"""Manifest-bound matched-depth empirical distribution scoring."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from immune_health.aggregation.distances import sliced_wasserstein_distance
from immune_health.aggregation.empirical_index import (
    EmpiricalDistributionStore,
    load_empirical_distribution_store,
)
from immune_health.healthy_reference.endpoint import validate_endpoint_inputs
from immune_health.provenance import atomic_write_json, sha256_file, stable_hash

from .kernel import (
    DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    KernelWeightResult,
    age_kernel_weights,
)
from .trajectory import REFERENCE_WEIGHTING_SCHEMES

EMPIRICAL_SCORING_SCHEMA = "immune-health-empirical-matched-depth-score/v1"
EMPIRICAL_SCORES = "empirical_matched_depth_scores.parquet"
EMPIRICAL_REPLICATES = "empirical_matched_depth_replicates.parquet"
EMPIRICAL_RELIABILITY = "empirical_reliability.parquet"
EMPIRICAL_SCORING_MANIFEST = "empirical_scoring_manifest.json"


@dataclass(frozen=True)
class _EndpointBinding:
    manifest_path: Path
    manifest: Mapping[str, Any]
    validation: Mapping[str, Any]
    metadata: pd.DataFrame


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_endpoint(path: Path, roles: str | Sequence[str]) -> _EndpointBinding:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Endpoint manifest is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    metadata_path = path.parent / str(manifest.get("metadata_path", ""))
    locations_path = path.parent / str(manifest.get("features_path", ""))
    validation = validate_endpoint_inputs(
        path,
        metadata_path,
        locations_path,
        expected_role=None,
    )
    allowed_roles = {roles} if isinstance(roles, str) else set(map(str, roles))
    if validation["role"] not in allowed_roles:
        raise ValueError(
            f"Endpoint role must be one of {sorted(allowed_roles)}, observed "
            f"{validation['role']!r}"
        )
    metadata = pd.read_parquet(metadata_path)
    return _EndpointBinding(path, manifest, validation, metadata)


def _validate_endpoint_pair(
    reference: _EndpointBinding, query: _EndpointBinding
) -> None:
    for field in ("reference_design", "heldout_dataset", "endpoint", "feature_ids"):
        if reference.manifest.get(field) != query.manifest.get(field):
            raise ValueError(f"Reference/query endpoint {field} differs")
    for field in ("model_manifest_sha256", "checkpoint_sha256", "fold_id", "seed"):
        left = reference.validation["source_provenance"].get(field)
        right = query.validation["source_provenance"].get(field)
        if left is None or right is None or str(left) != str(right):
            raise ValueError(
                f"Reference/query endpoint model provenance differs: {field}"
            )
    overlap = set(reference.metadata["biological_unit_id"].astype(str)) & set(
        query.metadata["biological_unit_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            "Reference/query empirical endpoints share biological units: "
            f"{sorted(overlap)[:5]}"
        )


def _validate_store_binding(
    store: EmpiricalDistributionStore,
    endpoint: _EndpointBinding,
) -> None:
    identity = endpoint.validation["endpoint"]
    if str(store.manifest.get("embedding_column")) != str(identity["gp_id"]):
        raise ValueError("Empirical index GP differs from endpoint identity")
    if (
        store.manifest.get("projection_output_manifest_sha256")
        != (endpoint.validation["projection_output_manifest_sha256"])
    ):
        raise ValueError(
            "Empirical index and endpoint come from different frozen projections"
        )
    endpoint_conversion_hash = endpoint.validation["source_provenance"].get(
        "arrow_conversion_manifest_sha256"
    )
    if (
        endpoint_conversion_hash is not None
        and store.manifest.get("arrow_conversion_manifest_sha256")
        != endpoint_conversion_hash
    ):
        raise ValueError(
            "Empirical index and endpoint aggregation bind different Arrow "
            "conversion manifests"
        )
    endpoint_aggregation_hash = endpoint.manifest.get("source_aggregate_table_sha256")
    if (
        not isinstance(endpoint_aggregation_hash, str)
        or store.manifest.get("aggregation_table_sha256") != endpoint_aggregation_hash
    ):
        raise ValueError(
            "Empirical index and endpoint do not bind the same donor aggregation table"
        )
    expected_dimension = len(endpoint.validation["feature_ids"])
    shape = store.manifest.get("source_embeddings_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or int(shape[1]) != (expected_dimension)
    ):
        raise ValueError("Empirical embedding dimension differs from endpoint")


def _group_lookup(store: EmpiricalDistributionStore) -> dict[tuple[str, ...], int]:
    keys = [
        tuple(map(str, values))
        for values in store.groups[
            ["observation_id", "lineage", "fine_type", "gp_id"]
        ].itertuples(index=False, name=None)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Empirical group table duplicates an endpoint key")
    return {key: index for index, key in enumerate(keys)}


def _bind_endpoint_groups(
    endpoint: _EndpointBinding,
    store: EmpiricalDistributionStore,
) -> np.ndarray:
    lookup = _group_lookup(store)
    identity = endpoint.validation["endpoint"]
    positions: list[int] = []
    unavailable: list[tuple[str, ...]] = []
    for row in endpoint.metadata.itertuples(index=False):
        key = (
            str(row.observation_id),
            str(identity["lineage"]),
            str(identity["fine_type"]),
            str(identity["gp_id"]),
        )
        position = lookup.get(key)
        if position is None:
            raise ValueError(f"Endpoint empirical distribution is missing: {key}")
        group = store.groups.iloc[position]
        if not bool(group["empirical_distance_eligible"]):
            unavailable.append(key)
        positions.append(position)
    if unavailable:
        raise ValueError(
            "Endpoint contains empirical distributions below the frozen eligibility "
            f"threshold; examples={unavailable[:5]}"
        )
    return np.asarray(positions, dtype=np.int64)


def _rng(seed: int, *components: object) -> np.random.Generator:
    digest = stable_hash([int(seed), *map(str, components)])
    return np.random.default_rng(int(digest[:16], 16))


def _directions(dimension: int, n_projections: int, seed: int) -> np.ndarray:
    directions = _rng(seed, "sliced_wasserstein_directions").normal(
        size=(n_projections, dimension)
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.ascontiguousarray(directions, dtype=np.float64)


def _source_rows(store: EmpiricalDistributionStore, group_position: int) -> np.ndarray:
    group = store.groups.iloc[int(group_position)]
    return store.embedding_rows[int(group["start"]) : int(group["stop"])]


def _sample_query(
    store: EmpiricalDistributionStore,
    group_position: int,
    depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source_rows = _source_rows(store, group_position)
    if len(source_rows) < depth:
        raise ValueError("query distribution has fewer cells than matched depth")
    local = rng.choice(len(source_rows), size=depth, replace=False)
    return np.asarray(store.embeddings[source_rows[local]], dtype=np.float32)


def _sample_reference_mixture(
    store: EmpiricalDistributionStore,
    group_positions: np.ndarray,
    weights: np.ndarray,
    depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_groups = rng.choice(
        len(group_positions), size=depth, replace=True, p=weights
    )
    result = np.empty((depth, store.embeddings.shape[1]), dtype=np.float32)
    for local_group in np.unique(selected_groups):
        output_rows = np.flatnonzero(selected_groups == local_group)
        source_rows = _source_rows(store, int(group_positions[local_group]))
        if len(source_rows) == 0:
            raise ValueError("reference empirical group contains no source rows")
        selected_cells = rng.choice(
            len(source_rows), size=len(output_rows), replace=True
        )
        result[output_rows] = store.embeddings[source_rows[selected_cells]]
    return result


def _kernel_weights(
    reference_metadata: pd.DataFrame,
    age: float,
    sex: str,
    *,
    bandwidth: float,
    minimum_exact_sex_donors: int,
    weighting_scheme: str,
) -> KernelWeightResult:
    return age_kernel_weights(
        reference_metadata["age"],
        age,
        sexes=reference_metadata["sex"],
        target_sex=sex,
        biological_unit_ids=reference_metadata["biological_unit_id"],
        datasets=reference_metadata["dataset"],
        weighting_scheme=weighting_scheme,
        bandwidth=bandwidth,
        minimum_exact_sex_donors=minimum_exact_sex_donors,
    )


def _distance(
    query: np.ndarray, reference: np.ndarray, projections: np.ndarray
) -> float:
    if len(query) != len(reference):
        raise AssertionError("matched-depth empirical samples differ in cell count")
    return sliced_wasserstein_distance(query, reference, projections=projections)


def _binding_record(
    endpoint: _EndpointBinding,
    empirical_manifest_path: Path,
    store: EmpiricalDistributionStore,
) -> dict[str, Any]:
    return {
        "endpoint_manifest": str(endpoint.manifest_path),
        "endpoint_manifest_file_sha256": sha256_file(endpoint.manifest_path),
        "endpoint_manifest_content_sha256": endpoint.manifest.get("manifest_sha256"),
        "endpoint_observation_id_ordered_sha256": endpoint.validation[
            "observation_id_ordered_sha256"
        ],
        "endpoint_projection_output_manifest_sha256": endpoint.validation[
            "projection_output_manifest_sha256"
        ],
        "endpoint_model_provenance": dict(endpoint.validation["source_provenance"]),
        "empirical_index_manifest": str(empirical_manifest_path.resolve()),
        "empirical_index_manifest_file_sha256": sha256_file(empirical_manifest_path),
        "empirical_index_manifest_content_sha256": store.manifest.get(
            "manifest_sha256"
        ),
        "empirical_source_cell_key_ordered_sha256": store.manifest[
            "source_cell_key_ordered_sha256"
        ],
        "empirical_source_embeddings_float32_payload_sha256": store.manifest[
            "source_embeddings_float32_payload_sha256"
        ],
        "empirical_groups_sha256": store.manifest["groups_sha256"],
        "empirical_rows_sha256": store.manifest["rows_sha256"],
        "aggregation_table_sha256": store.manifest["aggregation_table_sha256"],
    }


def score_empirical_matched_depth(
    *,
    reference_endpoint_manifest: Path,
    query_endpoint_manifest: Path,
    reference_empirical_index_manifest: Path,
    query_empirical_index_manifest: Path,
    output_dir: Path,
    depths: Sequence[int] = (25, 50, 100, 250, 500, 1000),
    n_replicates: int = 100,
    n_projections: int = 128,
    age_grid_size: int = 101,
    bandwidth: float = 10.0,
    minimum_exact_sex_donors: int = DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    weighting_scheme: str = "donor_pooled",
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score exact query endpoint distributions against a healthy mixture."""

    ordered_depths = tuple(sorted(set(map(int, depths))))
    if (
        not ordered_depths
        or ordered_depths[0] < 2
        or n_replicates < 2
        or n_projections < 1
        or age_grid_size < 2
        or bandwidth <= 0
        or minimum_exact_sex_donors < 1
    ):
        raise ValueError("empirical matched-depth settings are invalid")
    if weighting_scheme not in REFERENCE_WEIGHTING_SCHEMES:
        raise ValueError(f"unknown weighting scheme: {weighting_scheme}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / EMPIRICAL_SCORES
    replicates_path = output_dir / EMPIRICAL_REPLICATES
    reliability_path = output_dir / EMPIRICAL_RELIABILITY
    manifest_path = output_dir / EMPIRICAL_SCORING_MANIFEST
    existing = [
        path
        for path in (scores_path, replicates_path, reliability_path, manifest_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite empirical scores: {existing}")

    reference = _load_endpoint(reference_endpoint_manifest, "reference")
    query = _load_endpoint(query_endpoint_manifest, ("validation", "query"))
    _validate_endpoint_pair(reference, query)
    reference_store = load_empirical_distribution_store(
        reference_empirical_index_manifest
    )
    query_store = load_empirical_distribution_store(query_empirical_index_manifest)
    _validate_store_binding(reference_store, reference)
    _validate_store_binding(query_store, query)
    if reference_store.embeddings.shape[1] != query_store.embeddings.shape[1]:
        raise ValueError("Reference/query empirical embedding dimensions differ")
    reference_groups = _bind_endpoint_groups(reference, reference_store)
    query_groups = _bind_endpoint_groups(query, query_store)
    if reference.metadata["biological_unit_id"].nunique() < 2:
        raise ValueError("Empirical healthy reference requires at least two donors")

    dimension = int(reference_store.embeddings.shape[1])
    projections = _directions(dimension, n_projections, seed)
    projection_hash = hashlib.sha256(projections.tobytes()).hexdigest()
    age_grid = np.linspace(
        float(pd.to_numeric(reference.metadata["age"], errors="raise").min()),
        float(pd.to_numeric(reference.metadata["age"], errors="raise").max()),
        age_grid_size,
    )
    weight_cache: dict[tuple[float, str], KernelWeightResult] = {}

    def weights(age: float, sex: str) -> KernelWeightResult:
        key = (float(age), str(sex))
        if key not in weight_cache:
            weight_cache[key] = _kernel_weights(
                reference.metadata,
                age,
                sex,
                bandwidth=bandwidth,
                minimum_exact_sex_donors=minimum_exact_sex_donors,
                weighting_scheme=weighting_scheme,
            )
        return weight_cache[key]

    replicate_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    for query_position, row in enumerate(query.metadata.itertuples(index=False)):
        query_group = int(query_groups[query_position])
        n_query_cells = int(query_store.groups.iloc[query_group]["n_rows"])
        base = {
            column: getattr(row, column)
            for column in (
                "dataset",
                "donor_id",
                "biological_unit_id",
                "sample_id",
                "observation_id",
                "age",
                "sex",
                "lineage",
                "fine_type",
                "gp_id",
            )
            if hasattr(row, column)
        }
        for column in (
            "model_id",
            "model_manifest_sha256",
            "checkpoint_sha256",
            "fold_id",
            "seed",
        ):
            if column in query.validation["source_provenance"]:
                base[column] = query.validation["source_provenance"][column]
        base["projection_role"] = query.validation["role"]
        target_age = float(row.age)
        target_sex = str(row.sex)
        matched_support = weights(target_age, target_sex)
        for depth in ordered_depths:
            reliability_base = {
                **base,
                "matched_cell_depth": depth,
                "n_query_cells_available": n_query_cells,
                "fraction_query_cells_used": min(depth / n_query_cells, 1.0),
                "n_replicates_requested": n_replicates,
                "n_projections": n_projections,
                "reference_support_donors": matched_support.n_support_donors,
                "reference_support_cohorts": matched_support.n_support_cohorts,
                "reference_effective_support_donors": (
                    matched_support.effective_support_donors
                ),
                "reference_exact_sex_used": matched_support.exact_sex_used,
                "reference_age_extrapolation": (
                    not matched_support.target_age_in_support_range
                ),
            }
            if n_query_cells < depth:
                reliability_records.append(
                    {
                        **reliability_base,
                        "status": "insufficient_query_cells",
                        "n_replicates_computed": 0,
                    }
                )
                continue
            for replicate in range(n_replicates):
                query_sample = _sample_query(
                    query_store,
                    query_group,
                    depth,
                    _rng(seed, row.observation_id, depth, replicate, "query"),
                )
                matched_reference = _sample_reference_mixture(
                    reference_store,
                    reference_groups,
                    matched_support.weights,
                    depth,
                    _rng(
                        seed,
                        row.observation_id,
                        depth,
                        replicate,
                        "reference_age_matched",
                    ),
                )
                age_matched_distance = _distance(
                    query_sample, matched_reference, projections
                )
                grid_distances: list[float] = []
                for grid_index, candidate_age in enumerate(age_grid):
                    candidate_support = weights(float(candidate_age), target_sex)
                    candidate_reference = _sample_reference_mixture(
                        reference_store,
                        reference_groups,
                        candidate_support.weights,
                        depth,
                        _rng(
                            seed,
                            row.observation_id,
                            depth,
                            replicate,
                            "reference_grid",
                            grid_index,
                        ),
                    )
                    grid_distances.append(
                        _distance(query_sample, candidate_reference, projections)
                    )
                grid_distances_array = np.asarray(grid_distances, dtype=float)
                minimum = float(grid_distances_array.min())
                tied = np.flatnonzero(
                    np.isclose(grid_distances_array, minimum, rtol=1e-10, atol=1e-12)
                )
                predicted_age = float(age_grid[tied].mean())
                replicate_records.append(
                    {
                        **base,
                        "matched_cell_depth": depth,
                        "bootstrap_replicate": replicate,
                        "age_matched_empirical_sliced_wasserstein_distance": (
                            age_matched_distance
                        ),
                        "off_trajectory_empirical_sliced_wasserstein_distance": (
                            minimum
                        ),
                        "predicted_empirical_gp_age": predicted_age,
                        "empirical_gp_age_acceleration": predicted_age - target_age,
                    }
                )
            reliability_records.append(
                {
                    **reliability_base,
                    "status": "computed",
                    "n_replicates_computed": n_replicates,
                }
            )

    replicates = pd.DataFrame.from_records(replicate_records)
    reliability = pd.DataFrame.from_records(reliability_records)
    if replicates.empty:
        raise ValueError(
            "No requested matched depth is available for any query endpoint row"
        )
    group_columns = [
        column
        for column in (
            "dataset",
            "donor_id",
            "biological_unit_id",
            "sample_id",
            "observation_id",
            "age",
            "sex",
            "lineage",
            "fine_type",
            "gp_id",
            "model_id",
            "model_manifest_sha256",
            "checkpoint_sha256",
            "fold_id",
            "seed",
            "projection_role",
            "matched_cell_depth",
        )
        if column in replicates
    ]
    metrics = (
        "age_matched_empirical_sliced_wasserstein_distance",
        "off_trajectory_empirical_sliced_wasserstein_distance",
        "predicted_empirical_gp_age",
        "empirical_gp_age_acceleration",
    )
    aggregate = replicates.groupby(group_columns, observed=True, sort=True)[
        list(metrics)
    ].agg(["mean", "std"])
    aggregate.columns = [
        f"{metric}{'_cell_sampling_se' if statistic == 'std' else ''}"
        for metric, statistic in aggregate.columns
    ]
    scores = aggregate.reset_index()
    scores["cell_sampling_se"] = scores[
        "age_matched_empirical_sliced_wasserstein_distance_cell_sampling_se"
    ]
    reliability = reliability.merge(
        scores[
            [
                "observation_id",
                "matched_cell_depth",
                *metrics,
                "cell_sampling_se",
                "off_trajectory_empirical_sliced_wasserstein_distance_cell_sampling_se",
                "predicted_empirical_gp_age_cell_sampling_se",
            ]
        ],
        on=["observation_id", "matched_cell_depth"],
        how="left",
        validate="one_to_one",
    )
    reliability["reliability_interpretation"] = np.where(
        reliability["status"].eq("computed"),
        "descriptive_depth_and_resampling_stability",
        "metric_not_computed",
    )
    for _, positions in reliability.groupby(
        "observation_id", sort=False
    ).groups.items():
        selected = reliability.loc[positions]
        computed = selected.loc[selected["status"].eq("computed")]
        if computed.empty:
            continue
        deepest = computed.sort_values("matched_cell_depth").iloc[-1]
        reliability.loc[positions, "deepest_computed_depth"] = int(
            deepest["matched_cell_depth"]
        )
        reliability.loc[positions, "age_matched_distance_delta_from_deepest"] = (
            pd.to_numeric(
                selected["age_matched_empirical_sliced_wasserstein_distance"],
                errors="coerce",
            )
            - float(deepest["age_matched_empirical_sliced_wasserstein_distance"])
        )
        reliability.loc[positions, "predicted_age_delta_from_deepest"] = pd.to_numeric(
            selected["predicted_empirical_gp_age"], errors="coerce"
        ) - float(deepest["predicted_empirical_gp_age"])

    _atomic_parquet(scores_path, scores)
    _atomic_parquet(replicates_path, replicates)
    _atomic_parquet(reliability_path, reliability)
    payload: dict[str, Any] = {
        "schema_version": EMPIRICAL_SCORING_SCHEMA,
        "status": "complete",
        "endpoint": dict(reference.validation["endpoint"]),
        "reference_design": reference.validation["reference_design"],
        "heldout_dataset": reference.validation["heldout_dataset"],
        "target_role": query.validation["role"],
        "reference_binding": _binding_record(
            reference,
            Path(reference_empirical_index_manifest),
            reference_store,
        ),
        "query_binding": _binding_record(
            query,
            Path(query_empirical_index_manifest),
            query_store,
        ),
        "leakage_checks": {
            "reference_query_biological_units_disjoint": True,
            "same_model_checkpoint_fold_seed": True,
            "role_specific_projection_and_empirical_indices": True,
            "query_used_to_fit_reference_weights": False,
        },
        "settings": {
            "depths": list(ordered_depths),
            "n_replicates": n_replicates,
            "n_projections": n_projections,
            "projection_directions_sha256": projection_hash,
            "age_grid_size": age_grid_size,
            "age_grid_min": float(age_grid.min()),
            "age_grid_max": float(age_grid.max()),
            "bandwidth": bandwidth,
            "minimum_exact_sex_donors": minimum_exact_sex_donors,
            "weighting_scheme": weighting_scheme,
            "seed": seed,
            "query_sampling": "without_replacement_at_requested_depth",
            "reference_sampling": (
                "age_sex_kernel_weighted_donor_then_uniform_cell_with_replacement"
            ),
            "equal_query_reference_cell_depth": True,
        },
        "metrics": {
            "age_matched": metrics[0],
            "off_trajectory": metrics[1],
            "predicted_age": metrics[2],
            "age_acceleration": metrics[3],
            "cell_sampling_se": (
                "sample standard deviation across matched-depth cell-resampling "
                "replicates; an endpoint measurement-variability estimate, not the "
                "Monte Carlo standard error of the reported replicate mean"
            ),
            "cell_sampling_se_divides_by_sqrt_n_replicates": False,
            "spline_residual_covariance_used": False,
        },
        "outputs": {
            "scores": {
                "path": scores_path.name,
                "sha256": sha256_file(scores_path),
                "n_rows": len(scores),
            },
            "replicates": {
                "path": replicates_path.name,
                "sha256": sha256_file(replicates_path),
                "n_rows": len(replicates),
            },
            "reliability": {
                "path": reliability_path.name,
                "sha256": sha256_file(reliability_path),
                "n_rows": len(reliability),
            },
        },
        "storage_contract": {
            "reference_embeddings_memory_mapped": isinstance(
                reference_store.embeddings, np.memmap
            ),
            "query_embeddings_memory_mapped": isinstance(
                query_store.embeddings, np.memmap
            ),
            "embedding_values_duplicated_on_disk": False,
            "covariance_arrays_read_or_copied": False,
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return payload
