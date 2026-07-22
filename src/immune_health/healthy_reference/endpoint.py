"""Role-aware, manifest-bound donor GP endpoints for fitting and scoring."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from immune_health.baselines.pseudobulk import ensure_donor_observation_ids
from immune_health.provenance import atomic_write_json, sha256_file, stable_hash

ENDPOINT_SCHEMA = "immune-health-donor-gp-endpoint/v1"
PROJECTION_OUTPUT_SCHEMA = "immune-health-tripso-projection-output/v1"
ENDPOINT_METADATA = "endpoint_metadata.parquet"
ENDPOINT_FEATURES = "endpoint_locations.npy"
ENDPOINT_COVARIANCES = "endpoint_covariances.npy"
ENDPOINT_MANIFEST = "endpoint_manifest.json"
ENDPOINT_ROLES = frozenset({"reference", "validation", "query"})
REFERENCE_DESIGNS = frozenset({"lodo", "all_healthy"})


def _ordered_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _strict_state_available(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        invalid = sorted(normalized.loc[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"state_available contains invalid values: {invalid[:5]}")
    return mapped.astype(bool)


def _parse_location(value: object, *, row_label: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Measurable endpoint row {row_label} lacks location_summary")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid location_summary JSON in row {row_label}") from exc
    array = np.asarray(decoded, dtype=np.float32)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(
            f"location_summary in row {row_label} must be a finite nonempty vector"
        )
    return array


def _parse_covariance(value: object, *, row_label: str, dimension: int) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Measurable endpoint row {row_label} lacks covariance_summary"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid covariance_summary JSON in row {row_label}") from exc
    array = np.asarray(decoded, dtype=np.float32)
    if array.shape != (dimension, dimension) or not np.isfinite(array).all():
        raise ValueError(
            f"covariance_summary in row {row_label} must be a finite "
            f"{dimension}x{dimension} matrix"
        )
    if not np.allclose(array, array.T, rtol=1e-5, atol=1e-6):
        raise ValueError(f"covariance_summary in row {row_label} is not symmetric")
    return array


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


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


def _single_values(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].dropna().drop_duplicates()
        if len(values) > 1:
            raise ValueError(
                f"Endpoint rows mix multiple {column} values: {values.head().tolist()}"
            )
        if len(values) == 1:
            value = values.iloc[0]
            result[column] = value.item() if hasattr(value, "item") else value
    return result


def _validate_role_scope(
    *,
    role: str,
    reference_design: str,
    heldout_dataset: str | None,
    datasets: Sequence[str],
) -> None:
    if role not in ENDPOINT_ROLES:
        raise ValueError(f"Endpoint role must be one of {sorted(ENDPOINT_ROLES)}")
    if reference_design not in REFERENCE_DESIGNS:
        raise ValueError(f"Reference design must be one of {sorted(REFERENCE_DESIGNS)}")
    observed = set(map(str, datasets))
    if not observed:
        raise ValueError("Endpoint contains no datasets")
    if reference_design == "lodo":
        if heldout_dataset is None or not str(heldout_dataset).strip():
            raise ValueError("LODO endpoint requires a heldout_dataset declaration")
        heldout = str(heldout_dataset)
        if role == "reference" and heldout in observed:
            raise ValueError(
                "Reference LODO endpoint must contain adaptation rows only; "
                f"held-out dataset {heldout!r} is present"
            )
        if role == "validation" and heldout in observed:
            raise ValueError(
                "Inner-validation LODO endpoint cannot contain the outer held-out "
                f"dataset {heldout!r}"
            )
        if role == "query" and observed != {heldout}:
            raise ValueError(
                "Query LODO endpoint must contain exactly its held-out dataset; "
                f"expected {heldout!r}, observed {sorted(observed)}"
            )
    elif heldout_dataset is not None:
        raise ValueError("all_healthy endpoints cannot declare a heldout_dataset")


def _load_projection_output(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Projection output manifest is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        PROJECTION_OUTPUT_SCHEMA
    ):
        raise ValueError(f"Unsupported projection output manifest: {path}")
    claimed = payload.get("manifest_sha256")
    if claimed is not None:
        content = dict(payload)
        content.pop("manifest_sha256", None)
        if claimed != stable_hash(content):
            raise ValueError("Projection output manifest content hash does not match")
    return payload, sha256_file(path)


def _validate_aggregation_provenance(
    manifest_path: Path,
    aggregate_table_path: Path,
    *,
    projection_output_manifest_path: Path,
    projection_output_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Aggregation manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" or manifest.get("stage") != (
        "donor_distribution_aggregation"
    ):
        raise ValueError("Aggregation manifest is not a completed donor aggregation")
    table_record = manifest.get("fine_type_distribution_table")
    if not isinstance(table_record, Mapping):
        raise ValueError("Aggregation manifest lacks a hashed endpoint table")
    if Path(str(table_record.get("path", ""))).resolve() != aggregate_table_path:
        raise ValueError("Aggregate table path differs from aggregation manifest")
    if table_record.get("sha256") != sha256_file(aggregate_table_path):
        raise ValueError("Aggregate table changed after aggregation completed")
    conversion = manifest.get("arrow_conversion_validation")
    if not isinstance(conversion, Mapping):
        raise ValueError("Aggregation manifest lacks Arrow conversion validation")
    projection = conversion.get("projection_output")
    if not isinstance(projection, Mapping):
        raise ValueError("Aggregation did not validate a projection output manifest")
    if Path(str(projection.get("manifest_path", ""))).resolve() != (
        projection_output_manifest_path
    ):
        raise ValueError("Aggregation and endpoint projection manifests differ")
    if projection.get("manifest_sha256") != projection_output_manifest_sha256:
        raise ValueError("Projection output changed after donor aggregation")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "arrow_conversion_manifest": conversion.get("manifest_path"),
        "arrow_conversion_manifest_sha256": conversion.get("manifest_sha256"),
        "projection_output": dict(projection),
    }


def assemble_donor_gp_endpoint(
    aggregate_table_path: Path,
    aggregation_manifest_path: Path,
    projection_output_manifest_path: Path,
    output_dir: Path,
    *,
    lineage: str,
    fine_type: str,
    gp_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one exact donor-level GP endpoint and its aligned numeric matrix."""

    aggregate_table_path = Path(aggregate_table_path).resolve()
    projection_output_manifest_path = Path(projection_output_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not aggregate_table_path.is_file():
        raise FileNotFoundError(
            f"Aggregated fine-type table is missing: {aggregate_table_path}"
        )
    projection, projection_manifest_hash = _load_projection_output(
        projection_output_manifest_path
    )
    aggregation = _validate_aggregation_provenance(
        aggregation_manifest_path,
        aggregate_table_path,
        projection_output_manifest_path=projection_output_manifest_path,
        projection_output_manifest_sha256=projection_manifest_hash,
    )
    role = str(projection.get("projection_role", ""))
    reference_design = str(projection.get("reference_design", ""))
    raw_heldout = projection.get("heldout_dataset")
    heldout_dataset = None if raw_heldout in {None, ""} else str(raw_heldout)
    if str(projection.get("lineage", "")) != str(lineage):
        raise ValueError("Endpoint lineage differs from projection output manifest")
    gp_projection = projection.get("gp_projection")
    if not isinstance(gp_projection, Mapping) or str(gp_id) not in set(
        map(str, gp_projection.get("program_ids", []))
    ):
        raise ValueError("Endpoint GP is outside the frozen projection allowlist")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / ENDPOINT_METADATA
    features_path = output_dir / ENDPOINT_FEATURES
    covariances_path = output_dir / ENDPOINT_COVARIANCES
    manifest_path = output_dir / ENDPOINT_MANIFEST
    existing = [
        path
        for path in (
            metadata_path,
            features_path,
            covariances_path,
            manifest_path,
        )
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite endpoint outputs: {existing}")

    table = pd.read_parquet(aggregate_table_path)
    required = {
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
        "fine_type_state_eligible",
        "state_available",
        "location_summary",
        "covariance_summary",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Aggregated endpoint table lacks columns: {missing}")
    selected = table.loc[
        table["lineage"].astype(str).eq(str(lineage))
        & table["fine_type"].astype(str).eq(str(fine_type))
        & table["gp_id"].astype(str).eq(str(gp_id))
    ].copy()
    if selected.empty:
        raise ValueError(
            "Aggregated table contains no exact lineage/fine_type/gp_id endpoint"
        )
    endpoint_key = ["observation_id", "lineage", "fine_type", "gp_id"]
    if selected.duplicated(endpoint_key).any():
        raise ValueError("Aggregated table duplicates an observation endpoint")
    endpoint_eligibility = _strict_state_available(selected["fine_type_state_eligible"])
    if not endpoint_eligibility.all():
        raise ValueError(
            "Fine type is ontology-ineligible for state/GP-age endpoint assembly"
        )
    state_available = _strict_state_available(selected["state_available"])
    measurable = selected.loc[state_available].copy()
    if measurable.empty:
        raise ValueError("No measurable donor observations remain for this endpoint")
    measurable = ensure_donor_observation_ids(measurable).reset_index(drop=True)
    for column in ("dataset", "donor_id", "sample_id", "age", "sex"):
        if measurable[column].isna().any():
            raise ValueError(
                f"Endpoint metadata column {column!r} contains missing values"
            )
    ages = pd.to_numeric(measurable["age"], errors="coerce")
    if not np.isfinite(ages).all():
        raise ValueError("Endpoint ages must be finite numeric values")
    measurable["age"] = ages.astype(float)
    source_datasets = sorted(selected["dataset"].astype(str).unique().tolist())
    declared_datasets = projection.get("datasets")
    if isinstance(declared_datasets, list) and set(map(str, declared_datasets)) != set(
        source_datasets
    ):
        raise ValueError(
            "Aggregated endpoint datasets differ from projection output manifest"
        )
    source_biological_units = sorted(
        selected["biological_unit_id"].astype(str).unique().tolist()
    )
    if source_biological_units != sorted(
        map(str, projection.get("biological_unit_ids", []))
    ):
        raise ValueError(
            "Aggregated endpoint donor scope differs from projection output manifest"
        )
    datasets = sorted(measurable["dataset"].astype(str).unique().tolist())
    _validate_role_scope(
        role=role,
        reference_design=reference_design,
        heldout_dataset=heldout_dataset,
        datasets=datasets,
    )

    vectors = [
        _parse_location(value, row_label=str(observation))
        for value, observation in zip(
            measurable["location_summary"],
            measurable["observation_id"],
            strict=True,
        )
    ]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Endpoint location dimensions differ: {sorted(dimensions)}")
    features = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    dimension = features.shape[1]
    covariances = np.ascontiguousarray(
        np.stack(
            [
                _parse_covariance(
                    value,
                    row_label=str(observation),
                    dimension=dimension,
                )
                for value, observation in zip(
                    measurable["covariance_summary"],
                    measurable["observation_id"],
                    strict=True,
                )
            ]
        ),
        dtype=np.float32,
    )
    feature_ids = [f"{gp_id}::location_{index:04d}" for index in range(dimension)]
    identity = _single_values(
        selected,
        (
            "lineage",
            "fine_type",
            "gp_id",
            "model_id",
            "model_manifest",
            "model_manifest_sha256",
            "checkpoint_sha256",
            "fold_id",
            "seed",
            "projection_role",
            "eligible_for_model_selection",
            "outer_query_evaluation_only",
            "reference_design",
            "heldout_dataset",
            "projection_output_manifest",
            "projection_output_manifest_sha256",
            "projection_arrow_tree_sha256",
            "arrow_conversion_manifest",
            "arrow_conversion_manifest_sha256",
            "arrow_cell_key_ordered_sha256",
        ),
    )
    expected_identity = {
        "lineage": str(lineage),
        "fine_type": str(fine_type),
        "gp_id": str(gp_id),
    }
    if {key: str(identity.get(key)) for key in expected_identity} != expected_identity:
        raise AssertionError("Endpoint identity changed during assembly")
    expected_provenance = {
        "model_id": projection["hashes"]["model_manifest_sha256"],
        "model_manifest_sha256": projection["hashes"]["model_manifest_sha256"],
        "checkpoint_sha256": projection["hashes"]["checkpoint_sha256"],
        "fold_id": projection["fold_id"],
        "seed": projection["seed"],
        "projection_role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "reference_design": reference_design,
        "heldout_dataset": heldout_dataset,
        "model_manifest": projection["model_manifest"],
        "projection_output_manifest": str(projection_output_manifest_path),
        "projection_output_manifest_sha256": projection_manifest_hash,
        "projection_arrow_tree_sha256": projection["hashes"]["arrow_tree_sha256"],
        "arrow_conversion_manifest_sha256": aggregation[
            "arrow_conversion_manifest_sha256"
        ],
        "arrow_conversion_manifest": aggregation["arrow_conversion_manifest"],
        "arrow_cell_key_ordered_sha256": projection["cell_key_ordered_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected_provenance.items()
        if str(identity.get(key)) != str(value)
    }
    if mismatches:
        raise ValueError(
            "Aggregate provenance differs from the frozen projection chain: "
            f"{mismatches}"
        )

    measurable.insert(0, "endpoint_row", np.arange(len(measurable), dtype=np.int64))
    measurable["projection_role"] = role
    measurable["eligible_for_model_selection"] = role == "validation"
    measurable["outer_query_evaluation_only"] = role == "query"
    measurable["reference_design"] = reference_design
    measurable["heldout_dataset"] = heldout_dataset

    metadata = measurable.drop(columns=["location_summary", "covariance_summary"])
    _atomic_parquet(metadata_path, metadata)
    _atomic_npy(features_path, features)
    _atomic_npy(covariances_path, covariances)
    row_key_digest = _ordered_digest(metadata["observation_id"].astype(str).tolist())
    payload: dict[str, Any] = {
        "schema_version": ENDPOINT_SCHEMA,
        "role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "reference_design": reference_design,
        "heldout_dataset": heldout_dataset,
        "endpoint": expected_identity,
        "datasets": datasets,
        "source_projection_datasets": source_datasets,
        "source_aggregate_table": str(aggregate_table_path),
        "source_aggregate_table_sha256": sha256_file(aggregate_table_path),
        "aggregation_manifest": aggregation["manifest_path"],
        "aggregation_manifest_sha256": aggregation["manifest_sha256"],
        "projection_output_manifest": str(projection_output_manifest_path),
        "projection_output_manifest_sha256": projection_manifest_hash,
        "metadata_path": metadata_path.name,
        "features_path": features_path.name,
        "covariances_path": covariances_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "features_npy_sha256": sha256_file(features_path),
        "covariances_npy_sha256": sha256_file(covariances_path),
        "shape": list(features.shape),
        "covariance_shape": list(covariances.shape),
        "dtype": "float32",
        "feature_ids": feature_ids,
        "n_input_endpoint_rows": int(len(selected)),
        "n_measurable_rows": int(len(metadata)),
        "n_excluded_state_unavailable": int((~state_available).sum()),
        "n_biological_units": int(metadata["biological_unit_id"].nunique()),
        "observation_id_ordered_sha256": row_key_digest,
        "source_provenance": identity,
        "state_policy": "exclude_only_rows_with_state_available=false",
        "fit_contract": {
            "metadata_path": metadata_path.name,
            "features_path": features_path.name,
            "covariances_path": covariances_path.name,
            "endpoint_manifest": manifest_path.name,
            "required_weighting_schemes": ["donor_pooled", "cohort_balanced"],
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return payload


def validate_endpoint_inputs(
    manifest_path: Path,
    metadata_path: Path,
    features_path: Path,
    *,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Revalidate exact endpoint outputs immediately before fitting or scoring."""

    manifest_path = Path(manifest_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    features_path = Path(features_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Endpoint manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != ENDPOINT_SCHEMA:
        raise ValueError(f"Unsupported endpoint manifest: {manifest_path}")
    hash_payload = dict(manifest)
    observed_manifest_hash = hash_payload.pop("manifest_sha256", None)
    if observed_manifest_hash != stable_hash(hash_payload):
        raise ValueError("Endpoint manifest content hash does not match")
    role = str(manifest.get("role", ""))
    if role not in ENDPOINT_ROLES:
        raise ValueError(f"Endpoint role must be one of {sorted(ENDPOINT_ROLES)}")
    if manifest.get("eligible_for_model_selection") is not (role == "validation"):
        raise ValueError("Endpoint model-selection eligibility disagrees with role")
    if manifest.get("outer_query_evaluation_only") is not (role == "query"):
        raise ValueError("Endpoint outer-query evaluation flag disagrees with role")
    if expected_role is not None and role != expected_role:
        raise ValueError(f"Endpoint role must be {expected_role!r}, observed {role!r}")
    expected_metadata = (
        manifest_path.parent / str(manifest.get("metadata_path", ""))
    ).resolve()
    expected_features = (
        manifest_path.parent / str(manifest.get("features_path", ""))
    ).resolve()
    expected_covariances = (
        manifest_path.parent / str(manifest.get("covariances_path", ""))
    ).resolve()
    if metadata_path != expected_metadata:
        raise ValueError("Endpoint metadata path differs from manifest")
    if features_path != expected_features:
        raise ValueError("Endpoint feature path differs from manifest")
    if sha256_file(metadata_path) != manifest.get("metadata_sha256"):
        raise ValueError("Endpoint metadata SHA-256 does not match")
    if sha256_file(features_path) != manifest.get("features_npy_sha256"):
        raise ValueError("Endpoint feature SHA-256 does not match")
    if not expected_covariances.is_file():
        raise FileNotFoundError("Endpoint covariance array is missing")
    if sha256_file(expected_covariances) != manifest.get("covariances_npy_sha256"):
        raise ValueError("Endpoint covariance SHA-256 does not match")

    metadata = pd.read_parquet(metadata_path)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    covariances = np.load(expected_covariances, mmap_mode="r", allow_pickle=False)
    if list(features.shape) != manifest.get("shape") or len(metadata) != len(features):
        raise ValueError("Endpoint metadata/features shape does not match")
    if features.dtype != np.dtype("float32") or not np.isfinite(features).all():
        raise ValueError("Endpoint features must be finite float32 values")
    expected_covariance_shape = [len(metadata), features.shape[1], features.shape[1]]
    if list(covariances.shape) != expected_covariance_shape or list(
        covariances.shape
    ) != manifest.get("covariance_shape"):
        raise ValueError("Endpoint covariance shape does not match locations")
    if covariances.dtype != np.dtype("float32") or not np.isfinite(covariances).all():
        raise ValueError("Endpoint covariances must be finite float32 values")
    if not np.allclose(
        covariances,
        np.swapaxes(covariances, 1, 2),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Endpoint covariance matrices are not symmetric")
    if "endpoint_row" not in metadata or not np.array_equal(
        pd.to_numeric(metadata["endpoint_row"], errors="coerce").to_numpy(),
        np.arange(len(metadata)),
    ):
        raise ValueError("Endpoint metadata row order is invalid")
    endpoint = manifest.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise ValueError("Endpoint manifest identity is invalid")
    for column in ("lineage", "fine_type", "gp_id"):
        if column not in metadata:
            raise ValueError(f"Endpoint metadata lacks {column}")
        values = metadata[column].dropna().astype(str).unique()
        if len(values) != 1 or values[0] != str(endpoint.get(column)):
            raise ValueError(f"Endpoint metadata mixes or changes {column}")
    for column, expected in (
        ("projection_role", role),
        ("eligible_for_model_selection", str(role == "validation")),
        ("outer_query_evaluation_only", str(role == "query")),
        ("reference_design", str(manifest.get("reference_design", ""))),
    ):
        if column not in metadata:
            raise ValueError(f"Endpoint metadata lacks {column}")
        values = metadata[column].dropna().astype(str).unique()
        if len(values) != 1 or values[0] != expected:
            raise ValueError(f"Endpoint metadata mixes or changes {column}")
    manifest_heldout = manifest.get("heldout_dataset")
    heldout_values = metadata["heldout_dataset"].dropna().astype(str).unique()
    if manifest_heldout is None:
        if len(heldout_values):
            raise ValueError("Endpoint metadata adds an undeclared heldout_dataset")
    elif len(heldout_values) != 1 or heldout_values[0] != str(manifest_heldout):
        raise ValueError("Endpoint metadata changes heldout_dataset")
    datasets = sorted(metadata["dataset"].astype(str).unique().tolist())
    if datasets != sorted(map(str, manifest.get("datasets", []))):
        raise ValueError("Endpoint metadata datasets differ from manifest")
    _validate_role_scope(
        role=role,
        reference_design=str(manifest.get("reference_design", "")),
        heldout_dataset=(
            None if manifest_heldout in {None, ""} else str(manifest_heldout)
        ),
        datasets=datasets,
    )
    observed_order = _ordered_digest(metadata["observation_id"].astype(str).tolist())
    if observed_order != manifest.get("observation_id_ordered_sha256"):
        raise ValueError("Endpoint observation order does not match manifest")
    feature_ids = manifest.get("feature_ids")
    if not isinstance(feature_ids, list) or len(feature_ids) != features.shape[1]:
        raise ValueError("Endpoint feature IDs do not match feature width")
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "reference_design": str(manifest.get("reference_design")),
        "heldout_dataset": manifest_heldout,
        "datasets": datasets,
        "endpoint": dict(endpoint),
        "feature_ids": list(map(str, feature_ids)),
        "shape": list(features.shape),
        "covariances_path": str(expected_covariances),
        "covariances_npy_sha256": manifest["covariances_npy_sha256"],
        "covariance_shape": list(covariances.shape),
        "observation_id_ordered_sha256": observed_order,
        "source_provenance": dict(manifest.get("source_provenance", {})),
        "projection_output_manifest_sha256": manifest.get(
            "projection_output_manifest_sha256"
        ),
    }
