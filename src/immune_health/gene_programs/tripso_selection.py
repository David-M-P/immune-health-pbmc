"""Training-only selection of transferable fine-type TRIPSO gene programs.

The selector consumes donor-cross-fitted predicted ages from frozen, role=reference
endpoint runs.  It never reads query scores or raw TRIPSO coordinates.  Every input
is revalidated through the endpoint, model, checkpoint, and cross-fit hashes before
any statistical screen is calculated.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from immune_health.gene_programs.transferability import (
    TransferabilityConfig,
    select_transferable_gene_programs,
)
from immune_health.healthy_reference.endpoint import validate_endpoint_inputs
from immune_health.provenance import atomic_write_json, sha256_file, stable_hash
from immune_health.tripso_adapter.contracts import load_fold_input_manifest
from immune_health.tripso_adapter.provenance import validate_checkpoint_manifest

SELECTION_SCHEMA = "immune-health-tripso-gp-selection/v1"
CROSSFIT_SCHEMA = "immune-health-training-crossfit-scores/v1"
REFERENCE_SCHEMA = "immune-health-frozen-healthy-reference/v1"
MODEL_SCHEMA = "immune-health-tripso-model/v1"

KEY_COLUMNS = ("lineage", "fine_type", "gp_id")


@dataclass(frozen=True)
class TripsoGPSelectionConfig:
    """Prespecified transfer, stability, coverage, and nuisance thresholds."""

    minimum_donors_per_cohort: int = 20
    minimum_age_span: float = 10.0
    minimum_cohorts: int = 3
    minimum_sign_concordance: float = 0.75
    maximum_i2: float = 0.75
    maximum_fdr: float = 0.05
    minimum_absolute_standardized_slope_per_decade: float = 0.0
    minimum_seed_retention_fraction: float = 1.0
    minimum_seed_sign_concordance: float = 1.0
    minimum_state_observation_coverage: float = 0.0
    minimum_median_cells: float = 5.0
    maximum_absolute_depth_partial_correlation: float = 0.5
    maximum_absolute_composition_partial_correlation: float = 0.5
    minimum_seed_rank_correlation: float | None = None
    maximum_seed_effect_sd: float | None = None
    minimum_baseline_standardized_improvement: float | None = None

    def validate(self) -> None:
        TransferabilityConfig(
            minimum_donors_per_cohort=self.minimum_donors_per_cohort,
            minimum_age_span=self.minimum_age_span,
            minimum_cohorts=self.minimum_cohorts,
            minimum_sign_concordance=self.minimum_sign_concordance,
            maximum_i2=self.maximum_i2,
            maximum_fdr=self.maximum_fdr,
            minimum_absolute_standardized_slope_per_decade=(
                self.minimum_absolute_standardized_slope_per_decade
            ),
        ).validate()
        for name, value in (
            ("minimum_seed_retention_fraction", self.minimum_seed_retention_fraction),
            ("minimum_seed_sign_concordance", self.minimum_seed_sign_concordance),
            (
                "minimum_state_observation_coverage",
                self.minimum_state_observation_coverage,
            ),
            (
                "maximum_absolute_depth_partial_correlation",
                self.maximum_absolute_depth_partial_correlation,
            ),
            (
                "maximum_absolute_composition_partial_correlation",
                self.maximum_absolute_composition_partial_correlation,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.minimum_median_cells < 0:
            raise ValueError("minimum_median_cells cannot be negative")
        if self.minimum_seed_rank_correlation is not None and not (
            -1 <= self.minimum_seed_rank_correlation <= 1
        ):
            raise ValueError("minimum_seed_rank_correlation must be in [-1, 1]")
        if self.maximum_seed_effect_sd is not None and self.maximum_seed_effect_sd < 0:
            raise ValueError("maximum_seed_effect_sd cannot be negative")

    def transferability(self) -> TransferabilityConfig:
        return TransferabilityConfig(
            minimum_donors_per_cohort=self.minimum_donors_per_cohort,
            minimum_age_span=self.minimum_age_span,
            minimum_cohorts=self.minimum_cohorts,
            minimum_sign_concordance=self.minimum_sign_concordance,
            maximum_i2=self.maximum_i2,
            maximum_fdr=self.maximum_fdr,
            minimum_absolute_standardized_slope_per_decade=(
                self.minimum_absolute_standardized_slope_per_decade
            ),
        )


@dataclass(frozen=True)
class ValidatedCrossfitRun:
    """One hash-validated candidate endpoint for one independently trained seed."""

    reference_manifest_path: Path
    crossfit_path: Path
    endpoint_manifest_path: Path
    model_manifest_path: Path
    lineage: str
    fine_type: str
    gp_id: str
    fold_id: str
    heldout_dataset: str
    seed: int
    model_id: str
    model_signature: str
    n_input_endpoint_rows: int
    scores: pd.DataFrame
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class TripsoGPSelectionResult:
    """Auditable cohort/seed effects and one final row per GP/fine type."""

    effects: pd.DataFrame
    selection: pd.DataFrame
    manifest: Mapping[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _bound_path(parent: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def _single_string(frame: pd.DataFrame, column: str, label: str) -> str:
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{label} must contain exactly one {column}")
    return str(values[0])


def _assert_aligned(
    observed: pd.Series, expected: pd.Series, column: str, *, numeric: bool = False
) -> None:
    if numeric:
        left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
        matches = np.allclose(left, right, rtol=0.0, atol=1e-10, equal_nan=True)
    else:
        left = observed.astype("string").fillna("<NA>").to_numpy()
        right = expected.astype("string").fillna("<NA>").to_numpy()
        matches = np.array_equal(left, right)
    if not matches:
        raise ValueError(f"Cross-fit scores differ from endpoint metadata: {column}")


def _scientific_model_signature(model: Mapping[str, Any]) -> str:
    hashes = model.get("hashes", {})
    configuration = model.get("model_configuration", {})
    signature = {
        "fold_id": model.get("fold_id"),
        "held_out_dataset": model.get("held_out_dataset"),
        "lineage": model.get("lineage"),
        "model_type": model.get("model_type"),
        "sampler": model.get("sampler"),
        "resource_hashes": {
            key: hashes.get(key)
            for key in (
                "gp_library_sha256",
                "gene_vocabulary_sha256",
                "input_manifest_sha256",
                "projection_gp_candidates_sha256",
                "projection_gp_program_ids_ordered_sha256",
            )
        },
        "model_configuration": {
            key: configuration.get(key)
            for key in (
                "tokenizer",
                "preprocessing",
                "embedding_dimension",
                "gp_latent_dimension",
                "gene_embedding_dimension",
                "model_type",
                "feature_set",
                "hvg_size",
                "includes_all_retained_gp_genes",
                "geneformer_identity",
            )
        },
    }
    return stable_hash(signature)


def validate_crossfit_reference_run(
    reference_manifest_path: Path,
    *,
    lineage: str,
    fold_id: str,
    heldout_dataset: str,
    training_datasets: Sequence[str],
    weighting_scheme: str,
) -> ValidatedCrossfitRun:
    """Validate one reference endpoint and its bound donor-cross-fitted scores."""

    reference_manifest_path = Path(reference_manifest_path).resolve()
    reference = _read_json(reference_manifest_path, "healthy-reference manifest")
    if reference.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError(
            f"Unsupported healthy-reference manifest: {reference_manifest_path}"
        )
    arrays_path = _bound_path(
        reference_manifest_path.parent,
        reference.get("arrays_path"),
        "healthy-reference arrays",
    )
    if not arrays_path.is_file() or sha256_file(arrays_path) != reference.get(
        "arrays_sha256"
    ):
        raise ValueError("Healthy-reference arrays are missing or fail their hash")
    if reference.get("input_composition") != "reference_only":
        raise ValueError(
            "TRIPSO GP selection forbids legacy combined query/reference input"
        )
    if reference.get("final_all_healthy") is not False:
        raise ValueError("Fine-type LODO GP selection requires a LODO reference run")
    if str(reference.get("heldout_dataset")) != heldout_dataset:
        raise ValueError(
            "Healthy-reference heldout dataset differs from selection fold"
        )
    if reference.get("query_dataset_offset") != "forbidden":
        raise ValueError("Healthy-reference manifest does not forbid a query offset")
    if reference.get("weighting_scheme") != weighting_scheme:
        raise ValueError(
            "Healthy-reference weighting scheme differs from selection plan"
        )
    expected_datasets = tuple(sorted(map(str, training_datasets)))
    observed_datasets = tuple(sorted(map(str, reference.get("training_datasets", []))))
    if observed_datasets != expected_datasets or heldout_dataset in observed_datasets:
        raise ValueError(
            "Healthy-reference training cohorts differ or include the query"
        )

    endpoint_record = reference.get("endpoint_artifact")
    if not isinstance(endpoint_record, Mapping):
        raise ValueError("GP selection requires an endpoint-backed healthy reference")
    endpoint_manifest_path = _bound_path(
        reference_manifest_path.parent,
        endpoint_record.get("manifest_path"),
        "endpoint manifest",
    )
    endpoint_manifest = _read_json(endpoint_manifest_path, "endpoint manifest")
    metadata_path = _bound_path(
        endpoint_manifest_path.parent,
        endpoint_manifest.get("metadata_path"),
        "endpoint metadata",
    )
    features_path = _bound_path(
        endpoint_manifest_path.parent,
        endpoint_manifest.get("features_path"),
        "endpoint locations",
    )
    endpoint = validate_endpoint_inputs(
        endpoint_manifest_path,
        metadata_path,
        features_path,
        expected_role="reference",
    )
    if endpoint["reference_design"] != "lodo":
        raise ValueError("GP selection accepts only LODO reference endpoints")
    if endpoint["heldout_dataset"] != heldout_dataset:
        raise ValueError("Endpoint heldout dataset differs from the selection fold")
    if tuple(endpoint["datasets"]) != expected_datasets:
        raise ValueError("Endpoint cohorts differ from the required training cohorts")
    identity = endpoint["endpoint"]
    if str(identity.get("lineage")) != lineage:
        raise ValueError(
            "Endpoint lineage differs from the requested selection lineage"
        )
    for key in ("manifest_sha256", "observation_id_ordered_sha256"):
        if endpoint_record.get(key) != endpoint.get(key):
            raise ValueError(f"Healthy reference endpoint binding changed: {key}")

    source = endpoint.get("source_provenance", {})
    if str(source.get("fold_id")) != fold_id:
        raise ValueError("Endpoint fold differs from the requested selection fold")
    seed = int(source.get("seed"))
    if int(reference.get("seed")) != seed:
        raise ValueError("Endpoint and healthy-reference seeds differ")
    model_manifest_path = Path(str(source.get("model_manifest", ""))).resolve()
    model = validate_checkpoint_manifest(
        model_manifest_path,
        expected_fold_id=fold_id,
        expected_held_out_dataset=heldout_dataset,
        expected_lineage=lineage,
    )
    if model.get("schema_version") != MODEL_SCHEMA or int(model.get("seed")) != seed:
        raise ValueError("Endpoint source model identity or seed differs")
    model_file_hash = sha256_file(model_manifest_path)
    if str(source.get("model_manifest_sha256")) != model_file_hash:
        raise ValueError("Endpoint source model manifest hash changed")
    if str(source.get("model_id")) != model_file_hash:
        raise ValueError("Endpoint model_id is not the source model manifest hash")
    if str(source.get("checkpoint_sha256")) != str(
        model.get("hashes", {}).get("checkpoint_sha256")
    ):
        raise ValueError("Endpoint checkpoint hash differs from the source model")
    fold_input_path = _bound_path(
        model_manifest_path.parent,
        model.get("paths", {}).get("fold_input_manifest"),
        "TRIPSO fold-input manifest",
    )
    if sha256_file(fold_input_path) != model.get("hashes", {}).get(
        "input_manifest_sha256"
    ):
        raise ValueError("TRIPSO fold-input manifest changed after model fitting")
    fold_input = load_fold_input_manifest(fold_input_path)
    if (
        fold_input.get("reference_design") != "lodo"
        or str(fold_input.get("fold_id")) != fold_id
        or str(fold_input.get("held_out_dataset")) != heldout_dataset
        or str(fold_input.get("lineage")) != lineage
    ):
        raise ValueError("TRIPSO fold-input identity differs from the selection fold")
    fold_datasets = set(map(str, fold_input.get("datasets", ())))
    if fold_datasets != {*expected_datasets, heldout_dataset}:
        raise ValueError("TRIPSO fold-input cohort universe differs from selection")
    candidate_path = _bound_path(
        fold_input_path.parent,
        fold_input.get("inputs", {}).get("projection_gp_candidates_path"),
        "training-only projection GP candidates",
    )
    candidate_manifest = _read_json(candidate_path, "projection GP candidates")
    if str(identity.get("gp_id")) not in set(
        map(str, candidate_manifest.get("program_ids", ()))
    ):
        raise ValueError(
            "Endpoint GP was not in the fold-local training-only candidates"
        )
    adaptation_units = set(
        map(str, fold_input.get("adaptation_biological_unit_ids", ()))
    )
    validation_units = set(
        map(str, fold_input.get("validation_biological_unit_ids", ()))
    )
    query_units = set(map(str, fold_input.get("query_biological_unit_ids", ())))
    if any(unit.split("::", 1)[0] == heldout_dataset for unit in adaptation_units):
        raise ValueError("Held-out donors entered TRIPSO adaptation")
    if any(unit.split("::", 1)[0] == heldout_dataset for unit in validation_units):
        raise ValueError("Held-out donors entered TRIPSO inner validation")
    reference_unit_datasets = {
        unit.split("::", 1)[0] for unit in adaptation_units | validation_units
    }
    if reference_unit_datasets != set(expected_datasets):
        raise ValueError("TRIPSO reference donor cohorts differ from selection")
    if not query_units or any(
        unit.split("::", 1)[0] != heldout_dataset for unit in query_units
    ):
        raise ValueError("TRIPSO query donor scope differs from held-out cohort")

    crossfit_record = reference.get("training_crossfit_scores")
    if not isinstance(crossfit_record, Mapping):
        raise ValueError("Healthy reference lacks a bound training cross-fit artifact")
    if crossfit_record.get("schema_version") != CROSSFIT_SCHEMA:
        raise ValueError("Unsupported training cross-fit score schema")
    if crossfit_record.get("query_data_consulted") is not False:
        raise ValueError("Cross-fit manifest does not prove query exclusion")
    if crossfit_record.get("fit_scope") != "donor_grouped_training_only":
        raise ValueError(
            "Cross-fit scores are not declared donor-grouped training-only"
        )
    crossfit_path = _bound_path(
        reference_manifest_path.parent,
        crossfit_record.get("path"),
        "training cross-fit scores",
    )
    if not crossfit_path.is_file() or sha256_file(crossfit_path) != crossfit_record.get(
        "sha256"
    ):
        raise ValueError("Training cross-fit scores are missing or fail their hash")
    if crossfit_record.get("score_column") != "predicted_gp_age":
        raise ValueError("GP selection must use cross-fitted predicted_gp_age")
    if crossfit_record.get("fold_column") != "inner_crossfit_fold":
        raise ValueError("Cross-fit artifact lacks the protected inner-fold column")

    scores = pd.read_parquet(crossfit_path)
    endpoint_metadata = pd.read_parquet(metadata_path)
    required = {
        "row_index",
        "inner_crossfit_fold",
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
        "n_cells",
        "fine_type_fraction",
        "predicted_gp_age",
        "gp_age_acceleration",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Training cross-fit scores lack required columns: {missing}")
    if len(scores) != len(endpoint_metadata) or not np.array_equal(
        pd.to_numeric(scores["row_index"], errors="coerce").to_numpy(),
        np.arange(len(scores)),
    ):
        raise ValueError("Training cross-fit rows do not match endpoint row order")
    for column in (
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "observation_id",
        "sex",
        "lineage",
        "fine_type",
        "gp_id",
    ):
        _assert_aligned(scores[column], endpoint_metadata[column], column)
    for column in ("age", "n_cells", "fine_type_fraction"):
        _assert_aligned(scores[column], endpoint_metadata[column], column, numeric=True)
    if set(scores["dataset"].astype(str)) != set(expected_datasets):
        raise ValueError("Cross-fit scores omit a training cohort or add query data")
    if heldout_dataset in set(scores["dataset"].astype(str)):
        raise ValueError("Held-out query cohort entered GP selection scores")
    candidate = tuple(
        _single_string(scores, column, "cross-fit scores") for column in KEY_COLUMNS
    )
    if candidate != (
        str(identity["lineage"]),
        str(identity["fine_type"]),
        str(identity["gp_id"]),
    ):
        raise ValueError("Cross-fit candidate identity differs from its endpoint")
    predicted = pd.to_numeric(scores["predicted_gp_age"], errors="coerce").to_numpy()
    age = pd.to_numeric(scores["age"], errors="coerce").to_numpy()
    acceleration = pd.to_numeric(
        scores["gp_age_acceleration"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(predicted).all() or not np.isfinite(acceleration).all():
        raise ValueError("Cross-fitted predicted ages must be finite")
    if not np.allclose(acceleration, predicted - age, rtol=1e-8, atol=1e-8):
        raise ValueError("GP age acceleration is inconsistent with predicted age")
    inner_folds = pd.to_numeric(scores["inner_crossfit_fold"], errors="coerce")
    if inner_folds.isna().any() or inner_folds.nunique() < 2:
        raise ValueError("Cross-fit scores require at least two finite inner folds")
    donor_folds = (
        scores.assign(_inner_fold=inner_folds)
        .groupby("biological_unit_id", observed=True)["_inner_fold"]
        .nunique()
    )
    if donor_folds.gt(1).any():
        raise ValueError("One donor appears in multiple cross-fit validation folds")
    cells = pd.to_numeric(scores["n_cells"], errors="coerce")
    fractions = pd.to_numeric(scores["fine_type_fraction"], errors="coerce")
    if cells.isna().any() or cells.le(0).any():
        raise ValueError("Cross-fit endpoint cell counts must be positive")
    if fractions.isna().any() or fractions.le(0).any() or fractions.gt(1).any():
        raise ValueError("Cross-fit fine-type fractions must lie in (0, 1]")

    model_id = model_file_hash
    provenance = {
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "reference_arrays": str(arrays_path),
        "reference_arrays_sha256": sha256_file(arrays_path),
        "crossfit_scores": str(crossfit_path),
        "crossfit_scores_sha256": sha256_file(crossfit_path),
        "endpoint_manifest": str(endpoint_manifest_path),
        "endpoint_manifest_sha256": sha256_file(endpoint_manifest_path),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": model_file_hash,
        "checkpoint_sha256": model["hashes"]["checkpoint_sha256"],
        "fold_input_manifest": str(fold_input_path),
        "fold_input_manifest_sha256": sha256_file(fold_input_path),
        "projection_gp_candidates": str(candidate_path),
        "projection_gp_candidates_sha256": sha256_file(candidate_path),
    }
    return ValidatedCrossfitRun(
        reference_manifest_path=reference_manifest_path,
        crossfit_path=crossfit_path,
        endpoint_manifest_path=endpoint_manifest_path,
        model_manifest_path=model_manifest_path,
        lineage=candidate[0],
        fine_type=candidate[1],
        gp_id=candidate[2],
        fold_id=fold_id,
        heldout_dataset=heldout_dataset,
        seed=seed,
        model_id=model_id,
        model_signature=_scientific_model_signature(model),
        n_input_endpoint_rows=int(endpoint_manifest.get("n_input_endpoint_rows", 0)),
        scores=scores,
        provenance=provenance,
    )


def _weighted_residual(
    values: np.ndarray, design: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    cross = design.T @ (weights[:, None] * design)
    coefficient = np.linalg.pinv(cross) @ (design.T @ (weights * values))
    return values - design @ coefficient


def _donor_balanced_partial_correlation(frame: pd.DataFrame, covariate: str) -> float:
    """Age/sex/cohort-adjusted association with one total weight per donor."""

    acceleration = pd.to_numeric(
        frame["gp_age_acceleration"], errors="coerce"
    ).to_numpy(dtype=float)
    age = pd.to_numeric(frame["age"], errors="coerce").to_numpy(dtype=float)
    if covariate == "n_cells":
        nuisance = np.log1p(
            pd.to_numeric(frame[covariate], errors="coerce").to_numpy(dtype=float)
        )
    elif covariate == "fine_type_fraction":
        fraction = pd.to_numeric(frame[covariate], errors="coerce").to_numpy(
            dtype=float
        )
        nuisance = np.log(np.clip(fraction, 1e-6, 1 - 1e-6)) - np.log1p(
            -np.clip(fraction, 1e-6, 1 - 1e-6)
        )
    else:
        raise ValueError(f"Unsupported nuisance covariate: {covariate}")
    if not all(np.isfinite(value).all() for value in (acceleration, age, nuisance)):
        raise ValueError(f"Non-finite values prevent {covariate} diagnostics")
    donors = frame["biological_unit_id"].astype(str).to_numpy()
    _, donor_inverse, donor_counts = np.unique(
        donors, return_inverse=True, return_counts=True
    )
    weights = 1.0 / donor_counts[donor_inverse]
    confounders = pd.DataFrame(
        {
            "age": age,
            "sex": frame["sex"].astype(str),
            "dataset": frame["dataset"].astype(str),
        }
    )
    categorical = pd.get_dummies(
        confounders[["sex", "dataset"]], drop_first=True, dtype=float
    )
    design = np.column_stack(
        [np.ones(len(frame)), age - np.average(age, weights=weights), categorical]
    )
    outcome_residual = _weighted_residual(acceleration, design, weights)
    nuisance_residual = _weighted_residual(nuisance, design, weights)
    outcome_residual -= np.average(outcome_residual, weights=weights)
    nuisance_residual -= np.average(nuisance_residual, weights=weights)
    outcome_scale = float(np.average(outcome_residual**2, weights=weights))
    nuisance_scale = float(np.average(nuisance_residual**2, weights=weights))
    if outcome_scale <= 1e-15 or nuisance_scale <= 1e-15:
        return 0.0
    covariance = float(
        np.average(outcome_residual * nuisance_residual, weights=weights)
    )
    return float(covariance / np.sqrt(outcome_scale * nuisance_scale))


def _coverage(run: ValidatedCrossfitRun) -> dict[str, Any]:
    frame = run.scores
    if run.n_input_endpoint_rows < len(frame) or run.n_input_endpoint_rows < 1:
        raise ValueError("Endpoint input-row coverage denominator is invalid")
    return {
        "lineage": run.lineage,
        "fine_type": run.fine_type,
        "gp_id": run.gp_id,
        "seed": run.seed,
        "n_observations": int(len(frame)),
        "n_biological_units": int(frame["biological_unit_id"].nunique()),
        "minimum_donors_in_observed_cohort": int(
            frame.groupby("dataset", observed=True)["biological_unit_id"]
            .nunique()
            .min()
        ),
        "state_observation_coverage": float(len(frame) / run.n_input_endpoint_rows),
        "total_cells": int(pd.to_numeric(frame["n_cells"]).sum()),
        "median_cells": float(pd.to_numeric(frame["n_cells"]).median()),
        "depth_partial_correlation": _donor_balanced_partial_correlation(
            frame, "n_cells"
        ),
        "composition_partial_correlation": _donor_balanced_partial_correlation(
            frame, "fine_type_fraction"
        ),
    }


def _seed_rank_stability(seed_selection: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for fine_type, group in seed_selection.groupby("fine_type", observed=True):
        pivot = group.pivot(
            index="gp_id",
            columns="seed",
            values="mean_absolute_standardized_slope_per_decade",
        )
        correlations: list[float] = []
        for first, second in combinations(pivot.columns, 2):
            values = pivot[[first, second]].dropna()
            if len(values) < 2:
                continue
            correlation = float(spearmanr(values[first], values[second]).statistic)
            if np.isfinite(correlation):
                correlations.append(correlation)
        output[str(fine_type)] = min(correlations) if correlations else float("nan")
    return output


def _validate_simple_baseline(
    path: Path,
    *,
    lineage: str,
    heldout_dataset: str,
    training_datasets: Sequence[str],
    candidate_keys: set[tuple[str, str, str]],
    score_column: str,
    transferability: TransferabilityConfig,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    path = Path(path).resolve()
    baseline = pd.read_parquet(path)
    required = {
        *KEY_COLUMNS,
        "dataset",
        "biological_unit_id",
        "age",
        "sex",
        score_column,
    }
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise ValueError(f"Simple baseline lacks required columns: {missing}")
    baseline = baseline.loc[baseline["lineage"].astype(str).eq(lineage)].copy()
    observed_datasets = set(baseline["dataset"].astype(str))
    if heldout_dataset in observed_datasets:
        raise ValueError("Simple baseline contains the held-out query cohort")
    if observed_datasets != set(map(str, training_datasets)):
        raise ValueError("Simple baseline cohorts differ from the training cohorts")
    observed_keys = set(
        baseline.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None)
    )
    missing_keys = sorted(candidate_keys - observed_keys)
    if missing_keys:
        raise ValueError(
            f"Simple baseline omits candidate endpoints: {missing_keys[:5]}"
        )
    baseline = baseline.loc[
        baseline.loc[:, KEY_COLUMNS]
        .astype(str)
        .apply(tuple, axis=1)
        .isin(candidate_keys)
    ].copy()
    result = select_transferable_gene_programs(
        baseline,
        score_column=score_column,
        training_datasets=training_datasets,
        excluded_datasets=(heldout_dataset,),
        config=transferability,
    )
    renamed = result.selection.rename(
        columns={
            column: f"baseline_{column}"
            for column in result.selection.columns
            if column not in KEY_COLUMNS
        }
    )
    return renamed, {
        "path": str(path),
        "sha256": sha256_file(path),
        "score_column": score_column,
        "query_data_consulted": False,
    }


def select_transferable_tripso_gps(
    reference_manifests: Sequence[Path],
    *,
    lineage: str,
    fold_id: str,
    heldout_dataset: str,
    training_datasets: Sequence[str],
    required_seeds: Sequence[int],
    weighting_scheme: str = "donor_pooled",
    config: TripsoGPSelectionConfig = TripsoGPSelectionConfig(),
    simple_baseline: Path | None = None,
    simple_baseline_score_column: str = "gp_score",
) -> TripsoGPSelectionResult:
    """Select GP/fine-type endpoints using only training cross-fitted ages."""

    config.validate()
    if weighting_scheme not in {"donor_pooled", "cohort_balanced"}:
        raise ValueError("Unknown healthy-reference weighting scheme")
    paths = tuple(dict.fromkeys(Path(path).resolve() for path in reference_manifests))
    if not paths:
        raise ValueError("At least one healthy-reference manifest is required")
    cohorts = tuple(sorted(set(map(str, training_datasets))))
    seeds = tuple(sorted(set(map(int, required_seeds))))
    if len(cohorts) < config.minimum_cohorts:
        raise ValueError("Required training cohorts cannot meet minimum_cohorts")
    if len(seeds) < 2 or any(seed < 0 for seed in seeds):
        raise ValueError("At least two nonnegative required seeds must be declared")
    if heldout_dataset in cohorts:
        raise ValueError("Held-out query cohort cannot be a training cohort")

    runs = [
        validate_crossfit_reference_run(
            path,
            lineage=lineage,
            fold_id=fold_id,
            heldout_dataset=heldout_dataset,
            training_datasets=cohorts,
            weighting_scheme=weighting_scheme,
        )
        for path in paths
    ]
    signatures = {run.model_signature for run in runs}
    if len(signatures) != 1:
        raise ValueError("Reference runs mix scientific model configurations")
    observed_seeds = tuple(sorted({run.seed for run in runs}))
    if observed_seeds != seeds:
        raise ValueError(
            f"Reference seeds differ from required seeds: {observed_seeds} != {seeds}"
        )
    model_ids_by_seed = {
        seed: {run.model_id for run in runs if run.seed == seed} for seed in seeds
    }
    if any(len(values) != 1 for values in model_ids_by_seed.values()):
        raise ValueError("Each seed must contribute exactly one frozen TRIPSO model")
    candidates_by_seed = {
        seed: {
            (run.lineage, run.fine_type, run.gp_id) for run in runs if run.seed == seed
        }
        for seed in seeds
    }
    candidate_sets = list(candidates_by_seed.values())
    if not candidate_sets[0] or any(
        value != candidate_sets[0] for value in candidate_sets
    ):
        raise ValueError(
            "Every candidate GP/fine-type endpoint must exist for every seed"
        )
    candidate_keys = candidate_sets[0]
    run_keys = [(run.lineage, run.fine_type, run.gp_id, run.seed) for run in runs]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Duplicate reference run for one GP/fine-type/seed")

    combined: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for run in runs:
        frame = run.scores.copy()
        frame["seed"] = run.seed
        frame["model_id"] = run.model_id
        combined.append(frame)
        coverage_rows.append(_coverage(run))
    scores = pd.concat(combined, ignore_index=True)
    coverage = pd.DataFrame.from_records(coverage_rows)

    effect_frames: list[pd.DataFrame] = []
    seed_selections: list[pd.DataFrame] = []
    for seed in seeds:
        selected = scores.loc[scores["seed"].eq(seed)].copy()
        result = select_transferable_gene_programs(
            selected,
            score_column="predicted_gp_age",
            training_datasets=cohorts,
            excluded_datasets=(heldout_dataset,),
            config=config.transferability(),
        )
        effects = result.effects.copy()
        effects["seed"] = seed
        effects["model_id"] = next(iter(model_ids_by_seed[seed]))
        effects["fold_id"] = fold_id
        effects["heldout_dataset"] = heldout_dataset
        effects["score_column"] = "predicted_gp_age"
        effect_frames.append(effects)
        seed_selection = result.selection.copy()
        seed_selection["seed"] = seed
        seed_selection["model_id"] = next(iter(model_ids_by_seed[seed]))
        seed_selections.append(seed_selection)
    effects = (
        pd.concat(effect_frames, ignore_index=True)
        .sort_values([*KEY_COLUMNS, "seed", "dataset"])
        .reset_index(drop=True)
    )
    seed_selection = pd.concat(seed_selections, ignore_index=True).merge(
        coverage, on=[*KEY_COLUMNS, "seed"], validate="one_to_one"
    )
    rank_stability = _seed_rank_stability(seed_selection)

    baseline_selection: pd.DataFrame | None = None
    baseline_provenance: Mapping[str, Any] | None = None
    if simple_baseline is not None:
        baseline_selection, baseline_provenance = _validate_simple_baseline(
            simple_baseline,
            lineage=lineage,
            heldout_dataset=heldout_dataset,
            training_datasets=cohorts,
            candidate_keys=candidate_keys,
            score_column=simple_baseline_score_column,
            transferability=config.transferability(),
        )

    rows: list[dict[str, Any]] = []
    for key, group in seed_selection.groupby(
        list(KEY_COLUMNS), observed=True, sort=True
    ):
        candidate = tuple(map(str, key if isinstance(key, tuple) else (key,)))
        if tuple(sorted(group["seed"].astype(int))) != seeds:
            raise AssertionError("Candidate seed coverage changed after validation")
        slopes = group["meta_age_slope_per_year"].to_numpy(dtype=float)
        finite_slopes = slopes[np.isfinite(slopes)]
        pooled_sign = np.sign(np.median(finite_slopes)) if len(finite_slopes) else 0.0
        seed_sign_concordance = (
            float(np.mean(np.sign(finite_slopes) == pooled_sign))
            if len(finite_slopes) and pooled_sign != 0
            else 0.0
        )
        standardized = group["mean_absolute_standardized_slope_per_decade"].to_numpy(
            dtype=float
        )
        finite_standardized = standardized[np.isfinite(standardized)]
        record: dict[str, Any] = {
            **dict(zip(KEY_COLUMNS, candidate, strict=True)),
            "fold_id": fold_id,
            "heldout_dataset": heldout_dataset,
            "n_seeds": len(group),
            "seed_ids": ",".join(map(str, seeds)),
            "minimum_eligible_cohorts_across_seeds": int(
                group["n_cohorts_eligible"].min()
            ),
            "minimum_cohort_sign_concordance_across_seeds": float(
                group["sign_concordance"].min()
            ),
            "maximum_heterogeneity_i2_across_seeds": float(
                group["heterogeneity_i2"].max()
            ),
            "maximum_meta_fdr_across_seeds": float(group["meta_fdr"].max()),
            "median_meta_age_slope_per_year_across_seeds": (
                float(np.median(finite_slopes)) if len(finite_slopes) else np.nan
            ),
            "mean_absolute_standardized_slope_per_decade_across_seeds": (
                float(np.mean(finite_standardized))
                if len(finite_standardized)
                else np.nan
            ),
            "seed_effect_sd": (
                float(np.std(finite_standardized, ddof=1))
                if len(finite_standardized) > 1
                else 0.0
            ),
            "seed_retention_fraction": float(group["retained"].astype(bool).mean()),
            "seed_sign_concordance": seed_sign_concordance,
            "seed_rank_correlation": rank_stability[candidate[1]],
            "minimum_state_observation_coverage_across_seeds": float(
                group["state_observation_coverage"].min()
            ),
            "minimum_observed_donors_across_seeds": int(
                group["n_biological_units"].min()
            ),
            "minimum_median_cells_across_seeds": float(group["median_cells"].min()),
            "maximum_absolute_depth_partial_correlation_across_seeds": float(
                group["depth_partial_correlation"].abs().max()
            ),
            "maximum_absolute_composition_partial_correlation_across_seeds": float(
                group["composition_partial_correlation"].abs().max()
            ),
            "seed_failure_reasons": ";".join(
                sorted(
                    {
                        str(reason)
                        for reason in group.loc[
                            ~group["retained"].astype(bool), "selection_reason"
                        ]
                    }
                )
            ),
        }
        rows.append(record)
    selection = pd.DataFrame.from_records(rows)

    if baseline_selection is not None:
        selection = selection.merge(
            baseline_selection,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        baseline_effect = "baseline_mean_absolute_standardized_slope_per_decade"
        if selection[baseline_effect].isna().any():
            raise ValueError("Simple baseline comparison is incomplete")
        selection["standardized_effect_improvement_over_baseline"] = (
            selection["mean_absolute_standardized_slope_per_decade_across_seeds"]
            - selection[baseline_effect]
        )
    else:
        selection["standardized_effect_improvement_over_baseline"] = np.nan

    retained: list[bool] = []
    reasons: list[str] = []
    for row in selection.itertuples(index=False):
        failed: list[str] = []
        if row.minimum_eligible_cohorts_across_seeds < config.minimum_cohorts:
            failed.append("too_few_cohorts")
        if (
            not np.isfinite(row.minimum_cohort_sign_concordance_across_seeds)
            or row.minimum_cohort_sign_concordance_across_seeds
            < config.minimum_sign_concordance
        ):
            failed.append("inconsistent_cohort_direction")
        if (
            not np.isfinite(row.maximum_heterogeneity_i2_across_seeds)
            or row.maximum_heterogeneity_i2_across_seeds > config.maximum_i2
        ):
            failed.append("excessive_heterogeneity")
        if (
            not np.isfinite(row.maximum_meta_fdr_across_seeds)
            or row.maximum_meta_fdr_across_seeds > config.maximum_fdr
        ):
            failed.append("age_association_fdr")
        if (
            not np.isfinite(
                row.mean_absolute_standardized_slope_per_decade_across_seeds
            )
            or row.mean_absolute_standardized_slope_per_decade_across_seeds
            < config.minimum_absolute_standardized_slope_per_decade
        ):
            failed.append("effect_too_small")
        if row.seed_retention_fraction < config.minimum_seed_retention_fraction:
            failed.append("unstable_transferability_across_seeds")
        if row.seed_sign_concordance < config.minimum_seed_sign_concordance:
            failed.append("inconsistent_seed_direction")
        if config.minimum_seed_rank_correlation is not None and (
            not np.isfinite(row.seed_rank_correlation)
            or row.seed_rank_correlation < config.minimum_seed_rank_correlation
        ):
            failed.append("unstable_seed_rank")
        if (
            row.minimum_state_observation_coverage_across_seeds
            < config.minimum_state_observation_coverage
        ):
            failed.append("insufficient_state_coverage")
        if row.minimum_median_cells_across_seeds < config.minimum_median_cells:
            failed.append("insufficient_cell_coverage")
        if (
            row.maximum_absolute_depth_partial_correlation_across_seeds
            > config.maximum_absolute_depth_partial_correlation
        ):
            failed.append("excessive_cell_depth_dependence")
        if (
            row.maximum_absolute_composition_partial_correlation_across_seeds
            > config.maximum_absolute_composition_partial_correlation
        ):
            failed.append("excessive_composition_dependence")
        if (
            config.maximum_seed_effect_sd is not None
            and row.seed_effect_sd > config.maximum_seed_effect_sd
        ):
            failed.append("excessive_seed_effect_variability")
        if config.minimum_baseline_standardized_improvement is not None:
            if simple_baseline is None:
                failed.append("baseline_comparison_required")
            elif (
                row.standardized_effect_improvement_over_baseline
                < config.minimum_baseline_standardized_improvement
            ):
                failed.append("does_not_improve_simple_baseline")
        retained.append(not failed)
        reasons.append("retained" if not failed else "|".join(failed))
    selection["retained"] = retained
    selection["selection_reason"] = reasons
    selection = selection.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

    selected_endpoints = [
        {
            "lineage": str(row.lineage),
            "fine_type": str(row.fine_type),
            "gp_id": str(row.gp_id),
        }
        for row in selection.loc[selection["retained"]].itertuples(index=False)
    ]
    manifest: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "status": "complete" if selected_endpoints else "complete_no_candidates",
        "method": "donor_crossfit_predicted_gp_age",
        "score_column": "predicted_gp_age",
        "raw_tripso_coordinates_used_for_selection": False,
        "query_data_consulted": False,
        "reference_role_required": "reference",
        "lineage": lineage,
        "fold_id": fold_id,
        "heldout_dataset": heldout_dataset,
        "training_datasets": list(cohorts),
        "required_seeds": list(seeds),
        "weighting_scheme": weighting_scheme,
        "model_signature": next(iter(signatures)),
        "criteria": asdict(config),
        "n_candidates": len(selection),
        "n_selected_endpoints": len(selected_endpoints),
        "selected_endpoints": selected_endpoints,
        "selected_program_ids": sorted(
            {str(value["gp_id"]) for value in selected_endpoints}
        ),
        "input_artifacts": [
            {
                "lineage": run.lineage,
                "fine_type": run.fine_type,
                "gp_id": run.gp_id,
                "seed": run.seed,
                **dict(run.provenance),
            }
            for run in sorted(
                runs, key=lambda item: (item.fine_type, item.gp_id, item.seed)
            )
        ],
        "simple_baseline": baseline_provenance,
    }
    return TripsoGPSelectionResult(
        effects=effects, selection=selection, manifest=manifest
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_tripso_gp_selection(
    result: TripsoGPSelectionResult,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Atomically write selection tables and a canonical self-hashed manifest."""

    output_dir = Path(output_dir).resolve()
    effects_path = output_dir / "tripso_gp_cohort_seed_effects.parquet"
    selection_path = output_dir / "tripso_gp_selection.parquet"
    manifest_path = output_dir / "selected_tripso_gps.json"
    existing = [
        path for path in (effects_path, selection_path, manifest_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite GP selection outputs: {existing}")
    _atomic_parquet(effects_path, result.effects)
    _atomic_parquet(selection_path, result.selection)
    payload = {
        **dict(result.manifest),
        "outputs": {
            "cohort_seed_effects": {
                "path": effects_path.name,
                "sha256": sha256_file(effects_path),
            },
            "selection_audit": {
                "path": selection_path.name,
                "sha256": sha256_file(selection_path),
            },
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return effects_path, selection_path, manifest_path


def validate_tripso_gp_selection_manifest(path: Path) -> dict[str, Any]:
    """Validate a frozen selector artifact before downstream query scoring."""

    path = Path(path).resolve()
    manifest = _read_json(path, "TRIPSO GP selection manifest")
    if manifest.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError(f"Unsupported TRIPSO GP selection manifest: {path}")
    content = dict(manifest)
    claimed = content.pop("manifest_sha256", None)
    if claimed != stable_hash(content):
        raise ValueError("TRIPSO GP selection manifest content hash does not match")
    if (
        manifest.get("query_data_consulted") is not False
        or manifest.get("raw_tripso_coordinates_used_for_selection") is not False
        or manifest.get("reference_role_required") != "reference"
        or manifest.get("method") != "donor_crossfit_predicted_gp_age"
        or manifest.get("score_column") != "predicted_gp_age"
    ):
        raise ValueError("TRIPSO GP selection does not prove the approved input scope")
    heldout = str(manifest.get("heldout_dataset", ""))
    cohorts = tuple(map(str, manifest.get("training_datasets", ())))
    seeds = tuple(map(int, manifest.get("required_seeds", ())))
    if (
        not heldout
        or not cohorts
        or heldout in cohorts
        or len(cohorts) != len(set(cohorts))
        or len(seeds) < 2
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("TRIPSO GP selection has invalid cohort or seed scope")

    outputs = manifest.get("outputs")
    expected_outputs = {
        "cohort_seed_effects": "tripso_gp_cohort_seed_effects.parquet",
        "selection_audit": "tripso_gp_selection.parquet",
    }
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected_outputs):
        raise ValueError("TRIPSO GP selection output binding is incomplete")
    resolved_outputs: dict[str, Path] = {}
    for name, expected_name in expected_outputs.items():
        record = outputs[name]
        if not isinstance(record, Mapping) or record.get("path") != expected_name:
            raise ValueError("TRIPSO GP selection output name is invalid")
        output = _bound_path(path.parent, record.get("path"), "selection output")
        if output.parent != path.parent:
            raise ValueError(
                "TRIPSO GP selection output escaped its artifact directory"
            )
        if not output.is_file() or sha256_file(output) != record.get("sha256"):
            raise ValueError("TRIPSO GP selection output is missing or changed")
        resolved_outputs[name] = output

    selection = pd.read_parquet(resolved_outputs["selection_audit"])
    required_selection = {*KEY_COLUMNS, "retained", "selection_reason"}
    missing = sorted(required_selection - set(selection.columns))
    if missing or selection["retained"].isna().any():
        raise ValueError(f"TRIPSO GP selection audit is incomplete: {missing}")
    if len(selection) != int(manifest.get("n_candidates", -1)):
        raise ValueError("TRIPSO GP candidate count differs from its audit table")
    selected_from_table = sorted(
        selection.loc[selection["retained"].eq(True), list(KEY_COLUMNS)]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    selected_raw = manifest.get("selected_endpoints")
    if not isinstance(selected_raw, list):
        raise ValueError("TRIPSO GP selected_endpoints must be a list")
    selected_from_manifest: list[tuple[str, str, str]] = []
    for endpoint in selected_raw:
        if not isinstance(endpoint, Mapping) or set(endpoint) != set(KEY_COLUMNS):
            raise ValueError("TRIPSO GP selected endpoint identity is invalid")
        values = tuple(str(endpoint[column]).strip() for column in KEY_COLUMNS)
        if any(not value for value in values):
            raise ValueError("TRIPSO GP selected endpoint identity is empty")
        selected_from_manifest.append(values)
    if (
        len(selected_from_manifest) != len(set(selected_from_manifest))
        or sorted(selected_from_manifest) != selected_from_table
        or len(selected_from_manifest) != int(manifest.get("n_selected_endpoints", -1))
    ):
        raise ValueError("TRIPSO GP selected endpoints differ from the audit table")
    expected_programs = sorted({values[2] for values in selected_from_manifest})
    if manifest.get("selected_program_ids") != expected_programs:
        raise ValueError("TRIPSO GP selected program IDs are inconsistent")
    expected_status = "complete" if selected_from_manifest else "complete_no_candidates"
    if manifest.get("status") != expected_status:
        raise ValueError("TRIPSO GP selection status is inconsistent")

    effects = pd.read_parquet(resolved_outputs["cohort_seed_effects"])
    required_effects = {*KEY_COLUMNS, "dataset", "seed", "model_id", "fold_id"}
    if required_effects - set(effects.columns):
        raise ValueError("TRIPSO GP cohort/seed effect audit is incomplete")
    if set(effects["dataset"].astype(str)) != set(cohorts) or heldout in set(
        effects["dataset"].astype(str)
    ):
        raise ValueError("TRIPSO GP effects differ from the training cohort scope")
    if set(pd.to_numeric(effects["seed"], errors="coerce")) != set(seeds):
        raise ValueError("TRIPSO GP effects differ from the required seed scope")

    input_artifacts = manifest.get("input_artifacts")
    if not isinstance(input_artifacts, list) or not input_artifacts:
        raise ValueError("TRIPSO GP selection lacks its source artifact bindings")
    identities = {
        (
            str(record.get("lineage")),
            str(record.get("fine_type")),
            str(record.get("gp_id")),
            int(record.get("seed")),
        )
        for record in input_artifacts
        if isinstance(record, Mapping)
    }
    if len(identities) != len(input_artifacts) or {
        value[3] for value in identities
    } != set(seeds):
        raise ValueError("TRIPSO GP source artifact identities are incomplete")
    return manifest
