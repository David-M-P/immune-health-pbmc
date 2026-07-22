"""Input and output contracts for donor-aware immune-health analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.data.ids import add_stable_identifiers, validate_identifier_contract

REQUIRED_CELL_FIELDS = (
    "dataset",
    "donor_id",
    "sample_id",
    "age",
    "sex",
    "lineage",
    "ctype_low",
    "ctype_low_conf",
)

FINE_TYPE_GP_REQUIRED_COLUMNS = (
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
    "covariance_trace",
    "covariance_logdet",
    "dispersion",
    "predicted_gp_age",
    "gp_age_acceleration",
    "age_matched_distance",
    "off_trajectory_distance",
    "sliced_wasserstein_distance",
    "gaussian_wasserstein_distance",
    "cell_sampling_se",
    "reference_sampling_se",
    "seed_sd",
    "model_id",
    "fold_id",
    "seed",
    "annotation_version",
    "gp_library_version",
    "reference_version",
)

LINEAGE_GP_REQUIRED_COLUMNS = (
    "dataset",
    "donor_id",
    "observation_id",
    "age",
    "sex",
    "lineage",
    "gp_id",
    "observed_mixture_score",
    "composition_standardized_state_score",
    "composition_only_score",
    "within_fine_type_heterogeneity",
    "between_fine_type_heterogeneity",
    "total_lineage_heterogeneity",
)


@dataclass(frozen=True)
class CountValidation:
    shape: tuple[int, int]
    sparse: bool
    dtype: str
    inspected_values: int
    minimum: float | None
    maximum: float | None
    nonfinite_values: int
    negative_values: int
    non_integer_like_values: int


def validate_cell_metadata(
    frame: pd.DataFrame,
    *,
    allowed_datasets: Iterable[str] | None = None,
    allowed_lineages: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate metadata and return a copy with all stable identifiers."""
    missing_columns = sorted(set(REQUIRED_CELL_FIELDS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Cell metadata is missing required columns: {missing_columns}"
        )
    result = add_stable_identifiers(frame)
    id_report = validate_identifier_contract(result)

    age = pd.to_numeric(result["age"], errors="coerce")
    if age.isna().any():
        raise ValueError(
            f"Age is missing or nonnumeric for {int(age.isna().sum())} cells"
        )
    result["age"] = age

    sex = result["sex"].astype("string").str.strip().str.lower()
    invalid_sex = ~sex.isin(["female", "male", "unknown"])
    if invalid_sex.any():
        values = sorted(sex.loc[invalid_sex].dropna().unique().tolist())
        raise ValueError(f"Unexpected sex values: {values}")
    result["sex"] = sex

    if allowed_datasets is not None:
        unexpected = sorted(set(result["dataset"].astype(str)) - set(allowed_datasets))
        if unexpected:
            raise ValueError(f"Unexpected datasets: {unexpected}")
    if allowed_lineages is not None:
        unexpected = sorted(set(result["lineage"].astype(str)) - set(allowed_lineages))
        if unexpected:
            raise ValueError(f"Unexpected lineages: {unexpected}")

    report = {
        **id_report,
        "n_cells": len(result),
        "n_datasets": int(result["dataset"].nunique()),
        "n_lineages": int(result["lineage"].nunique()),
        "n_fine_types": int(result["ctype_low"].nunique()),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
    }
    return result, report


def _sample_values(values: Any, max_values: int) -> np.ndarray:
    array = np.asarray(values)
    if array.size <= max_values:
        return array.ravel()
    positions = np.linspace(0, array.size - 1, max_values, dtype=np.int64)
    return array.ravel()[positions]


def validate_raw_counts(
    counts: Any,
    *,
    max_values: int = 1_000_000,
    integer_tolerance: float = 1e-6,
) -> CountValidation:
    """Validate nonnegative integer-like raw counts without densifying sparse input."""
    if not hasattr(counts, "shape") or len(counts.shape) != 2:
        raise ValueError("Counts must be a two-dimensional matrix")
    is_sparse = sparse.issparse(counts)
    stored = counts.data if is_sparse else counts
    values = _sample_values(stored, max_values).astype(np.float64, copy=False)
    nonfinite = int((~np.isfinite(values)).sum())
    negative = int((values < 0).sum())
    fractional = int((np.abs(values - np.rint(values)) > integer_tolerance).sum())
    if nonfinite:
        raise ValueError(f"Counts contain {nonfinite} nonfinite sampled values")
    if negative:
        raise ValueError(f"Counts contain {negative} negative sampled values")
    if fractional:
        raise ValueError(f"Counts contain {fractional} non-integer-like sampled values")
    dtype = getattr(counts, "dtype", None)
    if dtype is None:
        dtype = np.asarray(counts).dtype
    return CountValidation(
        shape=(int(counts.shape[0]), int(counts.shape[1])),
        sparse=is_sparse,
        dtype=str(dtype),
        inspected_values=int(values.size),
        minimum=float(values.min()) if values.size else None,
        maximum=float(values.max()) if values.size else None,
        nonfinite_values=nonfinite,
        negative_values=negative,
        non_integer_like_values=fractional,
    )


def gene_coverage(
    query_genes: Iterable[str], frozen_vocabulary: Iterable[str]
) -> dict[str, Any]:
    """Report query coverage against a frozen training vocabulary."""
    query = set(map(str, query_genes))
    frozen = tuple(map(str, frozen_vocabulary))
    present = [gene for gene in frozen if gene in query]
    missing = [gene for gene in frozen if gene not in query]
    return {
        "n_frozen_genes": len(frozen),
        "n_present": len(present),
        "n_missing": len(missing),
        "coverage": len(present) / len(frozen) if frozen else 1.0,
        "missing_genes": missing,
    }


def require_gene_coverage(
    query_genes: Iterable[str],
    frozen_vocabulary: Iterable[str],
    minimum_coverage: float,
    *,
    allow_low_coverage: bool = False,
) -> dict[str, Any]:
    report = gene_coverage(query_genes, frozen_vocabulary)
    if report["coverage"] < minimum_coverage and not allow_low_coverage:
        raise ValueError(
            "Query gene coverage below safety threshold: "
            f"{report['coverage']:.3f} < {minimum_coverage:.3f}. "
            "Use the explicit low-coverage override only after reviewing the report."
        )
    report["override_used"] = bool(
        allow_low_coverage and report["coverage"] < minimum_coverage
    )
    return report


def validate_table_schema(
    frame: pd.DataFrame, required_columns: Iterable[str], table_name: str
) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
    object_cells = [
        column
        for column in frame.columns
        if frame[column].dtype == object
        and frame[column].map(lambda value: isinstance(value, (dict, set))).any()
    ]
    if object_cells:
        raise ValueError(
            f"{table_name} contains unsupported Python objects in: {object_cells}"
        )
