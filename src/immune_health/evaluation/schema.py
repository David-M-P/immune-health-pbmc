"""Validation and assembly of donor-level GP output tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

FINE_TYPE_GP_COLUMNS = (
    "dataset",
    "donor_id",
    "biological_unit_id",
    "sample_id",
    "source_observation_id",
    "observation_id",
    "age",
    "sex",
    "lineage",
    "fine_type",
    "gp_id",
    "n_cells",
    "fine_type_fraction",
    "annotation_confidence_summary",
    "location_summary",
    "covariance_summary",
    "covariance_trace",
    "covariance_logdet",
    "dispersion",
    "age_axis_q10",
    "age_axis_q25",
    "age_axis_q50",
    "age_axis_q75",
    "age_axis_q90",
    "healthy_tail_fraction",
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

LINEAGE_GP_COLUMNS = (
    "dataset",
    "donor_id",
    "biological_unit_id",
    "sample_id",
    "source_observation_id",
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

PROVENANCE_COLUMNS = (
    "model_id",
    "fold_id",
    "seed",
    "annotation_version",
    "gp_library_version",
    "reference_version",
)

FINE_TYPE_OPTIONAL_NUMERIC = (
    "predicted_gp_age",
    "gp_age_acceleration",
    "age_matched_distance",
    "off_trajectory_distance",
    "sliced_wasserstein_distance",
    "gaussian_wasserstein_distance",
    "cell_sampling_se",
    "reference_sampling_se",
    "seed_sd",
)

LINEAGE_SCORE_COLUMNS = (
    "observed_mixture_score",
    "composition_standardized_state_score",
    "composition_only_score",
    "within_fine_type_heterogeneity",
    "between_fine_type_heterogeneity",
    "total_lineage_heterogeneity",
    "cell_sampling_se",
    "reference_sampling_se",
    "seed_sd",
)


def _add_required_values(
    table: pd.DataFrame,
    provenance: Mapping[str, object],
    optional_numeric: Sequence[str],
) -> pd.DataFrame:
    result = table.copy()
    absent_provenance = [
        column
        for column in PROVENANCE_COLUMNS
        if column not in provenance and column not in result
    ]
    if absent_provenance:
        raise ValueError(f"missing required provenance values: {absent_provenance}")
    for column in PROVENANCE_COLUMNS:
        if column not in result:
            result[column] = provenance[column]
    for column in optional_numeric:
        if column not in result:
            result[column] = np.nan
    return result


def finalize_fine_type_output(
    aggregation_table: pd.DataFrame,
    provenance: Mapping[str, object],
) -> pd.DataFrame:
    """Add explicit unavailable score fields and validate the final table."""

    result = _add_required_values(
        aggregation_table, provenance, FINE_TYPE_OPTIONAL_NUMERIC
    )
    validate_fine_type_gp_schema(result)
    return result.loc[:, FINE_TYPE_GP_COLUMNS]


def finalize_lineage_output(
    lineage_table: pd.DataFrame,
    provenance: Mapping[str, object],
) -> pd.DataFrame:
    """Add explicit unavailable lineage fields and validate the final table."""

    result = _add_required_values(lineage_table, provenance, LINEAGE_SCORE_COLUMNS)
    validate_lineage_gp_schema(result)
    return result.loc[:, LINEAGE_GP_COLUMNS]


def _validate_identifiers(table: pd.DataFrame) -> None:
    expected_biological = (
        table["dataset"].astype("string") + "::" + table["donor_id"].astype("string")
    )
    expected_observation = (
        expected_biological + "::" + table["sample_id"].astype("string")
    )
    if "source_observation_id" in table:
        expected_source = (
            table["dataset"].astype("string")
            + "::"
            + table["sample_id"].astype("string")
        )
        if not table["source_observation_id"].astype("string").equals(expected_source):
            raise ValueError(
                "source_observation_id violates dataset::sample_id contract"
            )
    if not table["biological_unit_id"].astype("string").equals(expected_biological):
        raise ValueError("biological_unit_id violates dataset::donor_id contract")
    if not table["observation_id"].astype("string").equals(expected_observation):
        raise ValueError(
            "observation_id violates dataset::donor_id::sample_id contract"
        )


def _reject_python_objects(table: pd.DataFrame) -> None:
    for column in table.select_dtypes(include="object"):
        invalid = table[column].map(
            lambda value: isinstance(value, (dict, list, tuple, np.ndarray))
        )
        if invalid.any():
            raise ValueError(
                f"{column!r} contains arbitrary Python objects; serialize arrays"
            )


def _validate_json_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        for value in table[column].dropna():
            if not isinstance(value, str):
                raise ValueError(f"{column!r} must contain JSON text or missing values")
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(f"{column!r} JSON must encode an array")


def _validate_common(table: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"output table is missing columns: {missing}")
    if table.empty:
        raise ValueError("output table is empty")
    identifiers = (
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "observation_id",
        "lineage",
        "gp_id",
    )
    if table.loc[:, identifiers].isna().any().any():
        raise ValueError("required output identifiers contain missing values")
    if table.loc[:, PROVENANCE_COLUMNS].isna().any().any():
        raise ValueError("required model/reference provenance contains missing values")
    _validate_identifiers(table)
    _reject_python_objects(table)


def validate_fine_type_gp_schema(table: pd.DataFrame) -> None:
    """Validate keys, scalar columns and missing rare-type state semantics."""

    _validate_common(table, FINE_TYPE_GP_COLUMNS)
    if table["fine_type"].isna().any():
        raise ValueError("fine_type contains missing values")
    key = [
        "observation_id",
        "lineage",
        "fine_type",
        "gp_id",
        "model_id",
        "fold_id",
        "seed",
    ]
    if table.duplicated(key).any():
        raise ValueError("fine-type GP output contains duplicate stable keys")
    cells = pd.to_numeric(table["n_cells"], errors="coerce")
    if (
        cells.isna().any()
        or (cells < 0).any()
        or not np.allclose(cells, np.rint(cells))
    ):
        raise ValueError("n_cells must be nonnegative integers")
    fractions = pd.to_numeric(table["fine_type_fraction"], errors="coerce")
    if fractions.isna().any() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("fine_type_fraction must be between zero and one")
    totals = table.groupby(["observation_id", "lineage", "gp_id"], observed=True)[
        "fine_type_fraction"
    ].sum()
    if not np.allclose(totals, 1.0, atol=1e-7):
        raise ValueError("fine-type fractions do not close to one")
    _validate_json_columns(table, ("location_summary", "covariance_summary"))

    if "state_available" in table:
        rare = ~table["state_available"].astype(bool)
        scalar_state = ("covariance_trace", "covariance_logdet", "dispersion")
        serialized = table.loc[rare, ["location_summary", "covariance_summary"]]
        if serialized.notna().any().any():
            raise ValueError("unmeasured rare-type state must be missing, not zero")
        if table.loc[rare, scalar_state].notna().any().any():
            raise ValueError("unmeasured rare-type dispersion must be missing")


def validate_lineage_gp_schema(table: pd.DataFrame) -> None:
    """Validate a whole-lineage decomposition table."""

    _validate_common(table, LINEAGE_GP_COLUMNS)
    key = ["observation_id", "lineage", "gp_id", "model_id", "fold_id", "seed"]
    if table.duplicated(key).any():
        raise ValueError("lineage GP output contains duplicate stable keys")
    values = table.loc[
        :,
        [
            "within_fine_type_heterogeneity",
            "between_fine_type_heterogeneity",
            "total_lineage_heterogeneity",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    if (values.loc[complete] < -1e-10).any().any():
        raise ValueError("heterogeneity components cannot be negative")
    if not np.allclose(
        values.loc[complete, "total_lineage_heterogeneity"],
        values.loc[complete, "within_fine_type_heterogeneity"]
        + values.loc[complete, "between_fine_type_heterogeneity"],
        atol=1e-7,
    ):
        raise ValueError("total heterogeneity is not within + between")
