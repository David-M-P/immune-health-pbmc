"""Donor-aware age support and cross-cohort slope diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .trajectory import reference_row_weights


def _aligned_training_arrays(
    ages: Sequence[float],
    sexes: Sequence[str],
    datasets: Sequence[str],
    biological_unit_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    age = np.asarray(ages, dtype=float)
    sex = np.asarray(sexes, dtype=str)
    dataset = np.asarray(datasets, dtype=str)
    donors = np.asarray(biological_unit_ids, dtype=str)
    if age.ndim != 1 or any(
        value.shape != age.shape for value in (sex, dataset, donors)
    ):
        raise ValueError("age-support metadata vectors must align")
    if len(age) == 0 or not np.isfinite(age).all():
        raise ValueError("training ages must be nonempty and finite")
    metadata = pd.DataFrame(
        {"donor": donors, "dataset": dataset, "sex": sex}
    ).drop_duplicates()
    if metadata["donor"].duplicated().any():
        raise ValueError("dataset and sex must be constant within each donor")
    return age, sex, dataset, donors


def query_age_support(
    training_ages: Sequence[float],
    training_sexes: Sequence[str],
    training_datasets: Sequence[str],
    training_biological_unit_ids: Sequence[str],
    query_ages: Sequence[float],
    query_sexes: Sequence[str],
    *,
    window_years: float = 5.0,
    minimum_cohorts: int = 3,
    minimum_donors: int = 20,
    weighting_scheme: str = "donor_pooled",
) -> pd.DataFrame:
    """Flag local age/sex support without using any query outcomes.

    A support cohort contains at least one same-sex reference donor within the
    configured age window.  ``common_cohort_range`` is stricter: the target age
    must fall inside the observed same-sex range of every training cohort.
    """

    age, sex, dataset, donors = _aligned_training_arrays(
        training_ages,
        training_sexes,
        training_datasets,
        training_biological_unit_ids,
    )
    target_age = np.asarray(query_ages, dtype=float)
    target_sex = np.asarray(query_sexes, dtype=str)
    if target_age.ndim != 1 or target_sex.shape != target_age.shape:
        raise ValueError("query ages and sexes must align")
    if not np.isfinite(target_age).all():
        raise ValueError("query ages must be finite")
    if window_years <= 0 or minimum_cohorts < 1 or minimum_donors < 1:
        raise ValueError("age-support thresholds must be positive")

    row_weights = reference_row_weights(donors, dataset, scheme=weighting_scheme)
    levels = tuple(sorted(np.unique(dataset)))
    rows: list[dict[str, object]] = []
    for query_index, (value, sex_value) in enumerate(
        zip(target_age, target_sex, strict=True)
    ):
        same_sex = sex == sex_value
        local = same_sex & (np.abs(age - value) <= window_years)
        local_donors = np.unique(donors[local])
        supporting = tuple(sorted(np.unique(dataset[local])))
        sex_age = age[same_sex]
        in_sex_range = bool(
            len(sex_age) and float(sex_age.min()) <= value <= float(sex_age.max())
        )
        ranges = {level: age[same_sex & (dataset == level)] for level in levels}
        all_ranges_present = all(len(values) for values in ranges.values())
        common_low = (
            max(float(values.min()) for values in ranges.values())
            if all_ranges_present
            else float("nan")
        )
        common_high = (
            min(float(values.max()) for values in ranges.values())
            if all_ranges_present
            else float("nan")
        )
        common = bool(all_ranges_present and common_low <= value <= common_high)
        enough = (
            len(supporting) >= minimum_cohorts and len(local_donors) >= minimum_donors
        )
        if not in_sex_range:
            status = "out_of_range"
        elif common and enough:
            status = "common_support"
        elif enough:
            status = "supported"
        else:
            status = "limited_support"
        rows.append(
            {
                "query_index": query_index,
                "age_support_status": status,
                "age_support_window_years": float(window_years),
                "n_support_cohorts": len(supporting),
                "support_cohorts": "|".join(supporting),
                "n_support_donors": len(local_donors),
                "effective_support_weight": float(row_weights[local].sum()),
                "in_training_sex_age_range": in_sex_range,
                "in_common_cohort_sex_age_range": common,
                "common_cohort_sex_age_min": common_low,
                "common_cohort_sex_age_max": common_high,
            }
        )
    return pd.DataFrame.from_records(rows)


def age_support_grid(
    ages: Sequence[float],
    sexes: Sequence[str],
    datasets: Sequence[str],
    biological_unit_ids: Sequence[str],
    *,
    window_years: float = 5.0,
    minimum_cohorts: int = 3,
    minimum_donors: int = 20,
    weighting_scheme: str = "donor_pooled",
) -> pd.DataFrame:
    """Evaluate support at each integer age for every observed sex label."""

    age, sex, dataset, donors = _aligned_training_arrays(
        ages, sexes, datasets, biological_unit_ids
    )
    grid = np.arange(np.floor(age.min()), np.ceil(age.max()) + 1.0)
    query_age = np.tile(grid, len(np.unique(sex)))
    query_sex = np.repeat(np.sort(np.unique(sex)), len(grid))
    result = query_age_support(
        age,
        sex,
        dataset,
        donors,
        query_age,
        query_sex,
        window_years=window_years,
        minimum_cohorts=minimum_cohorts,
        minimum_donors=minimum_donors,
        weighting_scheme=weighting_scheme,
    )
    result.insert(1, "age", query_age)
    result.insert(2, "sex", query_sex)
    return result


def _clustered_age_effect(
    values: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    donors: np.ndarray,
) -> tuple[float, float, float]:
    weights = reference_row_weights(donors)
    centered_age = ages - np.average(ages, weights=weights)
    sex_levels = tuple(sorted(np.unique(sexes)))
    columns = [np.ones(len(values)), centered_age]
    columns.extend((sexes == level).astype(float) for level in sex_levels[1:])
    design = np.column_stack(columns)
    weighted_cross = design.T @ (weights[:, None] * design)
    bread = np.linalg.pinv(weighted_cross)
    coefficients = bread @ (design.T @ (weights * values))
    residuals = values - design @ coefficients
    meat = np.zeros_like(weighted_cross)
    unique_donors = np.unique(donors)
    for donor in unique_donors:
        selected = donors == donor
        score = design[selected].T @ (weights[selected] * residuals[selected])
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n_groups = len(unique_donors)
    n_rows, n_parameters = design.shape
    if n_groups > 1 and n_rows > n_parameters:
        covariance *= (n_groups / (n_groups - 1)) * (
            (n_rows - 1) / (n_rows - n_parameters)
        )
    slope = float(coefficients[1])
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    mean = float(np.average(values, weights=weights))
    scale = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    standardized_decade = slope * 10.0 / scale if scale > 0 else float("nan")
    return slope, standard_error, standardized_decade


def cohort_feature_age_effects(
    features: np.ndarray,
    ages: Sequence[float],
    sexes: Sequence[str],
    biological_unit_ids: Sequence[str],
    datasets: Sequence[str],
    *,
    feature_ids: Sequence[str] | None = None,
    minimum_donors: int = 20,
    minimum_age_span: float = 10.0,
) -> pd.DataFrame:
    """Estimate sex-adjusted within-cohort age slopes with donor-clustered SEs."""

    age, sex, dataset, donors = _aligned_training_arrays(
        ages, sexes, datasets, biological_unit_ids
    )
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) != len(age):
        raise ValueError("features and age-effect metadata must align")
    if not np.isfinite(matrix).all():
        raise ValueError("age-effect features must be finite")
    if minimum_donors < 2 or minimum_age_span <= 0:
        raise ValueError("age-effect support thresholds are invalid")
    labels = (
        tuple(f"feature_{index}" for index in range(matrix.shape[1]))
        if feature_ids is None
        else tuple(map(str, feature_ids))
    )
    if len(labels) != matrix.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("feature_ids must be unique and match the feature matrix")

    rows: list[dict[str, object]] = []
    for level in sorted(np.unique(dataset)):
        selected = dataset == level
        n_donors = len(np.unique(donors[selected]))
        age_span = float(np.ptp(age[selected]))
        eligible = n_donors >= minimum_donors and age_span >= minimum_age_span
        reason = (
            "eligible"
            if eligible
            else "too_few_donors"
            if n_donors < minimum_donors
            else "insufficient_age_span"
        )
        for feature_index, feature_id in enumerate(labels):
            if eligible:
                slope, standard_error, standardized = _clustered_age_effect(
                    matrix[selected, feature_index],
                    age[selected],
                    sex[selected],
                    donors[selected],
                )
                z_score = slope / standard_error if standard_error > 0 else float("nan")
                p_value = (
                    float(2.0 * norm.sf(abs(z_score)))
                    if np.isfinite(z_score)
                    else float("nan")
                )
            else:
                slope = standard_error = standardized = z_score = p_value = float("nan")
            rows.append(
                {
                    "dataset": str(level),
                    "feature_id": feature_id,
                    "feature_index": feature_index,
                    "n_donors": n_donors,
                    "n_rows": int(selected.sum()),
                    "age_min": float(age[selected].min()),
                    "age_max": float(age[selected].max()),
                    "age_span": age_span,
                    "eligible": eligible,
                    "eligibility_reason": reason,
                    "age_slope_per_year": slope,
                    "age_slope_se": standard_error,
                    "age_slope_per_decade": slope * 10.0,
                    "standardized_age_slope_per_decade": standardized,
                    "z_score": z_score,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame.from_records(rows)
