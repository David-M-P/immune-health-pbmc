"""Training-only selection of age-associated, cross-cohort gene programs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from immune_health.healthy_reference.diagnostics import cohort_feature_age_effects


@dataclass(frozen=True)
class TransferabilityConfig:
    """Prespecified support and heterogeneity thresholds for GP selection."""

    minimum_donors_per_cohort: int = 20
    minimum_age_span: float = 10.0
    minimum_cohorts: int = 3
    minimum_sign_concordance: float = 0.75
    maximum_i2: float = 0.75
    maximum_fdr: float = 0.05
    minimum_absolute_standardized_slope_per_decade: float = 0.0

    def validate(self) -> None:
        if self.minimum_donors_per_cohort < 2 or self.minimum_cohorts < 2:
            raise ValueError("transferability donor/cohort minima must be at least two")
        if self.minimum_age_span <= 0:
            raise ValueError("minimum_age_span must be positive")
        for name, value in (
            ("minimum_sign_concordance", self.minimum_sign_concordance),
            ("maximum_i2", self.maximum_i2),
            ("maximum_fdr", self.maximum_fdr),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.minimum_absolute_standardized_slope_per_decade < 0:
            raise ValueError("minimum standardized slope cannot be negative")


@dataclass(frozen=True)
class TransferableGPResult:
    """Cohort-specific effects and one auditable selection row per GP stratum."""

    effects: pd.DataFrame
    selection: pd.DataFrame
    training_datasets: tuple[str, ...]
    excluded_datasets: tuple[str, ...]


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().sort_values()
    if finite.empty:
        return result
    ranks = np.arange(1, len(finite) + 1, dtype=float)
    adjusted = np.minimum.accumulate((finite.to_numpy() * len(finite) / ranks)[::-1])[
        ::-1
    ]
    result.loc[finite.index] = np.minimum(adjusted, 1.0)
    return result


def _meta_analysis(effect: pd.DataFrame) -> dict[str, float | int]:
    valid = effect.loc[
        effect["eligible"]
        & np.isfinite(effect["age_slope_per_year"])
        & np.isfinite(effect["age_slope_se"])
        & effect["age_slope_se"].gt(0)
    ].copy()
    if valid.empty:
        return {
            "n_cohorts_eligible": 0,
            "meta_age_slope_per_year": float("nan"),
            "meta_age_slope_se": float("nan"),
            "meta_z_score": float("nan"),
            "meta_p_value": float("nan"),
            "heterogeneity_q": float("nan"),
            "heterogeneity_i2": float("nan"),
            "sign_concordance": float("nan"),
            "mean_absolute_standardized_slope_per_decade": float("nan"),
        }
    slopes = valid["age_slope_per_year"].to_numpy(dtype=float)
    variances = np.square(valid["age_slope_se"].to_numpy(dtype=float))
    weights = 1.0 / variances
    meta_slope = float(np.average(slopes, weights=weights))
    meta_se = float(np.sqrt(1.0 / weights.sum()))
    z_score = meta_slope / meta_se
    q_value = float(np.sum(weights * np.square(slopes - meta_slope)))
    degrees = len(slopes) - 1
    i2 = max(0.0, (q_value - degrees) / q_value) if q_value > 0 else 0.0
    reference_sign = np.sign(meta_slope)
    concordance = float(np.mean(np.sign(slopes) == reference_sign))
    standardized = valid["standardized_age_slope_per_decade"].to_numpy(dtype=float)
    finite_standardized = standardized[np.isfinite(standardized)]
    return {
        "n_cohorts_eligible": len(valid),
        "meta_age_slope_per_year": meta_slope,
        "meta_age_slope_se": meta_se,
        "meta_z_score": z_score,
        "meta_p_value": float(2.0 * norm.sf(abs(z_score))),
        "heterogeneity_q": q_value,
        "heterogeneity_i2": float(i2),
        "sign_concordance": concordance,
        "mean_absolute_standardized_slope_per_decade": (
            float(np.mean(np.abs(finite_standardized)))
            if len(finite_standardized)
            else float("nan")
        ),
    }


def select_transferable_gene_programs(
    scores: pd.DataFrame,
    *,
    program_column: str = "gp_id",
    score_column: str = "gp_score",
    dataset_column: str = "dataset",
    donor_column: str = "biological_unit_id",
    age_column: str = "age",
    sex_column: str = "sex",
    strata_columns: Sequence[str] = ("lineage", "fine_type"),
    training_datasets: Iterable[str] | None = None,
    excluded_datasets: Iterable[str] = (),
    config: TransferabilityConfig = TransferabilityConfig(),
) -> TransferableGPResult:
    """Select reproducible GP-age effects without consulting excluded cohorts.

    Effects are fitted independently within each cohort and stratum.  Repeated
    observations share one donor's total weight, and standard errors are clustered
    by donor.  Fixed-effect meta-analysis is used only as a selection summary;
    cohort slopes remain in the returned audit table.
    """

    config.validate()
    strata = tuple(map(str, strata_columns))
    if not strata:
        raise ValueError("at least one transferability stratum column is required")
    required = {
        program_column,
        score_column,
        dataset_column,
        donor_column,
        age_column,
        sex_column,
        *strata,
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"GP transferability scores lack columns: {missing}")
    frame = scores.copy()
    frame[dataset_column] = frame[dataset_column].astype(str)
    excluded = tuple(sorted(set(map(str, excluded_datasets))))
    if training_datasets is None:
        selected = tuple(sorted(set(frame[dataset_column]) - set(excluded)))
    else:
        selected = tuple(sorted(set(map(str, training_datasets))))
    if not selected:
        raise ValueError("no training datasets remain for GP transferability")
    if set(selected) & set(excluded):
        raise ValueError("training and excluded datasets overlap")
    unknown = set(selected) - set(frame[dataset_column])
    if unknown:
        raise ValueError(f"GP scores lack training datasets: {sorted(unknown)}")
    frame = frame.loc[frame[dataset_column].isin(selected)].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")

    effect_frames: list[pd.DataFrame] = []
    grouping: str | list[str] = list(strata) if len(strata) > 1 else strata[0]
    for stratum_values, stratum in frame.groupby(grouping, observed=True, sort=True):
        values = (
            stratum_values if isinstance(stratum_values, tuple) else (stratum_values,)
        )
        stratum_fields = dict(zip(strata, map(str, values), strict=True))
        for program_id, program in stratum.groupby(
            program_column, observed=True, sort=True
        ):
            program = program.loc[np.isfinite(program[score_column])]
            if program.empty:
                continue
            effects = cohort_feature_age_effects(
                program[[score_column]].to_numpy(dtype=float),
                program[age_column],
                program[sex_column],
                program[donor_column],
                program[dataset_column],
                feature_ids=[str(program_id)],
                minimum_donors=config.minimum_donors_per_cohort,
                minimum_age_span=config.minimum_age_span,
            )
            effects = effects.rename(columns={"feature_id": program_column})
            effects = effects.drop(columns="feature_index")
            for name, value in stratum_fields.items():
                effects[name] = value
            effect_frames.append(effects)
    if not effect_frames:
        raise ValueError("no finite GP scores were available for transferability")
    effects = pd.concat(effect_frames, ignore_index=True)

    records: list[dict[str, object]] = []
    selection_groups = [*strata, program_column]
    for key, group in effects.groupby(selection_groups, observed=True, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(selection_groups, key_values, strict=True))
        record.update(_meta_analysis(group))
        records.append(record)
    selection = pd.DataFrame.from_records(records)
    selection["meta_fdr"] = _benjamini_hochberg(selection["meta_p_value"])
    reasons: list[str] = []
    retained: list[bool] = []
    for row in selection.itertuples(index=False):
        failed: list[str] = []
        if row.n_cohorts_eligible < config.minimum_cohorts:
            failed.append("too_few_cohorts")
        if not np.isfinite(row.sign_concordance) or (
            row.sign_concordance < config.minimum_sign_concordance
        ):
            failed.append("inconsistent_direction")
        if not np.isfinite(row.heterogeneity_i2) or (
            row.heterogeneity_i2 > config.maximum_i2
        ):
            failed.append("excessive_heterogeneity")
        if not np.isfinite(row.meta_fdr) or row.meta_fdr > config.maximum_fdr:
            failed.append("age_association_fdr")
        if (
            not np.isfinite(row.mean_absolute_standardized_slope_per_decade)
            or row.mean_absolute_standardized_slope_per_decade
            < config.minimum_absolute_standardized_slope_per_decade
        ):
            failed.append("effect_too_small")
        retained.append(not failed)
        reasons.append("retained" if not failed else "|".join(failed))
    selection["retained"] = retained
    selection["selection_reason"] = reasons
    selection["minimum_cohorts"] = config.minimum_cohorts
    selection["minimum_sign_concordance"] = config.minimum_sign_concordance
    selection["maximum_i2"] = config.maximum_i2
    selection["maximum_fdr"] = config.maximum_fdr
    return TransferableGPResult(
        effects=effects.sort_values(selection_groups + ["dataset"]).reset_index(
            drop=True
        ),
        selection=selection.sort_values(selection_groups).reset_index(drop=True),
        training_datasets=selected,
        excluded_datasets=excluded,
    )
