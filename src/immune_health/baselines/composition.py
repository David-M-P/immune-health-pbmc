"""Fine-type composition summaries and an age/sex conditional comparator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .pseudobulk import ensure_donor_observation_ids

COMPOSITION_KEYS = (
    "dataset",
    "donor_id",
    "biological_unit_id",
    "sample_id",
    "source_observation_id",
    "observation_id",
    "lineage",
)


def build_composition_table(
    obs: pd.DataFrame,
    *,
    fine_type_col: str = "fine_type",
    fine_type_universe: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return long donor-observation compositions, including observed zeros.

    Rare types are never discarded.  A requested universe is expanded across
    every observation, making absence explicit in composition while leaving
    state estimation to the aggregation module.
    """

    frame = ensure_donor_observation_ids(obs)
    required = [*COMPOSITION_KEYS, fine_type_col]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing composition columns: {missing}")
    if frame[fine_type_col].isna().any():
        raise ValueError("fine_type contains missing values")

    metadata_columns = [column for column in ("age", "sex") if column in frame]
    key_columns = [*COMPOSITION_KEYS, *metadata_columns]
    counts = (
        frame.groupby([*key_columns, fine_type_col], observed=True, sort=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    if fine_type_universe is None:
        fine_types = tuple(sorted(frame[fine_type_col].astype(str).unique()))
    else:
        fine_types = tuple(dict.fromkeys(str(value) for value in fine_type_universe))
        observed = set(frame[fine_type_col].astype(str))
        absent = observed.difference(fine_types)
        if absent:
            raise ValueError(
                f"fine_type_universe omits observed labels: {sorted(absent)}"
            )
    observation_keys = [*COMPOSITION_KEYS]
    for column in metadata_columns:
        variation = frame.groupby(observation_keys, observed=True, sort=False)[
            column
        ].nunique(dropna=False)
        if (variation > 1).any():
            raise ValueError(f"{column!r} varies within one biological observation")
    observations = frame[key_columns].drop_duplicates(ignore_index=True)
    grid = observations.merge(
        pd.DataFrame({fine_type_col: fine_types}), how="cross", validate="many_to_many"
    )
    table = grid.merge(
        counts,
        on=[*key_columns, fine_type_col],
        how="left",
        validate="one_to_one",
    )
    table["n_cells"] = table["n_cells"].fillna(0).astype(np.int64)
    totals = table.groupby(key_columns, observed=True)["n_cells"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("an observation has no cells")
    table["fine_type_fraction"] = table["n_cells"] / totals
    return table


def composition_matrix(
    table: pd.DataFrame,
    *,
    fine_type_col: str = "fine_type",
    value_col: str = "fine_type_fraction",
    fine_types: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Pivot a long composition table into metadata, matrix and column labels."""

    key_columns = [
        column for column in (*COMPOSITION_KEYS, "age", "sex") if column in table
    ]
    if fine_types is None:
        labels = tuple(sorted(table[fine_type_col].astype(str).unique()))
    else:
        labels = tuple(str(value) for value in fine_types)
    wide = table.pivot(index=key_columns, columns=fine_type_col, values=value_col)
    wide = wide.reindex(columns=labels, fill_value=0.0).fillna(0.0)
    metadata = wide.index.to_frame(index=False)
    matrix = wide.to_numpy(dtype=float)
    return metadata, matrix, labels


def _closed(composition: np.ndarray, pseudocount: float) -> np.ndarray:
    values = np.asarray(composition, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("compositions must have shape (n, at least 2 parts)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("compositions must be finite and nonnegative")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    adjusted = values + pseudocount
    totals = adjusted.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("composition rows must have positive total")
    return adjusted / totals


def centered_log_ratio(
    composition: np.ndarray, *, pseudocount: float = 1e-6
) -> np.ndarray:
    """Apply closure and the centred log-ratio transform."""

    log_values = np.log(_closed(composition, pseudocount))
    return log_values - log_values.mean(axis=1, keepdims=True)


def aitchison_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    pseudocount: float = 1e-6,
) -> np.ndarray | float:
    """Euclidean distance between centred log-ratio compositions."""

    left = centered_log_ratio(first, pseudocount=pseudocount)
    right = centered_log_ratio(second, pseudocount=pseudocount)
    if left.shape != right.shape and left.shape[0] != 1 and right.shape[0] != 1:
        raise ValueError("composition arrays cannot be broadcast by row")
    distances = np.linalg.norm(left - right, axis=1)
    return float(distances[0]) if distances.size == 1 else distances


@dataclass
class AgeSexCompositionModel:
    """Transparent ridge model for expected CLR composition.

    Age is represented by linear and quadratic terms and sex by fixed effects.
    Repeated observations receive inverse donor-frequency weights so a donor,
    rather than each visit, is the biological unit during fitting.
    """

    ridge: float = 1e-6
    pseudocount: float = 1e-6

    def _design(self, ages: np.ndarray, sexes: np.ndarray) -> np.ndarray:
        if not hasattr(self, "age_mean_"):
            raise RuntimeError("composition model has not been fitted")
        scaled = (ages - self.age_mean_) / self.age_scale_
        columns = [np.ones_like(scaled), scaled, scaled**2]
        for level in self.sex_levels_[1:]:
            columns.append((sexes == level).astype(float))
        known = np.isin(sexes, self.sex_levels_)
        if not known.all():
            raise ValueError(f"unknown sex labels: {sorted(set(sexes[~known]))}")
        return np.column_stack(columns)

    def fit(
        self,
        compositions: np.ndarray,
        ages: Sequence[float],
        sexes: Sequence[str],
        biological_unit_ids: Sequence[str],
        *,
        fine_types: Sequence[str] | None = None,
    ) -> "AgeSexCompositionModel":
        values = np.asarray(compositions, dtype=float)
        age = np.asarray(ages, dtype=float)
        sex = np.asarray(sexes, dtype=str)
        groups = np.asarray(biological_unit_ids, dtype=str)
        if values.ndim != 2 or len(values) != len(age):
            raise ValueError("composition and covariate row counts differ")
        if len(sex) != len(age) or len(groups) != len(age):
            raise ValueError("composition covariate row counts differ")
        if not np.isfinite(age).all():
            raise ValueError("ages must be finite")
        if np.any(pd.isna(sexes)) or np.any(pd.isna(biological_unit_ids)):
            raise ValueError("sex and biological_unit_id cannot be missing")
        self.age_mean_ = float(age.mean())
        self.age_scale_ = float(age.std()) or 1.0
        self.sex_levels_ = tuple(sorted(np.unique(sex)))
        self.fine_types_ = (
            tuple(str(value) for value in fine_types)
            if fine_types is not None
            else tuple(f"part_{index}" for index in range(values.shape[1]))
        )
        if len(self.fine_types_) != values.shape[1]:
            raise ValueError("fine_types length does not match composition width")

        design = self._design(age, sex)
        target = centered_log_ratio(values, pseudocount=self.pseudocount)
        _, inverse, donor_counts = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        weights = 1.0 / donor_counts[inverse]
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_target = target * np.sqrt(weights[:, None])
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_target,
        )
        self.training_biological_units_ = frozenset(groups)
        return self

    def predict(self, ages: Sequence[float], sexes: Sequence[str]) -> np.ndarray:
        age = np.asarray(ages, dtype=float)
        sex = np.asarray(sexes, dtype=str)
        if len(age) != len(sex):
            raise ValueError("age and sex row counts differ")
        clr = self._design(age, sex) @ self.coefficients_
        clr -= clr.max(axis=1, keepdims=True)
        proportions = np.exp(clr)
        return proportions / proportions.sum(axis=1, keepdims=True)

    def distance(
        self,
        compositions: np.ndarray,
        ages: Sequence[float],
        sexes: Sequence[str],
    ) -> np.ndarray:
        expected = self.predict(ages, sexes)
        return np.atleast_1d(
            aitchison_distance(compositions, expected, pseudocount=self.pseudocount)
        )
