"""Donor-level metrics that report every held-out LODO fold separately."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def age_prediction_metrics(
    chronological_age: Sequence[float], predicted_age: Sequence[float]
) -> dict[str, float | int]:
    """Compute calibration and error metrics on biological observations."""

    observed = np.asarray(chronological_age, dtype=float)
    predicted = np.asarray(predicted_age, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) == 0:
        return {
            "n_observations": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }
    residual = predicted - observed
    if len(observed) >= 2 and np.ptp(observed) > 0:
        slope, intercept = np.polyfit(observed, predicted, 1)
        if np.ptp(predicted) > 0:
            pearson = float(pearsonr(observed, predicted)[0])
            spearman = float(spearmanr(observed, predicted)[0])
        else:
            pearson = spearman = np.nan
    else:
        slope = intercept = pearson = spearman = np.nan
    return {
        "n_observations": int(len(observed)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def evaluate_lodo(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("fold_id", "lineage", "gp_id"),
    minimum_subgroup_size: int = 5,
) -> pd.DataFrame:
    """Return per-fold full, age-overlap and powered sex-stratified metrics."""

    required = [*group_columns, "observation_id", "age", "predicted_gp_age"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"LODO predictions are missing columns: {missing}")
    if predictions.duplicated([*group_columns, "observation_id"]).any():
        raise ValueError("LODO metrics require one row per biological observation")
    records: list[dict[str, object]] = []
    grouped = predictions.groupby(list(group_columns), observed=True, sort=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_columns, keys))

        def add(subset: pd.DataFrame, label: str) -> None:
            records.append(
                {
                    **base,
                    "evaluation_subset": label,
                    **age_prediction_metrics(subset["age"], subset["predicted_gp_age"]),
                }
            )

        add(group, "full")
        if {"training_age_min", "training_age_max"}.issubset(group):
            overlap = group[
                group["age"].between(
                    group["training_age_min"], group["training_age_max"]
                )
            ]
            add(overlap, "age_overlap")
        if "sex" in group:
            for sex, subset in group.groupby("sex", observed=True):
                if len(subset) >= minimum_subgroup_size:
                    add(subset, f"sex={sex}")
    return pd.DataFrame.from_records(records)


def dataset_predictability(
    features: np.ndarray,
    datasets: Sequence[str],
    biological_unit_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float | int]:
    """Estimate donor-grouped predictability of dataset from biological outputs."""

    x = np.asarray(features, dtype=float)
    labels = np.asarray(datasets, dtype=str)
    groups = np.asarray(biological_unit_ids, dtype=str)
    if x.ndim != 2 or any(len(value) != len(x) for value in (labels, groups)):
        raise ValueError("features, dataset labels and donors must align")
    donor_dataset = pd.DataFrame({"group": groups, "dataset": labels})
    if (donor_dataset.groupby("group")["dataset"].nunique() > 1).any():
        raise ValueError("one biological unit cannot span multiple datasets")
    donors_per_dataset = donor_dataset.drop_duplicates().groupby("dataset").size()
    splits = min(n_splits, int(donors_per_dataset.min()))
    if splits < 2 or len(np.unique(labels)) < 2:
        raise ValueError("dataset predictability requires two folds and two datasets")
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    predicted = np.empty_like(labels)
    for train, test in splitter.split(x, labels, groups):
        if len(np.unique(labels[train])) < 2:
            raise ValueError("a grouped training fold contains only one dataset")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5_000, random_state=seed),
        )
        model.fit(x[train], labels[train])
        predicted[test] = model.predict(x[test])
    return {
        "n_observations": int(len(labels)),
        "n_donors": int(len(np.unique(groups))),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
