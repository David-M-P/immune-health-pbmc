"""Transparent spline trajectories in GP embedding space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from immune_health.aggregation.statistics import shrinkage_covariance

REFERENCE_WEIGHTING_SCHEMES = frozenset({"donor_pooled", "cohort_balanced"})


def _as_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite rows-by-dimensions matrix")
    return matrix


def _weighted_quantiles(
    values: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    probability = weights[order] / weights.sum()
    positions = np.cumsum(probability) - probability / 2.0
    return np.interp(probabilities, positions, ordered)


def reference_row_weights(
    biological_unit_ids: Sequence[str],
    datasets: Sequence[str] | None = None,
    *,
    scheme: str = "donor_pooled",
) -> np.ndarray:
    """Return row weights with one total unit per donor by default.

    ``donor_pooled`` gives every donor the same total weight.  In
    ``cohort_balanced`` mode donors remain equal within a cohort, while every
    cohort receives the same total weight.  Both schemes are scaled to sum to
    the number of donors so changing schemes does not silently alter the
    effective ridge penalty.
    """

    groups = np.asarray(biological_unit_ids, dtype=str)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("biological_unit_ids must be a nonempty vector")
    if scheme not in REFERENCE_WEIGHTING_SCHEMES:
        raise ValueError(
            f"unknown reference weighting scheme {scheme!r}; choose "
            f"{sorted(REFERENCE_WEIGHTING_SCHEMES)}"
        )
    unique_groups, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    if scheme == "donor_pooled":
        return weights
    if datasets is None:
        raise ValueError("cohort_balanced weighting requires dataset labels")
    dataset = np.asarray(datasets, dtype=str)
    if dataset.shape != groups.shape:
        raise ValueError("datasets and biological_unit_ids must align")
    donor_dataset = pd.DataFrame(
        {"donor": groups, "dataset": dataset}
    ).drop_duplicates()
    if donor_dataset["donor"].duplicated().any():
        raise ValueError("one biological unit maps to multiple datasets")
    donor_counts = donor_dataset.groupby("dataset", observed=True).size()
    if len(donor_counts) < 2:
        raise ValueError("cohort_balanced weighting requires at least two datasets")
    total_donors = len(unique_groups)
    cohort_target = total_donors / len(donor_counts)
    scale = dataset.copy().astype(object)
    for level, count in donor_counts.items():
        scale[dataset == str(level)] = cohort_target / float(count)
    return weights * scale.astype(float)


def fit_age_direction(features: np.ndarray, ages: Sequence[float]) -> np.ndarray:
    """Fit and normalize a multivariate training-only linear age direction."""

    matrix = _as_matrix(features)
    age = np.asarray(ages, dtype=float)
    if len(matrix) != len(age) or not np.isfinite(age).all():
        raise ValueError("features and ages must align and be finite")
    centered_age = age - age.mean()
    denominator = float(np.dot(centered_age, centered_age))
    if denominator == 0:
        raise ValueError("age direction requires variation in training ages")
    direction = centered_age @ (matrix - matrix.mean(axis=0)) / denominator
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise ValueError("training features have no estimable age direction")
    return direction / norm


class HealthyTrajectory:
    """Ridge-regularized regression spline with sex and dataset fixed effects.

    The age basis contains cubic polynomial terms and truncated-cubic knots.
    Dataset effects are centred over the training datasets.  Frozen query
    scoring sets those effects to zero, representing the average training
    cohort, and never estimates an offset from the query dataset.
    """

    def __init__(
        self,
        *,
        n_spline_knots: int = 3,
        ridge: float = 1e-3,
        age_grid_size: int = 101,
        covariance_regularization: float = 1e-8,
        weighting_scheme: str = "donor_pooled",
    ) -> None:
        if n_spline_knots < 0 or age_grid_size < 2:
            raise ValueError("invalid spline knots or age grid size")
        self.n_spline_knots = n_spline_knots
        self.ridge = ridge
        self.age_grid_size = age_grid_size
        self.covariance_regularization = covariance_regularization
        if weighting_scheme not in REFERENCE_WEIGHTING_SCHEMES:
            raise ValueError(
                f"unknown reference weighting scheme {weighting_scheme!r}; choose "
                f"{sorted(REFERENCE_WEIGHTING_SCHEMES)}"
            )
        self.weighting_scheme = weighting_scheme

    def _age_basis(self, ages: np.ndarray) -> list[np.ndarray]:
        scaled = (ages - self.age_mean_) / self.age_scale_
        columns = [scaled, scaled**2, scaled**3]
        columns.extend(np.maximum(scaled - knot, 0.0) ** 3 for knot in self.knots_)
        return columns

    def _design(
        self,
        ages: np.ndarray,
        sexes: np.ndarray,
        datasets: np.ndarray | None,
    ) -> np.ndarray:
        if not hasattr(self, "coefficients_") and not hasattr(self, "age_mean_"):
            raise RuntimeError("healthy trajectory has not been fitted")
        known_sex = np.isin(sexes, self.sex_levels_)
        if not known_sex.all():
            raise ValueError(f"unknown sex labels: {sorted(set(sexes[~known_sex]))}")
        columns = [np.ones(len(ages)), *self._age_basis(ages)]
        columns.extend((sexes == level).astype(float) for level in self.sex_levels_[1:])

        if self.dataset_levels_:
            if datasets is None:
                columns.extend(np.zeros(len(ages)) for _ in self.dataset_levels_)
            else:
                known_dataset = np.isin(datasets, self.dataset_levels_)
                if not known_dataset.all():
                    unknown = sorted(set(datasets[~known_dataset]))
                    raise ValueError(
                        f"unseen query datasets {unknown}; pass dataset=None for "
                        "strict zero-shot scoring"
                    )
                columns.extend(
                    (datasets == level).astype(float) - self.dataset_proportions_[level]
                    for level in self.dataset_levels_
                )
        return np.column_stack(columns)

    def fit(
        self,
        features: np.ndarray,
        ages: Sequence[float],
        sexes: Sequence[str],
        biological_unit_ids: Sequence[str],
        *,
        datasets: Sequence[str] | None = None,
    ) -> "HealthyTrajectory":
        matrix = _as_matrix(features)
        age = np.asarray(ages, dtype=float)
        sex = np.asarray(sexes, dtype=str)
        groups = np.asarray(biological_unit_ids, dtype=str)
        if any(len(value) != len(matrix) for value in (age, sex, groups)):
            raise ValueError("trajectory features and metadata must align by row")
        if not np.isfinite(age).all() or len(np.unique(groups)) < 2:
            raise ValueError(
                "finite ages and at least two biological units are required"
            )
        dataset = None if datasets is None else np.asarray(datasets, dtype=str)
        if dataset is not None and len(dataset) != len(matrix):
            raise ValueError("dataset labels and features must align by row")

        weights = reference_row_weights(groups, dataset, scheme=self.weighting_scheme)
        self.age_mean_ = float(np.average(age, weights=weights))
        variance = float(np.average((age - self.age_mean_) ** 2, weights=weights))
        self.age_scale_ = np.sqrt(variance) or 1.0
        self.age_range_ = (float(age.min()), float(age.max()))
        if self.n_spline_knots:
            probabilities = np.linspace(0.0, 1.0, self.n_spline_knots + 2)[1:-1]
            scaled = (age - self.age_mean_) / self.age_scale_
            self.knots_ = tuple(
                float(value)
                for value in np.unique(
                    _weighted_quantiles(scaled, probabilities, weights)
                )
            )
        else:
            self.knots_ = ()
        self.sex_levels_ = tuple(sorted(np.unique(sex)))
        if dataset is None:
            self.dataset_levels_ = ()
            self.dataset_proportions_ = {}
        else:
            self.dataset_levels_ = tuple(sorted(np.unique(dataset)))
            total_weight = weights.sum()
            self.dataset_proportions_ = {
                level: float(weights[dataset == level].sum() / total_weight)
                for level in self.dataset_levels_
            }

        design = self._design(age, sex, dataset)
        root_weight = np.sqrt(weights)[:, None]
        weighted_design = design * root_weight
        weighted_features = matrix * root_weight
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_features,
        )
        fitted = design @ self.coefficients_
        residuals = matrix - fitted
        donor_residuals = np.vstack(
            [residuals[groups == donor].mean(axis=0) for donor in np.unique(groups)]
        )
        self.residual_covariance_ = shrinkage_covariance(
            donor_residuals,
            location=np.zeros(matrix.shape[1]),
            regularization=self.covariance_regularization,
        )
        self.training_biological_units_ = frozenset(groups)
        self.training_biological_unit_rows_ = groups.copy()
        self.training_ages_ = age.copy()
        self.training_sexes_ = sex.copy()
        self.training_datasets_ = None if dataset is None else dataset.copy()
        self.training_row_weights_ = weights.copy()
        self.training_weight_summary_ = {
            str(level): float(weights[dataset == level].sum())
            for level in self.dataset_levels_
        }
        self.n_features_in_ = matrix.shape[1]
        direction_grid = np.linspace(*self.age_range_, max(self.age_grid_size, 10))
        try:
            self.age_direction_ = fit_age_direction(
                self.predict(
                    direction_grid,
                    np.repeat(self.sex_levels_[0], len(direction_grid)),
                ),
                direction_grid,
            )
        except ValueError:
            # A flat fitted trajectory has no identifiable ageing direction.
            self.age_direction_ = np.zeros(self.n_features_in_, dtype=float)
        return self

    def predict(
        self,
        ages: Sequence[float],
        sexes: Sequence[str],
        *,
        datasets: Sequence[str] | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("healthy trajectory has not been fitted")
        age = np.asarray(ages, dtype=float)
        sex = np.asarray(sexes, dtype=str)
        if len(age) != len(sex) or not np.isfinite(age).all():
            raise ValueError("ages and sexes must align and ages must be finite")
        dataset = None if datasets is None else np.asarray(datasets, dtype=str)
        if dataset is not None and len(dataset) != len(age):
            raise ValueError("datasets and ages must align")
        return self._design(age, sex, dataset) @ self.coefficients_

    def score(
        self,
        location: np.ndarray,
        chronological_age: float,
        sex: str,
        *,
        covariance: np.ndarray | None = None,
        age_grid: Sequence[float] | None = None,
        dataset: str | None = None,
    ) -> dict[str, float]:
        """Return location-only age position and trajectory distances.

        Donor within-cell covariance is a different object from covariance of
        donor residual locations. Distributional scoring belongs to the paired
        ``AgeKernelReference`` and is refused here to prevent conflation.
        """

        point = np.asarray(location, dtype=float)
        if point.shape != (self.n_features_in_,) or not np.isfinite(point).all():
            raise ValueError("query location differs from healthy embedding dimension")
        if covariance is not None:
            raise ValueError(
                "HealthyTrajectory is location-only; score donor covariance with "
                "AgeKernelReference.score_distribution"
            )
        if dataset is not None and dataset not in self.dataset_levels_:
            raise ValueError(
                "an unseen dataset cannot receive a fitted offset; use dataset=None"
            )
        query_dataset = None if dataset is None else [dataset]
        expected = self.predict([chronological_age], [sex], datasets=query_dataset)[0]
        if age_grid is None:
            grid = np.linspace(*self.age_range_, self.age_grid_size)
        else:
            grid = np.asarray(age_grid, dtype=float)
        if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
            raise ValueError("age_grid must contain at least two finite ages")
        trajectory = self.predict(
            grid,
            np.repeat(str(sex), len(grid)),
            datasets=None if dataset is None else np.repeat(dataset, len(grid)),
        )

        age_matched = float(np.linalg.norm(point - expected))
        distances = np.linalg.norm(trajectory - point, axis=1)
        minimum = float(np.min(distances))
        tied = np.flatnonzero(np.isclose(distances, minimum, rtol=1e-10, atol=1e-12))
        predicted_age = float(np.mean(grid[tied]))
        return {
            "predicted_gp_age": predicted_age,
            "gp_age_acceleration": predicted_age - float(chronological_age),
            "age_matched_distance": age_matched,
            "off_trajectory_distance": minimum,
            "age_matched_location_distance": age_matched,
            "off_trajectory_location_distance": minimum,
        }


@dataclass(frozen=True)
class CrossFitResult:
    """Out-of-fold healthy expectations and donor-level trajectory scores."""

    expected_locations: np.ndarray
    scores: pd.DataFrame


def cross_fit_trajectory(
    features: np.ndarray,
    ages: Sequence[float],
    sexes: Sequence[str],
    biological_unit_ids: Sequence[str],
    *,
    datasets: Sequence[str] | None = None,
    n_splits: int = 5,
    model_kwargs: Mapping[str, object] | None = None,
    use_dataset_effect_for_expectation: bool = False,
) -> CrossFitResult:
    """Fit healthy trajectories out of fold with donor-grouped partitions."""

    matrix = _as_matrix(features)
    age = np.asarray(ages, dtype=float)
    sex = np.asarray(sexes, dtype=str)
    groups = np.asarray(biological_unit_ids, dtype=str)
    dataset = None if datasets is None else np.asarray(datasets, dtype=str)
    if any(len(value) != len(matrix) for value in (age, sex, groups)):
        raise ValueError("cross-fit arrays must align by row")
    if dataset is not None and len(dataset) != len(matrix):
        raise ValueError("cross-fit dataset labels must align by row")
    folds = min(n_splits, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("cross-fitting requires at least two donors")
    expected = np.full_like(matrix, np.nan, dtype=float)
    records: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=folds)
    for fold_id, (train, validation) in enumerate(splitter.split(matrix, age, groups)):
        model = HealthyTrajectory(**dict(model_kwargs or {})).fit(
            matrix[train],
            age[train],
            sex[train],
            groups[train],
            datasets=None if dataset is None else dataset[train],
        )
        validation_dataset = (
            dataset[validation]
            if use_dataset_effect_for_expectation and dataset is not None
            else None
        )
        expected[validation] = model.predict(
            age[validation], sex[validation], datasets=validation_dataset
        )
        for index in validation:
            score = model.score(matrix[index], age[index], sex[index])
            records.append({"row_index": int(index), "fold_id": fold_id, **score})
    scores = (
        pd.DataFrame.from_records(records)
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    return CrossFitResult(expected, scores)
