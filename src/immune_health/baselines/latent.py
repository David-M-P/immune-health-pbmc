"""Training-only pseudobulk projection and donor-grouped age prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def log_cpm(
    counts: sparse.spmatrix | np.ndarray, *, scale: float = 1_000_000.0
) -> np.ndarray:
    """Return dense donor-level log1p-CPM features.

    Pseudobulk matrices are compact donor summaries; conversion is explicit at
    this boundary and callers can limit it with ``TrainOnlyPCA.max_dense_values``.
    """

    if sparse.issparse(counts):
        matrix = counts.tocsr(copy=False)
        library_size = np.asarray(matrix.sum(axis=1)).ravel()
        factors = np.divide(
            scale,
            library_size,
            out=np.zeros_like(library_size, dtype=float),
            where=library_size > 0,
        )
        normalized = (sparse.diags(factors) @ matrix).toarray()
    else:
        normalized = np.asarray(counts, dtype=float).copy()
        if normalized.ndim != 2:
            raise ValueError("counts must be two dimensional")
        library_size = normalized.sum(axis=1)
        normalized *= np.divide(
            scale,
            library_size,
            out=np.zeros_like(library_size, dtype=float),
            where=library_size > 0,
        )[:, None]
    if np.any(normalized < 0) or not np.isfinite(normalized).all():
        raise ValueError("counts must be finite and nonnegative")
    return np.log1p(normalized)


class TrainOnlyPCA:
    """PCA fitted once on training pseudobulks and frozen for query projection."""

    def __init__(
        self,
        n_components: int = 10,
        *,
        random_state: int = 0,
        max_dense_values: int = 50_000_000,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.max_dense_values = max_dense_values

    def _features(self, counts: sparse.spmatrix | np.ndarray) -> np.ndarray:
        shape = counts.shape
        if len(shape) != 2 or shape[0] * shape[1] > self.max_dense_values:
            raise ValueError(
                "donor-level matrix exceeds max_dense_values; reduce vocabulary"
            )
        return log_cpm(counts)

    def fit(
        self,
        training_counts: sparse.spmatrix | np.ndarray,
        *,
        feature_ids: Sequence[str] | None = None,
        training_biological_units: Sequence[str] | None = None,
    ) -> "TrainOnlyPCA":
        features = self._features(training_counts)
        maximum = min(features.shape[0], features.shape[1])
        if not 1 <= self.n_components <= maximum:
            raise ValueError(f"n_components must be between 1 and {maximum}")
        self.model_ = PCA(
            n_components=self.n_components,
            svd_solver="full",
            random_state=self.random_state,
        ).fit(features)
        self.n_features_in_ = features.shape[1]
        self.feature_ids_ = (
            tuple(str(value) for value in feature_ids)
            if feature_ids is not None
            else None
        )
        if (
            self.feature_ids_ is not None
            and len(self.feature_ids_) != self.n_features_in_
        ):
            raise ValueError("feature_ids length does not match training counts")
        units = () if training_biological_units is None else training_biological_units
        self.training_biological_units_ = frozenset(str(value) for value in units)
        return self

    def transform(
        self,
        query_counts: sparse.spmatrix | np.ndarray,
        *,
        feature_ids: Sequence[str] | None = None,
        query_biological_units: Sequence[str] | None = None,
        allow_training_units: bool = False,
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("PCA has not been fitted")
        if query_counts.shape[1] != self.n_features_in_:
            raise ValueError("query vocabulary width differs from frozen PCA")
        if self.feature_ids_ is not None:
            supplied = () if feature_ids is None else feature_ids
            query_ids = tuple(str(value) for value in supplied)
            if query_ids != self.feature_ids_:
                raise ValueError("query feature order differs from frozen vocabulary")
        if query_biological_units is not None and not allow_training_units:
            overlap = self.training_biological_units_.intersection(
                str(value) for value in query_biological_units
            )
            if overlap:
                raise ValueError(
                    f"query contains training donors: {sorted(overlap)[:3]}"
                )
        return self.model_.transform(self._features(query_counts))

    def fit_transform(
        self,
        training_counts: sparse.spmatrix | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        self.fit(training_counts, **kwargs)
        return self.model_.transform(self._features(training_counts))


class ElasticNetAgeModel:
    """Elastic-net age regression with inner donor-grouped tuning."""

    def __init__(
        self,
        *,
        alphas: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
        l1_ratios: Sequence[float] = (0.1, 0.5, 0.9),
        n_splits: int = 5,
        random_state: int = 0,
        max_iter: int = 20_000,
    ) -> None:
        self.alphas = tuple(float(value) for value in alphas)
        self.l1_ratios = tuple(float(value) for value in l1_ratios)
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_iter = max_iter

    def _pipeline(self, alpha: float, l1_ratio: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "elastic_net",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=self.max_iter,
                        selection="cyclic",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    @staticmethod
    def _donor_weights(groups: np.ndarray) -> np.ndarray:
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        return 1.0 / counts[inverse]

    def fit(
        self,
        features: np.ndarray,
        ages: Sequence[float],
        biological_unit_ids: Sequence[str],
    ) -> "ElasticNetAgeModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(ages, dtype=float)
        groups = np.asarray(biological_unit_ids, dtype=str)
        if x.ndim != 2 or len(x) != len(y) or len(groups) != len(y):
            raise ValueError("features, ages and donor groups must align by row")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("features and ages must be finite")
        unique_groups = np.unique(groups)
        splits = min(self.n_splits, len(unique_groups))
        if splits < 2:
            raise ValueError("at least two biological units are required for tuning")

        cv = GroupKFold(n_splits=splits)
        cv_splits = tuple(cv.split(x, y, groups))
        self.cv_group_splits_ = tuple(
            (
                frozenset(groups[train]),
                frozenset(groups[validation]),
            )
            for train, validation in cv_splits
        )
        records: list[dict[str, float]] = []
        best: tuple[float, float, float] | None = None
        weights = self._donor_weights(groups)
        for alpha in self.alphas:
            for l1_ratio in self.l1_ratios:
                fold_mae: list[float] = []
                for train, validation in cv_splits:
                    model = self._pipeline(alpha, l1_ratio)
                    model.fit(
                        x[train],
                        y[train],
                        scale__sample_weight=weights[train],
                        elastic_net__sample_weight=weights[train],
                    )
                    prediction = model.predict(x[validation])
                    fold_mae.append(mean_absolute_error(y[validation], prediction))
                score = float(np.mean(fold_mae))
                records.append(
                    {"alpha": alpha, "l1_ratio": l1_ratio, "mean_grouped_mae": score}
                )
                candidate = (score, alpha, l1_ratio)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, best_alpha, best_l1 = best
        self.model_ = self._pipeline(best_alpha, best_l1)
        self.model_.fit(
            x,
            y,
            scale__sample_weight=weights,
            elastic_net__sample_weight=weights,
        )
        self.best_params_ = {"alpha": best_alpha, "l1_ratio": best_l1}
        self.cv_results_ = pd.DataFrame.from_records(records)
        self.training_biological_units_ = frozenset(groups)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(
        self,
        features: np.ndarray,
        *,
        query_biological_units: Sequence[str] | None = None,
        allow_training_units: bool = False,
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("elastic-net age model has not been fitted")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("query features differ from the frozen age model")
        if query_biological_units is not None and not allow_training_units:
            overlap = self.training_biological_units_.intersection(
                str(value) for value in query_biological_units
            )
            if overlap:
                raise ValueError(
                    f"query contains training donors: {sorted(overlap)[:3]}"
                )
        return self.model_.predict(x)
