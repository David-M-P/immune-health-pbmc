"""Deterministic empirical and Gaussian distributional distances."""

from __future__ import annotations

import numpy as np


def _embedding(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty cells-by-dimensions matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Euclidean distance between empirical centroids."""

    left = _embedding(first, "first")
    right = _embedding(second, "second")
    if left.shape[1] != right.shape[1]:
        raise ValueError("embedding dimensions differ")
    return float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))


def _psd_sqrt(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    if float(values.min()) < -1e-7:
        raise ValueError("covariance is not positive semidefinite")
    return (vectors * np.sqrt(np.maximum(values, floor))) @ vectors.T


def gaussian_wasserstein_distance(
    mean_first: np.ndarray,
    covariance_first: np.ndarray,
    mean_second: np.ndarray,
    covariance_second: np.ndarray,
) -> float:
    """Gaussian 2-Wasserstein/Bures distance between two moment summaries."""

    first_mean = np.asarray(mean_first, dtype=float)
    second_mean = np.asarray(mean_second, dtype=float)
    first_cov = np.asarray(covariance_first, dtype=float)
    second_cov = np.asarray(covariance_second, dtype=float)
    dimension = len(first_mean)
    expected = (dimension, dimension)
    if first_mean.shape != second_mean.shape or first_cov.shape != expected:
        raise ValueError("first Gaussian dimensions are inconsistent")
    if second_cov.shape != expected:
        raise ValueError("second Gaussian dimensions are inconsistent")
    if not all(
        np.isfinite(value).all()
        for value in (first_mean, second_mean, first_cov, second_cov)
    ):
        raise ValueError("Gaussian moments must be finite")
    second_sqrt = _psd_sqrt(second_cov)
    middle = second_sqrt @ first_cov @ second_sqrt
    middle_sqrt = _psd_sqrt(middle)
    squared = float(
        np.dot(first_mean - second_mean, first_mean - second_mean)
        + np.trace(first_cov + second_cov - 2.0 * middle_sqrt)
    )
    return float(np.sqrt(max(squared, 0.0)))


def sliced_wasserstein_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_projections: int = 128,
    seed: int = 0,
    p: float = 2.0,
    projections: np.ndarray | None = None,
) -> float:
    """Sliced empirical Wasserstein-p distance over fixed random slices.

    Projection vectors come from a local seeded generator and are therefore
    deterministic without mutating NumPy's global random state.  Supplying the
    same explicit projections allows direct comparability across donors.
    """

    left = _embedding(first, "first")
    right = _embedding(second, "second")
    if left.shape[1] != right.shape[1]:
        raise ValueError("embedding dimensions differ")
    if not np.isfinite(p) or p < 1:
        raise ValueError("Wasserstein order p must be at least one")
    dimension = left.shape[1]
    if projections is None:
        if n_projections < 1:
            raise ValueError("n_projections must be positive")
        directions = np.random.default_rng(seed).normal(size=(n_projections, dimension))
    else:
        directions = np.asarray(projections, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != dimension:
            raise ValueError("projection dimensions differ from embeddings")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0) or not np.isfinite(directions).all():
        raise ValueError("projections must be finite nonzero vectors")
    directions = directions / norms[:, None]
    left_projected = left @ directions.T
    right_projected = right @ directions.T
    powers = [
        _wasserstein_power(left_projected[:, index], right_projected[:, index], p)
        for index in range(directions.shape[0])
    ]
    return float(np.mean(powers) ** (1.0 / p))


def _wasserstein_power(first: np.ndarray, second: np.ndarray, p: float) -> float:
    """Exact integral of |empirical quantile difference|**p in one dimension."""

    left = np.sort(first)
    right = np.sort(second)
    boundaries = np.unique(
        np.concatenate(
            [
                np.arange(len(left) + 1, dtype=float) / len(left),
                np.arange(len(right) + 1, dtype=float) / len(right),
            ]
        )
    )
    widths = np.diff(boundaries)
    midpoints = boundaries[:-1] + widths / 2.0
    left_index = np.minimum((midpoints * len(left)).astype(int), len(left) - 1)
    right_index = np.minimum((midpoints * len(right)).astype(int), len(right) - 1)
    return float(np.sum(widths * np.abs(left[left_index] - right[right_index]) ** p))
