"""Stable fine-type location and dispersion estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class DistributionEstimate:
    """Numerical summary of one donor-observation fine-type distribution."""

    n_cells: int
    location: np.ndarray | None
    covariance: np.ndarray | None
    covariance_trace: float
    covariance_logdet: float
    median_distance: float
    distance_q75: float
    distance_q90: float
    state_available: bool


def _matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("cell embeddings must have shape (cells, dimensions)")
    if not np.isfinite(matrix).all():
        raise ValueError("cell embeddings contain non-finite values")
    return matrix


def _positive_semidefinite(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(values, floor)
    return (vectors * clipped) @ vectors.T


def shrinkage_covariance(
    values: np.ndarray,
    *,
    location: np.ndarray | None = None,
    regularization: float = 1e-8,
) -> np.ndarray:
    """Estimate a positive-definite Ledoit-Wolf shrinkage covariance.

    When a robust location is supplied, the covariance is fitted to values
    centred on that location with ``assume_centered=True``.  This avoids an
    unrestricted sample covariance in small, high-dimensional fine types.
    """

    matrix = _matrix(values)
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    centre = (
        matrix.mean(axis=0) if location is None else np.asarray(location, dtype=float)
    )
    if centre.shape != (matrix.shape[1],):
        raise ValueError("location dimension differs from cell embeddings")
    centered = matrix - centre
    if matrix.shape[0] == 1:
        return np.eye(matrix.shape[1]) * regularization
    covariance = LedoitWolf(assume_centered=True).fit(centered).covariance_
    return _positive_semidefinite(covariance, regularization)


def summarize_distribution(
    values: np.ndarray,
    *,
    min_cells: int = 5,
    robust_location: bool = True,
    regularization: float = 1e-8,
) -> DistributionEstimate:
    """Summarize a distribution, retaining rare state as explicitly missing."""

    matrix = _matrix(values)
    n_cells = matrix.shape[0]
    if min_cells < 2:
        raise ValueError("min_cells must be at least two for state estimation")
    if n_cells < min_cells:
        missing = float("nan")
        return DistributionEstimate(
            n_cells=n_cells,
            location=None,
            covariance=None,
            covariance_trace=missing,
            covariance_logdet=missing,
            median_distance=missing,
            distance_q75=missing,
            distance_q90=missing,
            state_available=False,
        )

    location = np.median(matrix, axis=0) if robust_location else matrix.mean(axis=0)
    covariance = shrinkage_covariance(
        matrix, location=location, regularization=regularization
    )
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("regularized covariance is not positive definite")
    distances = np.linalg.norm(matrix - location, axis=1)
    return DistributionEstimate(
        n_cells=n_cells,
        location=location,
        covariance=covariance,
        covariance_trace=float(np.trace(covariance)),
        covariance_logdet=float(logdet),
        median_distance=float(np.median(distances)),
        distance_q75=float(np.quantile(distances, 0.75)),
        distance_q90=float(np.quantile(distances, 0.90)),
        state_available=True,
    )
