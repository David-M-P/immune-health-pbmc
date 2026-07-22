"""Fine-type mixture moments and exact covariance decomposition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from immune_health.baselines.composition import aitchison_distance

from .distances import gaussian_wasserstein_distance


@dataclass(frozen=True)
class CovarianceDecomposition:
    """Within, between and total covariance of a fine-type mixture."""

    mean: np.ndarray
    within_covariance: np.ndarray
    between_covariance: np.ndarray
    total_covariance: np.ndarray
    within_trace: float
    between_trace: float
    total_trace: float


def _groups_and_weights(
    groups: Mapping[str, np.ndarray], weights: Mapping[str, float] | None
) -> tuple[tuple[str, ...], list[np.ndarray], np.ndarray]:
    if not groups:
        raise ValueError("at least one fine-type distribution is required")
    labels = tuple(groups)
    matrices: list[np.ndarray] = []
    dimension: int | None = None
    for label in labels:
        matrix = np.asarray(groups[label], dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid distribution for fine type {label!r}")
        dimension = matrix.shape[1] if dimension is None else dimension
        if matrix.shape[1] != dimension:
            raise ValueError("fine-type embedding dimensions differ")
        matrices.append(matrix)
    if weights is None:
        probabilities = np.asarray([len(matrix) for matrix in matrices], dtype=float)
    else:
        missing = set(labels).difference(weights)
        extra_positive = {
            label
            for label, value in weights.items()
            if label not in groups and value > 0
        }
        if missing or extra_positive:
            raise ValueError(
                f"weights and measurable fine types differ; missing={sorted(missing)}, "
                f"unmeasured_positive={sorted(extra_positive)}"
            )
        probabilities = np.asarray([weights[label] for label in labels], dtype=float)
    if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
        raise ValueError("fine-type weights must be finite and nonnegative")
    if probabilities.sum() <= 0:
        raise ValueError("fine-type weights have zero total")
    probabilities /= probabilities.sum()
    return labels, matrices, probabilities


def covariance_decomposition(
    groups: Mapping[str, np.ndarray],
    weights: Mapping[str, float] | None = None,
) -> CovarianceDecomposition:
    """Decompose population covariance into weighted within and between terms.

    Population (``ddof=0``) covariance is used so the identity is exact for the
    empirical mixture.  Shrinkage state covariances remain available separately
    for Gaussian distance estimation.
    """

    _, matrices, probabilities = _groups_and_weights(groups, weights)
    means = np.vstack([matrix.mean(axis=0) for matrix in matrices])
    mixture_mean = probabilities @ means
    dimension = means.shape[1]
    within = np.zeros((dimension, dimension), dtype=float)
    between = np.zeros_like(within)
    for probability, matrix, mean in zip(probabilities, matrices, means):
        centered = matrix - mean
        covariance = centered.T @ centered / len(matrix)
        within += probability * covariance
        difference = mean - mixture_mean
        between += probability * np.outer(difference, difference)
    total = within + between
    return CovarianceDecomposition(
        mean=mixture_mean,
        within_covariance=within,
        between_covariance=between,
        total_covariance=total,
        within_trace=float(np.trace(within)),
        between_trace=float(np.trace(between)),
        total_trace=float(np.trace(total)),
    )


def mixture_moments(
    groups: Mapping[str, np.ndarray], weights: Mapping[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and total covariance for a fine-type mixture."""

    result = covariance_decomposition(groups, weights)
    return result.mean, result.total_covariance


def lineage_state_scores(
    query_groups: Mapping[str, np.ndarray],
    reference_groups: Mapping[str, np.ndarray],
    observed_weights: Mapping[str, float],
    expected_weights: Mapping[str, float],
) -> dict[str, float]:
    """Keep composition, standardized state and heterogeneity components apart.

    Positive composition weight without a measurable state raises an error;
    the function never silently replaces a rare unmeasured state with zero or
    renormalizes it away.
    """

    labels = tuple(expected_weights)
    for name, groups, weights in (
        ("query", query_groups, observed_weights),
        ("reference", reference_groups, expected_weights),
    ):
        unavailable = [
            label
            for label in labels
            if weights.get(label, 0.0) > 0 and label not in groups
        ]
        if unavailable:
            raise ValueError(f"{name} state is unmeasured for {unavailable}")

    query_observed_mean, query_observed_cov = mixture_moments(
        query_groups, observed_weights
    )
    query_standard_mean, query_standard_cov = mixture_moments(
        query_groups, expected_weights
    )
    reference_mean, reference_cov = mixture_moments(reference_groups, expected_weights)
    decomposition = covariance_decomposition(query_groups, observed_weights)
    observed_vector = np.asarray([observed_weights.get(label, 0.0) for label in labels])
    expected_vector = np.asarray([expected_weights[label] for label in labels])
    return {
        "observed_mixture_score": gaussian_wasserstein_distance(
            query_observed_mean,
            query_observed_cov,
            reference_mean,
            reference_cov,
        ),
        "composition_standardized_state_score": gaussian_wasserstein_distance(
            query_standard_mean,
            query_standard_cov,
            reference_mean,
            reference_cov,
        ),
        "composition_only_score": float(
            aitchison_distance(observed_vector, expected_vector)
        ),
        "within_fine_type_heterogeneity": decomposition.within_trace,
        "between_fine_type_heterogeneity": decomposition.between_trace,
        "total_lineage_heterogeneity": decomposition.total_trace,
    }
