"""Age-kernel healthy references with exact-sex matching and safe fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from immune_health.aggregation.distances import gaussian_wasserstein_distance

from .trajectory import REFERENCE_WEIGHTING_SCHEMES, reference_row_weights

DEFAULT_MINIMUM_EXACT_SEX_DONORS = 20


@dataclass(frozen=True)
class KernelWeightResult:
    """Normalized training-row weights and matching diagnostics."""

    weights: np.ndarray
    exact_sex_used: bool
    n_support_donors: int
    n_support_cohorts: int
    effective_support_donors: float
    target_age_in_support_range: bool
    support_age_min: float
    support_age_max: float


def age_kernel_weights(
    ages: Sequence[float],
    target_age: float,
    *,
    sexes: Sequence[str] | None = None,
    target_sex: str | None = None,
    biological_unit_ids: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    weighting_scheme: str = "donor_pooled",
    exclude_biological_units: Sequence[str] = (),
    bandwidth: float = 10.0,
    minimum_exact_sex_donors: int = DEFAULT_MINIMUM_EXACT_SEX_DONORS,
) -> KernelWeightResult:
    """Compute donor- or cohort-balanced Gaussian age-kernel weights.

    Exact-sex matching is used only when it has the configured donor support;
    otherwise both sexes are retained and ``exact_sex_used`` reports the
    documented fallback.  Exclusion enables leave-donor-out cross-fitting.
    """

    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or len(age) == 0 or not np.isfinite(age).all():
        raise ValueError("training ages must be a nonempty finite vector")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    groups = (
        np.asarray(biological_unit_ids, dtype=str)
        if biological_unit_ids is not None
        else np.asarray([f"row_{index}" for index in range(len(age))])
    )
    if len(groups) != len(age):
        raise ValueError("biological unit IDs and ages must align")
    if weighting_scheme not in REFERENCE_WEIGHTING_SCHEMES:
        raise ValueError(f"unknown reference weighting scheme: {weighting_scheme}")
    dataset = None if datasets is None else np.asarray(datasets, dtype=str)
    if dataset is not None and dataset.shape != groups.shape:
        raise ValueError("datasets and ages must align")
    if weighting_scheme == "cohort_balanced" and dataset is None:
        raise ValueError("cohort_balanced weighting requires dataset labels")
    allowed = ~np.isin(groups, np.asarray(exclude_biological_units, dtype=str))
    exact_sex = False
    if sexes is not None or target_sex is not None:
        if sexes is None or target_sex is None:
            raise ValueError("sexes and target_sex must be provided together")
        sex = np.asarray(sexes, dtype=str)
        if len(sex) != len(age):
            raise ValueError("sexes and ages must align")
        sex_mask = allowed & (sex == str(target_sex))
        exact_cohorts = 0 if dataset is None else len(np.unique(dataset[sex_mask]))
        if len(np.unique(groups[sex_mask])) >= minimum_exact_sex_donors and (
            weighting_scheme != "cohort_balanced" or exact_cohorts >= 2
        ):
            allowed = sex_mask
            exact_sex = True
    if not allowed.any():
        raise ValueError("no healthy-reference donors remain after exclusions")

    base_weights = reference_row_weights(
        groups[allowed],
        None if dataset is None else dataset[allowed],
        scheme=weighting_scheme,
    )
    allowed_groups = np.unique(groups[allowed])
    exponent = -0.5 * ((age - float(target_age)) / bandwidth) ** 2
    exponent -= np.max(exponent[allowed])
    weights = np.zeros(len(age), dtype=float)
    weights[allowed] = base_weights * np.exp(exponent[allowed])
    if weights.sum() <= 0:
        raise RuntimeError("age-kernel weights underflowed")
    weights /= weights.sum()
    donor_totals = np.asarray(
        [weights[groups == donor].sum() for donor in allowed_groups], dtype=float
    )
    effective_donors = float(1.0 / np.square(donor_totals).sum())
    support_ages = age[allowed]
    support_cohorts = 0 if dataset is None else len(np.unique(dataset[allowed]))
    return KernelWeightResult(
        weights=weights,
        exact_sex_used=exact_sex,
        n_support_donors=len(allowed_groups),
        n_support_cohorts=support_cohorts,
        effective_support_donors=effective_donors,
        target_age_in_support_range=bool(
            float(support_ages.min()) <= float(target_age) <= float(support_ages.max())
        ),
        support_age_min=float(support_ages.min()),
        support_age_max=float(support_ages.max()),
    )


def weighted_mixture_moments(
    locations: np.ndarray,
    covariances: np.ndarray,
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine donor Gaussian summaries, including between-donor covariance."""

    # Preserve endpoint float32/memmap storage. Converting the complete covariance
    # tensor to float64 would add roughly 0.5 GiB per 1,000 256-D donor rows.
    means = np.asarray(locations)
    covs = np.asarray(covariances)
    probability = np.asarray(weights, dtype=float).copy()
    if means.ndim != 2 or covs.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("location and covariance dimensions are inconsistent")
    if probability.shape != (len(means),) or np.any(probability < 0):
        raise ValueError("weights must be one nonnegative value per donor summary")
    if not np.isfinite(means).all() or not np.isfinite(probability).all():
        raise ValueError("healthy Gaussian summaries must be finite")
    if probability.sum() <= 0:
        raise ValueError("weights must have positive total")
    probability /= probability.sum()
    mean = probability @ means
    covariance = np.zeros(covs.shape[1:], dtype=np.float64)
    for weight, location, within in zip(probability, means, covs):
        if weight == 0:
            continue
        if not np.isfinite(within).all() or not np.allclose(
            within, within.T, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("healthy covariance summaries must be finite/symmetric")
        difference = location - mean
        covariance += weight * (within + np.outer(difference, difference))
    covariance = (covariance + covariance.T) / 2.0
    return np.asarray(mean, dtype=float), covariance


class AgeKernelReference:
    """Frozen empirical healthy reference over donor distribution moments."""

    def __init__(
        self,
        *,
        bandwidth: float = 10.0,
        minimum_exact_sex_donors: int = DEFAULT_MINIMUM_EXACT_SEX_DONORS,
        weighting_scheme: str = "donor_pooled",
        age_grid_size: int = 101,
    ) -> None:
        if bandwidth <= 0 or minimum_exact_sex_donors < 1 or age_grid_size < 2:
            raise ValueError("age-kernel hyperparameters are invalid")
        self.bandwidth = bandwidth
        self.minimum_exact_sex_donors = minimum_exact_sex_donors
        if weighting_scheme not in REFERENCE_WEIGHTING_SCHEMES:
            raise ValueError(f"unknown reference weighting scheme: {weighting_scheme}")
        self.weighting_scheme = weighting_scheme
        self.age_grid_size = age_grid_size

    def fit(
        self,
        locations: np.ndarray,
        covariances: np.ndarray,
        ages: Sequence[float],
        sexes: Sequence[str],
        biological_unit_ids: Sequence[str],
        *,
        datasets: Sequence[str],
    ) -> "AgeKernelReference":
        self.locations_ = np.asanyarray(locations)
        self.covariances_ = np.asanyarray(covariances)
        self.ages_ = np.asarray(ages, dtype=float).copy()
        self.sexes_ = np.asarray(sexes, dtype=str).copy()
        self.biological_unit_ids_ = np.asarray(biological_unit_ids, dtype=str).copy()
        self.datasets_ = np.asarray(datasets, dtype=str).copy()
        if self.locations_.ndim != 2 or self.covariances_.shape != (
            len(self.locations_),
            self.locations_.shape[1],
            self.locations_.shape[1],
        ):
            raise ValueError("location and covariance dimensions are inconsistent")
        if not np.isfinite(self.locations_).all() or not np.isfinite(self.ages_).all():
            raise ValueError("healthy locations and ages must be finite")
        if any(
            len(value) != len(self.locations_)
            for value in (
                self.ages_,
                self.sexes_,
                self.biological_unit_ids_,
                self.datasets_,
            )
        ):
            raise ValueError("healthy-reference arrays must align by row")
        # Validate donor/cohort mapping and persist the exact base weights used by
        # this estimand. Kernel localization is applied on top at query time.
        self.training_row_weights_ = reference_row_weights(
            self.biological_unit_ids_,
            self.datasets_,
            scheme=self.weighting_scheme,
        )
        self.training_biological_units_ = frozenset(self.biological_unit_ids_)
        self.age_range_ = (float(self.ages_.min()), float(self.ages_.max()))
        self.n_features_in_ = self.locations_.shape[1]
        self.endpoint_arrays_reused_without_copy_ = True
        self._grid_moment_cache_: dict[
            tuple[str, bytes], tuple[np.ndarray, np.ndarray]
        ] = {}
        return self

    def expected_moments(
        self,
        age: float,
        sex: str,
        *,
        exclude_biological_units: Sequence[str] = (),
    ) -> tuple[np.ndarray, np.ndarray, KernelWeightResult]:
        if not hasattr(self, "locations_"):
            raise RuntimeError("age-kernel reference has not been fitted")
        result = age_kernel_weights(
            self.ages_,
            age,
            sexes=self.sexes_,
            target_sex=sex,
            biological_unit_ids=self.biological_unit_ids_,
            datasets=self.datasets_,
            weighting_scheme=self.weighting_scheme,
            exclude_biological_units=exclude_biological_units,
            bandwidth=self.bandwidth,
            minimum_exact_sex_donors=self.minimum_exact_sex_donors,
        )
        mean, covariance = weighted_mixture_moments(
            self.locations_, self.covariances_, result.weights
        )
        return mean, covariance, result

    def score_distribution(
        self,
        location: np.ndarray,
        covariance: np.ndarray,
        age: float,
        sex: str,
        *,
        exclude_biological_units: Sequence[str] = (),
        age_grid: Sequence[float] | None = None,
    ) -> dict[str, float | bool | int]:
        """Score query moments against the age-matched healthy mixture.

        This is intentionally separate from ``HealthyTrajectory.score``: donor
        within-cell covariance is compared only with the kernel mixture of donor
        within-cell covariances, never with residual covariance of spline means.
        """

        point = np.asarray(location, dtype=float)
        query_covariance = np.asarray(covariance, dtype=float)
        if point.shape != (self.n_features_in_,):
            raise ValueError("query location differs from age-kernel dimension")
        if query_covariance.shape != (self.n_features_in_, self.n_features_in_):
            raise ValueError("query covariance differs from age-kernel dimension")
        expected_location, expected_covariance, result = self.expected_moments(
            age,
            sex,
            exclude_biological_units=exclude_biological_units,
        )
        distance = gaussian_wasserstein_distance(
            point,
            query_covariance,
            expected_location,
            expected_covariance,
        )
        grid = (
            np.linspace(*self.age_range_, self.age_grid_size)
            if age_grid is None
            else np.asarray(age_grid, dtype=float)
        )
        if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
            raise ValueError("age_grid must contain at least two finite ages")
        cache_key = (str(sex), np.asarray(grid, dtype=np.float64).tobytes())
        cache_allowed = len(exclude_biological_units) == 0
        cached = self._grid_moment_cache_.get(cache_key) if cache_allowed else None
        if cached is None:
            grid_locations: list[np.ndarray] = []
            grid_covariances: list[np.ndarray] = []
            for candidate_age in grid:
                candidate_location, candidate_covariance, _ = self.expected_moments(
                    float(candidate_age),
                    sex,
                    exclude_biological_units=exclude_biological_units,
                )
                grid_locations.append(candidate_location)
                grid_covariances.append(candidate_covariance)
            cached = (np.vstack(grid_locations), np.stack(grid_covariances))
            if cache_allowed:
                self._grid_moment_cache_[cache_key] = cached
        grid_locations, grid_covariances = cached
        trajectory_distances = []
        for candidate_location, candidate_covariance in zip(
            grid_locations, grid_covariances, strict=True
        ):
            trajectory_distances.append(
                gaussian_wasserstein_distance(
                    point,
                    query_covariance,
                    candidate_location,
                    candidate_covariance,
                )
            )
        trajectory_distances = np.asarray(trajectory_distances, dtype=float)
        minimum = float(trajectory_distances.min())
        tied = np.flatnonzero(
            np.isclose(trajectory_distances, minimum, rtol=1e-10, atol=1e-12)
        )
        predicted_distributional_age = float(np.mean(grid[tied]))
        return {
            "age_matched_gaussian_wasserstein_distance": distance,
            "off_trajectory_gaussian_wasserstein_distance": minimum,
            "predicted_distributional_gp_age": predicted_distributional_age,
            "distributional_gp_age_acceleration": (
                predicted_distributional_age - float(age)
            ),
            "age_kernel_exact_sex_used": result.exact_sex_used,
            "age_kernel_n_support_donors": result.n_support_donors,
            "age_kernel_n_support_cohorts": result.n_support_cohorts,
            "age_kernel_effective_support_donors": result.effective_support_donors,
            "age_kernel_age_extrapolation": not result.target_age_in_support_range,
            "age_kernel_support_age_min": result.support_age_min,
            "age_kernel_support_age_max": result.support_age_max,
        }
