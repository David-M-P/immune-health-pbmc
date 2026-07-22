from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from immune_health.cli._reference import write_age_kernel_reference
from immune_health.healthy_reference.kernel import (
    DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    AgeKernelReference,
    age_kernel_weights,
)
from immune_health.healthy_reference.trajectory import HealthyTrajectory


def _moments() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ages = np.asarray([30.0, 40.0, 50.0, 60.0, 35.0, 55.0])
    sexes = np.asarray(["female"] * len(ages))
    datasets = np.asarray(["large"] * 4 + ["small"] * 2)
    donors = np.asarray(
        [f"{dataset}::d{index}" for index, dataset in enumerate(datasets)]
    )
    locations = np.column_stack([ages / 10.0, np.sin(ages / 20.0)])
    covariances = np.stack(
        [np.diag([0.1 + index / 100.0, 0.2]) for index in range(len(ages))]
    )
    return locations, covariances, ages, sexes, datasets, donors


def test_age_kernel_cohort_balancing_changes_only_reference_estimand() -> None:
    _, _, _, _, datasets, donors = _moments()
    ages = np.repeat(50.0, len(donors))
    pooled = age_kernel_weights(
        ages,
        50.0,
        biological_unit_ids=donors,
        datasets=datasets,
        weighting_scheme="donor_pooled",
    )
    balanced = age_kernel_weights(
        ages,
        50.0,
        biological_unit_ids=donors,
        datasets=datasets,
        weighting_scheme="cohort_balanced",
    )
    assert np.isclose(pooled.weights[datasets == "large"].sum(), 4 / 6)
    assert np.isclose(balanced.weights[datasets == "large"].sum(), 0.5)
    assert np.isclose(balanced.weights[datasets == "small"].sum(), 0.5)
    assert balanced.n_support_cohorts == 2


def test_exact_sex_support_default_is_twenty_for_all_kernel_entry_points() -> None:
    ages = np.linspace(20.0, 70.0, 20)
    donors = np.asarray([f"cohort::d{index}" for index in range(20)])
    datasets = np.repeat("cohort", 20)

    insufficient = age_kernel_weights(
        ages,
        45.0,
        sexes=["female"] * 19 + ["male"],
        target_sex="female",
        biological_unit_ids=donors,
        datasets=datasets,
    )
    sufficient = age_kernel_weights(
        ages,
        45.0,
        sexes=["female"] * 20,
        target_sex="female",
        biological_unit_ids=donors,
        datasets=datasets,
    )

    assert DEFAULT_MINIMUM_EXACT_SEX_DONORS == 20
    assert AgeKernelReference().minimum_exact_sex_donors == 20
    assert insufficient.exact_sex_used is False
    assert sufficient.exact_sex_used is True


def test_age_kernel_distribution_score_and_safe_serialization(tmp_path: Path) -> None:
    locations, covariances, ages, sexes, datasets, donors = _moments()
    model = AgeKernelReference(
        bandwidth=8.0,
        minimum_exact_sex_donors=3,
        weighting_scheme="cohort_balanced",
    ).fit(
        locations,
        covariances,
        ages,
        sexes,
        donors,
        datasets=datasets,
    )
    score = model.score_distribution(locations[2], covariances[2], ages[2], sexes[2])
    assert np.isfinite(score["age_matched_gaussian_wasserstein_distance"])
    assert np.isfinite(score["off_trajectory_gaussian_wasserstein_distance"])
    assert np.isfinite(score["predicted_distributional_gp_age"])
    assert score["distributional_gp_age_acceleration"] == pytest.approx(
        score["predicted_distributional_gp_age"] - ages[2]
    )
    assert score["age_kernel_exact_sex_used"] is True
    assert score["age_kernel_n_support_cohorts"] == 2
    assert score["age_kernel_age_extrapolation"] is False
    model.score_distribution(locations[3], covariances[3], ages[3], sexes[3])
    assert len(model._grid_moment_cache_) == 1

    manifest_path = write_age_kernel_reference(
        model,
        tmp_path,
        metadata={
            "feature_ids": ["GP_A::location_0000", "GP_A::location_0001"],
            "endpoint_artifact": {
                "manifest_path": "/immutable/endpoint.json",
                "manifest_sha256": "endpoint-hash",
                "covariances_npy_sha256": "covariance-hash",
                "shape": list(locations.shape),
                "covariance_shape": list(covariances.shape),
                "feature_ids": [
                    "GP_A::location_0000",
                    "GP_A::location_0001",
                ],
                "observation_id_ordered_sha256": "row-hash",
            },
        },
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["covariance_semantics"]["spline_residual_covariance_used"] is False
    assert manifest["storage_contract"]["copied_covariance_archive_written"] is False
    assert not (tmp_path / "age_kernel_reference_arrays.npz").exists()


def test_spline_refuses_within_cell_covariance_conflation() -> None:
    locations, covariances, ages, sexes, datasets, donors = _moments()
    trajectory = HealthyTrajectory(n_spline_knots=0).fit(
        locations,
        ages,
        sexes,
        donors,
        datasets=datasets,
    )
    location_score = trajectory.score(locations[0], ages[0], sexes[0])
    assert (
        location_score["age_matched_location_distance"]
        == (location_score["age_matched_distance"])
    )
    with pytest.raises(ValueError, match="AgeKernelReference"):
        trajectory.score(
            locations[0],
            ages[0],
            sexes[0],
            covariance=covariances[0],
        )
