"""Training-only healthy-age trajectories and frozen query scoring."""

from .bootstrap import bootstrap_healthy_reference_scores
from .diagnostics import (
    age_support_grid,
    cohort_feature_age_effects,
    query_age_support,
)
from .empirical import score_empirical_matched_depth
from .kernel import (
    AgeKernelReference,
    KernelWeightResult,
    age_kernel_weights,
    weighted_mixture_moments,
)
from .trajectory import (
    REFERENCE_WEIGHTING_SCHEMES,
    CrossFitResult,
    HealthyTrajectory,
    cross_fit_trajectory,
    fit_age_direction,
    reference_row_weights,
)
from .uncertainty import combine_seed_score_tables

__all__ = [
    "AgeKernelReference",
    "CrossFitResult",
    "HealthyTrajectory",
    "KernelWeightResult",
    "REFERENCE_WEIGHTING_SCHEMES",
    "age_support_grid",
    "age_kernel_weights",
    "bootstrap_healthy_reference_scores",
    "combine_seed_score_tables",
    "cohort_feature_age_effects",
    "cross_fit_trajectory",
    "fit_age_direction",
    "query_age_support",
    "reference_row_weights",
    "score_empirical_matched_depth",
    "weighted_mixture_moments",
]
