"""Fine-type-aware donor distribution summaries and uncertainty tools."""

from .bootstrap import (
    fine_type_stratified_bootstrap,
    stratified_bootstrap_indices,
)
from .decomposition import (
    CovarianceDecomposition,
    covariance_decomposition,
    lineage_state_scores,
    mixture_moments,
)
from .distances import (
    centroid_distance,
    gaussian_wasserstein_distance,
    sliced_wasserstein_distance,
)
from .empirical_index import (
    EmpiricalDistributionStore,
    load_empirical_distribution_store,
    write_empirical_row_index,
)
from .statistics import (
    DistributionEstimate,
    shrinkage_covariance,
    summarize_distribution,
)
from .summarize import AggregationResult, aggregate_fine_type_distributions

__all__ = [
    "AggregationResult",
    "CovarianceDecomposition",
    "DistributionEstimate",
    "EmpiricalDistributionStore",
    "aggregate_fine_type_distributions",
    "centroid_distance",
    "covariance_decomposition",
    "fine_type_stratified_bootstrap",
    "gaussian_wasserstein_distance",
    "lineage_state_scores",
    "load_empirical_distribution_store",
    "mixture_moments",
    "shrinkage_covariance",
    "sliced_wasserstein_distance",
    "stratified_bootstrap_indices",
    "summarize_distribution",
    "write_empirical_row_index",
]
