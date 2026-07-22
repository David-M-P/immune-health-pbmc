"""Donor-level comparators used alongside TRIPSO.

All fit operations in this package consume donor/observation summaries.  Cells
are used only to make those summaries and are never treated as independent
biological replicates.
"""

from .composition import (
    AgeSexCompositionModel,
    aitchison_distance,
    build_composition_table,
    centered_log_ratio,
    composition_matrix,
)
from .gp_scores import score_gene_programs
from .latent import ElasticNetAgeModel, TrainOnlyPCA, log_cpm
from .pseudobulk import (
    PseudobulkResult,
    build_pseudobulk,
    ensure_donor_observation_ids,
)

__all__ = [
    "AgeSexCompositionModel",
    "ElasticNetAgeModel",
    "PseudobulkResult",
    "TrainOnlyPCA",
    "aitchison_distance",
    "build_composition_table",
    "build_pseudobulk",
    "centered_log_ratio",
    "composition_matrix",
    "ensure_donor_observation_ids",
    "log_cpm",
    "score_gene_programs",
]
