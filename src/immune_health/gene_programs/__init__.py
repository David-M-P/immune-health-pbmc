"""Strict gene identifiers, gene-program resources, and fold-local filtering."""

from immune_health.gene_programs.filters import (
    GPFilterConfig,
    GPFilterResult,
    HVGSelection,
    compute_training_gene_statistics,
    filter_gene_programs_training_only,
    select_hvgs_training_only,
)
from immune_health.gene_programs.identifiers import (
    AmbiguousGeneMappingError,
    GeneMappingResult,
    map_ensembl_to_symbols,
    strip_ensembl_version,
)
from immune_health.gene_programs.io import (
    GeneProgram,
    load_gene_programs,
    validate_gene_programs,
    validate_gp_resource,
)
from immune_health.gene_programs.synthetic import (
    synthetic_gene_mapping,
    synthetic_gene_programs,
    write_synthetic_gp_library,
)
from immune_health.gene_programs.transferability import (
    TransferabilityConfig,
    TransferableGPResult,
    select_transferable_gene_programs,
)
from immune_health.gene_programs.tripso_selection import (
    TripsoGPSelectionConfig,
    TripsoGPSelectionResult,
    select_transferable_tripso_gps,
    validate_crossfit_reference_run,
    validate_tripso_gp_selection_manifest,
    write_tripso_gp_selection,
)

__all__ = [
    "AmbiguousGeneMappingError",
    "GPFilterConfig",
    "GPFilterResult",
    "GeneMappingResult",
    "GeneProgram",
    "HVGSelection",
    "TransferabilityConfig",
    "TransferableGPResult",
    "TripsoGPSelectionConfig",
    "TripsoGPSelectionResult",
    "compute_training_gene_statistics",
    "filter_gene_programs_training_only",
    "load_gene_programs",
    "map_ensembl_to_symbols",
    "select_hvgs_training_only",
    "select_transferable_gene_programs",
    "select_transferable_tripso_gps",
    "strip_ensembl_version",
    "synthetic_gene_mapping",
    "synthetic_gene_programs",
    "validate_gene_programs",
    "validate_gp_resource",
    "validate_crossfit_reference_run",
    "validate_tripso_gp_selection_manifest",
    "write_synthetic_gp_library",
    "write_tripso_gp_selection",
]
