"""Training-fold-only gene-program and highly-variable-gene selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from immune_health.gene_programs.io import GeneProgram, validate_gene_programs

try:
    from scipy import sparse
except ImportError:  # pragma: no cover - an informative error is raised at use
    sparse = None


@dataclass(frozen=True)
class GPFilterConfig:
    """Reviewable defaults for fold-local program filtering."""

    minimum_mapped_genes: int = 10
    maximum_program_size: int = 200
    minimum_expression_coverage: float = 0.0
    minimum_donor_coverage: float = 0.0
    minimum_dataset_fraction: float = 1.0
    redundancy_jaccard_threshold: float = 0.8
    maximum_pairwise_overlap: int | None = None

    def validate(self) -> None:
        if self.minimum_mapped_genes < 1:
            raise ValueError("minimum_mapped_genes must be positive")
        if self.maximum_program_size < self.minimum_mapped_genes:
            raise ValueError("maximum_program_size is below minimum_mapped_genes")
        for name, value in (
            ("minimum_expression_coverage", self.minimum_expression_coverage),
            ("minimum_donor_coverage", self.minimum_donor_coverage),
            ("minimum_dataset_fraction", self.minimum_dataset_fraction),
            ("redundancy_jaccard_threshold", self.redundancy_jaccard_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.minimum_dataset_fraction == 0:
            raise ValueError("minimum_dataset_fraction must be greater than zero")
        if (
            self.maximum_pairwise_overlap is not None
            and self.maximum_pairwise_overlap < 0
        ):
            raise ValueError("maximum_pairwise_overlap must be nonnegative")


@dataclass(frozen=True)
class GPFilterResult:
    programs: tuple[GeneProgram, ...]
    report: pd.DataFrame
    gene_support: pd.DataFrame
    training_datasets: tuple[str, ...]
    heldout_dataset: str | None
    ignored_nontraining_rows: int


@dataclass(frozen=True)
class HVGSelection:
    selected_genes: tuple[str, ...]
    scores: pd.DataFrame
    training_datasets: tuple[str, ...]
    heldout_dataset: str
    n_training_cells: int


def _training_datasets(
    training_datasets: Iterable[str], heldout_dataset: str | None
) -> tuple[str, ...]:
    selected = tuple(sorted(set(map(str, training_datasets))))
    if not selected:
        raise ValueError("At least one training dataset is required")
    if heldout_dataset is not None and str(heldout_dataset) in selected:
        raise ValueError("Held-out dataset cannot be a training dataset")
    return selected


def _program_lineages(program: GeneProgram) -> set[str] | None:
    value = program.metadata.get(
        "lineages", program.metadata.get("applicable_lineages")
    )
    if value is None or value == "":
        return None
    if isinstance(value, str):
        normalized = value.replace("|", ",").replace(";", ",")
        return {item.strip() for item in normalized.split(",") if item.strip()}
    return {str(item) for item in value}


def filter_gene_programs_training_only(
    programs: Iterable[GeneProgram],
    gene_statistics: pd.DataFrame,
    *,
    training_datasets: Iterable[str],
    heldout_dataset: str | None,
    config: GPFilterConfig = GPFilterConfig(),
    lineage: str | None = None,
    dataset_column: str = "dataset",
    gene_column: str = "gene",
    expression_coverage_column: str = "expression_coverage",
    donor_coverage_column: str = "donor_coverage",
) -> GPFilterResult:
    """Filter programs using only the explicitly supplied reference datasets.

    ``heldout_dataset=None`` is the explicit final-reference mode: every supplied
    dataset is eligible, while all filtering still uses adaptation donors only.
    """

    config.validate()
    library = tuple(programs)
    validate_gene_programs(library)
    selected_datasets = _training_datasets(training_datasets, heldout_dataset)
    required = {
        dataset_column,
        gene_column,
        expression_coverage_column,
        donor_coverage_column,
    }
    missing = sorted(required - set(gene_statistics.columns))
    if missing:
        raise ValueError(f"Gene statistics are missing columns: {missing}")
    stats = gene_statistics.copy()
    stats[dataset_column] = stats[dataset_column].astype(str)
    stats[gene_column] = stats[gene_column].astype(str)
    ignored = int((~stats[dataset_column].isin(selected_datasets)).sum())
    stats = stats.loc[stats[dataset_column].isin(selected_datasets)].copy()
    duplicated = stats.duplicated([dataset_column, gene_column], keep=False)
    if duplicated.any():
        raise ValueError("Gene statistics contain duplicate dataset/gene rows")
    present = set(stats[dataset_column])
    missing_training = sorted(set(selected_datasets) - present)
    if missing_training:
        raise ValueError(f"Gene statistics lack training datasets: {missing_training}")
    stats[expression_coverage_column] = pd.to_numeric(
        stats[expression_coverage_column], errors="raise"
    )
    stats[donor_coverage_column] = pd.to_numeric(
        stats[donor_coverage_column], errors="raise"
    )
    for column in (expression_coverage_column, donor_coverage_column):
        if (~stats[column].between(0, 1)).any():
            raise ValueError(f"{column} values must be proportions between 0 and 1")

    stats["passes_training_coverage"] = stats[expression_coverage_column].ge(
        config.minimum_expression_coverage
    ) & stats[donor_coverage_column].ge(config.minimum_donor_coverage)
    support = (
        stats.groupby(gene_column, observed=True)
        .agg(
            datasets_observed=(dataset_column, "nunique"),
            datasets_passing=("passes_training_coverage", "sum"),
            minimum_expression_coverage=(expression_coverage_column, "min"),
            minimum_donor_coverage=(donor_coverage_column, "min"),
        )
        .reset_index()
        .rename(columns={gene_column: "gene"})
    )
    support["training_dataset_fraction"] = support["datasets_passing"] / len(
        selected_datasets
    )
    support["supported"] = support["training_dataset_fraction"].ge(
        config.minimum_dataset_fraction
    )
    mapped_genes = set(support["gene"].astype(str))
    supported_genes = set(support.loc[support["supported"], "gene"].astype(str))

    report_rows: dict[str, dict[str, object]] = {}
    candidates: list[GeneProgram] = []
    for program in library:
        mapped = tuple(gene for gene in program.genes if gene in mapped_genes)
        supported_members = tuple(gene for gene in mapped if gene in supported_genes)
        reasons: list[str] = []
        applicable = _program_lineages(program)
        if lineage is not None and applicable is not None and lineage not in applicable:
            reasons.append("not_applicable_to_lineage")
        if len(mapped) < config.minimum_mapped_genes:
            reasons.append("too_few_mapped_genes")
        if len(mapped) > config.maximum_program_size:
            reasons.append("program_too_large")
        if len(supported_members) < config.minimum_mapped_genes:
            reasons.append("too_few_training_supported_genes")
        report_rows[program.program_id] = {
            "program_id": program.program_id,
            "source": program.source,
            "lineage": lineage,
            "n_input_genes": len(program.genes),
            "n_mapped_training_genes": len(mapped),
            "n_training_supported_genes": len(supported_members),
            "retained": not reasons,
            "reason": "|".join(reasons) if reasons else "retained",
            "redundant_with": pd.NA,
            "pairwise_jaccard": np.nan,
            "pairwise_overlap": pd.NA,
        }
        if not reasons:
            metadata = dict(program.metadata)
            metadata.update(
                {
                    "filtered_on_training_datasets": selected_datasets,
                    "heldout_dataset_excluded": (
                        None if heldout_dataset is None else str(heldout_dataset)
                    ),
                }
            )
            candidates.append(
                replace(program, genes=supported_members, metadata=metadata)
            )

    retained: list[GeneProgram] = []
    ordered_candidates = sorted(
        candidates, key=lambda item: (-len(item.genes), item.program_id)
    )
    for candidate in ordered_candidates:
        candidate_genes = set(candidate.genes)
        conflict: tuple[GeneProgram, float, int, str] | None = None
        for accepted in retained:
            accepted_genes = set(accepted.genes)
            overlap = len(candidate_genes & accepted_genes)
            union = len(candidate_genes | accepted_genes)
            jaccard = overlap / union if union else 0.0
            reason = ""
            if jaccard >= config.redundancy_jaccard_threshold:
                reason = "redundant_jaccard"
            elif (
                config.maximum_pairwise_overlap is not None
                and overlap > config.maximum_pairwise_overlap
            ):
                reason = "excessive_pairwise_overlap"
            if reason:
                conflict = (accepted, jaccard, overlap, reason)
                break
        if conflict is None:
            retained.append(candidate)
            continue
        accepted, jaccard, overlap, reason = conflict
        row = report_rows[candidate.program_id]
        row["retained"] = False
        row["reason"] = reason
        row["redundant_with"] = accepted.program_id
        row["pairwise_jaccard"] = jaccard
        row["pairwise_overlap"] = overlap
    retained.sort(key=lambda item: item.program_id)
    report = (
        pd.DataFrame(report_rows.values())
        .sort_values("program_id")
        .reset_index(drop=True)
    )
    return GPFilterResult(
        programs=tuple(retained),
        report=report,
        gene_support=support.sort_values("gene").reset_index(drop=True),
        training_datasets=selected_datasets,
        heldout_dataset=(None if heldout_dataset is None else str(heldout_dataset)),
        ignored_nontraining_rows=ignored,
    )


def _validate_matrix_inputs(
    matrix: object, metadata: pd.DataFrame, gene_ids: Sequence[str]
) -> tuple[int, int]:
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError("Expression matrix must be two-dimensional")
    n_cells, n_genes = map(int, matrix.shape)
    if n_cells != len(metadata) or n_genes != len(gene_ids):
        raise ValueError(
            "Expression shape does not match metadata/gene IDs: "
            f"{matrix.shape}, {len(metadata)}, {len(gene_ids)}"
        )
    if len(set(map(str, gene_ids))) != len(gene_ids):
        raise ValueError("gene_ids must be unique")
    return n_cells, n_genes


def _training_mask(
    metadata: pd.DataFrame,
    training_datasets: Iterable[str],
    heldout_dataset: str,
    dataset_column: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    selected = _training_datasets(training_datasets, heldout_dataset)
    if dataset_column not in metadata:
        raise ValueError(f"Metadata lacks dataset column {dataset_column!r}")
    values = metadata[dataset_column].astype(str)
    mask = values.isin(selected).to_numpy()
    if not mask.any():
        raise ValueError("No training cells were selected")
    missing = sorted(set(selected) - set(values[mask]))
    if missing:
        raise ValueError(f"No cells found for training datasets: {missing}")
    return mask, selected


def select_hvgs_training_only(
    matrix: object,
    metadata: pd.DataFrame,
    gene_ids: Sequence[str],
    *,
    training_datasets: Iterable[str],
    heldout_dataset: str,
    n_top_genes: int,
    dataset_column: str = "dataset",
    minimum_training_cells_expressing: int = 1,
    minimum_mean: float = 0.0,
    maximum_mean: float = np.inf,
) -> HVGSelection:
    """Select overdispersed genes without reading held-out rows."""

    _validate_matrix_inputs(matrix, metadata, gene_ids)
    if n_top_genes < 1 or minimum_training_cells_expressing < 1:
        raise ValueError("n_top_genes and expression count threshold must be positive")
    mask, selected_datasets = _training_mask(
        metadata, training_datasets, heldout_dataset, dataset_column
    )
    training = matrix[mask]
    is_sparse = sparse is not None and sparse.issparse(training)
    if is_sparse:
        means = np.asarray(training.mean(axis=0)).ravel()
        second = np.asarray(training.power(2).mean(axis=0)).ravel()
        expressing = np.asarray((training > 0).getnnz(axis=0)).ravel()
    else:
        array = np.asarray(training)
        means = np.mean(array, axis=0)
        second = np.mean(np.square(array), axis=0)
        expressing = np.count_nonzero(array > 0, axis=0)
    variance = np.maximum(second - np.square(means), 0)
    dispersion = variance / np.maximum(means, np.finfo(float).eps)
    gene_array = np.asarray(list(map(str, gene_ids)), dtype=object)
    eligible = (
        np.isfinite(dispersion)
        & (expressing >= minimum_training_cells_expressing)
        & (means >= minimum_mean)
        & (means <= maximum_mean)
    )
    eligible_positions = np.flatnonzero(eligible)
    order = np.lexsort(
        (gene_array[eligible_positions], -dispersion[eligible_positions])
    )
    chosen_positions = eligible_positions[order[: min(n_top_genes, len(order))]]
    chosen = tuple(gene_array[chosen_positions].tolist())
    scores = pd.DataFrame(
        {
            "gene": gene_array,
            "training_mean": means,
            "training_variance": variance,
            "training_dispersion": dispersion,
            "training_cells_expressing": expressing,
            "eligible": eligible,
            "selected": np.isin(np.arange(len(gene_array)), chosen_positions),
        }
    ).sort_values(
        ["selected", "training_dispersion", "gene"],
        ascending=[False, False, True],
    )
    return HVGSelection(
        selected_genes=chosen,
        scores=scores.reset_index(drop=True),
        training_datasets=selected_datasets,
        heldout_dataset=str(heldout_dataset),
        n_training_cells=int(mask.sum()),
    )


def compute_training_gene_statistics(
    matrix: object,
    metadata: pd.DataFrame,
    gene_ids: Sequence[str],
    *,
    training_datasets: Iterable[str],
    heldout_dataset: str,
    dataset_column: str = "dataset",
    donor_column: str = "biological_unit_id",
) -> pd.DataFrame:
    """Compute cell/donor expression coverage from training rows only."""

    _validate_matrix_inputs(matrix, metadata, gene_ids)
    mask, selected_datasets = _training_mask(
        metadata, training_datasets, heldout_dataset, dataset_column
    )
    if donor_column not in metadata:
        raise ValueError(f"Metadata lacks donor column {donor_column!r}")
    genes = np.asarray(list(map(str, gene_ids)), dtype=object)
    rows: list[pd.DataFrame] = []
    dataset_values = metadata[dataset_column].astype(str).to_numpy()
    for dataset in selected_datasets:
        positions = np.flatnonzero(mask & (dataset_values == dataset))
        subset = matrix[positions]
        subset_meta = metadata.iloc[positions].reset_index(drop=True)
        is_sparse = sparse is not None and sparse.issparse(subset)
        if is_sparse:
            expressed = subset > 0
            cell_counts = np.asarray(expressed.getnnz(axis=0)).ravel()
        else:
            expressed = np.asarray(subset) > 0
            cell_counts = np.count_nonzero(expressed, axis=0)
        donor_expressed = np.zeros(len(genes), dtype=int)
        donor_groups = subset_meta.groupby(donor_column, observed=True).indices
        for donor_positions in donor_groups.values():
            if is_sparse:
                donor_has_gene = (
                    np.asarray(
                        expressed[np.asarray(donor_positions)].getnnz(axis=0)
                    ).ravel()
                    > 0
                )
            else:
                donor_has_gene = expressed[np.asarray(donor_positions)].any(axis=0)
            donor_expressed += donor_has_gene.astype(int)
        n_donors = len(donor_groups)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "gene": genes,
                    "n_cells": len(positions),
                    "n_donors": n_donors,
                    "expression_coverage": cell_counts / len(positions),
                    "donor_coverage": donor_expressed / n_donors,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
