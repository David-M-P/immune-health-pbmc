"""Documented expression- and rank-based pseudobulk gene-program scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def _log_cpm(counts: sparse.csr_matrix, scale: float) -> sparse.csr_matrix:
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    factors = np.divide(
        scale,
        library_size,
        out=np.zeros_like(library_size, dtype=float),
        where=library_size > 0,
    )
    normalized = (sparse.diags(factors) @ counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def score_gene_programs(
    counts: sparse.spmatrix,
    gene_ids: Sequence[str],
    programs: Mapping[str, Sequence[str]],
    *,
    method: str = "mean_log_cpm",
    cpm_scale: float = 1_000_000.0,
    minimum_genes: int = 1,
) -> pd.DataFrame:
    """Score programs on pseudobulks and return one row per summary and GP.

    ``mean_log_cpm`` is the arithmetic mean of per-gene ``log1p(CPM)``.
    ``mean_percentile_rank`` is the mean within-summary percentile rank of the
    program genes among the complete supplied gene vocabulary.  These methods
    are intentionally simple comparators; they are not silently substituted
    for package-specific enrichment statistics.
    """

    if not sparse.issparse(counts):
        raise TypeError("gene-program scores require a sparse pseudobulk matrix")
    matrix = counts.tocsr(copy=False)
    if matrix.shape[1] != len(gene_ids):
        raise ValueError("gene_ids length does not match count matrix")
    if minimum_genes < 1:
        raise ValueError("minimum_genes must be at least one")
    index = {str(gene): position for position, gene in enumerate(gene_ids)}
    if len(index) != len(gene_ids):
        raise ValueError("gene_ids must be unique")

    mapped: dict[str, list[int]] = {}
    coverage: dict[str, tuple[int, int]] = {}
    for gp_id, genes in programs.items():
        unique_genes = tuple(dict.fromkeys(str(gene) for gene in genes))
        positions = [index[gene] for gene in unique_genes if gene in index]
        mapped[str(gp_id)] = positions
        coverage[str(gp_id)] = (len(positions), len(unique_genes))

    if method == "mean_log_cpm":
        transformed = _log_cpm(matrix, cpm_scale)
        score_by_gp = {
            gp_id: np.asarray(transformed[:, positions].mean(axis=1)).ravel()
            if len(positions) >= minimum_genes
            else np.full(matrix.shape[0], np.nan)
            for gp_id, positions in mapped.items()
        }
    elif method == "mean_percentile_rank":
        score_by_gp = {gp_id: np.full(matrix.shape[0], np.nan) for gp_id in mapped}
        for row_index in range(matrix.shape[0]):
            row = matrix.getrow(row_index).toarray().ravel()
            ranks = rankdata(row, method="average") / len(row)
            for gp_id, positions in mapped.items():
                if len(positions) >= minimum_genes:
                    score_by_gp[gp_id][row_index] = float(ranks[positions].mean())
    else:
        raise ValueError("method must be 'mean_log_cpm' or 'mean_percentile_rank'")

    records: list[dict[str, object]] = []
    for gp_id, scores in score_by_gp.items():
        mapped_count, requested_count = coverage[gp_id]
        for row_index, score in enumerate(scores):
            records.append(
                {
                    "summary_index": row_index,
                    "gp_id": gp_id,
                    "gp_score": score,
                    "score_method": method,
                    "n_program_genes": requested_count,
                    "n_mapped_genes": mapped_count,
                    "gene_coverage": (
                        mapped_count / requested_count if requested_count else 0.0
                    ),
                }
            )
    return pd.DataFrame.from_records(records)
