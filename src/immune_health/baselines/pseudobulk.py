"""Sparse raw-count pseudobulk at donor-observation by fine-type resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

IDENTIFIER_COLUMNS = (
    "dataset",
    "donor_id",
    "biological_unit_id",
    "sample_id",
    "source_observation_id",
    "observation_id",
)


@dataclass(frozen=True)
class PseudobulkResult:
    """A compact sparse matrix plus one metadata row per pseudobulk row."""

    counts: sparse.csr_matrix
    metadata: pd.DataFrame
    gene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.counts.shape[0] != len(self.metadata):
            raise ValueError("pseudobulk matrix and metadata row counts differ")
        if self.counts.shape[1] != len(self.gene_ids):
            raise ValueError("pseudobulk matrix and gene vocabulary sizes differ")


def _required_strings(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing required metadata columns: {missing}")
    for column in columns:
        values = frame[column].astype("string")
        invalid = values.isna() | values.str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{column!r} contains missing or empty identifiers")


def ensure_donor_observation_ids(
    obs: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    donor_col: str = "donor_id",
    sample_col: str = "sample_id",
    copy: bool = True,
) -> pd.DataFrame:
    """Create or validate globally collision-safe biological identifiers.

    ``sample_id`` is retained as the source observation label.  It is not
    assumed to identify a donor: OneK1K sequencing pools, for example, contain
    multiple donors.  The approved observation key is therefore
    ``dataset::donor_id::sample_id``.
    """

    _required_strings(obs, (dataset_col, donor_col, sample_col))
    result = obs.copy() if copy else obs
    dataset = result[dataset_col].astype("string")
    donor = result[donor_col].astype("string")
    sample = result[sample_col].astype("string")
    biological = dataset + "::" + donor
    observation = biological + "::" + sample

    expected = {
        "biological_unit_id": biological,
        "source_observation_id": dataset + "::" + sample,
        "observation_id": observation,
    }
    for column, values in expected.items():
        if column in result:
            actual = result[column].astype("string")
            mismatch = actual.ne(values) | actual.isna()
            if mismatch.any():
                examples = result.loc[
                    mismatch, [dataset_col, donor_col, sample_col]
                ].head(3)
                raise ValueError(
                    f"{column!r} violates the approved identifier contract; "
                    f"examples={examples.to_dict(orient='records')}"
                )
        result[column] = values
    return result


def _validate_raw_sparse_counts(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise TypeError("raw counts must be a scipy sparse matrix")
    counts = matrix.tocsr(copy=False)
    if counts.ndim != 2:
        raise ValueError("raw counts must be two dimensional")
    if not np.isfinite(counts.data).all():
        raise ValueError("raw counts contain non-finite stored values")
    if np.any(counts.data < 0):
        raise ValueError("raw counts contain negative values")
    if not np.allclose(counts.data, np.rint(counts.data), atol=1e-8, rtol=0.0):
        raise ValueError("matrix is not raw integer-like count data")
    return counts


def build_pseudobulk(
    raw_counts: sparse.spmatrix,
    obs: pd.DataFrame,
    gene_ids: Iterable[str] | None = None,
    *,
    lineage_col: str = "lineage",
    fine_type_col: str = "fine_type",
    min_cells: int = 1,
    carry_columns: Sequence[str] = ("age", "sex"),
) -> PseudobulkResult:
    """Sum sparse raw counts by donor observation, lineage and fine type.

    The aggregation matrix is sparse, so the cell-by-gene input is never
    densified.  ``carry_columns`` must be constant inside every resulting
    group.  Groups below ``min_cells`` are omitted from the state matrix; use
    :func:`build_composition_table` on the original metadata to retain their
    composition information.
    """

    if min_cells < 1:
        raise ValueError("min_cells must be at least one")
    counts = _validate_raw_sparse_counts(raw_counts)
    if counts.shape[0] != len(obs):
        raise ValueError("count rows and observation metadata rows differ")
    frame = ensure_donor_observation_ids(obs)
    _required_strings(frame, (lineage_col, fine_type_col))

    group_columns = [
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
        lineage_col,
        fine_type_col,
    ]
    keys = pd.MultiIndex.from_frame(frame[group_columns])
    codes, unique_keys = pd.factorize(keys, sort=False)
    n_groups = len(unique_keys)
    aggregator = sparse.csr_matrix(
        (np.ones(len(frame), dtype=np.int64), (codes, np.arange(len(frame)))),
        shape=(n_groups, len(frame)),
    )
    aggregated = (aggregator @ counts).tocsr()
    n_cells = np.bincount(codes, minlength=n_groups)

    metadata = unique_keys.to_frame(index=False)
    metadata.columns = group_columns
    for column in carry_columns:
        if column not in frame:
            continue
        grouped = frame.groupby(group_columns, sort=False, observed=True)[column]
        if (grouped.nunique(dropna=False) > 1).any():
            raise ValueError(f"{column!r} is inconsistent within a pseudobulk group")
        metadata[column] = grouped.first().to_numpy()
    metadata["n_cells"] = n_cells
    metadata["library_size"] = np.asarray(aggregated.sum(axis=1)).ravel()

    keep = n_cells >= min_cells
    metadata = metadata.loc[keep].reset_index(drop=True)
    aggregated = aggregated[keep].tocsr()

    if gene_ids is None:
        genes = tuple(f"gene_{index}" for index in range(counts.shape[1]))
    else:
        genes = tuple(str(gene) for gene in gene_ids)
        if len(genes) != counts.shape[1]:
            raise ValueError("gene_ids length does not match the count matrix")
        if len(set(genes)) != len(genes):
            raise ValueError("gene_ids must be unique")
    return PseudobulkResult(aggregated, metadata, genes)
