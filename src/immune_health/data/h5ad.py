"""Read small, sparse subsets from merged H5ADs without modifying inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.data.ids import add_stable_identifiers


@dataclass(frozen=True)
class LineageData:
    counts: sparse.csr_matrix
    obs: pd.DataFrame
    gene_ids: np.ndarray
    source_path: Path


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _take_hdf5_rows(dataset: h5py.Dataset, rows: np.ndarray | None) -> np.ndarray:
    """Take arbitrary rows despite h5py's sorted-unique fancy-index rule."""
    if rows is None:
        return dataset[:]
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    return dataset[unique_rows][inverse]


def read_obs_column(
    group: h5py.Group, name: str, rows: np.ndarray | None = None
) -> pd.Series:
    obj = group[name]
    if isinstance(obj, h5py.Group):
        categories = [_decode(value) for value in obj["categories"][:]]
        codes = _take_hdf5_rows(obj["codes"], rows)
        return pd.Series(
            pd.Categorical.from_codes(codes, categories=categories), name=name
        )
    values = _take_hdf5_rows(obj, rows)
    if values.dtype.kind in {"O", "S", "U"}:
        values = np.asarray([_decode(value) for value in values], dtype=object)
    return pd.Series(values, name=name)


def read_obs(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Read selected observation columns with HDF5 opened strictly read-only."""
    row_array = None if rows is None else np.asarray(list(rows), dtype=np.int64)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        missing = sorted(set(columns) - set(obs.keys()))
        if missing:
            raise ValueError(f"H5AD observation table is missing columns: {missing}")
        return pd.DataFrame(
            {column: read_obs_column(obs, column, row_array) for column in columns}
        )


def read_csr_rows(path: Path, rows: Iterable[int]) -> sparse.csr_matrix:
    """Read arbitrary CSR rows without materializing the full cell-by-gene matrix."""
    row_array = np.asarray(list(rows), dtype=np.int64)
    if row_array.ndim != 1:
        raise ValueError("rows must be one-dimensional")
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        encoding = _decode(x.attrs.get("encoding-type", ""))
        if encoding != "csr_matrix":
            raise ValueError(f"Expected CSR .X, observed {encoding!r} in {path}")
        n_rows, n_genes = map(int, x.attrs["shape"])
        if ((row_array < 0) | (row_array >= n_rows)).any():
            raise IndexError("Requested H5AD row is out of bounds")
        if not len(row_array):
            return sparse.csr_matrix((0, n_genes), dtype=x["data"].dtype)
        # h5py and backed AnnData have version-dependent arbitrary-row indexing
        # behavior.  Read sorted unique rows as contiguous CSR storage runs, then
        # restore the requested order (and any duplicates) in memory.
        unique_rows, inverse = np.unique(row_array, return_inverse=True)
        indptr_source = x["indptr"]
        data_source = x["data"]
        indices_source = x["indices"]
        data_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        row_lengths: list[np.ndarray] = []
        run_starts = np.r_[0, np.flatnonzero(np.diff(unique_rows) > 1) + 1]
        run_stops = np.r_[run_starts[1:], len(unique_rows)]
        for first_index, stop_index in zip(run_starts, run_stops, strict=True):
            first_row = int(unique_rows[first_index])
            last_row = int(unique_rows[stop_index - 1])
            source_indptr = np.asarray(
                indptr_source[first_row : last_row + 2], dtype=np.int64
            )
            data_start, data_stop = int(source_indptr[0]), int(source_indptr[-1])
            data_parts.append(data_source[data_start:data_stop])
            index_parts.append(indices_source[data_start:data_stop])
            row_lengths.append(np.diff(source_indptr))
        data = (
            np.concatenate(data_parts)
            if data_parts
            else np.asarray([], dtype=data_source.dtype)
        )
        indices = (
            np.concatenate(index_parts)
            if index_parts
            else np.asarray([], dtype=indices_source.dtype)
        )
        lengths = np.concatenate(row_lengths)
        indptr = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
        unique_matrix = sparse.csr_matrix(
            (data, indices, indptr), shape=(len(unique_rows), n_genes)
        )
        return unique_matrix[inverse]


def read_gene_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        values = var["unified_ensembl"][:]
    return np.asarray([_decode(value) for value in values], dtype=str)


def choose_donor_balanced_rows(
    obs: pd.DataFrame,
    *,
    datasets: Iterable[str] | None = None,
    max_donors_per_dataset: int | None = None,
    max_cells_per_donor: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Select a deterministic donor-balanced development subset."""
    required = {"dataset", "donor_id"}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"Cannot select donors; missing metadata: {missing}")
    # Avoid constructing three string identifiers for every cell in a large
    # backed object. Stable IDs are added only after the small subset is chosen.
    frame = obs
    if datasets is not None:
        frame = frame.loc[frame["dataset"].astype(str).isin(set(datasets))]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, dataset_part in frame.groupby("dataset", observed=True, sort=True):
        donors = np.asarray(
            sorted(dataset_part["donor_id"].astype(str).unique()), dtype=object
        )
        if max_donors_per_dataset is not None and len(donors) > max_donors_per_dataset:
            donors = np.sort(
                rng.choice(donors, size=max_donors_per_dataset, replace=False)
            )
        for donor in donors:
            donor_rows = dataset_part.index[
                dataset_part["donor_id"].astype(str) == donor
            ].to_numpy(dtype=np.int64)
            if (
                max_cells_per_donor is not None
                and len(donor_rows) > max_cells_per_donor
            ):
                donor_rows = np.sort(
                    rng.choice(donor_rows, size=max_cells_per_donor, replace=False)
                )
            selected.extend(donor_rows.tolist())
    return np.asarray(selected, dtype=np.int64)


def load_small_lineage_subset(
    path: Path,
    *,
    datasets: Iterable[str] | None = None,
    max_donors_per_dataset: int = 3,
    max_cells_per_donor: int = 100,
    seed: int = 42,
) -> LineageData:
    """Load a donor-balanced sparse subset suitable for smoke tests."""
    columns = [
        "dataset",
        "donor_id",
        "sample_id",
        "age",
        "sex",
        "lineage",
        "ctype_low",
        "ctype_low_conf",
    ]
    full_obs = read_obs(path, columns)
    rows = choose_donor_balanced_rows(
        full_obs,
        datasets=datasets,
        max_donors_per_dataset=max_donors_per_dataset,
        max_cells_per_donor=max_cells_per_donor,
        seed=seed,
    )
    subset_obs = full_obs.iloc[rows].reset_index(drop=True)
    subset_obs = add_stable_identifiers(subset_obs)
    counts = read_csr_rows(path, rows)
    return LineageData(
        counts=counts,
        obs=subset_obs,
        gene_ids=read_gene_ids(path),
        source_path=path.resolve(),
    )


def validate_merged_h5ad(path: Path) -> dict[str, object]:
    """Validate structural properties without scanning or densifying counts."""
    with h5py.File(path, "r") as handle:
        if "X" not in handle or "obs" not in handle or "var" not in handle:
            raise ValueError(f"Not a valid merged AnnData structure: {path}")
        x = handle["X"]
        encoding = _decode(x.attrs.get("encoding-type", ""))
        if encoding != "csr_matrix":
            raise ValueError(f"Merged .X must be CSR, observed {encoding!r}")
        required_obs = {
            "dataset",
            "donor_id",
            "sample_id",
            "age",
            "sex",
            "lineage",
            "ctype_low",
            "ctype_low_conf",
        }
        missing = sorted(required_obs - set(handle["obs"].keys()))
        if missing:
            raise ValueError(f"Merged H5AD is missing obs columns: {missing}")
        return {
            "path": str(path.resolve()),
            "shape": [int(value) for value in x.attrs["shape"]],
            "x_encoding": encoding,
            "x_dtype": str(x["data"].dtype),
            "n_stored_values": int(x["data"].shape[0]),
            "layers": sorted(handle["layers"].keys()),
            "has_raw": "raw" in handle,
            "read_only": True,
        }
