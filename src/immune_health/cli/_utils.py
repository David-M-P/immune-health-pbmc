"""Small file/configuration helpers shared by CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.config import load_path_config, resolve_config_path

IDENTIFIER_CONTRACT = {
    "biological_unit_id": "dataset::donor_id",
    "source_observation_id": "dataset::sample_id",
    "observation_id": "dataset::donor_id::sample_id",
}


def load_config(path: Path | None) -> dict[str, Any]:
    """Load an optional YAML config and reject conflicting identifier settings."""

    if path is None:
        return {}
    config = load_path_config(path)
    identifiers = config.get("identifiers")
    if isinstance(identifiers, Mapping):
        conflicts = {
            key: identifiers.get(key)
            for key, expected in IDENTIFIER_CONTRACT.items()
            if key in identifiers and identifiers.get(key) != expected
        }
        if conflicts:
            raise ValueError(
                "Configuration conflicts with the approved identifier contract: "
                f"{conflicts}"
            )
    return config


def config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to its declaring config file."""

    return resolve_config_path(config, value)


def read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a supported rectangular table without format guessing by content."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Table does not exist: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".parquet" in suffixes:
        if nrows is not None:
            # Parquet metadata reads are already columnar; return an empty schema.
            return pd.read_parquet(path).head(nrows)
        return pd.read_parquet(path)
    if ".tsv" in suffixes or ".txt" in suffixes:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if ".csv" in suffixes:
        return pd.read_csv(path, nrows=nrows)
    raise ValueError(
        f"Unsupported table format for {path}; use Parquet, TSV[.gz], or CSV[.gz]"
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a table using its explicit extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".parquet" in suffixes:
        frame.to_parquet(path, index=False)
    elif ".tsv" in suffixes or ".txt" in suffixes:
        frame.to_csv(path, sep="\t", index=False)
    elif ".csv" in suffixes:
        frame.to_csv(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output table format for {path}; use Parquet, TSV, or CSV"
        )


def load_matrix(path: Path, *, require_sparse: bool = False) -> Any:
    """Load a NumPy array or SciPy sparse NPZ without implicit densification."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Matrix does not exist: {path}")
    if path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        try:
            matrix = sparse.load_npz(path)
        except (KeyError, ValueError):
            archive = np.load(path, allow_pickle=False)
            if len(archive.files) != 1:
                raise ValueError(
                    f"Dense NPZ {path} must contain exactly one named array"
                )
            matrix = archive[archive.files[0]]
    else:
        raise ValueError(f"Unsupported matrix format for {path}; use .npy or .npz")
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError(f"Matrix must be two-dimensional: {path}")
    if require_sparse and not sparse.issparse(matrix):
        raise TypeError(
            f"Raw cell-count matrix must be a SciPy sparse .npz, observed {path}"
        )
    return matrix


def read_gene_ids(path: Path) -> tuple[str, ...]:
    """Read an ordered one-column vocabulary from text, NumPy, or a table."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gene vocabulary does not exist: {path}")
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False).ravel().astype(str).tolist()
    elif path.suffix.lower() in {".txt", ".list"}:
        values = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        table = read_table(path)
        if table.shape[1] != 1:
            preferred = next(
                (
                    column
                    for column in ("gene", "gene_id", "ensembl_id", "symbol")
                    if column in table
                ),
                None,
            )
            if preferred is None:
                raise ValueError(
                    f"Gene table {path} needs one column or a recognized gene column"
                )
            values = table[preferred].dropna().astype(str).tolist()
        else:
            values = table.iloc[:, 0].dropna().astype(str).tolist()
    genes = tuple(value.strip() for value in values if value.strip())
    if not genes:
        raise ValueError(f"Gene vocabulary is empty: {path}")
    if len(set(genes)) != len(genes):
        raise ValueError(f"Gene vocabulary contains duplicates: {path}")
    return genes


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return value


def expand_path(path: str | Path) -> Path:
    """Expand environment variables and reject unresolved compute placeholders."""

    expanded = os.path.expandvars(str(path))
    if "${" in expanded or ("<" in expanded and ">" in expanded):
        raise ValueError(f"Unresolved path placeholder: {expanded!r}")
    return Path(expanded).expanduser()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def guard_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(existing)
        )


def json_plan(command: str, **details: Any) -> str:
    """Create a stable, human- and machine-readable dry-run plan."""

    return json.dumps(
        {
            "command": command,
            "dry_run": True,
            "identifier_contract": IDENTIFIER_CONTRACT,
            **details,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )
