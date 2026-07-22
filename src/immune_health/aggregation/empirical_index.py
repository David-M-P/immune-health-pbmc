"""Zero-copy source binding for donor/fine-type empirical distributions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from immune_health.provenance import atomic_write_json, sha256_file, stable_hash
from immune_health.tripso_adapter.arrow_bridge import (
    validate_arrow_conversion_for_aggregation,
)

from .summarize import DistributionKey

EMPIRICAL_INDEX_SCHEMA = "immune-health-empirical-row-index/v1"
EMPIRICAL_GROUPS = "empirical_distribution_groups.parquet"
EMPIRICAL_ROWS = "empirical_distribution_rows.npy"
EMPIRICAL_MANIFEST = "empirical_distribution_manifest.json"


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_empirical_row_index(
    output_dir: Path,
    metadata: pd.DataFrame,
    distribution_rows: Mapping[DistributionKey, np.ndarray],
    empirical_distance_keys: Sequence[DistributionKey] | frozenset[DistributionKey],
    *,
    conversion_validation: Mapping[str, Any],
    aggregation_table_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Index source NPY rows without serializing the embeddings a second time."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / EMPIRICAL_GROUPS
    rows_path = output_dir / EMPIRICAL_ROWS
    manifest_path = output_dir / EMPIRICAL_MANIFEST
    existing = [
        path for path in (groups_path, rows_path, manifest_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite empirical row index: {existing}")
    if "embedding_row" not in metadata:
        raise ValueError("Converted metadata lacks embedding_row for zero-copy index")
    embedding_rows = pd.to_numeric(metadata["embedding_row"], errors="coerce")
    if embedding_rows.isna().any() or embedding_rows.duplicated().any():
        raise ValueError("embedding_row must be unique finite integers")
    embedding_rows_array = embedding_rows.to_numpy(dtype=np.int64)
    if not np.array_equal(embedding_rows_array, embedding_rows.to_numpy(dtype=float)):
        raise ValueError("embedding_row contains non-integer values")

    eligible = set(empirical_distance_keys)
    flat_parts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    cursor = 0
    for key, positions in sorted(distribution_rows.items()):
        local_positions = np.asarray(positions, dtype=np.int64)
        if local_positions.ndim != 1 or (
            len(local_positions)
            and (local_positions.min() < 0 or local_positions.max() >= len(metadata))
        ):
            raise ValueError(f"Invalid metadata positions for empirical group {key}")
        selected_metadata = metadata.iloc[local_positions]
        for column, expected in zip(
            ("observation_id", "lineage", "fine_type"), key[:3], strict=True
        ):
            if (
                column not in selected_metadata
                or not selected_metadata[column].astype(str).eq(str(expected)).all()
            ):
                raise ValueError(
                    f"Empirical group {key} points to rows with a different {column}"
                )
        source_rows = np.ascontiguousarray(
            embedding_rows_array[local_positions], dtype=np.int64
        )
        start = cursor
        cursor += len(source_rows)
        flat_parts.append(source_rows)
        records.append(
            {
                "observation_id": str(key[0]),
                "lineage": str(key[1]),
                "fine_type": str(key[2]),
                "gp_id": str(key[3]),
                "start": start,
                "stop": cursor,
                "n_rows": len(source_rows),
                "empirical_distance_eligible": key in eligible,
            }
        )
    rows = (
        np.concatenate(flat_parts).astype(np.int64, copy=False)
        if flat_parts
        else np.empty(0, dtype=np.int64)
    )
    if len(rows) and len(np.unique(rows)) != len(rows):
        raise ValueError(
            "One source embedding row appears in multiple empirical groups"
        )
    source_n_rows = int(conversion_validation["n_rows"])
    if len(rows) and (rows.min() < 0 or rows.max() >= source_n_rows):
        raise ValueError("Empirical index points outside the converted embedding NPY")
    groups = pd.DataFrame.from_records(
        records,
        columns=(
            "observation_id",
            "lineage",
            "fine_type",
            "gp_id",
            "start",
            "stop",
            "n_rows",
            "empirical_distance_eligible",
        ),
    )
    _atomic_parquet(groups_path, groups)
    _atomic_npy(rows_path, rows)
    payload: dict[str, Any] = {
        "schema_version": EMPIRICAL_INDEX_SCHEMA,
        "storage_mode": "mmap_source_npy_plus_int64_row_gather",
        "source_embeddings_path": conversion_validation["embedding_path"],
        "source_embeddings_shape": conversion_validation["shape"],
        "source_embeddings_dtype": conversion_validation["dtype"],
        "source_embeddings_float32_payload_sha256": conversion_validation[
            "float32_payload_sha256"
        ],
        "source_metadata_path": conversion_validation["metadata_path"],
        "source_metadata_sha256": conversion_validation["metadata_sha256"],
        "source_cell_key_ordered_sha256": conversion_validation[
            "cell_key_ordered_sha256"
        ],
        "embedding_column": conversion_validation["embedding_column"],
        "arrow_conversion_manifest": conversion_validation["manifest_path"],
        "arrow_conversion_manifest_sha256": conversion_validation["manifest_sha256"],
        "projection_output_manifest_sha256": conversion_validation["projection_output"][
            "manifest_sha256"
        ],
        "groups_path": groups_path.name,
        "groups_sha256": sha256_file(groups_path),
        "rows_path": rows_path.name,
        "rows_sha256": sha256_file(rows_path),
        "rows_dtype": "int64",
        "n_indexed_rows": len(rows),
        "n_groups": len(groups),
        "n_empirical_distance_eligible_groups": int(
            groups.get("empirical_distance_eligible", pd.Series(dtype=bool)).sum()
        ),
        "copied_embedding_values": False,
    }
    if aggregation_table_path is not None:
        aggregation_table_path = Path(aggregation_table_path).resolve()
        if not aggregation_table_path.is_file():
            raise FileNotFoundError(
                f"Aggregation table is missing: {aggregation_table_path}"
            )
        payload["aggregation_table_path"] = str(aggregation_table_path)
        payload["aggregation_table_sha256"] = sha256_file(aggregation_table_path)
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return payload


@dataclass(frozen=True)
class EmpiricalDistributionStore:
    """Validated mmap source plus compact group-to-source-row lookup."""

    embeddings: np.ndarray
    embedding_rows: np.ndarray
    groups: pd.DataFrame
    manifest: Mapping[str, Any]

    def get(self, key: DistributionKey) -> np.ndarray:
        selected = self.groups.loc[
            self.groups["observation_id"].astype(str).eq(str(key[0]))
            & self.groups["lineage"].astype(str).eq(str(key[1]))
            & self.groups["fine_type"].astype(str).eq(str(key[2]))
            & self.groups["gp_id"].astype(str).eq(str(key[3]))
        ]
        if len(selected) != 1:
            raise KeyError(f"Empirical distribution key is absent or duplicated: {key}")
        row = selected.iloc[0]
        source_rows = self.embedding_rows[int(row["start"]) : int(row["stop"])]
        return np.asarray(self.embeddings[source_rows], dtype=np.float32)


def load_empirical_distribution_store(
    manifest_path: Path,
) -> EmpiricalDistributionStore:
    """Validate the full source chain, mmap it, and expose group-level gathers."""

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Empirical index manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != EMPIRICAL_INDEX_SCHEMA:
        raise ValueError(f"Unsupported empirical index manifest: {manifest_path}")
    content = dict(manifest)
    claimed_hash = content.pop("manifest_sha256", None)
    if claimed_hash != stable_hash(content):
        raise ValueError("Empirical index manifest content hash does not match")
    conversion = validate_arrow_conversion_for_aggregation(
        Path(str(manifest["arrow_conversion_manifest"])),
        Path(str(manifest["source_embeddings_path"])),
        Path(str(manifest["source_metadata_path"])),
        embedding_column=str(manifest["embedding_column"]),
    )
    comparisons = {
        "arrow_conversion_manifest_sha256": conversion["manifest_sha256"],
        "source_embeddings_shape": conversion["shape"],
        "source_embeddings_dtype": conversion["dtype"],
        "source_embeddings_float32_payload_sha256": conversion[
            "float32_payload_sha256"
        ],
        "source_metadata_sha256": conversion["metadata_sha256"],
        "source_cell_key_ordered_sha256": conversion["cell_key_ordered_sha256"],
        "projection_output_manifest_sha256": conversion["projection_output"][
            "manifest_sha256"
        ],
    }
    mismatches = {
        name: {"expected": manifest.get(name), "observed": value}
        for name, value in comparisons.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Empirical index source chain differs: {mismatches}")
    aggregation_path = manifest.get("aggregation_table_path")
    aggregation_hash = manifest.get("aggregation_table_sha256")
    if (aggregation_path is None) != (aggregation_hash is None):
        raise ValueError("Empirical index aggregation-table binding is incomplete")
    if (
        aggregation_path is not None
        and sha256_file(Path(str(aggregation_path))) != aggregation_hash
    ):
        raise ValueError("Empirical index aggregation table SHA-256 does not match")
    groups_path = manifest_path.parent / str(manifest["groups_path"])
    rows_path = manifest_path.parent / str(manifest["rows_path"])
    if sha256_file(groups_path) != manifest.get("groups_sha256"):
        raise ValueError("Empirical group table SHA-256 does not match")
    if sha256_file(rows_path) != manifest.get("rows_sha256"):
        raise ValueError("Empirical row index SHA-256 does not match")
    groups = pd.read_parquet(groups_path)
    required = {
        "observation_id",
        "lineage",
        "fine_type",
        "gp_id",
        "start",
        "stop",
        "n_rows",
        "empirical_distance_eligible",
    }
    if required - set(groups.columns):
        raise ValueError("Empirical group table schema is incomplete")
    rows = np.load(rows_path, mmap_mode="r", allow_pickle=False)
    embeddings = np.load(
        Path(str(manifest["source_embeddings_path"])),
        mmap_mode="r",
        allow_pickle=False,
    )
    if rows.dtype != np.dtype("int64") or rows.ndim != 1:
        raise ValueError("Empirical embedding rows must be a one-dimensional int64 NPY")
    starts = pd.to_numeric(groups["start"], errors="coerce").to_numpy(dtype=np.int64)
    stops = pd.to_numeric(groups["stop"], errors="coerce").to_numpy(dtype=np.int64)
    expected_starts = np.r_[0, stops[:-1]] if len(stops) else np.empty(0, dtype=int)
    if (
        len(groups) != int(manifest["n_groups"])
        or len(rows) != int(manifest["n_indexed_rows"])
        or not np.array_equal(starts, expected_starts)
        or (len(stops) and stops[-1] != len(rows))
        or not np.array_equal(
            stops - starts,
            pd.to_numeric(groups["n_rows"], errors="coerce").to_numpy(dtype=np.int64),
        )
    ):
        raise ValueError("Empirical group offsets are not contiguous and aligned")
    if len(rows) and (rows.min() < 0 or rows.max() >= len(embeddings)):
        raise ValueError("Empirical group index points outside source embeddings")
    return EmpiricalDistributionStore(embeddings, rows, groups, manifest)
