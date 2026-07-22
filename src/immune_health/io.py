"""Compact artifact I/O with adjacent, trackable provenance manifests."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from immune_health.provenance import atomic_write_json, stable_hash


def artifact_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def write_parquet_artifact(
    frame: pd.DataFrame,
    path: Path,
    *,
    schema_name: str,
    schema_version: str,
    provenance: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write Parquet plus a small JSON manifest."""
    manifest_path = artifact_manifest_path(path)
    if (path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact without --overwrite: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(path),
        "format": "parquet",
        "schema_name": schema_name,
        "schema_version": schema_version,
        "n_rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "provenance": provenance,
        "provenance_hash": stable_hash(provenance),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def read_parquet_artifact(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = artifact_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Artifact or provenance manifest is missing for {path}"
        )
    import json

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Artifact manifest is not complete: {manifest_path}")
    frame = pd.read_parquet(path)
    if len(frame) != manifest["n_rows"]:
        raise ValueError(f"Artifact row count differs from manifest: {path}")
    return frame, manifest
