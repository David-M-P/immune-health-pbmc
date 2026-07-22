"""Small, trackable provenance manifests and atomic completion markers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def software_environment(packages: Iterable[str] = ()) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def completion_marker(
    path: Path,
    *,
    stage: str,
    outputs: Iterable[Path],
    configuration: dict[str, Any],
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_records = []
    for output in outputs:
        if not output.exists():
            raise FileNotFoundError(f"Cannot mark incomplete stage; missing {output}")
        output_records.append(
            {"path": str(output), "size_bytes": output.stat().st_size}
        )
    payload = {
        "status": "complete",
        "stage": stage,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(repo_root),
        "configuration_hash": stable_hash(configuration),
        "outputs": output_records,
        "environment": software_environment(
            ["numpy", "pandas", "scipy", "scikit-learn", "anndata", "h5py"]
        ),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)
    return payload
