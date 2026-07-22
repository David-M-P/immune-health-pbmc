"""Reproducible provenance for TRIPSO model and checkpoint artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import atomic_write_json, canonical_json_hash, sha256_path

# Imported upstream revision documented in docs/TRIPSO_PROVENANCE.md. The parent
# repository tree hash and dirty flag below additionally reveal any local change.
VENDORED_TRIPSO_UPSTREAM_COMMIT = "5d19c88081b1a0c497fb6dc4637df063e7782a3a"


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def repository_provenance(repo_root: Path, vendor_root: Path) -> dict[str, Any]:
    """Record repository commit and the exact tracked vendored source tree."""
    repo_root = Path(repo_root).resolve()
    vendor_root = Path(vendor_root).resolve()
    try:
        vendor_relative = vendor_root.relative_to(repo_root).as_posix()
    except ValueError:
        vendor_relative = str(vendor_root)

    vendor_nested_commit = _git(vendor_root, "rev-parse", "HEAD")
    vendor_tree = _git(repo_root, "rev-parse", f"HEAD:{vendor_relative}")
    vendor_introducing_commit = _git(
        repo_root, "log", "-1", "--format=%H", "--", vendor_relative
    )
    return {
        "repository_commit": _git(repo_root, "rev-parse", "HEAD"),
        "repository_dirty": bool(_git(repo_root, "status", "--porcelain")),
        "vendor_path": vendor_relative,
        # A nested commit is used only when vendor_root really is a repository.
        "vendor_nested_commit": vendor_nested_commit
        if (vendor_root / ".git").exists()
        else None,
        "vendor_upstream_commit": VENDORED_TRIPSO_UPSTREAM_COMMIT,
        "vendor_upstream_commit_source": "docs/TRIPSO_PROVENANCE.md",
        "vendor_source_commit": vendor_introducing_commit,
        "vendor_tree_hash": vendor_tree,
    }


def software_environment(distributions: Sequence[str] | None = None) -> dict[str, Any]:
    """Capture interpreter/platform and installed package versions without imports."""
    names = distributions or (
        "anndata",
        "datasets",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scanpy",
        "scipy",
        "torch",
        "transformers",
        "tripso",
        "wandb",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def build_model_artifact_manifest(
    *,
    output_path: Path,
    repo_root: Path,
    vendor_root: Path,
    fold_input_manifest_path: Path,
    checkpoint_path: Path,
    fold_id: str,
    held_out_dataset: str | None,
    lineage: str,
    model_type: str,
    sampler_mode: str,
    alpha: float | None,
    fine_type_lambda: float | None,
    seed: int,
    gp_library_path: Path,
    gene_vocabulary_path: Path,
    training_metrics: Mapping[str, Any] | None = None,
    model_configuration: Mapping[str, Any] | None = None,
    asset_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write the immutable provenance required for every trained artifact."""
    fold_input_manifest_path = Path(fold_input_manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    for name, path in {
        "fold input manifest": fold_input_manifest_path,
        "checkpoint": checkpoint_path,
        "GP library": Path(gp_library_path),
        "gene vocabulary": Path(gene_vocabulary_path),
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot record missing {name}: {path}")

    with fold_input_manifest_path.open(encoding="utf-8") as handle:
        fold_manifest = json.load(handle)
    fold_hashes = fold_manifest.get("hashes", {})
    projection_candidate_hashes = {
        key: fold_hashes[key]
        for key in (
            "projection_gp_candidates_sha256",
            "projection_gp_program_ids_ordered_sha256",
        )
        if fold_hashes.get(key)
    }
    payload: dict[str, Any] = {
        "schema_version": "immune-health-tripso-model/v1",
        "repository": repository_provenance(repo_root, vendor_root),
        "fold_id": fold_id,
        "reference_design": fold_manifest.get("reference_design", "lodo"),
        "held_out_dataset": held_out_dataset,
        "lineage": lineage,
        "model_type": model_type,
        "sampler": {
            "mode": sampler_mode,
            "dataset_alpha": None if alpha is None else float(alpha),
            "fine_type_lambda": (
                None if fine_type_lambda is None else float(fine_type_lambda)
            ),
        },
        "inner_model_selection": dict(fold_manifest.get("inner_model_selection", {})),
        "seed": int(seed),
        "hashes": {
            "gp_library_sha256": sha256_path(Path(gp_library_path)),
            "gene_vocabulary_sha256": sha256_path(Path(gene_vocabulary_path)),
            "input_manifest_sha256": sha256_path(fold_input_manifest_path),
            "checkpoint_sha256": sha256_path(checkpoint_path),
            # Copy the training-only GP candidate binding into the model
            # artifact itself.  The fold manifest hash already commits to
            # these values, but keeping them explicit makes projection
            # validation and artifact auditing fail closed without inference.
            **projection_candidate_hashes,
            **dict(asset_hashes or {}),
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "fold_input_manifest": str(fold_input_manifest_path),
            "projection_gp_candidates": fold_manifest.get("inputs", {}).get(
                "projection_gp_candidates_path"
            ),
        },
        "software_environment": software_environment(),
        "model_configuration": dict(model_configuration or {}),
        "training_metrics": dict(training_metrics or {}),
        "fold_input_manifest_content_hash": canonical_json_hash(fold_manifest),
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    atomic_write_json(output_path, payload)
    return payload


def validate_checkpoint_manifest(
    manifest_path: Path,
    *,
    expected_fold_id: str | None = None,
    expected_held_out_dataset: str | None = None,
    expected_lineage: str | None = None,
    verify_checkpoint_hash: bool = True,
) -> dict[str, Any]:
    """Validate a checkpoint's provenance and immutable fold identity."""
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "immune-health-tripso-model/v1":
        raise ValueError(f"Unsupported TRIPSO model manifest: {manifest_path}")
    expected_values = {
        "fold_id": expected_fold_id,
        "held_out_dataset": expected_held_out_dataset,
        "lineage": expected_lineage,
    }
    for key, expected in expected_values.items():
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {expected!r}, "
                f"observed {manifest.get(key)!r}"
            )
    content = dict(manifest)
    claimed_manifest_hash = content.pop("manifest_sha256", None)
    if claimed_manifest_hash != canonical_json_hash(content):
        raise ValueError(f"Model manifest content hash is invalid: {manifest_path}")
    checkpoint = Path(manifest["paths"]["checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {checkpoint}")
    if verify_checkpoint_hash:
        observed = sha256_path(checkpoint)
        expected = manifest["hashes"]["checkpoint_sha256"]
        if observed != expected:
            raise ValueError(
                f"Checkpoint hash mismatch: expected {expected}, observed {observed}"
            )
    return manifest
