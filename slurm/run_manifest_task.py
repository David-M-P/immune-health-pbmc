#!/usr/bin/env python3
"""Run one JSONL manifest row with validation and atomic restart markers."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_record(
    path: Path, *, relative_path: str | None = None
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected output is not a regular file: {path}")
    before = path.stat()
    digest = _file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Expected output changed while it was inventoried: {path}")
    record: dict[str, Any] = {
        "kind": "file",
        "size_bytes": after.st_size,
        "sha256": digest,
    }
    if relative_path is not None:
        record["relative_path"] = relative_path
    return record


def _directory_record(path: Path, *, ignored_paths: frozenset[Path]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Expected output is not a real directory: {path}")

    def listed_entries() -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise ValueError(
                    "Symlinks are forbidden inside expected output directories: "
                    f"{child}"
                )
            resolved = child.resolve()
            if resolved in ignored_paths:
                continue
            relative = child.relative_to(path).as_posix()
            if child.is_dir():
                entries.append((relative, "directory"))
            elif child.is_file():
                entries.append((relative, "file"))
            else:
                raise ValueError(f"Unsupported output tree entry: {child}")
        return entries

    initial = listed_entries()
    inventory: list[dict[str, Any]] = []
    for relative, kind in initial:
        child = path / relative
        if kind == "directory":
            inventory.append({"relative_path": relative, "kind": "directory"})
        else:
            inventory.append(_regular_file_record(child, relative_path=relative))
    if listed_entries() != initial:
        raise RuntimeError(f"Expected output tree changed during inventory: {path}")
    return {
        "kind": "directory",
        "entries": inventory,
        "tree_sha256": _canonical_hash(inventory),
        "n_files": sum(entry["kind"] == "file" for entry in inventory),
        "n_directories": sum(entry["kind"] == "directory" for entry in inventory),
        "total_file_bytes": sum(int(entry.get("size_bytes", 0)) for entry in inventory),
    }


def _expected_output_inventory(
    job: Mapping[str, Any], *, ignored_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    configured = job.get("expected_outputs")
    if (
        not isinstance(configured, list)
        or not configured
        or not all(isinstance(value, str) and value for value in configured)
    ):
        raise ValueError("Manifest expected_outputs must be a non-empty path list")
    paths = [Path(value).resolve() for value in configured]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest expected_outputs contains duplicate paths")
    ignored = frozenset(path.resolve() for path in ignored_paths)
    inventory: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required expected output is absent: {path}")
        if path.is_symlink():
            raise ValueError(f"Expected output cannot be a symlink: {path}")
        if path.is_file():
            record = _regular_file_record(path)
        elif path.is_dir():
            record = _directory_record(path, ignored_paths=ignored)
        else:
            raise ValueError(f"Unsupported expected output type: {path}")
        inventory.append({"path": str(path), **record})
    return inventory


def _validate_completion_inventory(
    done: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    ignored_paths: Sequence[Path],
) -> None:
    if done.get("schema_version") != "immune-health-slurm-completion/v2":
        raise RuntimeError(
            "Completion marker predates deterministic output inventory; refusing "
            "to trust or overwrite it"
        )
    recorded = done.get("expected_output_inventory")
    if not isinstance(recorded, list) or done.get(
        "expected_output_inventory_sha256"
    ) != _canonical_hash(recorded):
        raise RuntimeError("Completion marker output inventory is absent or invalid")
    try:
        observed = _expected_output_inventory(job, ignored_paths=ignored_paths)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Completion marker is stale because outputs cannot be verified: {exc}"
        ) from exc
    if observed != recorded:
        raise RuntimeError(
            "Completion marker is stale because expected output content changed"
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded or ("<" in expanded and ">" in expanded):
            raise ValueError(f"Unresolved compute placeholder: {expanded!r}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_job(manifest_path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("Manifest index must be non-negative")
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                job = json.loads(line)
                break
        else:
            raise IndexError(f"Manifest has no row {index}: {manifest_path}")
    if job.get("schema_version") != "immune-health-slurm-job/v1":
        raise ValueError("Unsupported manifest job schema")
    if not JOB_ID_RE.fullmatch(str(job.get("job_id", ""))):
        raise ValueError(f"Unsafe job_id: {job.get('job_id')!r}")
    return _expand(job)


def validate_upstream(job: Mapping[str, Any]) -> None:
    for artifact in job.get("upstream_artifacts", []):
        path = Path(artifact["path"])
        if not path.exists():
            raise FileNotFoundError(f"Required upstream artifact is absent: {path}")
        expected = artifact.get("sha256")
        if expected:
            if not path.is_file():
                raise ValueError(f"Cannot hash non-file upstream artifact: {path}")
            observed = _file_hash(path)
            if observed != expected:
                raise ValueError(
                    f"Upstream hash mismatch for {path}: expected {expected}, "
                    f"observed {observed}"
                )
        required_json = artifact.get("json_require")
        if required_json:
            if not path.is_file():
                raise ValueError(f"Expected a JSON upstream file: {path}")
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            for key, expected_value in required_json.items():
                if payload.get(key) != expected_value:
                    raise ValueError(
                        f"Upstream JSON validation failed for {path}: {key} must "
                        f"equal {expected_value!r}, observed {payload.get(key)!r}"
                    )


def _optional_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "returncode": completed.returncode,
        "output": completed.stdout[-12000:],
    }


def resource_log(phase: str) -> dict[str, Any]:
    packages = {}
    for name in ("tripso", "torch", "pytorch-lightning", "numpy", "anndata"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    report = {
        "phase": phase,
        "timestamp_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "maximum_resident_set_size_kb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_GPUS",
            )
        },
        "gpu": _optional_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
    }
    if slurm_job_id:
        report["sstat"] = _optional_command(
            [
                "sstat",
                "--noheader",
                "--parsable2",
                "--jobs",
                slurm_job_id,
                "--format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU,AveDiskRead,AveDiskWrite",
            ]
        )
    return report


def run_job(job: Mapping[str, Any], *, dry_run: bool = False) -> int:
    if not job.get("runnable"):
        reason = job.get("pending_reason") or "an unresolved prerequisite"
        raise ValueError(f"Job {job['job_id']} is non-runnable: {reason}")
    command = job.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ValueError("Manifest command must be a non-empty argv list")

    output_dir = Path(job["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _canonical_hash(job)
    done_path = output_dir / ".done.json"
    failure_path = output_dir / ".failed.json"
    lock_path = output_dir / ".task.lock"
    job_spec_path = Path(job["job_spec_path"])
    control_paths = (
        done_path,
        failure_path,
        lock_path,
        job_spec_path,
        output_dir / "resources.before.json",
        output_dir / "resources.after.json",
    )

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Job output is already active: {output_dir}") from exc

        if done_path.exists():
            with done_path.open(encoding="utf-8") as handle:
                done = json.load(handle)
            if done.get("job_fingerprint") != fingerprint:
                raise RuntimeError(
                    "A completion marker exists for different job content; refusing "
                    f"to overwrite {output_dir}"
                )
            _validate_completion_inventory(
                done,
                job,
                ignored_paths=control_paths,
            )
            print(f"Already complete; skipping {job['job_id']}")
            return 0

        validate_upstream(job)
        _atomic_json(job_spec_path, job)
        _atomic_json(output_dir / "resources.before.json", resource_log("before"))
        if dry_run:
            print(json.dumps({"job_id": job["job_id"], "command": command}, indent=2))
            return 0

        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(job["seed"])
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Production training replaces the vendor W&B boundary with a local
        # Lightning CSV logger.  Force-disable W&B as a second no-network guard,
        # even if the submission shell inherited online credentials/settings.
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        started = _utc_now()
        try:
            completed = subprocess.run(
                command,
                cwd=job.get("working_directory") or None,
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            _atomic_json(output_dir / "resources.after.json", resource_log("after"))
            output_inventory = _expected_output_inventory(
                job,
                ignored_paths=control_paths,
            )
            _atomic_json(
                done_path,
                {
                    "schema_version": "immune-health-slurm-completion/v2",
                    "job_id": job["job_id"],
                    "job_fingerprint": fingerprint,
                    "started_utc": started,
                    "completed_utc": _utc_now(),
                    "returncode": 0,
                    "expected_output_inventory": output_inventory,
                    "expected_output_inventory_sha256": _canonical_hash(
                        output_inventory
                    ),
                },
            )
            return 0
        except BaseException as exc:
            _atomic_json(
                failure_path,
                {
                    "schema_version": "immune-health-slurm-failure/v1",
                    "job_id": job["job_id"],
                    "job_fingerprint": fingerprint,
                    "started_utc": started,
                    "failed_utc": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )[-20000:],
                    "resources": resource_log("failure"),
                },
            )
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = load_job(args.manifest, args.index)
    return run_job(job, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
