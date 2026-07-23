#!/usr/bin/env python3
"""Combine reviewed JSONL job manifests without changing their row semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

JOB_SCHEMA = "immune-health-slurm-job/v1"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Input manifest must be a regular file: {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Input manifest is empty: {source}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(
                "Blank manifest rows are unsupported because row indices are exact: "
                f"{source}:{line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict) or row.get("schema_version") != JOB_SCHEMA:
            raise ValueError(
                f"Unsupported job manifest row at {source}:{line_number}; "
                f"expected schema_version={JOB_SCHEMA!r}"
            )
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
            raise ValueError(f"Unsafe job_id at {source}:{line_number}: {job_id!r}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def combine_job_manifests(
    inputs: Sequence[Path], output: Path
) -> dict[str, Any]:
    """Validate and atomically concatenate manifests in the supplied order."""

    if not inputs:
        raise ValueError("At least one --input manifest is required")
    sources = tuple(Path(path) for path in inputs)
    resolved_sources = tuple(source.resolve() for source in sources)
    if len(resolved_sources) != len(set(resolved_sources)):
        raise ValueError("The same input manifest cannot be supplied more than once")
    destination = Path(output)
    if destination.is_symlink():
        raise ValueError("Output manifest cannot be a symlink")
    if destination.resolve() in resolved_sources:
        raise ValueError("Output manifest must be different from every input manifest")

    combined: list[dict[str, Any]] = []
    seen_job_ids: dict[str, Path] = {}
    input_records: list[dict[str, Any]] = []
    for source, resolved_source in zip(sources, resolved_sources, strict=True):
        rows = _load_rows(source)
        for row in rows:
            job_id = str(row["job_id"])
            previous = seen_job_ids.get(job_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate job_id {job_id!r} in {resolved_source}; "
                    f"first seen in {previous}"
                )
            seen_job_ids[job_id] = resolved_source
            combined.append(row)
        input_records.append(
            {
                "path": str(resolved_source),
                "row_count": len(rows),
                "sha256": _file_sha256(source),
            }
        )

    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in combined
    )
    _atomic_write(destination, content)
    return {
        "schema_version": "immune-health-combined-job-manifest/v1",
        "inputs": input_records,
        "output": str(destination.absolute()),
        "output_row_count": len(combined),
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="inputs",
        help="Input JSONL manifest; repeat to preserve and combine this order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = combine_job_manifests(args.inputs, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
