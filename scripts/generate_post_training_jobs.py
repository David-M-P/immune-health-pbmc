#!/usr/bin/env python3
"""Generate, but never submit, model-dependent frozen-projection jobs.

Each runnable Stage-1/2 row receives exact adaptation/reference and fixed
inner-validation projections. Outer-query projection is evaluation-only and is
generated only behind an explicit, hashed selected-job allowlist. Stage 3 receives
reference projection only. Binding and GPU projection remain restartable phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

JOB_SCHEMA = "immune-health-slurm-job/v1"
FOLD_SCHEMA = "immune-health-tripso-fold-input/v1"
MODEL_SCHEMA = "immune-health-tripso-model/v1"
TOKENIZATION_SCHEMA = "immune-health-tripso-tokenization/v1"
PROJECTION_SCHEMA = "immune-health-tripso-projection-input/v1"
DEFAULT_MAX_PROJECTED_BYTES = 250 * 1024**3
OUTER_QUERY_ALLOWLIST_SCHEMA = "immune-health-outer-query-evaluation-allowlist/v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_outer_query_allowlist(path: Path) -> tuple[set[str], dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != OUTER_QUERY_ALLOWLIST_SCHEMA:
        raise ValueError(f"Unsupported outer-query allowlist schema: {path}")
    content = dict(payload)
    claimed_hash = content.pop("manifest_sha256", None)
    if claimed_hash != _canonical_hash(content):
        raise ValueError("Outer-query selected-job allowlist self-hash is invalid")
    if payload.get("selection_basis") != "inner_validation_only":
        raise ValueError(
            "Outer-query allowlist selection_basis must be inner_validation_only"
        )
    if payload.get("outer_query_data_consulted_for_selection") is not False:
        raise ValueError("Outer-query allowlist must attest zero query consultation")
    raw_ids = payload.get("selected_training_job_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(not isinstance(value, str) or not value.strip() for value in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
    ):
        raise ValueError(
            "Outer-query allowlist needs unique nonempty selected_training_job_ids"
        )
    return set(raw_ids), payload


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != JOB_SCHEMA:
                raise ValueError(f"Unsupported job schema at {path}:{line_number}")
            rows.append(row)
    return rows


def _one_path(
    values: Iterable[Mapping[str, Any]],
    *,
    predicate: Any,
    description: str,
) -> str:
    paths = [str(value["path"]) for value in values if predicate(value)]
    if len(paths) != 1:
        raise ValueError(f"Training job must identify one {description}; got {paths}")
    return paths[0]


def _training_paths(job: Mapping[str, Any]) -> dict[str, str]:
    upstream = job.get("upstream_artifacts", [])
    fold_input = _one_path(
        upstream,
        predicate=lambda value: value.get("json_require", {}).get("schema_version")
        == FOLD_SCHEMA,
        description="fold-input manifest",
    )
    model_paths = [
        str(path)
        for path in job.get("expected_outputs", [])
        if str(path).endswith("/model_manifest.json")
    ]
    if len(model_paths) != 1:
        raise ValueError("Training job must expect exactly one model_manifest.json")
    fold_parent = str(PurePosixPath(fold_input).parent)
    return {
        "fold_input": fold_input,
        "model_manifest": model_paths[0],
        "reference_tokenization": (
            f"{fold_parent}/adaptation/tokenization_manifest.json"
        ),
        "validation_tokenization": (
            f"{fold_parent}/validation/tokenization_manifest.json"
        ),
        "query_tokenization": f"{fold_parent}/query/tokenization_manifest.json",
    }


def _job(
    *,
    parent: Mapping[str, Any],
    phase: str,
    role: str,
    runner_dir: str,
    command: Sequence[str],
    upstream: Sequence[Mapping[str, Any]],
    expected_outputs: Sequence[str],
    projection_input: str,
    projection_data_dir: str,
) -> dict[str, Any]:
    parent_id = str(parent["job_id"])
    job_id = f"posttrain-{phase}-{role}-{parent_id}"
    return {
        "schema_version": JOB_SCHEMA,
        "job_id": job_id,
        "stage": f"posttrain_{phase}_{role}",
        "parent_training_stage": parent.get("stage"),
        "parent_training_job_id": parent_id,
        "reference_design": (
            "all_healthy" if parent.get("design") == "final_all_healthy" else "lodo"
        ),
        "lineage": parent.get("lineage"),
        "heldout_dataset": parent.get("heldout_dataset"),
        "hvg_size": parent.get("hvg_size"),
        "sampler_mode": parent.get("sampler_mode"),
        "seed": int(parent["seed"]),
        "projection_role": role,
        "adapt": False,
        "optimizer_allowed": False,
        "all_tokenized_cells_required": True,
        "runnable": True,
        "output_dir": runner_dir,
        "projection_input": projection_input,
        "projection_data_dir": projection_data_dir,
        "runner_output_separate_from_projection_data": runner_dir
        != projection_data_dir,
        "working_directory": "${PROJECT_ROOT}",
        "job_spec_path": f"{runner_dir}/job_spec.json",
        "upstream_artifacts": list(upstream),
        "expected_outputs": list(expected_outputs),
        "command": list(command),
    }


def generate_post_training_jobs(
    training_jobs: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 128,
    precision: str = "32",
    max_projected_bytes: int = DEFAULT_MAX_PROJECTED_BYTES,
    allow_all_gps: bool = False,
    allow_oversized_projection: bool = False,
    enable_outer_query_evaluation: bool = False,
    outer_query_selected_job_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Expand every runnable training row into role-aware post-training jobs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(max_projected_bytes, bool) or max_projected_bytes < 1:
        raise ValueError("max_projected_bytes must be a positive integer")
    selected_query_ids = set(outer_query_selected_job_ids or ())
    if enable_outer_query_evaluation != bool(selected_query_ids):
        raise ValueError(
            "Outer-query evaluation requires both the explicit evaluation flag and "
            "a nonempty selected-job allowlist"
        )
    runnable_lodo_ids = {
        str(parent.get("job_id", ""))
        for parent in training_jobs
        if parent.get("runnable", False)
        and parent.get("stage") in {"stage1", "stage2"}
        and parent.get("design") == "lodo"
    }
    unknown_query_ids = sorted(selected_query_ids - runnable_lodo_ids)
    if unknown_query_ids:
        raise ValueError(
            "Outer-query allowlist contains non-runnable/non-LODO job IDs: "
            f"{unknown_query_ids[:10]}"
        )
    output = {
        "bind_reference": [],
        "bind_validation": [],
        "bind_query": [],
        "project_reference": [],
        "project_validation": [],
        "project_query": [],
    }
    seen: set[str] = set()
    for parent in training_jobs:
        if not parent.get("runnable", False):
            continue
        if parent.get("stage") not in {"stage1", "stage2", "stage3"}:
            raise ValueError(
                f"Unsupported parent training stage: {parent.get('stage')}"
            )
        parent_id = str(parent.get("job_id", ""))
        if not parent_id or parent_id in seen:
            raise ValueError(
                f"Duplicate or empty parent training job_id: {parent_id!r}"
            )
        seen.add(parent_id)
        paths = _training_paths(parent)
        base = f"{parent['output_dir']}/post_training"
        roles = ["reference"]
        if parent.get("design") == "lodo":
            if not parent.get("heldout_dataset"):
                raise ValueError("Runnable LODO training row lacks heldout_dataset")
            if (
                parent.get("model_selection_role") != "validation"
                or parent.get("inner_validation_fold") != 0
                or parent.get("outer_query_evaluation_only") is not True
                or parent.get("outer_query_used_for_selection") is not False
            ):
                raise ValueError(
                    "LODO parent does not prove fixed inner-validation selection "
                    "with a sealed outer query"
                )
            if parent.get("stage") not in {"stage1", "stage2"}:
                raise ValueError("LODO training is supported only in Stage 1/2")
            roles.append("validation")
            if parent_id in selected_query_ids:
                roles.append("query")
        elif parent.get("design") != "final_all_healthy":
            raise ValueError(f"Unsupported training design: {parent.get('design')}")

        for role in roles:
            tokenization = paths[f"{role}_tokenization"]
            projection_input = f"{base}/inputs/{role}_projection_input.json"
            projection_data_dir = f"{base}/projection_data/{role}"
            bind_runner = f"{base}/runner/bind_{role}"
            project_runner = f"{base}/runner/project_{role}"
            bind = _job(
                parent=parent,
                phase="bind",
                role=role,
                runner_dir=bind_runner,
                command=[
                    "python",
                    "-m",
                    "immune_health.cli.tokenize_tripso",
                    "build-projection-input",
                    "--role",
                    role,
                    "--tokenization-manifest",
                    tokenization,
                    "--model-manifest",
                    paths["model_manifest"],
                    "--output",
                    projection_input,
                    (
                        "--allow-all-gps"
                        if allow_all_gps
                        else "--use-fold-bound-gp-candidates"
                    ),
                    "--max-projected-bytes",
                    str(max_projected_bytes),
                    *(
                        ["--allow-oversized-projection"]
                        if allow_oversized_projection
                        else []
                    ),
                ],
                upstream=[
                    {
                        "path": paths["model_manifest"],
                        "json_require": {"schema_version": MODEL_SCHEMA},
                    },
                    {
                        "path": tokenization,
                        "json_require": {"schema_version": TOKENIZATION_SCHEMA},
                    },
                ],
                expected_outputs=[projection_input],
                projection_input=projection_input,
                projection_data_dir=projection_data_dir,
            )
            project = _job(
                parent=parent,
                phase="project",
                role=role,
                runner_dir=project_runner,
                command=[
                    "python",
                    "-m",
                    "immune_health.cli",
                    "project-tripso",
                    "--model-manifest",
                    paths["model_manifest"],
                    "--projection-manifest",
                    projection_input,
                    "--output-dir",
                    projection_data_dir,
                    "--vendor-root",
                    "${PROJECT_ROOT}/tripso_code/tripso",
                    "--batch-size",
                    str(batch_size),
                    "--precision",
                    str(precision),
                ],
                upstream=[
                    {
                        "path": paths["model_manifest"],
                        "json_require": {"schema_version": MODEL_SCHEMA},
                    },
                    {
                        "path": projection_input,
                        "json_require": {"schema_version": PROJECTION_SCHEMA},
                    },
                ],
                expected_outputs=[
                    f"{projection_data_dir}/embeddings/{role}_set",
                    f"{projection_data_dir}/projection_output_manifest.json",
                ],
                projection_input=projection_input,
                projection_data_dir=projection_data_dir,
            )
            output[f"bind_{role}"].append(bind)
            output[f"project_{role}"].append(project)
            policy = (
                "all_gps_bounded_diagnostic"
                if allow_all_gps
                else "fold_bound_training_candidates"
            )
            for generated in (bind, project):
                generated["gp_projection_policy"] = policy
                generated["maximum_projected_bytes"] = int(max_projected_bytes)
                generated["oversized_projection_override"] = bool(
                    allow_oversized_projection
                )
                generated["eligible_for_model_selection"] = role == "validation"
                generated["outer_query_evaluation_only"] = role == "query"
                generated["outer_query_allowlist_required"] = role == "query"
    return output


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = list(rows)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in values
    )
    _atomic_write(path, content)
    return len(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("slurm/manifests/post_training")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--precision", default="32")
    parser.add_argument(
        "--max-projected-bytes", type=int, default=DEFAULT_MAX_PROJECTED_BYTES
    )
    parser.add_argument(
        "--allow-all-gps",
        action="store_true",
        help="Explicit bounded diagnostic; production uses fold-bound candidates",
    )
    parser.add_argument("--allow-oversized-projection", action="store_true")
    parser.add_argument(
        "--enable-outer-query-evaluation",
        action="store_true",
        help=(
            "Generate held-out query jobs only for the separately allowlisted, "
            "inner-validation-selected training jobs"
        ),
    )
    parser.add_argument(
        "--outer-query-selected-job-allowlist",
        type=Path,
        help=(
            "Hashed JSON manifest of selected training job IDs; required together "
            "with --enable-outer-query-evaluation"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training_jobs: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for path in args.training_manifest:
        training_jobs.extend(_read_jobs(path))
        source_hashes[str(path.resolve())] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if args.enable_outer_query_evaluation != (
        args.outer_query_selected_job_allowlist is not None
    ):
        raise ValueError(
            "Pass --enable-outer-query-evaluation and "
            "--outer-query-selected-job-allowlist together"
        )
    selected_query_ids: set[str] = set()
    query_allowlist: dict[str, Any] | None = None
    query_allowlist_file_hash: str | None = None
    if args.outer_query_selected_job_allowlist is not None:
        selected_query_ids, query_allowlist = _read_outer_query_allowlist(
            args.outer_query_selected_job_allowlist
        )
        query_allowlist_file_hash = hashlib.sha256(
            args.outer_query_selected_job_allowlist.read_bytes()
        ).hexdigest()
    jobs = generate_post_training_jobs(
        training_jobs,
        batch_size=args.batch_size,
        precision=args.precision,
        max_projected_bytes=args.max_projected_bytes,
        allow_all_gps=args.allow_all_gps,
        allow_oversized_projection=args.allow_oversized_projection,
        enable_outer_query_evaluation=args.enable_outer_query_evaluation,
        outer_query_selected_job_ids=selected_query_ids,
    )
    summary = {
        "schema_version": "immune-health-post-training-summary/v1",
        "training_manifests": source_hashes,
        "runnable_training_jobs": sum(
            bool(row.get("runnable", False)) for row in training_jobs
        ),
        "counts": {name: len(rows) for name, rows in jobs.items()},
        "submitted": False,
        "per_gp_jobs_generated": False,
        "gp_projection_policy": (
            "all_gps_bounded_diagnostic"
            if args.allow_all_gps
            else "fold_bound_training_candidates"
        ),
        "maximum_projected_bytes": args.max_projected_bytes,
        "oversized_projection_override": args.allow_oversized_projection,
        "default_model_selection_role": "validation",
        "outer_query_evaluation_enabled": args.enable_outer_query_evaluation,
        "outer_query_selected_job_count": len(selected_query_ids),
        "outer_query_allowlist": (
            {
                "path": str(args.outer_query_selected_job_allowlist.resolve()),
                "sha256": query_allowlist_file_hash,
                "manifest_sha256": query_allowlist["manifest_sha256"],
                "selection_basis": query_allowlist["selection_basis"],
                "outer_query_data_consulted_for_selection": False,
            }
            if query_allowlist is not None
            else None
        ),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for name, rows in jobs.items():
        _write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    _atomic_write(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
