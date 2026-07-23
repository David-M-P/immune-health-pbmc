#!/usr/bin/env python3
"""Map one task in an exclusive node to one manifest row.

Concurrency is owned by one Slurm ``srun`` step.  Every task receives the same
block index and a distinct ``SLURM_PROCID``; this module maps that pair to one
selected JSONL row and then delegates to the ordinary, restart-safe manifest
runner.  GPU launchers retain the default expectation of one visible GPU per
worker.  CPU launchers must explicitly request zero visible GPUs, which also
hides any inherited node-level CUDA allocation from the child job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

JOB_SCHEMA = "immune-health-slurm-job/v1"
INDEX_TOKEN = re.compile(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$")
SINGLE_PROCESS_SLURM_ENV = {
    "SLURM_NTASKS": "1",
    "SLURM_NPROCS": "1",
    "SLURM_NTASKS_PER_NODE": "1",
    "SLURM_TASKS_PER_NODE": "1",
    "SLURM_STEP_NUM_TASKS": "1",
    "SLURM_STEP_TASKS_PER_NODE": "1",
    "SLURM_PROCID": "0",
    "SLURM_LOCALID": "0",
    "SLURM_NODEID": "0",
    "SLURM_GTIDS": "0",
}
DISTRIBUTED_ENV_NAMES = frozenset(
    {
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
)
DISTRIBUTED_ENV_PREFIXES = (
    "PMI_",
    "PMIX_",
    "OMPI_COMM_WORLD_",
    "MV2_COMM_WORLD_",
    "TORCHELASTIC_",
)


def manifest_row_count(path: Path) -> int:
    """Count and minimally validate the non-empty rows of one job manifest."""

    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for line_number, line in enumerate(rows, start=1):
        if not line.strip():
            raise ValueError(
                f"Blank manifest rows are unsupported because indices are exact: "
                f"{path}:{line_number}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != JOB_SCHEMA:
            raise ValueError(f"Unsupported job manifest row at {path}:{line_number}")
    return len(rows)


def parse_indices(spec: str | None, row_count: int) -> tuple[int, ...]:
    """Expand a Slurm-like index specification without a concurrency suffix."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    if spec is None:
        return tuple(range(row_count))
    value = spec.strip()
    if not value:
        raise ValueError("Index specification cannot be empty")
    if "%" in value:
        raise ValueError(
            "Index specification cannot contain a Slurm % concurrency suffix; "
            "throttle the node array at sbatch submission instead"
        )

    expanded: list[int] = []
    observed: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        match = INDEX_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid manifest index token: {token!r}")
        start = int(match.group(1))
        end_text = match.group(2)
        step_text = match.group(3)
        if end_text is None:
            if step_text is not None:  # Defensive; excluded by the regex.
                raise ValueError(f"A step requires a range: {token!r}")
            values = (start,)
        else:
            end = int(end_text)
            step = int(step_text or 1)
            if end < start:
                raise ValueError(f"Descending manifest range is unsupported: {token!r}")
            if step < 1:
                raise ValueError(f"Manifest range step must be positive: {token!r}")
            values = range(start, end + 1, step)
        for index in values:
            if index >= row_count:
                raise ValueError(
                    f"Manifest index {index} is outside 0-{row_count - 1}"
                )
            if index in observed:
                raise ValueError(f"Manifest index is selected more than once: {index}")
            observed.add(index)
            expanded.append(index)
    if not expanded:
        raise ValueError("Index specification selected no manifest rows")
    return tuple(expanded)


def indices_from_file(path: Path, row_count: int) -> tuple[int, ...]:
    """Read a reviewed index selection, allowing comments and one token per line."""

    tokens: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        content = raw.split("#", 1)[0].strip()
        if content:
            tokens.extend(part.strip() for part in content.split(",") if part.strip())
    if not tokens:
        raise ValueError(f"Index file selected no rows: {path}")
    return parse_indices(",".join(tokens), row_count)


def indices_for_block(
    indices: Sequence[int], block_index: int, workers_per_node: int
) -> tuple[int, ...]:
    """Return the selected manifest rows assigned to one node-array element."""

    if block_index < 0:
        raise ValueError("block_index must be non-negative")
    if workers_per_node < 1:
        raise ValueError("workers_per_node must be positive")
    start = block_index * workers_per_node
    if start >= len(indices):
        raise IndexError(
            f"block {block_index} starts after {len(indices)} selected manifest rows"
        )
    return tuple(indices[start : start + workers_per_node])


def nodepack_plan(
    *, manifest: Path, indices: Sequence[int], workers_per_node: int
) -> dict[str, object]:
    """Return the exact node-array plan for inspection before submission."""

    if workers_per_node < 1:
        raise ValueError("workers_per_node must be positive")
    n_blocks = math.ceil(len(indices) / workers_per_node)
    blocks = [
        {
            "block_index": block,
            "manifest_indices": list(
                indices_for_block(indices, block, workers_per_node)
            ),
        }
        for block in range(n_blocks)
    ]
    return {
        "schema_version": "immune-health-slurm-nodepack-plan/v1",
        "manifest": str(Path(manifest).resolve()),
        "manifest_row_count": manifest_row_count(manifest),
        "selected_row_count": len(indices),
        "workers_per_node": workers_per_node,
        "node_array_elements": n_blocks,
        "slurm_array_spec": f"0-{n_blocks - 1}",
        "blocks": blocks,
    }


def isolated_worker_environment(
    source: dict[str, str],
    *,
    block_index: int,
    worker_rank: int,
    manifest_index: int,
    expected_visible_gpus: int = 1,
) -> dict[str, str]:
    """Hide the multi-task Slurm world from one independent manifest worker.

    The enclosing ``srun`` uses multiple tasks solely for resource binding.
    TRIPSO models are intentionally independent, so each child must see a
    one-task Slurm environment or Lightning may infer a distributed job and hang
    or change sampler behavior.  GPU binding variables and job/array provenance
    are deliberately preserved for GPU workers.  A CPU worker explicitly hides
    CUDA devices even if the exclusive outer allocation inherited GPU visibility.
    """

    if expected_visible_gpus < 0:
        raise ValueError("expected_visible_gpus cannot be negative")
    if expected_visible_gpus > 1:
        raise ValueError("expected_visible_gpus must be either zero or one")

    env = dict(source)
    original_tasks = env.get("SLURM_NTASKS")
    original_procid = env.get("SLURM_PROCID")
    original_cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    for name in list(env):
        if name in DISTRIBUTED_ENV_NAMES or name.startswith(
            DISTRIBUTED_ENV_PREFIXES
        ):
            env.pop(name, None)
    env.update(SINGLE_PROCESS_SLURM_ENV)
    env["IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS"] = str(expected_visible_gpus)
    if expected_visible_gpus == 0:
        # Apply this after Slurm has created the task environment.  An exclusive
        # CPU preprocessing allocation can otherwise expose node GPUs even when
        # the worker never requested or uses them.
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["IMMUNE_HEALTH_NODEPACK_BLOCK_INDEX"] = str(block_index)
    env["IMMUNE_HEALTH_NODEPACK_WORKER_RANK"] = str(worker_rank)
    env["IMMUNE_HEALTH_NODEPACK_MANIFEST_INDEX"] = str(manifest_index)
    if original_tasks is not None:
        env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_SLURM_NTASKS"] = original_tasks
    if original_procid is not None:
        env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_SLURM_PROCID"] = original_procid
    if original_cuda_visible_devices is not None:
        env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_CUDA_VISIBLE_DEVICES"] = (
            original_cuda_visible_devices
        )
    return env


def run_nodepack(
    *,
    manifest: Path,
    block_index: int,
    workers_per_node: int,
    indices: Sequence[int],
    worker_rank: int,
    runner: Path,
    expected_visible_gpus: int = 1,
    dry_run: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> int:
    """Run the one manifest row assigned to this Slurm task rank."""

    if worker_rank < 0 or worker_rank >= workers_per_node:
        raise ValueError(
            f"worker_rank must satisfy 0 <= rank < {workers_per_node}; "
            f"observed {worker_rank}"
        )
    if expected_visible_gpus < 0:
        raise ValueError("expected_visible_gpus cannot be negative")
    if expected_visible_gpus > 1:
        raise ValueError("expected_visible_gpus must be either zero or one")
    block = indices_for_block(indices, block_index, workers_per_node)
    if worker_rank >= len(block):
        print(
            json.dumps(
                {
                    "status": "idle_tail_worker",
                    "block_index": block_index,
                    "worker_rank": worker_rank,
                    "workers_per_node": workers_per_node,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    manifest_index = int(block[worker_rank])
    command = [
        sys.executable,
        str(Path(runner).resolve()),
        "--manifest",
        str(Path(manifest).resolve()),
        "--index",
        str(manifest_index),
    ]
    mapping = {
        "status": "planned" if dry_run else "starting",
        "block_index": block_index,
        "worker_rank": worker_rank,
        "workers_per_node": workers_per_node,
        "manifest_index": manifest_index,
        "command": command,
    }
    print(json.dumps(mapping, sort_keys=True), flush=True)
    if dry_run:
        return 0

    env = isolated_worker_environment(
        dict(os.environ),
        block_index=block_index,
        worker_rank=worker_rank,
        manifest_index=manifest_index,
        expected_visible_gpus=expected_visible_gpus,
    )
    completed = command_runner(command, env=env, check=False)
    return int(completed.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--workers-per-node", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices")
    selection.add_argument("--indices-file", type=Path)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument(
        "--expected-visible-gpus",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Exact number of GPUs each child must see. The default of one "
            "preserves GPU node-pack behavior; CPU launchers must pass zero."
        ),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_manifest_task.py"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    row_count = manifest_row_count(args.manifest)
    indices = (
        indices_from_file(args.indices_file, row_count)
        if args.indices_file is not None
        else parse_indices(args.indices, row_count)
    )
    if args.plan_only:
        print(
            json.dumps(
                nodepack_plan(
                    manifest=args.manifest,
                    indices=indices,
                    workers_per_node=args.workers_per_node,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.block_index is None:
        raise ValueError("--block-index is required unless --plan-only is used")
    worker_rank = args.worker_rank
    if worker_rank is None:
        rank_text = os.environ.get("SLURM_PROCID")
        if rank_text is None:
            raise RuntimeError(
                "SLURM_PROCID is required inside the node-pack srun step; use "
                "--worker-rank only for a local dry run"
            )
        worker_rank = int(rank_text)
    return run_nodepack(
        manifest=args.manifest,
        block_index=args.block_index,
        workers_per_node=args.workers_per_node,
        indices=indices,
        worker_rank=worker_rank,
        runner=args.runner,
        expected_visible_gpus=args.expected_visible_gpus,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
