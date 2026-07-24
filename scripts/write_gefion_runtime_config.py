#!/usr/bin/env python3
"""Write placeholder-free Gefion runtime provenance from explicit arguments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--cpu-partition", required=True)
    parser.add_argument("--gpu-partition", required=True)
    parser.add_argument("--activation-script", type=Path, required=True)
    parser.add_argument("--cpu-walltime", required=True)
    parser.add_argument("--gpu-walltime", required=True)
    parser.add_argument("--projection-walltime", required=True)
    parser.add_argument("--cpu-memory", required=True)
    parser.add_argument("--gpu-memory", required=True)
    parser.add_argument("--cpu-workers", type=int, required=True)
    parser.add_argument("--cpu-cpus-per-worker", type=int, required=True)
    parser.add_argument("--gpu-workers", type=int, required=True)
    parser.add_argument("--gpu-cpus-per-worker", type=int, required=True)
    parser.add_argument("--cpu-node-concurrency", type=int, required=True)
    parser.add_argument("--gpu-node-concurrency", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    return parser.parse_args(argv)


def _git_value(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    status = _git_value(project_root, "status", "--porcelain")
    payload = {
        "schema_version": "immune-health-gefion-runtime/v1",
        "run": {
            "id": args.run_id,
            "output_root": str(args.output_root),
        },
        "cluster": {
            "name": "gefion",
            "scheduler": "slurm",
            "project_account": args.account,
            "partition_cpu": args.cpu_partition,
            "partition_gpu": args.gpu_partition,
        },
        "paths": {
            "project_root": str(project_root),
            "work_root": str(args.work_root),
            "data_root": str(args.data_root),
            "environment_root": str(args.environment_root),
        },
        "environment": {
            "activation_script": str(args.activation_script),
            "python": str(args.environment_root / "bin" / "python"),
        },
        "slurm": {
            "nodes_per_array_element": 1,
            "exclusive": True,
            "gpus_per_node": args.gpus_per_node,
            "cpu": {
                "partition": args.cpu_partition,
                "walltime": args.cpu_walltime,
                "memory": args.cpu_memory,
                "workers_per_node": args.cpu_workers,
                "cpus_per_worker": args.cpu_cpus_per_worker,
                "maximum_concurrent_nodes": args.cpu_node_concurrency,
                "cuda_visible_to_worker": False,
            },
            "gpu": {
                "partition": args.gpu_partition,
                "walltime": args.gpu_walltime,
                "projection_walltime": args.projection_walltime,
                "memory": args.gpu_memory,
                "workers_per_node": args.gpu_workers,
                "cpus_per_worker": args.gpu_cpus_per_worker,
                "gpus_per_worker": 1,
                "maximum_concurrent_nodes": args.gpu_node_concurrency,
            },
        },
        "git": {
            "commit": _git_value(project_root, "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    temporary.replace(args.output)
    print(f"Wrote placeholder-free Gefion provenance: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
