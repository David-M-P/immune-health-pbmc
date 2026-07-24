from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_writer_records_concrete_gefion_values(tmp_path: Path) -> None:
    output = tmp_path / "gefion.runtime.yaml"
    project_root = REPOSITORY_ROOT.resolve()
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "write_gefion_runtime_config.py"),
        "--output",
        str(output),
        "--project-root",
        str(project_root),
        "--work-root",
        "/gefion/work",
        "--data-root",
        "/gefion/work/data",
        "--environment-root",
        "/gefion/work/env",
        "--output-root",
        "/gefion/work/output",
        "--run-id",
        "test_run",
        "--account",
        "cu_0071",
        "--cpu-partition",
        "defq",
        "--gpu-partition",
        "defq",
        "--activation-script",
        str(project_root / "slurm" / "activate_packed_environment.sh"),
        "--cpu-walltime",
        "7-00:00:00",
        "--gpu-walltime",
        "7-00:00:00",
        "--projection-walltime",
        "7-00:00:00",
        "--cpu-memory",
        "400G",
        "--gpu-memory",
        "400G",
        "--cpu-workers",
        "4",
        "--cpu-cpus-per-worker",
        "4",
        "--gpu-workers",
        "8",
        "--gpu-cpus-per-worker",
        "20",
        "--cpu-node-concurrency",
        "1",
        "--gpu-node-concurrency",
        "1",
        "--gpus-per-node",
        "8",
    ]
    subprocess.run(command, check=True)

    runtime = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert runtime["schema_version"] == "immune-health-gefion-runtime/v1"
    assert runtime["cluster"] == {
        "name": "gefion",
        "scheduler": "slurm",
        "project_account": "cu_0071",
        "partition_cpu": "defq",
        "partition_gpu": "defq",
    }
    assert runtime["slurm"]["gpus_per_node"] == 8
    assert runtime["slurm"]["cpu"]["workers_per_node"] == 4
    assert runtime["slurm"]["gpu"]["workers_per_node"] == 8
    assert "<" not in output.read_text(encoding="utf-8")
