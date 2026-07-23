from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_nodepack_module():
    path = REPOSITORY_ROOT / "slurm" / "run_manifest_nodepack.py"
    spec = importlib.util.spec_from_file_location("run_manifest_nodepack_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_runner_module():
    path = REPOSITORY_ROOT / "slurm" / "run_manifest_task.py"
    spec = importlib.util.spec_from_file_location(
        "run_manifest_task_nodepack_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, row_count: int) -> None:
    rows = [
        {"schema_version": "immune-health-slurm-job/v1", "job_id": f"job-{index}"}
        for index in range(row_count)
    ]
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_nodepack_index_parser_supports_sparse_stepped_specs() -> None:
    nodepack = _load_nodepack_module()

    assert nodepack.parse_indices(None, 10) == tuple(range(10))
    assert nodepack.parse_indices("0-5,60-65", 150) == (
        0,
        1,
        2,
        3,
        4,
        5,
        60,
        61,
        62,
        63,
        64,
        65,
    )
    assert nodepack.parse_indices("0-24:5", 25) == (0, 5, 10, 15, 20)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("0,0", "(?i)more than once|duplicate"),
        ("-1", "(?i)invalid"),
        ("10", "(?i)outside|range"),
        ("0-10", "(?i)outside|range"),
        ("5-2", "(?i)descending"),
        ("0-5:0", "(?i)step"),
        ("0-7%2", "(?i)concurrency|throttle"),
    ],
)
def test_nodepack_index_parser_fails_closed(
    spec: str,
    message: str,
) -> None:
    nodepack = _load_nodepack_module()

    with pytest.raises(ValueError, match=message):
        nodepack.parse_indices(spec, 10)


def test_nodepack_dense_and_sparse_tail_blocks() -> None:
    nodepack = _load_nodepack_module()

    dense = nodepack.parse_indices(None, 150)
    assert nodepack.indices_for_block(dense, 0, 8) == tuple(range(8))
    assert nodepack.indices_for_block(dense, 18, 8) == (
        144,
        145,
        146,
        147,
        148,
        149,
    )

    sentinels = nodepack.parse_indices("0-5,60-65", 150)
    assert nodepack.indices_for_block(sentinels, 0, 8) == (
        0,
        1,
        2,
        3,
        4,
        5,
        60,
        61,
    )
    assert nodepack.indices_for_block(sentinels, 1, 8) == (62, 63, 64, 65)

    with pytest.raises(IndexError, match="(?i)block"):
        nodepack.indices_for_block(dense, 19, 8)


def test_nodepack_plan_uses_nineteen_nodes_for_150_rows(tmp_path: Path) -> None:
    nodepack = _load_nodepack_module()
    manifest = tmp_path / "stage1.jsonl"
    _write_manifest(manifest, 150)

    plan = nodepack.nodepack_plan(
        manifest=manifest,
        indices=nodepack.parse_indices(None, 150),
        workers_per_node=8,
    )

    assert plan["selected_row_count"] == 150
    assert plan["workers_per_node"] == 8
    assert plan["node_array_elements"] == 19
    assert plan["slurm_array_spec"] == "0-18"
    assert plan["blocks"][0]["manifest_indices"] == list(range(8))
    assert plan["blocks"][-1]["manifest_indices"] == list(range(144, 150))


def test_nodepack_sbatch_uses_one_eight_task_step_with_per_task_gpu_binding() -> None:
    text = (REPOSITORY_ROOT / "slurm" / "tripso_nodepack.sbatch").read_text(
        encoding="utf-8"
    )

    assert '--ntasks="${TRIPSO_WORKERS_PER_NODE}"' in text
    assert "--gpus-per-task=1" in text
    assert "--gpu-bind=single:1" in text
    assert "--exact" in text
    assert "--kill-on-bad-exit=0" in text
    assert "--wait=0" in text
    assert "--block-index" in text
    assert '"${SLURM_ARRAY_TASK_ID}"' in text
    assert "run_manifest_nodepack.py" in text
    assert "%t" in text
    assert "run_manifest_task.py" not in text


def test_cpu_nodepack_uses_cpu_bound_tasks_and_zero_gpu_contract() -> None:
    text = (REPOSITORY_ROOT / "slurm" / "cpu_nodepack.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --account=immunehealth" in text
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --time" not in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --cpus-per-task" not in text
    assert '--ntasks="${CPU_WORKERS_PER_NODE}"' in text
    assert '--cpus-per-task="${SLURM_CPUS_PER_TASK}"' in text
    assert "--cpu-bind=cores" in text
    assert "--gpus-per-task" not in text
    assert "--gpu-bind" not in text
    assert "--expected-visible-gpus 0" in text
    assert "--exact" in text
    assert "--kill-on-bad-exit=0" in text
    assert "--wait=0" in text
    assert "--block-index" in text
    assert '"${SLURM_ARRAY_TASK_ID}"' in text
    assert "run_manifest_nodepack.py" in text
    assert "%t" in text
    assert "run_manifest_task.py" not in text


def test_nodepack_rank_maps_to_one_row_and_isolates_lightning_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodepack = _load_nodepack_module()
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("{}\n" * 150, encoding="utf-8")
    runner = tmp_path / "run_manifest_task.py"
    runner.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_runner(command, *, env, check):
        captured.update(command=command, env=env, check=check)
        return SimpleNamespace(returncode=7)

    inherited = {
        "CUDA_VISIBLE_DEVICES": "3",
        "SLURM_NTASKS": "8",
        "SLURM_NTASKS_PER_NODE": "8",
        "SLURM_NPROCS": "8",
        "SLURM_PROCID": "3",
        "SLURM_LOCALID": "3",
        "SLURM_NODEID": "0",
        "SLURM_STEP_NUM_TASKS": "8",
        "WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_RANK": "3",
        "MASTER_ADDR": "node01",
        "MASTER_PORT": "48123",
        "PMI_RANK": "3",
        "PMI_SIZE": "8",
        "PMIX_RANK": "3",
        "OMPI_COMM_WORLD_RANK": "3",
        "OMPI_COMM_WORLD_SIZE": "8",
        "UNRELATED_SETTING": "preserved",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    result = nodepack.run_nodepack(
        manifest=manifest,
        block_index=1,
        workers_per_node=8,
        indices=tuple(range(150)),
        worker_rank=3,
        runner=runner,
        command_runner=fake_runner,
    )

    assert result == 7
    assert captured["command"] == [
        sys.executable,
        str(runner.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--index",
        "11",
    ]
    assert captured["check"] is False
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["CUDA_VISIBLE_DEVICES"] == "3"
    assert child_env["UNRELATED_SETTING"] == "preserved"
    assert child_env["IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS"] == "1"
    assert child_env["IMMUNE_HEALTH_NODEPACK_BLOCK_INDEX"] == "1"
    assert child_env["IMMUNE_HEALTH_NODEPACK_WORKER_RANK"] == "3"
    assert child_env["IMMUNE_HEALTH_NODEPACK_MANIFEST_INDEX"] == "11"
    assert child_env["SLURM_NTASKS"] == "1"
    assert child_env["SLURM_NTASKS_PER_NODE"] == "1"
    assert child_env["SLURM_NPROCS"] == "1"
    assert child_env["SLURM_PROCID"] == "0"
    assert child_env["SLURM_LOCALID"] == "0"
    assert child_env["SLURM_NODEID"] == "0"
    assert child_env["SLURM_STEP_NUM_TASKS"] == "1"
    for name in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PMI_RANK",
        "PMI_SIZE",
        "PMIX_RANK",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
    ):
        assert name not in child_env


def test_nodepack_cpu_worker_hides_inherited_gpus_and_expects_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodepack = _load_nodepack_module()
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("{}\n" * 8, encoding="utf-8")
    runner = tmp_path / "run_manifest_task.py"
    runner.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_runner(command, *, env, check):
        captured.update(command=command, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "2")

    result = nodepack.run_nodepack(
        manifest=manifest,
        block_index=0,
        workers_per_node=8,
        indices=tuple(range(8)),
        worker_rank=2,
        runner=runner,
        expected_visible_gpus=0,
        command_runner=fake_runner,
    )

    assert result == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["CUDA_VISIBLE_DEVICES"] == ""
    assert child_env["IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS"] == "0"
    assert (
        child_env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_CUDA_VISIBLE_DEVICES"]
        == "0,1,2,3,4,5,6,7"
    )
    assert child_env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_SLURM_NTASKS"] == "8"
    assert child_env["IMMUNE_HEALTH_NODEPACK_ORIGINAL_SLURM_PROCID"] == "2"
    assert child_env["SLURM_NTASKS"] == "1"
    assert child_env["SLURM_PROCID"] == "0"


def test_nodepack_rejects_negative_expected_visible_gpu_count(
    tmp_path: Path,
) -> None:
    nodepack = _load_nodepack_module()

    with pytest.raises(ValueError, match="cannot be negative"):
        nodepack.run_nodepack(
            manifest=tmp_path / "jobs.jsonl",
            block_index=0,
            workers_per_node=1,
            indices=(0,),
            worker_rank=0,
            runner=tmp_path / "runner.py",
            expected_visible_gpus=-1,
            dry_run=True,
        )

    with pytest.raises(ValueError, match="zero or one"):
        nodepack.run_nodepack(
            manifest=tmp_path / "jobs.jsonl",
            block_index=0,
            workers_per_node=1,
            indices=(0,),
            worker_rank=0,
            runner=tmp_path / "runner.py",
            expected_visible_gpus=2,
            dry_run=True,
        )


def test_nodepack_tail_rank_is_idle_without_starting_a_child(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nodepack = _load_nodepack_module()

    def unexpected_runner(*args, **kwargs):
        del args, kwargs
        raise AssertionError("An idle tail rank must not start a child process")

    result = nodepack.run_nodepack(
        manifest=tmp_path / "unused.jsonl",
        block_index=1,
        workers_per_node=8,
        indices=tuple(range(10)),
        worker_rank=2,
        runner=tmp_path / "unused.py",
        command_runner=unexpected_runner,
    )

    assert result == 0
    assert '"status": "idle_tail_worker"' in capsys.readouterr().out


def test_nodepack_dry_run_never_starts_child(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nodepack = _load_nodepack_module()

    def unexpected_runner(*args, **kwargs):
        del args, kwargs
        raise AssertionError("A dry run must not start a child process")

    result = nodepack.run_nodepack(
        manifest=tmp_path / "jobs.jsonl",
        block_index=0,
        workers_per_node=8,
        indices=tuple(range(8)),
        worker_rank=7,
        runner=tmp_path / "runner.py",
        dry_run=True,
        command_runner=unexpected_runner,
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"manifest_index": 7' in output
    assert '"status": "planned"' in output


def test_manifest_runner_fails_closed_if_worker_sees_more_than_one_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_manifest_runner_module()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 8))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS", "1")

    with pytest.raises(RuntimeError, match="expected exactly 1.*sees 8"):
        runner.validate_visible_gpu_expectation()

    fake_torch.cuda.device_count = lambda: 1
    assert runner.validate_visible_gpu_expectation() == {"expected": 1, "observed": 1}


def test_manifest_runner_accepts_explicit_zero_gpu_cpu_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_manifest_runner_module()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS", "0")

    assert runner.validate_visible_gpu_expectation() == {"expected": 0, "observed": 0}

    fake_torch.cuda.device_count = lambda: 1
    with pytest.raises(RuntimeError, match="expected exactly 0.*sees 1"):
        runner.validate_visible_gpu_expectation()
