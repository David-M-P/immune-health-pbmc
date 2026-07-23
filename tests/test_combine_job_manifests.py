from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
JOB_SCHEMA = "immune-health-slurm-job/v1"


def _load_module():
    path = REPOSITORY_ROOT / "scripts" / "combine_job_manifests.py"
    spec = importlib.util.spec_from_file_location("combine_job_manifests_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, job_ids: tuple[str, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": JOB_SCHEMA,
                    "job_id": job_id,
                    "runnable": True,
                }
            )
            + "\n"
            for job_id in job_ids
        ),
        encoding="utf-8",
    )


def test_combiner_preserves_input_and_row_order_and_reports_hashes(
    tmp_path: Path,
) -> None:
    combiner = _load_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "combined" / "jobs.jsonl"
    _write_manifest(first, ("first-0", "first-1"))
    _write_manifest(second, ("second-0",))

    summary = combiner.combine_job_manifests((first, second), output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["job_id"] for row in rows] == ["first-0", "first-1", "second-0"]
    assert summary["schema_version"] == "immune-health-combined-job-manifest/v1"
    assert summary["output_row_count"] == 3
    assert len(summary["output_sha256"]) == 64
    assert [record["row_count"] for record in summary["inputs"]] == [2, 1]
    assert all(len(record["sha256"]) == 64 for record in summary["inputs"])


def test_combiner_rejects_duplicate_job_ids_without_replacing_output(
    tmp_path: Path,
) -> None:
    combiner = _load_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "combined.jsonl"
    _write_manifest(first, ("shared-job",))
    _write_manifest(second, ("shared-job",))
    output.write_text("existing-output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate job_id"):
        combiner.combine_job_manifests((first, second), output)

    assert output.read_text(encoding="utf-8") == "existing-output\n"


@pytest.mark.parametrize(
    "contents",
    (
        "",
        "\n",
        "not-json\n",
        json.dumps({"schema_version": "wrong", "job_id": "job-0"}) + "\n",
        json.dumps({"schema_version": JOB_SCHEMA, "job_id": "unsafe id"}) + "\n",
    ),
)
def test_combiner_fails_closed_on_invalid_rows(
    tmp_path: Path,
    contents: str,
) -> None:
    combiner = _load_module()
    source = tmp_path / "source.jsonl"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        combiner.combine_job_manifests((source,), tmp_path / "output.jsonl")


def test_combiner_rejects_repeated_input_and_in_place_output(tmp_path: Path) -> None:
    combiner = _load_module()
    source = tmp_path / "source.jsonl"
    _write_manifest(source, ("job-0",))

    with pytest.raises(ValueError, match="more than once"):
        combiner.combine_job_manifests((source, source), tmp_path / "output.jsonl")
    with pytest.raises(ValueError, match="different"):
        combiner.combine_job_manifests((source,), source)


def test_combiner_rejects_input_and_output_symlinks(tmp_path: Path) -> None:
    combiner = _load_module()
    source = tmp_path / "source.jsonl"
    source_link = tmp_path / "source-link.jsonl"
    output_target = tmp_path / "output-target.jsonl"
    output_link = tmp_path / "output-link.jsonl"
    _write_manifest(source, ("job-0",))
    source_link.symlink_to(source)
    output_target.write_text("existing\n", encoding="utf-8")
    output_link.symlink_to(output_target)

    with pytest.raises(ValueError, match="regular file"):
        combiner.combine_job_manifests((source_link,), tmp_path / "output.jsonl")
    with pytest.raises(ValueError, match="symlink"):
        combiner.combine_job_manifests((source,), output_link)

    assert output_target.read_text(encoding="utf-8") == "existing\n"
