from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_record(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _project_job(tmp_path: Path, parent: str, role: str, seed: int) -> dict[str, Any]:
    data_dir = tmp_path / parent / "post_training" / "projection_data" / role
    return {
        "schema_version": "immune-health-slurm-job/v1",
        "job_id": f"posttrain-project-{role}-{parent}",
        "stage": f"posttrain_project_{role}",
        "parent_training_job_id": parent,
        "projection_role": role,
        "projection_data_dir": str(data_dir),
        "seed": seed,
        "runnable": True,
    }


def _model(
    tmp_path: Path,
    parent: str,
    *,
    endpoints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    metadata = {
        role: _file_record(tmp_path / "fixed" / f"{parent}-{role}.parquet", role)
        for role in ("reference", "validation", "query")
    }
    scoring = {}
    for role in ("validation", "query"):
        scoring[role] = {
            "query_genes": _file_record(
                tmp_path / "fixed" / f"{parent}-{role}-genes.txt", "ENSG1\n"
            ),
            "frozen_vocabulary": _file_record(
                tmp_path / "fixed" / f"{parent}-{role}-vocabulary.txt", "ENSG1\n"
            ),
            "gp_coverage": {"GP_A": 1.0, "GP_B": 0.9},
        }
    return {
        "parent_training_job_id": parent,
        "model_selection_status": "selected",
        "selection_group_id": "B-lodo-query-hybrid-hvg3000",
        "lineage": "B cells",
        "candidate_endpoints": endpoints or [{"gp_id": "GP_A", "fine_type": "Naive B"}],
        "cell_metadata": metadata,
        "scoring_resources": scoring,
        "n_cell_bootstrap": 25,
    }


def _plan(tmp_path: Path, models: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
    ontology = _file_record(tmp_path / "fixed" / "fine_types.json", "{}\n")
    payload = {
        "schema_version": "immune-health-downstream-candidate-plan/v1",
        "selection_scope": "training_or_inner_validation_only",
        "outer_query_results_used_for_selection": False,
        "healthy_reference": {"minimum_exact_sex_donors": 20},
        "fine_type_universe": ontology,
        "models": models,
    }
    downstream = _load_script(
        "downstream_hash_helper", "scripts/generate_downstream_jobs.py"
    )
    payload["manifest_sha256"] = downstream._canonical_hash(payload)
    path = tmp_path / "candidate_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def _fake_projection_state(job: dict[str, Any], gp_ids: list[str]):
    role = job["projection_role"]
    payload = {
        "schema_version": "immune-health-tripso-projection-output/v1",
        "projection_role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "reference_design": "lodo",
        "heldout_dataset": "query_cohort",
        "fold_id": "lodo_query_cohort",
        "lineage": "B cells",
        "seed": job["seed"],
        "datasets": ["train_a", "train_b"],
        "model_manifest": f"/models/{job['parent_training_job_id']}.json",
        "projection_input_manifest": f"/inputs/{role}.json",
    }
    data_dir = job["projection_data_dir"]
    return {
        "job": job,
        "payload": payload,
        "validation": {"gp_program_ids": gp_ids},
        "manifest": f"{data_dir}/projection_output_manifest.json",
        "manifest_file_sha256": "a" * 64,
        "arrow_dataset": f"{data_dir}/embeddings/{role}_set",
    }, None


def test_pass1_is_reference_validation_only_and_dependency_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream = _load_script(
        "downstream_pass1_test", "scripts/generate_downstream_jobs.py"
    )
    models = [_model(tmp_path, "train-a"), _model(tmp_path, "train-b")]
    plan, plan_path = _plan(tmp_path, models)
    jobs = [
        _project_job(tmp_path, parent, role, seed)
        for parent, seed in (("train-a", 42), ("train-b", 43))
        for role in ("reference", "validation", "query")
    ]
    monkeypatch.setattr(downstream, "_projection_state", _fake_projection_state)
    generated = downstream.generate_pass1(
        jobs,
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )

    assert generated["convert_query"] == []
    assert generated["aggregate_query"] == []
    assert generated["endpoint_query"] == []
    assert generated["score_query"] == []
    assert len(generated["convert_reference"]) == 2
    assert len(generated["convert_validation"]) == 2
    assert len(generated["fit_reference"]) == 4
    assert len(generated["score_validation"]) == 4
    assert len(generated["empirical_scoring"]) == 4
    for job in generated["fit_reference"]:
        option = job["command"].index("--age-kernel-minimum-exact-sex-donors")
        assert job["command"][option + 1] == "20"
        assert job["minimum_exact_sex_donors"] == 20
        assert any(
            artifact.get("sha256") == hashlib.sha256(plan_path.read_bytes()).hexdigest()
            for artifact in job["upstream_artifacts"]
        )
    for job in generated["empirical_scoring"]:
        option = job["command"].index("--minimum-exact-sex-donors")
        assert job["command"][option + 1] == "20"
        assert job["minimum_exact_sex_donors"] == 20
    assert all(
        "--fine-type-universe" in job["command"]
        for role in ("reference", "validation")
        for job in generated[f"aggregate_{role}"]
    )
    selector = generated["select_transferable"][0]
    assert selector["runnable"] is True
    assert selector["selector_input_role"] == "reference"
    assert selector["query_artifacts_allowed"] is False
    assert selector["command"].count("--required-seed") == 2
    assert all(
        "/query/" not in artifact["path"] for artifact in selector["upstream_artifacts"]
    )
    assert all(
        job["expected_outputs"]
        for rows in generated.values()
        for job in rows
        if job["runnable"]
    )
    assert all(
        not job["expected_outputs"]
        for rows in generated.values()
        for job in rows
        if not job["runnable"]
    )
    assert all(
        "not a healthy-reference score standard error" in job["uncertainty_estimand"]
        for job in generated["bootstrap_cell"]
    )


def test_all_healthy_fit_rows_have_required_cli_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream = _load_script(
        "downstream_all_healthy_test", "scripts/generate_downstream_jobs.py"
    )
    model = _model(tmp_path, "final-model")
    plan, plan_path = _plan(tmp_path, [model])
    reference = _project_job(tmp_path, "final-model", "reference", 42)

    def all_healthy_state(job: dict[str, Any], gp_ids: list[str]):
        state, reason = _fake_projection_state(job, gp_ids)
        state["payload"].update(
            {
                "reference_design": "all_healthy",
                "heldout_dataset": None,
                "fold_id": "all_healthy",
                "datasets": ["train_a", "train_b", "query_cohort"],
            }
        )
        return state, reason

    monkeypatch.setattr(downstream, "_projection_state", all_healthy_state)
    generated = downstream.generate_pass1(
        [reference],
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    assert generated["fit_reference"]
    assert all(
        "--final-all-healthy" in job["command"] for job in generated["fit_reference"]
    )
    assert generated["select_transferable"][0]["runnable"] is False


def test_candidate_plan_requires_hash_bound_exact_sex_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream = _load_script(
        "downstream_exact_sex_validation_test",
        "scripts/generate_downstream_jobs.py",
    )
    model = _model(tmp_path, "train-a")
    plan, plan_path = _plan(tmp_path, [model])
    monkeypatch.setattr(downstream, "_projection_state", _fake_projection_state)
    del plan["healthy_reference"]

    with pytest.raises(ValueError, match="healthy_reference settings object"):
        downstream.generate_pass1(
            [_project_job(tmp_path, "train-a", "reference", 42)],
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        )


def test_pass2_reuses_posttraining_allowlist_and_emits_selected_query_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream = _load_script(
        "downstream_pass2_test", "scripts/generate_downstream_jobs.py"
    )
    posttraining = _load_script(
        "posttraining_allowlist_compatibility_test",
        "scripts/generate_post_training_jobs.py",
    )
    model = _model(
        tmp_path,
        "train-a",
        endpoints=[
            {"gp_id": "GP_A", "fine_type": "Naive B"},
            {"gp_id": "GP_B", "fine_type": "Memory B"},
        ],
    )
    plan, plan_path = _plan(tmp_path, [model])
    selected_path = tmp_path / "selected_tripso_gps.json"
    selected_path.write_text("selector artifact placeholder", encoding="utf-8")
    selected_file_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    selected = {
        "schema_version": "immune-health-tripso-gp-selection/v1",
        "lineage": "B cells",
        "fold_id": "lodo_query_cohort",
        "heldout_dataset": "query_cohort",
        "required_seeds": [42, 43],
        "selected_endpoints": [
            {"lineage": "B cells", "fine_type": "Naive B", "gp_id": "GP_A"}
        ],
    }
    monkeypatch.setattr(
        downstream, "validate_tripso_gp_selection_manifest", lambda _: selected
    )
    monkeypatch.setattr(downstream, "_projection_state", _fake_projection_state)

    allowlist = {
        "schema_version": "immune-health-outer-query-evaluation-allowlist/v1",
        "selection_basis": "inner_validation_only",
        "outer_query_data_consulted_for_selection": False,
        "selected_training_job_ids": ["train-a"],
        "allowed_parent_training_job_ids": ["train-a"],
        "selection_manifest_sha256": selected_file_sha,
        "outer_query_evaluation_only": True,
        "outer_query_results_used_for_selection": False,
        "query_derived_evidence_used_for_selection": False,
    }
    allowlist["manifest_sha256"] = downstream._canonical_hash(allowlist)
    allowlist_path = tmp_path / "query_allowlist.json"
    allowlist_path.write_text(json.dumps(allowlist), encoding="utf-8")
    selected_ids, _ = posttraining._read_outer_query_allowlist(allowlist_path)
    assert selected_ids == {"train-a"}

    generated = downstream.generate_pass2(
        [_project_job(tmp_path, "train-a", "query", 42)],
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        selected_path=selected_path,
        allowlist=allowlist,
        allowlist_path=allowlist_path,
        allowlist_file_sha256=hashlib.sha256(allowlist_path.read_bytes()).hexdigest(),
    )
    assert len(generated["convert_query"]) == 1
    assert generated["convert_query"][0]["candidate_gp_ids"] == ["GP_A"]
    assert len(generated["aggregate_query"]) == 1
    assert len(generated["endpoint_query"]) == 1
    assert len(generated["score_query"]) == 2
    assert len(generated["empirical_scoring"]) == 2
    assert generated["convert_reference"] == []
    assert generated["score_validation"] == []
    assert all(job["outer_query_evaluation_only"] for job in generated["score_query"])
    assert all(
        any(
            artifact.get("json_require", {}).get("schema_version")
            == "immune-health-tripso-gp-selection/v1"
            for artifact in job["upstream_artifacts"]
        )
        for job in generated["score_query"]
    )

    divergent = dict(allowlist)
    divergent["allowed_parent_training_job_ids"] = ["different-job"]
    with pytest.raises(ValueError, match="must exactly equal"):
        downstream.generate_pass2(
            [_project_job(tmp_path, "train-a", "query", 42)],
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256="b" * 64,
            selected_path=selected_path,
            allowlist=divergent,
            allowlist_path=allowlist_path,
            allowlist_file_sha256="c" * 64,
        )
