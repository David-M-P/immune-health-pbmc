from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_generator():
    path = REPOSITORY_ROOT / "scripts" / "generate_reference_prep_jobs.py"
    spec = importlib.util.spec_from_file_location("reference_prep_jobs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tripso_generator():
    path = REPOSITORY_ROOT / "scripts" / "generate_job_manifests.py"
    spec = importlib.util.spec_from_file_location("tripso_job_manifests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def test_reference_prep_jobs_materialize_both_exact_hvg_variants() -> None:
    generator = _load_generator()
    config = generator.load_config(
        REPOSITORY_ROOT / "configs" / "experiments" / "reference_preparation.yaml"
    )
    jobs = generator.generate_jobs(config)
    assert len(jobs["visits"]) == 1
    assert len(jobs["final_fold"]) == 1
    assert len(jobs["features"]) == 30
    assert len(jobs["materialize"]) == 160
    assert len(jobs["lodo_tokenize"]) == 150
    assert len(jobs["lodo_bind"]) == 50
    assert len(jobs["final_tokenize"]) == 10
    assert len(jobs["final_bind"]) == 10
    assert {job["hvg_size"] for job in jobs["materialize"]} == {3000, 9000}
    assert {job["preparation_role"] for job in jobs["materialize"]} == {
        "adaptation",
        "validation",
        "query",
    }
    for job in jobs["materialize"]:
        size = job["hvg_size"]
        assert f"/hvg{size}/" in job["output_dir"]
        assert job["cell_downsampling"] is False

    lodo_features = [
        job for job in jobs["features"] if job.get("heldout_dataset") is not None
    ]
    assert len(lodo_features) == 25
    assert all("--global-one-visit-query" in job["command"] for job in lodo_features)
    assert all(
        job["command"][job["command"].index("--inner-validation-fold") + 1] == "0"
        for job in lodo_features
    )
    assert all(job["selection_uses_outer_query"] is False for job in lodo_features)
    for job in jobs["features"]:
        outputs = set(job["expected_outputs"])
        assert any(path.endswith("/model_genes_hvg3000.txt") for path in outputs)
        assert any(path.endswith("/model_genes_hvg9000.txt") for path in outputs)
        required_feature_sidecars = {
            "feature_manifest.json",
            "hvg3000.txt",
            "hvg9000.txt",
            "gene_programs_filtered.gmt",
            "gpdb_filtered.csv",
            "gene_program_terms.tsv",
            "cell_metadata.parquet",
            "training_gene_statistics.parquet",
            "gene_program_filter_report.parquet",
            "gene_program_gene_support.parquet",
            "hvg_scores.parquet",
            "hvg_dataset_scores.parquet",
            "model_gene_membership.parquet",
            "simple_gp_donor_scores.parquet",
            "simple_gp_age_effects.parquet",
            "simple_gp_transferability.parquet",
            "projection_gp_candidates.tsv",
            "projection_gp_candidates.json",
        }
        assert required_feature_sidecars <= {Path(path).name for path in outputs}

    lodo_materialize = [
        job for job in jobs["materialize"] if job.get("heldout_dataset") is not None
    ]
    assert len(lodo_materialize) == 150
    assert {job["preparation_role"] for job in lodo_materialize} == {
        "adaptation",
        "validation",
        "query",
    }
    assert len(jobs["lodo_tokenize"]) == len(lodo_materialize)
    assert {job["preparation_role"] for job in jobs["lodo_tokenize"]} == {
        "adaptation",
        "validation",
        "query",
    }
    assert all(
        "--inner-validation-fold" in job["command"]
        and "--inner-fold-column" in job["command"]
        and job["selection_uses_outer_query"] is False
        for job in jobs["lodo_bind"]
    )

    final_features = [
        job for job in jobs["features"] if job.get("reference_design") == "all_healthy"
    ]
    assert len(final_features) == 5
    assert all(job["heldout_dataset"] is None for job in final_features)
    assert all("--reference-design" in job["command"] for job in final_features)
    for job in jobs["final_bind"]:
        assert job["stage3_compatible_path"] is True
        assert "/all_healthy/hvg" in job["expected_outputs"][0]
        assert "--held-out-dataset" not in job["command"]
        assert job["command"][job["command"].index("--reference-design") + 1] == (
            "all_healthy"
        )


def test_every_stage1_fold_input_has_exactly_one_lodo_prep_producer() -> None:
    reference_generator = _load_generator()
    reference = reference_generator.generate_jobs(
        reference_generator.load_config(
            REPOSITORY_ROOT / "configs" / "experiments" / "reference_preparation.yaml"
        )
    )
    tripso_generator = _load_tripso_generator()
    tripso = tripso_generator.load_experiment(
        REPOSITORY_ROOT / "configs" / "experiments" / "tripso_lodo.yaml"
    )
    stage1 = tripso_generator.generate_jobs(tripso, "stage1", base_seed=42)
    required = {
        artifact["path"]
        for job in stage1
        if job["runnable"]
        for artifact in job["upstream_artifacts"]
        if artifact.get("json_require", {}).get("schema_version")
        == "immune-health-tripso-fold-input/v1"
    }
    produced = Counter(
        output for job in reference["lodo_bind"] for output in job["expected_outputs"]
    )
    assert len(required) == 50
    assert required == set(produced)
    assert set(produced.values()) == {1}


def test_reference_prep_example_environment_resolves_every_job_path(
    monkeypatch,
) -> None:
    generator = _load_generator()
    config = generator.load_config(
        REPOSITORY_ROOT / "configs" / "experiments" / "reference_preparation.yaml"
    )
    cluster = yaml.safe_load(
        (
            REPOSITORY_ROOT / "configs" / "slurm" / "reference_prep_cpu.example.yaml"
        ).read_text()
    )
    assert "OUTPUT_ROOT" in cluster["environment"]
    values = {
        "PROJECT_ROOT": "/cluster/project/repository",
        "DATA_ROOT": "/cluster/data/intermediate_data",
        "REFERENCE_PREP_OUTPUT_ROOT": "/cluster/work/reference_prep",
        "OUTPUT_ROOT": "/cluster/work/immune_health",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    jobs = generator.generate_jobs(config)
    for job in (row for rows in jobs.values() for row in rows):
        expanded = _expand(job)
        serialized = str(expanded)
        assert "${" not in serialized
        assert "<SET_" not in serialized
        for path in (
            expanded["output_dir"],
            expanded["working_directory"],
            *expanded["expected_outputs"],
            *(item["path"] for item in expanded["upstream_artifacts"]),
        ):
            assert Path(path).is_absolute(), (job["job_id"], path)
