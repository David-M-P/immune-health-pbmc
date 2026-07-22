from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from immune_health.tripso_adapter import (
    FrozenProjectionError,
    TripsoContractError,
    TripsoTrainingSpec,
    build_model_artifact_manifest,
    build_training_call,
    load_fold_input_manifest,
    make_identifiers,
    prepare_fold_input_manifest,
    project_frozen,
    run_mock_projection_smoke,
    run_tripso_training,
    validate_checkpoint_manifest,
    validate_fold_rows,
    validate_tripso_resources,
)
from immune_health.tripso_adapter import geneformer as geneformer_module
from immune_health.tripso_adapter.contracts import canonical_json_hash, sha256_path
from immune_health.tripso_adapter.geneformer import (
    EXPECTED_GENEFORMER_CONFIG,
    VALIDATED_GENEFORMER_MODEL,
    geneformer_runtime_compatibility,
    validate_geneformer_root,
)
from immune_health.tripso_adapter.training import _sequence_and_identifier_contract

REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fold_resources(tmp_path: Path) -> dict[str, Path]:
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    gp = tmp_path / "gp.csv"
    gp.write_text("gp_a\nGENE1\n", encoding="utf-8")
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("ENSG000001\n", encoding="utf-8")
    candidates = tmp_path / "projection_gp_candidates.json"
    candidate_payload = {
        "schema_version": "immune-health-projection-gp-candidates/v1",
        "selection_level": "donor_lineage_pseudobulk",
        "query_data_consulted": False,
        "program_ids": ["gp_a"],
        "program_ids_ordered_sha256": canonical_json_hash(["gp_a"]),
        "binding": {"gpdb_sha256": sha256_path(gp)},
    }
    candidate_payload["manifest_content_sha256"] = canonical_json_hash(
        candidate_payload
    )
    candidates.write_text(json.dumps(candidate_payload), encoding="utf-8")
    return {
        "tokenized": tokenized,
        "gp": gp,
        "vocabulary": vocabulary,
        "candidates": candidates,
    }


def _fold_rows() -> list[dict[str, object]]:
    rows = []
    for dataset, donor, sample, eligible in (
        ("aidav2", "d1", "s1", True),
        ("onek1k", "d2", "pool7", True),
        ("onek1k", "d3", "pool7", True),
        ("terekhova", "d4", "visit1", False),
        ("terekhova", "d4", "visit2", False),
    ):
        rows.append(
            {
                "dataset": dataset,
                "donor_id": donor,
                "sample_id": sample,
                **make_identifiers(dataset, donor, sample),
                "outer_role": "query" if dataset == "terekhova" else "reference",
                "eligible_for_reference_fitting": eligible,
            }
        )
    return rows


def _make_fold_manifest(tmp_path: Path) -> Path:
    resources = _fold_resources(tmp_path)
    path = tmp_path / "fold_input.json"
    prepare_fold_input_manifest(
        rows=_fold_rows(),
        output_path=path,
        fold_id="lodo_terekhova",
        held_out_dataset="terekhova",
        lineage="B cells",
        tokenized_dataset_path=resources["tokenized"],
        gp_library_path=resources["gp"],
        gene_vocabulary_path=resources["vocabulary"],
        projection_gp_candidates_path=resources["candidates"],
        partition_column="outer_role",
    )
    return path


def _make_scope_proven_fold_manifest(tmp_path: Path) -> Path:
    resources = _fold_resources(tmp_path)
    path = tmp_path / "scope_proven_fold_input.json"
    prepare_fold_input_manifest(
        rows=_fold_rows(),
        output_path=path,
        fold_id="lodo_terekhova",
        held_out_dataset="terekhova",
        lineage="B cells",
        tokenized_dataset_path=resources["tokenized"],
        gp_library_path=resources["gp"],
        gene_vocabulary_path=resources["vocabulary"],
        projection_gp_candidates_path=resources["candidates"],
        tokenized_biological_unit_ids=(
            "aidav2::d1",
            "onek1k::d2",
            "onek1k::d3",
        ),
        partition_column="outer_role",
    )
    return path


def test_approved_identifier_contract_is_donor_specific() -> None:
    first = make_identifiers("onek1k", "donor_a", "pool_1")
    second = make_identifiers("onek1k", "donor_b", "pool_1")

    assert first["biological_unit_id"] == "onek1k::donor_a"
    assert first["source_observation_id"] == second["source_observation_id"]
    assert first["observation_id"] != second["observation_id"]
    assert first["observation_id"] == "onek1k::donor_a::pool_1"


def test_fold_validation_accepts_repeated_samples_but_not_donor_leakage() -> None:
    validated = validate_fold_rows(
        _fold_rows(), "terekhova", partition_column="outer_role"
    )
    assert validated.adaptation_donors == (
        "aidav2::d1",
        "onek1k::d2",
        "onek1k::d3",
    )
    assert validated.query_donors == ("terekhova::d4",)
    assert len(validated.observations) == 5

    leaking = _fold_rows()
    leaking.append(
        {
            "dataset": "terekhova",
            "donor_id": "d4",
            "sample_id": "visit3",
            "outer_role": "query",
            "eligible_for_reference_fitting": True,
        }
    )
    with pytest.raises(TripsoContractError, match="Held-out dataset"):
        validate_fold_rows(leaking, "terekhova", partition_column="outer_role")


def test_fold_manifest_binds_training_donors_and_resources(tmp_path: Path) -> None:
    path = _make_fold_manifest(tmp_path)
    manifest = load_fold_input_manifest(path)

    assert manifest["biological_split_unit"] == "donor"
    assert manifest["vendor_internal_cell_split_scope"] == "adaptation_donors_only"
    assert "terekhova::d4" not in manifest["adaptation_biological_unit_ids"]
    assert manifest["query_biological_unit_ids"] == ["terekhova::d4"]
    assert len(manifest["hashes"]["gp_library_sha256"]) == 64
    assert manifest["tokenized_dataset_scope_validation"]["status"] == ("not_performed")

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["biological_split_unit"] = "cell"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TripsoContractError, match="donor biological splits"):
        load_fold_input_manifest(path)


def test_tokenized_dataset_requires_physical_donor_scope_proof(
    tmp_path: Path,
) -> None:
    resources = _fold_resources(tmp_path)
    path = tmp_path / "scope_proven.json"
    adaptation = ("aidav2::d1", "onek1k::d2", "onek1k::d3")
    manifest = prepare_fold_input_manifest(
        rows=_fold_rows(),
        output_path=path,
        fold_id="lodo_terekhova",
        held_out_dataset="terekhova",
        lineage="B cells",
        tokenized_dataset_path=resources["tokenized"],
        gp_library_path=resources["gp"],
        gene_vocabulary_path=resources["vocabulary"],
        projection_gp_candidates_path=resources["candidates"],
        tokenized_biological_unit_ids=adaptation,
        partition_column="outer_role",
    )
    assert manifest["tokenized_dataset_scope_validation"]["status"] == "passed"

    with pytest.raises(TripsoContractError, match="donor scope"):
        prepare_fold_input_manifest(
            rows=_fold_rows(),
            output_path=tmp_path / "leaking.json",
            fold_id="lodo_terekhova",
            held_out_dataset="terekhova",
            lineage="B cells",
            tokenized_dataset_path=resources["tokenized"],
            gp_library_path=resources["gp"],
            gene_vocabulary_path=resources["vocabulary"],
            projection_gp_candidates_path=resources["candidates"],
            tokenized_biological_unit_ids=(*adaptation, "terekhova::d4"),
            partition_column="outer_role",
        )


def test_training_wrapper_refuses_cell_split_and_vendor_weighted_sampler(
    tmp_path: Path,
) -> None:
    fold_path = _make_fold_manifest(tmp_path)
    with pytest.raises(TripsoContractError, match="Biological split unit"):
        build_training_call(
            TripsoTrainingSpec(
                fold_input_manifest_path=fold_path,
                output_dir=tmp_path / "model",
                model_type="Base",
                seed=7,
                parameters={"biological_split_unit": "cell"},
            )
        )
    with pytest.raises(TripsoContractError, match="not donor-hierarchical"):
        build_training_call(
            TripsoTrainingSpec(
                fold_input_manifest_path=fold_path,
                output_dir=tmp_path / "model",
                model_type="Base",
                seed=7,
                parameters={"sampler": "weighted"},
            )
        )

    call, invocation = build_training_call(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_path,
            output_dir=tmp_path / "model",
            model_type="Base",
            seed=7,
        )
    )
    assert call["seed"] == 7
    assert call["model_type"] == "Base"
    assert invocation["adaptation_biological_unit_ids"] == [
        "aidav2::d1",
        "onek1k::d2",
        "onek1k::d3",
    ]
    with pytest.raises(TripsoContractError, match="physical tokenized dataset"):
        run_tripso_training(
            TripsoTrainingSpec(
                fold_input_manifest_path=fold_path,
                output_dir=tmp_path / "real_model",
                model_type="Base",
                seed=7,
            ),
            train_fn=lambda **_: None,
        )


def test_training_replaces_vendor_wandb_surface_with_local_csv(tmp_path: Path) -> None:
    fold_path = _make_scope_proven_fold_manifest(tmp_path)
    output_dir = tmp_path / "offline_model"
    network_calls: list[str] = []

    class FakeCSVLogger:
        def __init__(self, *, save_dir: str, name: str, version: str) -> None:
            metric_dir = Path(save_dir) / name / version
            metric_dir.mkdir(parents=True)
            (metric_dir / "metrics.csv").write_text(
                "epoch,step,train/loss\n0,0,2.5\n0,1,1.25\n", encoding="utf-8"
            )

    def external_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        network_calls.append("called")
        raise AssertionError("external tracking must not be called")

    class FakeDataModule:
        """Expose the inspected vendor global for this tracking-only test."""

    surface: dict[str, object] = {
        "Path": Path,
        "pl": SimpleNamespace(loggers=SimpleNamespace(CSVLogger=FakeCSVLogger)),
        "configure_save_id": external_call,
        "configure_wandb": external_call,
        "configure_logger": external_call,
        "rank_zero_only": SimpleNamespace(rank=0),
        "wandb_api_readback": external_call,
        "txDataModule": FakeDataModule,
    }
    exec(
        """
def fake_vendor_train(dataset_path, gpdb_path, output_dir, model_type, seed, **kwargs):
    del dataset_path, gpdb_path, kwargs
    args = {"output_dir": output_dir, "model_type": model_type, "seed": seed}
    save_id = configure_save_id(args)
    configure_wandb(args, save_id)
    logger = configure_logger(args)
    assert logger is not None
    checkpoint = Path(output_dir) / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"offline-checkpoint")
    if rank_zero_only.rank == 0:
        wandb_api_readback()
    return save_id
""",
        surface,
    )
    fake_vendor_train = surface["fake_vendor_train"]
    originals = {
        name: surface[name]
        for name in (
            "configure_save_id",
            "configure_wandb",
            "configure_logger",
            "rank_zero_only",
        )
    }

    result = run_tripso_training(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_path,
            output_dir=output_dir,
            model_type="Base",
            seed=17,
            parameters={"fm_encoder_pkg": "from_scratch"},
        ),
        train_fn=fake_vendor_train,
    )

    assert result == "immune_health_Base_seed_17"
    assert network_calls == []
    assert all(surface[name] is original for name, original in originals.items())
    canonical_metrics = output_dir / "training_metrics.csv"
    assert canonical_metrics.is_file()
    completion = json.loads(
        (output_dir / "tripso_training_result.json").read_text(encoding="utf-8")
    )
    tracking = completion["experiment_tracking"]
    assert tracking["network_required"] is False
    assert tracking["surface_status"] == "inspected_vendor_tracking_replaced"
    assert tracking["metrics"]["status"] == "written"
    assert tracking["metrics"]["n_rows"] == 2
    assert tracking["metrics"]["last_logged_values"]["train/loss"] == 1.25


def test_training_call_installs_project_sampler_and_fold_vocabulary(
    tmp_path: Path,
) -> None:
    fold_path = _make_fold_manifest(tmp_path)
    call, invocation = build_training_call(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_path,
            output_dir=tmp_path / "model",
            model_type="Base",
            seed=11,
            parameters={
                "batch_size": 8,
                "gene_format": "ensembl",
                "fm_encoder_pkg": "from_scratch",
                "all_genes_from_fold_vocabulary": True,
                "bert_config": {
                    "tokenization_input_size": 4096,
                    "hidden_size": 512,
                },
                "sampler": None,
                "project_sampler": {
                    "mode": "hybrid",
                    "alpha": 0.5,
                    "fine_type_lambda": 0.7,
                    "n_cells_per_epoch": 16,
                },
            },
        )
    )

    assert "project_sampler" not in call
    assert "all_genes_from_fold_vocabulary" not in call
    assert call["all_genes"] == ["ENSG000001"]
    assert call["bert_config"]["max_seq_len"] == 1
    assert invocation["training_sampler_backend"].endswith("HierarchicalCellSampler")
    assert invocation["project_sampler"]["lineage"] == "B cells"
    assert invocation["project_sampler"]["batch_size"] == 8
    assert (
        invocation["sequence_and_identifier_contract"]["gene_identifier_format"]
        == "ensembl"
    )

    with pytest.raises(TripsoContractError, match="predominantly ENSG"):
        build_training_call(
            TripsoTrainingSpec(
                fold_input_manifest_path=fold_path,
                output_dir=tmp_path / "wrong_gene_format",
                model_type="Base",
                seed=11,
                parameters={
                    "gene_format": "symbol",
                    "fm_encoder_pkg": "from_scratch",
                    "all_genes_from_fold_vocabulary": True,
                    "bert_config": {"tokenization_input_size": 4096},
                },
            )
        )


def test_native_manifest_job_disables_project_sampler(tmp_path: Path) -> None:
    from immune_health.cli.main import _resolve_training_request

    fold_path = _make_fold_manifest(tmp_path)
    parameters_path = tmp_path / "parameters.json"
    parameters_path.write_text(
        json.dumps(
            {
                "fm_encoder_pkg": "from_scratch",
                "project_sampler": {
                    "enabled": True,
                    "mode": "hybrid",
                    "alpha": 0.5,
                    "fine_type_lambda": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "native_model"
    job_path = tmp_path / "native_job.json"
    job_path.write_text(
        json.dumps(
            {
                "schema_version": "immune-health-slurm-job/v1",
                "runnable": True,
                "seed": 42,
                "output_dir": str(output_dir),
                "sampler_mode": "native_all_cells",
                "sampling_backend": "vendor",
                "project_sampler_enabled": False,
                "project_sampler_mode": None,
                "alpha": None,
                "fine_type_lambda": None,
                "hvg_size": 3000,
                "feature_set": "hvg3000_plus_gp",
                "upstream_artifacts": [
                    {
                        "path": str(fold_path),
                        "json_require": {
                            "schema_version": ("immune-health-tripso-fold-input/v1")
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        job_spec=job_path,
        seed=42,
        model_type="Base",
        parameters_json=parameters_path,
        fold_input=None,
        output_dir=None,
        sampler_mode="hybrid",
        alpha=0.5,
        fine_type_lambda=0.7,
        base_model_dir=None,
    )

    _, invocation, resolved_output, job = _resolve_training_request(args)

    assert resolved_output == output_dir
    assert job["sampler_mode"] == "native_all_cells"
    assert invocation["project_sampler"]["enabled"] is False
    assert invocation["training_sampler_backend"] == "vendor"


def test_sequence_contract_flags_large_gene_universe_truncation() -> None:
    genes = [f"ENSG{index:011d}" for index in range(9000)]
    contract = _sequence_and_identifier_contract(
        {
            "fm_encoder_pkg": "from_scratch",
            "gene_format": "ensembl",
            "all_genes": genes,
            "bert_config": {
                "tokenization_input_size": 4096,
                "max_seq_len": len(genes),
            },
        }
    )
    assert contract["n_fold_gene_universe"] == 9000
    assert contract["per_cell_rank_token_limit"] == 4096
    assert contract["per_cell_truncation_expected"] is True


def test_geneformer_assets_are_exactly_pinned_and_forward_patch_is_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Geneformer"
    model = root / VALIDATED_GENEFORMER_MODEL
    model.mkdir(parents=True)
    config_path = model / "config.json"
    config_path.write_text(json.dumps(EXPECTED_GENEFORMER_CONFIG), encoding="utf-8")
    weights_path = model / "model.safetensors"
    weights_path.write_bytes(b"test-weights")
    from immune_health.tripso_adapter.contracts import sha256_path

    monkeypatch.setattr(
        geneformer_module,
        "EXPECTED_GENEFORMER_HASHES",
        {
            "config.json": sha256_path(config_path),
            "model.safetensors": sha256_path(weights_path),
        },
    )
    validation = validate_geneformer_root(root)
    assert validation["passed"] is True
    assert validation["hashes_pinned"] is True

    fold_path = _make_fold_manifest(tmp_path)
    call, invocation = build_training_call(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_path,
            output_dir=tmp_path / "full_geneformer_model",
            model_type="Base",
            seed=2,
            parameters={
                "fm_encoder_pkg": "geneformer",
                "fm_encoder_name": VALIDATED_GENEFORMER_MODEL,
                "geneformer_root": str(root),
            },
        )
    )
    assert call["calc_gene_loss"] is False
    assert call["calc_gp_loss"] is True
    assert call["warmup"] == 0
    assert "geneformer_root" not in call
    assert invocation["geneformer_validation"]["passed"] is True

    class FakeWrapper:
        def forward(self, input_dataset, masking):
            return input_dataset, masking

    class FakeBase:
        pass

    test_module = sys.modules[__name__]
    monkeypatch.setattr(test_module, "gfWrapper", FakeWrapper, raising=False)
    old_root = lambda: "/hard-coded/original"  # noqa: E731
    monkeypatch.setitem(globals(), "get_gf_repo", old_root)
    monkeypatch.setitem(globals(), "gpTransformerBase", FakeBase)

    def fake_train(
        dataset_path,
        gpdb_path,
        output_dir,
        model_type,
        seed,
    ):
        del dataset_path, gpdb_path, output_dir, model_type, seed

    original_forward = FakeWrapper.forward
    with geneformer_runtime_compatibility(fake_train, geneformer_root=root):
        assert globals()["get_gf_repo"]() == str(root.resolve())
        assert FakeWrapper().forward(
            "input", masking=True, return_mean_non_padding=True
        ) == ("input", True)
    assert globals()["get_gf_repo"]() == "/hard-coded/original"
    assert FakeWrapper.forward is original_forward


class _Parameter:
    def __init__(self, value: int) -> None:
        self.value = value
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> "_Parameter":
        self.requires_grad = value
        return self


class _Model:
    def __init__(self) -> None:
        self.weight = _Parameter(4)
        self.training = True

    def state_dict(self) -> dict[str, int]:
        return {"weight": self.weight.value}

    def parameters(self) -> list[_Parameter]:
        return [self.weight]

    def eval(self) -> "_Model":
        self.training = False
        return self

    def __call__(self, value: int) -> int:
        return self.weight.value * value


def test_frozen_projection_has_no_optimizer_or_state_update() -> None:
    model = _Model()
    assert project_frozen(model, [1, 2]) == [4, 8]
    assert model.training is False
    assert model.weight.requires_grad is False
    assert model.weight.value == 4

    with pytest.raises(FrozenProjectionError, match="optimizer"):
        project_frozen(model, [1], optimizer=object())

    mutating_model = _Model()

    def mutating_forward(active_model: _Model, value: int) -> int:
        active_model.weight.value += 1
        return value

    with pytest.raises(FrozenProjectionError, match="state"):
        project_frozen(mutating_model, [1], forward=mutating_forward)


def test_mock_smoke_never_claims_real_tripso_success() -> None:
    result = run_mock_projection_smoke()
    assert result["mock_adapter_smoke_passed"] is True
    assert result["real_tripso_import_tested"] is False
    assert result["real_tripso_training_smoke_passed"] is False


def test_resource_validation_never_synthesizes_vendor_assets(tmp_path: Path) -> None:
    vendor = tmp_path / "tripso"
    (vendor / "tripso" / "Utils").mkdir(parents=True)
    (vendor / "setup.py").write_text("", encoding="utf-8")
    (vendor / "requirements.txt").write_text("numpy==1.25.0\n", encoding="utf-8")
    gp = tmp_path / "gp.csv"
    gp.write_text("gp\nG1\n", encoding="utf-8")
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("G1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="never synthesized"):
        validate_tripso_resources(
            vendor_root=vendor,
            gp_library_path=gp,
            gene_vocabulary_path=vocabulary,
        )


def test_model_manifest_records_checkpoint_and_resource_hashes(tmp_path: Path) -> None:
    fold_path = _make_fold_manifest(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    resources = {
        "gp": tmp_path / "gp.csv",
        "vocabulary": tmp_path / "vocabulary.txt",
    }
    output = tmp_path / "model_manifest.json"
    build_model_artifact_manifest(
        output_path=output,
        repo_root=REPOSITORY_ROOT,
        vendor_root=REPOSITORY_ROOT / "tripso_code" / "tripso",
        fold_input_manifest_path=fold_path,
        checkpoint_path=checkpoint,
        fold_id="lodo_terekhova",
        held_out_dataset="terekhova",
        lineage="B cells",
        model_type="Base",
        sampler_mode="hybrid",
        alpha=0.5,
        fine_type_lambda=0.7,
        seed=42,
        gp_library_path=resources["gp"],
        gene_vocabulary_path=resources["vocabulary"],
        training_metrics={"train/loss": 1.0},
    )
    validated = validate_checkpoint_manifest(output)
    assert validated["hashes"]["checkpoint_sha256"]
    assert (
        validated["hashes"]["projection_gp_candidates_sha256"]
        == json.loads(fold_path.read_text(encoding="utf-8"))["hashes"][
            "projection_gp_candidates_sha256"
        ]
    )
    assert validated["hashes"]["projection_gp_program_ids_ordered_sha256"]
    assert validated["paths"]["projection_gp_candidates"]
    assert validated["repository"]["vendor_tree_hash"]
    assert validated["sampler"]["fine_type_lambda"] == 0.7


def test_stage_manifests_are_deterministic_and_pending_choices_are_not_runnable() -> (
    None
):
    generator = _load_script(
        "generate_job_manifests_test", "scripts/generate_job_manifests.py"
    )
    config = generator.load_experiment(
        REPOSITORY_ROOT / "configs" / "experiments" / "tripso_lodo.yaml"
    )
    stage1 = generator.generate_jobs(config, "stage1", base_seed=42)
    stage2 = generator.generate_jobs(config, "stage2", base_seed=42)
    stage3 = generator.generate_jobs(config, "stage3", base_seed=42)

    assert len(stage1) == 5 * 5 * 3 * 2
    assert all(job["runnable"] for job in stage1)
    assert len({job["job_id"] for job in stage1}) == len(stage1)
    assert stage1 == generator.generate_jobs(config, "stage1", base_seed=42)
    assert {job["hvg_size"] for job in stage1} == {3000, 9000}
    assert {job["feature_set"] for job in stage1} == {
        "hvg3000_plus_gp",
        "hvg9000_plus_gp",
    }
    assert {job["sampler_mode"] for job in stage1} == {
        "native_all_cells",
        "donor_uniform_observed",
        "hybrid",
    }
    native = [job for job in stage1 if job["sampler_mode"] == "native_all_cells"]
    assert native
    assert all(job["sampling_backend"] == "vendor" for job in native)
    assert all(job["project_sampler_enabled"] is False for job in native)
    assert all(job["alpha"] is None for job in native)
    donor_uniform = [
        job for job in stage1 if job["sampler_mode"] == "donor_uniform_observed"
    ]
    assert all(job["alpha"] == 1.0 for job in donor_uniform)
    assert all(job["fine_type_lambda"] == 1.0 for job in donor_uniform)
    assert all(
        job["project_sampler_mode"] == "observed_proportions" for job in donor_uniform
    )
    assert len(stage2) == 5 * 5 * 2 * 2
    assert {job["seed"] for job in stage2} == {43, 44}
    assert all(job["reused_seed_offsets"] == [0] for job in stage2)
    assert all(job["extends_stage"] == "stage1" for job in stage2)
    assert len(stage3) == 5 * 1 * 1 * 5
    assert not any(job["runnable"] for job in stage2 + stage3)
    assert all(job["command"] == [] for job in stage2 + stage3)

    retraining = json.loads(json.dumps(config))
    retraining["stages"]["stage2"]["seed_offsets"] = [0, 1, 2]
    with pytest.raises(ValueError, match="retrain reused seed offsets"):
        generator.generate_jobs(retraining, "stage2", base_seed=42)


def test_every_runnable_stage1_and_stage3_model_has_role_aware_posttrain_jobs() -> None:
    training_generator = _load_script(
        "generate_job_manifests_for_posttrain_test",
        "scripts/generate_job_manifests.py",
    )
    posttrain_generator = _load_script(
        "generate_post_training_jobs_test",
        "scripts/generate_post_training_jobs.py",
    )
    config = training_generator.load_experiment(
        REPOSITORY_ROOT / "configs" / "experiments" / "tripso_lodo.yaml"
    )
    stage1 = training_generator.generate_jobs(config, "stage1", base_seed=42)
    stage1_post = posttrain_generator.generate_post_training_jobs(stage1)
    expected_stage1_ids = {job["job_id"] for job in stage1 if job["runnable"]}
    assert len(expected_stage1_ids) == 150
    for phase in (
        "bind_reference",
        "bind_validation",
        "project_reference",
        "project_validation",
    ):
        rows = stage1_post[phase]
        assert {job["parent_training_job_id"] for job in rows} == expected_stage1_ids
        assert len(rows) == 150
        assert all(job["adapt"] is False for job in rows)
        assert all(job["optimizer_allowed"] is False for job in rows)
        assert all(
            job["runner_output_separate_from_projection_data"] is True for job in rows
        )
    assert all(
        job["expected_outputs"][0].endswith("/embeddings/reference_set")
        for job in stage1_post["project_reference"]
    )
    assert all(
        job["expected_outputs"][0].endswith("/embeddings/validation_set")
        for job in stage1_post["project_validation"]
    )
    assert stage1_post["bind_query"] == []
    assert stage1_post["project_query"] == []
    assert all(
        job["expected_outputs"][1].endswith("/projection_output_manifest.json")
        for job in (
            stage1_post["project_reference"] + stage1_post["project_validation"]
        )
    )
    assert all(
        "--use-fold-bound-gp-candidates" in job["command"]
        and job["gp_projection_policy"] == "fold_bound_training_candidates"
        for job in stage1_post["bind_reference"] + stage1_post["bind_validation"]
    )
    assert all(
        job["maximum_projected_bytes"] == 250 * 1024**3
        for rows in stage1_post.values()
        for job in rows
    )
    assert all(
        any("/adaptation/tokenization_manifest.json" in part for part in job["command"])
        for job in stage1_post["bind_reference"]
    )
    assert all(
        any("/validation/tokenization_manifest.json" in part for part in job["command"])
        and job["eligible_for_model_selection"] is True
        and job["outer_query_evaluation_only"] is False
        for job in stage1_post["bind_validation"]
    )

    selected_ids = set(sorted(expected_stage1_ids)[:2])
    outer_evaluation = posttrain_generator.generate_post_training_jobs(
        stage1,
        enable_outer_query_evaluation=True,
        outer_query_selected_job_ids=selected_ids,
    )
    for phase in ("bind_query", "project_query"):
        assert {
            job["parent_training_job_id"] for job in outer_evaluation[phase]
        } == selected_ids
        assert len(outer_evaluation[phase]) == 2
        assert all(
            job["outer_query_evaluation_only"] is True
            and job["eligible_for_model_selection"] is False
            and job["outer_query_allowlist_required"] is True
            for job in outer_evaluation[phase]
        )
    with pytest.raises(ValueError, match="requires both"):
        posttrain_generator.generate_post_training_jobs(
            stage1,
            outer_query_selected_job_ids=selected_ids,
        )

    resolved = json.loads(json.dumps(config))
    for lineage in resolved["lineages"]:
        resolved["stages"]["stage3"]["configuration_selection_by_lineage"][lineage] = [
            {"sampler": "hybrid", "hvg_size": 3000}
        ]
    stage3 = training_generator.generate_jobs(resolved, "stage3", base_seed=42)
    assert len(stage3) == 25 and all(job["runnable"] for job in stage3)
    stage3_post = posttrain_generator.generate_post_training_jobs(stage3)
    expected_stage3_ids = {job["job_id"] for job in stage3}
    assert {
        job["parent_training_job_id"] for job in stage3_post["bind_reference"]
    } == expected_stage3_ids
    assert {
        job["parent_training_job_id"] for job in stage3_post["project_reference"]
    } == expected_stage3_ids
    assert stage3_post["bind_query"] == []
    assert stage3_post["project_query"] == []
    assert stage3_post["bind_validation"] == []
    assert stage3_post["project_validation"] == []
    assert not any(
        "gp_id" in job or "--gp-id" in job["command"]
        for rows in stage1_post.values()
        for job in rows
    )


def test_slurm_script_has_fixed_account_and_no_guessed_resources() -> None:
    text = (REPOSITORY_ROOT / "slurm" / "tripso_array.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --account=immunehealth" in text
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --time" not in text
    assert "#SBATCH --gpus" not in text
    assert "#SBATCH --gres" not in text


def test_array_runner_writes_atomic_done_marker_and_skips_restart(
    tmp_path: Path,
) -> None:
    runner = _load_script("run_manifest_task_test", "slurm/run_manifest_task.py")
    upstream = tmp_path / "environment.json"
    upstream.write_text(json.dumps({"environment_passed": True}), encoding="utf-8")
    output = tmp_path / "job"
    expected = output / "result.txt"
    tracking_environment = output / "tracking_environment.json"
    command = [
        sys.executable,
        "-c",
        "import json, os; from pathlib import Path; "
        "Path(r'%s').parent.mkdir(parents=True, exist_ok=True); "
        "Path(r'%s').write_text('ok'); "
        "Path(r'%s').write_text(json.dumps({"
        "'WANDB_MODE': os.environ.get('WANDB_MODE'), "
        "'WANDB_SILENT': os.environ.get('WANDB_SILENT')}))"
        % (expected, expected, tracking_environment),
    ]
    job = {
        "schema_version": "immune-health-slurm-job/v1",
        "job_id": "test-job",
        "runnable": True,
        "seed": 11,
        "output_dir": str(output),
        "job_spec_path": str(output / "job_spec.json"),
        "upstream_artifacts": [
            {"path": str(upstream), "json_require": {"environment_passed": True}}
        ],
        "expected_outputs": [str(expected)],
        "working_directory": str(tmp_path),
        "command": command,
    }

    assert runner.run_job(job) == 0
    done = json.loads((output / ".done.json").read_text(encoding="utf-8"))
    assert done["job_id"] == "test-job"
    assert done["schema_version"] == "immune-health-slurm-completion/v2"
    assert done["expected_output_inventory"] == [
        {
            "kind": "file",
            "path": str(expected.resolve()),
            "sha256": runner._file_hash(expected),
            "size_bytes": 2,
        }
    ]
    tracking_env = json.loads(tracking_environment.read_text(encoding="utf-8"))
    assert tracking_env == {"WANDB_MODE": "disabled", "WANDB_SILENT": "true"}
    assert not (output / ".failed.json").exists()
    first_mtime = expected.stat().st_mtime_ns
    assert runner.run_job(job) == 0
    assert expected.stat().st_mtime_ns == first_mtime

    old_done = dict(done)
    old_done["schema_version"] = "immune-health-slurm-completion/v1"
    old_done.pop("expected_output_inventory")
    old_done.pop("expected_output_inventory_sha256")
    (output / ".done.json").write_text(json.dumps(old_done), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predates deterministic output inventory"):
        runner.run_job(job)

    (output / ".done.json").write_text(json.dumps(done), encoding="utf-8")
    expected.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected output content changed"):
        runner.run_job(job)
    expected.unlink()
    with pytest.raises(RuntimeError, match="outputs cannot be verified"):
        runner.run_job(job)


def test_array_runner_hashes_directory_trees_and_rejects_stale_marker(
    tmp_path: Path,
) -> None:
    runner = _load_script("run_manifest_directory_test", "slurm/run_manifest_task.py")
    output = tmp_path / "runner"
    expected_directory = tmp_path / "artifact"
    nested_file = expected_directory / "nested" / "data.bin"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; "
        f"p=Path(r'{nested_file}'); p.parent.mkdir(parents=True); "
        "p.write_bytes(b'abc')",
    ]
    job = {
        "schema_version": "immune-health-slurm-job/v1",
        "job_id": "directory-job",
        "runnable": True,
        "seed": 3,
        "output_dir": str(output),
        "job_spec_path": str(output / "job_spec.json"),
        "upstream_artifacts": [],
        "expected_outputs": [str(expected_directory)],
        "working_directory": str(tmp_path),
        "command": command,
    }

    assert runner.run_job(job) == 0
    done = json.loads((output / ".done.json").read_text(encoding="utf-8"))
    directory = done["expected_output_inventory"][0]
    assert directory["kind"] == "directory"
    assert directory["n_files"] == 1
    assert directory["tree_sha256"]
    nested_file.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="expected output content changed"):
        runner.run_job(job)
