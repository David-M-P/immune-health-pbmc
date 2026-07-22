from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import pytest
from datasets import Dataset, load_from_disk

from immune_health.tripso_adapter import projection as projection_module
from immune_health.tripso_adapter.contracts import (
    TripsoContractError,
    canonical_json_hash,
)
from immune_health.tripso_adapter.geneformer import (
    EXPECTED_GENEFORMER_CONFIG,
    EXPECTED_GENEFORMER_HASHES,
    GENEFORMER_ROOT_ENV,
    VALIDATED_GENEFORMER_MODEL,
    VALIDATED_GENEFORMER_REVISION,
)


def _model_manifest(encoder_package: str = "geneformer") -> dict[str, Any]:
    return {
        "model_type": "Base",
        "seed": 11,
        "paths": {"checkpoint": "/tmp/model/checkpoints/last.ckpt"},
        "model_configuration": {
            "vendor_call": {
                "fm_encoder_pkg": encoder_package,
                "fm_encoder_name": VALIDATED_GENEFORMER_MODEL,
            },
            "geneformer_identity": {
                "model_name": VALIDATED_GENEFORMER_MODEL,
                "source_revision": VALIDATED_GENEFORMER_REVISION,
                "config": EXPECTED_GENEFORMER_CONFIG,
                "hashes": EXPECTED_GENEFORMER_HASHES,
                "hashes_pinned": True,
            },
        },
    }


def test_projection_enters_and_restores_full_geneformer_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Geneformer"
    fake_train = lambda: None  # noqa: E731
    tripso = SimpleNamespace(train=fake_train)
    events: list[str] = []
    active = False

    monkeypatch.setattr(
        projection_module,
        "resolve_geneformer_root",
        lambda value: root if value is None else Path(value),
    )

    def fake_validate(path: Path, *, model_name: str) -> dict[str, Any]:
        assert path == root
        assert model_name == VALIDATED_GENEFORMER_MODEL
        events.append("validate")
        return {
            "passed": True,
            "root": str(root),
            "model_name": VALIDATED_GENEFORMER_MODEL,
            "source_revision": VALIDATED_GENEFORMER_REVISION,
            "config": EXPECTED_GENEFORMER_CONFIG,
            "hashes": EXPECTED_GENEFORMER_HASHES,
            "hashes_pinned": True,
        }

    @contextmanager
    def fake_runtime(train_fn: Any, *, geneformer_root: Path) -> Iterator[None]:
        nonlocal active
        assert train_fn is fake_train
        assert geneformer_root == root
        active = True
        events.append("enter")
        try:
            yield
        finally:
            active = False
            events.append("restore")

    monkeypatch.setattr(projection_module, "validate_geneformer_root", fake_validate)
    monkeypatch.setattr(
        projection_module, "geneformer_runtime_compatibility", fake_runtime
    )

    with projection_module._geneformer_projection_compatibility(
        _model_manifest(), tripso
    ) as validation:
        assert active is True
        assert validation["passed"] is True
        assert validation["root"] == str(root)
        events.append("projection")

    assert active is False
    assert events == ["validate", "enter", "projection", "restore"]


def test_full_geneformer_projection_fails_actionably_without_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GENEFORMER_ROOT_ENV, raising=False)
    with pytest.raises(TripsoContractError, match=GENEFORMER_ROOT_ENV):
        with projection_module._geneformer_projection_compatibility(
            _model_manifest(), SimpleNamespace(train=lambda: None)
        ):
            pytest.fail("projection compatibility should not be entered")


def test_static_embedding_projection_does_not_require_geneformer_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_root_lookup(value: object) -> Path:
        del value
        raise AssertionError("static-embedding projection must not resolve full assets")

    monkeypatch.setattr(
        projection_module, "resolve_geneformer_root", unexpected_root_lookup
    )
    with projection_module._geneformer_projection_compatibility(
        _model_manifest("from_scratch"), SimpleNamespace()
    ) as validation:
        assert validation is None


def test_full_projection_keeps_compatibility_active_through_checkpoint_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "projection"
    model_manifest_path = tmp_path / "model_manifest.json"
    model_manifest_path.write_text("{}", encoding="utf-8")
    fold_path = tmp_path / "fold_input.json"
    fold_path.write_text("{}", encoding="utf-8")
    events: list[str] = []
    compatibility_active = False

    @contextmanager
    def fake_projection_compatibility(
        model_manifest: dict[str, Any], tripso_module: Any
    ) -> Iterator[None]:
        nonlocal compatibility_active
        assert model_manifest["model_type"] == "Base"
        assert tripso_module is fake_tripso
        compatibility_active = True
        events.append("enter")
        try:
            yield
        finally:
            compatibility_active = False
            events.append("restore")

    class FakeLightningModule:
        emb_dataset: Any = None

        def state_dict(self) -> dict[str, int]:
            return {"weight": 1}

        def parameters(self) -> list[Any]:
            return []

        def eval(self) -> None:
            return None

        def test_step(self, batch: Any, batch_index: int) -> None:
            del batch, batch_index
            values = {
                "GP_A": [[1.0, 2.0], [3.0, 4.0]],
                "GP_B": [[5.0, 6.0], [7.0, 8.0]],
                "GP_A_prop_genes": [1.0, 1.0],
                "GP_B_prop_genes": [1.0, 1.0],
                "cell_key": ["held::c1", "held::c2"],
                "dataset": ["held", "held"],
                "biological_unit_id": ["held::q1", "held::q1"],
                "observation_id": ["held::q1::s1", "held::q1::s1"],
                "fine_type": ["memory B", "memory B"],
                "lineage": ["B cells", "B cells"],
            }
            self.emb_dataset = Dataset.from_dict(values)

    class FakeEvaluator:
        hparam_save = "all"
        fm_encoder_name = VALIDATED_GENEFORMER_MODEL
        max_len = 4096

        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            assert compatibility_active is True
            events.append("checkpoint_load")

        def _init_trainer(self, **kwargs: Any) -> FakeLightningModule:
            del kwargs
            assert compatibility_active is True
            return FakeLightningModule()

    class FakeDataModule:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def test(self, model: Any, data_module: Any) -> None:
            del data_module
            assert compatibility_active is True
            events.append("test")
            model.test_step({}, 0)
            destination = output_dir / "embeddings" / "query_set"
            destination.parent.mkdir(parents=True)
            model.emb_dataset.save_to_disk(str(destination))

    fake_pl = ModuleType("pytorch_lightning")
    fake_pl.Trainer = FakeTrainer  # type: ignore[attr-defined]
    fake_tripso = ModuleType("tripso")
    fake_tripso.__path__ = []  # type: ignore[attr-defined]
    fake_tripso.train = lambda: None  # type: ignore[attr-defined]
    fake_tripso.gpEval = FakeEvaluator  # type: ignore[attr-defined]
    fake_datamodules = ModuleType("tripso.Datamodules")
    fake_datamodules.__path__ = []  # type: ignore[attr-defined]
    fake_datamodule = ModuleType("tripso.Datamodules.datamodule")
    fake_datamodule.txDataModule = FakeDataModule  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)
    monkeypatch.setitem(sys.modules, "tripso", fake_tripso)
    monkeypatch.setitem(sys.modules, "tripso.Datamodules", fake_datamodules)
    monkeypatch.setitem(
        sys.modules,
        "tripso.Datamodules.datamodule",
        fake_datamodule,
    )
    validated_model = _model_manifest()
    validated_model.update(
        {
            "lineage": "B cells",
            "reference_design": "lodo",
            "held_out_dataset": "held",
            "paths": {
                "checkpoint": "/tmp/model/checkpoints/last.ckpt",
                "fold_input_manifest": str(fold_path),
            },
            "hashes": {
                "checkpoint_sha256": "checkpoint-hash",
                "input_manifest_sha256": "fold-hash",
            },
        }
    )
    monkeypatch.setattr(
        projection_module,
        "validate_frozen_query_resources",
        lambda **kwargs: validated_model,
    )
    monkeypatch.setattr(
        projection_module,
        "_geneformer_projection_compatibility",
        fake_projection_compatibility,
    )
    monkeypatch.setattr(projection_module, "_inference_context", nullcontext)
    monkeypatch.setattr(
        projection_module,
        "load_fold_input_manifest",
        lambda path: {
            "reference_design": "lodo",
            "held_out_dataset": "held",
            "fold_id": "lodo_held",
        },
    )

    digest = hashlib.sha256()
    for key in ("held::c1", "held::c2"):
        encoded = key.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    query_manifest = {
        "projection_role": "query",
        "tokenized_dataset_path": str(tmp_path / "query"),
        "gp_library_path": str(tmp_path / "gp.csv"),
        "seed": 11,
        "n_cells": 2,
        "cell_key_ordered_sha256": digest.hexdigest(),
        "biological_unit_ids": ["held::q1"],
        "projection_metadata_columns": [
            "cell_key",
            "dataset",
            "biological_unit_id",
            "observation_id",
            "fine_type",
            "lineage",
        ],
        "gp_projection": {
            "program_ids": ["GP_A"],
            "program_ids_ordered_sha256": canonical_json_hash(["GP_A"]),
            "embedding_dimension": 2,
            "include_cell_token": False,
            "include_gene_encoder_cls": False,
        },
        "hashes": {
            "gp_library_sha256": "gp-hash",
            "gene_vocabulary_sha256": "vocabulary-hash",
        },
        "manifest_sha256": "projection-input-content-hash",
    }

    result = projection_module.run_vendor_frozen_projection(
        model_manifest_path=model_manifest_path,
        query_manifest=query_manifest,
        output_dir=output_dir,
    )

    assert result == output_dir / "embeddings" / "query_set"
    projected = load_from_disk(str(result))
    assert projected.column_names == [
        "GP_A",
        "cell_key",
        "dataset",
        "biological_unit_id",
        "observation_id",
        "fine_type",
        "lineage",
    ]
    output_manifest = json.loads(
        (output_dir / "projection_output_manifest.json").read_text()
    )
    assert output_manifest["projection_role"] == "query"
    assert output_manifest["gp_projection"]["program_ids"] == ["GP_A"]
    assert (
        output_manifest["vendor_batch_filter"]["unselected_gp_vectors_persisted"]
        is False
    )
    assert compatibility_active is False
    assert events == ["enter", "checkpoint_load", "test", "restore"]
