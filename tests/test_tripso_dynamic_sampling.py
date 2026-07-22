from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from immune_health.tripso_adapter import TripsoContractError
from immune_health.tripso_adapter.dynamic_sampling import (
    HierarchicalEpochSampler,
    _strata_preserving_split_indices,
    make_dynamic_datamodule_class,
    make_identifier_safe_datamodule_class,
    normalize_dynamic_sampler_config,
)
from immune_health.tripso_adapter.training import _dynamic_datamodule_context


def _metadata() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cell = 0
    for dataset, donor, fine_type, count in (
        ("large", "large::rich", "naive", 40),
        ("large", "large::rich", "memory", 8),
        ("large", "large::small", "naive", 4),
        ("small", "small::one", "naive", 4),
        ("small", "small::two", "memory", 4),
    ):
        for _ in range(count):
            rows.append(
                {
                    "dataset": dataset,
                    "biological_unit_id": donor,
                    "fine_type": fine_type,
                    "lineage": "B cells",
                    "idx": cell,
                }
            )
            cell += 1
    return pd.DataFrame(rows)


def test_epoch_sampler_is_deterministic_batch_unique_and_audited(
    tmp_path: Path,
) -> None:
    config = normalize_dynamic_sampler_config(
        {
            "mode": "hybrid",
            "alpha": 0.5,
            "fine_type_lambda": 0.7,
            "n_cells_per_epoch": 128,
        },
        batch_size=16,
        seed=19,
        lineage="B cells",
    )
    first = HierarchicalEpochSampler(
        _metadata(), config, audit_output_dir=tmp_path / "first"
    )
    second = HierarchicalEpochSampler(
        _metadata(), config, audit_output_dir=tmp_path / "second"
    )
    first_positions = list(first)
    second_positions = list(second)

    assert first_positions == second_positions
    assert len(first_positions) == 128
    for start in range(0, 128, 16):
        assert len(set(first_positions[start : start + 16])) == 16
    summary = json.loads(
        (tmp_path / "first/epoch_0000_rank_000_summary.json").read_text()
    )
    assert summary["duplicate_cell_draws_within_batches"] == 0
    assert summary["n_unique_cell_positions"] <= summary["n_sampled"]
    assert (tmp_path / "first/epoch_0000_rank_000_distribution.tsv").is_file()

    # A second iterator advances the deterministic epoch rather than replaying it.
    assert list(first) != first_positions
    assert (tmp_path / "first/epoch_0001_rank_000_summary.json").is_file()


def test_dynamic_sampler_configuration_rejects_ambiguous_exposure() -> None:
    with pytest.raises(TripsoContractError, match="divisible by batch_size"):
        normalize_dynamic_sampler_config(
            {"n_cells_per_epoch": 127},
            batch_size=16,
            seed=1,
            lineage="Monocytes",
        )


def test_optimizer_split_never_loses_small_donor_fine_type_strata() -> None:
    rows: list[dict[str, object]] = []
    for fine_type, count in (
        ("singleton", 1),
        ("doublet", 2),
        ("triplet", 3),
        ("common", 10),
    ):
        rows.extend(
            {
                "dataset": "cohort",
                "biological_unit_id": "cohort::donor",
                "fine_type": fine_type,
            }
            for _ in range(count)
        )
    metadata = pd.DataFrame(rows)
    train, validation, test, strata = _strata_preserving_split_indices(
        metadata,
        strata_columns=("dataset", "biological_unit_id", "fine_type"),
        seed=17,
    )

    assert not (set(train) & set(validation))
    assert not (set(train) & set(test))
    assert set(train) | set(validation) | set(test) == set(range(len(metadata)))
    assert (strata["n_training_cells"] >= 1).all()
    small = strata.set_index("fine_type").loc[["singleton", "doublet", "triplet"]]
    assert (small["n_training_cells"] == small["n_source_cells"]).all()
    with pytest.raises(TripsoContractError, match="possible typo"):
        normalize_dynamic_sampler_config(
            {"fine_typ_lambda": 0.5},
            batch_size=16,
            seed=1,
            lineage="Monocytes",
        )


class _ArrowLike:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.column_names = list(frame.columns)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int | str) -> Any:
        if isinstance(item, str):
            return self.frame[item].tolist()
        return self.frame.iloc[item].to_dict()

    def select(self, indices: list[int]) -> "_ArrowLike":
        return _ArrowLike(self.frame.iloc[indices].reset_index(drop=True))

    def select_columns(self, columns: list[str]) -> "_ArrowLike":
        return _ArrowLike(self.frame.loc[:, columns])

    def to_pandas(self) -> pd.DataFrame:
        return self.frame.copy()


class _TokenizedDataset:
    def __init__(self, gdata: _ArrowLike) -> None:
        self.gdata = gdata

    def __len__(self) -> int:
        return len(self.gdata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.gdata[index]


class _CombinedDataset:
    def __init__(self, tokenized: _TokenizedDataset) -> None:
        self.tk_dataset = tokenized

    def __len__(self) -> int:
        return len(self.tk_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"tk": self.tk_dataset[index], "adata": None}


class _FakeVendorDataModule:
    def __init__(
        self,
        *,
        folder: _ArrowLike,
        batch_size: int,
        num_workers: int = 0,
        output_dir: str = "./",
        **_: Any,
    ) -> None:
        self.folder = folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.return_tuple = False

    def setup(self, stage: str | None = None) -> None:
        del stage
        tokenized = _TokenizedDataset(self.folder)
        self.dataset = _CombinedDataset(tokenized)
        train_size = int(0.8 * len(self.dataset))
        self.train_dataset = Subset(self.dataset, list(range(train_size)))
        self.metadata = [
            column
            for column in self.folder.column_names
            if column not in {"input_ids", "length"}
        ]

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            collate_fn=self.custom_collate,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def custom_collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        tokenized = [item["tk"] for item in batch]
        output: dict[str, Any] = {}
        for column in self.metadata:
            values = [item[column] for item in tokenized]
            output[column] = (
                torch.tensor(values, dtype=torch.long)
                if column.endswith("_id")
                else values
            )
        return output


def test_datamodule_bridge_preserves_string_identifiers_and_fine_types(
    tmp_path: Path,
) -> None:
    frame = _metadata().iloc[:32].copy()
    frame["observation_id"] = [f"obs::{index // 4}" for index in range(len(frame))]
    frame["input_ids"] = [[1, 2]] * len(frame)
    frame["length"] = 2
    config = normalize_dynamic_sampler_config(
        {"n_cells_per_epoch": 16},
        batch_size=8,
        seed=3,
        lineage="B cells",
    )
    dynamic_class = make_dynamic_datamodule_class(_FakeVendorDataModule, config)
    module = dynamic_class(
        folder=_ArrowLike(frame),
        batch_size=8,
        num_workers=0,
        output_dir=str(tmp_path),
    )
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))

    assert len(batch["fine_type"]) == 8
    assert all(isinstance(value, str) for value in batch["biological_unit_id"])
    assert all(isinstance(value, str) for value in batch["observation_id"])
    assert module._immune_health_epoch_sampler is not None
    contract = json.loads(
        (tmp_path / "sampling_audit/training_pool_contract.json").read_text()
    )
    assert contract["vendor_unrestricted_random_split_replaced"] is True
    assert contract["n_strata_absent_from_training"] == 0


def test_native_datamodule_keeps_vendor_sampling_but_preserves_string_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _metadata().iloc[:32].copy()
    frame["observation_id"] = [f"obs::{index // 4}" for index in range(len(frame))]
    frame["input_ids"] = [[1, 2]] * len(frame)
    frame["length"] = 2
    config = normalize_dynamic_sampler_config(
        {"enabled": False},
        batch_size=8,
        seed=3,
        lineage="B cells",
    )

    def fake_train() -> None:
        return None

    monkeypatch.setitem(fake_train.__globals__, "txDataModule", _FakeVendorDataModule)
    with _dynamic_datamodule_context(fake_train, config):
        native_class = fake_train.__globals__["txDataModule"]
        assert native_class is not _FakeVendorDataModule
        assert native_class._immune_health_identifier_safe is True
        module = native_class(
            folder=_ArrowLike(frame),
            batch_size=8,
            num_workers=0,
            output_dir=str(tmp_path),
        )
        module.setup("fit")
        vendor_train_indices = tuple(module.train_dataset.indices)
        loader = module.train_dataloader()
        batch = next(iter(loader))

        assert isinstance(loader.sampler, RandomSampler)
        assert vendor_train_indices == tuple(range(int(0.8 * len(frame))))
        assert not hasattr(module, "_immune_health_epoch_sampler")
        assert all(isinstance(value, str) for value in batch["biological_unit_id"])
        assert all(isinstance(value, str) for value in batch["observation_id"])
        assert not (tmp_path / "sampling_audit").exists()

    assert fake_train.__globals__["txDataModule"] is _FakeVendorDataModule


def test_dynamic_sampler_retains_ineligible_strata_without_uniform_uplift(
    tmp_path: Path,
) -> None:
    frame = _metadata().copy()
    frame["fine_type_balance_eligible"] = frame["fine_type"].ne("memory")
    frame["input_ids"] = [[1, 2]] * len(frame)
    frame["length"] = 2
    config = normalize_dynamic_sampler_config(
        {
            "mode": "hybrid",
            "fine_type_lambda": 0.7,
            "fine_type_balance_eligible_column": ("fine_type_balance_eligible"),
            "n_cells_per_epoch": 32,
        },
        batch_size=8,
        seed=29,
        lineage="B cells",
    )
    dynamic_class = make_dynamic_datamodule_class(_FakeVendorDataModule, config)
    module = dynamic_class(
        folder=_ArrowLike(frame),
        batch_size=8,
        num_workers=0,
        output_dir=str(tmp_path),
    )
    module.setup("fit")

    contract = json.loads(
        (tmp_path / "sampling_audit/training_pool_contract.json").read_text()
    )
    assert contract["schema_version"] == "immune-health-tripso-training-pool/v2"
    assert contract["fine_type_balance_eligibility_source"] == "metadata_column"
    assert contract["n_balance_ineligible_source_cells"] > 0
    assert contract["n_balance_ineligible_strata_absent_from_training"] == 0
    strata = pd.read_csv(tmp_path / "sampling_audit/training_pool_strata.tsv", sep="\t")
    ineligible = strata.loc[~strata["fine_type_balance_eligible"]]
    assert not ineligible.empty
    assert (ineligible["n_training_cells"] >= 1).all()

    sampler = module._immune_health_epoch_sampler
    assert sampler is not None
    fine_rows = sampler._sampler.intended_distribution.query("level == 'fine_type'")
    memory = fine_rows.loc[fine_rows["fine_type"].eq("memory")]
    assert not memory.empty
    assert memory["fine_type_balance_eligible"].eq(False).all()
    assert memory["uniform_fine_type_probability"].eq(0.0).all()
    list(sampler)
    summary = json.loads(
        (tmp_path / "sampling_audit/epoch_0000_rank_000_summary.json").read_text()
    )
    assert summary["n_balance_ineligible_fine_type_strata"] > 0


def test_dynamic_sampler_missing_eligibility_column_is_backward_compatible(
    tmp_path: Path,
) -> None:
    frame = _metadata().iloc[:32].copy()
    frame["input_ids"] = [[1, 2]] * len(frame)
    frame["length"] = 2
    config = normalize_dynamic_sampler_config(
        {"n_cells_per_epoch": 16},
        batch_size=8,
        seed=3,
        lineage="B cells",
    )
    dynamic_class = make_dynamic_datamodule_class(_FakeVendorDataModule, config)
    module = dynamic_class(
        folder=_ArrowLike(frame),
        batch_size=8,
        num_workers=0,
        output_dir=str(tmp_path),
    )
    module.setup("fit")

    contract = json.loads(
        (tmp_path / "sampling_audit/training_pool_contract.json").read_text()
    )
    assert contract["fine_type_balance_eligibility_source"] == (
        "default_all_eligible_missing_column"
    )
    assert contract["n_balance_ineligible_strata"] == 0
    assert module._immune_health_epoch_sampler is not None
    assert (
        module._immune_health_epoch_sampler._sampler.fine_type_balance_eligibility_source
        == "default_all_eligible_missing_column"
    )


def test_identifier_safe_factory_does_not_replace_vendor_train_loader() -> None:
    safe_class = make_identifier_safe_datamodule_class(_FakeVendorDataModule)
    assert safe_class.train_dataloader is _FakeVendorDataModule.train_dataloader
