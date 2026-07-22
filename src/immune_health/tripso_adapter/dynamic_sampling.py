"""Project-local donor-hierarchical sampling for the vendored TRIPSO loader.

The vendor API creates an unrestricted random 80/10/10 cell split internally.
This module replaces it with a deterministic, disjoint split that guarantees at
least one training cell from every dataset/donor/fine-type stratum, then replaces
only the training split's row sampler with :class:`HierarchicalCellSampler`.
Validation and test rows are therefore never drawn by the training sampler.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from immune_health.sampling.hierarchical import (
    SAMPLER_MODES,
    HierarchicalCellSampler,
    SamplingResult,
    _normalize_balance_eligibility,
)

from .contracts import TripsoContractError

DEFAULT_STRING_ID_COLUMNS = (
    "biological_unit_id",
    "observation_id",
    "source_observation_id",
    "donor_id",
    "sample_id",
    "source_cell_id",
    "cell_id",
)

_CONFIG_KEYS = {
    "enabled",
    "dataset_column",
    "donor_column",
    "fine_type_column",
    "fine_type_balance_eligible_column",
    "cell_id_column",
    "lineage_column",
    "alpha",
    "mode",
    "fine_type_lambda",
    "lambda_by_lineage",
    "min_cells_per_fine_type",
    "n_cells_per_epoch",
    "preserve_string_id_columns",
    "audit_subdirectory",
}


@dataclass(frozen=True)
class DynamicSamplerConfig:
    """Validated settings for one training split.

    ``n_cells_per_epoch=None`` means one full-batch-truncated draw per physical
    row in the vendor training subset.  A fixed integer is preferable when
    comparing folds because it equalises optimiser exposure.
    """

    enabled: bool = True
    dataset_column: str = "dataset"
    donor_column: str = "biological_unit_id"
    fine_type_column: str = "fine_type"
    fine_type_balance_eligible_column: str = "fine_type_balance_eligible"
    cell_id_column: str | None = "idx"
    lineage_column: str | None = "lineage"
    lineage: str | None = None
    alpha: float = 0.5
    mode: str = "hybrid"
    fine_type_lambda: float = 0.7
    lambda_by_lineage: Mapping[str, float] | None = None
    min_cells_per_fine_type: int = 1
    n_cells_per_epoch: int | None = None
    batch_size: int = 32
    seed: int = 0
    preserve_string_id_columns: tuple[str, ...] = DEFAULT_STRING_ID_COLUMNS
    audit_subdirectory: str = "sampling_audit"

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["epoch_size_policy"] = (
            "training_split_size_truncated_to_full_batches"
            if self.n_cells_per_epoch is None
            else "fixed"
        )
        result["optimizer_split_policy"] = (
            "strata_preserving_approximately_80_10_10"
            if self.enabled
            else "vendor_unrestricted_random_80_10_10"
        )
        result["optimizer_split_strata"] = (
            [
                self.dataset_column,
                self.donor_column,
                self.fine_type_column,
                self.fine_type_balance_eligible_column,
            ]
            if self.enabled
            else None
        )
        return result


def _nonempty_column_name(
    value: Any, name: str, *, allow_none: bool = False
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TripsoContractError(f"project_sampler.{name} must be a non-empty string")
    return value.strip()


def normalize_dynamic_sampler_config(
    raw: Mapping[str, Any],
    *,
    batch_size: int,
    seed: int,
    lineage: str | None,
) -> DynamicSamplerConfig:
    """Validate user settings and bind fold-specific values.

    Batch size, seed, and lineage are deliberately taken from the protected
    TRIPSO call/fold rather than duplicated in user configuration.
    """

    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise TripsoContractError(
            f"Unknown project_sampler options (possible typo): {unknown}"
        )
    if not isinstance(raw.get("enabled", True), bool):
        raise TripsoContractError("project_sampler.enabled must be boolean")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise TripsoContractError("TRIPSO batch_size must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TripsoContractError("TRIPSO seed must be a non-negative integer")

    mode = str(raw.get("mode", "hybrid"))
    if mode not in SAMPLER_MODES:
        raise TripsoContractError(
            f"project_sampler.mode must be one of {sorted(SAMPLER_MODES)}"
        )
    try:
        alpha = float(raw.get("alpha", 0.5))
        fine_type_lambda = float(raw.get("fine_type_lambda", 0.7))
    except (TypeError, ValueError) as exc:
        raise TripsoContractError(
            "project_sampler alpha and fine_type_lambda must be numeric"
        ) from exc
    if not 0 <= alpha <= 1:
        raise TripsoContractError("project_sampler.alpha must be between 0 and 1")
    if not 0 <= fine_type_lambda <= 1:
        raise TripsoContractError(
            "project_sampler.fine_type_lambda must be between 0 and 1"
        )

    minimum = raw.get("min_cells_per_fine_type", 1)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise TripsoContractError(
            "project_sampler.min_cells_per_fine_type must be a positive integer"
        )
    n_cells = raw.get("n_cells_per_epoch")
    if n_cells is not None:
        if (
            not isinstance(n_cells, int)
            or isinstance(n_cells, bool)
            or n_cells < batch_size
        ):
            raise TripsoContractError(
                "project_sampler.n_cells_per_epoch must be null or an integer at "
                "least as large as batch_size"
            )
        if n_cells % batch_size:
            raise TripsoContractError(
                "project_sampler.n_cells_per_epoch must be divisible by batch_size "
                "so the audit equals the cells consumed with drop_last=True"
            )

    lambda_by_lineage = raw.get("lambda_by_lineage")
    if lambda_by_lineage is not None:
        if not isinstance(lambda_by_lineage, Mapping):
            raise TripsoContractError(
                "project_sampler.lambda_by_lineage must be a mapping"
            )
        lambda_by_lineage = {
            str(key): float(value) for key, value in lambda_by_lineage.items()
        }
        invalid = {
            key: value
            for key, value in lambda_by_lineage.items()
            if not 0 <= value <= 1
        }
        if invalid:
            raise TripsoContractError(
                f"project_sampler.lambda_by_lineage values must be in [0, 1]: {invalid}"
            )

    preserve = raw.get("preserve_string_id_columns", DEFAULT_STRING_ID_COLUMNS)
    if (
        not isinstance(preserve, Sequence)
        or isinstance(preserve, (str, bytes))
        or not all(isinstance(item, str) and item.strip() for item in preserve)
    ):
        raise TripsoContractError(
            "project_sampler.preserve_string_id_columns must be a string list"
        )
    audit_subdirectory = _nonempty_column_name(
        raw.get("audit_subdirectory", "sampling_audit"), "audit_subdirectory"
    )
    assert audit_subdirectory is not None
    if Path(audit_subdirectory).is_absolute() or ".." in Path(audit_subdirectory).parts:
        raise TripsoContractError(
            "project_sampler.audit_subdirectory must be a relative child path"
        )

    return DynamicSamplerConfig(
        enabled=bool(raw.get("enabled", True)),
        dataset_column=_nonempty_column_name(
            raw.get("dataset_column", "dataset"), "dataset_column"
        ),
        donor_column=_nonempty_column_name(
            raw.get("donor_column", "biological_unit_id"), "donor_column"
        ),
        fine_type_column=_nonempty_column_name(
            raw.get("fine_type_column", "fine_type"), "fine_type_column"
        ),
        fine_type_balance_eligible_column=_nonempty_column_name(
            raw.get(
                "fine_type_balance_eligible_column",
                "fine_type_balance_eligible",
            ),
            "fine_type_balance_eligible_column",
        ),
        cell_id_column=_nonempty_column_name(
            raw.get("cell_id_column", "idx"), "cell_id_column", allow_none=True
        ),
        lineage_column=_nonempty_column_name(
            raw.get("lineage_column", "lineage"), "lineage_column", allow_none=True
        ),
        lineage=lineage,
        alpha=alpha,
        mode=mode,
        fine_type_lambda=fine_type_lambda,
        lambda_by_lineage=lambda_by_lineage,
        min_cells_per_fine_type=minimum,
        n_cells_per_epoch=n_cells,
        batch_size=batch_size,
        seed=seed,
        preserve_string_id_columns=tuple(
            dict.fromkeys(item.strip() for item in preserve)
        ),
        audit_subdirectory=audit_subdirectory,
    )


class HierarchicalEpochSampler(Sampler[int]):
    """A PyTorch sampler that materialises and audits one deterministic epoch."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        config: DynamicSamplerConfig,
        *,
        audit_output_dir: Path,
    ) -> None:
        self.config = config
        self.audit_output_dir = Path(audit_output_dir)
        self._epoch = 0
        self._explicit_epoch: int | None = None
        self.last_result: SamplingResult | None = None
        self._sampler = HierarchicalCellSampler(
            metadata,
            dataset_column=config.dataset_column,
            donor_column=config.donor_column,
            fine_type_column=config.fine_type_column,
            fine_type_balance_eligible_column=(
                config.fine_type_balance_eligible_column
            ),
            cell_id_column=config.cell_id_column,
            lineage_column=config.lineage_column,
            lineage=config.lineage,
            alpha=config.alpha,
            mode=config.mode,
            hybrid_lambda=config.fine_type_lambda,
            lambda_by_lineage=config.lambda_by_lineage,
            min_cells_per_fine_type=config.min_cells_per_fine_type,
            seed=config.seed,
        )
        requested = config.n_cells_per_epoch
        if requested is None:
            requested = (len(metadata) // config.batch_size) * config.batch_size
        if requested < config.batch_size:
            raise TripsoContractError(
                "The vendor training subset contains fewer than one full batch"
            )
        self.n_cells_per_epoch = int(requested)

    def __len__(self) -> int:
        return self.n_cells_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Support launchers that explicitly propagate the training epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._explicit_epoch = int(epoch)

    @staticmethod
    def _require_single_process() -> None:
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        declared_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if visible_gpus > 1 or declared_world_size > 1:
            raise RuntimeError(
                "The project donor-hierarchical sampler requires exactly one "
                "visible GPU/process. Request one GPU per job."
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size != 1:
                raise RuntimeError(
                    "The project donor-hierarchical sampler currently requires one "
                    "training process/GPU. Request one GPU per job; multi-process "
                    "sampling is blocked rather than silently changing exposure."
                )

    def __iter__(self) -> Iterator[int]:
        self._require_single_process()
        epoch = self._epoch if self._explicit_epoch is None else self._explicit_epoch
        result = self._sampler.sample_epoch(
            self.n_cells_per_epoch,
            epoch=epoch,
            batch_size=self.config.batch_size,
        )
        if result.duplicate_cell_draws_within_batches:
            raise RuntimeError(
                "Hierarchical sampler reused a cell inside a batch; training aborted"
            )
        result.write_log(
            self.audit_output_dir,
            prefix=f"epoch_{epoch:04d}_rank_000",
        )
        runtime = {
            **self.config.manifest(),
            "effective_n_cells_per_epoch": self.n_cells_per_epoch,
        }
        self.audit_output_dir.mkdir(parents=True, exist_ok=True)
        (self.audit_output_dir / "sampler_runtime_config.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.last_result = result
        self._epoch = max(self._epoch, epoch + 1)
        self._explicit_epoch = None
        return iter(result.cell_positions.tolist())


def _metadata_for_training_subset(
    gdata: Any,
    indices: Sequence[int],
    columns: Sequence[str],
) -> pd.DataFrame:
    available = set(getattr(gdata, "column_names", ()))
    missing = sorted(set(columns) - available)
    if missing:
        raise TripsoContractError(
            "Tokenized Arrow dataset is missing hierarchical-sampling metadata: "
            f"{missing}. Preserve these columns during tokenization."
        )
    selected = gdata.select([int(index) for index in indices])
    if hasattr(selected, "select_columns"):
        selected = selected.select_columns(list(columns))
    if hasattr(selected, "to_pandas"):
        return selected.to_pandas().reset_index(drop=True)
    return pd.DataFrame({column: selected[column] for column in columns})


def _strata_preserving_split_indices(
    metadata: pd.DataFrame,
    *,
    strata_columns: Sequence[str],
    seed: int,
) -> tuple[list[int], list[int], list[int], pd.DataFrame]:
    """Create disjoint optimiser splits without losing a donor/fine-type stratum.

    The vendored datamodule uses an unrestricted random 80/10/10 cell split.  A
    singleton donor/fine-type stratum can therefore disappear before the project
    sampler sees it.  Here each complete stratum contributes at least one cell to
    training (and approximately 80% for non-small strata); the remaining cells are
    divided between validation and test.  These are optimiser diagnostics inside
    adaptation donors, never biological model-selection folds.
    """

    columns = tuple(dict.fromkeys(map(str, strata_columns)))
    missing = sorted(set(columns) - set(metadata.columns))
    if missing:
        raise TripsoContractError(
            f"Cannot make a strata-preserving optimiser split; missing {missing}"
        )
    if metadata.empty:
        raise TripsoContractError("Cannot split an empty adaptation dataset")

    rng = np.random.default_rng(int(seed))
    training: list[int] = []
    remainder: list[int] = []
    group_key: str | list[str] = list(columns) if len(columns) > 1 else columns[0]
    records: list[dict[str, Any]] = []
    grouped = metadata.groupby(group_key, observed=True, sort=True).indices
    for raw_key, raw_positions in grouped.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        positions = np.asarray(raw_positions, dtype=np.int64)
        shuffled = rng.permutation(positions)
        # ceil, rather than floor, keeps all cells in strata of size 1--4 and
        # approaches the vendor's 80% training target for larger strata.
        n_training = max(1, int(np.ceil(0.8 * len(shuffled))))
        selected = shuffled[:n_training]
        held = shuffled[n_training:]
        training.extend(map(int, selected))
        remainder.extend(map(int, held))
        normalized_key = {
            column: (bool(value) if isinstance(value, (bool, np.bool_)) else str(value))
            for column, value in zip(columns, key, strict=True)
        }
        records.append(
            {
                **normalized_key,
                "n_source_cells": int(len(shuffled)),
                "n_training_cells": int(len(selected)),
                "n_diagnostic_cells": int(len(held)),
            }
        )

    remainder_array = rng.permutation(np.asarray(remainder, dtype=np.int64))
    n_validation = len(remainder_array) // 2
    validation = list(map(int, remainder_array[:n_validation]))
    test = list(map(int, remainder_array[n_validation:]))
    # Subset order is randomized deterministically so a systematic source-file
    # ordering cannot leak into batches; the epoch sampler performs further draws.
    training = list(map(int, rng.permutation(np.asarray(training, dtype=np.int64))))

    sets = [set(training), set(validation), set(test)]
    if any(left & right for i, left in enumerate(sets) for right in sets[i + 1 :]):
        raise AssertionError("Strata-preserving optimiser splits overlap")
    expected = set(range(len(metadata)))
    if set().union(*sets) != expected:
        raise AssertionError("Strata-preserving optimiser splits omit source cells")
    strata = pd.DataFrame.from_records(records)
    if strata.empty or (strata["n_training_cells"] < 1).any():
        raise AssertionError("An observed donor/fine-type stratum was lost")
    return training, validation, test, strata


def _write_training_pool_audit(
    output_dir: Path,
    *,
    strata: pd.DataFrame,
    n_source: int,
    training: Sequence[int],
    validation: Sequence[int],
    test: Sequence[int],
    fine_type_balance_eligible_column: str,
    fine_type_balance_eligibility_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strata_path = output_dir / "training_pool_strata.tsv"
    strata.to_csv(strata_path, sep="\t", index=False)
    balance_eligible = strata[fine_type_balance_eligible_column].astype(bool)
    payload = {
        "schema_version": "immune-health-tripso-training-pool/v2",
        "policy": "strata_preserving_approximately_80_10_10",
        "vendor_unrestricted_random_split_replaced": True,
        "scope": "optimizer diagnostics within physical adaptation donors",
        "biological_model_selection_split": False,
        "n_source_cells": int(n_source),
        "n_training_cells": int(len(training)),
        "n_validation_cells": int(len(validation)),
        "n_test_cells": int(len(test)),
        "n_source_strata": int(len(strata)),
        "n_strata_absent_from_training": int((strata["n_training_cells"] < 1).sum()),
        "fine_type_balance_eligible_column": (fine_type_balance_eligible_column),
        "fine_type_balance_eligibility_source": (fine_type_balance_eligibility_source),
        "n_balance_eligible_strata": int(balance_eligible.sum()),
        "n_balance_ineligible_strata": int((~balance_eligible).sum()),
        "n_balance_eligible_source_cells": int(
            strata.loc[balance_eligible, "n_source_cells"].sum()
        ),
        "n_balance_ineligible_source_cells": int(
            strata.loc[~balance_eligible, "n_source_cells"].sum()
        ),
        "n_balance_ineligible_strata_absent_from_training": int(
            (~balance_eligible & strata["n_training_cells"].lt(1)).sum()
        ),
        "balance_ineligible_optimizer_split_policy": (
            "retained_in_training; eligibility affects fine-type uplift only"
        ),
        "small_strata_policy": (
            "ceil(0.8*n) with at least one training cell; sizes 1-4 remain fully "
            "available to the donor-aware training sampler"
        ),
        "strata_table": strata_path.name,
    }
    (output_dir / "training_pool_contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_identifier_safe_datamodule_class(
    vendor_datamodule_class: type,
    preserve_string_id_columns: Sequence[str] = DEFAULT_STRING_ID_COLUMNS,
) -> type:
    """Preserve string identifiers without changing vendor sampling semantics."""

    configured = tuple(dict.fromkeys(map(str, preserve_string_id_columns)))

    class IdentifierSafeDataModule(vendor_datamodule_class):  # type: ignore[misc, valid-type]
        _immune_health_identifier_safe = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._immune_health_preserved_ids: tuple[str, ...] = ()
            super().__init__(*args, **kwargs)

        def setup(self, stage: str | None = None) -> None:
            super().setup(stage)
            try:
                gdata = self.dataset.tk_dataset.gdata
            except AttributeError as exc:
                raise RuntimeError(
                    "Vendored txDataModule internals changed; string identifier "
                    "transport could not be installed"
                ) from exc
            metadata_columns = tuple(getattr(self, "metadata", ()))
            preserve = set(configured) & set(metadata_columns)
            if len(gdata):
                first = gdata[0]
                preserve.update(
                    column
                    for column in metadata_columns
                    if column.endswith("_id") and isinstance(first.get(column), str)
                )
            self._immune_health_preserved_ids = tuple(sorted(preserve))

        def custom_collate(self, batch: list[Mapping[str, Any]]) -> Any:
            preserved = self._immune_health_preserved_ids
            original_metadata = self.metadata
            self.metadata = [
                item for item in original_metadata if item not in preserved
            ]
            try:
                output = super().custom_collate(batch)
            finally:
                self.metadata = original_metadata
            if self.return_tuple:
                return output
            for column in preserved:
                output[column] = [item["tk"][column] for item in batch]
            return output

    IdentifierSafeDataModule.__name__ = "IdentifierSafeDataModule"
    IdentifierSafeDataModule.__qualname__ = "IdentifierSafeDataModule"
    return IdentifierSafeDataModule


def make_dynamic_datamodule_class(
    vendor_datamodule_class: type,
    config: DynamicSamplerConfig,
) -> type:
    """Return a local subclass suitable for temporary injection into TRIPSO."""

    identifier_safe_class = make_identifier_safe_datamodule_class(
        vendor_datamodule_class,
        config.preserve_string_id_columns,
    )

    class DonorHierarchicalDataModule(identifier_safe_class):  # type: ignore[misc, valid-type]
        _immune_health_dynamic_sampler = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            output_dir = Path(kwargs.get("output_dir", "./")).resolve()
            self._immune_health_sampling_output = output_dir / config.audit_subdirectory
            self._immune_health_epoch_sampler: HierarchicalEpochSampler | None = None
            super().__init__(*args, **kwargs)

        def setup(self, stage: str | None = None) -> None:
            super().setup(stage)
            try:
                gdata = self.dataset.tk_dataset.gdata
            except AttributeError as exc:
                raise RuntimeError(
                    "Vendored txDataModule internals changed; dynamic sampling was "
                    "not installed"
                ) from exc

            required = [
                config.dataset_column,
                config.donor_column,
                config.fine_type_column,
            ]
            available_columns = set(getattr(gdata, "column_names", ()))
            eligibility_in_metadata = (
                config.fine_type_balance_eligible_column in available_columns
            )
            if eligibility_in_metadata:
                required.append(config.fine_type_balance_eligible_column)
            for optional in (config.cell_id_column, config.lineage_column):
                if optional is not None and optional in available_columns:
                    required.append(optional)
            required = list(dict.fromkeys(required))
            full_metadata = _metadata_for_training_subset(
                gdata, list(range(len(gdata))), required
            )
            split_metadata = full_metadata.copy()
            if eligibility_in_metadata:
                try:
                    split_metadata[config.fine_type_balance_eligible_column] = (
                        _normalize_balance_eligibility(
                            split_metadata[config.fine_type_balance_eligible_column],
                            column=config.fine_type_balance_eligible_column,
                        )
                    )
                except ValueError as exc:
                    raise TripsoContractError(str(exc)) from exc
                eligibility_source = "metadata_column"
            else:
                split_metadata[config.fine_type_balance_eligible_column] = True
                eligibility_source = "default_all_eligible_missing_column"
            train_indices, validation_indices, test_indices, strata = (
                _strata_preserving_split_indices(
                    split_metadata,
                    strata_columns=(
                        config.dataset_column,
                        config.donor_column,
                        config.fine_type_column,
                        config.fine_type_balance_eligible_column,
                    ),
                    seed=config.seed,
                )
            )
            # Replace the vendor's unrestricted random split before constructing
            # the hierarchical sampler.  Validation/test remain disjoint cell
            # diagnostics; biological tuning is handled by donor-level folds.
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, validation_indices)
            self.test_dataset = Subset(self.dataset, test_indices)
            self.train_size = len(train_indices)
            self.val_size = len(validation_indices)
            _write_training_pool_audit(
                self._immune_health_sampling_output,
                strata=strata,
                n_source=len(full_metadata),
                training=train_indices,
                validation=validation_indices,
                test=test_indices,
                fine_type_balance_eligible_column=(
                    config.fine_type_balance_eligible_column
                ),
                fine_type_balance_eligibility_source=eligibility_source,
            )
            metadata = _metadata_for_training_subset(gdata, train_indices, required)
            self._immune_health_epoch_sampler = HierarchicalEpochSampler(
                metadata,
                config,
                audit_output_dir=self._immune_health_sampling_output,
            )

        def train_dataloader(self) -> DataLoader[Any]:
            sampler = self._immune_health_epoch_sampler
            if sampler is None:
                raise RuntimeError("setup() must run before train_dataloader()")
            sampler._require_single_process()
            return DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
            )

    DonorHierarchicalDataModule.__name__ = "DonorHierarchicalDataModule"
    DonorHierarchicalDataModule.__qualname__ = "DonorHierarchicalDataModule"
    return DonorHierarchicalDataModule
