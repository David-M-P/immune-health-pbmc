"""Dataset -> donor -> fine-type -> cell hierarchical sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

SAMPLER_MODES = {
    "observed_proportions": 1.0,
    "fully_balanced": 0.0,
    "hybrid": None,
}


def _normalize_balance_eligibility(values: pd.Series, *, column: str) -> pd.Series:
    """Validate a nullable Arrow/pandas eligibility column as strict booleans."""

    if values.isna().any():
        raise ValueError(
            f"Sampler column {column!r} has {int(values.isna().sum())} missing values"
        )

    def normalize_one(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if int(value) in (0, 1):
                return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "false"}:
                return normalized == "true"
        raise ValueError(
            f"Sampler column {column!r} must contain only boolean values; "
            f"found {value!r}"
        )

    return values.map(normalize_one).astype(bool)


@dataclass(frozen=True)
class SamplingResult:
    """One deterministic epoch's sampled positions and audit information."""

    cell_positions: np.ndarray
    cell_ids: np.ndarray | None
    distribution_log: pd.DataFrame
    n_requested: int
    n_sampled: int
    duplicate_cell_draws_within_batches: int
    forced_replacement_cycles: int
    seed: int
    epoch: int
    rank: int
    world_size: int
    alpha: float
    mode: str
    lambda_value: float
    fine_type_balance_eligible_column: str
    fine_type_balance_eligibility_source: str

    def summary(self) -> dict[str, object]:
        """Return JSON-serializable epoch provenance."""

        fine_rows = self.distribution_log.loc[
            self.distribution_log["level"].eq("fine_type")
        ]
        balance_eligible = fine_rows["fine_type_balance_eligible"].eq(True)
        balance_ineligible = fine_rows["fine_type_balance_eligible"].eq(False)
        zero_probability = fine_rows["conditional_probability"].eq(0.0)
        fallback_rows = fine_rows["balance_fallback_to_observed"].eq(True)
        n_fallback_donors = int(
            fine_rows.loc[fallback_rows, ["dataset", "biological_unit_id"]]
            .drop_duplicates()
            .shape[0]
        )
        return {
            "n_requested": self.n_requested,
            "n_sampled": self.n_sampled,
            "n_unique_cell_positions": int(len(np.unique(self.cell_positions))),
            "duplicate_cell_draws_across_epoch": int(
                len(self.cell_positions) - len(np.unique(self.cell_positions))
            ),
            "duplicate_cell_draws_within_batches": (
                self.duplicate_cell_draws_within_batches
            ),
            "forced_replacement_cycles": self.forced_replacement_cycles,
            "seed": self.seed,
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "alpha": self.alpha,
            "mode": self.mode,
            "lambda": self.lambda_value,
            "fine_type_balance_eligible_column": (
                self.fine_type_balance_eligible_column
            ),
            "fine_type_balance_eligibility_source": (
                self.fine_type_balance_eligibility_source
            ),
            "n_balance_eligible_fine_type_strata": int(balance_eligible.sum()),
            "n_balance_ineligible_fine_type_strata": int(balance_ineligible.sum()),
            "n_zero_probability_fine_type_strata": int(zero_probability.sum()),
            "n_balance_ineligible_cells_in_training_pool": int(
                fine_rows.loc[balance_ineligible, "available_cells"].sum()
            ),
            "balance_ineligible_draws": int(
                fine_rows.loc[balance_ineligible, "realized_count"].sum()
            ),
            "n_donors_fallback_to_observed_no_balance_eligible_type": int(
                n_fallback_donors
            ),
            "balance_fallback_reason": (
                "no_balance_eligible_fine_type; donor uses observed proportions"
                if n_fallback_donors
                else None
            ),
        }

    def write_log(
        self, output_dir: str | Path, prefix: str = "sampling"
    ) -> dict[str, Path]:
        """Write intended/realized distributions and a compact JSON summary."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        table_path = output / f"{prefix}_distribution.tsv"
        summary_path = output / f"{prefix}_summary.json"
        self.distribution_log.to_csv(table_path, sep="\t", index=False)
        summary_path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n"
        )
        return {"distribution": table_path, "summary": summary_path}


class HierarchicalCellSampler:
    """Sample cells without allowing cell-rich donors to dominate silently.

    Dataset probabilities are proportional to eligible donor count ``** alpha``.
    Donors are uniform conditional on dataset.  Fine-type probabilities mix a
    donor's observed composition with a uniform distribution over its trusted,
    balance-eligible fine types.  Ineligible fine types retain only their observed
    component.  Cells are uniform within the selected stratum.

    A cell is not reused within one batch while any unused eligible cell remains.
    Quota is dynamically redistributed as fine types, donors, or datasets run out.
    Pools reset between batches and epochs, allowing normal stochastic training.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        *,
        dataset_column: str = "dataset",
        donor_column: str = "biological_unit_id",
        fine_type_column: str = "fine_type",
        fine_type_balance_eligible_column: str = "fine_type_balance_eligible",
        cell_id_column: str | None = "cell_id",
        lineage_column: str | None = "lineage",
        lineage: str | None = None,
        alpha: float = 0.5,
        mode: str = "hybrid",
        hybrid_lambda: float = 0.7,
        lambda_by_lineage: Mapping[str, float] | None = None,
        min_cells_per_fine_type: int = 1,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if mode not in SAMPLER_MODES:
            raise ValueError(
                f"Unknown sampler mode {mode!r}; choose {sorted(SAMPLER_MODES)}"
            )
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if min_cells_per_fine_type < 1:
            raise ValueError("min_cells_per_fine_type must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")

        required = {dataset_column, donor_column, fine_type_column}
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(f"Sampler metadata is missing columns: {missing}")
        input_metadata = metadata.reset_index(drop=True).copy()
        frame = input_metadata.copy()
        frame["_input_position"] = np.arange(len(frame), dtype=np.int64)
        for column in required:
            values = frame[column].astype("string")
            invalid = values.isna() | values.str.strip().eq("")
            if invalid.any():
                raise ValueError(
                    f"Sampler column {column!r} has {int(invalid.sum())} missing values"
                )
            frame[column] = values

        if (
            not isinstance(fine_type_balance_eligible_column, str)
            or not fine_type_balance_eligible_column.strip()
        ):
            raise ValueError(
                "fine_type_balance_eligible_column must be a non-empty string"
            )
        fine_type_balance_eligible_column = fine_type_balance_eligible_column.strip()
        internal_eligibility_column = "_fine_type_balance_eligible"
        if fine_type_balance_eligible_column in frame.columns:
            frame[internal_eligibility_column] = _normalize_balance_eligibility(
                frame[fine_type_balance_eligible_column],
                column=fine_type_balance_eligible_column,
            )
            eligibility_source = "metadata_column"
        else:
            # Backward compatibility for tokenized datasets created before the
            # reviewed ontology added an explicit eligibility flag.
            frame[internal_eligibility_column] = True
            eligibility_source = "default_all_eligible_missing_column"

        selected_lineage = lineage
        if lineage_column is not None and lineage_column in frame:
            observed_lineages = sorted(
                frame[lineage_column].dropna().astype(str).unique()
            )
            if selected_lineage is not None:
                frame = frame.loc[
                    frame[lineage_column].astype(str).eq(selected_lineage)
                ].copy()
                if frame.empty:
                    raise ValueError(f"No cells found for lineage {selected_lineage!r}")
            elif len(observed_lineages) == 1:
                selected_lineage = observed_lineages[0]
            elif len(observed_lineages) > 1:
                raise ValueError(
                    "A hierarchical sampler is lineage-specific; pass lineage=... "
                    "when metadata contains multiple lineages"
                )

        if mode == "hybrid":
            value = (
                lambda_by_lineage[selected_lineage]
                if lambda_by_lineage is not None
                and selected_lineage is not None
                and selected_lineage in lambda_by_lineage
                else hybrid_lambda
            )
        else:
            value = float(SAMPLER_MODES[mode])
        if not 0 <= value <= 1:
            raise ValueError("Fine-type lambda must be between 0 and 1")

        frame = frame.reset_index(drop=True)
        grouping = [dataset_column, donor_column, fine_type_column]
        eligibility_counts = frame.groupby(grouping, observed=True, sort=True)[
            internal_eligibility_column
        ].nunique()
        inconsistent = eligibility_counts[eligibility_counts != 1]
        if not inconsistent.empty:
            examples = [tuple(map(str, key)) for key in inconsistent.index[:5]]
            raise ValueError(
                "fine-type balance eligibility must be constant within each "
                "dataset/donor/fine-type stratum; inconsistent examples: "
                f"{examples}"
            )
        group_sizes = frame.groupby(grouping, observed=True, sort=True).size()
        eligible_keys = group_sizes[group_sizes >= min_cells_per_fine_type].index
        eligible_index = pd.MultiIndex.from_tuples(
            eligible_keys.tolist(), names=grouping
        )
        row_keys = pd.MultiIndex.from_frame(frame[grouping])
        frame = frame.loc[row_keys.isin(eligible_index)].copy()
        if frame.empty:
            raise ValueError("No eligible donor/fine-type strata remain")

        frame = frame.reset_index(drop=True)
        # Returned positions always index the caller's original metadata order,
        # even when a lineage or minimum-cell filter removes rows.
        self.metadata = input_metadata
        self.eligible_metadata = frame
        self.dataset_column = dataset_column
        self.donor_column = donor_column
        self.fine_type_column = fine_type_column
        self.fine_type_balance_eligible_column = fine_type_balance_eligible_column
        self.fine_type_balance_eligibility_source = eligibility_source
        self.cell_id_column = (
            cell_id_column if cell_id_column in input_metadata else None
        )
        self.lineage = selected_lineage
        self.alpha = float(alpha)
        self.mode = mode
        self.lambda_value = float(value)
        self.min_cells_per_fine_type = int(min_cells_per_fine_type)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

        self._cells: dict[tuple[str, str, str], np.ndarray] = {}
        grouped_positions = frame.groupby(
            grouping, observed=True, sort=True
        ).indices.items()
        for key, positions in grouped_positions:
            normalized = tuple(map(str, key))
            self._cells[normalized] = frame.iloc[np.asarray(positions)][
                "_input_position"
            ].to_numpy(dtype=np.int64, copy=True)
        self._fine_counts = {
            key: len(positions) for key, positions in self._cells.items()
        }
        grouped_eligibility = frame.groupby(grouping, observed=True, sort=True)[
            internal_eligibility_column
        ].first()
        self._fine_balance_eligible = {
            tuple(map(str, key)): bool(value)
            for key, value in grouped_eligibility.items()
        }
        self._donor_fines: dict[tuple[str, str], tuple[str, ...]] = {}
        for dataset, donor, fine_type in sorted(self._cells):
            key = (dataset, donor)
            self._donor_fines.setdefault(key, tuple())
            self._donor_fines[key] = (*self._donor_fines[key], fine_type)
        self._dataset_donors: dict[str, tuple[str, ...]] = {}
        for dataset, donor in sorted(self._donor_fines):
            self._dataset_donors.setdefault(dataset, tuple())
            self._dataset_donors[dataset] = (*self._dataset_donors[dataset], donor)

        donor_numbers = np.asarray(
            [
                len(self._dataset_donors[dataset])
                for dataset in sorted(self._dataset_donors)
            ],
            dtype=float,
        )
        dataset_weights = donor_numbers**self.alpha
        dataset_weights /= dataset_weights.sum()
        self._dataset_probability = dict(
            zip(sorted(self._dataset_donors), dataset_weights, strict=True)
        )
        self._fine_probability: dict[tuple[str, str], dict[str, float]] = {}
        self._observed_probability: dict[tuple[str, str], dict[str, float]] = {}
        self._uniform_probability: dict[tuple[str, str], dict[str, float]] = {}
        self._effective_lambda: dict[tuple[str, str], float] = {}
        self._balance_fallback_to_observed: dict[tuple[str, str], bool] = {}
        for donor_key, fine_types in self._donor_fines.items():
            counts = np.asarray(
                [self._fine_counts[(*donor_key, fine)] for fine in fine_types],
                dtype=float,
            )
            observed = counts / counts.sum()
            balance_eligible = np.asarray(
                [
                    self._fine_balance_eligible[(*donor_key, fine)]
                    for fine in fine_types
                ],
                dtype=bool,
            )
            uniform = balance_eligible.astype(float)
            if uniform.sum():
                uniform /= uniform.sum()
            fallback_to_observed = bool(not uniform.sum() and self.lambda_value < 1.0)
            effective_lambda = 1.0 if fallback_to_observed else self.lambda_value
            mixed = effective_lambda * observed + (1.0 - effective_lambda) * uniform
            if not np.isclose(mixed.sum(), 1.0):
                raise AssertionError("Fine-type sampling probabilities do not sum to 1")
            self._effective_lambda[donor_key] = float(effective_lambda)
            self._balance_fallback_to_observed[donor_key] = fallback_to_observed
            self._observed_probability[donor_key] = dict(
                zip(fine_types, observed, strict=True)
            )
            self._uniform_probability[donor_key] = dict(
                zip(fine_types, uniform, strict=True)
            )
            self._fine_probability[donor_key] = dict(
                zip(fine_types, mixed, strict=True)
            )
        self._sampling_donor_fines = {
            donor_key: tuple(
                fine
                for fine in fine_types
                if self._fine_probability[donor_key][fine] > 0.0
            )
            for donor_key, fine_types in self._donor_fines.items()
        }
        if any(not fine_types for fine_types in self._sampling_donor_fines.values()):
            raise AssertionError(
                "A donor has no positive-probability fine-type stratum"
            )

    @property
    def intended_distribution(self) -> pd.DataFrame:
        """Return dataset, donor, and fine-type target probabilities."""

        rows: list[dict[str, object]] = []
        for dataset, dataset_probability in self._dataset_probability.items():
            rows.append(
                {
                    "level": "dataset",
                    "dataset": dataset,
                    "biological_unit_id": pd.NA,
                    "fine_type": pd.NA,
                    "conditional_probability": dataset_probability,
                    "intended_probability": dataset_probability,
                    "observed_fine_type_probability": np.nan,
                    "uniform_fine_type_probability": np.nan,
                    "observed_component_probability": np.nan,
                    "uniform_component_probability": np.nan,
                    "fine_type_balance_eligible": pd.NA,
                    "balance_eligibility_source": (
                        self.fine_type_balance_eligibility_source
                    ),
                    "excluded_from_sampling": pd.NA,
                    "available_cells": pd.NA,
                    "configured_fine_type_lambda": self.lambda_value,
                    "effective_fine_type_lambda": pd.NA,
                    "balance_fallback_to_observed": pd.NA,
                    "balance_fallback_reason": pd.NA,
                }
            )
            donors = self._dataset_donors[dataset]
            conditional_donor = 1.0 / len(donors)
            for donor in donors:
                donor_key = (dataset, donor)
                donor_marginal = dataset_probability * conditional_donor
                rows.append(
                    {
                        "level": "donor",
                        "dataset": dataset,
                        "biological_unit_id": donor,
                        "fine_type": pd.NA,
                        "conditional_probability": conditional_donor,
                        "intended_probability": donor_marginal,
                        "observed_fine_type_probability": np.nan,
                        "uniform_fine_type_probability": np.nan,
                        "observed_component_probability": np.nan,
                        "uniform_component_probability": np.nan,
                        "fine_type_balance_eligible": pd.NA,
                        "balance_eligibility_source": (
                            self.fine_type_balance_eligibility_source
                        ),
                        "excluded_from_sampling": pd.NA,
                        "available_cells": pd.NA,
                        "configured_fine_type_lambda": self.lambda_value,
                        "effective_fine_type_lambda": self._effective_lambda[donor_key],
                        "balance_fallback_to_observed": (
                            self._balance_fallback_to_observed[donor_key]
                        ),
                        "balance_fallback_reason": (
                            "no_balance_eligible_fine_type"
                            if self._balance_fallback_to_observed[donor_key]
                            else pd.NA
                        ),
                    }
                )
                fine_probabilities = self._fine_probability[donor_key].items()
                for fine_type, fine_probability in fine_probabilities:
                    rows.append(
                        {
                            "level": "fine_type",
                            "dataset": dataset,
                            "biological_unit_id": donor,
                            "fine_type": fine_type,
                            "conditional_probability": fine_probability,
                            "intended_probability": donor_marginal * fine_probability,
                            "observed_fine_type_probability": (
                                self._observed_probability[donor_key][fine_type]
                            ),
                            "uniform_fine_type_probability": (
                                self._uniform_probability[donor_key][fine_type]
                            ),
                            "observed_component_probability": (
                                self._effective_lambda[donor_key]
                                * self._observed_probability[donor_key][fine_type]
                            ),
                            "uniform_component_probability": (
                                (1.0 - self._effective_lambda[donor_key])
                                * self._uniform_probability[donor_key][fine_type]
                            ),
                            "fine_type_balance_eligible": (
                                self._fine_balance_eligible[(dataset, donor, fine_type)]
                            ),
                            "balance_eligibility_source": (
                                self.fine_type_balance_eligibility_source
                            ),
                            "excluded_from_sampling": fine_probability == 0.0,
                            "available_cells": self._fine_counts[
                                (dataset, donor, fine_type)
                            ],
                            "configured_fine_type_lambda": self.lambda_value,
                            "effective_fine_type_lambda": (
                                self._effective_lambda[donor_key]
                            ),
                            "balance_fallback_to_observed": (
                                self._balance_fallback_to_observed[donor_key]
                            ),
                            "balance_fallback_reason": (
                                "no_balance_eligible_fine_type"
                                if self._balance_fallback_to_observed[donor_key]
                                else pd.NA
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _weighted_choice(
        rng: np.random.Generator, values: list[str], weights: np.ndarray
    ) -> str:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        return values[int(rng.choice(len(values), p=weights))]

    def _sample_one_batch(
        self, rng: np.random.Generator, size: int
    ) -> tuple[list[int], list[tuple[str, str, str]], int]:
        selected: list[int] = []
        selected_strata: list[tuple[str, str, str]] = []
        forced_cycles = 0

        while len(selected) < size:
            pool: dict[tuple[str, str, str], np.ndarray] = {}
            cursor: dict[tuple[str, str, str], int] = {key: 0 for key in self._cells}
            active_fines = {
                donor_key: set(fines)
                for donor_key, fines in self._sampling_donor_fines.items()
            }
            active_donors = {
                dataset: set(donors) for dataset, donors in self._dataset_donors.items()
            }
            active_datasets = set(self._dataset_donors)
            cycle_start = len(selected)

            while active_datasets and len(selected) < size:
                datasets = sorted(active_datasets)
                dataset = self._weighted_choice(
                    rng,
                    datasets,
                    np.asarray([self._dataset_probability[item] for item in datasets]),
                )
                donors = sorted(active_donors[dataset])
                donor = donors[int(rng.integers(len(donors)))]
                donor_key = (dataset, donor)
                fine_types = sorted(active_fines[donor_key])
                fine_type = self._weighted_choice(
                    rng,
                    fine_types,
                    np.asarray(
                        [self._fine_probability[donor_key][fine] for fine in fine_types]
                    ),
                )
                key = (dataset, donor, fine_type)
                if key not in pool:
                    pool[key] = rng.permutation(self._cells[key])
                position = int(pool[key][cursor[key]])
                cursor[key] += 1
                selected.append(position)
                selected_strata.append(key)

                if cursor[key] == len(pool[key]):
                    active_fines[donor_key].remove(fine_type)
                    if not active_fines[donor_key]:
                        active_donors[dataset].remove(donor)
                        if not active_donors[dataset]:
                            active_datasets.remove(dataset)
            if len(selected) < size:
                # Every cell was used once.  Reuse is now unavoidable, so begin
                # another independently shuffled cycle and record it explicitly.
                forced_cycles += 1
            if len(selected) == cycle_start:
                raise RuntimeError("Sampler could not draw from nonempty metadata")
        return selected, selected_strata, forced_cycles

    def _realized_log(self, strata: list[tuple[str, str, str]]) -> pd.DataFrame:
        intended = self.intended_distribution.copy()
        draws = pd.DataFrame(
            strata,
            columns=["dataset", "biological_unit_id", "fine_type"],
        )
        total = len(draws)
        dataset_counts = draws.groupby("dataset", observed=True).size()
        donor_counts = draws.groupby(
            ["dataset", "biological_unit_id"], observed=True
        ).size()
        fine_counts = draws.groupby(
            ["dataset", "biological_unit_id", "fine_type"], observed=True
        ).size()
        realized_count: list[int] = []
        conditional: list[float] = []
        marginal: list[float] = []
        for row in intended.itertuples(index=False):
            if row.level == "dataset":
                count = int(dataset_counts.get(row.dataset, 0))
                denominator = total
            elif row.level == "donor":
                count = int(donor_counts.get((row.dataset, row.biological_unit_id), 0))
                denominator = int(dataset_counts.get(row.dataset, 0))
            else:
                count = int(
                    fine_counts.get(
                        (row.dataset, row.biological_unit_id, row.fine_type), 0
                    )
                )
                denominator = int(
                    donor_counts.get((row.dataset, row.biological_unit_id), 0)
                )
            realized_count.append(count)
            conditional.append(count / denominator if denominator else np.nan)
            marginal.append(count / total if total else np.nan)
        intended["realized_count"] = realized_count
        intended["realized_conditional_probability"] = conditional
        intended["realized_probability"] = marginal
        intended["effective_cells"] = realized_count
        return intended

    def sample_epoch(
        self,
        n_cells: int,
        *,
        epoch: int = 0,
        batch_size: int = 256,
    ) -> SamplingResult:
        """Draw an epoch deterministically from seed, epoch, rank, and world size."""

        if n_cells < 0 or batch_size < 1 or epoch < 0:
            raise ValueError(
                "n_cells/epoch must be nonnegative and batch_size positive"
            )
        seed_sequence = np.random.SeedSequence(
            [self.seed, int(epoch), self.rank, self.world_size]
        )
        rng = np.random.default_rng(seed_sequence)
        selected: list[int] = []
        strata: list[tuple[str, str, str]] = []
        forced_cycles = 0
        for start in range(0, n_cells, batch_size):
            current_size = min(batch_size, n_cells - start)
            batch_positions, batch_strata, cycles = self._sample_one_batch(
                rng, current_size
            )
            selected.extend(batch_positions)
            strata.extend(batch_strata)
            forced_cycles += cycles
        positions = np.asarray(selected, dtype=np.int64)
        ids = (
            self.metadata.iloc[positions][self.cell_id_column].to_numpy(copy=True)
            if self.cell_id_column is not None
            else None
        )
        duplicates = 0
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            duplicates += len(batch) - len(np.unique(batch))
        return SamplingResult(
            cell_positions=positions,
            cell_ids=ids,
            distribution_log=self._realized_log(strata),
            n_requested=int(n_cells),
            n_sampled=len(positions),
            duplicate_cell_draws_within_batches=int(duplicates),
            forced_replacement_cycles=int(forced_cycles),
            seed=self.seed,
            epoch=int(epoch),
            rank=self.rank,
            world_size=self.world_size,
            alpha=self.alpha,
            mode=self.mode,
            lambda_value=self.lambda_value,
            fine_type_balance_eligible_column=(self.fine_type_balance_eligible_column),
            fine_type_balance_eligibility_source=(
                self.fine_type_balance_eligibility_source
            ),
        )
