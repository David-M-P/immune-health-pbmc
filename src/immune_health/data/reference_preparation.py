"""Leakage-safe CPU preparation of merged healthy-reference lineage objects.

The functions in this module deliberately separate three concerns:

1. selecting one Terekhova observation per donor for reference fitting;
2. learning gene-program support and HVGs from adaptation donors only; and
3. materialising role-specific H5AD files without cell downsampling.

The held-out query is never used to select programs or genes. Production uses one
deterministic Terekhova visit in every role; retaining all query visits is an
explicit longitudinal sensitivity. The final all-healthy path has no held-out
sentinel, and future queries are mapped to its frozen feature manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.baselines.gp_scores import score_gene_programs
from immune_health.data.h5ad import read_csr_rows
from immune_health.data.lineage_scope import (
    LINEAGE_DONOR_SCOPE_SCHEMA,
    MATERIALIZED_PREPARATION_ROLES,
    canonical_json_digest,
    validate_lineage_donor_scope,
)
from immune_health.data.ontology import (
    apply_fine_type_ontology,
    approved_ontology_identity,
    load_fine_type_ontology,
)
from immune_health.gene_programs.filters import (
    GPFilterConfig,
    filter_gene_programs_training_only,
)
from immune_health.gene_programs.io import GeneProgram, load_gene_programs
from immune_health.gene_programs.transferability import (
    TransferabilityConfig,
    select_transferable_gene_programs,
)
from immune_health.provenance import atomic_write_json, sha256_file
from immune_health.splits.lodo import add_stable_identifiers

VISIT_MANIFEST_SCHEMA = "immune-health-terekhova-one-visit/v1"
ALL_HEALTHY_FOLD_SCHEMA = "immune-health-all-healthy-reference-fold/v1"
FEATURE_MANIFEST_SCHEMA = "immune-health-fold-features/v1"
MATERIALIZED_SCHEMA = "immune-health-materialized-fold-h5ad/v1"
PROJECTION_GP_CANDIDATE_SCHEMA = "immune-health-projection-gp-candidates/v1"
REFERENCE_DESIGNS = frozenset({"lodo", "all_healthy"})


@dataclass(frozen=True)
class ReferenceFeatureConfig:
    """Reviewable scientific settings for fold-local feature preparation."""

    hvg_sizes: tuple[int, ...] = (3000, 9000)
    hvg_mean_bins: int = 20
    hvg_minimum_donor_fraction: float = 0.01
    hvg_minimum_dataset_fraction: float = 0.75
    gp_minimum_mapped_genes: int = 10
    gp_maximum_program_size: int = 200
    gp_minimum_expression_coverage: float = 0.001
    gp_minimum_donor_coverage: float = 0.05
    gp_minimum_dataset_fraction: float = 0.75
    gp_redundancy_jaccard_threshold: float = 0.8
    gp_transfer_minimum_donors_per_cohort: int = 20
    gp_transfer_minimum_age_span: float = 10.0
    gp_transfer_minimum_cohorts: int = 3
    gp_transfer_minimum_sign_concordance: float = 0.75
    gp_transfer_maximum_i2: float = 0.75
    gp_transfer_maximum_fdr: float = 0.05
    gp_transfer_minimum_absolute_standardized_slope_per_decade: float = 0.0
    gp_projection_control_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        sizes = tuple(sorted(set(map(int, self.hvg_sizes))))
        if not sizes or sizes[0] < 1:
            raise ValueError("At least one positive HVG size is required")
        if sizes != self.hvg_sizes:
            raise ValueError("hvg_sizes must be unique and sorted increasingly")
        if self.hvg_mean_bins < 2:
            raise ValueError("hvg_mean_bins must be at least two")
        for name, value in (
            ("hvg_minimum_donor_fraction", self.hvg_minimum_donor_fraction),
            ("hvg_minimum_dataset_fraction", self.hvg_minimum_dataset_fraction),
            ("gp_minimum_expression_coverage", self.gp_minimum_expression_coverage),
            ("gp_minimum_donor_coverage", self.gp_minimum_donor_coverage),
            ("gp_minimum_dataset_fraction", self.gp_minimum_dataset_fraction),
            (
                "gp_redundancy_jaccard_threshold",
                self.gp_redundancy_jaccard_threshold,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        TransferabilityConfig(
            minimum_donors_per_cohort=self.gp_transfer_minimum_donors_per_cohort,
            minimum_age_span=self.gp_transfer_minimum_age_span,
            minimum_cohorts=self.gp_transfer_minimum_cohorts,
            minimum_sign_concordance=self.gp_transfer_minimum_sign_concordance,
            maximum_i2=self.gp_transfer_maximum_i2,
            maximum_fdr=self.gp_transfer_maximum_fdr,
            minimum_absolute_standardized_slope_per_decade=(
                self.gp_transfer_minimum_absolute_standardized_slope_per_decade
            ),
        ).validate()
        controls = tuple(map(str, self.gp_projection_control_ids))
        if len(controls) != len(set(controls)) or any(not value for value in controls):
            raise ValueError(
                "gp_projection_control_ids must contain unique nonempty program IDs"
            )


@dataclass(frozen=True)
class DonorCountSummary:
    """Training-only cell support and donor-pseudobulk counts."""

    gene_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    donor_datasets: tuple[str, ...]
    pseudobulk_counts: np.ndarray
    datasets: tuple[str, ...]
    dataset_cell_counts: np.ndarray
    dataset_gene_cell_counts: np.ndarray


@dataclass(frozen=True)
class DonorAwareHVGSelection:
    """One common HVG ranking plus equal-dataset component scores."""

    ranked_genes: tuple[str, ...]
    scores: pd.DataFrame
    dataset_scores: pd.DataFrame


def _build_lineage_donor_scope(
    cells: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    lineage: str,
) -> dict[str, Any]:
    """Build a complete, self-hashed donor scope from physical lineage cells."""

    global_donors = sorted(
        set(fold_manifest["biological_unit_id"].dropna().astype(str))
    )
    role_donors = {
        role: sorted(
            set(
                cells.loc[
                    cells["preparation_role"].eq(role), "biological_unit_id"
                ].astype(str)
            )
        )
        for role in MATERIALIZED_PREPARATION_ROLES
    }
    materialized_donors = set().union(*(set(values) for values in role_donors.values()))
    unexpected = sorted(materialized_donors - set(global_donors))
    if unexpected:
        raise AssertionError(
            "Physical lineage cells contain donors outside the global fold: "
            f"{unexpected[:5]}"
        )
    missing = sorted(set(global_donors) - materialized_donors)
    payload: dict[str, Any] = {
        "schema_version": LINEAGE_DONOR_SCOPE_SCHEMA,
        "lineage": str(lineage),
        "scope_unit": "biological_unit_id",
        "scope_source": (
            "physical_per_lineage_cell_metadata_after_fold_and_visit_selection"
        ),
        "biological_unit_ids_by_preparation_role": role_donors,
        "n_biological_units_by_preparation_role": {
            role: len(values) for role, values in role_donors.items()
        },
        "biological_unit_ids_by_preparation_role_sha256": {
            role: canonical_json_digest(values) for role, values in role_donors.items()
        },
        "n_source_lineage_biological_units": int(
            cells["biological_unit_id"].astype(str).nunique()
        ),
        "n_global_fold_biological_units": len(global_donors),
        "global_fold_biological_unit_ids_sha256": canonical_json_digest(global_donors),
        "global_fold_biological_unit_ids_without_materialized_role_cells": missing,
        "n_global_fold_biological_units_without_materialized_role_cells": len(missing),
    }
    payload["scope_sha256"] = canonical_json_digest(payload)
    return validate_lineage_donor_scope(payload, lineage=lineage)


def _pipe_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = normalized.map(mapping)
    if result.isna().any():
        invalid = sorted(normalized.loc[result.isna()].dropna().unique().tolist())
        raise ValueError(f"{label} contains invalid booleans: {invalid[:5]}")
    return result.astype(bool)


def _visit_hash(seed: int, biological_unit_id: str, observation_id: str) -> str:
    value = f"{int(seed)}::{biological_unit_id}::{observation_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_terekhova_one_visit_manifest(
    donor_or_cell_metadata: pd.DataFrame,
    *,
    seed: int = 42,
    dataset_name: str = "terekhova",
) -> pd.DataFrame:
    """Select one observation per Terekhova donor by a stable seeded hash.

    The input may be the global donor manifest, where observations are stored as
    pipe-delimited values, or cell-level metadata.  Age is intentionally absent
    from the choice so longitudinal donors are not systematically biased younger
    or older.
    """

    frame = donor_or_cell_metadata.copy()
    required = {"dataset", "donor_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Visit metadata is missing columns: {missing}")
    frame["dataset"] = frame["dataset"].astype("string")
    frame = frame.loc[frame["dataset"].eq(str(dataset_name))].copy()
    if frame.empty:
        raise ValueError(f"No rows found for repeated-visit dataset {dataset_name!r}")

    rows: list[dict[str, str]] = []
    if "sample_id" in frame.columns:
        cell_level = add_stable_identifiers(frame, copy=True)
        for row in (
            cell_level[
                [
                    "dataset",
                    "donor_id",
                    "sample_id",
                    "biological_unit_id",
                    "observation_id",
                ]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            rows.append(row._asdict())
    elif {"sample_ids", "observation_ids", "biological_unit_id"}.issubset(
        frame.columns
    ):
        for row in frame.itertuples(index=False):
            samples = _pipe_values(getattr(row, "sample_ids"))
            observations = _pipe_values(getattr(row, "observation_ids"))
            if len(samples) != len(observations) or not samples:
                raise ValueError(
                    "Pipe-delimited sample_ids and observation_ids must be nonempty "
                    "and have equal lengths"
                )
            for sample, observation in zip(samples, observations, strict=True):
                rows.append(
                    {
                        "dataset": str(getattr(row, "dataset")),
                        "donor_id": str(getattr(row, "donor_id")),
                        "sample_id": sample,
                        "biological_unit_id": str(getattr(row, "biological_unit_id")),
                        "observation_id": observation,
                    }
                )
    else:
        raise ValueError(
            "Visit input must contain cell-level sample_id/observation_id or "
            "donor-level sample_ids/observation_ids"
        )

    visits = pd.DataFrame.from_records(rows).drop_duplicates()
    if visits["observation_id"].duplicated().any():
        raise ValueError("observation_id must identify exactly one donor visit")
    visits["selection_sha256"] = [
        _visit_hash(seed, biological, observation)
        for biological, observation in zip(
            visits["biological_unit_id"], visits["observation_id"], strict=True
        )
    ]
    visits = visits.sort_values(
        ["biological_unit_id", "selection_sha256", "observation_id"]
    ).reset_index(drop=True)
    visits["selected_for_reference"] = ~visits.duplicated(
        "biological_unit_id", keep="first"
    )
    visits["selection_reason"] = np.where(
        visits["selected_for_reference"],
        "lowest_seeded_hash",
        "nonselected_repeated_visit",
    )
    selected_counts = visits.groupby("biological_unit_id", observed=True)[
        "selected_for_reference"
    ].sum()
    if not selected_counts.eq(1).all():
        raise AssertionError(
            "Exactly one Terekhova observation must be selected per donor"
        )
    visits.insert(0, "selection_seed", int(seed))
    return visits.sort_values(["biological_unit_id", "observation_id"]).reset_index(
        drop=True
    )


def write_terekhova_one_visit_manifest(
    donor_or_cell_metadata: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int = 42,
    dataset_name: str = "terekhova",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the tabular visit decisions and a compact provenance manifest."""

    output_dir = Path(output_dir)
    table_path = output_dir / "terekhova_one_visit.tsv"
    manifest_path = output_dir / "terekhova_one_visit.json"
    existing = [path for path in (table_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite visit outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    visits = build_terekhova_one_visit_manifest(
        donor_or_cell_metadata, seed=seed, dataset_name=dataset_name
    )
    visits.to_csv(table_path, sep="\t", index=False)
    counts = visits.groupby("biological_unit_id", observed=True).size()
    payload = {
        "schema_version": VISIT_MANIFEST_SCHEMA,
        "dataset": str(dataset_name),
        "seed": int(seed),
        "selection_algorithm": (
            "minimum sha256(seed::biological_unit_id::observation_id), "
            "lexical observation_id tie-break"
        ),
        "age_used_for_selection": False,
        "n_donors": int(visits["biological_unit_id"].nunique()),
        "n_observations": int(len(visits)),
        "n_selected_observations": int(visits["selected_for_reference"].sum()),
        "visit_count_distribution": {
            str(int(key)): int(value)
            for key, value in counts.value_counts().sort_index().items()
        },
        "table": table_path.name,
        "table_sha256": sha256_file(table_path),
    }
    atomic_write_json(manifest_path, payload)
    return table_path, manifest_path


def build_all_healthy_reference_fold(
    donor_metadata: pd.DataFrame,
    *,
    healthy_datasets: Sequence[str],
    inner_validation_fold: int | None = None,
    inner_fold_column: str = "global_inner_fold",
) -> pd.DataFrame:
    """Build an explicit all-reference donor table for final model fitting.

    This is intentionally distinct from a LODO table with a fake held-out value.
    By default every donor in the five declared healthy cohorts is eligible for
    adaptation.  ``inner_validation_fold`` can reserve one precomputed donor fold
    for model selection; a subsequent final fit should omit it and use all donors.
    """

    datasets = tuple(dict.fromkeys(map(str, healthy_datasets)))
    if len(datasets) != 5:
        raise ValueError("Final healthy-reference fitting requires five unique cohorts")
    frame = donor_metadata.copy()
    required = {"dataset", "donor_id", "biological_unit_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Global donor metadata is missing columns: {missing}")
    frame["dataset"] = frame["dataset"].astype("string")
    frame = frame.loc[frame["dataset"].isin(datasets)].copy()
    observed = set(frame["dataset"].astype(str))
    if observed != set(datasets):
        raise ValueError(
            "Global donor metadata does not contain every declared healthy cohort; "
            f"missing={sorted(set(datasets) - observed)}"
        )
    if frame.empty or frame["biological_unit_id"].duplicated().any():
        raise ValueError(
            "Final reference table must contain one row per biological unit"
        )
    expected_ids = frame["dataset"].astype(str) + "::" + frame["donor_id"].astype(str)
    if not frame["biological_unit_id"].astype(str).eq(expected_ids).all():
        raise ValueError("biological_unit_id does not equal dataset::donor_id")

    if inner_fold_column in frame:
        inner = pd.to_numeric(frame[inner_fold_column], errors="coerce").astype("Int64")
    elif inner_validation_fold is not None:
        raise ValueError(
            f"inner validation requires donor column {inner_fold_column!r}"
        )
    else:
        inner = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    validation = (
        inner.eq(int(inner_validation_fold)).fillna(False)
        if inner_validation_fold is not None
        else pd.Series(False, index=frame.index)
    )

    frame.insert(0, "reference_design", "all_healthy")
    frame.insert(1, "fold_id", "all_healthy")
    frame.insert(2, "heldout_dataset", pd.NA)
    frame.insert(3, "outer_role", "reference")
    frame["inner_fold"] = inner
    frame["reference_partition"] = np.where(validation, "validation", "adaptation")
    frame["eligible_for_reference_fitting"] = ~validation.to_numpy(dtype=bool)
    return frame.sort_values(["dataset", "biological_unit_id"]).reset_index(drop=True)


def write_all_healthy_reference_fold(
    donor_metadata: pd.DataFrame,
    output_dir: Path,
    *,
    healthy_datasets: Sequence[str],
    inner_validation_fold: int | None = None,
    inner_fold_column: str = "global_inner_fold",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the final-reference donor table and its provenance manifest."""

    output_dir = Path(output_dir)
    table_path = output_dir / "all_healthy.tsv"
    manifest_path = output_dir / "all_healthy.json"
    existing = [path for path in (table_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final fold outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_all_healthy_reference_fold(
        donor_metadata,
        healthy_datasets=healthy_datasets,
        inner_validation_fold=inner_validation_fold,
        inner_fold_column=inner_fold_column,
    )
    _write_table_atomic(table, table_path)
    payload = {
        "schema_version": ALL_HEALTHY_FOLD_SCHEMA,
        "reference_design": "all_healthy",
        "heldout_dataset": None,
        "healthy_datasets": list(map(str, healthy_datasets)),
        "inner_validation_fold": inner_validation_fold,
        "inner_fold_column": inner_fold_column,
        "n_biological_units": int(table["biological_unit_id"].nunique()),
        "n_adaptation_biological_units": int(
            table["eligible_for_reference_fitting"].sum()
        ),
        "n_validation_biological_units": int(
            (~table["eligible_for_reference_fitting"]).sum()
        ),
        "donors_by_dataset": {
            str(key): int(value)
            for key, value in table["dataset"].value_counts().sort_index().items()
        },
        "table": table_path.name,
        "table_sha256": sha256_file(table_path),
    }
    atomic_write_json(manifest_path, payload)
    return table_path, manifest_path


def _safe_cell_metadata(obs: pd.DataFrame, obs_names: Sequence[str]) -> pd.DataFrame:
    frame = add_stable_identifiers(obs, copy=True)
    source_cell_ids = pd.Series(
        list(map(str, obs_names)), index=frame.index, dtype="string"
    )
    if source_cell_ids.isna().any() or source_cell_ids.str.strip().eq("").any():
        raise ValueError("AnnData observation names must be nonempty")
    if source_cell_ids.str.contains("::", regex=False).any():
        raise ValueError("Source cell IDs cannot contain the '::' separator")
    if source_cell_ids.duplicated().any():
        raise ValueError("AnnData observation names must be unique")
    frame["source_cell_id"] = source_cell_ids
    frame["cell_key"] = frame["dataset"].astype("string") + "::" + source_cell_ids
    if frame["cell_key"].duplicated().any():
        raise ValueError("dataset::source_cell_id cell keys must be unique")
    if "ctype_low" in frame and "fine_type" not in frame:
        frame["fine_type"] = frame["ctype_low"]
    if "ctype_low_conf" in frame and "fine_type_confidence" not in frame:
        frame["fine_type_confidence"] = frame["ctype_low_conf"]
    frame["original_row_position"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _apply_approved_fine_types(
    cells: pd.DataFrame, ontology: Mapping[str, Any]
) -> pd.DataFrame:
    """Apply canonical fine types from immutable raw annotation columns."""

    required = {"lineage", "ctype_low", "ctype_low_conf"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(
            f"Approved fine-type mapping requires raw annotation columns: {missing}"
        )
    mapped = apply_fine_type_ontology(
        cells,
        ontology,
        lineage_column="lineage",
        fine_type_column="ctype_low",
        confidence_column="ctype_low_conf",
        output_column="fine_type",
    )
    # Keep the raw fields unchanged and provide an explicit numeric alias used by
    # aggregation confidence summaries.
    mapped["fine_type_confidence"] = mapped["ctype_low_conf"]
    if len(mapped) != len(cells) or not mapped.index.equals(cells.index):
        raise AssertionError("Fine-type ontology application changed cell rows")
    if (
        mapped["fine_type_state_eligible"].astype(bool).any()
        and not mapped.loc[
            mapped["fine_type_state_eligible"].astype(bool),
            "fine_type_balance_eligible",
        ]
        .astype(bool)
        .all()
    ):
        raise AssertionError("State-eligible fine types must be balance eligible")
    return mapped


def _fine_type_mapping_qc(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw-to-canonical mapping without dropping special cells."""

    grouping = [
        "dataset",
        "lineage",
        "ctype_low",
        "fine_type",
        "fine_type_mapping_status",
        "fine_type_state_eligible",
        "fine_type_balance_eligible",
    ]
    required = set(grouping) | {
        "biological_unit_id",
        "observation_id",
        "ctype_low_conf",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Fine-type mapping QC lacks columns: {missing}")
    frame = cells.copy()
    frame["_fine_type_confidence"] = pd.to_numeric(
        frame["ctype_low_conf"], errors="coerce"
    )
    return (
        frame.groupby(grouping, observed=True, dropna=False, sort=True)
        .agg(
            n_cells=("cell_key", "size"),
            n_biological_units=("biological_unit_id", "nunique"),
            n_observations=("observation_id", "nunique"),
            annotation_confidence_mean=("_fine_type_confidence", "mean"),
            annotation_confidence_min=("_fine_type_confidence", "min"),
        )
        .reset_index()
    )


def annotate_fold_cell_roles(
    obs: pd.DataFrame,
    obs_names: Sequence[str],
    fold_manifest: pd.DataFrame,
    visit_manifest: pd.DataFrame,
    *,
    reference_design: str = "lodo",
    repeated_visit_dataset: str = "terekhova",
    inner_validation_fold: int | None = None,
    global_one_visit_query: bool = True,
    fine_type_ontology: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach donor-fold roles and visit eligibility to every source cell."""

    if reference_design not in REFERENCE_DESIGNS:
        raise ValueError(f"reference_design must be one of {sorted(REFERENCE_DESIGNS)}")
    cells = _safe_cell_metadata(obs, obs_names)
    if fine_type_ontology is not None:
        cells = _apply_approved_fine_types(cells, fine_type_ontology)
    fold = fold_manifest.copy()
    required_fold = {"biological_unit_id", "dataset", "outer_role"}
    if reference_design == "lodo":
        required_fold.add("heldout_dataset")
    missing = sorted(required_fold - set(fold.columns))
    if missing:
        raise ValueError(f"Fold manifest is missing columns: {missing}")
    if fold["biological_unit_id"].duplicated().any():
        raise ValueError("Fold manifest must have one row per biological unit")
    heldouts = (
        fold["heldout_dataset"].dropna().astype(str).unique()
        if "heldout_dataset" in fold
        else np.asarray([], dtype=str)
    )
    if reference_design == "lodo" and len(heldouts) != 1:
        raise ValueError("LODO manifest must declare exactly one held-out dataset")
    if reference_design == "all_healthy" and len(heldouts):
        raise ValueError("all_healthy design cannot declare a held-out dataset")
    outer_values = set(fold["outer_role"].dropna().astype(str))
    allowed_outer = (
        {"reference", "query"} if reference_design == "lodo" else {"reference"}
    )
    if not outer_values <= allowed_outer:
        raise ValueError(f"Unsupported outer roles: {sorted(outer_values)}")

    donor_columns = ["biological_unit_id", "outer_role"]
    for optional in (
        "heldout_dataset",
        "fold_id",
        "inner_fold",
        "eligible_for_reference_fitting",
        "reference_partition",
    ):
        if optional in fold:
            donor_columns.append(optional)
    donor_roles = fold[donor_columns].copy()
    cells = cells.merge(
        donor_roles,
        on="biological_unit_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if cells["outer_role"].isna().any():
        examples = cells.loc[cells["outer_role"].isna(), "biological_unit_id"].head()
        raise ValueError(f"Cells lack donor fold assignments: {examples.tolist()}")
    cells = cells.sort_values("original_row_position").reset_index(drop=True)

    visit_required = {"observation_id", "selected_for_reference"}
    missing_visit = sorted(visit_required - set(visit_manifest.columns))
    if missing_visit:
        raise ValueError(f"Visit manifest is missing columns: {missing_visit}")
    visits = visit_manifest[list(visit_required)].copy()
    if visits["observation_id"].duplicated().any():
        raise ValueError("Visit manifest observation_id values must be unique")
    visits["selected_for_reference"] = _strict_bool(
        visits["selected_for_reference"], "selected_for_reference"
    )
    selected_map = visits.set_index("observation_id")["selected_for_reference"]
    is_repeated_dataset = cells["dataset"].astype(str).eq(repeated_visit_dataset)
    mapped = cells["observation_id"].map(selected_map)
    if mapped.loc[is_repeated_dataset].isna().any():
        examples = cells.loc[
            is_repeated_dataset & mapped.isna(), "observation_id"
        ].head()
        raise ValueError(f"Terekhova cells lack visit decisions: {examples.tolist()}")
    cells["selected_reference_visit"] = np.where(
        is_repeated_dataset, mapped.fillna(False), True
    ).astype(bool)

    is_query = cells["outer_role"].astype(str).eq("query")
    query_visit_allowed = ~is_repeated_dataset | cells["selected_reference_visit"]
    if not global_one_visit_query:
        query_visit_allowed = np.ones(len(cells), dtype=bool)
    reference_visit_allowed = ~is_repeated_dataset | cells["selected_reference_visit"]
    if inner_validation_fold is None:
        if (
            reference_design == "all_healthy"
            and "eligible_for_reference_fitting" in cells
        ):
            eligible = _strict_bool(
                cells["eligible_for_reference_fitting"],
                "eligible_for_reference_fitting",
            )
            is_inner_validation = (~eligible).to_numpy()
        else:
            is_inner_validation = np.zeros(len(cells), dtype=bool)
    else:
        if "inner_fold" not in cells:
            raise ValueError("inner_validation_fold requires inner_fold in split table")
        inner = pd.to_numeric(cells["inner_fold"], errors="coerce")
        is_inner_validation = inner.eq(int(inner_validation_fold)).to_numpy()

    role = np.full(len(cells), "excluded_nonselected_visit", dtype=object)
    role[is_query.to_numpy() & np.asarray(query_visit_allowed)] = "query"
    reference = ~is_query.to_numpy() & np.asarray(reference_visit_allowed)
    role[reference & is_inner_validation] = "validation"
    role[reference & ~is_inner_validation] = "adaptation"
    cells["preparation_role"] = role
    cells["eligible_for_feature_selection"] = role == "adaptation"
    cells["global_one_visit_query"] = bool(global_one_visit_query)
    cells["reference_design"] = reference_design

    if reference_design == "all_healthy" and (role == "query").any():
        raise AssertionError("Final all-healthy preparation cannot contain query cells")

    # Repeated visits may be query observations, but a biological donor can only
    # occupy one outer role.
    role_counts = cells.groupby("biological_unit_id", observed=True)[
        "outer_role"
    ].nunique()
    if (role_counts > 1).any():
        raise AssertionError("A donor appears in multiple outer fold roles")
    return cells


def _as_raw_csr(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        result = matrix.tocsr()
    else:
        result = sparse.csr_matrix(np.asarray(matrix))
    if result.data.size:
        if not np.isfinite(result.data).all() or (result.data < 0).any():
            raise ValueError("Counts contain nonfinite or negative values")
        if not np.allclose(result.data, np.rint(result.data), atol=1e-6, rtol=0):
            raise ValueError("Expression matrix does not contain integer-like counts")
    return result


def summarize_training_counts(
    adata: ad.AnnData,
    cell_metadata: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> DonorCountSummary:
    """Stream adaptation cells into equally weighted donor pseudobulks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if len(adata) != len(cell_metadata):
        raise ValueError("AnnData and cell metadata rows differ")
    mask = cell_metadata["eligible_for_feature_selection"].astype(bool).to_numpy()
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise ValueError("No adaptation cells are eligible for feature selection")
    selected = cell_metadata.iloc[positions]
    donors = tuple(sorted(selected["biological_unit_id"].astype(str).unique()))
    donor_lookup = {donor: index for index, donor in enumerate(donors)}
    donor_codes = (
        selected["biological_unit_id"].astype(str).map(donor_lookup).to_numpy()
    )
    donor_dataset_table = (
        selected[["biological_unit_id", "dataset"]]
        .drop_duplicates()
        .sort_values("biological_unit_id")
    )
    donor_dataset_counts = donor_dataset_table.groupby(
        "biological_unit_id", observed=True
    )["dataset"].nunique()
    if (donor_dataset_counts > 1).any():
        raise ValueError("A biological unit maps to multiple datasets")
    donor_datasets = tuple(donor_dataset_table["dataset"].astype(str))
    if len(donor_datasets) != len(donors):
        raise AssertionError("Donor/dataset lookup is incomplete")

    datasets = tuple(sorted(selected["dataset"].astype(str).unique()))
    dataset_lookup = {dataset: index for index, dataset in enumerate(datasets)}
    dataset_codes = selected["dataset"].astype(str).map(dataset_lookup).to_numpy()
    genes = tuple(map(str, adata.var_names))
    n_donors, n_genes = len(donors), len(genes)
    pseudobulk = np.zeros((n_donors, n_genes), dtype=np.float64)
    dataset_cell_counts = np.zeros(len(datasets), dtype=np.int64)
    dataset_gene_cell_counts = np.zeros((len(datasets), n_genes), dtype=np.int64)

    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        source_positions = positions[start:stop]
        counts = _as_raw_csr(read_csr_rows(Path(adata.filename), source_positions))
        local_donors = donor_codes[start:stop]
        membership = sparse.csr_matrix(
            (
                np.ones(len(local_donors), dtype=np.float64),
                (local_donors, np.arange(len(local_donors))),
            ),
            shape=(n_donors, len(local_donors)),
        )
        aggregated = (membership @ counts).tocoo()
        pseudobulk[aggregated.row, aggregated.col] += aggregated.data

        local_datasets = dataset_codes[start:stop]
        for dataset_code in np.unique(local_datasets):
            dataset_rows = np.flatnonzero(local_datasets == dataset_code)
            dataset_cell_counts[dataset_code] += len(dataset_rows)
            dataset_gene_cell_counts[dataset_code] += np.asarray(
                (counts[dataset_rows] > 0).getnnz(axis=0)
            ).ravel()

    if (pseudobulk.sum(axis=1) <= 0).any():
        bad = np.asarray(donors)[pseudobulk.sum(axis=1) <= 0]
        raise ValueError(
            f"Donor pseudobulks have zero library size: {bad[:5].tolist()}"
        )
    return DonorCountSummary(
        gene_ids=genes,
        donor_ids=donors,
        donor_datasets=donor_datasets,
        pseudobulk_counts=pseudobulk,
        datasets=datasets,
        dataset_cell_counts=dataset_cell_counts,
        dataset_gene_cell_counts=dataset_gene_cell_counts,
    )


def training_gene_statistics(summary: DonorCountSummary) -> pd.DataFrame:
    """Return cell- and donor-expression support by training dataset."""

    genes = np.asarray(summary.gene_ids, dtype=object)
    donor_datasets = np.asarray(summary.donor_datasets, dtype=object)
    rows: list[pd.DataFrame] = []
    for dataset_index, dataset in enumerate(summary.datasets):
        donor_mask = donor_datasets == dataset
        n_donors = int(donor_mask.sum())
        if not n_donors or summary.dataset_cell_counts[dataset_index] <= 0:
            raise ValueError(f"Training dataset {dataset!r} has no cells or donors")
        donor_counts = np.count_nonzero(
            summary.pseudobulk_counts[donor_mask] > 0, axis=0
        )
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "gene": genes,
                    "n_cells": int(summary.dataset_cell_counts[dataset_index]),
                    "n_donors": n_donors,
                    "expression_coverage": (
                        summary.dataset_gene_cell_counts[dataset_index]
                        / summary.dataset_cell_counts[dataset_index]
                    ),
                    "donor_coverage": donor_counts / n_donors,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _mean_bin_zscore(
    means: np.ndarray, variances: np.ndarray, gene_ids: np.ndarray, n_bins: int
) -> np.ndarray:
    """Standardise donor variance within deterministic mean-expression bins."""

    order = np.lexsort((gene_ids, means))
    bins = np.empty(len(means), dtype=np.int64)
    bins[order] = np.minimum(
        (np.arange(len(means), dtype=np.int64) * n_bins) // max(len(means), 1),
        n_bins - 1,
    )
    result = np.zeros(len(means), dtype=np.float64)
    for bin_number in range(n_bins):
        mask = bins == bin_number
        values = np.log1p(variances[mask])
        if not len(values):
            continue
        scale = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if not np.isfinite(scale) or scale == 0:
            result[mask] = 0.0
        else:
            result[mask] = (values - values.mean()) / scale
    return result


def select_donor_dataset_aware_hvgs(
    summary: DonorCountSummary,
    *,
    mean_bins: int = 20,
    minimum_donor_fraction: float = 0.01,
    minimum_dataset_fraction: float = 0.75,
) -> DonorAwareHVGSelection:
    """Rank genes from donor pseudobulks, giving each dataset equal weight.

    Raw counts are summed within donor, library-normalised to 10,000, and log1p
    transformed.  Within each dataset, donor variance is standardised inside
    mean-expression bins.  Dataset-specific percentile ranks are then averaged,
    so a large cohort cannot dominate the ranking by contributing more cells or
    donors.
    """

    if mean_bins < 2:
        raise ValueError("mean_bins must be at least two")
    for value in (minimum_donor_fraction, minimum_dataset_fraction):
        if not 0 <= value <= 1:
            raise ValueError("HVG coverage fractions must be between zero and one")
    counts = np.asarray(summary.pseudobulk_counts, dtype=np.float64)
    library_sizes = counts.sum(axis=1)
    normalized = np.log1p(counts / library_sizes[:, None] * 10_000.0)
    genes = np.asarray(summary.gene_ids, dtype=object)
    donor_datasets = np.asarray(summary.donor_datasets, dtype=object)
    component_rows: list[pd.DataFrame] = []
    percentile_scores = np.zeros((len(summary.datasets), len(genes)), dtype=float)
    donor_coverages = np.zeros_like(percentile_scores)

    for dataset_index, dataset in enumerate(summary.datasets):
        donor_mask = donor_datasets == dataset
        values = normalized[donor_mask]
        raw_values = counts[donor_mask]
        means = values.mean(axis=0)
        variances = (
            values.var(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(genes))
        )
        coverage = np.mean(raw_values > 0, axis=0)
        zscore = _mean_bin_zscore(means, variances, genes, mean_bins)
        eligible = coverage >= minimum_donor_fraction
        eligible_positions = np.flatnonzero(eligible)
        order = np.lexsort((genes[eligible_positions], -zscore[eligible_positions]))
        ranked = eligible_positions[order]
        if len(ranked) == 1:
            percentile_scores[dataset_index, ranked] = 1.0
        elif len(ranked) > 1:
            percentile_scores[dataset_index, ranked] = 1.0 - (
                np.arange(len(ranked), dtype=float) / (len(ranked) - 1)
            )
        donor_coverages[dataset_index] = coverage
        component_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "gene": genes,
                    "n_donors": int(donor_mask.sum()),
                    "donor_log1p_cptt_mean": means,
                    "donor_log1p_cptt_variance": variances,
                    "mean_bin_variance_zscore": zscore,
                    "donor_coverage": coverage,
                    "eligible_in_dataset": eligible,
                    "dataset_percentile_score": percentile_scores[dataset_index],
                }
            )
        )

    required_datasets = max(
        1, int(math.ceil(minimum_dataset_fraction * len(summary.datasets)))
    )
    supported = donor_coverages >= minimum_donor_fraction
    n_supported = supported.sum(axis=0)
    eligible = n_supported >= required_datasets
    aggregate_score = percentile_scores.mean(axis=0)
    eligible_positions = np.flatnonzero(eligible)
    order = np.lexsort(
        (genes[eligible_positions], -aggregate_score[eligible_positions])
    )
    ranked_positions = eligible_positions[order]
    ranked_genes = tuple(genes[ranked_positions].tolist())
    aggregate_rank = np.full(len(genes), np.nan)
    aggregate_rank[ranked_positions] = np.arange(1, len(ranked_positions) + 1)
    scores = pd.DataFrame(
        {
            "gene": genes,
            "eligible": eligible,
            "n_training_datasets_supported": n_supported,
            "required_training_datasets": required_datasets,
            "minimum_donor_coverage": donor_coverages.min(axis=0),
            "median_donor_coverage": np.median(donor_coverages, axis=0),
            "mean_dataset_percentile_score": aggregate_score,
            "hvg_rank": pd.Series(aggregate_rank, dtype="Float64"),
        }
    ).sort_values(
        ["eligible", "mean_dataset_percentile_score", "gene"],
        ascending=[False, False, True],
    )
    return DonorAwareHVGSelection(
        ranked_genes=ranked_genes,
        scores=scores.reset_index(drop=True),
        dataset_scores=pd.concat(component_rows, ignore_index=True),
    )


def _write_text_atomic(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    suffix = ".parquet" if path.suffix == ".parquet" else path.suffix
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix in {".tsv", ".txt"}:
            frame.to_csv(temporary, sep="\t", index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported atomic table format: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_programs(
    programs: Iterable[GeneProgram], output_dir: Path
) -> dict[str, Path]:
    ordered = tuple(sorted(programs, key=lambda program: program.program_id))
    if not ordered:
        raise ValueError("Training-only filtering retained no gene programs")
    gmt = output_dir / "gene_programs_filtered.gmt"
    gpdb = output_dir / "gpdb_filtered.csv"
    terms = output_dir / "gene_program_terms.tsv"
    gmt_content = "".join(
        "\t".join((program.program_id, "training_fold_filtered", *program.genes)) + "\n"
        for program in ordered
    )
    _write_text_atomic(gmt, gmt_content)
    gpdb_frame = pd.DataFrame(
        {
            program.program_id: pd.Series(program.genes, dtype="string")
            for program in ordered
        }
    )
    _write_table_atomic(gpdb_frame, gpdb)
    _write_table_atomic(
        pd.DataFrame(
            {
                "program_id": [program.program_id for program in ordered],
                "source": [program.source for program in ordered],
                "category": [program.category for program in ordered],
                "direction": [program.direction for program in ordered],
                "n_genes": [len(program.genes) for program in ordered],
            }
        ),
        terms,
    )
    return {"gmt": gmt, "gpdb": gpdb, "terms": terms}


def _ordered_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _table_content_digest(frame: pd.DataFrame) -> str:
    """Hash table content independently of row and column ordering."""

    columns = sorted(map(str, frame.columns))
    canonical = frame.loc[:, columns].copy()
    for column in columns:
        canonical[column] = canonical[column].astype("string").fillna("<NA>")
    canonical = canonical.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_donor_metadata(
    cells: pd.DataFrame,
    summary: DonorCountSummary,
    *,
    lineage: str,
) -> pd.DataFrame:
    """Return one stable metadata row for each adaptation pseudobulk."""

    selected = cells.loc[cells["eligible_for_feature_selection"].astype(bool)].copy()
    required = {"biological_unit_id", "dataset", "age", "sex"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(
            "Training-only GP candidate selection requires donor metadata columns: "
            f"{missing}"
        )
    for column in ("dataset", "age", "sex"):
        varying = selected.groupby("biological_unit_id", observed=True)[column].nunique(
            dropna=False
        )
        if varying.gt(1).any():
            examples = varying.loc[varying.gt(1)].index.astype(str).tolist()[:5]
            raise ValueError(f"{column} varies within adaptation donor(s): {examples}")
    donor = (
        selected.groupby("biological_unit_id", observed=True, sort=False)
        .agg(
            dataset=("dataset", "first"),
            age=("age", "first"),
            sex=("sex", "first"),
            n_lineage_cells=("biological_unit_id", "size"),
        )
        .reindex(summary.donor_ids)
        .reset_index()
    )
    if donor[["dataset", "age", "sex", "n_lineage_cells"]].isna().any().any():
        raise ValueError("Adaptation donor pseudobulks lack complete age/sex metadata")
    if tuple(donor["biological_unit_id"].astype(str)) != summary.donor_ids:
        raise AssertionError("Donor candidate-score metadata order is not reproducible")
    if tuple(donor["dataset"].astype(str)) != summary.donor_datasets:
        raise AssertionError("Donor candidate-score dataset order differs from counts")
    donor["age"] = pd.to_numeric(donor["age"], errors="raise")
    donor["lineage"] = str(lineage)
    donor["library_size"] = summary.pseudobulk_counts.sum(axis=1)
    return donor


def _write_projection_gp_candidates(
    summary: DonorCountSummary,
    cells: pd.DataFrame,
    programs: Sequence[GeneProgram],
    output_dir: Path,
    *,
    lineage: str,
    reference_design: str,
    heldout_dataset: str | None,
    config: ReferenceFeatureConfig,
    gpdb_path: Path,
    source_gene_program_path: Path,
    fold_manifest: pd.DataFrame,
    visit_manifest: pd.DataFrame,
    fine_type_ontology_identity: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, Path]]:
    """Create a permissive training-only storage gate for frozen projection.

    This broad-lineage pseudobulk screen limits which learned GP vectors are
    persisted for every cell.  It is deliberately not the final fine-type-level
    biological selection, which is performed after reference projection.
    """

    ordered_programs = tuple(sorted(programs, key=lambda item: item.program_id))
    program_ids = tuple(program.program_id for program in ordered_programs)
    if not program_ids or len(program_ids) != len(set(program_ids)):
        raise ValueError("Filtered GP programs must have unique nonempty IDs")
    controls = tuple(map(str, config.gp_projection_control_ids))
    missing_controls = sorted(set(controls) - set(program_ids))
    if missing_controls:
        raise ValueError(
            "Prespecified projection controls did not survive training-only GP "
            f"filtering: {missing_controls}"
        )

    donor = _training_donor_metadata(cells, summary, lineage=lineage)
    scores = score_gene_programs(
        sparse.csr_matrix(summary.pseudobulk_counts),
        summary.gene_ids,
        {program.program_id: program.genes for program in ordered_programs},
        method="mean_log_cpm",
        cpm_scale=1_000_000.0,
        minimum_genes=config.gp_minimum_mapped_genes,
    ).merge(
        donor.reset_index(names="summary_index"),
        on="summary_index",
        how="left",
        validate="many_to_one",
    )
    if scores["dataset"].isna().any():
        raise AssertionError("GP candidate scores failed to join donor metadata")
    transfer_config = TransferabilityConfig(
        minimum_donors_per_cohort=config.gp_transfer_minimum_donors_per_cohort,
        minimum_age_span=config.gp_transfer_minimum_age_span,
        minimum_cohorts=config.gp_transfer_minimum_cohorts,
        minimum_sign_concordance=config.gp_transfer_minimum_sign_concordance,
        maximum_i2=config.gp_transfer_maximum_i2,
        maximum_fdr=config.gp_transfer_maximum_fdr,
        minimum_absolute_standardized_slope_per_decade=(
            config.gp_transfer_minimum_absolute_standardized_slope_per_decade
        ),
    )
    transfer = select_transferable_gene_programs(
        scores,
        strata_columns=("lineage",),
        training_datasets=summary.datasets,
        excluded_datasets=(),
        config=transfer_config,
    )
    selected_by_effect = set(
        transfer.selection.loc[transfer.selection["retained"], "gp_id"].astype(str)
    )
    candidate_set = selected_by_effect | set(controls)
    ordered_candidates = tuple(
        program_id for program_id in program_ids if program_id in candidate_set
    )
    if not ordered_candidates:
        raise ValueError(
            "Training-only GP candidate prefilter retained zero programs. Projection "
            "is blocked: review prespecified transferability thresholds or declare "
            "explicit control IDs; no data-driven fallback is permitted."
        )

    selection_by_program = transfer.selection.set_index("gp_id", drop=False)
    allowlist_rows: list[dict[str, object]] = []
    for program_id in ordered_candidates:
        selection_row = selection_by_program.loc[program_id]
        if isinstance(selection_row, pd.DataFrame):
            raise AssertionError("Projection candidate selection duplicated a GP ID")
        allowlist_rows.append(
            {
                "program_id": program_id,
                "gpdb_column_index": program_ids.index(program_id),
                "selected_by_transferability": program_id in selected_by_effect,
                "prespecified_control": program_id in controls,
                "selection_level": "donor_lineage_pseudobulk",
                "meta_age_slope_per_year": selection_row["meta_age_slope_per_year"],
                "meta_fdr": selection_row["meta_fdr"],
                "heterogeneity_i2": selection_row["heterogeneity_i2"],
                "sign_concordance": selection_row["sign_concordance"],
                "n_cohorts_eligible": selection_row["n_cohorts_eligible"],
            }
        )
    allowlist = pd.DataFrame.from_records(allowlist_rows)

    paths = {
        "scores": output_dir / "simple_gp_donor_scores.parquet",
        "effects": output_dir / "simple_gp_age_effects.parquet",
        "selection": output_dir / "simple_gp_transferability.parquet",
        "allowlist": output_dir / "projection_gp_candidates.tsv",
        "manifest": output_dir / "projection_gp_candidates.json",
    }
    _write_table_atomic(scores, paths["scores"])
    _write_table_atomic(transfer.effects, paths["effects"])
    _write_table_atomic(transfer.selection, paths["selection"])
    _write_table_atomic(allowlist, paths["allowlist"])

    training_cells = cells.loc[cells["eligible_for_feature_selection"].astype(bool)]
    transfer_payload = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in transfer_config.__dict__.items()
    }
    payload: dict[str, object] = {
        "schema_version": PROJECTION_GP_CANDIDATE_SCHEMA,
        "selection_level": "donor_lineage_pseudobulk",
        "purpose": "training_only_projection_storage_gate",
        "final_inference_level": "donor_observation_lineage_fine_type_gp",
        "final_fine_type_selection_required": True,
        "reference_design": reference_design,
        "lineage": str(lineage),
        "heldout_dataset": heldout_dataset,
        "training_datasets": list(summary.datasets),
        "query_data_consulted": False,
        "score_method": "mean_log_cpm",
        "score_scale": 1_000_000.0,
        "transferability_config": transfer_payload,
        "prespecified_control_ids": list(controls),
        "candidate_union_rule": (
            "transferability_retained_union_prespecified_controls"
        ),
        "zero_candidate_policy": "fail_closed",
        "n_filtered_programs": len(program_ids),
        "n_transferability_selected": len(selected_by_effect),
        "n_prespecified_controls_selected": len(set(controls)),
        "n_projection_candidates": len(ordered_candidates),
        "program_ids": list(ordered_candidates),
        # This digest deliberately uses the canonical JSON list representation;
        # tokenization and projection validate the exact same ordered payload.
        "program_ids_ordered_sha256": _canonical_json_digest(list(ordered_candidates)),
        "binding": {
            "training_biological_unit_ids_ordered_sha256": _ordered_digest(
                summary.donor_ids
            ),
            "training_cell_key_ordered_sha256": _ordered_digest(
                training_cells["cell_key"].astype(str)
            ),
            "gpdb_path": gpdb_path.name,
            "gpdb_sha256": sha256_file(gpdb_path),
            "source_gene_program_sha256": sha256_file(source_gene_program_path),
            "fold_table_content_sha256": _table_content_digest(fold_manifest),
            "visit_table_content_sha256": _table_content_digest(visit_manifest),
            "fine_type_ontology_sha256": fine_type_ontology_identity["sha256"],
        },
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    payload["manifest_content_sha256"] = _canonical_json_digest(payload)
    atomic_write_json(paths["manifest"], payload)
    return payload, paths


def prepare_fold_features(
    input_h5ad: Path,
    fold_manifest: pd.DataFrame,
    visit_manifest: pd.DataFrame,
    gene_program_path: Path,
    output_dir: Path,
    *,
    lineage: str,
    fine_type_ontology_path: Path,
    reference_design: str = "lodo",
    config: ReferenceFeatureConfig = ReferenceFeatureConfig(),
    chunk_size: int = 20_000,
    inner_validation_fold: int | None = None,
    global_one_visit_query: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Learn adaptation-only programs and HVGs and write frozen vocabularies."""

    if reference_design not in REFERENCE_DESIGNS:
        raise ValueError(f"reference_design must be one of {sorted(REFERENCE_DESIGNS)}")
    config.validate()
    input_h5ad = Path(input_h5ad).resolve()
    gene_program_path = Path(gene_program_path).resolve()
    fine_type_ontology_path = Path(fine_type_ontology_path).resolve()
    fine_type_ontology = load_fine_type_ontology(
        fine_type_ontology_path, require_approved=True
    )
    ontology_identity = approved_ontology_identity(fine_type_ontology) | {
        "sha256": sha256_file(fine_type_ontology_path)
    }
    output_dir = Path(output_dir)
    manifest_path = output_dir / "feature_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Feature preparation is already complete: {manifest_path}"
        )
    if not input_h5ad.is_file() or not gene_program_path.is_file():
        raise FileNotFoundError("Input H5AD and gene-program GMT must both exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        cells = annotate_fold_cell_roles(
            adata.obs,
            adata.obs_names,
            fold_manifest,
            visit_manifest,
            reference_design=reference_design,
            inner_validation_fold=inner_validation_fold,
            global_one_visit_query=global_one_visit_query,
            fine_type_ontology=fine_type_ontology,
        )
        summary = summarize_training_counts(adata, cells, chunk_size=chunk_size)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    gene_stats = training_gene_statistics(summary)
    programs = load_gene_programs(gene_program_path)
    heldout_values = (
        fold_manifest["heldout_dataset"].dropna().astype(str).unique()
        if "heldout_dataset" in fold_manifest
        else np.asarray([], dtype=str)
    )
    if reference_design == "lodo":
        if len(heldout_values) != 1:
            raise ValueError("LODO manifest must identify one held-out dataset")
        heldout: str | None = str(heldout_values[0])
    else:
        if len(heldout_values):
            raise ValueError("all_healthy design cannot identify a held-out dataset")
        heldout = None
    training_datasets = tuple(summary.datasets)
    gp_config = GPFilterConfig(
        minimum_mapped_genes=config.gp_minimum_mapped_genes,
        maximum_program_size=config.gp_maximum_program_size,
        minimum_expression_coverage=config.gp_minimum_expression_coverage,
        minimum_donor_coverage=config.gp_minimum_donor_coverage,
        minimum_dataset_fraction=config.gp_minimum_dataset_fraction,
        redundancy_jaccard_threshold=config.gp_redundancy_jaccard_threshold,
    )
    filtered = filter_gene_programs_training_only(
        programs,
        gene_stats,
        training_datasets=training_datasets,
        heldout_dataset=heldout,
        config=gp_config,
        lineage=lineage,
    )
    hvg = select_donor_dataset_aware_hvgs(
        summary,
        mean_bins=config.hvg_mean_bins,
        minimum_donor_fraction=config.hvg_minimum_donor_fraction,
        minimum_dataset_fraction=config.hvg_minimum_dataset_fraction,
    )
    if len(hvg.ranked_genes) < max(config.hvg_sizes):
        raise ValueError(
            f"Only {len(hvg.ranked_genes)} genes pass HVG support; "
            f"{max(config.hvg_sizes)} requested"
        )

    gp_genes = {gene for program in filtered.programs for gene in program.genes}
    source_genes = tuple(summary.gene_ids)
    source_gene_set = set(source_genes)
    missing_gp = sorted(gp_genes - source_gene_set)
    if missing_gp:
        raise AssertionError(
            f"Filtered GP genes are absent from source: {missing_gp[:5]}"
        )

    paths: dict[str, Path] = {
        "cell_metadata": output_dir / "cell_metadata.parquet",
        "fine_type_mapping_qc": output_dir / "fine_type_mapping_qc.tsv",
        "training_gene_statistics": output_dir / "training_gene_statistics.parquet",
        "gp_filter_report": output_dir / "gene_program_filter_report.parquet",
        "gp_gene_support": output_dir / "gene_program_gene_support.parquet",
        "hvg_scores": output_dir / "hvg_scores.parquet",
        "hvg_dataset_scores": output_dir / "hvg_dataset_scores.parquet",
    }
    _write_table_atomic(cells, paths["cell_metadata"])
    _write_table_atomic(_fine_type_mapping_qc(cells), paths["fine_type_mapping_qc"])
    _write_table_atomic(gene_stats, paths["training_gene_statistics"])
    _write_table_atomic(filtered.report, paths["gp_filter_report"])
    _write_table_atomic(filtered.gene_support, paths["gp_gene_support"])
    _write_table_atomic(hvg.scores, paths["hvg_scores"])
    _write_table_atomic(hvg.dataset_scores, paths["hvg_dataset_scores"])
    program_paths = _write_programs(filtered.programs, output_dir)
    candidate_manifest, candidate_paths = _write_projection_gp_candidates(
        summary,
        cells,
        filtered.programs,
        output_dir,
        lineage=lineage,
        reference_design=reference_design,
        heldout_dataset=heldout,
        config=config,
        gpdb_path=program_paths["gpdb"],
        source_gene_program_path=gene_program_path,
        fold_manifest=fold_manifest,
        visit_manifest=visit_manifest,
        fine_type_ontology_identity=ontology_identity,
    )

    ontology_snapshot = output_dir / "fine_type_ontology.approved.yaml"
    _write_text_atomic(ontology_snapshot, fine_type_ontology_path.read_text())
    if sha256_file(ontology_snapshot) != ontology_identity["sha256"]:
        raise AssertionError("Fine-type ontology snapshot differs from source")
    source_candidate_record = fine_type_ontology.get("source_candidate", {})
    source_candidate = fine_type_ontology_path.parent / str(
        source_candidate_record.get("path", "")
    )
    candidate_snapshot = output_dir / source_candidate.name
    _write_text_atomic(candidate_snapshot, source_candidate.read_text())
    if sha256_file(candidate_snapshot) != ontology_identity["source_candidate_sha256"]:
        raise AssertionError("Fine-type source-candidate snapshot differs")

    vocabularies: dict[str, dict[str, object]] = {}
    rank_lookup = {gene: rank for rank, gene in enumerate(hvg.ranked_genes, start=1)}
    for size in config.hvg_sizes:
        selected_hvgs = set(hvg.ranked_genes[:size])
        union = tuple(
            gene for gene in source_genes if gene in selected_hvgs or gene in gp_genes
        )
        hvg_path = output_dir / f"hvg{size}.txt"
        union_path = output_dir / f"model_genes_hvg{size}.txt"
        _write_text_atomic(
            hvg_path,
            "".join(f"{gene}\n" for gene in hvg.ranked_genes[:size]),
        )
        _write_text_atomic(union_path, "".join(f"{gene}\n" for gene in union))
        vocabularies[str(size)] = {
            "hvg_path": hvg_path.name,
            "hvg_sha256": sha256_file(hvg_path),
            "model_genes_path": union_path.name,
            "model_genes_sha256": sha256_file(union_path),
            "n_hvgs": int(size),
            "n_gp_genes": len(gp_genes),
            "n_hvg_gp_overlap": len(selected_hvgs & gp_genes),
            "n_model_genes": len(union),
            "gene_order": "source_h5ad_var_order",
        }
    gene_membership = pd.DataFrame(
        {
            "gene": source_genes,
            "source_position": np.arange(len(source_genes), dtype=np.int64),
            "is_retained_gp_gene": [gene in gp_genes for gene in source_genes],
            "hvg_rank": [rank_lookup.get(gene, pd.NA) for gene in source_genes],
        }
    )
    for size in config.hvg_sizes:
        gene_membership[f"selected_hvg{size}"] = gene_membership["hvg_rank"].apply(
            lambda value: pd.notna(value) and int(value) <= size
        )
        gene_membership[f"model_gene_hvg{size}"] = (
            gene_membership["is_retained_gp_gene"]
            | gene_membership[f"selected_hvg{size}"]
        )
    membership_path = output_dir / "model_gene_membership.parquet"
    _write_table_atomic(gene_membership, membership_path)

    role_counts = cells["preparation_role"].value_counts().sort_index()
    lineage_donor_scope = _build_lineage_donor_scope(
        cells,
        fold_manifest,
        lineage=lineage,
    )
    payload: dict[str, object] = {
        "schema_version": FEATURE_MANIFEST_SCHEMA,
        "reference_design": reference_design,
        "lineage": lineage,
        "heldout_dataset": heldout,
        "training_datasets": list(training_datasets),
        "inner_validation_fold": inner_validation_fold,
        "input_h5ad": str(input_h5ad),
        "input_h5ad_size_bytes": input_h5ad.stat().st_size,
        "input_h5ad_mtime_ns": input_h5ad.stat().st_mtime_ns,
        "source_shape": [len(cells), len(source_genes)],
        "all_source_cells_preserved_in_cell_manifest": True,
        "raw_fine_type_columns_preserved": ["ctype_low", "ctype_low_conf"],
        "cell_downsampling_performed": False,
        "fine_type_ontology": ontology_identity
        | {
            "path": ontology_snapshot.name,
            "source_path": str(fine_type_ontology_path),
            "state_eligibility_column": "fine_type_state_eligible",
            "balance_eligibility_column": "fine_type_balance_eligible",
            "special_categories_retained_in_composition": True,
            "special_categories_state_eligible": False,
            "special_categories_balance_eligible": False,
        },
        "terekhova_reference_policy": "one_seeded_hash_selected_observation_per_donor",
        "terekhova_query_policy": (
            "not_applicable_no_query"
            if reference_design == "all_healthy"
            else (
                "one_seeded_hash_selected_observation_per_donor"
                if global_one_visit_query
                else "all_query_observations"
            )
        ),
        "cell_counts_by_preparation_role": {
            str(key): int(value) for key, value in role_counts.items()
        },
        # The donor split is global, whereas each merged lineage can physically
        # omit donors with zero cells.  Downstream fold binding must use this
        # immutable lineage inventory rather than expecting all global donors.
        "lineage_donor_scope": lineage_donor_scope,
        "n_training_donors": len(summary.donor_ids),
        "training_donors_by_dataset": {
            str(key): int(value)
            for key, value in pd.Series(summary.donor_datasets)
            .value_counts()
            .sort_index()
            .items()
        },
        "training_cell_key_ordered_sha256": _ordered_digest(
            cells.loc[cells["eligible_for_feature_selection"], "cell_key"]
        ),
        "all_cell_key_ordered_sha256": _ordered_digest(cells["cell_key"]),
        "tripso_metadata_transport": {
            "required_safe_string_key": "cell_key",
            "custom_attr_columns": ["cell_key"],
            "external_metadata_join_required": True,
            "warning": (
                "Do not transport string metadata ending in _id through the vendor "
                "datamodule; it casts such fields to integer tensors."
            ),
        },
        "feature_config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in config.__dict__.items()
        },
        "hvg_method": (
            "donor pseudobulk log1p(counts-per-10k), variance standardised in "
            "mean bins per dataset, equal-weight mean dataset percentile"
        ),
        "n_input_gene_programs": len(programs),
        "n_retained_gene_programs": len(filtered.programs),
        "n_retained_gp_genes": len(gp_genes),
        "n_projection_gp_candidates": candidate_manifest["n_projection_candidates"],
        "projection_gp_candidate_program_ids_ordered_sha256": candidate_manifest[
            "program_ids_ordered_sha256"
        ],
        "projection_gp_candidate_manifest_content_sha256": candidate_manifest[
            "manifest_content_sha256"
        ],
        "vocabularies": vocabularies,
        "files": {key: path.name for key, path in paths.items()}
        | {
            "gene_programs_filtered_gmt": program_paths["gmt"].name,
            "gpdb_filtered_csv": program_paths["gpdb"].name,
            "gene_program_terms": program_paths["terms"].name,
            "model_gene_membership": membership_path.name,
            "simple_gp_donor_scores": candidate_paths["scores"].name,
            "simple_gp_age_effects": candidate_paths["effects"].name,
            "simple_gp_transferability": candidate_paths["selection"].name,
            "projection_gp_candidates_tsv": candidate_paths["allowlist"].name,
            "projection_gp_candidates_json": candidate_paths["manifest"].name,
            "fine_type_ontology": ontology_snapshot.name,
            "fine_type_ontology_source_candidate": candidate_snapshot.name,
        },
        "projection_gp_candidate_hashes": {
            key: sha256_file(path) for key, path in candidate_paths.items()
        },
        "input_hashes": {
            "gene_program_sha256": sha256_file(gene_program_path),
            "fold_table_content_sha256": _table_content_digest(fold_manifest),
            "visit_table_content_sha256": _table_content_digest(visit_manifest),
            "fine_type_ontology_sha256": ontology_identity["sha256"],
        },
    }
    atomic_write_json(manifest_path, payload)
    return payload


def _read_gene_list(path: Path) -> tuple[str, ...]:
    genes = tuple(
        line.strip() for line in path.read_text().splitlines() if line.strip()
    )
    if not genes or len(genes) != len(set(genes)):
        raise ValueError(f"Gene list is empty or duplicated: {path}")
    return genes


def _write_ensembl_var_column(path: Path, genes: Sequence[str]) -> None:
    """Add the tokenizer-required string column without loading the full H5AD.

    ``anndata.experimental.concat_on_disk`` in the pinned environment cannot merge
    arbitrary ``var`` Series.  The standard H5AD dataframe/string encodings are
    small and can be written directly after the count matrix has been concatenated.
    """

    with h5py.File(path, "r+") as handle:
        var = handle["var"]
        if "ensembl_id" in var:
            del var["ensembl_id"]
        dataset = var.create_dataset(
            "ensembl_id",
            data=np.asarray(list(genes), dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset.attrs["encoding-type"] = "string-array"
        dataset.attrs["encoding-version"] = "0.2.0"
        existing = [
            str(value.decode() if isinstance(value, bytes) else value)
            for value in var.attrs.get("column-order", [])
            if str(value.decode() if isinstance(value, bytes) else value)
            != "ensembl_id"
        ]
        var.attrs["column-order"] = np.asarray(
            [*existing, "ensembl_id"], dtype=h5py.string_dtype(encoding="utf-8")
        )


def materialize_fold_h5ad(
    input_h5ad: Path,
    preparation_dir: Path,
    output_h5ad: Path,
    *,
    role: str,
    hvg_size: int = 9000,
    row_chunk_size: int = 25_000,
    max_loaded_elements: int = 100_000_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write a memory-bounded, role-specific H5AD without cell downsampling."""

    if role not in {"adaptation", "validation", "query"}:
        raise ValueError("role must be adaptation, validation, or query")
    if row_chunk_size < 1 or max_loaded_elements < 1:
        raise ValueError("Chunk settings must be positive")
    input_h5ad = Path(input_h5ad).resolve()
    preparation_dir = Path(preparation_dir).resolve()
    output_h5ad = Path(output_h5ad).resolve()
    output_manifest = output_h5ad.with_suffix(".manifest.json")
    if (output_h5ad.exists() or output_manifest.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite materialized output: {output_h5ad}"
        )
    with (preparation_dir / "feature_manifest.json").open(encoding="utf-8") as handle:
        feature_manifest = json.load(handle)
    if feature_manifest.get("schema_version") != FEATURE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported feature preparation manifest")
    if Path(feature_manifest["input_h5ad"]).resolve() != input_h5ad:
        raise ValueError("Materialization input differs from feature-selection H5AD")
    lineage_donor_scope_value = feature_manifest.get("lineage_donor_scope")
    if not isinstance(lineage_donor_scope_value, Mapping):
        raise ValueError(
            "Feature manifest lacks the lineage-specific donor scope; rerun "
            "feature preparation with the current code"
        )
    lineage_donor_scope = validate_lineage_donor_scope(
        lineage_donor_scope_value,
        lineage=str(feature_manifest.get("lineage", "")),
    )
    ontology_record = feature_manifest.get("fine_type_ontology")
    if not isinstance(ontology_record, Mapping):
        raise ValueError("Feature manifest lacks its approved fine-type ontology")
    ontology_snapshot = preparation_dir / str(ontology_record.get("path", ""))
    if not ontology_snapshot.is_file() or sha256_file(ontology_snapshot) != str(
        ontology_record.get("sha256", "")
    ):
        raise ValueError("Prepared fine-type ontology is missing or changed")
    load_fine_type_ontology(ontology_snapshot, require_approved=True)
    vocabulary = feature_manifest["vocabularies"].get(str(int(hvg_size)))
    if vocabulary is None:
        raise ValueError(f"HVG size {hvg_size} was not prepared")
    gene_path = preparation_dir / vocabulary["model_genes_path"]
    if sha256_file(gene_path) != vocabulary["model_genes_sha256"]:
        raise ValueError("Prepared model-gene list hash does not match its manifest")
    genes = _read_gene_list(gene_path)
    cells = pd.read_parquet(
        preparation_dir / feature_manifest["files"]["cell_metadata"]
    )
    selected = cells.loc[cells["preparation_role"].eq(role)].copy()
    if selected.empty:
        raise ValueError(f"No cells have preparation role {role!r}")
    observed_role_donors = sorted(set(selected["biological_unit_id"].astype(str)))
    expected_role_donors = lineage_donor_scope[
        "biological_unit_ids_by_preparation_role"
    ][role]
    if observed_role_donors != expected_role_donors:
        raise ValueError(
            "Prepared cell metadata differs from its lineage donor scope for "
            f"role {role!r}"
        )
    required_fine_type_columns = {
        "ctype_low",
        "ctype_low_conf",
        "fine_type",
        "fine_type_state_eligible",
        "fine_type_balance_eligible",
    }
    missing_fine_type_columns = sorted(
        required_fine_type_columns - set(selected.columns)
    )
    if missing_fine_type_columns:
        raise ValueError(
            "Prepared cells lack approved fine-type columns: "
            f"{missing_fine_type_columns}"
        )
    if (
        _ordered_digest(cells["cell_key"])
        != feature_manifest["all_cell_key_ordered_sha256"]
    ):
        raise ValueError("Prepared cell metadata key order does not match its manifest")
    positions = selected["original_row_position"].to_numpy(dtype=np.int64)
    if not np.all(np.diff(positions) > 0):
        raise ValueError("Selected source positions must be strictly increasing")

    source = ad.read_h5ad(input_h5ad, backed="r")
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_h5ad.stem}.shards.", dir=output_h5ad.parent)
    )
    temporary_output = output_h5ad.parent / f".{output_h5ad.stem}.partial.h5ad"
    try:
        if len(source) != len(cells):
            raise ValueError("Source H5AD and prepared cell metadata rows differ")
        source_keys = np.asarray(source.obs_names, dtype=str)
        if not np.array_equal(
            source_keys, cells["source_cell_id"].astype(str).to_numpy()
        ):
            raise ValueError(
                "Prepared source_cell_id order no longer matches source H5AD"
            )
        source_gene_set = set(map(str, source.var_names))
        missing = [gene for gene in genes if gene not in source_gene_set]
        if missing:
            raise ValueError(f"Prepared genes are absent from source: {missing[:5]}")
        gene_positions = source.var_names.get_indexer(genes)
        if (gene_positions < 0).any():
            raise AssertionError("Prepared gene lookup is incomplete")
        shard_paths: list[Path] = []
        for shard_number, start in enumerate(range(0, len(positions), row_chunk_size)):
            stop = min(start + row_chunk_size, len(positions))
            shard_positions = positions[start:stop]
            # Geneformer's rank-value tokenizer normalises by each cell's total
            # library size.  That denominator must come from the complete merged
            # count matrix, not from the selected 3k/9k feature union; otherwise
            # the two feature sensitivities would receive different normalisation.
            full_row_counts = read_csr_rows(input_h5ad, shard_positions)
            n_counts = np.asarray(full_row_counts.sum(axis=1)).reshape(-1)
            if (
                len(n_counts) != len(shard_positions)
                or not np.isfinite(n_counts).all()
                or (n_counts <= 0).any()
            ):
                raise ValueError(
                    "Full-source library sizes must be finite and positive for "
                    "every materialized cell"
                )
            part = ad.AnnData(
                X=full_row_counts[:, gene_positions],
                obs=source.obs.iloc[shard_positions].copy(),
                var=source.var.iloc[gene_positions].copy(),
            )
            role_metadata = selected.iloc[start:stop].copy()
            role_metadata.index = part.obs_names
            for column in role_metadata.columns:
                # The explicit all-healthy design has no held-out cohort. Do not
                # serialize an all-null object column into HDF5 (and do not invent
                # a sentinel string that could later be mistaken for a cohort).
                if role_metadata[column].isna().all():
                    continue
                if column in part.obs and column not in {
                    "cell_key",
                    "source_cell_id",
                    "biological_unit_id",
                    "source_observation_id",
                    "observation_id",
                    "preparation_role",
                    "selected_reference_visit",
                    "eligible_for_feature_selection",
                    "global_one_visit_query",
                    "reference_design",
                    "original_row_position",
                    "outer_role",
                    "heldout_dataset",
                    "fold_id",
                    "inner_fold",
                    "fine_type",
                    "fine_type_confidence",
                    "fine_type_state_eligible",
                    "fine_type_balance_eligible",
                    "fine_type_mapping_status",
                }:
                    continue
                part.obs[column] = role_metadata[column].to_numpy()
            part.obs["n_counts"] = n_counts.astype(np.float64, copy=False)
            # The merged files use Ensembl IDs as var_names, while the inspected
            # vendor tokenizer explicitly requires this column as well.
            # Merged source objects are count-only.  Explicit reconstruction keeps
            # model inputs lean even if a future source gains unrelated embeddings.
            lean = ad.AnnData(X=part.X, obs=part.obs.copy(), var=part.var.copy())
            shard_path = temporary_dir / f"shard_{shard_number:06d}.h5ad"
            lean.write_h5ad(shard_path, compression="gzip")
            shard_paths.append(shard_path)
        if temporary_output.exists():
            temporary_output.unlink()
        ad.experimental.concat_on_disk(
            shard_paths,
            temporary_output,
            max_loaded_elems=max_loaded_elements,
            axis=0,
            join="inner",
            # Drop source-side var annotations here.  The one required identifier
            # column is written below using its standard H5AD string encoding.
            merge=None,
        )
        _write_ensembl_var_column(temporary_output, genes)
        check = ad.read_h5ad(temporary_output, backed="r")
        try:
            if check.shape != (len(selected), len(genes)):
                raise AssertionError("Materialized H5AD shape is incomplete")
            observed_keys = tuple(map(str, check.obs["cell_key"]))
            expected_keys = tuple(selected["cell_key"].astype(str))
            if observed_keys != expected_keys:
                raise AssertionError("Materialized H5AD cell order changed")
        finally:
            check.file.close()
        if output_h5ad.exists():
            output_h5ad.unlink()
        os.replace(temporary_output, output_h5ad)
    finally:
        source.file.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if temporary_output.exists():
            temporary_output.unlink()

    payload: dict[str, object] = {
        "schema_version": MATERIALIZED_SCHEMA,
        "reference_design": feature_manifest.get("reference_design", "lodo"),
        "role": role,
        "lineage": feature_manifest["lineage"],
        "heldout_dataset": feature_manifest["heldout_dataset"],
        "hvg_size": int(hvg_size),
        "source_h5ad": str(input_h5ad),
        "feature_manifest": str(preparation_dir / "feature_manifest.json"),
        "output_h5ad": str(output_h5ad),
        "shape": [len(selected), len(genes)],
        "cell_downsampling_performed": False,
        "lineage_donor_scope": lineage_donor_scope,
        "raw_fine_type_columns_preserved": ["ctype_low", "ctype_low_conf"],
        "fine_type_ontology": {
            key: value for key, value in ontology_record.items() if key != "source_path"
        },
        "gene_identifier_column": "ensembl_id",
        "library_size_column": "n_counts",
        "library_size_source": "full_source_h5ad_gene_universe_before_feature_subset",
        "cell_key_ordered_sha256": _ordered_digest(selected["cell_key"]),
        "model_gene_ordered_sha256": _ordered_digest(genes),
        "output_size_bytes": output_h5ad.stat().st_size,
    }
    atomic_write_json(output_manifest, payload)
    return payload


def materialize_frozen_query_h5ad(
    input_h5ad: Path,
    final_preparation_dir: Path,
    output_h5ad: Path,
    *,
    hvg_size: int,
    lineage: str,
    minimum_gene_coverage: float = 0.8,
    allow_training_dataset: bool = False,
    row_chunk_size: int = 25_000,
    max_loaded_elements: int = 100_000_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Map an unseen query to final-reference genes without relearning features.

    Missing frozen genes are represented by zero columns in the exact training
    order. Query counts never enter GP filtering or HVG ranking, and the complete
    query gene universe supplies the Geneformer library-size denominator.
    """

    if not 0 <= minimum_gene_coverage <= 1:
        raise ValueError("minimum_gene_coverage must be between zero and one")
    if row_chunk_size < 1 or max_loaded_elements < 1:
        raise ValueError("Chunk settings must be positive")
    input_h5ad = Path(input_h5ad).resolve()
    final_preparation_dir = Path(final_preparation_dir).resolve()
    output_h5ad = Path(output_h5ad).resolve()
    output_manifest = output_h5ad.with_suffix(".manifest.json")
    if (output_h5ad.exists() or output_manifest.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen query: {output_h5ad}")
    with (final_preparation_dir / "feature_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        feature_manifest = json.load(handle)
    if feature_manifest.get("schema_version") != FEATURE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported final feature manifest")
    if feature_manifest.get("reference_design") != "all_healthy":
        raise ValueError("Frozen queries require an all_healthy feature manifest")
    if feature_manifest.get("heldout_dataset") is not None:
        raise ValueError("Final feature manifest unexpectedly declares a heldout")
    if str(feature_manifest.get("lineage")) != str(lineage):
        raise ValueError("Query lineage differs from final feature manifest")
    ontology_record = feature_manifest.get("fine_type_ontology")
    if not isinstance(ontology_record, Mapping):
        raise ValueError("Final feature manifest lacks its approved fine-type ontology")
    ontology_snapshot = final_preparation_dir / str(ontology_record.get("path", ""))
    if not ontology_snapshot.is_file() or sha256_file(ontology_snapshot) != str(
        ontology_record.get("sha256", "")
    ):
        raise ValueError("Frozen fine-type ontology is missing or changed")
    fine_type_ontology = load_fine_type_ontology(
        ontology_snapshot, require_approved=True
    )
    vocabulary = feature_manifest["vocabularies"].get(str(int(hvg_size)))
    if vocabulary is None:
        raise ValueError(f"HVG size {hvg_size} was not frozen")
    gene_path = final_preparation_dir / vocabulary["model_genes_path"]
    if sha256_file(gene_path) != vocabulary["model_genes_sha256"]:
        raise ValueError("Frozen model-gene list hash does not match its manifest")
    genes = _read_gene_list(gene_path)

    source = ad.read_h5ad(input_h5ad, backed="r")
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_h5ad.stem}.shards.", dir=output_h5ad.parent)
    )
    temporary_output = output_h5ad.parent / f".{output_h5ad.stem}.partial.h5ad"
    try:
        cells = _safe_cell_metadata(source.obs, source.obs_names)
        if "lineage" in cells:
            observed_lineages = set(cells["lineage"].dropna().astype(str))
            if observed_lineages != {str(lineage)}:
                raise ValueError(
                    f"Query lineage values differ from {lineage!r}: "
                    f"{sorted(observed_lineages)}"
                )
        else:
            cells["lineage"] = str(lineage)
        cells = _apply_approved_fine_types(cells, fine_type_ontology)
        query_datasets = set(cells["dataset"].astype(str))
        training_datasets = set(map(str, feature_manifest["training_datasets"]))
        overlap = sorted(query_datasets & training_datasets)
        if overlap and not allow_training_dataset:
            raise ValueError(
                "Frozen production query reuses final-reference cohort names; "
                f"overlap={overlap}. Use the explicit override only for a labelled "
                "within-cohort sensitivity."
            )
        cells["preparation_role"] = "query"
        cells["eligible_for_feature_selection"] = False
        cells["selected_reference_visit"] = False
        cells["global_one_visit_query"] = False
        cells["outer_role"] = "query"
        cells["reference_design"] = "frozen_query"

        source_genes = tuple(map(str, source.var_names))
        if len(source_genes) != len(set(source_genes)):
            raise ValueError("Query H5AD var_names must be unique")
        source_lookup = {gene: index for index, gene in enumerate(source_genes)}
        available_target_positions = np.asarray(
            [index for index, gene in enumerate(genes) if gene in source_lookup],
            dtype=np.int64,
        )
        available_source_positions = np.asarray(
            [source_lookup[genes[index]] for index in available_target_positions],
            dtype=np.int64,
        )
        coverage = len(available_target_positions) / len(genes)
        if coverage < minimum_gene_coverage:
            raise ValueError(
                f"Frozen query gene coverage {coverage:.3f} is below "
                f"{minimum_gene_coverage:.3f}"
            )
        remap = sparse.csr_matrix(
            (
                np.ones(len(available_target_positions), dtype=np.float32),
                (
                    np.arange(len(available_target_positions)),
                    available_target_positions,
                ),
            ),
            shape=(len(available_target_positions), len(genes)),
        )

        shard_paths: list[Path] = []
        for shard_number, start in enumerate(range(0, len(cells), row_chunk_size)):
            stop = min(start + row_chunk_size, len(cells))
            positions = np.arange(start, stop, dtype=np.int64)
            full_counts = _as_raw_csr(read_csr_rows(input_h5ad, positions))
            n_counts = np.asarray(full_counts.sum(axis=1)).reshape(-1)
            if not np.isfinite(n_counts).all() or (n_counts <= 0).any():
                raise ValueError("Every frozen-query cell needs positive full counts")
            selected_counts = full_counts[:, available_source_positions] @ remap
            part_obs = source.obs.iloc[start:stop].copy()
            role_metadata = cells.iloc[start:stop].copy()
            role_metadata.index = part_obs.index
            for column in role_metadata.columns:
                if role_metadata[column].isna().all():
                    continue
                part_obs[column] = role_metadata[column].to_numpy()
            part_obs["n_counts"] = n_counts.astype(np.float64, copy=False)
            part = ad.AnnData(
                X=selected_counts.tocsr(),
                obs=part_obs,
                var=pd.DataFrame(index=pd.Index(genes, name=source.var_names.name)),
            )
            shard_path = temporary_dir / f"shard_{shard_number:06d}.h5ad"
            part.write_h5ad(shard_path, compression="gzip")
            shard_paths.append(shard_path)

        ad.experimental.concat_on_disk(
            shard_paths,
            temporary_output,
            max_loaded_elems=max_loaded_elements,
            axis=0,
            join="inner",
            merge=None,
        )
        _write_ensembl_var_column(temporary_output, genes)
        check = ad.read_h5ad(temporary_output, backed="r")
        try:
            if check.shape != (len(cells), len(genes)):
                raise AssertionError("Frozen query materialization is incomplete")
            if tuple(check.obs["cell_key"].astype(str)) != tuple(
                cells["cell_key"].astype(str)
            ):
                raise AssertionError("Frozen query cell order changed")
        finally:
            check.file.close()
        if output_h5ad.exists():
            output_h5ad.unlink()
        os.replace(temporary_output, output_h5ad)
    finally:
        source.file.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if temporary_output.exists():
            temporary_output.unlink()

    missing_genes = [gene for gene in genes if gene not in source_lookup]
    payload: dict[str, object] = {
        "schema_version": MATERIALIZED_SCHEMA,
        "reference_design": "frozen_query",
        "role": "query",
        "lineage": str(lineage),
        "heldout_dataset": None,
        "hvg_size": int(hvg_size),
        "source_h5ad": str(input_h5ad),
        "feature_manifest": str(final_preparation_dir / "feature_manifest.json"),
        "output_h5ad": str(output_h5ad),
        "shape": [len(cells), len(genes)],
        "query_datasets": sorted(query_datasets),
        "training_datasets": sorted(training_datasets),
        "feature_selection_on_query_performed": False,
        "cell_downsampling_performed": False,
        "raw_fine_type_columns_preserved": ["ctype_low", "ctype_low_conf"],
        "fine_type_ontology": {
            key: value for key, value in ontology_record.items() if key != "source_path"
        },
        "gene_identifier_column": "ensembl_id",
        "library_size_column": "n_counts",
        "library_size_source": "full_query_h5ad_gene_universe_before_feature_mapping",
        "frozen_gene_coverage": coverage,
        "n_frozen_genes_missing_from_query": len(missing_genes),
        "frozen_genes_missing_from_query": missing_genes,
        "cell_key_ordered_sha256": _ordered_digest(cells["cell_key"]),
        "model_gene_ordered_sha256": _ordered_digest(genes),
        "frozen_feature_manifest_sha256": sha256_file(
            final_preparation_dir / "feature_manifest.json"
        ),
        "frozen_vocabulary_sha256": sha256_file(gene_path),
        "output_size_bytes": output_h5ad.stat().st_size,
    }
    atomic_write_json(output_manifest, payload)
    return payload
