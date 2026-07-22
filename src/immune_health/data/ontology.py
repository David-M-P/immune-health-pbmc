"""Reviewable fine-cell-type ontology candidates from audited exact labels.

Candidate generation is deliberately conservative: every observed label maps
only to itself.  Biologically distinct labels are never merged automatically.
The resulting YAML remains unusable by default until scientific approval is
recorded explicitly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

SPECIAL_FINE_TYPE_CATEGORIES = ("other_confident", "low_confidence")
APPROVED_ONTOLOGY_STATUS = "approved"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce").fillna(0)
    keep = numeric.notna() & weight.gt(0)
    if not keep.any():
        return np.nan
    return float(np.average(numeric[keep], weights=weight[keep]))


def summarize_fine_type_labels(
    records: pd.DataFrame,
    *,
    dataset_column: str = "dataset",
    lineage_column: str = "lineage",
    fine_type_column: str = "fine_type",
    donor_column: str = "biological_unit_id",
    confidence_column: str = "annotation_confidence",
    confidence_threshold: float = 0.9,
    poor_donor_coverage_below: int = 10,
) -> pd.DataFrame:
    """Summarize exact labels from cells or the audited fine-type table.

    Audit aliases ``ctype_low`` and ``confidence_mean`` are accepted.  Marker
    sanity is reported as not evaluated because a label summary has no gene
    expression matrix; callers may merge an independently reviewed marker report.
    """

    frame = records.copy()
    aliases = {
        fine_type_column: (fine_type_column, "ctype_low"),
        confidence_column: (confidence_column, "confidence_mean"),
    }
    for target, candidates in aliases.items():
        if target not in frame:
            source = next((name for name in candidates if name in frame), None)
            if source is not None:
                frame[target] = frame[source]
    required = {dataset_column, lineage_column, fine_type_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fine-type records are missing columns: {missing}")
    for column in required:
        invalid = frame[column].astype("string").isna() | frame[column].astype(
            "string"
        ).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Fine-type column {column!r} contains missing labels")

    grouping = [dataset_column, lineage_column, fine_type_column]
    if "n_cells" in frame:
        duplicates = frame.duplicated(grouping, keep=False)
        if duplicates.any():
            raise ValueError(
                "Pre-aggregated fine-type input must have one row per "
                "dataset/lineage/exact label"
            )
        summary = frame[grouping].copy()
        summary["n_cells"] = pd.to_numeric(frame["n_cells"], errors="raise").astype(int)
        if "n_donors" in frame:
            summary["n_donors"] = pd.to_numeric(
                frame["n_donors"], errors="raise"
            ).astype(int)
        else:
            summary["n_donors"] = pd.NA
        summary["annotation_confidence_mean"] = (
            pd.to_numeric(frame[confidence_column], errors="coerce")
            if confidence_column in frame
            else np.nan
        )
        low_count_column = next(
            (
                column
                for column in (
                    f"confidence_lt_{confidence_threshold:g}",
                    "confidence_lt_0_9",
                )
                if column in frame
            ),
            None,
        )
        summary["n_below_confidence_threshold"] = (
            pd.to_numeric(frame[low_count_column], errors="coerce")
            if low_count_column is not None
            else pd.NA
        )
    else:
        if donor_column not in frame:
            if {dataset_column, "donor_id"}.issubset(frame.columns):
                frame[donor_column] = (
                    frame[dataset_column].astype("string")
                    + "::"
                    + frame["donor_id"].astype("string")
                )
            else:
                raise ValueError(
                    "Cell-level ontology summaries require biological_unit_id "
                    "or donor_id"
                )
        if confidence_column in frame:
            frame["_confidence_numeric"] = pd.to_numeric(
                frame[confidence_column], errors="coerce"
            )
            frame["_below_threshold"] = frame["_confidence_numeric"].lt(
                confidence_threshold
            )
        else:
            frame["_confidence_numeric"] = np.nan
            frame["_below_threshold"] = False
        grouped = frame.groupby(grouping, observed=True, sort=True)
        summary = grouped.size().rename("n_cells").reset_index()
        donor_counts = grouped[donor_column].nunique().rename("n_donors").reset_index()
        summary = summary.merge(donor_counts, on=grouping, validate="one_to_one")
        if confidence_column in frame:
            confidence = (
                grouped["_confidence_numeric"]
                .mean()
                .rename("annotation_confidence_mean")
                .reset_index()
            )
            below = (
                grouped["_below_threshold"]
                .sum()
                .rename("n_below_confidence_threshold")
                .reset_index()
            )
            summary = summary.merge(confidence, on=grouping, validate="one_to_one")
            summary = summary.merge(below, on=grouping, validate="one_to_one")
        else:
            summary["annotation_confidence_mean"] = np.nan
            summary["n_below_confidence_threshold"] = pd.NA

    support = (
        summary.groupby([lineage_column, fine_type_column], observed=True)[
            dataset_column
        ]
        .nunique()
        .rename("dataset_support_within_lineage")
        .reset_index()
    )
    summary = summary.merge(
        support, on=[lineage_column, fine_type_column], validate="many_to_one"
    )
    summary["found_in_one_dataset"] = summary["dataset_support_within_lineage"].eq(1)
    summary["poor_donor_coverage"] = pd.to_numeric(
        summary["n_donors"], errors="coerce"
    ).lt(poor_donor_coverage_below)
    summary["poor_mean_confidence"] = pd.to_numeric(
        summary["annotation_confidence_mean"], errors="coerce"
    ).lt(confidence_threshold)
    summary["marker_expression_sanity"] = "not_evaluated"
    summary["marker_expression_note"] = (
        "Requires a lineage-specific expression matrix and approved marker panel"
    )
    return summary.sort_values(grouping).reset_index(drop=True)


def generate_candidate_ontology(
    records: pd.DataFrame,
    *,
    dataset_column: str = "dataset",
    lineage_column: str = "lineage",
    fine_type_column: str = "fine_type",
    confidence_column: str = "annotation_confidence",
    minimum_confidence: float = 0.9,
    minimum_cells_for_state: int | Mapping[str, int] = 30,
    poor_donor_coverage_below: int = 10,
) -> dict[str, Any]:
    """Generate an identity-mapping ontology that requires manual approval."""

    summary = summarize_fine_type_labels(
        records,
        dataset_column=dataset_column,
        lineage_column=lineage_column,
        fine_type_column=fine_type_column,
        confidence_column=confidence_column,
        confidence_threshold=minimum_confidence,
        poor_donor_coverage_below=poor_donor_coverage_below,
    )
    lineages: dict[str, Any] = {}
    for lineage, lineage_frame in summary.groupby(
        lineage_column, observed=True, sort=True
    ):
        threshold = (
            int(minimum_cells_for_state.get(str(lineage), 30))
            if isinstance(minimum_cells_for_state, Mapping)
            else int(minimum_cells_for_state)
        )
        mappings: list[dict[str, Any]] = []
        for label, label_frame in lineage_frame.groupby(
            fine_type_column, observed=True, sort=True
        ):
            n_cells = int(label_frame["n_cells"].sum())
            donor_values = pd.to_numeric(label_frame["n_donors"], errors="coerce")
            n_donors = int(donor_values.sum()) if donor_values.notna().all() else None
            mappings.append(
                {
                    "original_label": str(label),
                    "canonical_fine_type": str(label),
                    "generated_identity_mapping": True,
                    "datasets": sorted(
                        label_frame[dataset_column].astype(str).tolist()
                    ),
                    "n_cells": n_cells,
                    "n_donors_dataset_qualified": n_donors,
                    "annotation_confidence_mean": _weighted_mean(
                        label_frame["annotation_confidence_mean"],
                        label_frame["n_cells"],
                    ),
                    "found_in_one_dataset": bool(
                        label_frame["found_in_one_dataset"].all()
                    ),
                    "poor_donor_coverage_any_dataset": bool(
                        label_frame["poor_donor_coverage"].any()
                    ),
                    "marker_expression_sanity": "not_evaluated",
                    "eligible_for_state_by_total_cells": n_cells >= threshold,
                    "retain_in_composition": True,
                }
            )
        lineages[str(lineage)] = {
            "minimum_cells_for_state": threshold,
            "retain_rare_types_in_composition": True,
            "mappings": mappings,
        }

    ontology: dict[str, Any] = {
        "schema_version": "1.0",
        "generated": True,
        "requires_approval": True,
        "approval": {
            "status": "pending_scientific_review",
            "approved_by": None,
            "approved_at": None,
        },
        "generation_policy": (
            "Exact audited labels receive identity mappings; no biological merges "
            "are inferred."
        ),
        "minimum_annotation_confidence": float(minimum_confidence),
        "special_categories": {
            "other_confident": {
                "description": "Confident label not mapped by the approved ontology",
                "retain_in_composition": True,
            },
            "low_confidence": {
                "description": "Missing or below-threshold annotation confidence",
                "retain_in_composition": True,
            },
        },
        "lineages": lineages,
    }
    validate_ontology(ontology)
    return ontology


def validate_ontology(ontology: Mapping[str, Any]) -> None:
    """Validate special categories and one-to-one original-label mappings."""

    missing_categories = set(SPECIAL_FINE_TYPE_CATEGORIES) - set(
        ontology.get("special_categories", {})
    )
    if missing_categories:
        raise ValueError(
            f"Ontology lacks special categories: {sorted(missing_categories)}"
        )
    for category in SPECIAL_FINE_TYPE_CATEGORIES:
        settings = ontology["special_categories"][category]
        if settings.get("retain_in_composition") is not True:
            raise ValueError(
                f"Special fine type {category!r} must remain in composition"
            )
        if settings.get("state_eligible", False) is not False:
            raise ValueError(f"Special fine type {category!r} cannot be state eligible")
        if settings.get("balance_eligible", False) is not False:
            raise ValueError(
                f"Special fine type {category!r} cannot receive balancing uplift"
            )
    if "lineages" not in ontology or not ontology["lineages"]:
        raise ValueError("Ontology contains no lineage mappings")
    for lineage, config in ontology["lineages"].items():
        if int(config.get("minimum_cells_for_state", 0)) < 1:
            raise ValueError(f"Lineage {lineage!r} has an invalid state threshold")
        seen: dict[str, str] = {}
        for mapping in config.get("mappings", []):
            original = str(mapping["original_label"])
            canonical = str(mapping["canonical_fine_type"])
            if original in seen and seen[original] != canonical:
                raise ValueError(
                    f"Lineage {lineage!r} maps label {original!r} to multiple types"
                )
            seen[original] = canonical
            state_eligible = bool(
                mapping.get(
                    "state_eligible",
                    mapping.get("eligible_for_state_by_total_cells", True),
                )
            )
            balance_eligible = bool(mapping.get("balance_eligible", True))
            if state_eligible and not balance_eligible:
                raise ValueError(
                    f"State-eligible mapping {lineage!r}/{original!r} must also be "
                    "eligible for fine-type balancing"
                )
            if canonical in SPECIAL_FINE_TYPE_CATEGORIES and (
                state_eligible or balance_eligible
            ):
                raise ValueError(
                    f"Mapping {lineage!r}/{original!r} targets special category "
                    f"{canonical!r} but remains eligible"
                )


def ontology_is_approved(ontology: Mapping[str, Any]) -> bool:
    """Return whether scientific approval is complete and internally consistent."""

    approval = ontology.get("approval")
    return bool(
        isinstance(approval, Mapping)
        and approval.get("status") == APPROVED_ONTOLOGY_STATUS
        and str(approval.get("approved_by", "")).strip()
        and str(approval.get("approved_at", "")).strip()
        and ontology.get("requires_approval") is False
        and set(map(str, ontology.get("approved_lineages", ())))
        == set(map(str, ontology.get("lineages", {})))
    )


def load_fine_type_ontology(
    path: str | Path, *, require_approved: bool = True
) -> dict[str, Any]:
    """Load and validate a YAML ontology, failing closed in production."""

    ontology_path = Path(path).resolve()
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Fine-type ontology is missing: {ontology_path}")
    with ontology_path.open(encoding="utf-8") as handle:
        ontology = yaml.safe_load(handle)
    if not isinstance(ontology, dict):
        raise ValueError("Fine-type ontology must be a YAML mapping")
    validate_ontology(ontology)
    source_candidate = ontology.get("source_candidate")
    if source_candidate is not None:
        if not isinstance(source_candidate, Mapping):
            raise ValueError("Ontology source_candidate must be a mapping")
        source_path = ontology_path.parent / str(source_candidate.get("path", ""))
        expected_hash = str(source_candidate.get("sha256", ""))
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Fine-type ontology source candidate is missing: {source_path}"
            )
        observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if not expected_hash or observed_hash != expected_hash:
            raise ValueError("Fine-type ontology source candidate hash differs")
    if require_approved and not ontology_is_approved(ontology):
        raise ValueError(
            "Fine-type ontology requires scientific approval with status, "
            "approver, date, and requires_approval=false"
        )
    return ontology


def approved_ontology_identity(ontology: Mapping[str, Any]) -> dict[str, Any]:
    """Return portable scientific identity fields for downstream manifests."""

    validate_ontology(ontology)
    if not ontology_is_approved(ontology):
        raise ValueError("Cannot bind an unapproved fine-type ontology")
    approval = ontology["approval"]
    return {
        "ontology_id": str(ontology.get("ontology_id", "")),
        "schema_version": str(ontology.get("schema_version", "")),
        "minimum_annotation_confidence": float(
            ontology["minimum_annotation_confidence"]
        ),
        "approval_status": str(approval["status"]),
        "approved_by": str(approval["approved_by"]),
        "approved_at": str(approval["approved_at"]),
        "special_categories": list(SPECIAL_FINE_TYPE_CATEGORIES),
        "approved_lineages": list(map(str, ontology["approved_lineages"])),
        "source_candidate_sha256": str(
            ontology.get("source_candidate", {}).get("sha256", "")
        ),
    }


def write_candidate_ontology(ontology: Mapping[str, Any], path: str | Path) -> Path:
    """Write a generated candidate YAML after strict validation."""

    validate_ontology(ontology)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(dict(ontology), sort_keys=False))
    return output


def apply_fine_type_ontology(
    records: pd.DataFrame,
    ontology: Mapping[str, Any],
    *,
    lineage_column: str = "lineage",
    fine_type_column: str = "fine_type",
    confidence_column: str = "annotation_confidence",
    output_column: str = "canonical_fine_type",
    state_eligible_column: str = "fine_type_state_eligible",
    balance_eligible_column: str = "fine_type_balance_eligible",
    mapping_status_column: str = "fine_type_mapping_status",
    allow_unapproved: bool = False,
) -> pd.DataFrame:
    """Apply an approved ontology, retaining explicit uncertainty categories."""

    validate_ontology(ontology)
    if not ontology_is_approved(ontology) and not allow_unapproved:
        raise ValueError(
            "Fine-type ontology is generated and requires scientific approval"
        )
    required = {lineage_column, fine_type_column, confidence_column}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Cannot apply ontology; missing columns: {missing}")
    if ontology_is_approved(ontology):
        approved_lineages = set(map(str, ontology["approved_lineages"]))
        observed_lineages = set(records[lineage_column].dropna().astype(str))
        outside_scope = sorted(observed_lineages - approved_lineages)
        if outside_scope:
            raise ValueError(
                "Fine-type ontology is not scientifically approved for lineages: "
                f"{outside_scope}"
            )
    mapping: dict[tuple[str, str], tuple[str, bool, bool, str]] = {}
    for lineage, config in ontology["lineages"].items():
        for item in config.get("mappings", []):
            canonical = str(item["canonical_fine_type"])
            state_eligible = bool(
                item.get(
                    "state_eligible",
                    item.get("eligible_for_state_by_total_cells", True),
                )
            )
            balance_eligible = bool(item.get("balance_eligible", True))
            mapping[(str(lineage), str(item["original_label"]))] = (
                canonical,
                state_eligible,
                balance_eligible,
                str(item.get("disposition", "approved_identity")),
            )
    threshold = float(ontology["minimum_annotation_confidence"])
    frame = records.copy()
    confidence = pd.to_numeric(frame[confidence_column], errors="coerce")
    result: list[str] = []
    state_flags: list[bool] = []
    balance_flags: list[bool] = []
    statuses: list[str] = []
    for lineage, label, value in zip(
        frame[lineage_column].astype(str),
        frame[fine_type_column].astype(str),
        confidence,
        strict=True,
    ):
        if pd.isna(value) or value < threshold:
            result.append("low_confidence")
            state_flags.append(False)
            balance_flags.append(False)
            statuses.append("below_confidence_threshold")
        else:
            resolved = mapping.get((lineage, label))
            if resolved is None:
                result.append("other_confident")
                state_flags.append(False)
                balance_flags.append(False)
                statuses.append("unmapped_confident")
            else:
                canonical, state_eligible, balance_eligible, status = resolved
                result.append(canonical)
                state_flags.append(state_eligible)
                balance_flags.append(balance_eligible)
                statuses.append(status)
    frame[output_column] = pd.Series(result, index=frame.index, dtype="string")
    frame[state_eligible_column] = pd.Series(state_flags, index=frame.index, dtype=bool)
    frame[balance_eligible_column] = pd.Series(
        balance_flags, index=frame.index, dtype=bool
    )
    frame[mapping_status_column] = pd.Series(
        statuses, index=frame.index, dtype="string"
    )
    return frame
