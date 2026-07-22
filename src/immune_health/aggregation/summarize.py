"""Aggregate per-cell GP embeddings without collapsing fine cell types."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from immune_health.baselines.pseudobulk import ensure_donor_observation_ids

from .statistics import DistributionEstimate, summarize_distribution

DistributionKey = tuple[str, str, str, str]
SPECIAL_FINE_TYPES = frozenset({"low_confidence", "other_confident"})


@dataclass(frozen=True)
class AggregationResult:
    """Serializable fine-type table plus empirical arrays for later distances."""

    table: pd.DataFrame
    distributions: Mapping[DistributionKey, np.ndarray]
    distribution_rows: Mapping[DistributionKey, np.ndarray]
    estimates: Mapping[DistributionKey, DistributionEstimate]
    empirical_distance_keys: frozenset[DistributionKey]


def _json_array(values: np.ndarray | None) -> str | pd.NA:
    if values is None:
        return pd.NA
    return json.dumps(np.asarray(values, dtype=float).tolist(), separators=(",", ":"))


def _strict_boolean(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        invalid = sorted(normalized.loc[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"{column} contains invalid booleans: {invalid[:5]}")
    return mapped.astype(bool)


def _age_axis_fields(
    values: np.ndarray,
    direction: np.ndarray | None,
    healthy_interval: tuple[float, float] | None,
) -> dict[str, float]:
    names = (
        "age_axis_q10",
        "age_axis_q25",
        "age_axis_q50",
        "age_axis_q75",
        "age_axis_q90",
    )
    if direction is None:
        return {
            **{name: float("nan") for name in names},
            "healthy_tail_fraction": float("nan"),
        }
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (values.shape[1],) or np.linalg.norm(vector) == 0:
        raise ValueError("age_direction must be nonzero and match embedding width")
    projected = values @ (vector / np.linalg.norm(vector))
    quantiles = np.quantile(projected, [0.10, 0.25, 0.50, 0.75, 0.90])
    fields = dict(zip(names, (float(value) for value in quantiles)))
    if healthy_interval is None:
        fields["healthy_tail_fraction"] = float("nan")
    else:
        low, high = healthy_interval
        fields["healthy_tail_fraction"] = float(
            np.mean((projected < low) | (projected > high))
        )
    return fields


def aggregate_fine_type_distributions(
    embeddings: np.ndarray,
    obs: pd.DataFrame,
    *,
    gp_id: str,
    fine_type_col: str = "fine_type",
    fine_type_state_eligible_col: str = "fine_type_state_eligible",
    fine_type_universe: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    min_state_cells: int = 5,
    min_empirical_cells: int = 25,
    robust_location: bool = True,
    age_direction: np.ndarray | None = None,
    healthy_interval: tuple[float, float] | None = None,
    annotation_confidence_col: str | None = None,
    provenance: Mapping[str, object] | None = None,
    retain_empirical_arrays: bool = False,
) -> AggregationResult:
    """Build donor-observation by fine-type GP distribution summaries.

    Groups below ``min_state_cells`` retain ``n_cells`` and composition but all
    state fields are missing. Ontology-ineligible special categories are retained
    in observed composition but can never produce a state or GP-age endpoint.
    Zero-cell state-eligible types from the lineage-specific ``fine_type_universe``
    receive explicit composition rows with missing state. Measurable groups below
    ``min_empirical_cells`` receive shrinkage moments but are flagged as too
    limited for empirical Wasserstein distance. Arrays remain separated by stable
    keys so downstream distance and bootstrap calculations do not serialize
    arbitrary Python objects into table cells.
    """

    values = np.asarray(embeddings)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(obs):
        raise ValueError("embeddings and observation metadata must align by row")
    for start in range(0, len(values), 65_536):
        if not np.isfinite(values[start : start + 65_536]).all():
            raise ValueError("embeddings contain non-finite values")
    if min_empirical_cells < min_state_cells:
        raise ValueError("min_empirical_cells cannot be below min_state_cells")
    frame = ensure_donor_observation_ids(obs)
    required = ["lineage", fine_type_col, "dataset", "donor_id", "sample_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing aggregation metadata columns: {missing}")
    if fine_type_state_eligible_col in frame:
        frame["_fine_type_state_eligible"] = _strict_boolean(
            frame[fine_type_state_eligible_col], fine_type_state_eligible_col
        )
    else:
        # Compatibility for old synthetic callers. Production CLI requires the
        # ontology-derived column explicitly.
        frame["_fine_type_state_eligible"] = ~frame[fine_type_col].astype(str).isin(
            SPECIAL_FINE_TYPES
        )
    eligibility_variation = frame.groupby(
        ["lineage", fine_type_col], observed=True, sort=False
    )["_fine_type_state_eligible"].nunique()
    if eligibility_variation.gt(1).any():
        raise ValueError("fine_type_state_eligible varies within one lineage/fine_type")
    grouping = [
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
        "lineage",
        fine_type_col,
    ]
    stable_columns = [column for column in ("age", "sex") if column in frame]
    observation_group = ["observation_id", "lineage"]
    for column in stable_columns:
        variation = frame.groupby(observation_group, observed=True, sort=False)[
            column
        ].nunique(dropna=False)
        if (variation > 1).any():
            raise ValueError(f"{column!r} varies within one biological observation")
    records: list[dict[str, object]] = []
    distributions: dict[DistributionKey, np.ndarray] = {}
    distribution_rows: dict[DistributionKey, np.ndarray] = {}
    estimates: dict[DistributionKey, DistributionEstimate] = {}
    empirical_distance_keys: set[DistributionKey] = set()
    provenance_values = dict(provenance or {})

    observed_by_lineage = {
        str(lineage): tuple(sorted(subset[fine_type_col].astype(str).drop_duplicates()))
        for lineage, subset in frame.groupby("lineage", observed=True, sort=False)
    }
    if fine_type_universe is None:
        universe_by_lineage = {
            str(lineage): tuple(
                sorted(
                    subset.loc[subset["_fine_type_state_eligible"], fine_type_col]
                    .astype(str)
                    .drop_duplicates()
                )
            )
            for lineage, subset in frame.groupby("lineage", observed=True, sort=False)
        }
    elif isinstance(fine_type_universe, Mapping):
        universe_by_lineage = {
            str(lineage): tuple(dict.fromkeys(str(value) for value in labels))
            for lineage, labels in fine_type_universe.items()
        }
    else:
        shared = tuple(dict.fromkeys(str(value) for value in fine_type_universe))
        universe_by_lineage = {lineage: shared for lineage in observed_by_lineage}
    for lineage, observed_types in observed_by_lineage.items():
        if lineage not in universe_by_lineage:
            raise ValueError(f"fine_type_universe is missing lineage {lineage!r}")
        omitted = set(observed_types).difference(universe_by_lineage[lineage])
        omitted_eligible = {
            fine_type
            for fine_type in omitted
            if frame.loc[
                frame["lineage"].astype(str).eq(lineage)
                & frame[fine_type_col].astype(str).eq(fine_type),
                "_fine_type_state_eligible",
            ].any()
        }
        if omitted_eligible:
            raise ValueError(
                f"fine_type_universe for {lineage!r} omits state-eligible "
                f"types {sorted(omitted_eligible)}"
            )

    group_indices = frame.groupby(grouping, observed=True, sort=False).indices
    lineage_totals = frame.groupby(["observation_id", "lineage"], observed=True).size()
    for group_values, indices in group_indices.items():
        index = np.asarray(indices, dtype=int)
        matrix = values[index]
        metadata = dict(zip(grouping, group_values))
        fine_type = str(metadata[fine_type_col])
        key = (
            str(metadata["observation_id"]),
            str(metadata["lineage"]),
            fine_type,
            str(gp_id),
        )
        eligible_values = frame.iloc[index]["_fine_type_state_eligible"].unique()
        if len(eligible_values) != 1:
            raise ValueError("Fine-type state eligibility varies within a group")
        fine_type_state_eligible = bool(eligible_values[0])
        if fine_type_state_eligible:
            estimate = summarize_distribution(
                matrix,
                min_cells=min_state_cells,
                robust_location=robust_location,
            )
        else:
            missing_state = float("nan")
            estimate = DistributionEstimate(
                n_cells=len(index),
                location=None,
                covariance=None,
                covariance_trace=missing_state,
                covariance_logdet=missing_state,
                median_distance=missing_state,
                distance_q75=missing_state,
                distance_q90=missing_state,
                state_available=False,
            )
        estimates[key] = estimate
        if estimate.state_available:
            distribution_rows[key] = index.copy()
            if retain_empirical_arrays:
                distributions[key] = matrix.copy()
            if len(index) >= min_empirical_cells:
                empirical_distance_keys.add(key)
        total = int(
            lineage_totals.loc[(metadata["observation_id"], metadata["lineage"])]
        )
        record: dict[str, object] = {
            **metadata,
            "fine_type": fine_type,
            "gp_id": str(gp_id),
            "n_cells": len(index),
            "fine_type_fraction": len(index) / total,
            "fine_type_state_eligible": fine_type_state_eligible,
            "location_summary": _json_array(estimate.location),
            "covariance_summary": _json_array(estimate.covariance),
            "covariance_trace": estimate.covariance_trace,
            "covariance_logdet": estimate.covariance_logdet,
            "dispersion": estimate.median_distance,
            "state_available": estimate.state_available,
            "empirical_distribution_available": key in empirical_distance_keys,
            "state_quality": (
                "ineligible_fine_type"
                if not fine_type_state_eligible
                else "sufficient"
                if key in empirical_distance_keys
                else "limited"
                if estimate.state_available
                else "insufficient"
            ),
            **_age_axis_fields(
                matrix if fine_type_state_eligible else np.empty((0, values.shape[1])),
                age_direction if fine_type_state_eligible else None,
                healthy_interval if fine_type_state_eligible else None,
            ),
            **provenance_values,
        }
        for column in stable_columns:
            unique = frame.iloc[index][column].drop_duplicates()
            if len(unique) != 1:
                raise ValueError(f"{column!r} varies within aggregation group")
            record[column] = unique.iloc[0]
        if annotation_confidence_col is None:
            record["annotation_confidence_summary"] = pd.NA
        else:
            confidence = pd.to_numeric(
                frame.iloc[index][annotation_confidence_col], errors="coerce"
            ).dropna()
            record["annotation_confidence_summary"] = (
                json.dumps(
                    {
                        "mean": float(confidence.mean()),
                        "median": float(confidence.median()),
                    },
                    separators=(",", ":"),
                )
                if len(confidence)
                else pd.NA
            )
        records.append(record)

    observation_columns = [
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
        "lineage",
        *stable_columns,
    ]
    observation_rows = frame[observation_columns].drop_duplicates()
    existing = {
        (
            str(record["observation_id"]),
            str(record["lineage"]),
            str(record["fine_type"]),
        )
        for record in records
    }
    for observation in observation_rows.to_dict(orient="records"):
        lineage = str(observation["lineage"])
        for fine_type in universe_by_lineage[lineage]:
            combination = (
                str(observation["observation_id"]),
                lineage,
                fine_type,
            )
            if combination in existing:
                continue
            key = (*combination, str(gp_id))
            missing = float("nan")
            estimate = DistributionEstimate(
                n_cells=0,
                location=None,
                covariance=None,
                covariance_trace=missing,
                covariance_logdet=missing,
                median_distance=missing,
                distance_q75=missing,
                distance_q90=missing,
                state_available=False,
            )
            estimates[key] = estimate
            records.append(
                {
                    **observation,
                    "fine_type": fine_type,
                    "gp_id": str(gp_id),
                    "n_cells": 0,
                    "fine_type_fraction": 0.0,
                    "fine_type_state_eligible": True,
                    "location_summary": pd.NA,
                    "covariance_summary": pd.NA,
                    "covariance_trace": missing,
                    "covariance_logdet": missing,
                    "dispersion": missing,
                    "state_available": False,
                    "empirical_distribution_available": False,
                    "state_quality": "insufficient",
                    **_age_axis_fields(np.empty((0, values.shape[1])), None, None),
                    "annotation_confidence_summary": pd.NA,
                    **provenance_values,
                }
            )
    return AggregationResult(
        pd.DataFrame.from_records(records),
        distributions,
        distribution_rows,
        estimates,
        frozenset(empirical_distance_keys),
    )
