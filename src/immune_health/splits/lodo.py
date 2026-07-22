"""Global donor splits for five-dataset leave-one-dataset-out analyses.

The split table is donor-level and can therefore be joined unchanged to every
lineage.  Samples, visits, fine types, and cells never receive independent
assignments.  ``source_observation_id`` intentionally remains a technical
sample/pool identifier; it is allowed to be shared by donors (as in OneK1K).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REFERENCE_DATASETS = (
    "aidav2",
    "immuneindonesia",
    "immunobiologyaging",
    "onek1k",
    "terekhova",
)


@dataclass(frozen=True)
class IdentifierColumns:
    """Column names used to construct globally stable identifiers."""

    dataset: str = "dataset"
    donor: str = "donor_id"
    sample: str = "sample_id"
    biological_unit: str = "biological_unit_id"
    source_observation: str = "source_observation_id"
    observation: str = "observation_id"
    separator: str = "::"


def _required_strings(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required identifier columns: {missing}")
    normalized = frame.loc[:, columns].astype("string")
    invalid = normalized.isna() | normalized.apply(lambda x: x.str.strip().eq(""))
    if invalid.any(axis=None):
        bad = {
            column: int(invalid[column].sum())
            for column in columns
            if invalid[column].any()
        }
        raise ValueError(f"Identifier columns contain missing/empty values: {bad}")
    return normalized


def _set_or_validate_column(
    frame: pd.DataFrame, column: str, expected: pd.Series
) -> None:
    if column in frame:
        actual = frame[column].astype("string")
        mismatch = ~actual.eq(expected)
        if mismatch.any():
            examples = frame.loc[mismatch, [column]].head(3).to_dict("records")
            raise ValueError(
                f"Existing {column!r} conflicts with the required definition; "
                f"examples={examples}"
            )
    frame[column] = expected.astype("string")


def add_stable_identifiers(
    records: pd.DataFrame,
    columns: IdentifierColumns = IdentifierColumns(),
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add and validate donor, source-sample, and donor-observation IDs.

    The approved definitions are::

        biological_unit_id = dataset::donor_id
        source_observation_id = dataset::sample_id
        observation_id = dataset::donor_id::sample_id

    The source observation may be a pool shared across donors.  The final
    observation identifier is donor-specific and must map to exactly one donor.
    """

    frame = records.copy() if copy else records
    values = _required_strings(frame, [columns.dataset, columns.donor, columns.sample])
    separator = columns.separator
    biological = values[columns.dataset] + separator + values[columns.donor]
    source_observation = values[columns.dataset] + separator + values[columns.sample]
    observation = biological + separator + values[columns.sample]
    _set_or_validate_column(frame, columns.biological_unit, biological)
    _set_or_validate_column(frame, columns.source_observation, source_observation)
    _set_or_validate_column(frame, columns.observation, observation)

    donor_counts = frame.groupby(columns.observation, observed=True)[
        columns.biological_unit
    ].nunique()
    if (donor_counts > 1).any():
        raise AssertionError("A donor-specific observation_id maps to multiple donors")
    return frame


def _first_consistent(series: pd.Series, label: str, donor: str) -> object:
    nonmissing = series.dropna()
    if nonmissing.dtype == object or isinstance(nonmissing.dtype, pd.StringDtype):
        nonmissing = nonmissing[nonmissing.astype("string").str.strip().ne("")]
    unique = pd.unique(nonmissing)
    if len(unique) > 1:
        raise ValueError(f"Donor {donor!r} has inconsistent {label}: {unique.tolist()}")
    return unique[0] if len(unique) else pd.NA


def _donor_level_table(
    records: pd.DataFrame, columns: IdentifierColumns
) -> pd.DataFrame:
    required = _required_strings(records, [columns.dataset, columns.donor])
    frame = records.copy()
    biological = required[columns.dataset] + columns.separator + required[columns.donor]
    _set_or_validate_column(frame, columns.biological_unit, biological)

    rows: list[dict[str, object]] = []
    for biological_unit, group in frame.groupby(
        columns.biological_unit, observed=True, sort=True
    ):
        dataset = _first_consistent(
            group[columns.dataset], "dataset", str(biological_unit)
        )
        donor = _first_consistent(
            group[columns.donor], "donor_id", str(biological_unit)
        )
        row: dict[str, object] = {
            columns.dataset: dataset,
            columns.donor: donor,
            columns.biological_unit: biological_unit,
        }

        if "n_sex_values" in group:
            n_sex_values = pd.to_numeric(group["n_sex_values"], errors="coerce").max()
            if pd.notna(n_sex_values) and n_sex_values > 1:
                raise ValueError(
                    f"Donor {biological_unit!r} has inconsistent audited sex values"
                )
        if "sex" in group:
            row["sex"] = _first_consistent(group["sex"], "sex", str(biological_unit))
        else:
            row["sex"] = "unknown"

        if "age" in group:
            age = pd.to_numeric(group["age"], errors="coerce")
            row["age_min"] = float(age.min()) if age.notna().any() else np.nan
            row["age_max"] = float(age.max()) if age.notna().any() else np.nan
            row["age_for_stratification"] = (
                float(age.median()) if age.notna().any() else np.nan
            )
        elif {"age_min", "age_max"}.issubset(group.columns):
            age_min = pd.to_numeric(group["age_min"], errors="coerce").min()
            age_max = pd.to_numeric(group["age_max"], errors="coerce").max()
            row["age_min"] = float(age_min) if pd.notna(age_min) else np.nan
            row["age_max"] = float(age_max) if pd.notna(age_max) else np.nan
            row["age_for_stratification"] = (
                float((age_min + age_max) / 2)
                if pd.notna(age_min) and pd.notna(age_max)
                else np.nan
            )
        else:
            row["age_min"] = np.nan
            row["age_max"] = np.nan
            row["age_for_stratification"] = np.nan

        if columns.sample in group:
            sample_values = sorted(
                group[columns.sample].dropna().astype("string").unique().tolist()
            )
            row["n_source_observations"] = len(sample_values)
            row["sample_ids"] = "|".join(sample_values)
            row["source_observation_ids"] = "|".join(
                f"{dataset}{columns.separator}{sample}" for sample in sample_values
            )
            row["observation_ids"] = "|".join(
                f"{biological_unit}{columns.separator}{sample}"
                for sample in sample_values
            )
        elif "sample_ids" in group:
            # The read-only audit table is already aggregated by donor/fine type
            # and stores the contributing samples as a pipe-delimited scalar.
            # Recover the exact donor-level union so the global manifest remains
            # auditable without revisiting millions of cell rows.
            sample_values = sorted(
                {
                    sample.strip()
                    for value in group["sample_ids"].dropna().astype(str)
                    for sample in value.split("|")
                    if sample.strip()
                }
            )
            row["n_source_observations"] = len(sample_values)
            row["sample_ids"] = "|".join(sample_values)
            row["source_observation_ids"] = "|".join(
                f"{dataset}{columns.separator}{sample}" for sample in sample_values
            )
            row["observation_ids"] = "|".join(
                f"{biological_unit}{columns.separator}{sample}"
                for sample in sample_values
            )
        elif "n_samples" in group:
            n_samples = pd.to_numeric(group["n_samples"], errors="coerce").max()
            row["n_source_observations"] = (
                int(n_samples) if pd.notna(n_samples) else pd.NA
            )
            row["sample_ids"] = pd.NA
            row["source_observation_ids"] = pd.NA
            row["observation_ids"] = pd.NA
        else:
            row["n_source_observations"] = pd.NA
            row["sample_ids"] = pd.NA
            row["source_observation_ids"] = pd.NA
            row["observation_ids"] = pd.NA
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(columns.biological_unit).reset_index(drop=True)
    )


def _age_strata(values: pd.Series, age_bin_edges: Sequence[float]) -> pd.Series:
    edges = np.asarray(age_bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("age_bin_edges must be a strictly increasing sequence")
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.cut(numeric, bins=edges, include_lowest=True).astype("string")
    return result.fillna("age_unknown")


def _assign_inner_folds(
    donors: pd.DataFrame,
    *,
    n_inner_folds: int,
    seed: int,
    dataset_column: str,
) -> pd.Series:
    if n_inner_folds < 2:
        raise ValueError("n_inner_folds must be at least 2")
    assignments = pd.Series(pd.NA, index=donors.index, dtype="Int64")
    rng = np.random.default_rng(seed)

    # Assignment is performed within dataset, preserving the outer LODO unit.
    for _, dataset_group in donors.groupby(dataset_column, sort=True, observed=True):
        if len(dataset_group) < n_inner_folds:
            raise ValueError(
                "Each dataset must contain at least n_inner_folds biological units"
            )
        fold_load = np.zeros(n_inner_folds, dtype=int)
        strata = dataset_group.groupby(
            ["age_bin", "sex_stratum"], sort=True, dropna=False, observed=True
        )
        ordered_strata = sorted(
            strata.indices.items(), key=lambda item: (-len(item[1]), str(item[0]))
        )
        for _, index_positions in ordered_strata:
            labels = dataset_group.iloc[index_positions].index.to_numpy(copy=True)
            labels = np.sort(labels)
            rng.shuffle(labels)
            stratum_load = np.zeros(n_inner_folds, dtype=int)
            for label in labels:
                best = np.flatnonzero(
                    (stratum_load == stratum_load.min())
                    & (fold_load == fold_load[stratum_load == stratum_load.min()].min())
                )
                chosen = int(rng.choice(best))
                assignments.loc[label] = chosen
                stratum_load[chosen] += 1
                fold_load[chosen] += 1
    return assignments


def build_global_donor_manifest(
    records: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = REFERENCE_DATASETS,
    n_inner_folds: int = 3,
    age_bin_edges: Sequence[float] = (0, 30, 45, 60, 75, np.inf),
    seed: int = 42,
    columns: IdentifierColumns = IdentifierColumns(),
) -> pd.DataFrame:
    """Collapse metadata to donors and make one lineage-independent assignment."""

    donors = _donor_level_table(records, columns)
    observed = set(donors[columns.dataset].astype(str))
    if datasets is None:
        selected = tuple(sorted(observed))
    else:
        selected = tuple(str(dataset) for dataset in datasets)
        missing = sorted(set(selected) - observed)
        extra = sorted(observed - set(selected))
        if missing or extra:
            raise ValueError(
                "Dataset labels do not match the requested reference set; "
                f"missing={missing}, unexpected={extra}"
            )
    if len(selected) < 2:
        raise ValueError("LODO requires at least two datasets")

    donors["age_bin"] = _age_strata(donors["age_for_stratification"], age_bin_edges)
    donors["sex_stratum"] = (
        donors["sex"].astype("string").str.strip().str.lower().fillna("unknown")
    )
    donors["global_inner_fold"] = _assign_inner_folds(
        donors,
        n_inner_folds=n_inner_folds,
        seed=seed,
        dataset_column=columns.dataset,
    )
    donors["split_seed"] = int(seed)
    donors["n_inner_folds"] = int(n_inner_folds)
    return donors.sort_values([columns.dataset, columns.biological_unit]).reset_index(
        drop=True
    )


def build_lodo_tables(
    global_donor_manifest: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    columns: IdentifierColumns = IdentifierColumns(),
) -> dict[str, pd.DataFrame]:
    """Create one donor-level outer-fold table for each held-out dataset."""

    required = {columns.dataset, columns.biological_unit, "global_inner_fold"}
    missing = sorted(required - set(global_donor_manifest.columns))
    if missing:
        raise ValueError(f"Global donor manifest is missing columns: {missing}")
    observed = sorted(global_donor_manifest[columns.dataset].astype(str).unique())
    heldouts = tuple(observed if datasets is None else map(str, datasets))
    if set(heldouts) != set(observed):
        raise ValueError("LODO held-out datasets must equal the manifest datasets")

    result: dict[str, pd.DataFrame] = {}
    for heldout in heldouts:
        table = global_donor_manifest.copy()
        is_query = table[columns.dataset].astype(str).eq(heldout)
        table.insert(0, "fold_id", f"lodo_{heldout}")
        table.insert(1, "heldout_dataset", heldout)
        table.insert(2, "outer_role", np.where(is_query, "query", "reference"))
        table["inner_fold"] = table["global_inner_fold"].where(~is_query, pd.NA)
        table["eligible_for_reference_fitting"] = ~is_query
        assert_lodo_integrity(table, heldout, columns=columns)
        result[heldout] = table
    return result


def assert_lodo_integrity(
    table: pd.DataFrame,
    heldout_dataset: str,
    *,
    columns: IdentifierColumns = IdentifierColumns(),
) -> None:
    """Raise if a LODO table permits donor or held-out-dataset leakage."""

    required = {
        columns.dataset,
        columns.biological_unit,
        "heldout_dataset",
        "outer_role",
        "inner_fold",
        "eligible_for_reference_fitting",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AssertionError(f"LODO table lacks integrity columns: {missing}")
    if not table["heldout_dataset"].astype(str).eq(str(heldout_dataset)).all():
        raise AssertionError("heldout_dataset is inconsistent within a LODO table")
    query_expected = table[columns.dataset].astype(str).eq(str(heldout_dataset))
    if not table.loc[query_expected, "outer_role"].eq("query").all():
        raise AssertionError("Held-out dataset contains non-query rows")
    if not table.loc[~query_expected, "outer_role"].eq("reference").all():
        raise AssertionError("A training dataset contains query rows")
    if table.loc[query_expected, "eligible_for_reference_fitting"].astype(bool).any():
        raise AssertionError("Query donor is eligible for reference fitting")
    reference_eligibility = table.loc[
        ~query_expected, "eligible_for_reference_fitting"
    ].astype(bool)
    if not reference_eligibility.all():
        raise AssertionError("Reference donors were unexpectedly excluded")
    if table.loc[query_expected, "inner_fold"].notna().any():
        raise AssertionError("Held-out donors received a training-time inner fold")

    per_donor = table.groupby(columns.biological_unit, observed=True).agg(
        n_datasets=(columns.dataset, "nunique"),
        n_roles=("outer_role", "nunique"),
        n_inner=("inner_fold", lambda x: x.dropna().nunique()),
    )
    if (per_donor["n_datasets"] != 1).any() or (per_donor["n_roles"] != 1).any():
        raise AssertionError("A biological donor crosses datasets or outer partitions")
    if (per_donor["n_inner"] > 1).any():
        raise AssertionError("A biological donor crosses internal folds")
    if "lineage" in table:
        across_lineages = table.groupby(columns.biological_unit, observed=True).agg(
            role_count=("outer_role", "nunique"),
            fold_count=("inner_fold", lambda x: x.dropna().nunique()),
        )
        if (across_lineages > 1).any(axis=None):
            raise AssertionError("Global donor assignment differs across lineages")
    if columns.observation in table:
        observation_donors = table.groupby(columns.observation, observed=True)[
            columns.biological_unit
        ].nunique()
        if (observation_donors > 1).any():
            raise AssertionError("A donor-specific observation crosses donors")


def assert_partition_disjoint(
    partitions: Mapping[str, Iterable[str]],
) -> None:
    """Assert that biological-unit collections are pairwise disjoint."""

    normalized = {name: set(map(str, values)) for name, values in partitions.items()}
    names = sorted(normalized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise AssertionError(
                    f"Biological units overlap between {left!r} and {right!r}: "
                    f"{examples}"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_lodo_manifests(
    global_donor_manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    datasets: Sequence[str] | None = None,
    columns: IdentifierColumns = IdentifierColumns(),
    source_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write compact donor-level split tables plus a provenance manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_path = output / "global_donor_manifest.tsv"
    global_donor_manifest.to_csv(global_path, sep="\t", index=False)
    tables = build_lodo_tables(
        global_donor_manifest, datasets=datasets, columns=columns
    )
    for provenance_column in ("split_seed", "n_inner_folds"):
        if (
            provenance_column in global_donor_manifest
            and global_donor_manifest[provenance_column].nunique(dropna=False) != 1
        ):
            raise ValueError(
                f"Global manifest has inconsistent {provenance_column!r} values"
            )
    paths: dict[str, Path] = {"global": global_path}
    fold_summary: dict[str, dict[str, object]] = {}
    for heldout, table in tables.items():
        path = output / f"lodo_{heldout}.tsv"
        table.to_csv(path, sep="\t", index=False)
        paths[heldout] = path
        fold_summary[heldout] = {
            "path": path.name,
            "sha256": _sha256(path),
            "reference_donors": int(table["eligible_for_reference_fitting"].sum()),
            "query_donors": int((table["outer_role"] == "query").sum()),
        }
    manifest = {
        "schema_version": "1.0",
        "identifier_columns": asdict(columns),
        "identifier_definitions": {
            columns.biological_unit: "dataset::donor_id",
            columns.source_observation: "dataset::sample_id",
            columns.observation: "dataset::donor_id::sample_id",
        },
        "source_observation_may_span_donors": True,
        "datasets": sorted(
            global_donor_manifest[columns.dataset].astype(str).unique().tolist()
        ),
        "biological_units_by_dataset": {
            str(dataset): int(count)
            for dataset, count in global_donor_manifest.groupby(
                columns.dataset, observed=True
            )
            .size()
            .items()
        },
        "split_seed": int(global_donor_manifest["split_seed"].iloc[0])
        if "split_seed" in global_donor_manifest
        else None,
        "n_inner_folds": int(global_donor_manifest["n_inner_folds"].iloc[0])
        if "n_inner_folds" in global_donor_manifest
        else None,
        "global_manifest": {
            "path": global_path.name,
            "sha256": _sha256(global_path),
            "n_donors": int(len(global_donor_manifest)),
        },
        "folds": fold_summary,
    }
    if source_path is not None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Split source table does not exist: {source}")
        manifest["source"] = {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
        }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["manifest"] = manifest_path
    return paths
