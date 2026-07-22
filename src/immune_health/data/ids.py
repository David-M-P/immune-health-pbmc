"""Stable donor and observation identifiers used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class IdentifierSpec:
    """Fields used for biological and observation identities.

    OneK1K's sample identifier is a sequencing-pool number shared by donors.
    Therefore the approved donor-observation identity includes the donor, while
    ``source_observation_id`` deliberately retains the pool/library identity.
    """

    dataset: str = "dataset"
    donor: str = "donor_id"
    sample: str = "sample_id"
    separator: str = "::"

    @property
    def biological_fields(self) -> tuple[str, str]:
        return self.dataset, self.donor

    @property
    def source_observation_fields(self) -> tuple[str, str]:
        return self.dataset, self.sample

    @property
    def observation_fields(self) -> tuple[str, str, str]:
        return self.dataset, self.donor, self.sample


DEFAULT_IDENTIFIER_SPEC = IdentifierSpec()


def _compose(
    frame: pd.DataFrame,
    fields: Iterable[str],
    separator: str,
    label: str,
) -> pd.Series:
    fields = tuple(fields)
    missing_columns = [field for field in fields if field not in frame]
    if missing_columns:
        raise ValueError(
            f"Cannot construct {label}; missing columns: {missing_columns}"
        )
    values = frame.loc[:, fields].astype("string")
    missing = values.isna().any(axis=1) | values.apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=1)
    if missing.any():
        examples = frame.index[missing].astype(str).tolist()[:5]
        raise ValueError(f"Cannot construct {label}; missing values at rows {examples}")
    contains_separator = values.apply(
        lambda column: column.str.contains(separator, regex=False)
    ).any(axis=1)
    if contains_separator.any():
        examples = frame.index[contains_separator].astype(str).tolist()[:5]
        raise ValueError(
            f"Cannot construct {label}; source value contains separator "
            f"{separator!r} at rows {examples}"
        )
    result = values.iloc[:, 0].str.strip()
    for field in fields[1:]:
        result = result + separator + values[field].str.strip()
    # Object dtype remains portable across the pinned AnnData/HDF5 writer; the
    # pandas StringArray categorical representation is not supported there.
    return result.astype(object).rename(label)


def add_stable_identifiers(
    frame: pd.DataFrame,
    spec: IdentifierSpec = DEFAULT_IDENTIFIER_SPEC,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add the approved stable IDs without overwriting source metadata."""
    result = frame.copy() if copy else frame
    expected = {
        "biological_unit_id": _compose(
            result, spec.biological_fields, spec.separator, "biological_unit_id"
        ),
        "source_observation_id": _compose(
            result,
            spec.source_observation_fields,
            spec.separator,
            "source_observation_id",
        ),
        "observation_id": _compose(
            result, spec.observation_fields, spec.separator, "observation_id"
        ),
    }
    for column, values in expected.items():
        if column in result:
            actual = result[column].astype("string")
            mismatch = actual.isna() | actual.ne(values.astype("string"))
            if mismatch.any():
                examples = result.index[mismatch].astype(str).tolist()[:5]
                raise ValueError(
                    f"Existing {column} violates the approved identifier contract "
                    f"at rows {examples}"
                )
        result[column] = values
    return result


def validate_identifier_contract(frame: pd.DataFrame) -> dict[str, int]:
    """Validate donor-observation mappings and return collision diagnostics."""
    required = {
        "dataset",
        "donor_id",
        "sample_id",
        "biological_unit_id",
        "source_observation_id",
        "observation_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Identifier contract is missing columns: {missing}")

    biological_mappings = (
        frame[["biological_unit_id", "dataset", "donor_id"]]
        .drop_duplicates()
        .groupby("biological_unit_id", observed=True)
        .size()
    )
    if (biological_mappings > 1).any():
        raise ValueError("A biological_unit_id maps to multiple dataset/donor pairs")

    observation_donors = (
        frame[["observation_id", "biological_unit_id"]]
        .drop_duplicates()
        .groupby("observation_id", observed=True)["biological_unit_id"]
        .nunique()
    )
    if (observation_donors > 1).any():
        raise ValueError("An observation_id maps to multiple biological donors")

    source_observation_donors = (
        frame[["source_observation_id", "biological_unit_id"]]
        .drop_duplicates()
        .groupby("source_observation_id", observed=True)["biological_unit_id"]
        .nunique()
    )
    return {
        "n_biological_units": int(frame["biological_unit_id"].nunique()),
        "n_observations": int(frame["observation_id"].nunique()),
        "n_source_observations": int(frame["source_observation_id"].nunique()),
        "n_source_observations_shared_across_donors": int(
            (source_observation_donors > 1).sum()
        ),
    }
