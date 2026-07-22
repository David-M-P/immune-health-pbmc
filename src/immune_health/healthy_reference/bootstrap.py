"""Healthy-reference uncertainty from resampling biological donors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .trajectory import HealthyTrajectory


def bootstrap_healthy_reference_scores(
    reference_features: np.ndarray,
    ages: Sequence[float],
    sexes: Sequence[str],
    biological_unit_ids: Sequence[str],
    query_location: np.ndarray,
    query_age: float,
    query_sex: str,
    *,
    datasets: Sequence[str] | None = None,
    n_bootstrap: int = 100,
    seed: int = 0,
    model_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Resample healthy donors, refit the trajectory and rescore one query.

    Every sampled donor contributes all of its rows.  Duplicate bootstrap draws
    receive distinct replicate group IDs, avoiding any cell- or row-level
    resampling masquerading as additional people.
    """

    features = np.asarray(reference_features, dtype=float)
    age = np.asarray(ages, dtype=float)
    sex = np.asarray(sexes, dtype=str)
    donors = np.asarray(biological_unit_ids, dtype=str)
    dataset = None if datasets is None else np.asarray(datasets, dtype=str)
    if features.ndim != 2 or any(
        len(value) != len(features) for value in (age, sex, donors)
    ):
        raise ValueError("healthy-reference features and metadata must align")
    if dataset is not None and len(dataset) != len(features):
        raise ValueError("dataset labels and reference features must align")
    unique_donors = np.unique(donors)
    if len(unique_donors) < 2 or n_bootstrap < 1:
        raise ValueError("bootstrap requires at least two donors and one replicate")
    donor_rows = {donor: np.flatnonzero(donors == donor) for donor in unique_donors}
    donor_strata: dict[tuple[str, str], list[str]] = {}
    for donor in unique_donors:
        selected = donor_rows[donor]
        donor_sexes = np.unique(sex[selected])
        donor_datasets = (
            np.asarray(["__all__"]) if dataset is None else np.unique(dataset[selected])
        )
        if len(donor_sexes) != 1 or len(donor_datasets) != 1:
            raise ValueError("sex and dataset must be constant within each donor")
        stratum = (str(donor_datasets[0]), str(donor_sexes[0]))
        donor_strata.setdefault(stratum, []).append(str(donor))
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []
    for replicate in range(n_bootstrap):
        draw = np.concatenate(
            [
                rng.choice(members, size=len(members), replace=True)
                for members in donor_strata.values()
            ]
        )
        rows: list[np.ndarray] = []
        replicate_groups: list[np.ndarray] = []
        for draw_index, donor in enumerate(draw):
            selected = donor_rows[str(donor)]
            rows.append(selected)
            replicate_groups.append(
                np.repeat(f"{donor}::bootstrap_{draw_index}", len(selected))
            )
        index = np.concatenate(rows)
        groups = np.concatenate(replicate_groups)
        model = HealthyTrajectory(**dict(model_kwargs or {})).fit(
            features[index],
            age[index],
            sex[index],
            groups,
            datasets=None if dataset is None else dataset[index],
        )
        score = model.score(query_location, query_age, query_sex)
        records.append({"bootstrap_replicate": replicate, **score})
    return pd.DataFrame.from_records(records)
