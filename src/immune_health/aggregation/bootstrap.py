"""Within-observation uncertainty from fine-type-stratified resampling."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np


def _target_counts(
    labels: np.ndarray,
    strata: tuple[str, ...],
    rng: np.random.Generator,
    *,
    mode: str,
    standardized_proportions: Mapping[str, float] | None,
    resample_composition: bool,
) -> np.ndarray:
    observed = np.asarray([(labels == stratum).sum() for stratum in strata], dtype=int)
    total = int(observed.sum())
    if mode == "observed_mixture":
        if not resample_composition:
            return observed
        probabilities = observed / total
    elif mode == "composition_standardized":
        if standardized_proportions is None:
            raise ValueError(
                "standardized proportions are required for standardized mode"
            )
        probabilities = np.asarray(
            [standardized_proportions.get(stratum, 0.0) for stratum in strata],
            dtype=float,
        )
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("standardized proportions must be nonnegative and nonzero")
        probabilities /= probabilities.sum()
        unavailable = (observed == 0) & (probabilities > 0)
        if unavailable.any():
            missing = [strata[index] for index in np.flatnonzero(unavailable)]
            raise ValueError(f"cannot standardize unmeasured fine types: {missing}")
        if not resample_composition:
            raw = probabilities * total
            counts = np.floor(raw).astype(int)
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[: total - counts.sum()]] += 1
            return counts
    else:
        raise ValueError(
            "mode must be 'observed_mixture' or 'composition_standardized'"
        )
    return rng.multinomial(total, probabilities)


def stratified_bootstrap_indices(
    fine_types: Sequence[str],
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
    mode: str = "observed_mixture",
    standardized_proportions: Mapping[str, float] | None = None,
    resample_composition: bool = False,
) -> Iterator[np.ndarray]:
    """Yield deterministic cell indices resampled only within fine-type strata."""

    labels = np.asarray(fine_types, dtype=str)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("fine_types must be a nonempty one-dimensional sequence")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    strata = tuple(dict.fromkeys(labels))
    members = {stratum: np.flatnonzero(labels == stratum) for stratum in strata}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        counts = _target_counts(
            labels,
            strata,
            rng,
            mode=mode,
            standardized_proportions=standardized_proportions,
            resample_composition=resample_composition,
        )
        pieces = [
            rng.choice(members[stratum], size=count, replace=True)
            for stratum, count in zip(strata, counts)
            if count > 0
        ]
        yield np.concatenate(pieces) if pieces else np.asarray([], dtype=int)


def fine_type_stratified_bootstrap(
    values: np.ndarray,
    fine_types: Sequence[str],
    statistic: Callable[[np.ndarray, np.ndarray], float | np.ndarray],
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
    mode: str = "observed_mixture",
    standardized_proportions: Mapping[str, float] | None = None,
    resample_composition: bool = False,
) -> np.ndarray:
    """Recalculate a user-supplied statistic for stratified bootstrap samples."""

    matrix = np.asarray(values)
    labels = np.asarray(fine_types, dtype=str)
    if len(matrix) != len(labels):
        raise ValueError("values and fine_types row counts differ")
    estimates = [
        statistic(matrix[index], labels[index])
        for index in stratified_bootstrap_indices(
            labels,
            n_bootstrap=n_bootstrap,
            seed=seed,
            mode=mode,
            standardized_proportions=standardized_proportions,
            resample_composition=resample_composition,
        )
    ]
    return np.asarray(estimates)
