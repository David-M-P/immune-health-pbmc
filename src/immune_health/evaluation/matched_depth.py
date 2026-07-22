"""Repeated equal-cell-depth sensitivity for distributional distances."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from immune_health.aggregation.distances import sliced_wasserstein_distance


def matched_depth_sensitivity(
    query_cells: np.ndarray,
    reference_cells: np.ndarray,
    *,
    depths: Sequence[int] = (25, 50, 100, 250, 500, 1_000),
    n_replicates: int = 20,
    seed: int = 0,
    distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_projections: int = 128,
) -> pd.DataFrame:
    """Compare equal-size donor/reference samples without replacement."""

    query = np.asarray(query_cells, dtype=float)
    reference = np.asarray(reference_cells, dtype=float)
    if query.ndim == 1:
        query = query[:, None]
    if reference.ndim == 1:
        reference = reference[:, None]
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference cell embeddings must share dimensions")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if distance is None:

        def distance(left: np.ndarray, right: np.ndarray) -> float:
            return sliced_wasserstein_distance(
                left, right, n_projections=n_projections, seed=seed
            )

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for depth_value in depths:
        depth = int(depth_value)
        if depth < 1:
            raise ValueError("matched depths must be positive")
        if depth > len(query) or depth > len(reference):
            records.append(
                {
                    "cell_depth": depth,
                    "replicate": -1,
                    "distance": np.nan,
                    "status": "insufficient_depth",
                    "query_cells_available": len(query),
                    "reference_cells_available": len(reference),
                }
            )
            continue
        for replicate in range(n_replicates):
            query_index = rng.choice(len(query), size=depth, replace=False)
            reference_index = rng.choice(len(reference), size=depth, replace=False)
            records.append(
                {
                    "cell_depth": depth,
                    "replicate": replicate,
                    "distance": float(
                        distance(query[query_index], reference[reference_index])
                    ),
                    "status": "ok",
                    "query_cells_available": len(query),
                    "reference_cells_available": len(reference),
                }
            )
    return pd.DataFrame.from_records(records)


def reliability_curve(matched_depth_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched-depth mean, sampling SD and replicate count."""

    required = {"cell_depth", "distance", "status"}
    if not required.issubset(matched_depth_results):
        raise ValueError(f"matched-depth table requires {sorted(required)}")
    valid = matched_depth_results[matched_depth_results["status"] == "ok"]
    return (
        valid.groupby("cell_depth", observed=True)["distance"]
        .agg(mean_distance="mean", sampling_sd="std", n_replicates="size")
        .reset_index()
    )
