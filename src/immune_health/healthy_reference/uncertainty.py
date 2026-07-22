"""Manifest-bound combination of scalar scores across independent model seeds."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from immune_health.provenance import atomic_write_json, sha256_file, stable_hash

SEED_COMBINATION_SCHEMA = "immune-health-seed-score-combination/v1"
DEFAULT_SCALAR_METRICS = (
    "age_matched_location_distance",
    "off_trajectory_location_distance",
    "age_matched_gaussian_wasserstein_distance",
    "off_trajectory_gaussian_wasserstein_distance",
    "predicted_gp_age",
    "gp_age_acceleration",
    "predicted_distributional_gp_age",
    "distributional_gp_age_acceleration",
    "age_matched_empirical_sliced_wasserstein_distance",
    "off_trajectory_empirical_sliced_wasserstein_distance",
    "predicted_empirical_gp_age",
    "empirical_gp_age_acceleration",
)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def combine_seed_score_tables(
    input_paths: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    metrics: Sequence[str] = (),
    required_seeds: Sequence[int] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute SD across scalar scores; never average embedding coordinates."""

    paths = tuple(Path(path).resolve() for path in input_paths)
    if len(paths) < 2 or len(paths) != len(set(paths)):
        raise ValueError("Seed combination requires at least two unique score tables")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Seed score tables are missing: {missing}")
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError("Refusing to overwrite seed-combination outputs")

    frames = [pd.read_parquet(path) for path in paths]
    required_identity = {
        "fold_id",
        "dataset",
        "biological_unit_id",
        "observation_id",
        "lineage",
        "fine_type",
        "gp_id",
        "age",
        "sex",
        "seed",
    }
    for path, frame in zip(paths, frames, strict=True):
        missing_columns = sorted(required_identity - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Seed score table {path} lacks {missing_columns}")
    selected_metrics = tuple(dict.fromkeys(map(str, metrics)))
    if not selected_metrics:
        selected_metrics = tuple(
            metric
            for metric in DEFAULT_SCALAR_METRICS
            if all(metric in frame for frame in frames)
        )
    if not selected_metrics:
        raise ValueError("No common approved scalar score metric is available")
    unknown = sorted(set(selected_metrics) - set(DEFAULT_SCALAR_METRICS))
    if unknown:
        raise ValueError(f"Unapproved/non-scalar seed metrics requested: {unknown}")

    combined = pd.concat(frames, ignore_index=True)
    combined["seed"] = pd.to_numeric(combined["seed"], errors="raise").astype(int)
    for metric in selected_metrics:
        combined[metric] = pd.to_numeric(combined[metric], errors="raise")
        if not np.isfinite(combined[metric]).all():
            raise ValueError(f"Seed metric contains non-finite values: {metric}")
    key_columns = [
        column
        for column in (
            "fold_id",
            "dataset",
            "donor_id",
            "biological_unit_id",
            "sample_id",
            "observation_id",
            "age",
            "sex",
            "lineage",
            "fine_type",
            "gp_id",
            "matched_cell_depth",
        )
        if column in combined
    ]
    if combined.duplicated([*key_columns, "seed"]).any():
        raise ValueError("A biological endpoint has duplicate rows within one seed")
    expected_seeds = set(map(int, required_seeds))
    if expected_seeds and len(expected_seeds) < 2:
        raise ValueError("required_seeds must contain at least two unique seeds")

    records: list[dict[str, Any]] = []
    for key, group in combined.groupby(key_columns, observed=True, sort=True):
        observed_seeds = set(group["seed"].astype(int))
        if len(observed_seeds) < 2:
            raise ValueError(f"Endpoint has fewer than two model seeds: {key}")
        if expected_seeds and observed_seeds != expected_seeds:
            raise ValueError(
                f"Endpoint seed set differs; expected={sorted(expected_seeds)}, "
                f"observed={sorted(observed_seeds)}"
            )
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(key_columns, values, strict=True))
        for metric in selected_metrics:
            metric_values = group[metric].to_numpy(dtype=float)
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "n_seeds": len(metric_values),
                    "seeds": "|".join(map(str, sorted(observed_seeds))),
                    "seed_mean": float(metric_values.mean()),
                    "seed_sd": float(metric_values.std(ddof=1)),
                }
            )
    result = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output_path, result)
    payload: dict[str, Any] = {
        "schema_version": SEED_COMBINATION_SCHEMA,
        "status": "complete",
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "seeds": sorted(frame["seed"].astype(int).unique().tolist()),
                "n_rows": len(frame),
            }
            for path, frame in zip(paths, frames, strict=True)
        ],
        "metrics": list(selected_metrics),
        "required_seeds": sorted(expected_seeds),
        "combination_unit": "scalar_endpoint_score_after_seed_specific_calibration",
        "embedding_coordinates_averaged": False,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "n_rows": len(result),
            "columns": list(result.columns),
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return payload
