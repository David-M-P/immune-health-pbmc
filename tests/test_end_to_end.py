from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from immune_health.io import read_parquet_artifact
from immune_health.pipeline import run_synthetic_lodo_smoke


def test_synthetic_counts_to_frozen_query_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    summary = run_synthetic_lodo_smoke(output, seed=19)

    assert summary["status"] == "complete"
    assert summary["heldout_dataset"] == "heldout_e"
    assert summary["n_training_units"] == 16
    assert summary["n_query_units"] == 4
    assert summary["mock_tripso"]["mock_adapter_smoke_passed"] is True
    assert summary["mock_tripso"]["real_tripso_training_smoke_passed"] is False

    scores, manifest = read_parquet_artifact(output / "query_scores.parquet")
    assert manifest["status"] == "complete"
    assert set(scores["dataset"]) == {"heldout_e"}
    assert scores["observation_id"].str.startswith("heldout_e::").all()
    assert scores["predicted_gp_age"].notna().all()
    assert scores["off_trajectory_distance"].ge(0).all()

    done = json.loads((output / "synthetic_lodo_smoke.done.json").read_text())
    assert done["status"] == "complete"
    assert done["stage"] == "synthetic_lodo_smoke"


def test_synthetic_vertical_slice_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_synthetic_lodo_smoke(first, seed=23)
    run_synthetic_lodo_smoke(second, seed=23)
    first_scores = pd.read_parquet(first / "query_scores.parquet")
    second_scores = pd.read_parquet(second / "query_scores.parquet")
    pd.testing.assert_frame_equal(first_scores, second_scores)
