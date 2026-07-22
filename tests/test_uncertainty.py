from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immune_health.cli.main import main
from immune_health.healthy_reference.uncertainty import combine_seed_score_tables


def _seed_score(path: Path, seed: int, value: float) -> Path:
    pd.DataFrame(
        {
            "fold_id": ["lodo_query"],
            "dataset": ["query"],
            "donor_id": ["q1"],
            "biological_unit_id": ["query::q1"],
            "sample_id": ["visit_1"],
            "observation_id": ["query::q1::visit_1"],
            "lineage": ["B cells"],
            "fine_type": ["Naive B"],
            "gp_id": ["GP_A"],
            "age": [50.0],
            "sex": ["female"],
            "seed": [seed],
            "matched_cell_depth": [100],
            "age_matched_empirical_sliced_wasserstein_distance": [value],
        }
    ).to_parquet(path, index=False)
    return path


def test_seed_combination_reports_scalar_sd_and_never_averages_coordinates(
    tmp_path: Path,
) -> None:
    first = _seed_score(tmp_path / "seed_1.parquet", 1, 2.0)
    second = _seed_score(tmp_path / "seed_2.parquet", 2, 4.0)
    output = tmp_path / "seed_summary.parquet"
    assert (
        main(
            [
                "combine-seed-scores",
                "--scores",
                str(first),
                "--scores",
                str(second),
                "--output",
                str(output),
                "--required-seed",
                "1",
                "--required-seed",
                "2",
            ]
        )
        == 0
    )
    result = pd.read_parquet(output)
    assert result["seed_mean"].tolist() == [3.0]
    assert result["seed_sd"].tolist() == pytest.approx([np.sqrt(2.0)])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["embedding_coordinates_averaged"] is False
    assert manifest["combination_unit"] == (
        "scalar_endpoint_score_after_seed_specific_calibration"
    )


def test_seed_combination_fails_closed_for_an_incomplete_seed_set(
    tmp_path: Path,
) -> None:
    first = _seed_score(tmp_path / "seed_1.parquet", 1, 2.0)
    second = _seed_score(tmp_path / "seed_2.parquet", 2, 4.0)
    with pytest.raises(ValueError, match="Endpoint seed set differs"):
        combine_seed_score_tables(
            [first, second],
            tmp_path / "bad_summary.parquet",
            tmp_path / "bad_summary.manifest.json",
            required_seeds=[1, 3],
        )
