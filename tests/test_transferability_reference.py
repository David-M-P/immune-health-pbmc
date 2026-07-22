from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.cli.main import main
from immune_health.gene_programs.transferability import (
    TransferabilityConfig,
    select_transferable_gene_programs,
)
from immune_health.healthy_reference.diagnostics import (
    cohort_feature_age_effects,
    query_age_support,
)
from immune_health.healthy_reference.trajectory import (
    HealthyTrajectory,
    reference_row_weights,
)


def test_reference_weights_preserve_donors_and_balance_cohorts() -> None:
    donors = np.asarray(["a::1", "a::1", "a::2", "a::3", "b::1", "b::1"])
    datasets = np.asarray(["a", "a", "a", "a", "b", "b"])
    pooled = reference_row_weights(donors, datasets, scheme="donor_pooled")
    assert np.allclose(
        [pooled[donors == donor].sum() for donor in np.unique(donors)], 1.0
    )
    assert np.isclose(pooled[datasets == "a"].sum(), 3.0)
    assert np.isclose(pooled[datasets == "b"].sum(), 1.0)

    balanced = reference_row_weights(donors, datasets, scheme="cohort_balanced")
    assert np.isclose(balanced.sum(), 4.0)
    assert np.isclose(balanced[datasets == "a"].sum(), 2.0)
    assert np.isclose(balanced[datasets == "b"].sum(), 2.0)


def test_query_age_support_distinguishes_common_limited_and_out_of_range() -> None:
    rows = []
    for dataset, low, high in (("a", 20, 70), ("b", 30, 65), ("c", 40, 80)):
        for index, age in enumerate(np.linspace(low, high, 21)):
            rows.append(
                {
                    "dataset": dataset,
                    "donor": f"{dataset}::{index}",
                    "age": age,
                    "sex": "female",
                }
            )
    metadata = pd.DataFrame(rows)
    support = query_age_support(
        metadata["age"],
        metadata["sex"],
        metadata["dataset"],
        metadata["donor"],
        [50.0, 25.0, 90.0],
        ["female", "female", "female"],
        window_years=6.0,
        minimum_cohorts=2,
        minimum_donors=4,
    )
    assert support.loc[0, "age_support_status"] == "common_support"
    assert support.loc[1, "age_support_status"] in {
        "supported",
        "limited_support",
    }
    assert not support.loc[1, "in_common_cohort_sex_age_range"]
    assert support.loc[2, "age_support_status"] == "out_of_range"


def _gp_score_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(91)
    rows = []
    slopes = {
        "stable_age": (0.12, 0.12, 0.12, 0.12),
        "inconsistent_age": (0.12, -0.12, 0.12, -0.12),
    }
    for dataset_index, dataset in enumerate(("a", "b", "c", "d")):
        for donor_index, age in enumerate(np.linspace(25, 75, 30)):
            donor = f"{dataset}::d{donor_index}"
            sex = "female" if donor_index % 2 == 0 else "male"
            for program, program_slopes in slopes.items():
                score = (
                    program_slopes[dataset_index] * age
                    + 0.2 * (sex == "male")
                    + rng.normal(0, 0.08)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "biological_unit_id": donor,
                        "observation_id": f"{donor}::visit_1",
                        "age": age,
                        "sex": sex,
                        "lineage": "B cells",
                        "fine_type": "Naive B",
                        "gp_id": program,
                        "gp_score": score,
                    }
                )
    return pd.DataFrame(rows)


def test_cohort_slopes_and_transferable_gp_selection() -> None:
    scores = _gp_score_fixture()
    stable = scores.loc[scores["gp_id"].eq("stable_age")]
    effects = cohort_feature_age_effects(
        stable[["gp_score"]].to_numpy(),
        stable["age"],
        stable["sex"],
        stable["biological_unit_id"],
        stable["dataset"],
        feature_ids=["stable_age"],
        minimum_donors=20,
        minimum_age_span=20,
    )
    assert effects["eligible"].all()
    assert (effects["age_slope_per_year"] > 0.1).all()

    result = select_transferable_gene_programs(
        scores,
        config=TransferabilityConfig(
            minimum_donors_per_cohort=20,
            minimum_age_span=20,
            minimum_cohorts=4,
            minimum_sign_concordance=0.75,
            maximum_i2=0.75,
            maximum_fdr=0.05,
        ),
    )
    selection = result.selection.set_index("gp_id")
    assert bool(selection.loc["stable_age", "retained"])
    assert not bool(selection.loc["inconsistent_age", "retained"])
    assert selection.loc["stable_age", "sign_concordance"] == 1.0


def test_cli_serializes_weighting_diagnostics_and_query_support(tmp_path: Path) -> None:
    rows = []
    for dataset, offset in (("a", 0.0), ("b", 0.2), ("query", -0.1)):
        for index, age in enumerate(np.linspace(30, 70, 8)):
            rows.append(
                {
                    "dataset": dataset,
                    "donor_id": f"d{index}",
                    "sample_id": "visit_1",
                    "age": age,
                    "sex": "female",
                    "feature": age / 10.0 + offset,
                }
            )
    metadata = pd.DataFrame(rows)
    metadata_path = tmp_path / "metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    reference_dir = tmp_path / "reference"
    assert (
        main(
            [
                "fit-healthy-reference",
                "--metadata",
                str(metadata_path),
                "--feature-column",
                "feature",
                "--output-dir",
                str(reference_dir),
                "--heldout-dataset",
                "query",
                "--legacy-combined-lodo-input",
                "--n-inner-folds",
                "2",
                "--n-spline-knots",
                "0",
                "--weighting-scheme",
                "cohort_balanced",
                "--minimum-support-donors",
                "2",
                "--minimum-support-cohorts",
                "2",
                "--slope-minimum-donors",
                "4",
            ]
        )
        == 0
    )
    manifest = json.loads((reference_dir / "healthy_reference.json").read_text())
    assert manifest["weighting_scheme"] == "cohort_balanced"
    assert (reference_dir / "age_support_grid.parquet").is_file()
    assert (reference_dir / "cohort_age_slope_diagnostics.parquet").is_file()

    query = metadata.loc[metadata["dataset"].eq("query")].head(1)
    query_path = tmp_path / "query.tsv"
    query.to_csv(query_path, sep="\t", index=False)
    genes = tmp_path / "genes.txt"
    genes.write_text("ENSG1\n")
    output = tmp_path / "scores.parquet"
    report = tmp_path / "scores.json"
    assert (
        main(
            [
                "score-query",
                "--reference-manifest",
                str(reference_dir / "healthy_reference.json"),
                "--query-metadata",
                str(query_path),
                "--feature-column",
                "feature",
                "--query-genes",
                str(genes),
                "--frozen-vocabulary",
                str(genes),
                "--gp-coverage",
                "1.0",
                "--minimum-support-donors",
                "2",
                "--minimum-support-cohorts",
                "2",
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    scored = pd.read_parquet(output)
    assert scored.loc[0, "age_support_status"] in {
        "common_support",
        "supported",
        "limited_support",
    }
    assert json.loads(report.read_text())["reference_weighting_scheme"] == (
        "cohort_balanced"
    )


def test_cohort_balanced_trajectory_equalizes_training_weight() -> None:
    ages = np.asarray([20, 30, 40, 50, 60, 70], dtype=float)
    datasets = np.asarray(["large", "large", "large", "large", "small", "small"])
    donors = np.asarray(
        [f"{dataset}::{index}" for index, dataset in enumerate(datasets)]
    )
    model = HealthyTrajectory(n_spline_knots=0, weighting_scheme="cohort_balanced").fit(
        ages[:, None],
        ages,
        np.repeat("female", len(ages)),
        donors,
        datasets=datasets,
    )
    assert np.isclose(model.training_weight_summary_["large"], 3.0)
    assert np.isclose(model.training_weight_summary_["small"], 3.0)


def test_pseudobulk_cli_writes_transferable_gp_audit_tables(tmp_path: Path) -> None:
    rows = []
    counts = []
    for dataset in ("a", "b", "c", "d"):
        for index, age in enumerate((25, 35, 45, 55, 65)):
            rows.append(
                {
                    "dataset": dataset,
                    "donor_id": f"d{index}",
                    "sample_id": "visit_1",
                    "age": age,
                    "sex": "female" if index % 2 == 0 else "male",
                    "lineage": "B cells",
                    "ctype_low": "Naive B",
                    "ctype_low_conf": 0.99,
                }
            )
            counts.append([age + index, 100 - age + index])
    metadata = tmp_path / "cells.tsv"
    pd.DataFrame(rows).to_csv(metadata, sep="\t", index=False)
    count_path = tmp_path / "counts.npz"
    sparse.save_npz(count_path, sparse.csr_matrix(np.asarray(counts)))
    genes = tmp_path / "genes.txt"
    genes.write_text("ENSG1\nENSG2\n")
    programs = tmp_path / "programs.gmt"
    programs.write_text("AGE_UP\tage-associated\tENSG1\n")
    output = tmp_path / "baseline"
    assert (
        main(
            [
                "build-pseudobulk-baselines",
                "--counts",
                str(count_path),
                "--metadata",
                str(metadata),
                "--genes",
                str(genes),
                "--gp-resource",
                str(programs),
                "--minimum-gp-genes",
                "1",
                "--min-cells",
                "1",
                "--select-transferable-gps",
                "--transfer-minimum-donors",
                "3",
                "--transfer-minimum-age-span",
                "20",
                "--transfer-minimum-cohorts",
                "3",
                "--transfer-maximum-i2",
                "1",
                "--transfer-maximum-fdr",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    effects = pd.read_parquet(output / "gp_age_effects.parquet")
    selection = pd.read_parquet(output / "transferable_gp_selection.parquet")
    assert set(effects["dataset"]) == {"a", "b", "c", "d"}
    assert selection["gp_id"].tolist() == ["AGE_UP"]
    assert selection["n_cohorts_eligible"].tolist() == [4]
