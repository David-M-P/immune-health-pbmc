from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from immune_health.cli.main import build_parser, main

COMMANDS = {
    "audit-data",
    "build-fine-type-ontology",
    "make-lodo-folds",
    "validate-gene-programs",
    "build-sampling-manifest",
    "build-pseudobulk-baselines",
    "train-tripso",
    "project-tripso",
    "aggregate-donor-distributions",
    "assemble-donor-gp-endpoint",
    "fit-healthy-reference",
    "select-transferable-tripso-gps",
    "score-query",
    "score-empirical-endpoint",
    "combine-seed-scores",
    "bootstrap-scores",
    "evaluate-lodo",
    "build-comparison-report",
}


def _write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_every_command_has_common_options() -> None:
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparser_action.choices) == COMMANDS
    for command, command_parser in subparser_action.choices.items():
        destinations = {action.dest for action in command_parser._actions}
        assert {"config", "dry_run", "seed", "log_level"} <= destinations, command


def test_exact_sex_support_defaults_are_aligned_at_twenty() -> None:
    parser = build_parser()
    fit = parser.parse_args(
        [
            "fit-healthy-reference",
            "--metadata",
            "metadata.parquet",
            "--output-dir",
            "reference",
        ]
    )
    empirical = parser.parse_args(
        [
            "score-empirical-endpoint",
            "--reference-endpoint-manifest",
            "reference.json",
            "--query-endpoint-manifest",
            "query.json",
            "--reference-empirical-index",
            "reference-index.json",
            "--query-empirical-index",
            "query-index.json",
            "--output-dir",
            "empirical",
        ]
    )

    assert fit.age_kernel_minimum_exact_sex_donors == 20
    assert empirical.minimum_exact_sex_donors == 20


def test_lodo_cli_writes_collision_safe_donor_manifests(tmp_path: Path) -> None:
    rows = []
    for dataset in ("cohort_a", "cohort_b"):
        for donor_number in range(3):
            rows.extend(
                {
                    "dataset": dataset,
                    "donor_id": f"d{donor_number}",
                    "sample_id": f"pool_{donor_number % 2}",
                    "age": 20 + donor_number * 10,
                    "sex": "female" if donor_number % 2 == 0 else "male",
                    "lineage": lineage,
                }
                for lineage in ("B cells", "CD4_like")
            )
    metadata = _write_tsv(pd.DataFrame(rows), tmp_path / "metadata.tsv")
    output = tmp_path / "splits"
    result = main(
        [
            "make-lodo-folds",
            "--metadata",
            str(metadata),
            "--output-dir",
            str(output),
            "--dataset",
            "cohort_a",
            "--dataset",
            "cohort_b",
            "--n-inner-folds",
            "2",
            "--seed",
            "7",
        ]
    )
    assert result == 0
    global_manifest = pd.read_csv(output / "global_donor_manifest.tsv", sep="\t")
    assert (
        global_manifest["biological_unit_id"].str.match(r"^cohort_[ab]::d[0-2]$").all()
    )
    assert global_manifest["biological_unit_id"].is_unique
    fold = pd.read_csv(output / "lodo_cohort_b.tsv", sep="\t")
    assert set(fold.loc[fold["outer_role"] == "query", "dataset"]) == {"cohort_b"}
    contract = json.loads((output / "split_manifest.json").read_text())[
        "identifier_definitions"
    ]
    assert contract["observation_id"] == "dataset::donor_id::sample_id"


def test_ontology_and_sampling_cli_are_executable(tmp_path: Path) -> None:
    fine_summary = pd.DataFrame(
        {
            "dataset": ["a", "b"],
            "lineage": ["B cells", "B cells"],
            "fine_type": ["Naive B", "Memory B"],
            "n_cells": [100, 80],
            "n_donors": [10, 8],
            "confidence_mean": [0.96, 0.94],
            "confidence_lt_0_9": [2, 3],
        }
    )
    fine_path = _write_tsv(fine_summary, tmp_path / "fine.tsv")
    ontology_path = tmp_path / "ontology.yaml"
    summary_path = tmp_path / "ontology_summary.tsv"
    assert (
        main(
            [
                "build-fine-type-ontology",
                "--input",
                str(fine_path),
                "--output",
                str(ontology_path),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )
    assert "pending_scientific_review" in ontology_path.read_text()

    cells = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "donor_id": ["d1", "d1", "d2", "d2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
            "lineage": ["B cells"] * 4,
            "ctype_low": ["Naive B", "Memory B"] * 2,
            "cell_id": ["c1", "c2", "c3", "c4"],
        }
    )
    cell_path = _write_tsv(cells, tmp_path / "cells.tsv")
    sampling_output = tmp_path / "sampling"
    assert (
        main(
            [
                "build-sampling-manifest",
                "--metadata",
                str(cell_path),
                "--output-dir",
                str(sampling_output),
                "--lineage",
                "B cells",
                "--n-cells",
                "8",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )
    manifest = json.loads((sampling_output / "sampling_manifest.json").read_text())
    assert manifest["identifier_contract"]["biological_unit_id"] == (
        "dataset::donor_id"
    )
    assert len(pd.read_parquet(sampling_output / "selected_cells.parquet")) == 8


def test_frozen_healthy_reference_cli_scores_unseen_query(tmp_path: Path) -> None:
    reference_rows = []
    for dataset in ("train_a", "train_b"):
        for index in range(4):
            age = 25.0 + index * 12
            reference_rows.append(
                {
                    "dataset": dataset,
                    "donor_id": f"{dataset}_d{index}",
                    "sample_id": "visit_1",
                    "age": age,
                    "sex": "female",
                    "feature_1": age / 10 + (dataset == "train_b") * 0.1,
                    "feature_2": age / 20,
                }
            )
    reference_rows.append(
        {
            "dataset": "heldout",
            "donor_id": "heldout_d1",
            "sample_id": "visit_1",
            "age": 50.0,
            "sex": "female",
            "feature_1": 5.0,
            "feature_2": 2.5,
        }
    )
    reference_path = _write_tsv(
        pd.DataFrame(reference_rows), tmp_path / "reference.tsv"
    )
    reference_output = tmp_path / "healthy"
    assert (
        main(
            [
                "fit-healthy-reference",
                "--metadata",
                str(reference_path),
                "--feature-column",
                "feature_1",
                "--feature-column",
                "feature_2",
                "--output-dir",
                str(reference_output),
                "--heldout-dataset",
                "heldout",
                "--legacy-combined-lodo-input",
                "--n-inner-folds",
                "2",
                "--n-spline-knots",
                "0",
            ]
        )
        == 0
    )
    fitted = json.loads((reference_output / "healthy_reference.json").read_text())
    assert "heldout" not in fitted["training_datasets"]

    query = pd.DataFrame(
        {
            "dataset": ["future_cohort"],
            "donor_id": ["future_d1"],
            "sample_id": ["visit_2"],
            "age": [49.0],
            "sex": ["female"],
            "feature_1": [5.1],
            "feature_2": [2.4],
        }
    )
    query_path = _write_tsv(query, tmp_path / "query.tsv")
    genes = tmp_path / "genes.txt"
    genes.write_text("ENSG1\nENSG2\n")
    scored = tmp_path / "scored.parquet"
    report = tmp_path / "query_report.json"
    assert (
        main(
            [
                "score-query",
                "--reference-manifest",
                str(reference_output / "healthy_reference.json"),
                "--query-metadata",
                str(query_path),
                "--feature-column",
                "feature_1",
                "--feature-column",
                "feature_2",
                "--query-genes",
                str(genes),
                "--frozen-vocabulary",
                str(genes),
                "--gp-coverage",
                "1.0",
                "--output",
                str(scored),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    result = pd.read_parquet(scored)
    assert result.loc[0, "observation_id"] == "future_cohort::future_d1::visit_2"
    assert pd.notna(result.loc[0, "predicted_gp_age"])
    diagnostics = json.loads(report.read_text())
    assert diagnostics["query_dataset_offset"] == "not_fitted"
