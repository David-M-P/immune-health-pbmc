from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from immune_health.cli.main import main
from immune_health.healthy_reference.endpoint import (
    assemble_donor_gp_endpoint,
    validate_endpoint_inputs,
)
from immune_health.provenance import sha256_file, stable_hash


def _write_projection_manifest(
    directory: Path,
    *,
    role: str,
    datasets: list[str],
    biological_units: list[str],
) -> Path:
    directory.mkdir(parents=True)
    model_manifest = directory.parent / "frozen_model" / "model_manifest.json"
    payload = {
        "schema_version": "immune-health-tripso-projection-output/v1",
        "projection_role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "inner_model_selection": {
            "enabled": role == "validation",
            "validation_fold": 0 if role == "validation" else None,
            "fold_column": "inner_fold" if role == "validation" else None,
            "selection_role": "validation" if role == "validation" else None,
            "outer_query_used_for_model_selection": False,
        },
        "reference_design": "lodo",
        "heldout_dataset": "query",
        "fold_id": "lodo_query",
        "lineage": "B cells",
        "model_type": "Base",
        "seed": 17,
        "adapt": False,
        "optimizer_used": False,
        "all_tokenized_cells_projected": True,
        "model_manifest": str(model_manifest.resolve()),
        "arrow_dataset": f"embeddings/{role}_set",
        "n_cells": 100,
        "datasets": datasets,
        "biological_unit_ids": biological_units,
        "biological_unit_ids_sha256": stable_hash(biological_units),
        "cell_key_ordered_sha256": "cell-key-digest",
        "gp_projection": {
            "program_ids": ["GP_A", "GP_B"],
            "program_ids_ordered_sha256": stable_hash(["GP_A", "GP_B"]),
            "n_programs": 2,
        },
        "arrow_files": [
            {"path": "data.arrow", "size_bytes": 123, "sha256": "arrow-hash"}
        ],
        "hashes": {
            "arrow_tree_sha256": "arrow-tree-hash",
            "model_manifest_sha256": "model-manifest-hash",
            "checkpoint_sha256": "checkpoint-hash",
            "gp_program_ids_ordered_sha256": stable_hash(["GP_A", "GP_B"]),
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    path = directory / "projection_output_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_aggregate(
    directory: Path,
    *,
    role: str,
    dataset_sizes: dict[str, int],
    donor_offset: int = 0,
) -> tuple[Path, Path, Path]:
    biological_units = [
        f"{dataset}::{dataset}_d{index + donor_offset}"
        for dataset, size in dataset_sizes.items()
        for index in range(size)
    ]
    projection_path = _write_projection_manifest(
        directory / "projection",
        role=role,
        datasets=sorted(dataset_sizes),
        biological_units=sorted(biological_units),
    )
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection_file_hash = sha256_file(projection_path)
    conversion_path = directory / "conversion" / "arrow_conversion_manifest.json"
    conversion_hash = f"{role}-conversion-hash"
    rows: list[dict[str, object]] = []
    for dataset_number, (dataset, size) in enumerate(dataset_sizes.items()):
        for index in range(size):
            age = float(25 + 8 * index + 2 * dataset_number)
            donor_id = f"{dataset}_d{index + donor_offset}"
            rows.append(
                {
                    "dataset": dataset,
                    "donor_id": donor_id,
                    "biological_unit_id": f"{dataset}::{donor_id}",
                    "sample_id": "visit_1",
                    "source_observation_id": f"{dataset}::visit_1",
                    "observation_id": f"{dataset}::{donor_id}::visit_1",
                    "age": age,
                    "sex": "female",
                    "lineage": "B cells",
                    "fine_type": "Naive B",
                    "gp_id": "GP_A",
                    "fine_type_state_eligible": True,
                    "state_available": True,
                    "location_summary": json.dumps(
                        [age / 10.0 + dataset_number / 20.0, age / 20.0]
                    ),
                    "covariance_summary": json.dumps([[0.2, 0.01], [0.01, 0.1]]),
                    "model_id": "model-manifest-hash",
                    "model_manifest": projection["model_manifest"],
                    "model_manifest_sha256": "model-manifest-hash",
                    "checkpoint_sha256": "checkpoint-hash",
                    "fold_id": "lodo_query",
                    "seed": 17,
                    "projection_role": role,
                    "eligible_for_model_selection": role == "validation",
                    "outer_query_evaluation_only": role == "query",
                    "reference_design": "lodo",
                    "heldout_dataset": "query",
                    "projection_output_manifest": str(projection_path.resolve()),
                    "projection_output_manifest_sha256": projection_file_hash,
                    "projection_arrow_tree_sha256": "arrow-tree-hash",
                    "arrow_conversion_manifest": str(conversion_path.resolve()),
                    "arrow_conversion_manifest_sha256": conversion_hash,
                    "arrow_cell_key_ordered_sha256": "cell-key-digest",
                }
            )
    aggregate_path = directory / "fine_type_distributions.parquet"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(aggregate_path, index=False)
    aggregation_manifest = directory / "donor_distribution_aggregation_manifest.json"
    aggregation_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "donor_distribution_aggregation",
                "fine_type_distribution_table": {
                    "path": str(aggregate_path.resolve()),
                    "sha256": sha256_file(aggregate_path),
                },
                "arrow_conversion_validation": {
                    "manifest_path": str(conversion_path.resolve()),
                    "manifest_sha256": conversion_hash,
                    "projection_output": {
                        "manifest_path": str(projection_path.resolve()),
                        "manifest_sha256": projection_file_hash,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return aggregate_path, aggregation_manifest, projection_path


def _assemble(
    directory: Path,
    *,
    role: str,
    dataset_sizes: dict[str, int],
    donor_offset: int = 0,
) -> Path:
    aggregate, aggregation_manifest, projection_manifest = _write_aggregate(
        directory / "source",
        role=role,
        dataset_sizes=dataset_sizes,
        donor_offset=donor_offset,
    )
    output = directory / "endpoint"
    assemble_donor_gp_endpoint(
        aggregate,
        aggregation_manifest,
        projection_manifest,
        output,
        lineage="B cells",
        fine_type="Naive B",
        gp_id="GP_A",
    )
    return output


def test_separate_reference_and_query_endpoint_fit_and_score(tmp_path: Path) -> None:
    reference = _assemble(
        tmp_path / "reference",
        role="reference",
        dataset_sizes={"train_a": 6, "train_b": 6},
    )
    query = _assemble(
        tmp_path / "query",
        role="query",
        dataset_sizes={"query": 2},
    )
    validation = _assemble(
        tmp_path / "validation",
        role="validation",
        dataset_sizes={"train_a": 2, "train_b": 2},
        donor_offset=100,
    )
    reference_validation = validate_endpoint_inputs(
        reference / "endpoint_manifest.json",
        reference / "endpoint_metadata.parquet",
        reference / "endpoint_locations.npy",
        expected_role="reference",
    )
    assert reference_validation["heldout_dataset"] == "query"
    assert "query" not in reference_validation["datasets"]

    fitted = tmp_path / "fitted"
    assert (
        main(
            [
                "fit-healthy-reference",
                "--metadata",
                str(reference / "endpoint_metadata.parquet"),
                "--features",
                str(reference / "endpoint_locations.npy"),
                "--endpoint-manifest",
                str(reference / "endpoint_manifest.json"),
                "--output-dir",
                str(fitted),
                "--n-inner-folds",
                "2",
                "--n-spline-knots",
                "0",
                "--slope-minimum-donors",
                "2",
            ]
        )
        == 0
    )
    frozen = json.loads((fitted / "healthy_reference.json").read_text())
    assert frozen["heldout_dataset"] == "query"
    assert frozen["input_composition"] == "reference_only"
    assert frozen["endpoint_artifact"]["role"] == "reference"
    assert "query" not in frozen["training_datasets"]
    assert (fitted / "age_kernel_reference.json").is_file()
    assert not (fitted / "age_kernel_reference_arrays.npz").exists()
    kernel_manifest = json.loads(
        (fitted / "age_kernel_reference.json").read_text(encoding="utf-8")
    )
    assert (
        kernel_manifest["storage_contract"]["copied_covariance_archive_written"]
        is False
    )

    genes = tmp_path / "genes.txt"
    genes.write_text("ENSG1\n", encoding="utf-8")
    scores = tmp_path / "query_scores.parquet"
    report = tmp_path / "query_report.json"
    assert (
        main(
            [
                "score-query",
                "--reference-manifest",
                str(fitted / "healthy_reference.json"),
                "--query-metadata",
                str(query / "endpoint_metadata.parquet"),
                "--features",
                str(query / "endpoint_locations.npy"),
                "--endpoint-manifest",
                str(query / "endpoint_manifest.json"),
                "--query-genes",
                str(genes),
                "--frozen-vocabulary",
                str(genes),
                "--gp-coverage",
                "1.0",
                "--output",
                str(scores),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    scored = pd.read_parquet(scores)
    assert len(scored) == 2
    assert scored["age_matched_gaussian_wasserstein_distance"].notna().all()
    assert scored["age_matched_location_distance"].notna().all()
    assert scored["off_trajectory_gaussian_wasserstein_distance"].notna().all()
    assert scored["predicted_distributional_gp_age"].notna().all()
    query_report = json.loads(report.read_text())
    assert query_report["query_endpoint_artifact"]["role"] == "query"
    assert query_report["scoring_role"] == "query"
    assert (
        query_report["distributional_scoring"][
            "location_spline_residual_covariance_used"
        ]
        is False
    )

    validation_scores = tmp_path / "validation_scores.parquet"
    validation_report = tmp_path / "validation_report.json"
    assert (
        main(
            [
                "score-query",
                "--reference-manifest",
                str(fitted / "healthy_reference.json"),
                "--query-metadata",
                str(validation / "endpoint_metadata.parquet"),
                "--features",
                str(validation / "endpoint_locations.npy"),
                "--endpoint-manifest",
                str(validation / "endpoint_manifest.json"),
                "--query-genes",
                str(genes),
                "--frozen-vocabulary",
                str(genes),
                "--gp-coverage",
                "1.0",
                "--output",
                str(validation_scores),
                "--report",
                str(validation_report),
            ]
        )
        == 0
    )
    validation_payload = json.loads(validation_report.read_text())
    assert validation_payload["scoring_role"] == "validation"
    assert (
        validation_payload["target_endpoint_artifact"]["eligible_for_model_selection"]
        is True
    )


def test_endpoint_rejects_spoofed_aggregate_model_provenance(tmp_path: Path) -> None:
    aggregate, aggregation_manifest, projection_manifest = _write_aggregate(
        tmp_path / "source",
        role="reference",
        dataset_sizes={"train_a": 3, "train_b": 3},
    )
    table = pd.read_parquet(aggregate)
    table["model_manifest_sha256"] = "spoofed-model"
    table.to_parquet(aggregate, index=False)
    aggregation = json.loads(aggregation_manifest.read_text(encoding="utf-8"))
    aggregation["fine_type_distribution_table"]["sha256"] = sha256_file(aggregate)
    aggregation_manifest.write_text(json.dumps(aggregation), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance differs"):
        assemble_donor_gp_endpoint(
            aggregate,
            aggregation_manifest,
            projection_manifest,
            tmp_path / "endpoint",
            lineage="B cells",
            fine_type="Naive B",
            gp_id="GP_A",
        )


def test_endpoint_rejects_ontology_ineligible_fine_type(tmp_path: Path) -> None:
    aggregate, aggregation_manifest, projection_manifest = _write_aggregate(
        tmp_path / "source",
        role="reference",
        dataset_sizes={"train_a": 3, "train_b": 3},
    )
    table = pd.read_parquet(aggregate)
    table["fine_type_state_eligible"] = False
    table.to_parquet(aggregate, index=False)
    aggregation = json.loads(aggregation_manifest.read_text(encoding="utf-8"))
    aggregation["fine_type_distribution_table"]["sha256"] = sha256_file(aggregate)
    aggregation_manifest.write_text(json.dumps(aggregation), encoding="utf-8")
    with pytest.raises(ValueError, match="ontology-ineligible"):
        assemble_donor_gp_endpoint(
            aggregate,
            aggregation_manifest,
            projection_manifest,
            tmp_path / "endpoint",
            lineage="B cells",
            fine_type="Naive B",
            gp_id="GP_A",
        )


def test_combined_lodo_input_requires_explicit_legacy_flag(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "dataset": ["train", "train", "query"],
            "donor_id": ["d1", "d2", "d3"],
            "sample_id": ["v1", "v1", "v1"],
            "age": [30.0, 50.0, 40.0],
            "sex": ["female", "female", "female"],
            "feature": [1.0, 2.0, 1.5],
        }
    )
    metadata = tmp_path / "combined.tsv"
    table.to_csv(metadata, sep="\t", index=False)
    assert (
        main(
            [
                "fit-healthy-reference",
                "--metadata",
                str(metadata),
                "--feature-column",
                "feature",
                "--output-dir",
                str(tmp_path / "reference"),
                "--heldout-dataset",
                "query",
                "--n-inner-folds",
                "2",
            ]
        )
        == 2
    )
