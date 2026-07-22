from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from immune_health.aggregation.empirical_index import (
    load_empirical_distribution_store,
)
from immune_health.cli.main import main
from immune_health.provenance import sha256_file, stable_hash
from immune_health.tripso_adapter.arrow_bridge import (
    convert_tripso_arrow_embeddings,
)
from tests.test_arrow_bridge import _ordered_key_digest, _write_huggingface_arrow


def _write_projection_artifact(
    directory: Path,
    *,
    role: str,
    donors_by_dataset: dict[str, list[str]],
    model_manifest: Path,
) -> tuple[Path, pd.DataFrame]:
    directory.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    keys: list[str] = []
    vectors: list[list[float]] = []
    ages = (26.0, 39.0, 53.0, 68.0)
    cell_offsets = ((-0.12, 0.04), (0.0, -0.08), (0.14, 0.06))
    for cohort_index, (dataset, donors) in enumerate(donors_by_dataset.items()):
        for donor_index, donor_id in enumerate(donors):
            age = ages[donor_index % len(ages)] + 2.0 * cohort_index
            observation_id = f"{dataset}::{donor_id}::visit_1"
            for cell_index, (first_offset, second_offset) in enumerate(cell_offsets):
                cell_key = f"{dataset}::{donor_id}::cell_{cell_index}"
                keys.append(cell_key)
                vectors.append(
                    [
                        age / 10.0 + 0.1 * cohort_index + first_offset,
                        age / 20.0 - 0.05 * cohort_index + second_offset,
                    ]
                )
                rows.append(
                    {
                        "cell_key": cell_key,
                        "dataset": dataset,
                        "donor_id": donor_id,
                        "sample_id": "visit_1",
                        "observation_id": observation_id,
                        "lineage": "B cells",
                        "fine_type": "Naive B",
                        "fine_type_state_eligible": True,
                        "fine_type_balance_eligible": True,
                        "age": age,
                        "sex": "female",
                    }
                )

    arrow_dir = _write_huggingface_arrow(
        directory / "arrow",
        pa.table({"cell_key": keys, "GP_A": vectors}),
    )
    metadata = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    metadata_path = directory / "source_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    files = [
        {
            "path": path.relative_to(arrow_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(arrow_dir.rglob("*"))
        if path.is_file()
    ]
    biological_units = sorted(
        f"{dataset}::{donor_id}"
        for dataset, donors in donors_by_dataset.items()
        for donor_id in donors
    )
    inner_selection = {
        "enabled": role == "validation",
        "validation_fold": 0 if role == "validation" else None,
        "fold_column": "inner_fold" if role == "validation" else None,
        "selection_role": "validation" if role == "validation" else None,
        "outer_query_used_for_model_selection": False,
    }
    projection = {
        "schema_version": "immune-health-tripso-projection-output/v1",
        "projection_role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": False,
        "inner_model_selection": inner_selection,
        "reference_design": "lodo",
        "heldout_dataset": "outer_query",
        "fold_id": "lodo_outer_query",
        "lineage": "B cells",
        "model_type": "Base",
        "seed": 17,
        "adapt": False,
        "optimizer_used": False,
        "all_tokenized_cells_projected": True,
        "model_manifest": str(model_manifest.resolve()),
        "arrow_dataset": arrow_dir.name,
        "n_cells": len(keys),
        "datasets": sorted(donors_by_dataset),
        "biological_unit_ids": biological_units,
        "biological_unit_ids_sha256": stable_hash(biological_units),
        "cell_key_ordered_sha256": _ordered_key_digest(keys),
        "gp_projection": {
            "program_ids": ["GP_A"],
            "program_ids_ordered_sha256": stable_hash(["GP_A"]),
            "n_programs": 1,
        },
        "arrow_files": files,
        "hashes": {
            "arrow_tree_sha256": stable_hash(files),
            "model_manifest_sha256": "frozen-model-hash",
            "checkpoint_sha256": "frozen-checkpoint-hash",
            "gp_program_ids_ordered_sha256": stable_hash(["GP_A"]),
        },
    }
    projection["manifest_sha256"] = stable_hash(projection)
    projection_path = directory / "projection_output_manifest.json"
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    return projection_path, metadata_path


def _convert_aggregate_and_assemble(
    directory: Path,
    *,
    role: str,
    donors_by_dataset: dict[str, list[str]],
    model_manifest: Path,
    fine_type_ontology: Path,
) -> Path:
    projection_path, metadata_path = _write_projection_artifact(
        directory,
        role=role,
        donors_by_dataset=donors_by_dataset,
        model_manifest=model_manifest,
    )
    conversion_dir = directory / "conversion"
    conversion = convert_tripso_arrow_embeddings(
        directory / "arrow",
        metadata_path,
        conversion_dir,
        projection_output_manifest=projection_path,
        embedding_columns=["GP_A"],
    )
    embedding_path = conversion_dir / conversion["embedding_outputs"]["GP_A"]["path"]
    aggregation_dir = directory / "aggregation"
    assert (
        main(
            [
                "aggregate-donor-distributions",
                "--embeddings",
                str(embedding_path),
                "--metadata",
                str(conversion_dir / "cell_metadata.parquet"),
                "--arrow-conversion-manifest",
                str(conversion_dir / "arrow_conversion_manifest.json"),
                "--gp-id",
                "GP_A",
                "--fine-type-universe",
                str(fine_type_ontology),
                "--min-state-cells",
                "2",
                "--min-empirical-cells",
                "2",
                "--output-dir",
                str(aggregation_dir),
            ]
        )
        == 0
    )
    endpoint_dir = directory / "endpoint"
    assert (
        main(
            [
                "assemble-donor-gp-endpoint",
                "--aggregate-table",
                str(aggregation_dir / "fine_type_distributions.parquet"),
                "--aggregation-manifest",
                str(aggregation_dir / "donor_distribution_aggregation_manifest.json"),
                "--projection-output-manifest",
                str(projection_path),
                "--lineage",
                "B cells",
                "--fine-type",
                "Naive B",
                "--gp-id",
                "GP_A",
                "--output-dir",
                str(endpoint_dir),
            ]
        )
        == 0
    )
    return endpoint_dir


def test_production_arrow_to_validation_score_vertical_slice(tmp_path: Path) -> None:
    model_manifest = tmp_path / "frozen_model" / "model_manifest.json"
    fine_type_ontology = tmp_path / "frozen_fine_type_ontology.json"
    fine_type_ontology.write_text(
        json.dumps(
            {
                "lineages": {
                    "B cells": {
                        "mappings": [
                            {"canonical_fine_type": "Naive B"},
                            {"canonical_fine_type": "Memory B"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ontology_sha256 = sha256_file(fine_type_ontology)
    reference = _convert_aggregate_and_assemble(
        tmp_path / "reference",
        role="reference",
        donors_by_dataset={
            "train_a": [f"shared_{index}" for index in range(4)],
            "train_b": [f"shared_{index}" for index in range(4)],
        },
        model_manifest=model_manifest,
        fine_type_ontology=fine_type_ontology,
    )
    validation = _convert_aggregate_and_assemble(
        tmp_path / "validation",
        role="validation",
        donors_by_dataset={
            "train_a": ["validation_0", "validation_1"],
            "train_b": ["validation_0", "validation_1"],
        },
        model_manifest=model_manifest,
        fine_type_ontology=fine_type_ontology,
    )

    reference_metadata = pd.read_parquet(reference / "endpoint_metadata.parquet")
    validation_metadata = pd.read_parquet(validation / "endpoint_metadata.parquet")
    assert reference_metadata["observation_id"].is_unique
    assert validation_metadata["observation_id"].is_unique
    assert (
        reference_metadata["biological_unit_id"]
        == reference_metadata["dataset"] + "::" + reference_metadata["donor_id"]
    ).all()
    assert (
        reference_metadata["observation_id"]
        == reference_metadata["biological_unit_id"]
        + "::"
        + reference_metadata["sample_id"]
    ).all()
    reference_units = set(reference_metadata["biological_unit_id"])
    validation_units = set(validation_metadata["biological_unit_id"])
    assert reference_units.isdisjoint(validation_units)
    assert set(reference_metadata["donor_id"]) == {
        f"shared_{index}" for index in range(4)
    }
    assert len(reference_units) == 8

    for root, expected_role in (
        (tmp_path / "reference", "reference"),
        (tmp_path / "validation", "validation"),
    ):
        aggregate = pd.read_parquet(
            root / "aggregation" / "fine_type_distributions.parquet"
        )
        unobserved = aggregate.loc[aggregate["fine_type"].eq("Memory B")]
        assert len(unobserved) == aggregate["observation_id"].nunique()
        assert unobserved["n_cells"].eq(0).all()
        assert not unobserved["state_available"].any()
        empirical_path = root / "aggregation" / "empirical_distribution_manifest.json"
        empirical = json.loads(empirical_path.read_text(encoding="utf-8"))
        assert empirical["copied_embedding_values"] is False
        assert not (root / "aggregation" / "empirical_distributions.npz").exists()
        store = load_empirical_distribution_store(empirical_path)
        assert isinstance(store.embeddings, np.memmap)
        endpoint = json.loads(
            (root / "endpoint" / "endpoint_manifest.json").read_text(encoding="utf-8")
        )
        assert endpoint["role"] == expected_role
        assert endpoint["source_provenance"]["projection_role"] == expected_role
        assert endpoint["source_provenance"]["model_id"] == "frozen-model-hash"
    assert sha256_file(fine_type_ontology) == ontology_sha256

    fitted = tmp_path / "fitted_reference"
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
    assert frozen["input_composition"] == "reference_only"
    assert frozen["endpoint_artifact"]["role"] == "reference"
    assert frozen["distributional_reference"]["endpoint_locations_reused"] is True
    assert frozen["distributional_reference"]["endpoint_covariances_reused"] is True
    kernel = json.loads((fitted / "age_kernel_reference.json").read_text())
    assert kernel["storage_contract"]["copied_covariance_archive_written"] is False
    assert not (fitted / "age_kernel_reference_arrays.npz").exists()

    genes = tmp_path / "genes.txt"
    genes.write_text("ENSG000001\n", encoding="utf-8")
    scores = tmp_path / "validation_scores.parquet"
    report = tmp_path / "validation_score_report.json"
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
                str(scores),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    scored = pd.read_parquet(scores)
    assert len(scored) == len(validation_metadata)
    assert scored["age_matched_gaussian_wasserstein_distance"].notna().all()
    score_report = json.loads(report.read_text(encoding="utf-8"))
    assert score_report["scoring_role"] == "validation"
    assert (
        score_report["target_endpoint_artifact"]["eligible_for_model_selection"] is True
    )
    assert (
        score_report["target_endpoint_artifact"]["outer_query_evaluation_only"] is False
    )
