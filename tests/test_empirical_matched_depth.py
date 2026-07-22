from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immune_health.aggregation.empirical_index import EmpiricalDistributionStore
from immune_health.cli.main import main
from immune_health.healthy_reference import empirical
from immune_health.provenance import sha256_file, stable_hash


def _ordered_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _endpoint(
    root: Path,
    *,
    role: str,
    rows: list[tuple[str, str, str, float]],
    model_hash: str = "same-model",
) -> Path:
    root.mkdir(parents=True)
    metadata_records = []
    for endpoint_row, (dataset, donor, observation, age) in enumerate(rows):
        metadata_records.append(
            {
                "endpoint_row": endpoint_row,
                "dataset": dataset,
                "donor_id": donor,
                "biological_unit_id": f"{dataset}::{donor}",
                "sample_id": "visit_1",
                "observation_id": observation,
                "age": age,
                "sex": "female",
                "lineage": "B cells",
                "fine_type": "Naive B",
                "gp_id": "GP_A",
                "projection_role": role,
                "eligible_for_model_selection": role == "validation",
                "outer_query_evaluation_only": role == "query",
                "reference_design": "lodo",
                "heldout_dataset": "query",
            }
        )
    metadata = pd.DataFrame(metadata_records)
    locations = np.column_stack(
        [metadata["age"].to_numpy() / 10.0, metadata["age"].to_numpy() / 20.0]
    ).astype(np.float32)
    covariances = np.repeat(
        np.asarray([[[0.2, 0.01], [0.01, 0.1]]], dtype=np.float32),
        len(metadata),
        axis=0,
    )
    metadata_path = root / "endpoint_metadata.parquet"
    locations_path = root / "endpoint_locations.npy"
    covariances_path = root / "endpoint_covariances.npy"
    metadata.to_parquet(metadata_path, index=False)
    np.save(locations_path, locations, allow_pickle=False)
    np.save(covariances_path, covariances, allow_pickle=False)
    projection_hash = f"{role}-projection-hash"
    payload = {
        "schema_version": "immune-health-donor-gp-endpoint/v1",
        "role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "reference_design": "lodo",
        "heldout_dataset": "query",
        "endpoint": {
            "lineage": "B cells",
            "fine_type": "Naive B",
            "gp_id": "GP_A",
        },
        "datasets": sorted(metadata["dataset"].unique().tolist()),
        "metadata_path": metadata_path.name,
        "features_path": locations_path.name,
        "covariances_path": covariances_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "features_npy_sha256": sha256_file(locations_path),
        "covariances_npy_sha256": sha256_file(covariances_path),
        "shape": list(locations.shape),
        "covariance_shape": list(covariances.shape),
        "dtype": "float32",
        "feature_ids": ["GP_A::location_0000", "GP_A::location_0001"],
        "observation_id_ordered_sha256": _ordered_digest(
            metadata["observation_id"].astype(str).tolist()
        ),
        "source_provenance": {
            "model_manifest_sha256": model_hash,
            "checkpoint_sha256": "same-checkpoint",
            "fold_id": "lodo_query",
            "seed": 17,
            "arrow_conversion_manifest_sha256": f"{role}-conversion-hash",
        },
        "projection_output_manifest_sha256": projection_hash,
        "source_aggregate_table_sha256": f"{role}-aggregation-hash",
    }
    payload["manifest_sha256"] = stable_hash(payload)
    path = root / "endpoint_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _store(
    root: Path,
    endpoint_rows: list[tuple[str, str, str, float]],
    *,
    role: str,
    eligible: bool = True,
) -> tuple[Path, EmpiricalDistributionStore]:
    root.mkdir(parents=True)
    cells_per_group = 8
    vectors = []
    records = []
    cursor = 0
    for group_number, (_, _, observation, age) in enumerate(endpoint_rows):
        for cell in range(cells_per_group):
            vectors.append(
                [
                    age / 10.0 + group_number / 20.0 + cell / 100.0,
                    age / 20.0 - cell / 200.0,
                ]
            )
        records.append(
            {
                "observation_id": observation,
                "lineage": "B cells",
                "fine_type": "Naive B",
                "gp_id": "GP_A",
                "start": cursor,
                "stop": cursor + cells_per_group,
                "n_rows": cells_per_group,
                "empirical_distance_eligible": eligible,
            }
        )
        cursor += cells_per_group
    embeddings_path = root / "embeddings.npy"
    rows_path = root / "rows.npy"
    np.save(embeddings_path, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    np.save(rows_path, np.arange(len(vectors), dtype=np.int64), allow_pickle=False)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    embedding_rows = np.load(rows_path, mmap_mode="r", allow_pickle=False)
    groups = pd.DataFrame(records)
    manifest = {
        "embedding_column": "GP_A",
        "projection_output_manifest_sha256": f"{role}-projection-hash",
        "arrow_conversion_manifest_sha256": f"{role}-conversion-hash",
        "source_embeddings_shape": list(embeddings.shape),
        "source_cell_key_ordered_sha256": f"{role}-cell-keys",
        "source_embeddings_float32_payload_sha256": f"{role}-payload",
        "groups_sha256": f"{role}-groups",
        "rows_sha256": f"{role}-rows",
        "aggregation_table_sha256": f"{role}-aggregation-hash",
        "manifest_sha256": f"{role}-manifest-content",
    }
    manifest_path = root / "empirical_distribution_manifest.json"
    manifest_path.write_text(json.dumps({"transferred": True}), encoding="utf-8")
    return manifest_path, EmpiricalDistributionStore(
        embeddings, embedding_rows, groups, manifest
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reference_rows = [
        ("train_a", "a1", "train_a::a1::visit_1", 30.0),
        ("train_a", "a2", "train_a::a2::visit_1", 40.0),
        ("train_b", "b1", "train_b::b1::visit_1", 50.0),
        ("train_b", "b2", "train_b::b2::visit_1", 60.0),
    ]
    query_rows = [("query", "q1", "query::q1::visit_1", 45.0)]
    reference_endpoint = _endpoint(
        tmp_path / "reference_endpoint", role="reference", rows=reference_rows
    )
    query_endpoint = _endpoint(
        tmp_path / "query_endpoint", role="query", rows=query_rows
    )
    reference_index, reference_store = _store(
        tmp_path / "reference_index", reference_rows, role="reference"
    )
    query_index, query_store = _store(
        tmp_path / "query_index", query_rows, role="query"
    )
    stores = {
        reference_index.resolve(): reference_store,
        query_index.resolve(): query_store,
    }
    monkeypatch.setattr(
        empirical,
        "load_empirical_distribution_store",
        lambda path: stores[Path(path).resolve()],
    )
    return reference_endpoint, query_endpoint, reference_index, query_index, stores


def test_empirical_matched_depth_cli_is_deterministic_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_endpoint, query_endpoint, reference_index, query_index, _ = _fixture(
        tmp_path, monkeypatch
    )
    output = tmp_path / "scores"
    arguments = [
        "score-empirical-endpoint",
        "--reference-endpoint-manifest",
        str(reference_endpoint),
        "--query-endpoint-manifest",
        str(query_endpoint),
        "--reference-empirical-index",
        str(reference_index),
        "--query-empirical-index",
        str(query_index),
        "--output-dir",
        str(output),
        "--depth",
        "4",
        "--depth",
        "10",
        "--n-replicates",
        "3",
        "--n-projections",
        "4",
        "--age-grid-size",
        "5",
        "--minimum-exact-sex-donors",
        "2",
        "--seed",
        "29",
    ]
    assert main(arguments) == 0
    scores = pd.read_parquet(output / "empirical_matched_depth_scores.parquet")
    replicates = pd.read_parquet(output / "empirical_matched_depth_replicates.parquet")
    reliability = pd.read_parquet(output / "empirical_reliability.parquet")
    assert scores["matched_cell_depth"].tolist() == [4]
    assert scores["seed"].tolist() == [17]
    assert scores["projection_role"].tolist() == ["query"]
    assert len(replicates) == 3
    assert np.isfinite(scores["cell_sampling_se"]).all()
    assert np.isfinite(scores["predicted_empirical_gp_age"]).all()
    assert set(reliability["status"]) == {
        "computed",
        "insufficient_query_cells",
    }
    manifest = json.loads((output / "empirical_scoring_manifest.json").read_text())
    assert manifest["leakage_checks"]["reference_query_biological_units_disjoint"]
    assert manifest["storage_contract"]["embedding_values_duplicated_on_disk"] is False
    assert manifest["storage_contract"]["covariance_arrays_read_or_copied"] is False
    assert manifest["metrics"]["spline_residual_covariance_used"] is False

    second_output = tmp_path / "scores_again"
    second_arguments = arguments.copy()
    second_arguments[second_arguments.index(str(output))] = str(second_output)
    assert main(second_arguments) == 0
    pd.testing.assert_frame_equal(
        replicates,
        pd.read_parquet(second_output / "empirical_matched_depth_replicates.parquet"),
    )


def test_empirical_scoring_rejects_cross_paired_model_and_ineligible_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_endpoint, _, reference_index, query_index, stores = _fixture(
        tmp_path, monkeypatch
    )
    query_rows = [("query", "q1", "query::q1::visit_1", 45.0)]
    wrong_query = _endpoint(
        tmp_path / "wrong_query",
        role="query",
        rows=query_rows,
        model_hash="other-model",
    )
    with pytest.raises(ValueError, match="model provenance differs"):
        empirical.score_empirical_matched_depth(
            reference_endpoint_manifest=reference_endpoint,
            query_endpoint_manifest=wrong_query,
            reference_empirical_index_manifest=reference_index,
            query_empirical_index_manifest=query_index,
            output_dir=tmp_path / "wrong_pair",
            depths=[4],
            n_replicates=2,
            n_projections=2,
            age_grid_size=3,
        )

    query_store = stores[query_index.resolve()]
    ineligible_groups = query_store.groups.copy()
    ineligible_groups["empirical_distance_eligible"] = False
    stores[query_index.resolve()] = EmpiricalDistributionStore(
        query_store.embeddings,
        query_store.embedding_rows,
        ineligible_groups,
        query_store.manifest,
    )
    correct_query = tmp_path / "query_endpoint" / "endpoint_manifest.json"
    with pytest.raises(ValueError, match="below the frozen eligibility"):
        empirical.score_empirical_matched_depth(
            reference_endpoint_manifest=reference_endpoint,
            query_endpoint_manifest=correct_query,
            reference_empirical_index_manifest=reference_index,
            query_empirical_index_manifest=query_index,
            output_dir=tmp_path / "ineligible",
            depths=[4],
            n_replicates=2,
            n_projections=2,
            age_grid_size=3,
        )


def test_empirical_scoring_accepts_inner_validation_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_endpoint, _, reference_index, _, stores = _fixture(tmp_path, monkeypatch)
    validation_rows = [
        ("train_validation", "v1", "train_validation::v1::visit_1", 45.0)
    ]
    validation_endpoint = _endpoint(
        tmp_path / "validation_endpoint",
        role="validation",
        rows=validation_rows,
    )
    validation_index, validation_store = _store(
        tmp_path / "validation_index",
        validation_rows,
        role="validation",
    )
    stores[validation_index.resolve()] = validation_store

    output = tmp_path / "validation_scores"
    manifest = empirical.score_empirical_matched_depth(
        reference_endpoint_manifest=reference_endpoint,
        query_endpoint_manifest=validation_endpoint,
        reference_empirical_index_manifest=reference_index,
        query_empirical_index_manifest=validation_index,
        output_dir=output,
        depths=[4],
        n_replicates=2,
        n_projections=2,
        age_grid_size=3,
    )
    assert manifest["target_role"] == "validation"
    assert manifest["settings"]["minimum_exact_sex_donors"] == 20
    scores = pd.read_parquet(output / "empirical_matched_depth_scores.parquet")
    assert scores["projection_role"].tolist() == ["validation"]
