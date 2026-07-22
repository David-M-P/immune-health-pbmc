from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from immune_health.aggregation.empirical_index import (
    load_empirical_distribution_store,
    write_empirical_row_index,
)
from immune_health.cli.main import main as immune_health_main
from immune_health.provenance import sha256_file, stable_hash
from immune_health.tripso_adapter.arrow_bridge import (
    convert_tripso_arrow_embeddings,
    validate_arrow_conversion_for_aggregation,
)


def _write_huggingface_arrow(path: Path, table: pa.Table) -> Path:
    path.mkdir()
    arrow_path = path / "data-00000-of-00001.arrow"
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=2)
    (path / "state.json").write_text(
        json.dumps({"_data_files": [{"filename": "data-00000-of-00001.arrow"}]}),
        encoding="utf-8",
    )
    return path


def _ordered_key_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_projection_output_manifest(
    arrow_dir: Path,
    *,
    keys: list[str],
    datasets: list[str],
    lineage: str = "B cells",
    programs: list[str] | None = None,
) -> Path:
    records = [
        {
            "path": path.relative_to(arrow_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(arrow_dir.rglob("*"))
        if path.is_file()
    ]
    program_ids = programs or ["GP_A"]
    biological_units = sorted(
        {f"{dataset}::d{index + 1}" for index, dataset in enumerate(datasets)}
    )
    payload = {
        "schema_version": "immune-health-tripso-projection-output/v1",
        "projection_role": "reference",
        "eligible_for_model_selection": False,
        "outer_query_evaluation_only": False,
        "inner_model_selection": {},
        "reference_design": "all_healthy",
        "heldout_dataset": None,
        "fold_id": "all_healthy",
        "lineage": lineage,
        "model_type": "Base",
        "seed": 17,
        "adapt": False,
        "optimizer_used": False,
        "all_tokenized_cells_projected": True,
        "model_manifest": str((arrow_dir.parent / "model_manifest.json").resolve()),
        "arrow_dataset": arrow_dir.name,
        "n_cells": len(keys),
        "datasets": sorted(set(datasets)),
        "biological_unit_ids": biological_units,
        "biological_unit_ids_sha256": stable_hash(biological_units),
        "cell_key_ordered_sha256": _ordered_key_digest(keys),
        "gp_projection": {
            "program_ids": program_ids,
            "program_ids_ordered_sha256": stable_hash(program_ids),
            "n_programs": len(program_ids),
        },
        "arrow_files": records,
        "hashes": {
            "arrow_tree_sha256": stable_hash(records),
            "model_manifest_sha256": "model-hash",
            "checkpoint_sha256": "checkpoint-hash",
            "gp_program_ids_ordered_sha256": stable_hash(program_ids),
        },
    }
    payload["manifest_sha256"] = stable_hash(payload)
    path = arrow_dir.parent / f"{arrow_dir.name}_projection_output_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_arrow_bridge_joins_by_safe_cell_key_and_preserves_arrow_order(
    tmp_path: Path,
) -> None:
    keys = ["a::cell-2", "a::cell-1", "b::cell-3"]
    vectors = [[2.0, 2.5], [1.0, 1.5], [3.0, 3.5]]
    table = pa.table(
        {
            "cell_key": keys,
            "GP_A": vectors,
            "GP_A_prop_genes": [0.5, 1.0, 0.25],
        }
    )
    arrow_dir = _write_huggingface_arrow(tmp_path / "arrow", table)
    projection_manifest = _write_projection_output_manifest(
        arrow_dir, keys=keys, datasets=["a", "a", "b"]
    )
    metadata = pd.DataFrame(
        {
            "cell_key": [keys[2], keys[0], keys[1]],
            "dataset": ["b", "a", "a"],
            "donor_id": ["d3", "d2", "d1"],
            "sample_id": ["s3", "s2", "s1"],
            "observation_id": ["b::d3::s3", "a::d2::s2", "a::d1::s1"],
            "lineage": ["B cells"] * 3,
            "fine_type": ["Memory B", "Naive B", "Naive B"],
            "fine_type_state_eligible": [True] * 3,
            "fine_type_balance_eligible": [True] * 3,
            "age": [60.0, 40.0, 30.0],
            "sex": ["female", "female", "male"],
        }
    )
    metadata_path = tmp_path / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    output = tmp_path / "converted"
    manifest = convert_tripso_arrow_embeddings(
        arrow_dir,
        metadata_path,
        output,
        projection_output_manifest=projection_manifest,
        embedding_columns=["GP_A"],
    )
    embedding_path = output / manifest["embedding_outputs"]["GP_A"]["path"]
    observed = np.load(embedding_path, allow_pickle=False)
    assert observed.dtype == np.float32
    assert np.array_equal(observed, np.asarray(vectors, dtype=np.float32))
    aligned = pd.read_parquet(output / "cell_metadata.parquet")
    assert aligned["cell_key"].tolist() == keys
    assert aligned["observation_id"].tolist() == [
        "a::d2::s2",
        "a::d1::s1",
        "b::d3::s3",
    ]
    assert manifest["alignment_validation"]["row_count_only_alignment"] is False
    validation = validate_arrow_conversion_for_aggregation(
        output / "arrow_conversion_manifest.json",
        embedding_path,
        output / "cell_metadata.parquet",
        embedding_column="GP_A",
    )
    assert validation["passed"] is True
    assert validation["cell_key_ordered_sha256"] == manifest["cell_key_ordered_sha256"]
    index_output = tmp_path / "empirical_index"
    first_key = ("a::d2::s2", "B cells", "Naive B", "GP_A")
    middle_key = ("a::d1::s1", "B cells", "Naive B", "GP_A")
    second_key = ("b::d3::s3", "B cells", "Memory B", "GP_A")
    index_manifest = write_empirical_row_index(
        index_output,
        aligned,
        {
            first_key: np.asarray([0], dtype=np.int64),
            middle_key: np.asarray([1], dtype=np.int64),
            second_key: np.asarray([2], dtype=np.int64),
        },
        frozenset({first_key}),
        conversion_validation=validation,
    )
    assert index_manifest["copied_embedding_values"] is False
    assert not (index_output / "empirical_distributions.npz").exists()
    store = load_empirical_distribution_store(
        index_output / "empirical_distribution_manifest.json"
    )
    assert isinstance(store.embeddings, np.memmap)
    np.testing.assert_array_equal(
        store.get(first_key), np.asarray(vectors[:1], dtype=np.float32)
    )
    aggregation_output = tmp_path / "aggregation"
    assert (
        immune_health_main(
            [
                "aggregate-donor-distributions",
                "--embeddings",
                str(embedding_path),
                "--metadata",
                str(output / "cell_metadata.parquet"),
                "--arrow-conversion-manifest",
                str(output / "arrow_conversion_manifest.json"),
                "--gp-id",
                "GP_A",
                "--output-dir",
                str(aggregation_output),
                "--min-state-cells",
                "2",
                "--min-empirical-cells",
                "2",
            ]
        )
        == 0
    )
    assert (aggregation_output / "empirical_distribution_manifest.json").is_file()
    assert (aggregation_output / "empirical_distribution_rows.npy").is_file()
    assert not (aggregation_output / "empirical_distributions.npz").exists()


def test_arrow_bridge_rejects_string_key_named_like_integer_id(tmp_path: Path) -> None:
    table = pa.table({"cell_key": ["a::c1"], "GP_A": [[1.0, 2.0]]})
    arrow_dir = _write_huggingface_arrow(tmp_path / "arrow", table)
    projection_manifest = _write_projection_output_manifest(
        arrow_dir, keys=["a::c1"], datasets=["a"]
    )
    metadata = pd.DataFrame(
        {
            "cell_key": ["a::c1"],
            "dataset": ["a"],
            "donor_id": ["d1"],
            "sample_id": ["s1"],
            "observation_id": ["a::d1::s1"],
            "lineage": ["B cells"],
            "fine_type": ["Naive B"],
            "age": [40.0],
            "sex": ["female"],
        }
    )
    metadata_path = tmp_path / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    try:
        convert_tripso_arrow_embeddings(
            arrow_dir,
            metadata_path,
            tmp_path / "output",
            projection_output_manifest=projection_manifest,
            embedding_columns=["GP_A"],
            cell_key_column="cell_id",
        )
    except ValueError as exc:
        assert "must not end in '_id'" in str(exc)
    else:
        raise AssertionError("Unsafe string *_id key was accepted")


def test_aggregation_binding_rejects_cross_paired_same_row_count_outputs(
    tmp_path: Path,
) -> None:
    converted: list[tuple[Path, dict[str, object]]] = []
    for label, keys in (
        ("first", ["a::cell-1", "a::cell-2"]),
        ("second", ["b::cell-1", "b::cell-2"]),
    ):
        table = pa.table({"cell_key": keys, "GP_A": [[1.0, 2.0], [3.0, 4.0]]})
        arrow_dir = _write_huggingface_arrow(tmp_path / f"arrow_{label}", table)
        projection_manifest = _write_projection_output_manifest(
            arrow_dir, keys=keys, datasets=[label, label]
        )
        metadata = pd.DataFrame(
            {
                "cell_key": keys,
                "dataset": [label, label],
                "donor_id": ["d1", "d2"],
                "sample_id": ["s1", "s2"],
                "observation_id": [f"{label}::d1::s1", f"{label}::d2::s2"],
                "lineage": ["B cells", "B cells"],
                "fine_type": ["Naive B", "Memory B"],
                "age": [40.0, 60.0],
                "sex": ["female", "male"],
            }
        )
        source_metadata = tmp_path / f"metadata_{label}.parquet"
        metadata.to_parquet(source_metadata, index=False)
        output = tmp_path / f"converted_{label}"
        manifest = convert_tripso_arrow_embeddings(
            arrow_dir,
            source_metadata,
            output,
            projection_output_manifest=projection_manifest,
            embedding_columns=["GP_A"],
        )
        converted.append((output, manifest))

    first_output, first_manifest = converted[0]
    second_output, _ = converted[1]
    first_embedding = first_output / first_manifest["embedding_outputs"]["GP_A"]["path"]
    with pytest.raises(ValueError, match="Metadata path is not the output bound"):
        validate_arrow_conversion_for_aggregation(
            first_output / "arrow_conversion_manifest.json",
            first_embedding,
            second_output / "cell_metadata.parquet",
            embedding_column="GP_A",
        )
