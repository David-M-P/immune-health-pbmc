from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, load_from_disk
from scipy import sparse

from immune_health.data.lineage_scope import (
    LINEAGE_DONOR_SCOPE_SCHEMA,
    canonical_json_digest,
)
from immune_health.tripso_adapter.contracts import (
    canonical_json_hash,
    load_fold_input_manifest,
    sha256_path,
)
from immune_health.tripso_adapter.projection import (
    make_query_only_datamodule_class,
    validate_frozen_query_resources,
)
from immune_health.tripso_adapter.provenance import build_model_artifact_manifest
from immune_health.tripso_adapter.tokenization import (
    build_fold_input_from_tokenization,
    build_query_input_from_tokenization,
    build_reference_input_from_tokenization,
    build_validation_input_from_tokenization,
    load_tokenization_manifest,
    relocate_tokenization_manifest,
    tokenize_fold_h5ad,
)
from immune_health.tripso_adapter.training import (
    TripsoTrainingSpec,
    build_training_call,
)


class _FakeInspectedTokenizer:
    """Small vendor-surface double; orchestration, not rank math, is under test."""

    def __init__(
        self,
        *,
        custom_attr_name_dict,
        nproc,
        model_input_size,
        special_token,
        collapse_gene_ids,
        keep_counts,
        gene_mapping_file,
        token_dictionary_file,
        gene_median_file,
    ) -> None:
        self.custom_attr_name_dict = custom_attr_name_dict
        self.model_input_size = model_input_size

    def tokenize_files(self, directory: Path, file_format: str = "h5ad"):
        input_path = next(Path(directory).glob("*.h5ad"))
        adata = ad.read_h5ad(input_path)
        metadata = {
            output: adata.obs[source].tolist()
            for source, output in self.custom_attr_name_dict.items()
        }
        sequences = []
        for key in adata.obs["cell_key"].astype(str):
            # One cell deliberately crosses the inspected 4094 ranked-gene limit.
            length = 4095 if key.endswith("c0") else 2
            sequences.append(np.arange(4, 4 + length, dtype=np.int64))
        return sequences, metadata, []

    def create_dataset(
        self,
        tokenized_cells,
        cell_metadata,
        tokenized_counts,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        sequences = [
            [2, *sequence[: self.model_input_size - 2].tolist(), 3]
            for sequence in tokenized_cells
        ]
        return Dataset.from_dict(
            {
                "input_ids": sequences,
                "length": [len(sequence) for sequence in sequences],
                **cell_metadata,
            }
        )


def _resources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    genes = [f"ENSG{index:011d}" for index in range(1, 4)]
    vocabulary = tmp_path / "model_genes_hvg3000.txt"
    vocabulary.write_text("".join(f"{gene}\n" for gene in genes))
    gp = tmp_path / "gpdb_filtered.csv"
    pd.DataFrame({"GP_A": [genes[0], genes[1]], "GP_B": [genes[1], genes[2]]}).to_csv(
        gp, index=False
    )
    token_dictionary = tmp_path / "tokens.pkl"
    median_dictionary = tmp_path / "medians.pkl"
    with token_dictionary.open("wb") as handle:
        pickle.dump(
            {"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3}
            | {gene: index + 4 for index, gene in enumerate(genes)},
            handle,
        )
    with median_dictionary.open("wb") as handle:
        pickle.dump({gene: 1.0 for gene in genes}, handle)
    return vocabulary, gp, token_dictionary, median_dictionary


def _h5ad(
    path: Path,
    *,
    role: str,
    dataset: str,
    donors: tuple[str, ...],
    lineage: str = "B cells",
) -> Path:
    rows = []
    for index in range(4):
        donor = donors[index % len(donors)]
        rows.append(
            {
                "cell_key": f"{dataset}::c{index}",
                "dataset": dataset,
                "donor_id": donor,
                "biological_unit_id": f"{dataset}::{donor}",
                "sample_id": "s1",
                "source_observation_id": f"{dataset}::s1",
                "observation_id": f"{dataset}::{donor}::s1",
                "fine_type": "naive B" if index % 2 == 0 else "memory B",
                "ctype_low": "naive B" if index % 2 == 0 else "memory B",
                "ctype_low_conf": 0.95,
                "fine_type_state_eligible": True,
                "fine_type_balance_eligible": True,
                "fine_type_mapping_status": "approved_identity",
                "lineage": lineage,
                "preparation_role": role,
                "n_counts": 100 + index,
            }
        )
    genes = [f"ENSG{index:011d}" for index in range(1, 4)]
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 2, 3]] * 4, dtype=np.float32)),
        obs=pd.DataFrame(rows, index=[f"source-{index}" for index in range(4)]),
        var=pd.DataFrame({"ensembl_id": genes}, index=genes),
    )
    adata.write_h5ad(path)
    return path


def _tokenize(
    tmp_path: Path,
    *,
    role: str,
    dataset: str,
    donors: tuple[str, ...],
    name: str,
    lineage: str = "B cells",
) -> tuple[dict, Path, Path]:
    vocabulary, gp, token_dictionary, median_dictionary = _resources(tmp_path)
    candidate_path = tmp_path / "projection_gp_candidates.json"
    candidate = {
        "schema_version": "immune-health-projection-gp-candidates/v1",
        "selection_level": "donor_lineage_pseudobulk",
        "query_data_consulted": False,
        "program_ids": ["GP_A", "GP_B"],
        "program_ids_ordered_sha256": canonical_json_hash(["GP_A", "GP_B"]),
        "binding": {"gpdb_sha256": sha256_path(gp)},
    }
    candidate["manifest_content_sha256"] = canonical_json_hash(candidate)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    h5ad = _h5ad(
        tmp_path / f"{name}.h5ad",
        role=role,
        dataset=dataset,
        donors=donors,
        lineage=lineage,
    )
    output = tmp_path / name
    manifest = tokenize_fold_h5ad(
        input_h5ad=h5ad,
        gene_vocabulary_path=vocabulary,
        gp_library_path=gp,
        projection_gp_candidates_path=candidate_path,
        output_dir=output,
        vendor_root=tmp_path,
        role=role,
        row_chunk_size=2,
        nproc=1,
        minimum_tokenizable_gp_genes=1,
        _vendor_surface=(
            _FakeInspectedTokenizer,
            token_dictionary,
            median_dictionary,
            Path(__file__),
        ),
    )
    return manifest, vocabulary, gp


def _lineage_scope(
    *,
    lineage: str,
    adaptation: tuple[str, ...],
    validation: tuple[str, ...] = (),
    query: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> dict:
    role_donors = {
        "adaptation": sorted(adaptation),
        "validation": sorted(validation),
        "query": sorted(query),
    }
    global_donors = sorted(
        set(adaptation) | set(validation) | set(query) | set(missing)
    )
    payload = {
        "schema_version": LINEAGE_DONOR_SCOPE_SCHEMA,
        "lineage": lineage,
        "scope_unit": "biological_unit_id",
        "scope_source": (
            "physical_per_lineage_cell_metadata_after_fold_and_visit_selection"
        ),
        "biological_unit_ids_by_preparation_role": role_donors,
        "n_biological_units_by_preparation_role": {
            role: len(donors) for role, donors in role_donors.items()
        },
        "biological_unit_ids_by_preparation_role_sha256": {
            role: canonical_json_digest(donors) for role, donors in role_donors.items()
        },
        "n_source_lineage_biological_units": len(
            set(adaptation) | set(validation) | set(query)
        ),
        "n_global_fold_biological_units": len(global_donors),
        "global_fold_biological_unit_ids_sha256": canonical_json_digest(global_donors),
        "global_fold_biological_unit_ids_without_materialized_role_cells": sorted(
            missing
        ),
        "n_global_fold_biological_units_without_materialized_role_cells": len(missing),
    }
    payload["scope_sha256"] = canonical_json_digest(payload)
    return payload


def _attach_lineage_scope(manifest_path: Path, scope: dict) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["lineage_donor_scope"] = scope
    payload["manifest_sha256"] = canonical_json_hash(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_tokenization_preserves_all_cells_metadata_and_reports_truncation(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _tokenize(
        tmp_path,
        role="adaptation",
        dataset="train",
        donors=("d1", "d2"),
        name="adaptation_tokens",
    )
    dataset = load_from_disk(manifest["tokenized_dataset_path"])
    assert len(dataset) == 4
    assert {
        "cell_key",
        "dataset",
        "biological_unit_id",
        "observation_id",
        "fine_type",
        "fine_type_state_eligible",
        "fine_type_balance_eligible",
        "lineage",
        "idx",
        "length_uncropped",
        "was_truncated",
    } <= set(dataset.column_names)
    assert dataset["cell_key"] == [f"train::c{index}" for index in range(4)]
    assert set(dataset["biological_unit_id"]) == {"train::d1", "train::d2"}
    assert manifest["cell_downsampling_performed"] is False
    assert manifest["hvg_calculation_performed"] is False
    assert manifest["sequence_qc"]["n_cells_truncated"] == 1
    assert manifest["sequence_qc"]["fraction_cells_truncated"] == 0.25
    assert not (
        Path(manifest["tokenized_dataset_path"]).parent / "tokenized_chunks"
    ).exists()


def test_tokenization_relocation_verifies_exact_sftp_copy_and_rebinds_fold(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_cluster"
    source_root.mkdir()
    source, _, _ = _tokenize(
        source_root,
        role="adaptation",
        dataset="train",
        donors=("d1", "d2"),
        name="adaptation_tokens",
    )
    destination_root = tmp_path / "gefion"
    shutil.copytree(source_root, destination_root)
    transferred_source_manifest = destination_root / "source_manifest.json"
    shutil.copy2(
        destination_root / "adaptation_tokens" / "tokenization_manifest.json",
        transferred_source_manifest,
    )
    # Prove the import does not rely on any source-cluster absolute path.
    shutil.rmtree(source_root)

    relocated_manifest_path = (
        destination_root / "adaptation_tokens" / "tokenization_manifest.json"
    )
    relocated = relocate_tokenization_manifest(
        source_manifest_path=transferred_source_manifest,
        output_manifest_path=relocated_manifest_path,
        tokenized_dataset_path=(
            destination_root / "adaptation_tokens" / "tokenized.dataset"
        ),
        input_h5ad=destination_root / "adaptation_tokens.h5ad",
        gene_vocabulary_path=destination_root / "model_genes_hvg3000.txt",
        gp_library_path=destination_root / "gpdb_filtered.csv",
        projection_gp_candidates_path=(
            destination_root / "projection_gp_candidates.json"
        ),
        vendor_root=destination_root,
        overwrite=True,
        _vendor_surface=(
            _FakeInspectedTokenizer,
            destination_root / "tokens.pkl",
            destination_root / "medians.pkl",
            Path(__file__),
        ),
    )
    assert relocated["relocation"]["scientific_inputs_changed"] is False
    assert (
        relocated["relocation"]["source_manifest_content_sha256"]
        == source["manifest_sha256"]
    )
    assert Path(relocated["tokenized_dataset_path"]).is_relative_to(destination_root)
    assert Path(relocated["input_h5ad"]).is_relative_to(destination_root)

    fold_table = destination_root / "fold.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "donor_id": donor,
                "sample_id": "s1",
                "outer_role": "reference",
                "eligible_for_reference_fitting": True,
            }
            for donor in ("d1", "d2")
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    fold = build_fold_input_from_tokenization(
        tokenization_manifest_path=relocated_manifest_path,
        fold_table_path=fold_table,
        output_path=destination_root / "fold_input.json",
        fold_id="all_healthy",
        held_out_dataset=None,
        lineage="B cells",
        reference_design="all_healthy",
    )
    assert Path(fold["inputs"]["tokenized_dataset_path"]).is_relative_to(
        destination_root
    )

    arrow_file = next(
        (destination_root / "adaptation_tokens" / "tokenized.dataset").rglob("*.arrow")
    )
    arrow_file.write_bytes(arrow_file.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="file inventory differs"):
        load_tokenization_manifest(relocated_manifest_path)


def test_tokenization_relocation_rejects_changed_immutable_resource(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_cluster"
    source_root.mkdir()
    _tokenize(
        source_root,
        role="query",
        dataset="query",
        donors=("q1",),
        name="query_tokens",
    )
    destination_root = tmp_path / "gefion"
    shutil.copytree(source_root, destination_root)
    source_manifest = destination_root / "source_manifest.json"
    shutil.copy2(
        destination_root / "query_tokens" / "tokenization_manifest.json",
        source_manifest,
    )
    vocabulary = destination_root / "model_genes_hvg3000.txt"
    vocabulary.write_text(vocabulary.read_text() + "ENSG99999999999\n")
    with pytest.raises(ValueError, match="not byte-identical"):
        relocate_tokenization_manifest(
            source_manifest_path=source_manifest,
            output_manifest_path=destination_root / "relocated.json",
            tokenized_dataset_path=(
                destination_root / "query_tokens" / "tokenized.dataset"
            ),
            input_h5ad=destination_root / "query_tokens.h5ad",
            gene_vocabulary_path=vocabulary,
            gp_library_path=destination_root / "gpdb_filtered.csv",
            projection_gp_candidates_path=(
                destination_root / "projection_gp_candidates.json"
            ),
            vendor_root=destination_root,
            _vendor_surface=(
                _FakeInspectedTokenizer,
                destination_root / "tokens.pkl",
                destination_root / "medians.pkl",
                Path(__file__),
            ),
        )


def test_physical_donor_scope_builds_fold_and_query_frozen_contract(
    tmp_path: Path,
) -> None:
    training, vocabulary, gp = _tokenize(
        tmp_path,
        role="adaptation",
        dataset="train",
        donors=("d1", "d2"),
        name="training_tokens",
    )
    fold_table = tmp_path / "lodo_held.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "donor_id": donor,
                "sample_id": "s1",
                "outer_role": "reference",
                "eligible_for_reference_fitting": True,
            }
            for donor in ("d1", "d2")
        ]
        + [
            {
                "dataset": "train",
                "donor_id": "d3",
                "sample_id": "s1",
                "outer_role": "validation",
                "eligible_for_reference_fitting": False,
            },
            {
                "dataset": "held",
                "donor_id": "q1",
                "sample_id": "s1",
                "outer_role": "query",
                "eligible_for_reference_fitting": False,
            },
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    fold_input_path = tmp_path / "fold_input.json"
    fold = build_fold_input_from_tokenization(
        tokenization_manifest_path=tmp_path
        / "training_tokens"
        / "tokenization_manifest.json",
        fold_table_path=fold_table,
        output_path=fold_input_path,
        fold_id="lodo_held",
        held_out_dataset="held",
        lineage="B cells",
    )
    assert fold["tokenized_dataset_scope_validation"]["status"] == "passed"
    assert fold["adaptation_biological_unit_ids"] == ["train::d1", "train::d2"]
    assert "tokenization_manifest_sha256" in fold["hashes"]

    checkpoint = tmp_path / "model" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    model_manifest_path = tmp_path / "model" / "model_manifest.json"
    build_model_artifact_manifest(
        output_path=model_manifest_path,
        repo_root=Path(__file__).parents[1],
        vendor_root=Path(__file__).parents[1] / "tripso_code" / "tripso",
        fold_input_manifest_path=fold_input_path,
        checkpoint_path=checkpoint,
        fold_id="lodo_held",
        held_out_dataset="held",
        lineage="B cells",
        model_type="Base",
        sampler_mode="hybrid",
        alpha=0.5,
        fine_type_lambda=0.7,
        seed=42,
        gp_library_path=gp,
        gene_vocabulary_path=vocabulary,
        model_configuration={
            "tokenizer": "geneformer_may2025",
            "preprocessing": "fold_input_manifest",
            "embedding_dimension": 256,
            "model_type": "Base",
        },
    )
    query, _, _ = _tokenize(
        tmp_path,
        role="query",
        dataset="held",
        donors=("q1",),
        name="query_tokens",
    )
    query_input_path = tmp_path / "query_input.json"
    query_input = build_query_input_from_tokenization(
        tokenization_manifest_path=tmp_path
        / "query_tokens"
        / "tokenization_manifest.json",
        model_manifest_path=model_manifest_path,
        output_path=query_input_path,
        allow_all_gps=True,
    )
    assert query_input["adapt"] is False
    assert (
        query_input["hashes"]["gene_vocabulary_sha256"]
        == training["hashes"]["gene_vocabulary_sha256"]
    )
    assert query_input["biological_unit_ids"] == ["held::q1"]
    assert query_input["projection_role"] == "query"
    assert query_input["n_cells"] == 4

    reference_input_path = tmp_path / "reference_input.json"
    reference_input = build_reference_input_from_tokenization(
        tokenization_manifest_path=tmp_path
        / "training_tokens"
        / "tokenization_manifest.json",
        model_manifest_path=model_manifest_path,
        output_path=reference_input_path,
        allow_all_gps=True,
    )
    assert reference_input["projection_role"] == "reference"
    assert reference_input["biological_unit_ids"] == ["train::d1", "train::d2"]
    assert reference_input["adapt"] is False
    assert reference_input["optimizer_allowed"] is False
    validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=reference_input,
    )
    validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=query_input,
    )

    _tokenize(
        tmp_path,
        role="validation",
        dataset="train",
        donors=("d3",),
        name="validation_tokens",
    )
    validation_input = build_validation_input_from_tokenization(
        tokenization_manifest_path=(
            tmp_path / "validation_tokens" / "tokenization_manifest.json"
        ),
        model_manifest_path=model_manifest_path,
        output_path=tmp_path / "validation_input.json",
        use_fold_bound_gp_candidates=True,
    )
    assert validation_input["projection_role"] == "validation"
    assert validation_input["biological_unit_ids"] == ["train::d3"]
    validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=validation_input,
    )

    bound_reference = build_reference_input_from_tokenization(
        tokenization_manifest_path=tmp_path
        / "training_tokens"
        / "tokenization_manifest.json",
        model_manifest_path=model_manifest_path,
        output_path=tmp_path / "bound_reference_input.json",
        use_fold_bound_gp_candidates=True,
    )
    assert bound_reference["gp_projection"]["mode"] == ("frozen_training_candidates")
    assert bound_reference["gp_projection"]["program_ids"] == ["GP_A", "GP_B"]
    assert bound_reference["gp_projection"]["estimated_gp_vector_bytes"] == (
        4 * 2 * 256 * 4
    )
    validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=bound_reference,
    )
    with pytest.raises(ValueError, match="exceeds the configured byte guard"):
        build_reference_input_from_tokenization(
            tokenization_manifest_path=tmp_path
            / "training_tokens"
            / "tokenization_manifest.json",
            model_manifest_path=model_manifest_path,
            output_path=tmp_path / "oversized_reference_input.json",
            use_fold_bound_gp_candidates=True,
            max_projected_bytes=1,
        )

    # Runtime validation re-reads Arrow scope; a self-consistent hand-edited JSON
    # donor declaration is still rejected.
    forged = dict(query_input)
    forged["biological_unit_ids"] = ["train::d1"]
    forged.pop("manifest_sha256")
    forged["manifest_sha256"] = canonical_json_hash(forged)
    with pytest.raises(
        ValueError, match="declaration differs from physical Arrow donors"
    ):
        validate_frozen_query_resources(
            model_manifest_path=model_manifest_path,
            query_manifest=forged,
        )

    _tokenize(
        tmp_path,
        role="query",
        dataset="train",
        donors=("d3",),
        name="validation_overlap_tokens",
    )
    with pytest.raises(ValueError, match="adaptation/validation biological units"):
        build_query_input_from_tokenization(
            tokenization_manifest_path=tmp_path
            / "validation_overlap_tokens"
            / "tokenization_manifest.json",
            model_manifest_path=model_manifest_path,
            output_path=tmp_path / "forbidden_validation_query.json",
            allow_all_gps=True,
        )


def test_fixed_inner_fold_rebinds_only_physical_adaptation_donors(
    tmp_path: Path,
) -> None:
    _tokenize(
        tmp_path,
        role="adaptation",
        dataset="train",
        donors=("d2",),
        name="fixed_inner_training",
    )
    fold_table = tmp_path / "fixed_inner.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "donor_id": "d1",
                "outer_role": "reference",
                "inner_fold": 0,
                "eligible_for_reference_fitting": True,
            },
            {
                "dataset": "train",
                "donor_id": "d2",
                "outer_role": "reference",
                "inner_fold": 1,
                "eligible_for_reference_fitting": True,
            },
            {
                "dataset": "held",
                "donor_id": "q1",
                "outer_role": "query",
                "inner_fold": pd.NA,
                "eligible_for_reference_fitting": False,
            },
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    output = tmp_path / "fixed_inner_fold_input.json"
    manifest = build_fold_input_from_tokenization(
        tokenization_manifest_path=(
            tmp_path / "fixed_inner_training" / "tokenization_manifest.json"
        ),
        fold_table_path=fold_table,
        output_path=output,
        fold_id="lodo_held",
        held_out_dataset="held",
        lineage="B cells",
        inner_validation_fold=0,
        inner_fold_column="inner_fold",
    )
    assert manifest["adaptation_biological_unit_ids"] == ["train::d2"]
    assert manifest["validation_biological_unit_ids"] == ["train::d1"]
    assert manifest["query_biological_unit_ids"] == ["held::q1"]
    assert manifest["inner_model_selection"] == {
        "enabled": True,
        "validation_fold": 0,
        "fold_column": "inner_fold",
        "selection_role": "validation",
        "outer_query_used_for_model_selection": False,
    }
    assert load_fold_input_manifest(output)["inner_model_selection"]["enabled"]


def test_lineage_scope_excludes_absent_donors_without_reassigning_roles(
    tmp_path: Path,
) -> None:
    lineage = "Monocytes"
    scope = _lineage_scope(
        lineage=lineage,
        adaptation=("train::a_present",),
        validation=("train::v_present",),
        query=("held::q_present",),
        missing=("held::q_absent", "train::a_absent", "train::v_absent"),
    )
    training, vocabulary, gp = _tokenize(
        tmp_path,
        role="adaptation",
        dataset="train",
        donors=("a_present",),
        name="lineage_training",
        lineage=lineage,
    )
    training_manifest_path = (
        tmp_path / "lineage_training" / "tokenization_manifest.json"
    )
    _attach_lineage_scope(training_manifest_path, scope)

    fold_table = tmp_path / "lineage_lodo.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "donor_id": "a_present",
                "outer_role": "reference",
                "inner_fold": 1,
            },
            {
                "dataset": "train",
                "donor_id": "a_absent",
                "outer_role": "reference",
                "inner_fold": 2,
            },
            {
                "dataset": "train",
                "donor_id": "v_present",
                "outer_role": "reference",
                "inner_fold": 0,
            },
            {
                "dataset": "train",
                "donor_id": "v_absent",
                "outer_role": "reference",
                "inner_fold": 0,
            },
            {
                "dataset": "held",
                "donor_id": "q_present",
                "outer_role": "query",
                "inner_fold": pd.NA,
            },
            {
                "dataset": "held",
                "donor_id": "q_absent",
                "outer_role": "query",
                "inner_fold": pd.NA,
            },
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    fold_input_path = tmp_path / "lineage_fold_input.json"
    fold = build_fold_input_from_tokenization(
        tokenization_manifest_path=training_manifest_path,
        fold_table_path=fold_table,
        output_path=fold_input_path,
        fold_id="lodo_held",
        held_out_dataset="held",
        lineage=lineage,
        inner_validation_fold=0,
        inner_fold_column="inner_fold",
    )
    assert fold["adaptation_biological_unit_ids"] == ["train::a_present"]
    assert fold["validation_biological_unit_ids"] == ["train::v_present"]
    assert fold["query_biological_unit_ids"] == ["held::q_present"]
    scope_audit = fold["lineage_donor_scope_validation"]
    assert scope_audit["status"] == "passed"
    assert scope_audit["global_fold_biological_unit_ids_excluded_by_original_role"] == {
        "adaptation": ["train::a_absent"],
        "validation": ["train::v_absent"],
        "query": ["held::q_absent"],
    }
    assert load_fold_input_manifest(fold_input_path) == fold

    checkpoint = tmp_path / "lineage_model" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    model_manifest_path = tmp_path / "lineage_model" / "model_manifest.json"
    build_model_artifact_manifest(
        output_path=model_manifest_path,
        repo_root=Path(__file__).parents[1],
        vendor_root=Path(__file__).parents[1] / "tripso_code" / "tripso",
        fold_input_manifest_path=fold_input_path,
        checkpoint_path=checkpoint,
        fold_id="lodo_held",
        held_out_dataset="held",
        lineage=lineage,
        model_type="Base",
        sampler_mode="hybrid",
        alpha=0.5,
        fine_type_lambda=0.7,
        seed=42,
        gp_library_path=gp,
        gene_vocabulary_path=vocabulary,
        model_configuration={
            "tokenizer": "geneformer_may2025",
            "preprocessing": "fold_input_manifest",
            "embedding_dimension": 256,
            "model_type": "Base",
        },
    )

    for role, dataset, donor, builder in (
        ("validation", "train", "v_present", build_validation_input_from_tokenization),
        ("query", "held", "q_present", build_query_input_from_tokenization),
    ):
        _tokenize(
            tmp_path,
            role=role,
            dataset=dataset,
            donors=(donor,),
            name=f"lineage_{role}",
            lineage=lineage,
        )
        tokenization_path = tmp_path / f"lineage_{role}" / "tokenization_manifest.json"
        _attach_lineage_scope(tokenization_path, scope)
        projection = builder(
            tokenization_manifest_path=tokenization_path,
            model_manifest_path=model_manifest_path,
            output_path=tmp_path / f"lineage_{role}_input.json",
            allow_all_gps=True,
        )
        assert projection["biological_unit_ids"] == [f"{dataset}::{donor}"]
        validate_frozen_query_resources(
            model_manifest_path=model_manifest_path,
            query_manifest=projection,
        )

    reference = build_reference_input_from_tokenization(
        tokenization_manifest_path=training_manifest_path,
        model_manifest_path=model_manifest_path,
        output_path=tmp_path / "lineage_reference_input.json",
        allow_all_gps=True,
    )
    assert reference["biological_unit_ids"] == ["train::a_present"]
    validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=reference,
    )


@pytest.mark.parametrize(
    ("lineage", "dataset", "missing"),
    [
        ("NK_ILC", "aidav2", ("JP_RIK_H001",)),
        (
            "Monocytes",
            "onek1k",
            ("1071_1072", "193_194", "281_282", "641_642"),
        ),
    ],
)
def test_known_lineage_absences_are_valid_all_healthy_exclusions(
    tmp_path: Path,
    lineage: str,
    dataset: str,
    missing: tuple[str, ...],
) -> None:
    present = "present_donor"
    physical_id = f"{dataset}::{present}"
    missing_ids = tuple(f"{dataset}::{donor}" for donor in missing)
    scope = _lineage_scope(
        lineage=lineage,
        adaptation=(physical_id,),
        missing=missing_ids,
    )
    _tokenize(
        tmp_path,
        role="adaptation",
        dataset=dataset,
        donors=(present,),
        name=f"{lineage}_all_healthy".lower(),
        lineage=lineage,
    )
    tokenization_path = (
        tmp_path / f"{lineage}_all_healthy".lower() / "tokenization_manifest.json"
    )
    _attach_lineage_scope(tokenization_path, scope)
    fold_table = tmp_path / f"{lineage}_all_healthy.tsv".lower()
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "donor_id": donor,
                "outer_role": "reference",
                "reference_partition": "adaptation",
                "eligible_for_reference_fitting": True,
            }
            for donor in (present, *missing)
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    manifest = build_fold_input_from_tokenization(
        tokenization_manifest_path=tokenization_path,
        fold_table_path=fold_table,
        output_path=tmp_path / f"{lineage}_all_healthy_fold.json".lower(),
        fold_id="all_healthy",
        held_out_dataset=None,
        lineage=lineage,
        partition_column="reference_partition",
        reference_design="all_healthy",
    )
    assert manifest["adaptation_biological_unit_ids"] == [physical_id]
    assert manifest["validation_biological_unit_ids"] == []
    assert manifest["query_biological_unit_ids"] == []
    assert manifest["lineage_donor_scope_validation"][
        "global_fold_biological_unit_ids_excluded_by_original_role"
    ]["adaptation"] == sorted(missing_ids)


def test_all_healthy_fold_input_requires_no_heldout_or_query_donors(
    tmp_path: Path,
) -> None:
    _tokenize(
        tmp_path,
        role="adaptation",
        dataset="train",
        donors=("d1", "d2"),
        name="all_healthy_tokens",
    )
    fold_table = tmp_path / "all_healthy.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "donor_id": donor,
                "sample_id": "s1",
                "reference_partition": "adaptation",
                "eligible_for_reference_fitting": True,
            }
            for donor in ("d1", "d2")
        ]
    ).to_csv(fold_table, sep="\t", index=False)
    output = tmp_path / "all_healthy_fold_input.json"
    manifest = build_fold_input_from_tokenization(
        tokenization_manifest_path=(
            tmp_path / "all_healthy_tokens" / "tokenization_manifest.json"
        ),
        fold_table_path=fold_table,
        output_path=output,
        fold_id="all_healthy",
        held_out_dataset=None,
        lineage="B cells",
        partition_column="reference_partition",
        reference_design="all_healthy",
    )
    assert manifest["reference_design"] == "all_healthy"
    assert manifest["held_out_dataset"] is None
    assert manifest["query_biological_unit_ids"] == []
    assert manifest["adaptation_biological_unit_ids"] == ["train::d1", "train::d2"]
    assert load_fold_input_manifest(output)["reference_design"] == "all_healthy"
    call, invocation = build_training_call(
        TripsoTrainingSpec(
            fold_input_manifest_path=output,
            output_dir=tmp_path / "final_model",
            model_type="Base",
            seed=42,
            parameters={"fm_encoder_pkg": "from_scratch"},
        )
    )
    assert call["dataset_path"].endswith("tokenized.dataset")
    assert invocation["reference_design"] == "all_healthy"
    assert invocation["held_out_dataset"] is None


def test_query_projection_collator_preserves_string_identifiers() -> None:
    rows = [
        {
            "tk": {
                "biological_unit_id": "held::q1",
                "observation_id": "held::q1::s1",
                "fine_type": "memory B",
            }
        }
    ]

    class FakeVendorDataModule:
        def __init__(self) -> None:
            self.return_tuple = False
            self.metadata = ["biological_unit_id", "observation_id", "fine_type"]
            gdata = [rows[0]["tk"]]
            self.dataset = SimpleNamespace(tk_dataset=SimpleNamespace(gdata=gdata))

        def setup(self, stage=None) -> None:
            self.test_dataset = "vendor_subset"

        def custom_collate(self, batch):
            output = {}
            for column in self.metadata:
                values = [item["tk"][column] for item in batch]
                if column.endswith("_id"):
                    values = [int(value) for value in values]
                output[column] = values
            return output

    query_class = make_query_only_datamodule_class(FakeVendorDataModule)
    module = query_class()
    module.setup("test")
    batch = module.custom_collate(rows)
    assert module.test_dataset is module.dataset
    assert batch["biological_unit_id"] == ["held::q1"]
    assert batch["observation_id"] == ["held::q1::s1"]
    assert batch["fine_type"] == ["memory B"]
