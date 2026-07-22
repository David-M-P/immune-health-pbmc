from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.config import (
    load_path_config,
    resolve_config_path,
    validate_reference_data_config,
)
from immune_health.data.contracts import (
    gene_coverage,
    require_gene_coverage,
    validate_cell_metadata,
    validate_raw_counts,
)
from immune_health.data.h5ad import load_small_lineage_subset, validate_merged_h5ad
from immune_health.data.ids import add_stable_identifiers, validate_identifier_contract
from immune_health.data.synthetic import make_synthetic_pbmc
from immune_health.data.vocabulary import build_training_vocabulary

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_approved_observation_id_is_collision_safe_for_pooled_samples() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["onek1k", "onek1k"],
            "donor_id": ["donor_a", "donor_b"],
            "sample_id": ["pool_1", "pool_1"],
        }
    )
    result = add_stable_identifiers(frame)
    report = validate_identifier_contract(result)

    assert result["source_observation_id"].nunique() == 1
    assert result["observation_id"].nunique() == 2
    assert report["n_source_observations_shared_across_donors"] == 1
    assert result.loc[0, "observation_id"] == "onek1k::donor_a::pool_1"


def test_reference_config_declares_audited_lineages_and_identifiers() -> None:
    config = load_path_config(REPOSITORY_ROOT / "configs/data/reference.yaml")
    validate_reference_data_config(config)
    merged_root = resolve_config_path(config, config["paths"]["merged_lineage_root"])
    assert merged_root.name == "merged"
    assert config["identifiers"]["observation_id"] == "dataset::donor_id::sample_id"


def test_sparse_raw_count_validation_never_densifies() -> None:
    counts = sparse.csr_matrix(np.asarray([[0.0, 2.0], [3.0, 0.0]]))
    before = counts.copy()
    report = validate_raw_counts(counts)
    assert report.sparse
    assert report.non_integer_like_values == 0
    assert (counts != before).nnz == 0


def test_query_gene_coverage_requires_explicit_override() -> None:
    report = gene_coverage(["A", "B"], ["A", "B", "C", "D"])
    assert report["coverage"] == 0.5
    try:
        require_gene_coverage(["A"], ["A", "B"], 0.8)
    except ValueError as error:
        assert "below safety threshold" in str(error)
    else:
        raise AssertionError("Low query coverage was not rejected")


def test_synthetic_anndata_has_sparse_counts_and_repeated_observations() -> None:
    adata, truth = make_synthetic_pbmc(
        donors_per_dataset=2, cells_per_donor_lineage=8, seed=7
    )
    assert sparse.isspmatrix_csr(adata.X)
    assert truth.held_out_dataset in set(adata.obs["dataset"])
    assert adata.obs["biological_unit_id"].nunique() == 10
    repeated = adata.obs.groupby("biological_unit_id")["observation_id"].nunique()
    assert (repeated > 1).sum() == 5
    validated, report = validate_cell_metadata(adata.obs.reset_index(drop=True))
    assert len(validated) == adata.n_obs
    assert report["n_biological_units"] == 10


def test_read_only_sparse_h5ad_subset(tmp_path: Path) -> None:
    adata, _ = make_synthetic_pbmc(
        donors_per_dataset=2, cells_per_donor_lineage=8, seed=11
    )
    adata = adata[adata.obs["lineage"] == "B cells"].copy()
    path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(path)
    before_mtime = path.stat().st_mtime_ns
    structure = validate_merged_h5ad(path)
    subset = load_small_lineage_subset(
        path, max_donors_per_dataset=1, max_cells_per_donor=5, seed=3
    )
    assert structure["read_only"] is True
    assert sparse.isspmatrix_csr(subset.counts)
    assert subset.counts.shape[0] == len(subset.obs)
    assert path.stat().st_mtime_ns == before_mtime


def test_training_vocabulary_never_opens_heldout_source(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sources = []
    for dataset, genes in (
        ("train_a", ["ENSG1", "ENSG2"]),
        ("train_b", ["ENSG2", "ENSG1"]),
    ):
        source_id = f"{dataset}_source"
        directory = data_root / "lineages" / source_id
        directory.mkdir(parents=True)
        path = directory / "B_cells.h5ad"
        tiny = ad.AnnData(
            X=sparse.csr_matrix(np.ones((1, len(genes)))),
            var=pd.DataFrame({"unified_ensembl": genes}, index=genes),
        )
        tiny.write_h5ad(path)
        sources.append(
            {
                "dataset": dataset,
                "source_dataset_id": source_id,
                "lineage_filename": path.name,
            }
        )
    heldout_dir = data_root / "lineages" / "heldout_source"
    heldout_dir.mkdir(parents=True)
    heldout_path = heldout_dir / "B_cells.h5ad"
    heldout_path.write_text("deliberately not an HDF5 file")
    sources.append(
        {
            "dataset": "heldout",
            "source_dataset_id": "heldout_source",
            "lineage_filename": heldout_path.name,
        }
    )
    merge_manifest = tmp_path / "merge_manifest.json"
    merge_manifest.write_text(json.dumps({"sources": sources}))

    result = build_training_vocabulary(merge_manifest, data_root, "heldout")
    assert result.genes == ("ENSG1", "ENSG2")
    assert str(heldout_path.resolve()) in result.excluded_sources
    assert str(heldout_path.resolve()) not in result.opened_sources
