from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from immune_health.data.ids import add_stable_identifiers
from immune_health.io import read_parquet_artifact
from immune_health.smoke import IDENTIFIER_COLUMNS, run_real_baseline_smoke


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    genes = [f"ENSG{index:011d}" for index in range(1, 13)]
    rng = np.random.default_rng(17)
    records: list[dict[str, object]] = []
    matrices: list[np.ndarray] = []
    datasets = ["dataset_a", "dataset_b", "dataset_c", "dataset_d", "dataset_e"]
    for dataset_index, dataset in enumerate(datasets):
        for donor_index in range(2):
            donor = f"donor_{donor_index}"
            for cell_index in range(6):
                fine_type = "B_naive" if cell_index < 3 else "B_memory"
                records.append(
                    {
                        "dataset": dataset,
                        "donor_id": donor,
                        "sample_id": f"sample_{donor_index}",
                        "age": 25 + dataset_index * 10 + donor_index,
                        "sex": "female" if donor_index == 0 else "male",
                        "lineage": "B cells",
                        "ctype_low": fine_type,
                        "ctype_low_conf": 0.95,
                    }
                )
                counts = rng.poisson(2, size=len(genes)).astype(float)
                counts[:3] += dataset_index + 1
                matrices.append(counts)
    obs = add_stable_identifiers(pd.DataFrame.from_records(records))
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.vstack(matrices)),
        obs=obs,
        var=pd.DataFrame({"unified_ensembl": genes}, index=genes),
    )
    merged_root = tmp_path / "merged"
    h5ad_dir = merged_root / "B_cells"
    h5ad_dir.mkdir(parents=True)
    h5ad_path = h5ad_dir / "merged.h5ad"
    adata.write_h5ad(h5ad_path)

    lineage_entries = {
        name: {"role": "primary", "path": "B_cells/merged.h5ad"}
        for name in ("B cells", "NK_ILC", "Monocytes", "CD4_like", "CD8_like")
    }
    data_config = {
        "paths": {"merged_lineage_root": str(merged_root)},
        "identifiers": {
            "biological_unit_id": "dataset::donor_id",
            "source_observation_id": "dataset::sample_id",
            "observation_id": "dataset::donor_id::sample_id",
        },
        "metadata_fields": {},
        "counts": {},
        "genes": {},
        "datasets": [{"name": name} for name in datasets],
        "lineages": lineage_entries,
    }
    data_config_path = tmp_path / "reference.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config))

    gmt_path = tmp_path / "curated.gmt"
    gmt_path.write_text(
        "CURATED_A\tfixture_curated\t"
        + "\t".join(genes[:5])
        + "\nCURATED_B\tfixture_curated\t"
        + "\t".join(genes[5:10])
        + "\n"
    )
    gp_config_path = tmp_path / "gene_programs.yaml"
    gp_config_path.write_text(
        yaml.safe_dump(
            {
                "resources": {
                    "existing_curated_gmt": {
                        "path": str(gmt_path),
                        "required": True,
                    }
                },
                "filters": {"minimum_mapped_genes": 2},
            }
        )
    )
    return data_config_path, gp_config_path


def test_real_baseline_smoke_is_read_only_and_keeps_stable_ids(
    tmp_path: Path,
) -> None:
    data_config_path, gp_config_path = _write_fixture(tmp_path)
    h5ad_path = tmp_path / "merged/B_cells/merged.h5ad"
    before_mtime = h5ad_path.stat().st_mtime_ns
    output_dir = tmp_path / "output"

    summary = run_real_baseline_smoke(
        output_dir,
        data_config_path=data_config_path,
        gene_program_config_path=gp_config_path,
        max_donors_per_dataset=1,
        max_cells_per_donor=4,
        n_pca_components=2,
        maximum_gene_programs=2,
        seed=9,
    )

    assert summary["status"] == "complete"
    assert summary["heldout_dataset"] == "dataset_e"
    assert summary["training_datasets"] == [
        "dataset_a",
        "dataset_b",
        "dataset_c",
        "dataset_d",
    ]
    assert summary["pca_frozen_across_query_projection"] is True
    assert summary["tripso_executed"] is False
    assert summary["production_fold_vocabulary_used"] is False
    assert h5ad_path.stat().st_mtime_ns == before_mtime

    for path in summary["outputs"].values():
        table, manifest = read_parquet_artifact(Path(path))
        assert set(IDENTIFIER_COLUMNS).issubset(table.columns)
        assert manifest["provenance"]["tripso_executed"] is False
        assert table["observation_id"].str.count("::").eq(2).all()

    projections = pd.read_parquet(summary["outputs"]["pca_projection"])
    assert set(projections.loc[projections["outer_role"] == "query", "dataset"]) == {
        "dataset_e"
    }
    assert "dataset_e" not in set(
        projections.loc[projections["outer_role"] == "reference", "dataset"]
    )
    gp_scores = pd.read_parquet(summary["outputs"]["simple_gp_scores"])
    assert gp_scores["gp_id"].nunique() == 2
    assert gp_scores["gp_score"].notna().all()

    done = json.loads((output_dir / "real_baseline_smoke.done.json").read_text())
    assert done["status"] == "complete"
    assert done["tripso_executed"] is False
