"""Bounded, read-only smoke checks on audited merged PBMC data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.baselines import (
    TrainOnlyPCA,
    build_composition_table,
    build_pseudobulk,
    score_gene_programs,
)
from immune_health.config import (
    load_path_config,
    resolve_config_path,
    validate_reference_data_config,
)
from immune_health.data.contracts import validate_cell_metadata, validate_raw_counts
from immune_health.data.h5ad import load_small_lineage_subset, validate_merged_h5ad
from immune_health.gene_programs import GeneProgram, load_gene_programs
from immune_health.io import artifact_manifest_path, write_parquet_artifact
from immune_health.provenance import (
    atomic_write_json,
    completion_marker,
    sha256_file,
    stable_hash,
)

IDENTIFIER_COLUMNS = (
    "dataset",
    "donor_id",
    "biological_unit_id",
    "sample_id",
    "source_observation_id",
    "observation_id",
)


def _file_identity(path: Path) -> dict[str, Any]:
    """Return an auditable identity without scanning a multi-gigabyte input."""

    stat = path.stat()
    identity = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    return {
        **identity,
        "identity_hash_type": "sha256(path,size_bytes,mtime_ns)",
        "identity_sha256": stable_hash(identity),
    }


def _sparse_subset_hash(
    counts: sparse.csr_matrix,
    gene_ids: Sequence[str],
    metadata: pd.DataFrame,
) -> str:
    """Hash the selected sparse content and stable IDs, not the complete H5AD."""

    digest = hashlib.sha256()
    matrix = counts.tocsr(copy=False)
    for name, values in (
        ("data", matrix.data),
        ("indices", matrix.indices),
        ("indptr", matrix.indptr),
        ("shape", np.asarray(matrix.shape, dtype=np.int64)),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    digest.update("\n".join(map(str, gene_ids)).encode())
    digest.update(
        metadata.loc[:, IDENTIFIER_COLUMNS]
        .astype(str)
        .to_csv(index=False, lineterminator="\n")
        .encode()
    )
    return digest.hexdigest()


def _pca_hash(model: TrainOnlyPCA) -> str:
    digest = hashlib.sha256()
    for name in (
        "components_",
        "mean_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
    ):
        array = np.ascontiguousarray(getattr(model.model_, name))
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_identifier_columns(frame: pd.DataFrame, table_name: str) -> None:
    missing = sorted(set(IDENTIFIER_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} dropped approved identifiers: {missing}")


def _configured_lineage_path(config: Mapping[str, Any], lineage_name: str) -> Path:
    lineages = config["lineages"]
    if lineage_name not in lineages:
        raise ValueError(f"Lineage {lineage_name!r} is not configured")
    root = resolve_config_path(config, config["paths"]["merged_lineage_root"])
    return (root / str(lineages[lineage_name]["path"])).resolve()


def _select_gene_programs(
    programs: Sequence[GeneProgram],
    gene_ids: Sequence[str],
    *,
    maximum_programs: int,
    minimum_mapped_genes: int,
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    if maximum_programs < 1:
        raise ValueError("maximum_programs must be at least one")
    available = set(map(str, gene_ids))
    coverage = []
    program_by_id = {program.program_id: program for program in programs}
    for program in programs:
        mapped = tuple(gene for gene in program.genes if gene in available)
        coverage.append(
            {
                "program_id": program.program_id,
                "source": program.source,
                "n_program_genes": len(program.genes),
                "n_mapped_genes": len(mapped),
                "gene_coverage": len(mapped) / len(program.genes),
            }
        )
    eligible = [
        row for row in coverage if row["n_mapped_genes"] >= minimum_mapped_genes
    ]
    eligible.sort(key=lambda row: (-row["n_mapped_genes"], row["program_id"]))
    selected_rows = eligible[:maximum_programs]
    if not selected_rows:
        raise ValueError(
            "No configured curated gene program has at least "
            f"{minimum_mapped_genes} genes in the merged Ensembl vocabulary"
        )
    selected = {
        row["program_id"]: tuple(program_by_id[row["program_id"]].genes)
        for row in selected_rows
    }
    return selected, selected_rows


def run_real_baseline_smoke(
    output_dir: Path,
    *,
    data_config_path: Path = Path("configs/data/reference.yaml"),
    gene_program_config_path: Path = Path("configs/gene_programs/default.yaml"),
    lineage_name: str = "B cells",
    heldout_dataset: str | None = None,
    max_donors_per_dataset: int = 2,
    max_cells_per_donor: int = 50,
    n_pca_components: int = 3,
    maximum_gene_programs: int = 3,
    minimum_mapped_genes: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a small merged-B-cell baseline check with a frozen query projection.

    This function deliberately does not train or invoke TRIPSO. It also does
    not construct a production fold-specific vocabulary: the audited merged
    H5AD vocabulary is used only for this bounded integration smoke.
    """

    output_dir = Path(output_dir).resolve()
    data_config_path = Path(data_config_path).resolve()
    gene_program_config_path = Path(gene_program_config_path).resolve()
    data_config = load_path_config(data_config_path)
    gp_config = load_path_config(gene_program_config_path)
    validate_reference_data_config(data_config)

    dataset_names = tuple(str(item["name"]) for item in data_config["datasets"])
    if len(dataset_names) != 5 or len(set(dataset_names)) != 5:
        raise ValueError(
            "The real baseline smoke requires exactly five distinct reference datasets"
        )
    query_dataset = heldout_dataset or dataset_names[-1]
    if query_dataset not in dataset_names:
        raise ValueError(f"Held-out dataset is not configured: {query_dataset}")
    training_datasets = tuple(name for name in dataset_names if name != query_dataset)
    if len(training_datasets) != 4:
        raise AssertionError("Expected four training datasets and one query dataset")

    h5ad_path = _configured_lineage_path(data_config, lineage_name)
    if not h5ad_path.is_file():
        raise FileNotFoundError(
            f"Configured merged lineage H5AD is missing: {h5ad_path}"
        )
    source_identity_before = _file_identity(h5ad_path)
    structure = validate_merged_h5ad(h5ad_path)

    resource_entry = gp_config.get("resources", {}).get("existing_curated_gmt")
    if not isinstance(resource_entry, Mapping) or not resource_entry.get("path"):
        raise ValueError(
            "Gene-program config lacks resources.existing_curated_gmt.path"
        )
    gp_resource_path = resolve_config_path(gp_config, resource_entry["path"])
    programs = load_gene_programs(gp_resource_path, format="gmt")
    configured_minimum = int(
        gp_config.get("filters", {}).get("minimum_mapped_genes", 1)
    )
    minimum_genes = (
        configured_minimum
        if minimum_mapped_genes is None
        else int(minimum_mapped_genes)
    )

    subset = load_small_lineage_subset(
        h5ad_path,
        datasets=dataset_names,
        max_donors_per_dataset=max_donors_per_dataset,
        max_cells_per_donor=max_cells_per_donor,
        seed=seed,
    )
    cells, metadata_report = validate_cell_metadata(
        subset.obs,
        allowed_datasets=dataset_names,
        allowed_lineages=[lineage_name],
    )
    count_report = validate_raw_counts(subset.counts)
    cells = cells.rename(columns={"ctype_low": "fine_type"})
    _require_identifier_columns(cells, "selected_cells")

    composition = build_composition_table(cells, fine_type_col="fine_type")
    pseudobulk = build_pseudobulk(
        subset.counts,
        cells,
        subset.gene_ids,
        fine_type_col="fine_type",
        min_cells=1,
    )
    _require_identifier_columns(composition, "fine_type_composition")
    _require_identifier_columns(pseudobulk.metadata, "pseudobulk_metadata")

    training_mask = (
        pseudobulk.metadata["dataset"].astype(str).isin(training_datasets).to_numpy()
    )
    query_mask = pseudobulk.metadata["dataset"].astype(str).eq(query_dataset).to_numpy()
    invalid_split = (
        not training_mask.any()
        or not query_mask.any()
        or (training_mask & query_mask).any()
    )
    if invalid_split:
        raise ValueError(
            "The selected subset does not contain a valid 4+1 dataset split"
        )
    training_units = pseudobulk.metadata.loc[
        training_mask, "biological_unit_id"
    ].astype(str)
    query_units = pseudobulk.metadata.loc[query_mask, "biological_unit_id"].astype(str)
    if set(training_units) & set(query_units):
        raise ValueError("Held-out biological units overlap training units")

    maximum_components = min(int(training_mask.sum()), int(pseudobulk.counts.shape[1]))
    components = min(int(n_pca_components), maximum_components)
    if components < 1:
        raise ValueError("PCA requires at least one training pseudobulk")
    pca = TrainOnlyPCA(n_components=components, random_state=seed)
    training_coordinates = pca.fit_transform(
        pseudobulk.counts[training_mask],
        feature_ids=pseudobulk.gene_ids,
        training_biological_units=training_units,
    )
    pca_hash_before_query = _pca_hash(pca)
    query_coordinates = pca.transform(
        pseudobulk.counts[query_mask],
        feature_ids=pseudobulk.gene_ids,
        query_biological_units=query_units,
    )
    pca_hash_after_query = _pca_hash(pca)
    if pca_hash_before_query != pca_hash_after_query:
        raise RuntimeError("Frozen training PCA changed during query projection")

    coordinate_columns = [f"pca_{index + 1}" for index in range(components)]
    training_projection = pseudobulk.metadata.loc[training_mask].reset_index(drop=True)
    training_projection = pd.concat(
        [
            training_projection,
            pd.DataFrame(training_coordinates, columns=coordinate_columns),
        ],
        axis=1,
    )
    training_projection["outer_role"] = "reference"
    query_projection = pseudobulk.metadata.loc[query_mask].reset_index(drop=True)
    query_projection = pd.concat(
        [query_projection, pd.DataFrame(query_coordinates, columns=coordinate_columns)],
        axis=1,
    )
    query_projection["outer_role"] = "query"
    projections = pd.concat([training_projection, query_projection], ignore_index=True)
    _require_identifier_columns(projections, "pca_projection")

    selected_programs, program_coverage = _select_gene_programs(
        programs,
        pseudobulk.gene_ids,
        maximum_programs=maximum_gene_programs,
        minimum_mapped_genes=minimum_genes,
    )
    raw_gp_scores = score_gene_programs(
        pseudobulk.counts,
        pseudobulk.gene_ids,
        selected_programs,
        method="mean_log_cpm",
        minimum_genes=minimum_genes,
    )
    metadata_with_index = pseudobulk.metadata.reset_index(names="summary_index")
    gp_scores = raw_gp_scores.merge(
        metadata_with_index,
        on="summary_index",
        how="left",
        validate="many_to_one",
    )
    _require_identifier_columns(gp_scores, "simple_gp_scores")
    if gp_scores["gp_score"].isna().any():
        raise RuntimeError("A selected curated gene program produced missing scores")

    source_identity_after = _file_identity(h5ad_path)
    if source_identity_after != source_identity_before:
        raise RuntimeError("Input H5AD identity changed during a read-only smoke run")
    subset_hash = _sparse_subset_hash(subset.counts, subset.gene_ids, cells)
    parameters = {
        "lineage_name": lineage_name,
        "heldout_dataset": query_dataset,
        "training_datasets": list(training_datasets),
        "max_donors_per_dataset": max_donors_per_dataset,
        "max_cells_per_donor": max_cells_per_donor,
        "n_pca_components": components,
        "maximum_gene_programs": maximum_gene_programs,
        "minimum_mapped_genes": minimum_genes,
        "seed": seed,
    }
    provenance = {
        "run_type": "real_data_baseline_smoke",
        "scope": "bounded integration smoke; not production training",
        "tripso_executed": False,
        "production_fold_vocabulary_used": False,
        "identifier_contract": {
            "biological_unit_id": "dataset::donor_id",
            "source_observation_id": "dataset::sample_id",
            "observation_id": "dataset::donor_id::sample_id",
        },
        "parameters": parameters,
        "configuration": {
            "data_config_path": str(data_config_path),
            "data_config_sha256": sha256_file(data_config_path),
            "gene_program_config_path": str(gene_program_config_path),
            "gene_program_config_sha256": sha256_file(gene_program_config_path),
            "resolved_configuration_sha256": stable_hash(
                {"data": data_config, "gene_programs": gp_config, **parameters}
            ),
        },
        "inputs": {
            "merged_h5ad": source_identity_before,
            "selected_sparse_subset_sha256": subset_hash,
            "gene_program_resource_path": str(gp_resource_path),
            "gene_program_resource_sha256": sha256_file(gp_resource_path),
        },
        "pca": {
            "fit_datasets": list(training_datasets),
            "projected_dataset": query_dataset,
            "model_sha256_before_query": pca_hash_before_query,
            "model_sha256_after_query": pca_hash_after_query,
            "query_refit": False,
        },
        "gene_program_score_definition": (
            "arithmetic mean across mapped program genes of per-gene log1p(CPM), "
            "using donor-observation-by-fine-type raw-count pseudobulks"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "selected_cells": cells,
        "fine_type_composition": composition,
        "pseudobulk_metadata": pseudobulk.metadata,
        "pca_projection": projections,
        "simple_gp_scores": gp_scores,
    }
    artifact_paths = {name: output_dir / f"{name}.parquet" for name in tables}
    summary_path = output_dir / "smoke_summary.json"
    done_path = output_dir / "real_baseline_smoke.done.json"
    all_expected = [
        summary_path,
        done_path,
        *artifact_paths.values(),
        *(artifact_manifest_path(path) for path in artifact_paths.values()),
    ]
    existing = [path for path in all_expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing smoke artifacts: "
            + ", ".join(str(path) for path in existing[:3])
        )
    for name, frame in tables.items():
        _require_identifier_columns(frame, name)
        write_parquet_artifact(
            frame,
            artifact_paths[name],
            schema_name=f"real_baseline_smoke.{name}",
            schema_version="1.0",
            provenance=provenance,
            overwrite=overwrite,
        )

    summary = {
        "status": "complete",
        "run_type": "real_data_baseline_smoke",
        "scope": "baseline smoke only; not TRIPSO and not a production-vocabulary run",
        "lineage": lineage_name,
        "heldout_dataset": query_dataset,
        "training_datasets": list(training_datasets),
        "n_cells": len(cells),
        "n_biological_units": int(cells["biological_unit_id"].nunique()),
        "n_observations": int(cells["observation_id"].nunique()),
        "n_fine_types": int(cells["fine_type"].nunique()),
        "pseudobulk_shape": list(pseudobulk.counts.shape),
        "count_validation": vars(count_report),
        "metadata_validation": metadata_report,
        "h5ad_structure": structure,
        "selected_gene_programs": program_coverage,
        "pca_frozen_across_query_projection": (
            pca_hash_before_query == pca_hash_after_query
        ),
        "tripso_executed": False,
        "production_fold_vocabulary_used": False,
        "provenance_hash": stable_hash(provenance),
        "provenance": provenance,
        "outputs": {name: str(path) for name, path in artifact_paths.items()},
    }
    atomic_write_json(summary_path, summary)
    completion_marker(
        done_path,
        stage="real_data_baseline_smoke",
        outputs=[summary_path, *artifact_paths.values()],
        configuration=provenance,
        repo_root=Path(__file__).resolve().parents[2],
        extra={
            "tripso_executed": False,
            "production_fold_vocabulary_used": False,
        },
    )
    return summary


__all__ = ["run_real_baseline_smoke"]
