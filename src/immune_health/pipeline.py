"""Small functioning vertical slices shared by CLI and smoke tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.aggregation import aggregate_fine_type_distributions
from immune_health.baselines import TrainOnlyPCA, build_pseudobulk, score_gene_programs
from immune_health.data.synthetic import make_synthetic_pbmc
from immune_health.evaluation import evaluate_lodo
from immune_health.healthy_reference import HealthyTrajectory
from immune_health.io import write_parquet_artifact
from immune_health.provenance import atomic_write_json, completion_marker
from immune_health.splits import (
    build_global_donor_manifest,
    build_lodo_tables,
    write_lodo_manifests,
)
from immune_health.tripso_adapter import run_mock_projection_smoke


def _model_hash(model: HealthyTrajectory) -> str:
    digest = hashlib.sha256()
    for name in ("coefficients_", "residual_covariance_", "age_direction_"):
        array = np.asarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def score_frozen_query(
    model: HealthyTrajectory,
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Score unseen donor observations without fitting a query offset."""
    required = {
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
        "age",
        "sex",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Frozen query metadata is missing columns: {missing}")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(metadata):
        raise ValueError("Frozen query features and metadata do not align")
    overlap = set(metadata["biological_unit_id"].astype(str)).intersection(
        model.training_biological_units_
    )
    if overlap:
        raise ValueError(
            f"Query donors entered reference fitting: {sorted(overlap)[:3]}"
        )
    before = _model_hash(model)
    records: list[dict[str, Any]] = []
    for position, row in metadata.reset_index(drop=True).iterrows():
        # dataset=None is deliberate: an unseen query offset must never be fitted.
        score = model.score(
            matrix[position], float(row["age"]), str(row["sex"]), dataset=None
        )
        records.append({**row.to_dict(), **score})
    if _model_hash(model) != before:
        raise RuntimeError("Frozen healthy reference changed during query scoring")
    return pd.DataFrame.from_records(records)


def _collapse_pseudobulk_observations(
    counts: sparse.csr_matrix, metadata: pd.DataFrame
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    codes, observations = pd.factorize(metadata["observation_id"], sort=False)
    aggregator = sparse.csr_matrix(
        (np.ones(len(metadata)), (codes, np.arange(len(metadata)))),
        shape=(len(observations), len(metadata)),
    )
    collapsed = (aggregator @ counts).tocsr()
    rows = []
    stable = [
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
        "age",
        "sex",
        "lineage",
    ]
    for code in range(len(observations)):
        part = metadata.loc[codes == code]
        for column in stable:
            if part[column].nunique(dropna=False) != 1:
                raise ValueError(f"{column} varies within a biological observation")
        row = part.iloc[0][stable].to_dict()
        row["n_fine_types"] = int(part["fine_type"].nunique())
        row["n_cells"] = int(part["n_cells"].sum())
        row["library_size"] = float(collapsed[code].sum())
        rows.append(row)
    return collapsed, pd.DataFrame(rows)


def run_synthetic_lodo_smoke(output_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    """Run counts→LODO→baseline→mock adapter→frozen query→report."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adata, truth = make_synthetic_pbmc(seed=seed)
    obs = adata.obs.reset_index().rename(columns={"ctype_low": "fine_type"})

    global_manifest = build_global_donor_manifest(
        obs,
        datasets=None,
        n_inner_folds=3,
        seed=seed,
    )
    split_paths = write_lodo_manifests(
        global_manifest,
        output_dir / "splits",
        datasets=sorted(obs["dataset"].astype(str).unique()),
    )
    folds = build_lodo_tables(global_manifest)
    fold = folds[truth.held_out_dataset]
    query_units = set(
        fold.loc[fold["outer_role"] == "query", "biological_unit_id"].astype(str)
    )
    reference_units = set(
        fold.loc[fold["outer_role"] == "reference", "biological_unit_id"].astype(str)
    )
    if query_units & reference_units:
        raise AssertionError("Synthetic query and reference donors overlap")

    b_mask = obs["lineage"].astype(str).eq("B cells").to_numpy()
    b_obs = obs.loc[b_mask].reset_index(drop=True)
    b_counts = adata.X[b_mask].tocsr()
    pseudobulk = build_pseudobulk(
        b_counts,
        b_obs,
        adata.var_names,
        fine_type_col="fine_type",
        min_cells=1,
    )
    observation_counts, observation_metadata = _collapse_pseudobulk_observations(
        pseudobulk.counts, pseudobulk.metadata
    )
    query_mask = (
        observation_metadata["biological_unit_id"].astype(str).isin(query_units)
    ).to_numpy()
    training_mask = (
        observation_metadata["biological_unit_id"].astype(str).isin(reference_units)
    ).to_numpy()
    if (
        not query_mask.any()
        or not training_mask.any()
        or (query_mask & training_mask).any()
    ):
        raise AssertionError("Synthetic outer fold is invalid")

    pca = TrainOnlyPCA(n_components=3, random_state=seed)
    training_coordinates = pca.fit_transform(
        observation_counts[training_mask],
        feature_ids=pseudobulk.gene_ids,
        training_biological_units=observation_metadata.loc[
            training_mask, "biological_unit_id"
        ],
    )
    query_coordinates = pca.transform(
        observation_counts[query_mask],
        feature_ids=pseudobulk.gene_ids,
        query_biological_units=observation_metadata.loc[
            query_mask, "biological_unit_id"
        ],
    )
    training_metadata = observation_metadata.loc[training_mask].reset_index(drop=True)
    query_metadata = observation_metadata.loc[query_mask].reset_index(drop=True)
    trajectory = HealthyTrajectory(n_spline_knots=1, age_grid_size=51).fit(
        training_coordinates,
        training_metadata["age"],
        training_metadata["sex"],
        training_metadata["biological_unit_id"],
        datasets=training_metadata["dataset"],
    )
    predictions = score_frozen_query(trajectory, query_coordinates, query_metadata)
    predictions["fold_id"] = f"lodo_{truth.held_out_dataset}"
    predictions["gp_id"] = "PSEUDOBULK_PCA"
    metrics = evaluate_lodo(predictions)

    gp_scores = score_gene_programs(
        observation_counts,
        pseudobulk.gene_ids,
        {"SYNTHETIC_AGE_UP": truth.age_up_genes},
        method="mean_log_cpm",
        minimum_genes=2,
    )
    cell_embeddings = np.log1p(b_counts[:, :3].toarray())
    aggregation = aggregate_fine_type_distributions(
        cell_embeddings,
        b_obs,
        gp_id="SYNTHETIC_EMBEDDING",
        fine_type_col="fine_type",
        min_state_cells=5,
        annotation_confidence_col="ctype_low_conf",
        provenance={"fold_id": f"lodo_{truth.held_out_dataset}", "seed": seed},
    )
    mock_tripso = run_mock_projection_smoke()

    sparse.save_npz(output_dir / "pseudobulk_counts.npz", observation_counts)
    provenance = {
        "seed": seed,
        "heldout_dataset": truth.held_out_dataset,
        "query_path": "same score_frozen_query interface used for external cohorts",
        "training_datasets": sorted(set(training_metadata["dataset"].astype(str))),
    }
    artifact_paths = []
    for frame, name, schema in (
        (observation_metadata, "pseudobulk_metadata.parquet", "pseudobulk_observation"),
        (gp_scores, "simple_gp_scores.parquet", "simple_gp_score"),
        (
            aggregation.table,
            "fine_type_distributions.parquet",
            "fine_type_distribution",
        ),
        (predictions, "query_scores.parquet", "query_score"),
        (metrics, "lodo_metrics.parquet", "lodo_metric"),
    ):
        path = output_dir / name
        write_parquet_artifact(
            frame,
            path,
            schema_name=schema,
            schema_version="1.0",
            provenance=provenance,
            overwrite=True,
        )
        artifact_paths.append(path)
    summary = {
        "status": "complete",
        "seed": seed,
        "heldout_dataset": truth.held_out_dataset,
        "n_cells": int(adata.n_obs),
        "n_biological_units": int(obs["biological_unit_id"].nunique()),
        "n_training_units": len(reference_units),
        "n_query_units": len(query_units),
        "pseudobulk_shape": list(observation_counts.shape),
        "mock_tripso": mock_tripso,
        "split_outputs": {key: str(path) for key, path in split_paths.items()},
    }
    summary_path = output_dir / "smoke_summary.json"
    atomic_write_json(summary_path, summary)
    completion_marker(
        output_dir / "synthetic_lodo_smoke.done.json",
        stage="synthetic_lodo_smoke",
        outputs=[summary_path, *artifact_paths],
        configuration=provenance,
        repo_root=Path(__file__).resolve().parents[2],
    )
    return summary
