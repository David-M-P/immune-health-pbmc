from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immune_health.cli.main import main
from immune_health.gene_programs.tripso_selection import (
    TripsoGPSelectionConfig,
    select_transferable_tripso_gps,
    validate_tripso_gp_selection_manifest,
    write_tripso_gp_selection,
)
from immune_health.healthy_reference.endpoint import validate_endpoint_inputs
from immune_health.provenance import sha256_file, stable_hash
from immune_health.tripso_adapter.contracts import (
    canonical_json_hash,
    prepare_fold_input_manifest,
)
from immune_health.tripso_adapter.provenance import build_model_artifact_manifest

REPO_ROOT = Path(__file__).parents[1]
COHORTS = ("cohort_a", "cohort_b", "cohort_c", "cohort_d")
SEEDS = (11, 12)
PROGRAMS = ("stable_age", "inconsistent_age")


def _ordered_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_fold_and_models(tmp_path: Path) -> dict[int, Path]:
    resources = tmp_path / "resources"
    tokenized = resources / "tokenized"
    tokenized.mkdir(parents=True)
    gp_library = resources / "gp.csv"
    gp_library.write_text(
        "stable_age,inconsistent_age\nGENE1,GENE2\n", encoding="utf-8"
    )
    vocabulary = resources / "vocabulary.txt"
    vocabulary.write_text("ENSG000001\nENSG000002\n", encoding="utf-8")
    candidates = resources / "projection_gp_candidates.json"
    candidate_payload = {
        "schema_version": "immune-health-projection-gp-candidates/v1",
        "selection_level": "donor_lineage_pseudobulk",
        "query_data_consulted": False,
        "program_ids": list(PROGRAMS),
        "program_ids_ordered_sha256": canonical_json_hash(list(PROGRAMS)),
        "binding": {"gpdb_sha256": sha256_file(gp_library)},
    }
    candidate_payload["manifest_content_sha256"] = canonical_json_hash(
        candidate_payload
    )
    candidates.write_text(json.dumps(candidate_payload), encoding="utf-8")

    rows: list[dict[str, object]] = []
    adaptation_units: list[str] = []
    for cohort in COHORTS:
        for donor_index in range(8):
            donor = f"d{donor_index}"
            adaptation_units.append(f"{cohort}::{donor}")
            rows.append(
                {
                    "dataset": cohort,
                    "donor_id": donor,
                    "sample_id": "visit_1",
                    "outer_role": "reference",
                }
            )
    rows.extend(
        {
            "dataset": "query",
            "donor_id": f"q{index}",
            "sample_id": "visit_1",
            "outer_role": "query",
        }
        for index in range(2)
    )
    fold_manifest = tmp_path / "fold_input.json"
    prepare_fold_input_manifest(
        rows=rows,
        output_path=fold_manifest,
        fold_id="lodo_query",
        held_out_dataset="query",
        lineage="B cells",
        tokenized_dataset_path=tokenized,
        gp_library_path=gp_library,
        gene_vocabulary_path=vocabulary,
        projection_gp_candidates_path=candidates,
        tokenized_biological_unit_ids=adaptation_units,
        partition_column="outer_role",
    )

    models: dict[int, Path] = {}
    for seed in SEEDS:
        model_dir = tmp_path / f"model_{seed}"
        model_dir.mkdir()
        checkpoint = model_dir / "model.ckpt"
        checkpoint.write_bytes(f"frozen-model-seed-{seed}".encode())
        manifest = model_dir / "model_manifest.json"
        build_model_artifact_manifest(
            output_path=manifest,
            repo_root=REPO_ROOT,
            vendor_root=REPO_ROOT / "tripso_code" / "tripso",
            fold_input_manifest_path=fold_manifest,
            checkpoint_path=checkpoint,
            fold_id="lodo_query",
            held_out_dataset="query",
            lineage="B cells",
            model_type="Base",
            sampler_mode="hybrid",
            alpha=0.5,
            fine_type_lambda=0.7,
            seed=seed,
            gp_library_path=gp_library,
            gene_vocabulary_path=vocabulary,
            model_configuration={
                "model_type": "Base",
                "embedding_dimension": 256,
                "feature_set": "3000_hvg_plus_gp",
            },
        )
        models[seed] = manifest
    return models


def _endpoint_metadata(program: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_index, cohort in enumerate(COHORTS):
        for donor_index, age in enumerate(np.linspace(22.0, 78.0, 8)):
            donor_id = f"d{donor_index}"
            rows.append(
                {
                    "dataset": cohort,
                    "donor_id": donor_id,
                    "biological_unit_id": f"{cohort}::{donor_id}",
                    "sample_id": "visit_1",
                    "source_observation_id": f"{cohort}::visit_1",
                    "observation_id": f"{cohort}::{donor_id}::visit_1",
                    "age": age + cohort_index * 0.25,
                    "sex": "female" if donor_index % 2 == 0 else "male",
                    "lineage": "B cells",
                    "fine_type": "Naive B",
                    "gp_id": program,
                    "n_cells": 30 + ((donor_index * 3 + cohort_index) % 9),
                    "fine_type_fraction": 0.1
                    + ((donor_index + cohort_index) % 5) * 0.01,
                    "projection_role": "reference",
                    "eligible_for_model_selection": False,
                    "outer_query_evaluation_only": False,
                    "reference_design": "lodo",
                    "heldout_dataset": "query",
                }
            )
    result = pd.DataFrame(rows)
    result.insert(0, "endpoint_row", np.arange(len(result), dtype=np.int64))
    return result


def _write_reference_run(
    tmp_path: Path,
    *,
    seed: int,
    program: str,
    model_manifest: Path,
) -> Path:
    output = tmp_path / f"reference_{seed}_{program}"
    output.mkdir()
    metadata = _endpoint_metadata(program)
    metadata_path = output / "endpoint_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    locations = np.column_stack(
        (
            metadata["age"].to_numpy(dtype=float) / 100.0,
            np.arange(len(metadata), dtype=float) / 100.0,
        )
    ).astype(np.float32)
    locations_path = output / "endpoint_locations.npy"
    np.save(locations_path, locations, allow_pickle=False)
    covariance = np.repeat(np.eye(2, dtype=np.float32)[None, :, :], len(metadata), 0)
    covariance_path = output / "endpoint_covariances.npy"
    np.save(covariance_path, covariance, allow_pickle=False)

    model_id = sha256_file(model_manifest)
    model = json.loads(model_manifest.read_text(encoding="utf-8"))
    endpoint_payload = {
        "schema_version": "immune-health-donor-gp-endpoint/v1",
        "role": "reference",
        "eligible_for_model_selection": False,
        "outer_query_evaluation_only": False,
        "reference_design": "lodo",
        "heldout_dataset": "query",
        "endpoint": {
            "lineage": "B cells",
            "fine_type": "Naive B",
            "gp_id": program,
        },
        "datasets": list(COHORTS),
        "metadata_path": metadata_path.name,
        "features_path": locations_path.name,
        "covariances_path": covariance_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "features_npy_sha256": sha256_file(locations_path),
        "covariances_npy_sha256": sha256_file(covariance_path),
        "shape": list(locations.shape),
        "covariance_shape": list(covariance.shape),
        "dtype": "float32",
        "feature_ids": [f"{program}::location_0000", f"{program}::location_0001"],
        "n_input_endpoint_rows": len(metadata) + 2,
        "n_measurable_rows": len(metadata),
        "observation_id_ordered_sha256": _ordered_digest(
            metadata["observation_id"].astype(str).tolist()
        ),
        "source_provenance": {
            "model_id": model_id,
            "model_manifest": str(model_manifest.resolve()),
            "model_manifest_sha256": model_id,
            "checkpoint_sha256": model["hashes"]["checkpoint_sha256"],
            "fold_id": "lodo_query",
            "seed": seed,
        },
    }
    endpoint_payload["manifest_sha256"] = stable_hash(endpoint_payload)
    endpoint_manifest = output / "endpoint_manifest.json"
    endpoint_manifest.write_text(json.dumps(endpoint_payload), encoding="utf-8")
    endpoint_validation = validate_endpoint_inputs(
        endpoint_manifest,
        metadata_path,
        locations_path,
        expected_role="reference",
    )

    rng = np.random.default_rng(seed + (0 if program == "stable_age" else 1000))
    cohort_index = metadata["dataset"].map(dict(zip(COHORTS, range(4), strict=True)))
    signs = np.where(cohort_index.to_numpy() % 2 == 0, 1.0, -1.0)
    age = metadata["age"].to_numpy(dtype=float)
    slope = np.ones(len(metadata)) if program == "stable_age" else signs
    predicted = 50.0 + slope * (age - 50.0) + rng.normal(0.0, 0.35, len(age))
    crossfit = metadata.reset_index(names="row_index")
    crossfit["inner_crossfit_fold"] = (
        crossfit["donor_id"].str.removeprefix("d").astype(int) % 2
    )
    crossfit["predicted_gp_age"] = predicted
    crossfit["gp_age_acceleration"] = predicted - age
    crossfit_path = output / "training_crossfit_scores.parquet"
    crossfit.to_parquet(crossfit_path, index=False)
    arrays = output / "healthy_reference_arrays.npz"
    np.savez_compressed(arrays, placeholder=np.asarray([seed], dtype=int))
    reference_payload = {
        "schema_version": "immune-health-frozen-healthy-reference/v1",
        "arrays_path": arrays.name,
        "arrays_sha256": sha256_file(arrays),
        "input_composition": "reference_only",
        "final_all_healthy": False,
        "heldout_dataset": "query",
        "query_dataset_offset": "forbidden",
        "weighting_scheme": "donor_pooled",
        "training_datasets": list(COHORTS),
        "seed": seed,
        "endpoint_artifact": endpoint_validation,
        "training_crossfit_scores": {
            "schema_version": "immune-health-training-crossfit-scores/v1",
            "path": crossfit_path.name,
            "sha256": sha256_file(crossfit_path),
            "score_column": "predicted_gp_age",
            "acceleration_column": "gp_age_acceleration",
            "fold_column": "inner_crossfit_fold",
            "fit_scope": "donor_grouped_training_only",
            "query_data_consulted": False,
        },
    }
    reference_manifest = output / "healthy_reference.json"
    reference_manifest.write_text(json.dumps(reference_payload), encoding="utf-8")
    return reference_manifest


def _fixture(tmp_path: Path) -> list[Path]:
    models = _write_fold_and_models(tmp_path)
    return [
        _write_reference_run(
            tmp_path,
            seed=seed,
            program=program,
            model_manifest=models[seed],
        )
        for seed in SEEDS
        for program in PROGRAMS
    ]


def _config() -> TripsoGPSelectionConfig:
    return TripsoGPSelectionConfig(
        minimum_donors_per_cohort=6,
        minimum_age_span=30.0,
        minimum_cohorts=4,
        minimum_sign_concordance=0.75,
        maximum_i2=1.0,
        maximum_fdr=1.0,
        minimum_seed_retention_fraction=1.0,
        minimum_seed_sign_concordance=1.0,
        minimum_median_cells=1.0,
        maximum_absolute_depth_partial_correlation=1.0,
        maximum_absolute_composition_partial_correlation=1.0,
    )


def _write_simple_baseline(path: Path) -> Path:
    frames: list[pd.DataFrame] = []
    for program in PROGRAMS:
        frame = _endpoint_metadata(program)
        cohort_index = frame["dataset"].map(dict(zip(COHORTS, range(4), strict=True)))
        sign = (
            np.ones(len(frame))
            if program == "stable_age"
            else np.where(cohort_index.to_numpy() % 2 == 0, 1.0, -1.0)
        )
        frame["gp_score"] = sign * frame["age"].to_numpy(dtype=float)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def test_selector_uses_training_crossfit_scores_and_writes_self_hashed_outputs(
    tmp_path: Path,
) -> None:
    manifests = _fixture(tmp_path)
    result = select_transferable_tripso_gps(
        manifests,
        lineage="B cells",
        fold_id="lodo_query",
        heldout_dataset="query",
        training_datasets=COHORTS,
        required_seeds=SEEDS,
        config=_config(),
    )
    selected = result.selection.set_index("gp_id")
    assert bool(selected.loc["stable_age", "retained"])
    assert not bool(selected.loc["inconsistent_age", "retained"])
    assert set(result.effects["dataset"]) == set(COHORTS)
    assert set(result.effects["seed"]) == set(SEEDS)
    assert result.manifest["query_data_consulted"] is False
    assert result.manifest["raw_tripso_coordinates_used_for_selection"] is False

    baseline = _write_simple_baseline(tmp_path / "simple_baseline.parquet")
    compared = select_transferable_tripso_gps(
        manifests,
        lineage="B cells",
        fold_id="lodo_query",
        heldout_dataset="query",
        training_datasets=COHORTS,
        required_seeds=SEEDS,
        config=_config(),
        simple_baseline=baseline,
    )
    assert "baseline_mean_absolute_standardized_slope_per_decade" in (
        compared.selection
    )
    assert compared.manifest["simple_baseline"]["sha256"] == sha256_file(baseline)

    output = tmp_path / "selection"
    effects_path, selection_path, manifest_path = write_tripso_gp_selection(
        result, output
    )
    assert effects_path.name == "tripso_gp_cohort_seed_effects.parquet"
    assert selection_path.name == "tripso_gp_selection.parquet"
    validated = validate_tripso_gp_selection_manifest(manifest_path)
    assert validated["selected_endpoints"] == [
        {"lineage": "B cells", "fine_type": "Naive B", "gp_id": "stable_age"}
    ]
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["status"] = "tampered"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        validate_tripso_gp_selection_manifest(manifest_path)


def test_selector_fails_closed_for_missing_seed_and_crossfit_query_leakage(
    tmp_path: Path,
) -> None:
    manifests = _fixture(tmp_path)
    with pytest.raises(ValueError, match="seeds differ"):
        select_transferable_tripso_gps(
            [path for path in manifests if f"reference_{SEEDS[0]}_" in str(path)],
            lineage="B cells",
            fold_id="lodo_query",
            heldout_dataset="query",
            training_datasets=COHORTS,
            required_seeds=SEEDS,
            config=_config(),
        )

    reference_path = manifests[0]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    crossfit_path = (
        reference_path.parent / reference["training_crossfit_scores"]["path"]
    )
    crossfit = pd.read_parquet(crossfit_path)
    crossfit.loc[0, "dataset"] = "query"
    crossfit.to_parquet(crossfit_path, index=False)
    reference["training_crossfit_scores"]["sha256"] = sha256_file(crossfit_path)
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint metadata|query"):
        select_transferable_tripso_gps(
            manifests,
            lineage="B cells",
            fold_id="lodo_query",
            heldout_dataset="query",
            training_datasets=COHORTS,
            required_seeds=SEEDS,
            config=_config(),
        )


def test_selector_cli_accepts_plain_manifest_list(tmp_path: Path) -> None:
    manifests = _fixture(tmp_path)
    manifest_list = tmp_path / "reference_manifests.txt"
    manifest_list.write_text(
        "\n".join(str(path.relative_to(tmp_path)) for path in manifests) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "cli_selection"
    arguments = [
        "select-transferable-tripso-gps",
        "--reference-manifest-list",
        str(manifest_list),
        "--lineage",
        "B cells",
        "--fold-id",
        "lodo_query",
        "--heldout-dataset",
        "query",
        "--weighting-scheme",
        "donor_pooled",
        "--output-dir",
        str(output),
        "--minimum-donors-per-cohort",
        "6",
        "--minimum-age-span",
        "30",
        "--minimum-cohorts",
        "4",
        "--minimum-sign-concordance",
        "0.75",
        "--maximum-i2",
        "1",
        "--maximum-fdr",
        "1",
        "--minimum-median-cells",
        "1",
        "--maximum-absolute-depth-partial-correlation",
        "1",
        "--maximum-absolute-composition-partial-correlation",
        "1",
    ]
    for cohort in COHORTS:
        arguments.extend(("--required-training-dataset", cohort))
    for seed in SEEDS:
        arguments.extend(("--required-seed", str(seed)))
    assert main(arguments) == 0
    validated = validate_tripso_gp_selection_manifest(
        output / "selected_tripso_gps.json"
    )
    assert validated["n_selected_endpoints"] == 1
