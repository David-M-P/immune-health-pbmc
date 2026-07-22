#!/usr/bin/env python3
"""Generate restartable JSONL jobs for staged CPU reference preparation."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

SCHEMA = "immune-health-reference-preparation/v1"
JOB_SCHEMA = "immune-health-slurm-job/v1"


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise ValueError(f"Cannot create safe slug from {value!r}")
    return result


def _gp_candidate_args(features: Mapping[str, Any]) -> list[str]:
    values = [
        "--gp-transfer-minimum-donors-per-cohort",
        str(features["gp_transfer_minimum_donors_per_cohort"]),
        "--gp-transfer-minimum-age-span",
        str(features["gp_transfer_minimum_age_span"]),
        "--gp-transfer-minimum-cohorts",
        str(features["gp_transfer_minimum_cohorts"]),
        "--gp-transfer-minimum-sign-concordance",
        str(features["gp_transfer_minimum_sign_concordance"]),
        "--gp-transfer-maximum-i2",
        str(features["gp_transfer_maximum_i2"]),
        "--gp-transfer-maximum-fdr",
        str(features["gp_transfer_maximum_fdr"]),
        "--gp-transfer-minimum-absolute-standardized-slope-per-decade",
        str(features["gp_transfer_minimum_absolute_standardized_slope_per_decade"]),
    ]
    for program_id in features["gp_projection_control_ids"]:
        values.extend(("--gp-projection-control-id", str(program_id)))
    return values


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA:
        raise ValueError(f"Unsupported reference-preparation config: {path}")
    datasets = config.get("healthy_datasets", [])
    if len(datasets) != 5 or len(set(datasets)) != 5:
        raise ValueError("Reference preparation requires five unique healthy datasets")
    lineages = config.get("lineages")
    if not isinstance(lineages, dict) or not lineages:
        raise ValueError("Reference preparation requires a lineage directory mapping")
    paths = config.get("paths", {})
    if not str(paths.get("fine_type_ontology", "")).strip():
        raise ValueError("paths.fine_type_ontology is required")
    if not str(paths.get("fine_type_ontology_source_candidate", "")).strip():
        raise ValueError("paths.fine_type_ontology_source_candidate is required")
    annotation = config.get("fine_type_annotation", {})
    required_annotation_fields = {
        "source_label_field",
        "source_confidence_field",
        "canonical_field",
        "minimum_confidence",
        "approval_status",
        "approved_by",
        "approved_at",
        "retain_special_categories_in_composition",
        "special_categories_balance_eligible",
        "special_categories_state_eligible",
    }
    missing_annotation = sorted(required_annotation_fields - set(annotation))
    if missing_annotation:
        raise ValueError(
            f"fine_type_annotation lacks required fields: {missing_annotation}"
        )
    for field in (
        "source_label_field",
        "source_confidence_field",
        "canonical_field",
        "approval_status",
        "approved_by",
        "approved_at",
    ):
        if not isinstance(annotation[field], str) or not annotation[field].strip():
            raise ValueError(f"fine_type_annotation.{field} must be nonempty")
    confidence = annotation["minimum_confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("fine_type_annotation.minimum_confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError(
            "fine_type_annotation.minimum_confidence must be between zero and one"
        )
    if annotation["approval_status"] != "approved":
        raise ValueError("fine_type_annotation.approval_status must be approved")
    required_policy_flags = {
        "retain_special_categories_in_composition": True,
        "special_categories_balance_eligible": False,
        "special_categories_state_eligible": False,
    }
    for field, expected in required_policy_flags.items():
        if annotation[field] is not expected:
            raise ValueError(
                f"fine_type_annotation.{field} must be {str(expected).lower()}"
            )
    sizes = config.get("features", {}).get("hvg_sizes", [])
    if sizes != sorted(set(sizes)) or not sizes:
        raise ValueError("features.hvg_sizes must be sorted and unique")
    features = config["features"]
    required_candidate_fields = {
        "gp_transfer_minimum_donors_per_cohort",
        "gp_transfer_minimum_age_span",
        "gp_transfer_minimum_cohorts",
        "gp_transfer_minimum_sign_concordance",
        "gp_transfer_maximum_i2",
        "gp_transfer_maximum_fdr",
        "gp_transfer_minimum_absolute_standardized_slope_per_decade",
        "gp_projection_control_ids",
    }
    missing_candidates = sorted(required_candidate_fields - set(features))
    if missing_candidates:
        raise ValueError(
            f"Reference preparation lacks GP candidate settings: {missing_candidates}"
        )
    controls = features["gp_projection_control_ids"]
    if (
        not isinstance(controls, list)
        or len(controls) != len(set(map(str, controls)))
        or any(not str(value) for value in controls)
    ):
        raise ValueError("features.gp_projection_control_ids must be unique strings")
    materialization = config.get("materialization", {})
    materialization_sizes = materialization.get("hvg_sizes", [])
    if (
        materialization_sizes != sorted(set(materialization_sizes))
        or not materialization_sizes
        or not set(materialization_sizes) <= set(sizes)
    ):
        raise ValueError(
            "materialization.hvg_sizes must be a sorted subset of feature sizes"
        )
    if materialization.get("cell_downsampling") is not False:
        raise ValueError("This preparation path does not permit cell downsampling")
    if set(materialization.get("roles", [])) != {
        "adaptation",
        "validation",
        "query",
    }:
        raise ValueError(
            "LODO materialization must explicitly include adaptation, validation, "
            "and query roles"
        )
    preserve_query_visits = config.get("visit_selection", {}).get(
        "preserve_all_visits_when_query"
    )
    if not isinstance(preserve_query_visits, bool):
        raise ValueError(
            "visit_selection.preserve_all_visits_when_query must be boolean"
        )
    lodo = config.get("lodo_reference", {})
    inner_validation = lodo.get("inner_validation_fold")
    if not isinstance(inner_validation, int) or isinstance(inner_validation, bool):
        raise ValueError(
            "lodo_reference.inner_validation_fold must be a fixed nonnegative integer"
        )
    if inner_validation < 0:
        raise ValueError(
            "lodo_reference.inner_validation_fold must be a fixed nonnegative integer"
        )
    if lodo.get("inner_fold_column") != "inner_fold":
        raise ValueError(
            "lodo_reference.inner_fold_column must be the precomputed LODO "
            "column 'inner_fold'"
        )
    if lodo.get("selection_uses_outer_query") is not False:
        raise ValueError(
            "lodo_reference.selection_uses_outer_query must be explicitly false"
        )
    final = config.get("final_reference", {})
    if final.get("enabled") is not True or final.get("design") != "all_healthy":
        raise ValueError("final_reference must explicitly enable all_healthy design")
    inner_validation = final.get("inner_validation_fold")
    if inner_validation is not None and (
        not isinstance(inner_validation, int) or inner_validation < 0
    ):
        raise ValueError("final_reference.inner_validation_fold must be null or >= 0")
    tokenization = config.get("tokenization", {})
    if tokenization.get("lodo_enabled") is not True:
        raise ValueError("LODO tokenization must be explicitly enabled")
    if tokenization.get("final_reference_enabled") is not True:
        raise ValueError("Final-reference tokenization must be explicitly enabled")
    for field in (
        "output_root",
        "vendor_root",
        "row_chunk_size",
        "nproc",
        "minimum_tokenizable_gp_genes",
    ):
        if field not in tokenization:
            raise ValueError(f"tokenization.{field} is required")
    return config


def _job(
    *,
    job_id: str,
    stage: str,
    seed: int,
    output_dir: str,
    command: Sequence[str],
    upstream: Sequence[str],
    expected_outputs: Sequence[str],
    details: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "schema_version": JOB_SCHEMA,
        "job_id": job_id,
        "stage": stage,
        "seed": int(seed),
        "runnable": True,
        "output_dir": output_dir,
        "working_directory": "${PROJECT_ROOT}",
        "job_spec_path": f"{output_dir}/job_spec.json",
        "upstream_artifacts": [{"path": path} for path in upstream],
        "expected_outputs": list(expected_outputs),
        "command": list(command),
        **dict(details),
    }


def generate_jobs(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    paths = config["paths"]
    annotation = config["fine_type_annotation"]
    features = config["features"]
    materialization = config["materialization"]
    lodo_reference = config["lodo_reference"]
    final_reference = config["final_reference"]
    tokenization = config["tokenization"]
    ontology_candidate = paths["fine_type_ontology_source_candidate"]
    ontology_candidate_name = Path(str(ontology_candidate)).name
    seed = int(config["seed"])
    output_root = paths["output_root"]
    visit_dir = f"{output_root}/visit_selection"
    visit_table = f"{visit_dir}/terekhova_one_visit.tsv"
    visit_json = f"{visit_dir}/terekhova_one_visit.json"
    visits = [
        _job(
            job_id=f"reference-visits-s{seed}",
            stage="visits",
            seed=seed,
            output_dir=visit_dir,
            command=[
                "python",
                "-m",
                "immune_health.cli.prepare_reference",
                "build-terekhova-visits",
                "--metadata",
                "${PROJECT_ROOT}/splits/global_donor_manifest.tsv",
                "--output-dir",
                visit_dir,
                "--seed",
                str(seed),
            ],
            upstream=["${PROJECT_ROOT}/splits/global_donor_manifest.tsv"],
            expected_outputs=[visit_table, visit_json],
            details={"selection_uses_age": False},
        )
    ]

    final_fold_dir = final_reference["fold_output_dir"]
    final_fold_table = f"{final_fold_dir}/all_healthy.tsv"
    final_fold_manifest = f"{final_fold_dir}/all_healthy.json"
    final_fold_command = [
        "python",
        "-m",
        "immune_health.cli.prepare_reference",
        "build-all-healthy-fold",
        "--metadata",
        "${PROJECT_ROOT}/splits/global_donor_manifest.tsv",
        "--output-dir",
        final_fold_dir,
        "--inner-fold-column",
        str(final_reference["inner_fold_column"]),
    ]
    for dataset in config["healthy_datasets"]:
        final_fold_command.extend(("--healthy-dataset", dataset))
    if final_reference.get("inner_validation_fold") is not None:
        final_fold_command.extend(
            (
                "--inner-validation-fold",
                str(final_reference["inner_validation_fold"]),
            )
        )
    final_fold_jobs = [
        _job(
            job_id=f"reference-all-healthy-fold-s{seed}",
            stage="final_fold",
            seed=seed,
            output_dir=final_fold_dir,
            command=final_fold_command,
            upstream=["${PROJECT_ROOT}/splits/global_donor_manifest.tsv"],
            expected_outputs=[final_fold_table, final_fold_manifest],
            details={
                "reference_design": "all_healthy",
                "heldout_dataset": None,
                "inner_validation_fold": final_reference.get("inner_validation_fold"),
            },
        )
    ]

    feature_jobs: list[dict[str, Any]] = []
    materialize_jobs: list[dict[str, Any]] = []
    lodo_tokenize_jobs: list[dict[str, Any]] = []
    lodo_bind_jobs: list[dict[str, Any]] = []
    final_tokenize_jobs: list[dict[str, Any]] = []
    final_bind_jobs: list[dict[str, Any]] = []
    for lineage_dir, lineage_name in config["lineages"].items():
        input_h5ad = f"{paths['merged_root']}/{lineage_dir}/merged.h5ad"
        for heldout in config["healthy_datasets"]:
            fold_slug = f"lodo_{_slug(heldout)}"
            fold_table = f"{paths['split_root']}/lodo_{heldout}.tsv"
            feature_dir = f"{output_root}/features/{lineage_dir}/{fold_slug}"
            feature_manifest = f"{feature_dir}/feature_manifest.json"
            command = [
                "python",
                "-m",
                "immune_health.cli.prepare_reference",
                "select-fold-features",
                "--input-h5ad",
                input_h5ad,
                "--fold-manifest",
                fold_table,
                "--visit-manifest",
                visit_table,
                "--gene-programs",
                paths["gene_programs"],
                "--fine-type-ontology",
                paths["fine_type_ontology"],
                "--output-dir",
                feature_dir,
                "--lineage",
                lineage_name,
                "--inner-validation-fold",
                str(lodo_reference["inner_validation_fold"]),
                "--chunk-size",
                str(features["count_chunk_size"]),
                "--hvg-mean-bins",
                str(features["hvg_mean_bins"]),
                "--hvg-minimum-donor-fraction",
                str(features["hvg_minimum_donor_fraction"]),
                "--hvg-minimum-dataset-fraction",
                str(features["hvg_minimum_dataset_fraction"]),
                "--gp-minimum-mapped-genes",
                str(features["gp_minimum_mapped_genes"]),
                "--gp-maximum-program-size",
                str(features["gp_maximum_program_size"]),
                "--gp-minimum-expression-coverage",
                str(features["gp_minimum_expression_coverage"]),
                "--gp-minimum-donor-coverage",
                str(features["gp_minimum_donor_coverage"]),
                "--gp-minimum-dataset-fraction",
                str(features["gp_minimum_dataset_fraction"]),
                "--gp-redundancy-jaccard-threshold",
                str(features["gp_redundancy_jaccard_threshold"]),
            ]
            for size in features["hvg_sizes"]:
                command.extend(("--hvg-size", str(size)))
            command.extend(_gp_candidate_args(features))
            if not config["visit_selection"]["preserve_all_visits_when_query"]:
                command.append("--global-one-visit-query")
            feature_jobs.append(
                _job(
                    job_id=f"features-{_slug(lineage_dir)}-{fold_slug}-s{seed}",
                    stage="features",
                    seed=seed,
                    output_dir=feature_dir,
                    command=command,
                    upstream=[
                        input_h5ad,
                        fold_table,
                        visit_table,
                        visit_json,
                        paths["gene_programs"],
                        paths["fine_type_ontology"],
                        ontology_candidate,
                    ],
                    expected_outputs=[
                        feature_manifest,
                        *[
                            f"{feature_dir}/model_genes_hvg{size}.txt"
                            for size in features["hvg_sizes"]
                        ],
                        f"{feature_dir}/gene_programs_filtered.gmt",
                        f"{feature_dir}/projection_gp_candidates.tsv",
                        f"{feature_dir}/projection_gp_candidates.json",
                        f"{feature_dir}/fine_type_mapping_qc.tsv",
                        f"{feature_dir}/fine_type_ontology.approved.yaml",
                        f"{feature_dir}/{ontology_candidate_name}",
                    ],
                    details={
                        "lineage": lineage_name,
                        "heldout_dataset": heldout,
                        "inner_validation_fold": lodo_reference[
                            "inner_validation_fold"
                        ],
                        "selection_uses_outer_query": False,
                        "cell_downsampling": False,
                        "fine_type_ontology": paths["fine_type_ontology"],
                        "fine_type_ontology_source_candidate": ontology_candidate,
                        "fine_type_ontology_approved_by": annotation["approved_by"],
                    },
                )
            )
            for hvg_size in materialization["hvg_sizes"]:
                for role in materialization["roles"]:
                    role_dir = (
                        f"{output_root}/materialized/{lineage_dir}/{fold_slug}/"
                        f"hvg{hvg_size}/{role}"
                    )
                    output_h5ad = f"{role_dir}/model_input.h5ad"
                    materialize_jobs.append(
                        _job(
                            job_id=(
                                f"materialize-{_slug(lineage_dir)}-{fold_slug}-"
                                f"hvg{hvg_size}-{role}-s{seed}"
                            ),
                            stage="materialize",
                            seed=seed,
                            output_dir=role_dir,
                            command=[
                                "python",
                                "-m",
                                "immune_health.cli.prepare_reference",
                                "materialize-fold-h5ad",
                                "--input-h5ad",
                                input_h5ad,
                                "--preparation-dir",
                                feature_dir,
                                "--output-h5ad",
                                output_h5ad,
                                "--role",
                                role,
                                "--hvg-size",
                                str(hvg_size),
                                "--row-chunk-size",
                                str(materialization["row_chunk_size"]),
                                "--max-loaded-elements",
                                str(materialization["max_loaded_elements"]),
                            ],
                            upstream=[input_h5ad, feature_manifest],
                            expected_outputs=[
                                output_h5ad,
                                f"{role_dir}/model_input.manifest.json",
                            ],
                            details={
                                "lineage": lineage_name,
                                "heldout_dataset": heldout,
                                "preparation_role": role,
                                "hvg_size": hvg_size,
                                "cell_downsampling": False,
                            },
                        )
                    )

                lineage_slug = _slug(lineage_name)
                vocabulary = f"{feature_dir}/model_genes_hvg{hvg_size}.txt"
                gp_library = f"{feature_dir}/gpdb_filtered.csv"
                projection_gp_candidates = (
                    f"{feature_dir}/projection_gp_candidates.json"
                )
                for role in materialization["roles"]:
                    source_h5ad = (
                        f"{output_root}/materialized/{lineage_dir}/{fold_slug}/"
                        f"hvg{hvg_size}/{role}/model_input.h5ad"
                    )
                    token_dir = (
                        f"{tokenization['output_root']}/{lineage_slug}/{fold_slug}/"
                        f"hvg{hvg_size}/{role}"
                    )
                    token_manifest = f"{token_dir}/tokenization_manifest.json"
                    lodo_tokenize_jobs.append(
                        _job(
                            job_id=(
                                f"tokenize-{lineage_slug}-{fold_slug}-hvg{hvg_size}-"
                                f"{role}-s{seed}"
                            ),
                            stage="lodo_tokenize",
                            seed=seed,
                            output_dir=token_dir,
                            command=[
                                "python",
                                "-m",
                                "immune_health.cli.tokenize_tripso",
                                "tokenize",
                                "--input-h5ad",
                                source_h5ad,
                                "--gene-vocabulary",
                                vocabulary,
                                "--gp-library",
                                gp_library,
                                "--projection-gp-candidates",
                                projection_gp_candidates,
                                "--output-dir",
                                token_dir,
                                "--vendor-root",
                                str(tokenization["vendor_root"]),
                                "--role",
                                role,
                                "--row-chunk-size",
                                str(tokenization["row_chunk_size"]),
                                "--nproc",
                                str(tokenization["nproc"]),
                                "--minimum-tokenizable-gp-genes",
                                str(tokenization["minimum_tokenizable_gp_genes"]),
                            ],
                            upstream=[
                                source_h5ad,
                                f"{Path(source_h5ad).with_suffix('.manifest.json')}",
                                vocabulary,
                                gp_library,
                                projection_gp_candidates,
                            ],
                            expected_outputs=[
                                token_manifest,
                                f"{token_dir}/tokenized.dataset",
                                f"{token_dir}/sequence_qc.json",
                            ],
                            details={
                                "lineage": lineage_name,
                                "reference_design": "lodo",
                                "heldout_dataset": heldout,
                                "preparation_role": role,
                                "hvg_size": hvg_size,
                                "cell_downsampling": False,
                                "hvg_calculation_during_tokenization": False,
                            },
                        )
                    )

                adaptation_token_manifest = (
                    f"{tokenization['output_root']}/{lineage_slug}/{fold_slug}/"
                    f"hvg{hvg_size}/adaptation/tokenization_manifest.json"
                )
                fold_input = (
                    f"{tokenization['output_root']}/{lineage_slug}/{fold_slug}/"
                    f"hvg{hvg_size}/fold_input.json"
                )
                lodo_bind_jobs.append(
                    _job(
                        job_id=(
                            f"bind-{lineage_slug}-{fold_slug}-hvg{hvg_size}-s{seed}"
                        ),
                        stage="lodo_bind",
                        seed=seed,
                        output_dir=str(Path(fold_input).parent),
                        command=[
                            "python",
                            "-m",
                            "immune_health.cli.tokenize_tripso",
                            "build-fold-input",
                            "--tokenization-manifest",
                            adaptation_token_manifest,
                            "--fold-table",
                            fold_table,
                            "--output",
                            fold_input,
                            "--fold-id",
                            fold_slug,
                            "--held-out-dataset",
                            heldout,
                            "--reference-design",
                            "lodo",
                            "--lineage",
                            lineage_name,
                            "--partition-column",
                            "outer_role",
                            "--inner-validation-fold",
                            str(lodo_reference["inner_validation_fold"]),
                            "--inner-fold-column",
                            str(lodo_reference["inner_fold_column"]),
                        ],
                        upstream=[adaptation_token_manifest, fold_table],
                        expected_outputs=[fold_input],
                        details={
                            "lineage": lineage_name,
                            "reference_design": "lodo",
                            "heldout_dataset": heldout,
                            "inner_validation_fold": lodo_reference[
                                "inner_validation_fold"
                            ],
                            "inner_fold_column": lodo_reference["inner_fold_column"],
                            "selection_uses_outer_query": False,
                            "hvg_size": hvg_size,
                            "stage1_compatible_path": True,
                        },
                    )
                )

        # Final production reference: all five cohorts are adaptation data, with
        # one deterministic Terekhova visit and no fabricated held-out cohort.
        final_slug = "all_healthy"
        final_feature_dir = f"{output_root}/features/{lineage_dir}/{final_slug}"
        final_feature_manifest = f"{final_feature_dir}/feature_manifest.json"
        final_feature_command = [
            "python",
            "-m",
            "immune_health.cli.prepare_reference",
            "select-fold-features",
            "--input-h5ad",
            input_h5ad,
            "--fold-manifest",
            final_fold_table,
            "--visit-manifest",
            visit_table,
            "--gene-programs",
            paths["gene_programs"],
            "--fine-type-ontology",
            paths["fine_type_ontology"],
            "--output-dir",
            final_feature_dir,
            "--lineage",
            lineage_name,
            "--reference-design",
            "all_healthy",
            "--global-one-visit-query",
            "--chunk-size",
            str(features["count_chunk_size"]),
            "--hvg-mean-bins",
            str(features["hvg_mean_bins"]),
            "--hvg-minimum-donor-fraction",
            str(features["hvg_minimum_donor_fraction"]),
            "--hvg-minimum-dataset-fraction",
            str(features["hvg_minimum_dataset_fraction"]),
            "--gp-minimum-mapped-genes",
            str(features["gp_minimum_mapped_genes"]),
            "--gp-maximum-program-size",
            str(features["gp_maximum_program_size"]),
            "--gp-minimum-expression-coverage",
            str(features["gp_minimum_expression_coverage"]),
            "--gp-minimum-donor-coverage",
            str(features["gp_minimum_donor_coverage"]),
            "--gp-minimum-dataset-fraction",
            str(features["gp_minimum_dataset_fraction"]),
            "--gp-redundancy-jaccard-threshold",
            str(features["gp_redundancy_jaccard_threshold"]),
        ]
        for size in features["hvg_sizes"]:
            final_feature_command.extend(("--hvg-size", str(size)))
        final_feature_command.extend(_gp_candidate_args(features))
        feature_jobs.append(
            _job(
                job_id=f"features-{_slug(lineage_dir)}-all-healthy-s{seed}",
                stage="features",
                seed=seed,
                output_dir=final_feature_dir,
                command=final_feature_command,
                upstream=[
                    input_h5ad,
                    final_fold_table,
                    final_fold_manifest,
                    visit_table,
                    visit_json,
                    paths["gene_programs"],
                    paths["fine_type_ontology"],
                    ontology_candidate,
                ],
                expected_outputs=[
                    final_feature_manifest,
                    *[
                        f"{final_feature_dir}/model_genes_hvg{size}.txt"
                        for size in features["hvg_sizes"]
                    ],
                    f"{final_feature_dir}/gene_programs_filtered.gmt",
                    f"{final_feature_dir}/projection_gp_candidates.tsv",
                    f"{final_feature_dir}/projection_gp_candidates.json",
                    f"{final_feature_dir}/fine_type_mapping_qc.tsv",
                    f"{final_feature_dir}/fine_type_ontology.approved.yaml",
                    f"{final_feature_dir}/{ontology_candidate_name}",
                ],
                details={
                    "lineage": lineage_name,
                    "reference_design": "all_healthy",
                    "heldout_dataset": None,
                    "inner_validation_fold": final_reference.get(
                        "inner_validation_fold"
                    ),
                    "cell_downsampling": False,
                    "fine_type_ontology": paths["fine_type_ontology"],
                    "fine_type_ontology_source_candidate": ontology_candidate,
                    "fine_type_ontology_approved_by": annotation["approved_by"],
                },
            )
        )

        final_roles = ["adaptation"]
        if final_reference.get("inner_validation_fold") is not None:
            final_roles.append("validation")
        for hvg_size in materialization["hvg_sizes"]:
            role_h5ads: dict[str, str] = {}
            for role in final_roles:
                role_dir = (
                    f"{output_root}/materialized/{lineage_dir}/all_healthy/"
                    f"hvg{hvg_size}/{role}"
                )
                output_h5ad = f"{role_dir}/model_input.h5ad"
                role_h5ads[role] = output_h5ad
                materialize_jobs.append(
                    _job(
                        job_id=(
                            f"materialize-{_slug(lineage_dir)}-all-healthy-"
                            f"hvg{hvg_size}-{role}-s{seed}"
                        ),
                        stage="materialize",
                        seed=seed,
                        output_dir=role_dir,
                        command=[
                            "python",
                            "-m",
                            "immune_health.cli.prepare_reference",
                            "materialize-fold-h5ad",
                            "--input-h5ad",
                            input_h5ad,
                            "--preparation-dir",
                            final_feature_dir,
                            "--output-h5ad",
                            output_h5ad,
                            "--role",
                            role,
                            "--hvg-size",
                            str(hvg_size),
                            "--row-chunk-size",
                            str(materialization["row_chunk_size"]),
                            "--max-loaded-elements",
                            str(materialization["max_loaded_elements"]),
                        ],
                        upstream=[input_h5ad, final_feature_manifest],
                        expected_outputs=[
                            output_h5ad,
                            f"{role_dir}/model_input.manifest.json",
                        ],
                        details={
                            "lineage": lineage_name,
                            "reference_design": "all_healthy",
                            "heldout_dataset": None,
                            "preparation_role": role,
                            "hvg_size": hvg_size,
                            "cell_downsampling": False,
                        },
                    )
                )

            lineage_slug = _slug(lineage_name)
            token_dir = (
                f"{tokenization['output_root']}/{lineage_slug}/all_healthy/"
                f"hvg{hvg_size}/adaptation"
            )
            token_manifest = f"{token_dir}/tokenization_manifest.json"
            vocabulary = f"{final_feature_dir}/model_genes_hvg{hvg_size}.txt"
            gp_library = f"{final_feature_dir}/gpdb_filtered.csv"
            projection_gp_candidates = (
                f"{final_feature_dir}/projection_gp_candidates.json"
            )
            final_tokenize_jobs.append(
                _job(
                    job_id=(
                        f"tokenize-{lineage_slug}-all-healthy-hvg{hvg_size}-s{seed}"
                    ),
                    stage="final_tokenize",
                    seed=seed,
                    output_dir=token_dir,
                    command=[
                        "python",
                        "-m",
                        "immune_health.cli.tokenize_tripso",
                        "tokenize",
                        "--input-h5ad",
                        role_h5ads["adaptation"],
                        "--gene-vocabulary",
                        vocabulary,
                        "--gp-library",
                        gp_library,
                        "--projection-gp-candidates",
                        projection_gp_candidates,
                        "--output-dir",
                        token_dir,
                        "--vendor-root",
                        str(tokenization["vendor_root"]),
                        "--role",
                        "adaptation",
                        "--row-chunk-size",
                        str(tokenization["row_chunk_size"]),
                        "--nproc",
                        str(tokenization["nproc"]),
                        "--minimum-tokenizable-gp-genes",
                        str(tokenization["minimum_tokenizable_gp_genes"]),
                    ],
                    upstream=[
                        role_h5ads["adaptation"],
                        f"{Path(role_h5ads['adaptation']).with_suffix('.manifest.json')}",
                        vocabulary,
                        gp_library,
                        projection_gp_candidates,
                    ],
                    expected_outputs=[
                        token_manifest,
                        f"{token_dir}/tokenized.dataset",
                        f"{token_dir}/sequence_qc.json",
                    ],
                    details={
                        "lineage": lineage_name,
                        "reference_design": "all_healthy",
                        "heldout_dataset": None,
                        "hvg_size": hvg_size,
                        "cell_downsampling": False,
                        "hvg_calculation_during_tokenization": False,
                    },
                )
            )

            fold_input = (
                f"{tokenization['output_root']}/{lineage_slug}/all_healthy/"
                f"hvg{hvg_size}/fold_input.json"
            )
            final_bind_jobs.append(
                _job(
                    job_id=f"bind-{lineage_slug}-all-healthy-hvg{hvg_size}-s{seed}",
                    stage="final_bind",
                    seed=seed,
                    output_dir=str(Path(fold_input).parent),
                    command=[
                        "python",
                        "-m",
                        "immune_health.cli.tokenize_tripso",
                        "build-fold-input",
                        "--tokenization-manifest",
                        token_manifest,
                        "--fold-table",
                        final_fold_table,
                        "--output",
                        fold_input,
                        "--fold-id",
                        "all_healthy",
                        "--reference-design",
                        "all_healthy",
                        "--lineage",
                        lineage_name,
                        "--partition-column",
                        "reference_partition",
                    ],
                    upstream=[token_manifest, final_fold_table, final_fold_manifest],
                    expected_outputs=[fold_input],
                    details={
                        "lineage": lineage_name,
                        "reference_design": "all_healthy",
                        "heldout_dataset": None,
                        "hvg_size": hvg_size,
                        "stage3_compatible_path": True,
                    },
                )
            )
    return {
        "visits": visits,
        "final_fold": final_fold_jobs,
        "features": feature_jobs,
        "materialize": materialize_jobs,
        "lodo_tokenize": lodo_tokenize_jobs,
        "lodo_bind": lodo_bind_jobs,
        "final_tokenize": final_tokenize_jobs,
        "final_bind": final_bind_jobs,
    }


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    records = list(rows)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return len(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/reference_preparation.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("slurm/manifests/reference_prep")
    )
    parser.add_argument(
        "--stage",
        choices=(
            "visits",
            "final_fold",
            "features",
            "materialize",
            "lodo_tokenize",
            "lodo_bind",
            "final_tokenize",
            "final_bind",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = generate_jobs(load_config(args.config))
    stages = tuple(jobs) if args.stage == "all" else (args.stage,)
    summary = {stage: len(jobs[stage]) for stage in stages}
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for stage in stages:
        _atomic_jsonl(args.output_dir / f"{stage}.jsonl", jobs[stage])
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "immune-health-reference-preparation-jobs/v1",
                "job_counts": summary,
                "submission_order": [
                    "visits",
                    "final_fold",
                    "features",
                    "materialize",
                    "lodo_tokenize",
                    "lodo_bind",
                    "final_tokenize",
                    "final_bind",
                ],
                "requires_afterok_dependencies": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
