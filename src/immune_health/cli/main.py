"""Central command line for donor-aware PBMC immune-health analyses."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.aggregation.bootstrap import (
    fine_type_stratified_bootstrap,
)
from immune_health.aggregation.empirical_index import write_empirical_row_index
from immune_health.aggregation.summarize import aggregate_fine_type_distributions
from immune_health.baselines.composition import build_composition_table
from immune_health.baselines.gp_scores import score_gene_programs
from immune_health.baselines.latent import ElasticNetAgeModel, TrainOnlyPCA
from immune_health.baselines.pseudobulk import (
    build_pseudobulk,
    ensure_donor_observation_ids,
)
from immune_health.data.audit import run_audit
from immune_health.data.contracts import (
    require_gene_coverage,
    validate_cell_metadata,
    validate_raw_counts,
)
from immune_health.data.ontology import (
    generate_candidate_ontology,
    load_fine_type_ontology,
    summarize_fine_type_labels,
    write_candidate_ontology,
)
from immune_health.evaluation.metrics import evaluate_lodo as evaluate_lodo_metrics
from immune_health.gene_programs.io import load_gene_programs, validate_gp_resource
from immune_health.gene_programs.transferability import (
    TransferabilityConfig,
    select_transferable_gene_programs,
)
from immune_health.gene_programs.tripso_selection import (
    TripsoGPSelectionConfig,
    select_transferable_tripso_gps,
    write_tripso_gp_selection,
)
from immune_health.healthy_reference.bootstrap import (
    bootstrap_healthy_reference_scores,
)
from immune_health.healthy_reference.diagnostics import (
    age_support_grid,
    cohort_feature_age_effects,
    query_age_support,
)
from immune_health.healthy_reference.empirical import score_empirical_matched_depth
from immune_health.healthy_reference.endpoint import (
    assemble_donor_gp_endpoint,
    validate_endpoint_inputs,
)
from immune_health.healthy_reference.kernel import (
    DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    AgeKernelReference,
)
from immune_health.healthy_reference.trajectory import (
    HealthyTrajectory,
    cross_fit_trajectory,
)
from immune_health.healthy_reference.uncertainty import combine_seed_score_tables
from immune_health.provenance import (
    atomic_write_json,
    completion_marker,
    sha256_file,
    stable_hash,
)
from immune_health.reporting.comparison import build_comparison_report
from immune_health.sampling.hierarchical import HierarchicalCellSampler
from immune_health.splits.lodo import (
    REFERENCE_DATASETS,
    build_global_donor_manifest,
    write_lodo_manifests,
)
from immune_health.tripso_adapter.arrow_bridge import (
    validate_arrow_conversion_for_aggregation,
)
from immune_health.tripso_adapter.contracts import (
    REQUIRED_VENDOR_ASSETS,
)
from immune_health.tripso_adapter.contracts import (
    sha256_path as tripso_sha256_path,
)
from immune_health.tripso_adapter.projection import (
    run_vendor_frozen_projection,
    validate_frozen_query_resources,
)
from immune_health.tripso_adapter.provenance import build_model_artifact_manifest
from immune_health.tripso_adapter.training import (
    TripsoTrainingSpec,
    build_training_call,
    run_tripso_training,
)

from ._reference import (
    load_age_kernel_reference,
    load_reference,
    write_age_kernel_reference,
    write_reference,
)
from ._utils import (
    IDENTIFIER_CONTRACT,
    config_path,
    expand_path,
    guard_outputs,
    json_plan,
    load_config,
    load_matrix,
    read_gene_ids,
    read_json,
    read_table,
    require_columns,
    write_table,
)

LOGGER = logging.getLogger("immune_health.cli")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_CONFIG = REPO_ROOT / "configs/data/reference.yaml"
DEFAULT_GP_CONFIG = REPO_ROOT / "configs/gene_programs/default.yaml"
DEFAULT_PROVENANCE = Path(
    "/faststorage/project/CancerEvolution_shared/Projects/David/phd/scih/docs/"
    "raw_to_lineage_split_and_merge.md"
)


def _configured(
    args: argparse.Namespace, default: Path | None = None
) -> dict[str, Any]:
    path = args.config if args.config is not None else default
    return load_config(path)


def _path_from_config(config: Mapping[str, Any], section: str, key: str) -> Path | None:
    value = config.get(section, {}).get(key)
    return None if value in {None, ""} else config_path(config, value)


def _normalise_cell_metadata(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    validate_full_contract: bool = False,
) -> pd.DataFrame:
    """Map configured audit fields to package-level names and add stable IDs."""

    result = frame.copy()
    fields = dict((config or {}).get("metadata_fields", {}))
    aliases = {
        "dataset": fields.get("dataset", "dataset"),
        "donor_id": fields.get("donor_id", "donor_id"),
        "sample_id": fields.get("sample_id", "sample_id"),
        "age": fields.get("age", "age"),
        "sex": fields.get("sex", "sex"),
        "lineage": fields.get("lineage", "lineage"),
        "ctype_low": fields.get("fine_type", "ctype_low"),
        "ctype_low_conf": fields.get("fine_type_confidence", "ctype_low_conf"),
    }
    for target, source in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    if "fine_type" not in result and "ctype_low" in result:
        result["fine_type"] = result["ctype_low"]
    if validate_full_contract:
        result, _ = validate_cell_metadata(result)
    else:
        result = ensure_donor_observation_ids(result)
    return result


def _write_command_manifest(
    output_dir: Path,
    *,
    stage: str,
    outputs: Sequence[Path],
    args: argparse.Namespace,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    marker = Path(output_dir) / f"{stage}_manifest.json"
    completion_marker(
        marker,
        stage=stage,
        outputs=outputs,
        configuration={
            key: value
            for key, value in vars(args).items()
            if key not in {"handler", "log_level"}
        },
        repo_root=REPO_ROOT,
        extra={
            "identifier_contract": IDENTIFIER_CONTRACT,
            **dict(extra or {}),
        },
    )
    return marker


def command_audit_data(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    data_root = args.data_root or _path_from_config(
        config, "paths", "reference_data_root"
    )
    if data_root is None:
        raise ValueError("audit-data requires --data-root or paths.reference_data_root")
    provenance = args.provenance or DEFAULT_PROVENANCE
    output_dir = args.output_dir
    missing = [path for path in (data_root, provenance) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Required audit inputs do not exist: {missing}")
    merged = Path(data_root) / "reference_lineages" / "merged"
    if not merged.is_dir():
        raise FileNotFoundError(f"Merged lineage root does not exist: {merged}")
    if args.dry_run:
        print(
            json_plan(
                "audit-data",
                data_root=Path(data_root).resolve(),
                provenance=Path(provenance).resolve(),
                output_dir=output_dir.resolve(),
                validation=(
                    "Paths and merged directory validated; H5AD metadata and matrices "
                    "were not opened"
                ),
            )
        )
        return 0
    run_audit(
        data_root=Path(data_root),
        output_dir=output_dir,
        provenance_path=Path(provenance),
        repo_root=REPO_ROOT,
    )
    return 0


def command_build_ontology(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    input_path = args.input
    output_path = args.output
    summary_path = args.summary_output
    header = read_table(input_path, nrows=0)
    require_columns(header, ("dataset", "lineage", "fine_type"), "fine-type audit")
    minimum_confidence = args.minimum_confidence
    if minimum_confidence is None:
        minimum_confidence = float(
            config.get("annotation", {}).get("minimum_confidence_default", 0.9)
        )
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum annotation confidence must be between 0 and 1")
    if args.minimum_cells_for_state < 1 or args.poor_donor_coverage_below < 1:
        raise ValueError("ontology cell and donor thresholds must be positive")
    if args.dry_run:
        print(
            json_plan(
                "build-fine-type-ontology",
                input=input_path.resolve(),
                output=output_path.resolve(),
                summary_output=summary_path.resolve(),
                minimum_confidence=minimum_confidence,
                policy=(
                    "Exact-label identity mappings only; generated ontology remains "
                    "pending scientific approval"
                ),
            )
        )
        return 0
    guard_outputs((output_path, summary_path), overwrite=args.overwrite)
    records = read_table(input_path)
    summary = summarize_fine_type_labels(
        records,
        confidence_threshold=minimum_confidence,
        poor_donor_coverage_below=args.poor_donor_coverage_below,
    )
    ontology = generate_candidate_ontology(
        records,
        minimum_confidence=minimum_confidence,
        minimum_cells_for_state=args.minimum_cells_for_state,
        poor_donor_coverage_below=args.poor_donor_coverage_below,
    )
    write_candidate_ontology(ontology, output_path)
    write_table(summary, summary_path)
    return 0


def _configured_datasets(config: Mapping[str, Any]) -> tuple[str, ...]:
    entries = config.get("datasets")
    if not isinstance(entries, list):
        return REFERENCE_DATASETS
    values = tuple(
        str(entry["name"] if isinstance(entry, Mapping) else entry) for entry in entries
    )
    return values or REFERENCE_DATASETS


def command_make_lodo_folds(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    header = read_table(args.metadata, nrows=0)
    require_columns(header, ("dataset", "donor_id"), "split metadata")
    datasets = tuple(args.dataset or _configured_datasets(config))
    if args.n_inner_folds < 2:
        raise ValueError("n-inner-folds must be at least 2")
    edges = np.asarray(args.age_bin_edges, dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("age-bin-edges must be strictly increasing")
    expected_paths = [
        args.output_dir / "global_donor_manifest.tsv",
        args.output_dir / "split_manifest.json",
        *(args.output_dir / f"lodo_{dataset}.tsv" for dataset in datasets),
    ]
    if args.dry_run:
        print(
            json_plan(
                "make-lodo-folds",
                metadata=args.metadata.resolve(),
                output_dir=args.output_dir.resolve(),
                datasets=datasets,
                n_inner_folds=args.n_inner_folds,
                age_bin_edges=args.age_bin_edges,
                biological_split_unit="donor",
                outputs=expected_paths,
            )
        )
        return 0
    guard_outputs(expected_paths, overwrite=args.overwrite)
    records = read_table(args.metadata)
    manifest = build_global_donor_manifest(
        records,
        datasets=datasets,
        n_inner_folds=args.n_inner_folds,
        age_bin_edges=args.age_bin_edges,
        seed=args.seed,
    )
    write_lodo_manifests(
        manifest,
        args.output_dir,
        datasets=datasets,
        source_path=args.metadata,
    )
    return 0


def _configured_gp_resources(
    config: Mapping[str, Any], explicit: Sequence[str]
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    if explicit:
        for value in explicit:
            if "=" in value:
                name, raw_path = value.split("=", 1)
            else:
                raw_path = value
                name = Path(raw_path).stem
            resources.append(
                {"name": name, "path": Path(raw_path).resolve(), "required": True}
            )
        return resources
    for name, entry in config.get("resources", {}).items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"GP resource {name!r} must be a mapping")
        raw_path = entry.get("path")
        resources.append(
            {
                "name": str(name),
                "path": (
                    None if raw_path in {None, ""} else config_path(config, raw_path)
                ),
                "required": bool(entry.get("required", False)),
                "status": entry.get("status"),
            }
        )
    return resources


def command_validate_gene_programs(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_GP_CONFIG)
    resources = _configured_gp_resources(config, args.resource)
    if not resources:
        raise ValueError(
            "No GP resources configured; use --resource NAME=PATH or a GP config"
        )
    missing_required = [
        item["name"]
        for item in resources
        if item["required"]
        and (item["path"] is None or not Path(item["path"]).is_file())
    ]
    if missing_required:
        raise FileNotFoundError(
            "Required GP resources are missing: " + ", ".join(missing_required)
        )
    if args.dry_run:
        print(
            json_plan(
                "validate-gene-programs",
                resources=resources,
                output=args.output.resolve(),
                production=not args.allow_test_resources,
                scope=(
                    "Resource syntax/availability only; expression and donor coverage "
                    "must be filtered separately inside each training fold"
                ),
            )
        )
        return 0
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    guard_outputs((args.output, manifest_path), overwrite=args.overwrite)
    rows: list[dict[str, Any]] = []
    validated_resources: list[dict[str, Any]] = []
    for item in resources:
        path = item["path"]
        if path is None or not Path(path).is_file():
            rows.append(
                {
                    "resource": item["name"],
                    "path": path,
                    "required": item["required"],
                    "status": item.get("status", "optional_missing"),
                    "program_id": pd.NA,
                    "n_genes": pd.NA,
                }
            )
            continue
        programs = validate_gp_resource(
            path,
            source=item["name"],
            production=not args.allow_test_resources,
        )
        resolved_path = Path(path).resolve()
        validated_resources.append(
            {
                "name": item["name"],
                "path": str(resolved_path),
                "sha256": sha256_file(resolved_path),
                "n_programs": len(programs),
            }
        )
        rows.extend(
            {
                "resource": item["name"],
                "path": str(Path(path).resolve()),
                "required": item["required"],
                "status": "valid",
                "program_id": program.program_id,
                "source": program.source,
                "category": program.category,
                "direction": program.direction,
                "n_genes": len(program.genes),
                "n_unique_genes": len(set(program.genes)),
            }
            for program in programs
        )
    result = pd.DataFrame(rows)
    write_table(result, args.output)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "immune-health-gp-resource-validation/v1",
            "status": "complete",
            "production_validation": not args.allow_test_resources,
            "n_configured_resources": len(resources),
            "n_valid_programs": int(result["program_id"].notna().sum()),
            "validated_resources": validated_resources,
            "missing_optional_resources": sorted(
                result.loc[
                    result["program_id"].isna() & ~result["required"].astype(bool),
                    "resource",
                ].astype(str)
            ),
            "training_fold_filtering_performed": False,
            "training_fold_filtering_note": (
                "Expression/donor coverage and redundancy filtering require a "
                "declared LODO training fold and are intentionally not inferred here."
            ),
            "output": str(args.output.resolve()),
        },
    )
    return 0


def command_build_sampling_manifest(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    header = read_table(args.metadata, nrows=0)
    fields = config.get("metadata_fields", {})
    fine_source = fields.get("fine_type", "ctype_low")
    fine_column = fine_source if fine_source in header else "fine_type"
    required = {
        fields.get("dataset", "dataset"),
        fields.get("donor_id", "donor_id"),
        fine_column,
    }
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"Sampler metadata header is missing columns: {missing}")
    if not 0 <= args.alpha <= 1 or not 0 <= args.fine_type_lambda <= 1:
        raise ValueError("sampler alpha and fine-type lambda must be between 0 and 1")
    if args.n_cells < 0 or args.batch_size < 1 or args.epoch < 0:
        raise ValueError("sampler sizes/epoch must be nonnegative and batch positive")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("distributed rank must satisfy 0 <= rank < world-size")
    outputs = (
        args.output_dir / "selected_cells.parquet",
        args.output_dir / "sampling_distribution.tsv",
        args.output_dir / "sampling_summary.json",
        args.output_dir / "sampling_manifest.json",
    )
    if args.dry_run:
        print(
            json_plan(
                "build-sampling-manifest",
                metadata=args.metadata.resolve(),
                output_dir=args.output_dir.resolve(),
                lineage=args.lineage,
                n_cells=args.n_cells,
                batch_size=args.batch_size,
                mode=args.mode,
                alpha=args.alpha,
                fine_type_lambda=args.fine_type_lambda,
                distributed_stream={"rank": args.rank, "world_size": args.world_size},
            )
        )
        return 0
    guard_outputs(outputs, overwrite=args.overwrite)
    metadata = read_table(args.metadata)
    metadata = _normalise_cell_metadata(metadata, config)
    sampler = HierarchicalCellSampler(
        metadata,
        lineage=args.lineage,
        alpha=args.alpha,
        mode=args.mode,
        hybrid_lambda=args.fine_type_lambda,
        min_cells_per_fine_type=args.min_cells_per_fine_type,
        seed=args.seed,
        rank=args.rank,
        world_size=args.world_size,
    )
    result = sampler.sample_epoch(
        args.n_cells, epoch=args.epoch, batch_size=args.batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame({"cell_position": result.cell_positions})
    if result.cell_ids is not None:
        selected["cell_id"] = result.cell_ids
    write_table(selected, outputs[0])
    result.write_log(args.output_dir, prefix="sampling")
    _write_command_manifest(
        args.output_dir,
        stage="sampling",
        outputs=outputs[:3],
        args=args,
        extra={"summary": result.summary()},
    )
    return 0


def _program_mapping(path: Path) -> dict[str, tuple[str, ...]]:
    return {program.program_id: program.genes for program in load_gene_programs(path)}


def command_build_pseudobulk_baselines(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    header = read_table(args.metadata, nrows=0)
    fields = config.get("metadata_fields", {})
    required_sources = {
        fields.get("dataset", "dataset"),
        fields.get("donor_id", "donor_id"),
        fields.get("sample_id", "sample_id"),
        fields.get("lineage", "lineage"),
        fields.get("fine_type", "ctype_low"),
    }
    missing = sorted(required_sources - set(header.columns))
    if missing:
        raise ValueError(f"Baseline metadata header is missing columns: {missing}")
    if args.counts.suffix.lower() != ".npz":
        raise ValueError("Raw counts must be supplied as a sparse SciPy .npz")
    for path in (args.counts, args.genes):
        if not path.is_file():
            raise FileNotFoundError(f"Baseline input is missing: {path}")
    if args.gp_resource is not None and not args.gp_resource.is_file():
        raise FileNotFoundError(f"GP resource is missing: {args.gp_resource}")
    if args.min_cells < 1 or args.minimum_gp_genes < 1:
        raise ValueError("pseudobulk and GP minimum counts must be positive")
    if args.fit_age_model and args.heldout_dataset is None:
        raise ValueError("--fit-age-model requires a declared --heldout-dataset")
    if args.select_transferable_gps and args.gp_resource is None:
        raise ValueError("--select-transferable-gps requires --gp-resource")
    if args.select_transferable_gps:
        TransferabilityConfig(
            minimum_donors_per_cohort=args.transfer_minimum_donors,
            minimum_age_span=args.transfer_minimum_age_span,
            minimum_cohorts=args.transfer_minimum_cohorts,
            minimum_sign_concordance=args.transfer_minimum_sign_concordance,
            maximum_i2=args.transfer_maximum_i2,
            maximum_fdr=args.transfer_maximum_fdr,
            minimum_absolute_standardized_slope_per_decade=(
                args.transfer_minimum_effect
            ),
        ).validate()
    outputs = {
        "composition": args.output_dir / "composition.parquet",
        "counts": args.output_dir / "pseudobulk_counts.npz",
        "metadata": args.output_dir / "pseudobulk_metadata.parquet",
        "genes": args.output_dir / "gene_ids.txt",
    }
    optional_planned: list[Path] = []
    if args.gp_resource is not None:
        optional_planned.append(args.output_dir / "simple_gp_scores.parquet")
        if args.select_transferable_gps:
            optional_planned.extend(
                (
                    args.output_dir / "gp_age_effects.parquet",
                    args.output_dir / "transferable_gp_selection.parquet",
                )
            )
    if args.heldout_dataset is not None:
        optional_planned.append(args.output_dir / "pseudobulk_pca.parquet")
        if args.fit_age_model:
            optional_planned.append(args.output_dir / "elastic_net_age_query.parquet")
    command_manifest = args.output_dir / "pseudobulk_baselines_manifest.json"
    if args.dry_run:
        print(
            json_plan(
                "build-pseudobulk-baselines",
                counts=args.counts.resolve(),
                metadata=args.metadata.resolve(),
                genes=args.genes.resolve(),
                output_dir=args.output_dir.resolve(),
                min_cells=args.min_cells,
                heldout_dataset=args.heldout_dataset,
                gp_resource=args.gp_resource,
                select_transferable_gps=args.select_transferable_gps,
                planned_components=(
                    "composition, sparse raw-count pseudobulk, optional GP scores, "
                    "optional training-only PCA/elastic-net projection"
                ),
            )
        )
        return 0
    guard_outputs(
        [*outputs.values(), *optional_planned, command_manifest],
        overwrite=args.overwrite,
    )
    counts = load_matrix(args.counts, require_sparse=True)
    count_report = validate_raw_counts(counts)
    genes = read_gene_ids(args.genes)
    if counts.shape[1] != len(genes):
        raise ValueError("Count matrix width differs from ordered gene vocabulary")
    metadata = _normalise_cell_metadata(
        read_table(args.metadata), config, validate_full_contract=True
    )
    if len(metadata) != counts.shape[0]:
        raise ValueError("Count rows and cell metadata rows differ")
    composition = build_composition_table(metadata)
    pseudobulk = build_pseudobulk(
        counts,
        metadata,
        genes,
        min_cells=args.min_cells,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(composition, outputs["composition"])
    sparse.save_npz(outputs["counts"], pseudobulk.counts)
    write_table(pseudobulk.metadata, outputs["metadata"])
    outputs["genes"].write_text("\n".join(pseudobulk.gene_ids) + "\n")

    optional_outputs: list[Path] = []
    if args.gp_resource is not None:
        gp_scores = score_gene_programs(
            pseudobulk.counts,
            pseudobulk.gene_ids,
            _program_mapping(args.gp_resource),
            method=args.gp_score_method,
            minimum_genes=args.minimum_gp_genes,
        ).merge(
            pseudobulk.metadata.reset_index(names="summary_index"),
            on="summary_index",
            validate="many_to_one",
        )
        gp_path = args.output_dir / "simple_gp_scores.parquet"
        write_table(gp_scores, gp_path)
        optional_outputs.append(gp_path)
        if args.select_transferable_gps:
            excluded = () if args.heldout_dataset is None else (args.heldout_dataset,)
            transferability = select_transferable_gene_programs(
                gp_scores,
                excluded_datasets=excluded,
                config=TransferabilityConfig(
                    minimum_donors_per_cohort=args.transfer_minimum_donors,
                    minimum_age_span=args.transfer_minimum_age_span,
                    minimum_cohorts=args.transfer_minimum_cohorts,
                    minimum_sign_concordance=args.transfer_minimum_sign_concordance,
                    maximum_i2=args.transfer_maximum_i2,
                    maximum_fdr=args.transfer_maximum_fdr,
                    minimum_absolute_standardized_slope_per_decade=(
                        args.transfer_minimum_effect
                    ),
                ),
            )
            effect_path = args.output_dir / "gp_age_effects.parquet"
            selection_path = args.output_dir / "transferable_gp_selection.parquet"
            write_table(transferability.effects, effect_path)
            write_table(transferability.selection, selection_path)
            optional_outputs.extend((effect_path, selection_path))

    if args.heldout_dataset is not None:
        is_query = pseudobulk.metadata["dataset"].astype(str).eq(args.heldout_dataset)
        if not is_query.any() or is_query.all():
            raise ValueError(
                "heldout_dataset must select at least one query and one training "
                "summary"
            )
        training = np.flatnonzero(~is_query.to_numpy())
        query = np.flatnonzero(is_query.to_numpy())
        model = TrainOnlyPCA(
            n_components=args.n_components,
            random_state=args.seed,
            max_dense_values=args.max_dense_values,
        )
        training_coordinates = model.fit_transform(
            pseudobulk.counts[training],
            feature_ids=pseudobulk.gene_ids,
            training_biological_units=pseudobulk.metadata.iloc[training][
                "biological_unit_id"
            ],
        )
        query_coordinates = model.transform(
            pseudobulk.counts[query],
            feature_ids=pseudobulk.gene_ids,
            query_biological_units=pseudobulk.metadata.iloc[query][
                "biological_unit_id"
            ],
        )
        coordinates = np.vstack((training_coordinates, query_coordinates))
        order = np.concatenate((training, query))
        coordinate_frame = pseudobulk.metadata.iloc[order].reset_index(drop=True)
        for index in range(coordinates.shape[1]):
            coordinate_frame[f"pc_{index + 1}"] = coordinates[:, index]
        coordinate_frame["outer_role"] = np.where(
            coordinate_frame["dataset"].astype(str).eq(args.heldout_dataset),
            "query",
            "reference",
        )
        pca_path = args.output_dir / "pseudobulk_pca.parquet"
        write_table(coordinate_frame, pca_path)
        optional_outputs.append(pca_path)
        if args.fit_age_model:
            age_model = ElasticNetAgeModel(
                n_splits=args.age_inner_folds, random_state=args.seed
            ).fit(
                training_coordinates,
                pseudobulk.metadata.iloc[training]["age"],
                pseudobulk.metadata.iloc[training]["biological_unit_id"],
            )
            age_output = pseudobulk.metadata.iloc[query].copy().reset_index(drop=True)
            age_output["predicted_age"] = age_model.predict(query_coordinates)
            age_path = args.output_dir / "elastic_net_age_query.parquet"
            write_table(age_output, age_path)
            optional_outputs.append(age_path)

    _write_command_manifest(
        args.output_dir,
        stage="pseudobulk_baselines",
        outputs=[*outputs.values(), *optional_outputs],
        args=args,
        extra={"count_validation": vars(count_report)},
    )
    return 0


@contextmanager
def _vendor_on_path(vendor_root: Path) -> Iterator[None]:
    """Expose the checked-out vendor package without modifying it."""

    value = str(Path(vendor_root).resolve())
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted and value in sys.path:
            sys.path.remove(value)


def _load_parameters(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_json(path)


def _fold_input_from_job(job: Mapping[str, Any]) -> Path:
    matches = [
        expand_path(item["path"])
        for item in job.get("upstream_artifacts", [])
        if item.get("json_require", {}).get("schema_version")
        == "immune-health-tripso-fold-input/v1"
    ]
    if len(matches) != 1:
        raise ValueError(
            "TRIPSO job spec must identify exactly one fold-input manifest upstream"
        )
    return matches[0]


def _resolve_training_request(
    args: argparse.Namespace,
) -> tuple[TripsoTrainingSpec, dict[str, Any], Path, dict[str, Any]]:
    job: dict[str, Any] = {}
    if args.job_spec is not None:
        job = read_json(args.job_spec)
        if job.get("schema_version") != "immune-health-slurm-job/v1":
            raise ValueError(f"Unsupported job-spec schema: {args.job_spec}")
        if not job.get("runnable", False):
            raise ValueError("Job spec is pending a scientific configuration selection")
        if int(job["seed"]) != args.seed:
            raise ValueError(
                f"CLI seed {args.seed} differs from job-spec seed {job['seed']}"
            )
        fold_input = _fold_input_from_job(job)
        output_dir = expand_path(job["output_dir"])
        model_type = str(job.get("model_type", args.model_type))
        parameters = _load_parameters(args.parameters_json)
        parameters.setdefault("biological_split_unit", "donor")
    else:
        if args.fold_input is None or args.output_dir is None:
            raise ValueError(
                "train-tripso requires --fold-input and --output-dir, or --job-spec"
            )
        fold_input = args.fold_input
        output_dir = args.output_dir
        model_type = args.model_type
        parameters = _load_parameters(args.parameters_json)
        parameters.setdefault("biological_split_unit", "donor")
    raw_project_sampler = parameters.get("project_sampler")
    if raw_project_sampler is not None and not isinstance(raw_project_sampler, Mapping):
        raise ValueError("parameters-json project_sampler must be a JSON mapping")
    project_sampler = dict(raw_project_sampler or {})
    project_sampler.setdefault("enabled", True)
    if job:
        # The generated job is the scientific record for a staged sampler
        # comparison, so its selected mode must override a generic parameter file.
        project_sampler_enabled = job.get("project_sampler_enabled", True)
        if not isinstance(project_sampler_enabled, bool):
            raise ValueError("job-spec project_sampler_enabled must be boolean")
        project_sampler["enabled"] = project_sampler_enabled
        if project_sampler_enabled:
            project_mode = job.get("project_sampler_mode") or job["sampler_mode"]
            if job.get("alpha") is None or job.get("fine_type_lambda") is None:
                raise ValueError(
                    "Hierarchical sampler jobs require numeric alpha and lambda"
                )
            project_sampler.update(
                {
                    "mode": str(project_mode),
                    "alpha": float(job["alpha"]),
                    "fine_type_lambda": float(job["fine_type_lambda"]),
                }
            )
    else:
        project_sampler.setdefault("mode", args.sampler_mode)
        project_sampler.setdefault("alpha", args.alpha)
        project_sampler.setdefault("fine_type_lambda", args.fine_type_lambda)
    parameters["project_sampler"] = project_sampler
    parameters.setdefault("sampler", None)
    if model_type in {"Global", "Global_LoRA"}:
        if args.base_model_dir is None:
            raise ValueError("Global TRIPSO training requires --base-model-dir")
        parameters["path_to_base_model"] = str(args.base_model_dir.resolve())
    spec = TripsoTrainingSpec(
        fold_input_manifest_path=fold_input,
        output_dir=output_dir,
        model_type=model_type,
        seed=args.seed,
        parameters=parameters,
        dry_run=False,
    )
    call, invocation = build_training_call(spec)
    return spec, invocation, output_dir, job


def command_train_tripso(args: argparse.Namespace) -> int:
    _configured(args)
    vendor_root = args.vendor_root.resolve()
    for path in (vendor_root / "setup.py", vendor_root / "requirements.txt"):
        if not path.is_file():
            raise FileNotFoundError(f"Vendored TRIPSO resource is missing: {path}")
    spec, invocation, output_dir, job = _resolve_training_request(args)
    fold = read_json(spec.fold_input_manifest_path)
    if args.environment_report is not None:
        environment = read_json(args.environment_report)
        if not environment.get("environment_passed", False):
            raise RuntimeError(
                "TRIPSO environment report does not pass; real training is blocked"
            )
    effective_sampler = invocation.get("project_sampler") or {}
    if effective_sampler.get("enabled", False):
        if not 0 <= float(effective_sampler["alpha"]) <= 1:
            raise ValueError("TRIPSO sampler alpha must be between 0 and 1")
        if not 0 <= float(effective_sampler["fine_type_lambda"]) <= 1:
            raise ValueError("TRIPSO fine-type lambda must be between 0 and 1")
    if args.dry_run:
        print(
            json_plan(
                "train-tripso",
                vendor_root=vendor_root,
                fold_input=spec.fold_input_manifest_path.resolve(),
                output_dir=output_dir.resolve(),
                model_type=spec.model_type,
                seed=spec.seed,
                invocation=invocation,
                note=(
                    "No TRIPSO import, GPU allocation, or model training was performed"
                ),
            )
        )
        return 0
    checkpoint = output_dir / "checkpoints" / "last.ckpt"
    model_manifest_path = output_dir / "model_manifest.json"
    guard_outputs((checkpoint, model_manifest_path), overwrite=args.overwrite)
    with _vendor_on_path(vendor_root):
        run_tripso_training(spec)
    training_result = read_json(output_dir / "tripso_training_result.json")
    sampler_enabled = bool(effective_sampler.get("enabled", False))
    sampler_mode = (
        str(job.get("sampler_mode") or effective_sampler.get("mode", args.sampler_mode))
        if sampler_enabled
        else str(job.get("sampler_mode") or "native_all_cells")
    )
    alpha = float(effective_sampler["alpha"]) if sampler_enabled else None
    fine_type_lambda = (
        float(effective_sampler["fine_type_lambda"]) if sampler_enabled else None
    )
    inputs = fold["inputs"]
    vendor_call = invocation["vendor_call"]
    gene_embedding_dimension = (vendor_call.get("bert_config") or {}).get("hidden_size")
    if gene_embedding_dimension is None:
        gene_embedding_dimension = (
            (invocation.get("geneformer_validation") or {}).get("config") or {}
        ).get("hidden_size")
    gp_latent_dimension = vendor_call.get("gp_latent_size")
    if gp_latent_dimension is None:
        gp_latent_dimension = gene_embedding_dimension or args.embedding_dimension
    geneformer_validation = invocation.get("geneformer_validation")
    geneformer_identity: dict[str, Any] | None = None
    geneformer_asset_hashes: dict[str, str] = {}
    if isinstance(geneformer_validation, Mapping) and geneformer_validation.get(
        "passed"
    ):
        geneformer_identity = {
            "model_name": geneformer_validation.get("model_name"),
            "source_revision": geneformer_validation.get("source_revision"),
            "requested_source_revision": geneformer_validation.get(
                "requested_source_revision"
            ),
            "config": geneformer_validation.get("config"),
            "hashes": geneformer_validation.get("hashes"),
            "hashes_pinned": geneformer_validation.get("hashes_pinned"),
        }
        geneformer_asset_hashes = {
            f"geneformer_asset:{name}": str(value)
            for name, value in (geneformer_validation.get("hashes") or {}).items()
        }
    build_model_artifact_manifest(
        output_path=model_manifest_path,
        repo_root=REPO_ROOT,
        vendor_root=vendor_root,
        fold_input_manifest_path=spec.fold_input_manifest_path,
        checkpoint_path=checkpoint,
        fold_id=str(fold["fold_id"]),
        held_out_dataset=(
            None
            if fold.get("held_out_dataset") is None
            else str(fold["held_out_dataset"])
        ),
        lineage=str(fold["lineage"]),
        model_type=spec.model_type,
        sampler_mode=sampler_mode,
        alpha=alpha,
        fine_type_lambda=fine_type_lambda,
        seed=args.seed,
        gp_library_path=Path(inputs["gp_library_path"]),
        gene_vocabulary_path=Path(inputs["gene_vocabulary_path"]),
        training_metrics=(
            training_result.get("experiment_tracking", {}).get("metrics", {})
        ),
        model_configuration={
            "tokenizer": args.tokenizer,
            "preprocessing": args.preprocessing,
            "embedding_dimension": gp_latent_dimension,
            "gp_latent_dimension": gp_latent_dimension,
            "gene_embedding_dimension": gene_embedding_dimension,
            "model_type": spec.model_type,
            "vendor_call": vendor_call,
            "project_sampler": effective_sampler,
            "sampling_backend": invocation["training_sampler_backend"],
            "experiment_tracking": training_result.get("experiment_tracking", {}),
            "feature_set": job.get("feature_set"),
            "hvg_size": job.get("hvg_size"),
            "includes_all_retained_gp_genes": job.get("includes_all_retained_gp_genes"),
            "geneformer_identity": geneformer_identity,
        },
        asset_hashes={
            f"vendor_asset:{relative_name}": tripso_sha256_path(
                vendor_root / relative_name
            )
            for relative_name in REQUIRED_VENDOR_ASSETS
        }
        | {
            **geneformer_asset_hashes,
        },
    )
    return 0


def command_project_tripso(args: argparse.Namespace) -> int:
    _configured(args)
    query_manifest = read_json(args.query_manifest)
    model = validate_frozen_query_resources(
        model_manifest_path=args.model_manifest,
        query_manifest=query_manifest,
        vendor_root=args.vendor_root,
    )
    if args.batch_size < 1:
        raise ValueError("projection batch size must be positive")
    if args.dry_run:
        print(
            json_plan(
                "project-tripso",
                model_manifest=args.model_manifest.resolve(),
                query_manifest=args.query_manifest.resolve(),
                output_dir=args.output_dir.resolve(),
                frozen_hashes=model["hashes"],
                optimizer_updates=False,
                adaptation=False,
                note="Resources were hash-validated; TRIPSO was not imported",
            )
        )
        return 0
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            "Projection data output already exists; use a new path or pass "
            f"--overwrite: {output_dir}"
        )
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.with_name(f".{output_dir.name}.backup.{os.getpid()}")
        if backup.exists():
            raise FileExistsError(f"Projection backup path already exists: {backup}")
        os.replace(output_dir, backup)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial.", dir=output_dir.parent)
    )
    try:
        with _vendor_on_path(args.vendor_root):
            run_vendor_frozen_projection(
                model_manifest_path=args.model_manifest,
                query_manifest=query_manifest,
                output_dir=staging,
                batch_size=args.batch_size,
                precision=args.precision,
                vendor_root=args.vendor_root,
                projection_manifest_path=args.query_manifest,
            )
        os.replace(staging, output_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    return 0


def _fine_type_universe(path: Path | None) -> Any:
    if path is None:
        return None
    value = (
        load_fine_type_ontology(path, require_approved=True)
        if path.suffix.lower() in {".yaml", ".yml"}
        else read_json(path)
    )
    if "lineages" in value:
        return {
            lineage: list(
                dict.fromkeys(
                    str(item["canonical_fine_type"])
                    for item in settings.get("mappings", [])
                    if bool(
                        item.get(
                            "state_eligible",
                            item.get("eligible_for_state_by_total_cells", True),
                        )
                    )
                )
            )
            for lineage, settings in value["lineages"].items()
        }
    return value


def command_aggregate_distributions(args: argparse.Namespace) -> int:
    config = _configured(args, DEFAULT_REFERENCE_CONFIG)
    header = read_table(args.metadata, nrows=0)
    fields = config.get("metadata_fields", {})
    normalized_header = _normalise_cell_metadata(header, config)
    require_columns(
        normalized_header,
        (
            "dataset",
            "donor_id",
            "sample_id",
            "age",
            "sex",
            "lineage",
            "fine_type",
            "fine_type_state_eligible",
        ),
        "aggregation metadata",
    )
    if not args.embeddings.is_file():
        raise FileNotFoundError(f"Embedding array is missing: {args.embeddings}")
    if args.min_state_cells < 2 or args.min_empirical_cells < args.min_state_cells:
        raise ValueError(
            "aggregation thresholds require 2 <= min-state <= min-empirical"
        )
    conversion_validation = validate_arrow_conversion_for_aggregation(
        args.arrow_conversion_manifest,
        args.embeddings,
        args.metadata,
        embedding_column=args.gp_id,
    )
    projection_validation = conversion_validation["projection_output"]
    output_table = args.output_dir / "fine_type_distributions.parquet"
    output_groups = args.output_dir / "empirical_distribution_groups.parquet"
    output_rows = args.output_dir / "empirical_distribution_rows.npy"
    output_index = args.output_dir / "empirical_distribution_manifest.json"
    if args.dry_run:
        print(
            json_plan(
                "aggregate-donor-distributions",
                embeddings=args.embeddings.resolve(),
                metadata=args.metadata.resolve(),
                arrow_conversion_manifest=args.arrow_conversion_manifest.resolve(),
                projection_output_manifest=projection_validation["manifest_path"],
                projection_role=projection_validation["projection_role"],
                reference_design=projection_validation["reference_design"],
                heldout_dataset=projection_validation["heldout_dataset"],
                gp_id=args.gp_id,
                output_dir=args.output_dir.resolve(),
                min_state_cells=args.min_state_cells,
                min_empirical_cells=args.min_empirical_cells,
                unit="dataset x donor x observation x lineage x fine_type x GP",
                empirical_storage="mmap converted NPY plus int64 row gather",
            )
        )
        return 0
    guard_outputs(
        (output_table, output_groups, output_rows, output_index),
        overwrite=args.overwrite,
    )
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    metadata = _normalise_cell_metadata(read_table(args.metadata), config)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Embedding rows and cell metadata rows differ")
    age_direction = (
        None
        if args.age_direction is None
        else np.load(args.age_direction, allow_pickle=False)
    )
    confidence_col = fields.get("fine_type_confidence", "ctype_low_conf")
    result = aggregate_fine_type_distributions(
        embeddings,
        metadata,
        gp_id=args.gp_id,
        fine_type_universe=_fine_type_universe(args.fine_type_universe),
        min_state_cells=args.min_state_cells,
        min_empirical_cells=args.min_empirical_cells,
        robust_location=not args.use_mean_location,
        age_direction=age_direction,
        annotation_confidence_col=(
            confidence_col if confidence_col in metadata else None
        ),
        provenance={
            "model_id": projection_validation["model_manifest_sha256"],
            "model_manifest": projection_validation["model_manifest"],
            "model_manifest_sha256": projection_validation["model_manifest_sha256"],
            "checkpoint_sha256": projection_validation["checkpoint_sha256"],
            "fold_id": projection_validation["fold_id"],
            "seed": projection_validation["seed"],
            "projection_role": projection_validation["projection_role"],
            "eligible_for_model_selection": projection_validation[
                "eligible_for_model_selection"
            ],
            "outer_query_evaluation_only": projection_validation[
                "outer_query_evaluation_only"
            ],
            "reference_design": projection_validation["reference_design"],
            "heldout_dataset": projection_validation["heldout_dataset"],
            "projection_output_manifest": projection_validation["manifest_path"],
            "projection_output_manifest_sha256": projection_validation[
                "manifest_sha256"
            ],
            "projection_arrow_tree_sha256": projection_validation["arrow_tree_sha256"],
            "arrow_conversion_manifest": conversion_validation["manifest_path"],
            "arrow_conversion_manifest_sha256": conversion_validation[
                "manifest_sha256"
            ],
            "arrow_cell_key_ordered_sha256": conversion_validation[
                "cell_key_ordered_sha256"
            ],
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(result.table, output_table)
    write_empirical_row_index(
        args.output_dir,
        metadata,
        result.distribution_rows,
        result.empirical_distance_keys,
        conversion_validation=conversion_validation,
        aggregation_table_path=output_table,
        overwrite=args.overwrite,
    )
    _write_command_manifest(
        args.output_dir,
        stage="donor_distribution_aggregation",
        outputs=(output_table, output_groups, output_rows, output_index),
        args=args,
        extra={
            "arrow_conversion_validation": conversion_validation,
            "fine_type_distribution_table": {
                "path": str(output_table.resolve()),
                "sha256": sha256_file(output_table),
            },
            "fine_type_ontology": (
                {
                    "path": str(args.fine_type_universe.resolve()),
                    "sha256": sha256_file(args.fine_type_universe),
                    "state_eligible_types_only_in_frozen_universe": True,
                    "special_categories_retained_for_composition_only": True,
                }
                if args.fine_type_universe is not None
                else None
            ),
        },
    )
    return 0


def _feature_matrix(
    table: pd.DataFrame,
    path: Path | None,
    columns: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if (path is None) == (not columns):
        raise ValueError("Provide exactly one of --features or --feature-column")
    if path is not None:
        values = load_matrix(path)
        if sparse.issparse(values):
            values = values.toarray()
        labels = tuple(f"feature_{index}" for index in range(values.shape[1]))
        return np.asarray(values, dtype=float), labels
    assert columns is not None
    require_columns(table, columns, "feature table")
    return table.loc[:, columns].to_numpy(dtype=float), tuple(columns)


def command_assemble_donor_gp_endpoint(args: argparse.Namespace) -> int:
    """Materialize one role-aware donor GP endpoint from aggregation output."""

    _configured(args)
    header = read_table(args.aggregate_table, nrows=0)
    require_columns(
        header,
        (
            "dataset",
            "donor_id",
            "sample_id",
            "observation_id",
            "age",
            "sex",
            "lineage",
            "fine_type",
            "gp_id",
            "state_available",
            "location_summary",
            "covariance_summary",
        ),
        "aggregated endpoint table",
    )
    if not args.projection_output_manifest.is_file():
        raise FileNotFoundError(
            f"Projection output manifest is missing: {args.projection_output_manifest}"
        )
    if not args.aggregation_manifest.is_file():
        raise FileNotFoundError(
            f"Aggregation manifest is missing: {args.aggregation_manifest}"
        )
    if args.dry_run:
        projection = read_json(args.projection_output_manifest)
        print(
            json_plan(
                "assemble-donor-gp-endpoint",
                aggregate_table=args.aggregate_table.resolve(),
                aggregation_manifest=args.aggregation_manifest.resolve(),
                projection_output_manifest=(args.projection_output_manifest.resolve()),
                projection_role=projection.get("projection_role"),
                reference_design=projection.get("reference_design"),
                heldout_dataset=projection.get("heldout_dataset"),
                lineage=args.lineage,
                fine_type=args.fine_type,
                gp_id=args.gp_id,
                output_dir=args.output_dir.resolve(),
                selector="exact lineage x fine_type x gp_id",
            )
        )
        return 0
    assemble_donor_gp_endpoint(
        args.aggregate_table,
        args.aggregation_manifest,
        args.projection_output_manifest,
        args.output_dir,
        lineage=args.lineage,
        fine_type=args.fine_type,
        gp_id=args.gp_id,
        overwrite=args.overwrite,
    )
    return 0


def _validate_endpoint_metadata_uniqueness(
    metadata: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Reject accidental mixtures of separately modelled endpoint identities."""

    if "location_summary" in metadata:
        raise ValueError(
            f"{label} still contains JSON location_summary values; run "
            "assemble-donor-gp-endpoint and supply its manifest-bound NPY"
        )
    identity_columns = ("lineage", "fine_type", "gp_id")
    present = [column for column in identity_columns if column in metadata]
    if present and len(present) != len(identity_columns):
        missing = sorted(set(identity_columns) - set(present))
        raise ValueError(f"{label} has an incomplete endpoint identity: {missing}")
    for column in present:
        values = metadata[column].dropna().astype(str).unique()
        if len(values) > 1:
            raise ValueError(f"{label} mixes multiple {column} values")


def _validate_endpoint_pair(
    reference: Mapping[str, Any], query: Mapping[str, Any]
) -> None:
    target_role = query.get("role")
    if reference.get("role") != "reference" or target_role not in {
        "validation",
        "query",
    }:
        raise ValueError(
            "Endpoint pair must contain reference then validation/query roles"
        )
    if query.get("eligible_for_model_selection") is not (target_role == "validation"):
        raise ValueError("Target endpoint model-selection eligibility is invalid")
    if query.get("outer_query_evaluation_only") is not (target_role == "query"):
        raise ValueError("Target endpoint outer-query evaluation flag is invalid")
    for field in ("reference_design", "heldout_dataset", "endpoint", "feature_ids"):
        if reference.get(field) != query.get(field):
            raise ValueError(f"Reference/query endpoint {field} differs")
    reference_source = reference.get("source_provenance", {})
    query_source = query.get("source_provenance", {})
    for field in ("model_id", "fold_id", "seed"):
        if field not in reference_source or field not in query_source:
            raise ValueError(
                f"Reference/query endpoint provenance lacks required {field}"
            )
        if str(reference_source[field]) != str(query_source[field]):
            raise ValueError(
                f"Reference/query endpoint model provenance differs: {field}"
            )


def _validate_feature_source(
    header: pd.DataFrame,
    path: Path | None,
    columns: Sequence[str] | None,
    *,
    label: str,
) -> None:
    if (path is None) == (not columns):
        raise ValueError("Provide exactly one of --features or --feature-column")
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"{label} feature matrix is missing: {path}")
    if columns:
        require_columns(header, columns, f"{label} feature table")


def command_fit_healthy_reference(args: argparse.Namespace) -> int:
    _configured(args)
    header = read_table(args.metadata, nrows=0)
    require_columns(
        header,
        ("dataset", "donor_id", "sample_id", "age", "sex"),
        "healthy-reference metadata",
    )
    _validate_feature_source(
        header,
        args.features,
        args.feature_column,
        label="healthy-reference",
    )
    _validate_endpoint_metadata_uniqueness(header, label="healthy-reference metadata")
    endpoint_validation: dict[str, Any] | None = None
    if args.endpoint_manifest is not None:
        if args.features is None or args.feature_column:
            raise ValueError(
                "--endpoint-manifest requires its exact --features NPY and forbids "
                "--feature-column"
            )
        endpoint_validation = validate_endpoint_inputs(
            args.endpoint_manifest,
            args.metadata,
            args.features,
            expected_role="reference",
        )
    elif args.features is not None and {
        "lineage",
        "fine_type",
        "gp_id",
    }.issubset(header.columns):
        raise ValueError("Endpoint-like external features require --endpoint-manifest")

    heldout_dataset = args.heldout_dataset
    if endpoint_validation is not None:
        manifest_heldout = endpoint_validation["heldout_dataset"]
        if heldout_dataset is not None and heldout_dataset != manifest_heldout:
            raise ValueError(
                "--heldout-dataset differs from the reference endpoint manifest"
            )
        if heldout_dataset is None:
            heldout_dataset = manifest_heldout
        if endpoint_validation["reference_design"] == "lodo":
            if args.final_all_healthy:
                raise ValueError(
                    "A LODO reference endpoint cannot use --final-all-healthy"
                )
        elif not args.final_all_healthy:
            raise ValueError(
                "An all_healthy reference endpoint requires --final-all-healthy"
            )
    if heldout_dataset is None and not args.final_all_healthy:
        raise ValueError(
            "Specify --heldout-dataset for LODO or --final-all-healthy explicitly"
        )
    if heldout_dataset is not None and args.final_all_healthy:
        raise ValueError(
            "LODO heldout and final-all-healthy modes are mutually exclusive"
        )
    if args.legacy_combined_lodo_input:
        if endpoint_validation is not None:
            raise ValueError(
                "--legacy-combined-lodo-input cannot be used with an endpoint manifest"
            )
        if heldout_dataset is None:
            raise ValueError("--legacy-combined-lodo-input requires --heldout-dataset")
    elif heldout_dataset is not None and endpoint_validation is None:
        raise ValueError(
            "Combined reference/query LODO metadata is legacy behavior; pass "
            "--legacy-combined-lodo-input explicitly, or use a role=reference "
            "endpoint manifest containing adaptation rows only"
        )
    if args.n_inner_folds < 2 or args.age_grid_size < 2:
        raise ValueError("healthy-reference folds and age-grid size must be at least 2")
    if (
        args.age_kernel_bandwidth <= 0
        or args.age_kernel_minimum_exact_sex_donors < 1
        or args.age_support_window_years <= 0
        or args.minimum_support_cohorts < 1
        or args.minimum_support_donors < 1
        or args.slope_minimum_donors < 2
        or args.slope_minimum_age_span <= 0
    ):
        raise ValueError("healthy-reference support thresholds are invalid")
    outputs = (
        args.output_dir / "healthy_reference.json",
        args.output_dir / "healthy_reference_arrays.npz",
        args.output_dir / "training_crossfit_scores.parquet",
        args.output_dir / "age_support_grid.parquet",
        args.output_dir / "cohort_age_slope_diagnostics.parquet",
    )
    if endpoint_validation is not None:
        outputs += (
            args.output_dir / "age_kernel_reference.json",
            # Guard/remove a legacy duplicated covariance archive so an
            # overwritten run cannot leave a misleading multi-gigabyte file.
            args.output_dir / "age_kernel_reference_arrays.npz",
        )
    if args.dry_run:
        print(
            json_plan(
                "fit-healthy-reference",
                metadata=args.metadata.resolve(),
                features=args.features,
                feature_columns=args.feature_column,
                endpoint_manifest=args.endpoint_manifest,
                endpoint_validation=endpoint_validation,
                output_dir=args.output_dir.resolve(),
                heldout_dataset=heldout_dataset,
                final_all_healthy=args.final_all_healthy,
                input_composition=(
                    "legacy_combined_reference_and_query"
                    if args.legacy_combined_lodo_input
                    else "reference_only"
                ),
                cross_fitting="donor-grouped",
                weighting_scheme=args.weighting_scheme,
                distributional_reference=(
                    "paired_age_kernel_gaussian_wasserstein"
                    if endpoint_validation is not None
                    else "unavailable_without_endpoint_covariances"
                ),
                age_kernel_bandwidth=args.age_kernel_bandwidth,
                query_dataset_offset="forbidden",
            )
        )
        return 0
    guard_outputs(outputs, overwrite=args.overwrite)
    if endpoint_validation is not None and args.overwrite:
        (args.output_dir / "age_kernel_reference_arrays.npz").unlink(missing_ok=True)
    metadata = ensure_donor_observation_ids(read_table(args.metadata))
    _validate_endpoint_metadata_uniqueness(metadata, label="healthy-reference metadata")
    features, feature_ids = _feature_matrix(
        metadata, args.features, args.feature_column
    )
    if endpoint_validation is not None:
        feature_ids = tuple(endpoint_validation["feature_ids"])
    if len(features) != len(metadata):
        raise ValueError("Healthy-reference features and metadata rows differ")
    if args.legacy_combined_lodo_input:
        assert heldout_dataset is not None
        query_mask = metadata["dataset"].astype(str).eq(heldout_dataset)
        if not query_mask.any():
            raise ValueError(
                f"Held-out dataset {heldout_dataset!r} is absent from metadata"
            )
        if query_mask.all():
            raise ValueError("Legacy combined LODO metadata has no adaptation rows")
        training_mask = ~query_mask.to_numpy()
    else:
        training_mask = np.ones(len(metadata), dtype=bool)
        if (
            heldout_dataset is not None
            and metadata["dataset"].astype(str).eq(heldout_dataset).any()
        ):
            raise ValueError(
                "Reference-only LODO input physically contains the held-out dataset"
            )
    training = metadata.loc[training_mask].reset_index(drop=True)
    training_features = features[training_mask]
    model_kwargs = {
        "n_spline_knots": args.n_spline_knots,
        "ridge": args.ridge,
        "age_grid_size": args.age_grid_size,
        "weighting_scheme": args.weighting_scheme,
    }
    crossfit = cross_fit_trajectory(
        training_features,
        training["age"],
        training["sex"],
        training["biological_unit_id"],
        datasets=training["dataset"],
        n_splits=args.n_inner_folds,
        model_kwargs=model_kwargs,
    )
    crossfit_scores = crossfit.scores.rename(columns={"fold_id": "inner_crossfit_fold"})
    crossfit_table = training.reset_index(names="row_index").merge(
        crossfit_scores, on="row_index", validate="one_to_one"
    )
    model = HealthyTrajectory(**model_kwargs).fit(
        training_features,
        training["age"],
        training["sex"],
        training["biological_unit_id"],
        datasets=training["dataset"],
    )
    support = age_support_grid(
        training["age"],
        training["sex"],
        training["dataset"],
        training["biological_unit_id"],
        window_years=args.age_support_window_years,
        minimum_cohorts=args.minimum_support_cohorts,
        minimum_donors=args.minimum_support_donors,
        weighting_scheme=args.weighting_scheme,
    )
    slope_diagnostics = cohort_feature_age_effects(
        training_features,
        training["age"],
        training["sex"],
        training["biological_unit_id"],
        training["dataset"],
        feature_ids=feature_ids,
        minimum_donors=args.slope_minimum_donors,
        minimum_age_span=args.slope_minimum_age_span,
    )
    distributional_reference: dict[str, Any] = {
        "status": "not_available",
        "reason": "fit input has no manifest-bound donor covariance endpoint",
    }
    if endpoint_validation is not None:
        if not training_mask.all():
            raise ValueError(
                "Distributional reference fitting requires the physically separate "
                "role=reference endpoint, not a row-filtered combined table"
            )
        assert args.features is not None
        endpoint_locations = np.load(args.features, mmap_mode="r", allow_pickle=False)
        endpoint_covariances = np.load(
            endpoint_validation["covariances_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        kernel = AgeKernelReference(
            bandwidth=args.age_kernel_bandwidth,
            minimum_exact_sex_donors=args.age_kernel_minimum_exact_sex_donors,
            weighting_scheme=args.weighting_scheme,
            age_grid_size=args.age_grid_size,
        ).fit(
            endpoint_locations,
            endpoint_covariances,
            training["age"],
            training["sex"],
            training["biological_unit_id"],
            datasets=training["dataset"],
        )
        kernel_manifest_path = write_age_kernel_reference(
            kernel,
            args.output_dir,
            metadata={
                "identifier_contract": IDENTIFIER_CONTRACT,
                "feature_ids": list(feature_ids),
                "endpoint_artifact": endpoint_validation,
                "heldout_dataset": heldout_dataset,
                "final_all_healthy": args.final_all_healthy,
                "distance_metric": "gaussian_2_wasserstein_bures",
                "weighting_scheme": args.weighting_scheme,
                "empirical_sliced_wasserstein": {
                    "status": "not_fitted",
                    "reason": (
                        "moment endpoint does not itself provide the separately "
                        "validated donor row-index/cell-embedding artifact"
                    ),
                },
            },
        )
        distributional_reference = {
            "status": "available",
            "model_class": "AgeKernelReference",
            "manifest_path": kernel_manifest_path.name,
            "manifest_sha256": sha256_file(kernel_manifest_path),
            "endpoint_locations_reused": True,
            "endpoint_covariances_reused": True,
            "copied_covariance_archive_written": False,
            "distance_column": "age_matched_gaussian_wasserstein_distance",
            "off_trajectory_distance_column": (
                "off_trajectory_gaussian_wasserstein_distance"
            ),
            "predicted_age_column": "predicted_distributional_gp_age",
            "weighting_scheme": args.weighting_scheme,
        }
    # Write diagnostic tables first so the frozen reference can bind the exact
    # cross-fitted scores used by downstream training-only GP selection.  The
    # selector refuses an unbound or modified score table.
    write_table(crossfit_table, outputs[2])
    write_table(support, outputs[3])
    write_table(slope_diagnostics, outputs[4])
    write_reference(
        model,
        args.output_dir,
        metadata={
            "identifier_contract": IDENTIFIER_CONTRACT,
            "feature_ids": list(feature_ids),
            "heldout_dataset": heldout_dataset,
            "final_all_healthy": args.final_all_healthy,
            "input_composition": (
                "legacy_combined_reference_and_query"
                if args.legacy_combined_lodo_input
                else "reference_only"
            ),
            "endpoint_artifact": endpoint_validation,
            "distributional_reference": distributional_reference,
            "n_training_rows": len(training),
            "n_training_biological_units": int(
                training["biological_unit_id"].nunique()
            ),
            "training_datasets": sorted(training["dataset"].astype(str).unique()),
            "seed": args.seed,
            "query_dataset_offset": "forbidden",
            "weighting_scheme": args.weighting_scheme,
            "age_support": {
                "window_years": args.age_support_window_years,
                "minimum_cohorts": args.minimum_support_cohorts,
                "minimum_donors": args.minimum_support_donors,
                "grid_path": outputs[3].name,
            },
            "cohort_age_slope_diagnostics_path": outputs[4].name,
            "training_crossfit_scores": {
                "schema_version": "immune-health-training-crossfit-scores/v1",
                "path": outputs[2].name,
                "sha256": sha256_file(outputs[2]),
                "score_column": "predicted_gp_age",
                "acceleration_column": "gp_age_acceleration",
                "fold_column": "inner_crossfit_fold",
                "fit_scope": "donor_grouped_training_only",
                "query_data_consulted": False,
            },
        },
    )
    return 0


def _selection_reference_manifests(args: argparse.Namespace) -> tuple[Path, ...]:
    """Collect repeat arguments and one-column/TSV manifest lists deterministically."""

    paths = [Path(path).resolve() for path in args.reference_manifest]
    for list_path_value in args.reference_manifest_list:
        list_path = Path(list_path_value).resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"Reference-manifest list is missing: {list_path}")
        lines = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            raise ValueError(f"Reference-manifest list is empty: {list_path}")
        delimiter = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
        header = lines[0].split(delimiter) if delimiter is not None else [lines[0]]
        if "reference_manifest" in header:
            column = header.index("reference_manifest")
            values = []
            for line in lines[1:]:
                fields = line.split(delimiter)
                if column >= len(fields) or not fields[column].strip():
                    raise ValueError(
                        f"Malformed reference_manifest row in {list_path}: {line!r}"
                    )
                values.append(fields[column].strip())
        else:
            if delimiter is not None:
                raise ValueError(
                    f"Delimited list {list_path} needs a reference_manifest column"
                )
            values = lines
        for value in values:
            candidate = Path(value)
            paths.append(
                (
                    candidate
                    if candidate.is_absolute()
                    else list_path.parent / candidate
                ).resolve()
            )
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise ValueError(
            "Provide --reference-manifest and/or --reference-manifest-list"
        )
    return unique


def _tripso_gp_selection_config(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> TripsoGPSelectionConfig:
    section = config.get("tripso_gp_selection", {})
    if not isinstance(section, Mapping):
        raise ValueError("tripso_gp_selection configuration must be a mapping")
    defaults = TripsoGPSelectionConfig()

    def choose(name: str) -> Any:
        value = getattr(args, name)
        return section.get(name, getattr(defaults, name)) if value is None else value

    return TripsoGPSelectionConfig(
        **{name: choose(name) for name in TripsoGPSelectionConfig.__dataclass_fields__}
    )


def command_select_transferable_tripso_gps(args: argparse.Namespace) -> int:
    """Freeze the final fine-type/GP set from training cross-fit scores only."""

    config = _configured(args)
    section = config.get("tripso_gp_selection", {})
    if not isinstance(section, Mapping):
        raise ValueError("tripso_gp_selection configuration must be a mapping")
    training_datasets = tuple(
        args.required_training_dataset or section.get("required_training_datasets", ())
    )
    required_seeds = tuple(args.required_seed or section.get("required_seeds", ()))
    if not training_datasets:
        raise ValueError(
            "Declare every adaptation cohort with --required-training-dataset or config"
        )
    if not required_seeds:
        raise ValueError("Declare every seed with --required-seed or config")
    selection_config = _tripso_gp_selection_config(args, config)
    manifests = _selection_reference_manifests(args)
    result = select_transferable_tripso_gps(
        manifests,
        lineage=args.lineage,
        fold_id=args.fold_id,
        heldout_dataset=args.heldout_dataset,
        training_datasets=training_datasets,
        required_seeds=required_seeds,
        weighting_scheme=args.weighting_scheme,
        config=selection_config,
        simple_baseline=args.simple_baseline,
        simple_baseline_score_column=args.simple_baseline_score_column,
    )
    if args.dry_run:
        print(
            json_plan(
                "select-transferable-tripso-gps",
                lineage=args.lineage,
                fold_id=args.fold_id,
                heldout_dataset=args.heldout_dataset,
                training_datasets=sorted(set(map(str, training_datasets))),
                required_seeds=sorted(set(map(int, required_seeds))),
                weighting_scheme=args.weighting_scheme,
                n_reference_manifests=len(manifests),
                n_candidates=len(result.selection),
                n_selected=int(result.selection["retained"].sum()),
                validation=(
                    "All endpoint/model/checkpoint/cross-fit hashes and training-only "
                    "cohort/seed contracts passed; outputs were not written"
                ),
            )
        )
        return 0
    write_tripso_gp_selection(result, args.output_dir, overwrite=args.overwrite)
    return 0


def command_score_query(args: argparse.Namespace) -> int:
    _configured(args)
    model, reference_manifest = load_reference(args.reference_manifest)
    query_header = read_table(args.query_metadata, nrows=0)
    require_columns(
        query_header,
        ("dataset", "donor_id", "sample_id", "age", "sex"),
        "query metadata",
    )
    _validate_feature_source(
        query_header, args.features, args.feature_column, label="query"
    )
    _validate_endpoint_metadata_uniqueness(query_header, label="query metadata")
    reference_endpoint = reference_manifest.get("endpoint_artifact")
    endpoint_validation: dict[str, Any] | None = None
    if args.endpoint_manifest is not None:
        if args.features is None or args.feature_column:
            raise ValueError(
                "--endpoint-manifest requires its exact --features NPY and forbids "
                "--feature-column"
            )
        endpoint_validation = validate_endpoint_inputs(
            args.endpoint_manifest,
            args.query_metadata,
            args.features,
            expected_role=None,
        )
        if endpoint_validation["role"] not in {"validation", "query"}:
            raise ValueError("Scoring endpoint must have role validation or query")
    elif args.features is not None and {
        "lineage",
        "fine_type",
        "gp_id",
    }.issubset(query_header.columns):
        raise ValueError("Endpoint-like query features require --endpoint-manifest")
    if isinstance(reference_endpoint, Mapping):
        if endpoint_validation is None:
            raise ValueError(
                "This frozen reference is endpoint-backed; score-query requires "
                "the corresponding role=validation/query --endpoint-manifest"
            )
        _validate_endpoint_pair(reference_endpoint, endpoint_validation)
    age_kernel_model: AgeKernelReference | None = None
    age_kernel_manifest: dict[str, Any] | None = None
    distributional_record = reference_manifest.get("distributional_reference")
    if isinstance(reference_endpoint, Mapping):
        if not isinstance(distributional_record, Mapping) or (
            distributional_record.get("status") != "available"
        ):
            raise ValueError(
                "Endpoint-backed reference lacks its paired AgeKernelReference"
            )
        age_kernel_path = args.reference_manifest.parent / str(
            distributional_record.get("manifest_path", "")
        )
        if not age_kernel_path.is_file() or sha256_file(
            age_kernel_path
        ) != distributional_record.get("manifest_sha256"):
            raise ValueError(
                "Paired age-kernel reference is missing or differs from the "
                "location-reference binding"
            )
        age_kernel_model, age_kernel_manifest = load_age_kernel_reference(
            age_kernel_path
        )
        if age_kernel_manifest.get("endpoint_artifact") != reference_endpoint:
            raise ValueError(
                "Age-kernel and location references bind different endpoints"
            )
        if age_kernel_model.weighting_scheme != model.weighting_scheme:
            raise ValueError(
                "Age-kernel and location references use different weighting schemes"
            )
    for name, value in (
        ("minimum_gene_coverage", args.minimum_gene_coverage),
        ("minimum_gp_coverage", args.minimum_gp_coverage),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if (
        args.age_support_window_years <= 0
        or args.minimum_support_cohorts < 1
        or args.minimum_support_donors < 1
    ):
        raise ValueError("query age-support thresholds must be positive")
    query_genes = read_gene_ids(args.query_genes)
    frozen_genes = read_gene_ids(args.frozen_vocabulary)
    coverage = require_gene_coverage(
        query_genes,
        frozen_genes,
        args.minimum_gene_coverage,
        allow_low_coverage=args.allow_low_coverage,
    )
    query_resource_manifest: dict[str, Any] | None = None
    if args.query_manifest is not None:
        query_resource_manifest = read_json(args.query_manifest)
    gp_coverage = args.gp_coverage
    if gp_coverage is None and query_resource_manifest is not None:
        candidate = query_resource_manifest.get("gp_coverage")
        if candidate is not None:
            gp_coverage = float(candidate)
    if gp_coverage is None:
        raise ValueError(
            "score-query requires --gp-coverage (or gp_coverage in the query "
            "manifest) so the frozen safety threshold can be enforced"
        )
    if gp_coverage < args.minimum_gp_coverage:
        if not args.allow_low_coverage:
            raise ValueError(
                "Query GP coverage is below the safety threshold: "
                f"{gp_coverage:.3f} < {args.minimum_gp_coverage:.3f}"
            )
    if (args.model_manifest is None) != (args.query_manifest is None):
        raise ValueError(
            "--model-manifest and --query-manifest must be supplied together"
        )
    if args.model_manifest is not None:
        validate_frozen_query_resources(
            model_manifest_path=args.model_manifest,
            query_manifest=query_resource_manifest,
        )
    if args.raw_counts is not None:
        if args.raw_metadata is None:
            raise ValueError("--raw-counts requires matching --raw-metadata")
        if not args.raw_counts.is_file():
            raise FileNotFoundError(f"Raw query counts are missing: {args.raw_counts}")
        raw_header = read_table(args.raw_metadata, nrows=0)
        require_columns(
            raw_header,
            (
                "dataset",
                "donor_id",
                "sample_id",
                "age",
                "sex",
                "lineage",
                "ctype_low",
                "ctype_low_conf",
            ),
            "raw query metadata",
        )
    if args.dry_run:
        print(
            json_plan(
                "score-query",
                reference_manifest=args.reference_manifest.resolve(),
                query_metadata=args.query_metadata.resolve(),
                features=args.features,
                feature_columns=args.feature_column,
                endpoint_manifest=args.endpoint_manifest,
                endpoint_validation=endpoint_validation,
                scoring_role=(
                    endpoint_validation["role"]
                    if endpoint_validation is not None
                    else "query"
                ),
                distributional_reference=distributional_record,
                output=args.output.resolve(),
                gene_coverage=coverage,
                gp_coverage=gp_coverage,
                frozen_projection_validated=args.model_manifest is not None,
                query_dataset_offset="forbidden",
                age_support_window_years=args.age_support_window_years,
                minimum_support_cohorts=args.minimum_support_cohorts,
                minimum_support_donors=args.minimum_support_donors,
            )
        )
        return 0
    guard_outputs((args.output, args.report), overwrite=args.overwrite)
    query = ensure_donor_observation_ids(read_table(args.query_metadata))
    _validate_endpoint_metadata_uniqueness(query, label="query metadata")
    features, feature_ids = _feature_matrix(query, args.features, args.feature_column)
    if endpoint_validation is not None:
        feature_ids = tuple(endpoint_validation["feature_ids"])
    if len(features) != len(query):
        raise ValueError("Query feature rows and query metadata rows differ")
    expected_features = tuple(map(str, reference_manifest.get("feature_ids", ())))
    if expected_features and feature_ids != expected_features:
        raise ValueError(
            "Query feature order differs from the frozen healthy-reference features"
        )
    overlap = set(query["biological_unit_id"].astype(str)) & set(
        model.training_biological_units_
    )
    if overlap:
        raise ValueError(
            f"Query contains healthy-reference training donors: {sorted(overlap)[:5]}"
        )
    if age_kernel_model is not None:
        assert age_kernel_manifest is not None
        if tuple(map(str, age_kernel_manifest.get("feature_ids", ()))) != feature_ids:
            raise ValueError(
                "Query feature order differs from the paired age-kernel reference"
            )
        if age_kernel_model.training_biological_units_ != (
            model.training_biological_units_
        ):
            raise ValueError(
                "Age-kernel and location references use different training donors"
            )
    raw_validation: dict[str, Any] | None = None
    if args.raw_counts is not None:
        raw_counts = load_matrix(args.raw_counts, require_sparse=True)
        raw_metadata, raw_id_report = validate_cell_metadata(
            read_table(args.raw_metadata)
        )
        if len(raw_metadata) != raw_counts.shape[0]:
            raise ValueError("Raw query count rows and raw metadata rows differ")
        if raw_counts.shape[1] != len(query_genes):
            raise ValueError("Raw query count width differs from query gene list")
        raw_validation = {
            "counts": vars(validate_raw_counts(raw_counts)),
            "metadata": raw_id_report,
        }
    query_covariances: np.ndarray | None = None
    if age_kernel_model is not None:
        assert endpoint_validation is not None
        query_covariances = np.load(
            endpoint_validation["covariances_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        if query_covariances.shape != (
            len(query),
            features.shape[1],
            features.shape[1],
        ):
            raise ValueError("Query endpoint covariance dimensions are inconsistent")
    scores: list[dict[str, Any]] = []
    for index, row in enumerate(query.itertuples(index=False)):
        score = model.score(features[index], row.age, str(row.sex), dataset=None)
        if age_kernel_model is not None:
            assert query_covariances is not None
            score.update(
                age_kernel_model.score_distribution(
                    features[index],
                    query_covariances[index],
                    row.age,
                    str(row.sex),
                )
            )
        scores.append(score)
    support_settings = reference_manifest.get("age_support", {})
    if all(
        getattr(model, name, None) is not None
        for name in (
            "training_ages_",
            "training_sexes_",
            "training_datasets_",
            "training_biological_unit_rows_",
        )
    ):
        support = query_age_support(
            model.training_ages_,
            model.training_sexes_,
            model.training_datasets_,
            model.training_biological_unit_rows_,
            query["age"],
            query["sex"],
            window_years=args.age_support_window_years,
            minimum_cohorts=args.minimum_support_cohorts,
            minimum_donors=args.minimum_support_donors,
            weighting_scheme=model.weighting_scheme,
        ).drop(columns="query_index")
    else:
        support = pd.DataFrame(
            {
                "age_support_status": np.repeat("not_available", len(query)),
                "age_support_window_years": args.age_support_window_years,
                "n_support_cohorts": pd.array([pd.NA] * len(query), dtype="Int64"),
                "support_cohorts": pd.NA,
                "n_support_donors": pd.array([pd.NA] * len(query), dtype="Int64"),
                "effective_support_weight": np.nan,
                "in_training_sex_age_range": pd.NA,
                "in_common_cohort_sex_age_range": pd.NA,
                "common_cohort_sex_age_min": np.nan,
                "common_cohort_sex_age_max": np.nan,
            }
        )
    output = pd.concat(
        [query.reset_index(drop=True), pd.DataFrame(scores), support], axis=1
    )
    write_table(output, args.output)
    ages = pd.to_numeric(query["age"], errors="raise")
    outside = ~ages.between(*model.age_range_)
    report = {
        "schema_version": "immune-health-query-score/v1",
        "identifier_contract": IDENTIFIER_CONTRACT,
        "reference_manifest": str(args.reference_manifest.resolve()),
        "query_endpoint_artifact": endpoint_validation,
        "target_endpoint_artifact": endpoint_validation,
        "scoring_role": (
            endpoint_validation["role"] if endpoint_validation is not None else "query"
        ),
        "distributional_reference": distributional_record,
        "distributional_scoring": {
            "gaussian_2_wasserstein_bures": (
                "computed" if age_kernel_model is not None else "not_available"
            ),
            "age_matched_distance_column": (
                "age_matched_gaussian_wasserstein_distance"
            ),
            "off_trajectory_distance_column": (
                "off_trajectory_gaussian_wasserstein_distance"
            ),
            "predicted_age_column": "predicted_distributional_gp_age",
            "age_acceleration_column": "distributional_gp_age_acceleration",
            "location_spline_residual_covariance_used": False,
            "empirical_sliced_wasserstein": {
                "status": "not_computed",
                "reason": (
                    "score-query was not supplied a separately manifest-validated "
                    "donor row-index/cell-embedding artifact"
                ),
            },
        },
        "n_rows": len(query),
        "n_biological_units": int(query["biological_unit_id"].nunique()),
        "query_datasets": sorted(query["dataset"].astype(str).unique()),
        "gene_coverage": coverage,
        "gp_coverage": gp_coverage,
        "minimum_gp_coverage": args.minimum_gp_coverage,
        "low_coverage_override": args.allow_low_coverage,
        "ood": {
            "training_age_range": list(model.age_range_),
            "rows_outside_training_age_range": int(outside.sum()),
            "fraction_outside_training_age_range": float(outside.mean()),
            "feature_dimension": features.shape[1],
        },
        "raw_input_validation": raw_validation or "not_supplied",
        "frozen_projection_resources_validated": args.model_manifest is not None,
        "query_dataset_offset": "not_fitted",
        "reference_weighting_scheme": model.weighting_scheme,
        "reference_age_support_settings": support_settings,
        "query_age_support_settings": {
            "window_years": args.age_support_window_years,
            "minimum_cohorts": args.minimum_support_cohorts,
            "minimum_donors": args.minimum_support_donors,
        },
        "query_age_support_status_counts": {
            str(key): int(value)
            for key, value in support["age_support_status"].value_counts().items()
        },
    }
    atomic_write_json(args.report, report)
    return 0


def command_score_empirical_endpoint(args: argparse.Namespace) -> int:
    """Run the exact role-paired matched-depth empirical sensitivity."""

    _configured(args)
    depths = args.depth or [25, 50, 100, 250, 500, 1000]
    inputs = (
        args.reference_endpoint_manifest,
        args.query_endpoint_manifest,
        args.reference_empirical_index,
        args.query_empirical_index,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Empirical scoring requires every endpoint/index artifact: "
            + ", ".join(missing)
        )
    if args.dry_run:
        print(
            json_plan(
                "score-empirical-endpoint",
                reference_endpoint_manifest=args.reference_endpoint_manifest.resolve(),
                query_endpoint_manifest=args.query_endpoint_manifest.resolve(),
                reference_empirical_index=args.reference_empirical_index.resolve(),
                query_empirical_index=args.query_empirical_index.resolve(),
                output_dir=args.output_dir.resolve(),
                depths=sorted(set(depths)),
                n_replicates=args.n_replicates,
                n_projections=args.n_projections,
                age_grid_size=args.age_grid_size,
                bandwidth=args.age_kernel_bandwidth,
                weighting_scheme=args.weighting_scheme,
                equal_query_reference_cell_depth=True,
                source_embedding_storage="validated memory map; no duplicate values",
                spline_residual_covariance_used=False,
            )
        )
        return 0
    score_empirical_matched_depth(
        reference_endpoint_manifest=args.reference_endpoint_manifest,
        query_endpoint_manifest=args.query_endpoint_manifest,
        reference_empirical_index_manifest=args.reference_empirical_index,
        query_empirical_index_manifest=args.query_empirical_index,
        output_dir=args.output_dir,
        depths=depths,
        n_replicates=args.n_replicates,
        n_projections=args.n_projections,
        age_grid_size=args.age_grid_size,
        bandwidth=args.age_kernel_bandwidth,
        minimum_exact_sex_donors=args.minimum_exact_sex_donors,
        weighting_scheme=args.weighting_scheme,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return 0


def _bootstrap_cell_scores(args: argparse.Namespace) -> pd.DataFrame:
    metadata = ensure_donor_observation_ids(read_table(args.metadata))
    if "fine_type" not in metadata and "ctype_low" in metadata:
        metadata["fine_type"] = metadata["ctype_low"]
    require_columns(metadata, ("observation_id", "fine_type"), "bootstrap metadata")
    features = load_matrix(args.features)
    if sparse.issparse(features):
        features = features.toarray()
    features = np.asarray(features, dtype=float)
    if len(features) != len(metadata):
        raise ValueError("Bootstrap feature rows and metadata rows differ")
    standardized = (
        None
        if args.standardized_proportions is None
        else {
            str(key): float(value)
            for key, value in read_json(args.standardized_proportions).items()
        }
    )
    rows: list[dict[str, Any]] = []
    grouping = [
        column
        for column in (
            "dataset",
            "donor_id",
            "biological_unit_id",
            "sample_id",
            "source_observation_id",
            "observation_id",
            "lineage",
            "gp_id",
        )
        if column in metadata
    ]
    for key, positions in metadata.groupby(
        grouping, observed=True, sort=True
    ).indices.items():
        selected = np.asarray(positions, dtype=int)
        subset = features[selected]
        labels = metadata.iloc[selected]["fine_type"].astype(str).to_numpy()
        estimates = fine_type_stratified_bootstrap(
            subset,
            labels,
            lambda values, _: values.mean(axis=0),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            mode=args.bootstrap_mode,
            standardized_proportions=standardized,
            resample_composition=args.resample_composition,
        )
        values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(grouping, values))
        for replicate, estimate in enumerate(estimates):
            row = {**base, "bootstrap_replicate": replicate}
            row.update(
                {
                    f"feature_{index}": float(value)
                    for index, value in enumerate(np.ravel(estimate))
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_reference_scores(args: argparse.Namespace) -> pd.DataFrame:
    reference = ensure_donor_observation_ids(read_table(args.metadata))
    query = ensure_donor_observation_ids(read_table(args.query_metadata))
    if len(query) != 1:
        raise ValueError("Reference bootstrap currently scores exactly one query row")
    reference_features = load_matrix(args.features)
    query_features = load_matrix(args.query_features)
    if sparse.issparse(reference_features):
        reference_features = reference_features.toarray()
    if sparse.issparse(query_features):
        query_features = query_features.toarray()
    if len(reference_features) != len(reference) or len(query_features) != 1:
        raise ValueError("Reference/query bootstrap arrays do not match metadata")
    query_row = query.iloc[0]
    if str(query_row["dataset"]) in set(reference["dataset"].astype(str)):
        raise ValueError(
            "Reference bootstrap input contains the query dataset; LODO leakage blocked"
        )
    if str(query_row["biological_unit_id"]) in set(
        reference["biological_unit_id"].astype(str)
    ):
        raise ValueError("Reference bootstrap input contains the query donor")
    result = bootstrap_healthy_reference_scores(
        np.asarray(reference_features, dtype=float),
        reference["age"],
        reference["sex"],
        reference["biological_unit_id"],
        np.asarray(query_features[0], dtype=float),
        float(query_row["age"]),
        str(query_row["sex"]),
        datasets=reference["dataset"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        model_kwargs={
            "n_spline_knots": args.n_spline_knots,
            "ridge": args.ridge,
            "age_grid_size": args.age_grid_size,
            "weighting_scheme": args.weighting_scheme,
        },
    )
    for column in (
        "dataset",
        "donor_id",
        "biological_unit_id",
        "sample_id",
        "source_observation_id",
        "observation_id",
    ):
        result[column] = query_row[column]
    return result


def command_bootstrap_scores(args: argparse.Namespace) -> int:
    _configured(args)
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.output.with_suffix(".manifest.json")
    )
    for path in (args.metadata, args.features):
        if not path.is_file():
            raise FileNotFoundError(f"Bootstrap input is missing: {path}")
    if args.uncertainty_layer == "reference":
        if args.query_metadata is None or args.query_features is None:
            raise ValueError(
                "Reference bootstrap requires --query-metadata and --query-features"
            )
        for path in (args.query_metadata, args.query_features):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Reference-bootstrap query input is missing: {path}"
                )
    if args.n_bootstrap < 1:
        raise ValueError("n-bootstrap must be positive")
    if (
        args.bootstrap_mode == "composition_standardized"
        and args.standardized_proportions is None
        and args.uncertainty_layer == "cell"
    ):
        raise ValueError(
            "composition-standardized cell bootstrap requires "
            "--standardized-proportions"
        )
    if args.dry_run:
        print(
            json_plan(
                "bootstrap-scores",
                uncertainty_layer=args.uncertainty_layer,
                metadata=args.metadata.resolve(),
                features=args.features.resolve(),
                output=args.output.resolve(),
                manifest=manifest_path.resolve(),
                n_bootstrap=args.n_bootstrap,
                bootstrap_mode=args.bootstrap_mode,
                biological_resampling_unit=(
                    "cells within fine-type strata and within one observation"
                    if args.uncertainty_layer == "cell"
                    else "whole healthy donors"
                ),
            )
        )
        return 0
    guard_outputs((args.output, manifest_path), overwrite=args.overwrite)
    result = (
        _bootstrap_cell_scores(args)
        if args.uncertainty_layer == "cell"
        else _bootstrap_reference_scores(args)
    )
    write_table(result, args.output)
    payload: dict[str, Any] = {
        "schema_version": "immune-health-uncertainty-replicates/v1",
        "status": "complete",
        "uncertainty_layer": args.uncertainty_layer,
        "biological_resampling_unit": (
            "cells_within_observation_fine_type_strata"
            if args.uncertainty_layer == "cell"
            else "whole_donors_within_cohort_and_sex"
        ),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "input_bindings": {
            "metadata": {
                "path": str(args.metadata.resolve()),
                "sha256": sha256_file(args.metadata),
            },
            "features": {
                "path": str(args.features.resolve()),
                "sha256": sha256_file(args.features),
            },
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "n_rows": len(result),
            "columns": list(result.columns),
        },
        "summary_metric_status": {
            "replicate_table_written": True,
            "cell_sampling_se_computed": False,
            "reference_sampling_se_computed": False,
            "seed_sd_computed": False,
            "reason": (
                "This stage writes auditable bootstrap replicates; a downstream "
                "endpoint-specific summarizer must compute and label an SD."
            ),
        },
        "spline_residual_covariance_used_for_within_cell_dispersion": False,
    }
    if args.uncertainty_layer == "reference":
        assert args.query_metadata is not None and args.query_features is not None
        payload["input_bindings"]["query_metadata"] = {
            "path": str(args.query_metadata.resolve()),
            "sha256": sha256_file(args.query_metadata),
        }
        payload["input_bindings"]["query_features"] = {
            "path": str(args.query_features.resolve()),
            "sha256": sha256_file(args.query_features),
        }
        payload["weighting_scheme"] = args.weighting_scheme
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return 0


def command_combine_seed_scores(args: argparse.Namespace) -> int:
    _configured(args)
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.output.with_suffix(".manifest.json")
    )
    if args.dry_run:
        print(
            json_plan(
                "combine-seed-scores",
                score_tables=[path.resolve() for path in args.scores],
                output=args.output.resolve(),
                manifest=manifest_path.resolve(),
                metrics=args.metric or "approved common scalar metrics",
                required_seeds=args.required_seed,
                embedding_coordinates_averaged=False,
            )
        )
        return 0
    combine_seed_score_tables(
        args.scores,
        args.output,
        manifest_path,
        metrics=args.metric,
        required_seeds=args.required_seed,
        overwrite=args.overwrite,
    )
    return 0


def command_evaluate_lodo(args: argparse.Namespace) -> int:
    _configured(args)
    header = read_table(args.predictions, nrows=0)
    require_columns(
        header,
        ("fold_id", "lineage", "gp_id", "observation_id", "age", "predicted_gp_age"),
        "LODO predictions",
    )
    if args.minimum_subgroup_size < 1:
        raise ValueError("minimum subgroup size must be positive")
    if args.dry_run:
        print(
            json_plan(
                "evaluate-lodo",
                predictions=args.predictions.resolve(),
                output=args.output.resolve(),
                wide_output=args.wide_output,
                method=args.method,
                reporting=(
                    "Per held-out fold first, including age-overlap and powered sex "
                    "subgroups when present"
                ),
            )
        )
        return 0
    paths = [args.output]
    if args.wide_output is not None:
        paths.append(args.wide_output)
    guard_outputs(paths, overwrite=args.overwrite)
    predictions = read_table(args.predictions)
    group_columns = ["fold_id", "lineage", "gp_id"]
    if "method" in predictions:
        group_columns.append("method")
    metrics = evaluate_lodo_metrics(
        predictions,
        group_columns=group_columns,
        minimum_subgroup_size=args.minimum_subgroup_size,
    )
    if "heldout_dataset" in predictions:
        fold_map = predictions.groupby("fold_id", observed=True)["heldout_dataset"].agg(
            lambda values: tuple(pd.unique(values.astype(str)))
        )
        inconsistent = fold_map[fold_map.map(len) != 1]
        if not inconsistent.empty:
            raise ValueError("heldout_dataset is inconsistent inside a fold")
        mapping = {fold: values[0] for fold, values in fold_map.items()}
    else:
        mapping = {
            fold: str(fold).removeprefix("lodo_")
            for fold in metrics["fold_id"].astype(str).unique()
        }
    metrics["heldout_dataset"] = metrics["fold_id"].map(mapping)
    if "method" not in metrics:
        metrics["method"] = args.method
    metrics["method"] = (
        metrics["method"].astype(str)
        + "::"
        + metrics["gp_id"].astype(str)
        + "::"
        + metrics["evaluation_subset"].astype(str)
    )
    metric_columns = [
        "n_observations",
        "mae",
        "rmse",
        "calibration_intercept",
        "calibration_slope",
        "pearson_r",
        "spearman_r",
    ]
    long = metrics.melt(
        id_vars=[
            "heldout_dataset",
            "fold_id",
            "lineage",
            "gp_id",
            "evaluation_subset",
            "method",
        ],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    write_table(long, args.output)
    if args.wide_output is not None:
        write_table(metrics, args.wide_output)
    return 0


def command_build_comparison_report(args: argparse.Namespace) -> int:
    config = _configured(args)
    header = read_table(args.metrics, nrows=0)
    require_columns(
        header,
        ("heldout_dataset", "lineage", "method", "metric", "value"),
        "comparison metrics",
    )
    expected = tuple(args.expected_heldout)
    if args.require_all_folds and not expected:
        expected = _configured_datasets(config) if config else REFERENCE_DATASETS
    if args.dry_run:
        print(
            json_plan(
                "build-comparison-report",
                metrics=args.metrics.resolve(),
                output=args.output.resolve(),
                expected_heldout_datasets=expected,
                order="Every held-out dataset before any aggregate",
            )
        )
        return 0
    guard_outputs((args.output,), overwrite=args.overwrite)
    build_comparison_report(
        read_table(args.metrics),
        args.output,
        expected_heldout_datasets=expected or None,
    )
    return 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="Command configuration YAML (command-specific defaults apply)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/contracts and print intended work without computation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacement of this command's output artifacts",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="immune-health",
        description=(
            "Donor-aware immune-health reference construction and frozen query scoring"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-data", help="Run the memory-conscious, read-only H5AD data audit"
    )
    _add_common_options(audit)
    audit.add_argument("--data-root", type=Path)
    audit.add_argument("--provenance", type=Path)
    audit.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "reports/data_audit"
    )
    audit.set_defaults(handler=command_audit_data)

    ontology = subparsers.add_parser(
        "build-fine-type-ontology",
        help="Generate a conservative exact-label ontology candidate",
    )
    _add_common_options(ontology)
    _add_overwrite(ontology)
    ontology.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "reports/data_audit/fine_type_summary.tsv",
    )
    ontology.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs/data/fine_type_ontology.generated.yaml",
    )
    ontology.add_argument(
        "--summary-output",
        type=Path,
        default=REPO_ROOT
        / "reports/data_audit/fine_type_ontology_candidate_summary.tsv",
    )
    ontology.add_argument("--minimum-confidence", type=float)
    ontology.add_argument("--minimum-cells-for-state", type=int, default=30)
    ontology.add_argument("--poor-donor-coverage-below", type=int, default=10)
    ontology.set_defaults(handler=command_build_ontology)

    folds = subparsers.add_parser(
        "make-lodo-folds", help="Create one global donor manifest and five LODO folds"
    )
    _add_common_options(folds)
    _add_overwrite(folds)
    folds.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "reports/data_audit/donor_summary.tsv.gz",
    )
    folds.add_argument("--output-dir", type=Path, default=REPO_ROOT / "splits")
    folds.add_argument("--dataset", action="append", default=[])
    folds.add_argument("--n-inner-folds", type=int, default=3)
    folds.add_argument(
        "--age-bin-edges",
        type=float,
        nargs="+",
        default=(0, 30, 45, 60, 75, np.inf),
    )
    folds.set_defaults(handler=command_make_lodo_folds)

    gp = subparsers.add_parser(
        "validate-gene-programs",
        help="Validate configured real GP resources without using held-out data",
    )
    _add_common_options(gp)
    _add_overwrite(gp)
    gp.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit resource; repeat to validate multiple libraries",
    )
    gp.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/gene_programs/resource_validation.tsv",
    )
    gp.add_argument("--allow-test-resources", action="store_true")
    gp.set_defaults(handler=command_validate_gene_programs)

    sampling = subparsers.add_parser(
        "build-sampling-manifest",
        help="Draw a deterministic dataset-donor-fine-type-cell epoch manifest",
    )
    _add_common_options(sampling)
    _add_overwrite(sampling)
    sampling.add_argument("--metadata", type=Path, required=True)
    sampling.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "runs/sampling"
    )
    sampling.add_argument("--lineage")
    sampling.add_argument("--n-cells", type=int, default=10_000)
    sampling.add_argument("--batch-size", type=int, default=256)
    sampling.add_argument("--epoch", type=int, default=0)
    sampling.add_argument("--alpha", type=float, default=0.5)
    sampling.add_argument(
        "--mode",
        choices=("observed_proportions", "fully_balanced", "hybrid"),
        default="hybrid",
    )
    sampling.add_argument("--fine-type-lambda", type=float, default=0.7)
    sampling.add_argument("--min-cells-per-fine-type", type=int, default=1)
    sampling.add_argument("--rank", type=int, default=0)
    sampling.add_argument("--world-size", type=int, default=1)
    sampling.set_defaults(handler=command_build_sampling_manifest)

    baseline = subparsers.add_parser(
        "build-pseudobulk-baselines",
        help="Build sparse donor-observation baselines from raw counts",
    )
    _add_common_options(baseline)
    _add_overwrite(baseline)
    baseline.add_argument("--counts", type=Path, required=True)
    baseline.add_argument("--metadata", type=Path, required=True)
    baseline.add_argument("--genes", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--min-cells", type=int, default=5)
    baseline.add_argument("--gp-resource", type=Path)
    baseline.add_argument(
        "--gp-score-method",
        choices=("mean_log_cpm", "mean_percentile_rank"),
        default="mean_log_cpm",
    )
    baseline.add_argument("--minimum-gp-genes", type=int, default=10)
    baseline.add_argument("--select-transferable-gps", action="store_true")
    baseline.add_argument("--transfer-minimum-donors", type=int, default=20)
    baseline.add_argument("--transfer-minimum-age-span", type=float, default=10.0)
    baseline.add_argument("--transfer-minimum-cohorts", type=int, default=3)
    baseline.add_argument(
        "--transfer-minimum-sign-concordance", type=float, default=0.75
    )
    baseline.add_argument("--transfer-maximum-i2", type=float, default=0.75)
    baseline.add_argument("--transfer-maximum-fdr", type=float, default=0.05)
    baseline.add_argument("--transfer-minimum-effect", type=float, default=0.0)
    baseline.add_argument("--heldout-dataset")
    baseline.add_argument("--n-components", type=int, default=10)
    baseline.add_argument("--max-dense-values", type=int, default=50_000_000)
    baseline.add_argument("--fit-age-model", action="store_true")
    baseline.add_argument("--age-inner-folds", type=int, default=5)
    baseline.set_defaults(handler=command_build_pseudobulk_baselines)

    train = subparsers.add_parser(
        "train-tripso",
        help="Run donor-gated TRIPSO training through the inspected vendor API",
    )
    _add_common_options(train)
    _add_overwrite(train)
    train.add_argument("--fold-input", type=Path)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--job-spec", type=Path)
    train.add_argument(
        "--vendor-root", type=Path, default=REPO_ROOT / "tripso_code/tripso"
    )
    train.add_argument(
        "--model-type", choices=("Base", "Global", "Global_LoRA"), default="Base"
    )
    train.add_argument("--base-model-dir", type=Path)
    train.add_argument("--parameters-json", type=Path)
    train.add_argument("--environment-report", type=Path)
    train.add_argument("--sampler-mode", default="hybrid")
    train.add_argument("--alpha", type=float, default=0.5)
    train.add_argument("--fine-type-lambda", type=float, default=0.7)
    train.add_argument("--tokenizer", default="geneformer_may2025")
    train.add_argument("--preprocessing", default="fold_input_manifest")
    train.add_argument("--embedding-dimension", type=int, default=256)
    train.set_defaults(handler=command_train_tripso)

    project = subparsers.add_parser(
        "project-tripso",
        help="Project all reference or query cells with frozen weights/resources",
    )
    _add_common_options(project)
    _add_overwrite(project)
    project.add_argument("--model-manifest", type=Path, required=True)
    project.add_argument(
        "--projection-manifest",
        "--query-manifest",
        dest="query_manifest",
        type=Path,
        required=True,
        help="Model-bound reference/validation/query projection-input manifest",
    )
    project.add_argument("--output-dir", type=Path, required=True)
    project.add_argument(
        "--vendor-root", type=Path, default=REPO_ROOT / "tripso_code/tripso"
    )
    project.add_argument("--batch-size", type=int, default=128)
    project.add_argument("--precision", default="32")
    project.set_defaults(handler=command_project_tripso)

    aggregate = subparsers.add_parser(
        "aggregate-donor-distributions",
        help="Retain fine-type GP distributions and shrinkage state summaries",
    )
    _add_common_options(aggregate)
    _add_overwrite(aggregate)
    aggregate.add_argument("--embeddings", type=Path, required=True)
    aggregate.add_argument("--metadata", type=Path, required=True)
    aggregate.add_argument("--arrow-conversion-manifest", type=Path, required=True)
    aggregate.add_argument("--gp-id", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--fine-type-universe", type=Path)
    aggregate.add_argument("--min-state-cells", type=int, default=5)
    aggregate.add_argument("--min-empirical-cells", type=int, default=25)
    aggregate.add_argument("--age-direction", type=Path)
    aggregate.add_argument("--use-mean-location", action="store_true")
    aggregate.set_defaults(handler=command_aggregate_distributions)

    endpoint = subparsers.add_parser(
        "assemble-donor-gp-endpoint",
        help="Materialize one role-aware lineage/fine-type/GP donor endpoint",
    )
    _add_common_options(endpoint)
    _add_overwrite(endpoint)
    endpoint.add_argument("--aggregate-table", type=Path, required=True)
    endpoint.add_argument("--aggregation-manifest", type=Path, required=True)
    endpoint.add_argument("--projection-output-manifest", type=Path, required=True)
    endpoint.add_argument("--lineage", required=True)
    endpoint.add_argument("--fine-type", required=True)
    endpoint.add_argument("--gp-id", required=True)
    endpoint.add_argument("--output-dir", type=Path, required=True)
    endpoint.set_defaults(handler=command_assemble_donor_gp_endpoint)

    healthy = subparsers.add_parser(
        "fit-healthy-reference",
        help="Fit and serialize a donor-weighted frozen healthy-age trajectory",
    )
    _add_common_options(healthy)
    _add_overwrite(healthy)
    healthy.add_argument("--metadata", type=Path, required=True)
    healthy.add_argument("--features", type=Path)
    healthy.add_argument("--feature-column", action="append")
    healthy.add_argument("--endpoint-manifest", type=Path)
    healthy.add_argument("--output-dir", type=Path, required=True)
    healthy.add_argument("--heldout-dataset")
    healthy.add_argument("--final-all-healthy", action="store_true")
    healthy.add_argument(
        "--legacy-combined-lodo-input",
        action="store_true",
        help=(
            "Explicitly permit one table containing adaptation and held-out rows; "
            "production endpoint manifests use physically separate role artifacts"
        ),
    )
    healthy.add_argument("--n-inner-folds", type=int, default=5)
    healthy.add_argument("--n-spline-knots", type=int, default=3)
    healthy.add_argument("--ridge", type=float, default=1e-3)
    healthy.add_argument("--age-grid-size", type=int, default=101)
    healthy.add_argument("--age-kernel-bandwidth", type=float, default=10.0)
    healthy.add_argument(
        "--age-kernel-minimum-exact-sex-donors",
        type=int,
        default=DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    )
    healthy.add_argument(
        "--weighting-scheme",
        choices=("donor_pooled", "cohort_balanced"),
        default="donor_pooled",
    )
    healthy.add_argument("--age-support-window-years", type=float, default=5.0)
    healthy.add_argument("--minimum-support-cohorts", type=int, default=3)
    healthy.add_argument("--minimum-support-donors", type=int, default=20)
    healthy.add_argument("--slope-minimum-donors", type=int, default=20)
    healthy.add_argument("--slope-minimum-age-span", type=float, default=10.0)
    healthy.set_defaults(handler=command_fit_healthy_reference)

    selector = subparsers.add_parser(
        "select-transferable-tripso-gps",
        help=(
            "Freeze fine-type/GP endpoints from reference-only donor cross-fit scores"
        ),
    )
    _add_common_options(selector)
    _add_overwrite(selector)
    selector.add_argument(
        "--reference-manifest", type=Path, action="append", default=[]
    )
    selector.add_argument(
        "--reference-manifest-list", type=Path, action="append", default=[]
    )
    selector.add_argument("--lineage", required=True)
    selector.add_argument("--fold-id", required=True)
    selector.add_argument("--heldout-dataset", required=True)
    selector.add_argument("--required-training-dataset", action="append", default=[])
    selector.add_argument("--required-seed", type=int, action="append", default=[])
    selector.add_argument(
        "--weighting-scheme",
        choices=("donor_pooled", "cohort_balanced"),
        default="donor_pooled",
    )
    selector.add_argument("--output-dir", type=Path, required=True)
    selector.add_argument("--simple-baseline", type=Path)
    selector.add_argument("--simple-baseline-score-column", default="gp_score")
    selector.add_argument("--minimum-donors-per-cohort", type=int)
    selector.add_argument("--minimum-age-span", type=float)
    selector.add_argument("--minimum-cohorts", type=int)
    selector.add_argument("--minimum-sign-concordance", type=float)
    selector.add_argument("--maximum-i2", type=float)
    selector.add_argument("--maximum-fdr", type=float)
    selector.add_argument(
        "--minimum-absolute-standardized-slope-per-decade", type=float
    )
    selector.add_argument("--minimum-seed-retention-fraction", type=float)
    selector.add_argument("--minimum-seed-sign-concordance", type=float)
    selector.add_argument("--minimum-state-observation-coverage", type=float)
    selector.add_argument("--minimum-median-cells", type=float)
    selector.add_argument("--maximum-absolute-depth-partial-correlation", type=float)
    selector.add_argument(
        "--maximum-absolute-composition-partial-correlation", type=float
    )
    selector.add_argument("--minimum-seed-rank-correlation", type=float)
    selector.add_argument("--maximum-seed-effect-sd", type=float)
    selector.add_argument("--minimum-baseline-standardized-improvement", type=float)
    selector.set_defaults(handler=command_select_transferable_tripso_gps)

    score = subparsers.add_parser(
        "score-query",
        help=(
            "Score fixed inner-validation or locked outer-query donors against a "
            "frozen healthy reference"
        ),
    )
    _add_common_options(score)
    _add_overwrite(score)
    score.add_argument("--reference-manifest", type=Path, required=True)
    score.add_argument("--query-metadata", type=Path, required=True)
    score.add_argument("--features", type=Path)
    score.add_argument("--feature-column", action="append")
    score.add_argument("--endpoint-manifest", type=Path)
    score.add_argument("--query-genes", type=Path, required=True)
    score.add_argument("--frozen-vocabulary", type=Path, required=True)
    score.add_argument("--minimum-gene-coverage", type=float, default=0.8)
    score.add_argument("--gp-coverage", type=float)
    score.add_argument("--minimum-gp-coverage", type=float, default=0.7)
    score.add_argument("--allow-low-coverage", action="store_true")
    score.add_argument("--age-support-window-years", type=float, default=5.0)
    score.add_argument("--minimum-support-cohorts", type=int, default=3)
    score.add_argument("--minimum-support-donors", type=int, default=20)
    score.add_argument("--raw-counts", type=Path)
    score.add_argument("--raw-metadata", type=Path)
    score.add_argument("--model-manifest", type=Path)
    score.add_argument("--query-manifest", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.set_defaults(handler=command_score_query)

    empirical = subparsers.add_parser(
        "score-empirical-endpoint",
        help=(
            "Matched-depth sliced-Wasserstein scoring for exact reference/query "
            "endpoint empirical distributions"
        ),
    )
    _add_common_options(empirical)
    _add_overwrite(empirical)
    empirical.add_argument("--reference-endpoint-manifest", type=Path, required=True)
    empirical.add_argument("--query-endpoint-manifest", type=Path, required=True)
    empirical.add_argument("--reference-empirical-index", type=Path, required=True)
    empirical.add_argument("--query-empirical-index", type=Path, required=True)
    empirical.add_argument("--output-dir", type=Path, required=True)
    empirical.add_argument(
        "--depth",
        type=int,
        action="append",
        help="Matched query/reference cell depth; repeat for a reliability curve",
    )
    empirical.add_argument("--n-replicates", type=int, default=100)
    empirical.add_argument("--n-projections", type=int, default=128)
    empirical.add_argument("--age-grid-size", type=int, default=101)
    empirical.add_argument("--age-kernel-bandwidth", type=float, default=10.0)
    empirical.add_argument(
        "--minimum-exact-sex-donors",
        type=int,
        default=DEFAULT_MINIMUM_EXACT_SEX_DONORS,
    )
    empirical.add_argument(
        "--weighting-scheme",
        choices=("donor_pooled", "cohort_balanced"),
        default="donor_pooled",
    )
    empirical.set_defaults(handler=command_score_empirical_endpoint)

    bootstrap = subparsers.add_parser(
        "bootstrap-scores",
        help="Estimate cell-sampling or healthy-reference uncertainty separately",
    )
    _add_common_options(bootstrap)
    _add_overwrite(bootstrap)
    bootstrap.add_argument(
        "--uncertainty-layer", choices=("cell", "reference"), default="cell"
    )
    bootstrap.add_argument("--metadata", type=Path, required=True)
    bootstrap.add_argument("--features", type=Path, required=True)
    bootstrap.add_argument("--query-metadata", type=Path)
    bootstrap.add_argument("--query-features", type=Path)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--manifest", type=Path)
    bootstrap.add_argument("--n-bootstrap", type=int, default=100)
    bootstrap.add_argument(
        "--bootstrap-mode",
        choices=("observed_mixture", "composition_standardized"),
        default="observed_mixture",
    )
    bootstrap.add_argument("--standardized-proportions", type=Path)
    bootstrap.add_argument("--resample-composition", action="store_true")
    bootstrap.add_argument("--n-spline-knots", type=int, default=3)
    bootstrap.add_argument("--ridge", type=float, default=1e-3)
    bootstrap.add_argument("--age-grid-size", type=int, default=101)
    bootstrap.add_argument(
        "--weighting-scheme",
        choices=("donor_pooled", "cohort_balanced"),
        default="donor_pooled",
    )
    bootstrap.set_defaults(handler=command_bootstrap_scores)

    seed_combination = subparsers.add_parser(
        "combine-seed-scores",
        help=(
            "Compute seed SD from calibrated scalar scores without averaging "
            "coordinates"
        ),
    )
    _add_common_options(seed_combination)
    _add_overwrite(seed_combination)
    seed_combination.add_argument("--scores", type=Path, action="append", required=True)
    seed_combination.add_argument("--output", type=Path, required=True)
    seed_combination.add_argument("--manifest", type=Path)
    seed_combination.add_argument("--metric", action="append", default=[])
    seed_combination.add_argument(
        "--required-seed", type=int, action="append", default=[]
    )
    seed_combination.set_defaults(handler=command_combine_seed_scores)

    evaluate = subparsers.add_parser(
        "evaluate-lodo",
        help="Compute donor-observation metrics separately for every LODO fold",
    )
    _add_common_options(evaluate)
    _add_overwrite(evaluate)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--wide-output", type=Path)
    evaluate.add_argument("--method", default="tripso")
    evaluate.add_argument("--minimum-subgroup-size", type=int, default=5)
    evaluate.set_defaults(handler=command_evaluate_lodo)

    comparison = subparsers.add_parser(
        "build-comparison-report",
        help="Write fold-first method comparisons before cross-fold summaries",
    )
    _add_common_options(comparison)
    _add_overwrite(comparison)
    comparison.add_argument("--metrics", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--expected-heldout", action="append", default=[])
    comparison.add_argument("--require-all-folds", action="store_true")
    comparison.set_defaults(handler=command_build_comparison_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except (
        AssertionError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
