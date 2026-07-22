"""Focused CLI for staged, CPU-only reference-lineage preparation.

Run as ``python -m immune_health.cli.prepare_reference``.  This module remains
separate from the central CLI so reference preparation can be deployed and
reviewed independently of model-training commands.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from immune_health.data.reference_preparation import (
    ReferenceFeatureConfig,
    materialize_fold_h5ad,
    materialize_frozen_query_h5ad,
    prepare_fold_features,
    write_all_healthy_reference_fold,
    write_terekhova_one_visit_manifest,
)

LOGGER = logging.getLogger(__name__)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if ".parquet" in suffixes:
        return pd.read_parquet(path)
    if ".tsv" in suffixes or ".txt" in suffixes:
        return pd.read_csv(path, sep="\t")
    if ".csv" in suffixes:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _feature_config(args: argparse.Namespace) -> ReferenceFeatureConfig:
    hvg_sizes = args.hvg_size or [3000, 9000]
    return ReferenceFeatureConfig(
        hvg_sizes=tuple(sorted(set(hvg_sizes))),
        hvg_mean_bins=args.hvg_mean_bins,
        hvg_minimum_donor_fraction=args.hvg_minimum_donor_fraction,
        hvg_minimum_dataset_fraction=args.hvg_minimum_dataset_fraction,
        gp_minimum_mapped_genes=args.gp_minimum_mapped_genes,
        gp_maximum_program_size=args.gp_maximum_program_size,
        gp_minimum_expression_coverage=args.gp_minimum_expression_coverage,
        gp_minimum_donor_coverage=args.gp_minimum_donor_coverage,
        gp_minimum_dataset_fraction=args.gp_minimum_dataset_fraction,
        gp_redundancy_jaccard_threshold=args.gp_redundancy_jaccard_threshold,
        gp_transfer_minimum_donors_per_cohort=(
            args.gp_transfer_minimum_donors_per_cohort
        ),
        gp_transfer_minimum_age_span=args.gp_transfer_minimum_age_span,
        gp_transfer_minimum_cohorts=args.gp_transfer_minimum_cohorts,
        gp_transfer_minimum_sign_concordance=(
            args.gp_transfer_minimum_sign_concordance
        ),
        gp_transfer_maximum_i2=args.gp_transfer_maximum_i2,
        gp_transfer_maximum_fdr=args.gp_transfer_maximum_fdr,
        gp_transfer_minimum_absolute_standardized_slope_per_decade=(
            args.gp_transfer_minimum_absolute_standardized_slope_per_decade
        ),
        gp_projection_control_ids=tuple(args.gp_projection_control_id),
    )


def _run_visits(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "build-terekhova-visits",
                    "input": str(args.metadata),
                    "output_dir": str(args.output_dir),
                    "seed": args.seed,
                    "selection": (
                        "minimum sha256(seed::biological_unit_id::observation_id)"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    table_path, manifest_path = write_terekhova_one_visit_manifest(
        _read_table(args.metadata),
        args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    LOGGER.info("Wrote %s and %s", table_path, manifest_path)
    return 0


def _run_all_healthy_fold(args: argparse.Namespace) -> int:
    datasets = tuple(args.healthy_dataset)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "build-all-healthy-fold",
                    "reference_design": "all_healthy",
                    "input": str(args.metadata),
                    "output_dir": str(args.output_dir),
                    "healthy_datasets": list(datasets),
                    "heldout_dataset": None,
                    "inner_validation_fold": args.inner_validation_fold,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    table_path, manifest_path = write_all_healthy_reference_fold(
        _read_table(args.metadata),
        args.output_dir,
        healthy_datasets=datasets,
        inner_validation_fold=args.inner_validation_fold,
        inner_fold_column=args.inner_fold_column,
        overwrite=args.overwrite,
    )
    LOGGER.info("Wrote %s and %s", table_path, manifest_path)
    return 0


def _run_features(args: argparse.Namespace) -> int:
    config = _feature_config(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "select-fold-features",
                    "input_h5ad": str(args.input_h5ad),
                    "fold_manifest": str(args.fold_manifest),
                    "visit_manifest": str(args.visit_manifest),
                    "gene_programs": str(args.gene_programs),
                    "fine_type_ontology": str(args.fine_type_ontology),
                    "output_dir": str(args.output_dir),
                    "lineage": args.lineage,
                    "reference_design": args.reference_design,
                    "hvg_sizes": list(config.hvg_sizes),
                    "inner_validation_fold": args.inner_validation_fold,
                    "global_one_visit_query": args.global_one_visit_query,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = prepare_fold_features(
        args.input_h5ad,
        _read_table(args.fold_manifest),
        _read_table(args.visit_manifest),
        args.gene_programs,
        args.output_dir,
        lineage=args.lineage,
        fine_type_ontology_path=args.fine_type_ontology,
        reference_design=args.reference_design,
        config=config,
        chunk_size=args.chunk_size,
        inner_validation_fold=args.inner_validation_fold,
        global_one_visit_query=args.global_one_visit_query,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Prepared %s: %s programs, vocabularies %s",
        args.output_dir,
        manifest["n_retained_gene_programs"],
        sorted(manifest["vocabularies"]),
    )
    return 0


def _run_materialize(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "materialize-fold-h5ad",
                    "input_h5ad": str(args.input_h5ad),
                    "preparation_dir": str(args.preparation_dir),
                    "output_h5ad": str(args.output_h5ad),
                    "role": args.role,
                    "hvg_size": args.hvg_size,
                    "cell_downsampling": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = materialize_fold_h5ad(
        args.input_h5ad,
        args.preparation_dir,
        args.output_h5ad,
        role=args.role,
        hvg_size=args.hvg_size,
        row_chunk_size=args.row_chunk_size,
        max_loaded_elements=args.max_loaded_elements,
        overwrite=args.overwrite,
    )
    LOGGER.info("Materialized %s cells to %s", manifest["shape"][0], args.output_h5ad)
    return 0


def _run_frozen_query(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "materialize-frozen-query-h5ad",
                    "input_h5ad": str(args.input_h5ad),
                    "final_preparation_dir": str(args.final_preparation_dir),
                    "output_h5ad": str(args.output_h5ad),
                    "lineage": args.lineage,
                    "hvg_size": args.hvg_size,
                    "feature_selection_on_query": False,
                    "cell_downsampling": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = materialize_frozen_query_h5ad(
        args.input_h5ad,
        args.final_preparation_dir,
        args.output_h5ad,
        hvg_size=args.hvg_size,
        lineage=args.lineage,
        minimum_gene_coverage=args.minimum_gene_coverage,
        allow_training_dataset=args.allow_training_dataset,
        row_chunk_size=args.row_chunk_size,
        max_loaded_elements=args.max_loaded_elements,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Mapped %s query cells at %.2f%% frozen-gene coverage to %s",
        manifest["shape"][0],
        100 * manifest["frozen_gene_coverage"],
        args.output_h5ad,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    visits = subparsers.add_parser(
        "build-terekhova-visits",
        help="Select one reference-fitting visit per Terekhova donor",
    )
    visits.add_argument("--metadata", type=Path, required=True)
    visits.add_argument("--output-dir", type=Path, required=True)
    visits.add_argument("--seed", type=int, default=42)
    visits.add_argument("--overwrite", action="store_true")
    visits.add_argument("--dry-run", action="store_true")
    visits.set_defaults(handler=_run_visits)

    all_healthy = subparsers.add_parser(
        "build-all-healthy-fold",
        help="Declare all five healthy cohorts as final reference data",
    )
    all_healthy.add_argument("--metadata", type=Path, required=True)
    all_healthy.add_argument("--output-dir", type=Path, required=True)
    all_healthy.add_argument("--healthy-dataset", action="append", required=True)
    all_healthy.add_argument("--inner-validation-fold", type=int)
    all_healthy.add_argument("--inner-fold-column", default="global_inner_fold")
    all_healthy.add_argument("--overwrite", action="store_true")
    all_healthy.add_argument("--dry-run", action="store_true")
    all_healthy.set_defaults(handler=_run_all_healthy_fold)

    features = subparsers.add_parser(
        "select-fold-features",
        help="Learn training-only programs and donor/dataset-aware HVGs",
    )
    features.add_argument("--input-h5ad", type=Path, required=True)
    features.add_argument("--fold-manifest", type=Path, required=True)
    features.add_argument("--visit-manifest", type=Path, required=True)
    features.add_argument("--gene-programs", type=Path, required=True)
    features.add_argument(
        "--fine-type-ontology",
        type=Path,
        required=True,
        help="Scientifically approved, immutable raw-to-canonical fine-type YAML",
    )
    features.add_argument("--output-dir", type=Path, required=True)
    features.add_argument("--lineage", required=True)
    features.add_argument(
        "--reference-design",
        choices=("lodo", "all_healthy"),
        default="lodo",
    )
    features.add_argument("--hvg-size", type=int, action="append")
    features.add_argument("--hvg-mean-bins", type=int, default=20)
    features.add_argument("--hvg-minimum-donor-fraction", type=float, default=0.01)
    features.add_argument("--hvg-minimum-dataset-fraction", type=float, default=0.75)
    features.add_argument("--gp-minimum-mapped-genes", type=int, default=10)
    features.add_argument("--gp-maximum-program-size", type=int, default=200)
    features.add_argument("--gp-minimum-expression-coverage", type=float, default=0.001)
    features.add_argument("--gp-minimum-donor-coverage", type=float, default=0.05)
    features.add_argument("--gp-minimum-dataset-fraction", type=float, default=0.75)
    features.add_argument("--gp-redundancy-jaccard-threshold", type=float, default=0.8)
    features.add_argument(
        "--gp-transfer-minimum-donors-per-cohort", type=int, default=20
    )
    features.add_argument("--gp-transfer-minimum-age-span", type=float, default=10.0)
    features.add_argument("--gp-transfer-minimum-cohorts", type=int, default=3)
    features.add_argument(
        "--gp-transfer-minimum-sign-concordance", type=float, default=0.75
    )
    features.add_argument("--gp-transfer-maximum-i2", type=float, default=0.75)
    features.add_argument("--gp-transfer-maximum-fdr", type=float, default=0.05)
    features.add_argument(
        "--gp-transfer-minimum-absolute-standardized-slope-per-decade",
        type=float,
        default=0.0,
    )
    features.add_argument(
        "--gp-projection-control-id",
        action="append",
        default=[],
        help=(
            "Prespecified filtered GP to retain as a projection control; repeat as "
            "needed. This is never inferred from query data."
        ),
    )
    features.add_argument("--chunk-size", type=int, default=20_000)
    features.add_argument("--inner-validation-fold", type=int)
    query_visits = features.add_mutually_exclusive_group()
    query_visits.add_argument(
        "--global-one-visit-query",
        dest="global_one_visit_query",
        action="store_true",
        help="Keep one deterministic Terekhova visit in every role (default)",
    )
    query_visits.add_argument(
        "--preserve-all-query-visits",
        dest="global_one_visit_query",
        action="store_false",
        help="Explicit longitudinal sensitivity: retain all Terekhova query visits",
    )
    features.set_defaults(global_one_visit_query=True)
    features.add_argument("--overwrite", action="store_true")
    features.add_argument("--dry-run", action="store_true")
    features.set_defaults(handler=_run_features)

    materialize = subparsers.add_parser(
        "materialize-fold-h5ad",
        help="Write one role-specific H5AD using one exact prepared vocabulary",
    )
    materialize.add_argument("--input-h5ad", type=Path, required=True)
    materialize.add_argument("--preparation-dir", type=Path, required=True)
    materialize.add_argument("--output-h5ad", type=Path, required=True)
    materialize.add_argument(
        "--role", choices=("adaptation", "validation", "query"), required=True
    )
    materialize.add_argument("--hvg-size", type=int, default=9000)
    materialize.add_argument("--row-chunk-size", type=int, default=25_000)
    materialize.add_argument("--max-loaded-elements", type=int, default=100_000_000)
    materialize.add_argument("--overwrite", action="store_true")
    materialize.add_argument("--dry-run", action="store_true")
    materialize.set_defaults(handler=_run_materialize)

    query = subparsers.add_parser(
        "materialize-frozen-query-h5ad",
        help="Map an unseen query to a final all-healthy frozen vocabulary",
    )
    query.add_argument("--input-h5ad", type=Path, required=True)
    query.add_argument("--final-preparation-dir", type=Path, required=True)
    query.add_argument("--output-h5ad", type=Path, required=True)
    query.add_argument("--lineage", required=True)
    query.add_argument("--hvg-size", type=int, required=True)
    query.add_argument("--minimum-gene-coverage", type=float, default=0.8)
    query.add_argument("--allow-training-dataset", action="store_true")
    query.add_argument("--row-chunk-size", type=int, default=25_000)
    query.add_argument("--max-loaded-elements", type=int, default=100_000_000)
    query.add_argument("--overwrite", action="store_true")
    query.add_argument("--dry-run", action="store_true")
    query.set_defaults(handler=_run_frozen_query)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
