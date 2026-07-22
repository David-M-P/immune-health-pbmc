"""Focused CLI for exact, all-cell TRIPSO tokenization and manifest binding.

Run as ``python -m immune_health.cli.tokenize_tripso``.  Tokenization is kept
separate from GPU training so CPU and GPU stages can be transferred or scheduled
independently.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from immune_health.tripso_adapter.tokenization import (
    DEFAULT_MAX_PROJECTED_BYTES,
    build_fold_input_from_tokenization,
    build_projection_input_from_tokenization,
    build_query_input_from_tokenization,
    relocate_tokenization_manifest,
    tokenize_fold_h5ad,
)

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _run_tokenize(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "tokenize",
                    "input_h5ad": str(args.input_h5ad),
                    "gene_vocabulary": str(args.gene_vocabulary),
                    "gp_library": str(args.gp_library),
                    "projection_gp_candidates": str(args.projection_gp_candidates),
                    "output_dir": str(args.output_dir),
                    "role": args.role,
                    "row_chunk_size": args.row_chunk_size,
                    "nproc": args.nproc,
                    "calculate_hvg": False,
                    "cell_subsampling": False,
                    "model_input_size": 4096,
                    "required_metadata": [
                        "cell_key",
                        "dataset",
                        "biological_unit_id",
                        "observation_id",
                        "fine_type",
                        "fine_type_state_eligible",
                        "fine_type_balance_eligible",
                        "lineage",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = tokenize_fold_h5ad(
        input_h5ad=args.input_h5ad,
        gene_vocabulary_path=args.gene_vocabulary,
        gp_library_path=args.gp_library,
        projection_gp_candidates_path=args.projection_gp_candidates,
        output_dir=args.output_dir,
        vendor_root=args.vendor_root,
        role=args.role,
        row_chunk_size=args.row_chunk_size,
        nproc=args.nproc,
        minimum_tokenizable_gp_genes=args.minimum_tokenizable_gp_genes,
        keep_chunks=args.keep_chunks,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Tokenized %s cells from %s biological units; %.2f%% of cells were truncated",
        manifest["shape"][0],
        manifest["n_biological_units"],
        100 * manifest["sequence_qc"]["fraction_cells_truncated"],
    )
    return 0


def _run_relocate(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "relocate-tokenization",
                    "source_manifest": str(args.source_manifest),
                    "output_manifest": str(args.output_manifest),
                    "tokenized_dataset": str(args.tokenized_dataset),
                    "input_h5ad": str(args.input_h5ad),
                    "gene_vocabulary": str(args.gene_vocabulary),
                    "gp_library": str(args.gp_library),
                    "projection_gp_candidates": str(args.projection_gp_candidates),
                    "vendor_root": str(args.vendor_root),
                    "checks": [
                        "source manifest self-hash",
                        "exact immutable resource hashes",
                        "exact Arrow per-file inventory",
                        "physical row/donor/dataset/lineage scope",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = relocate_tokenization_manifest(
        source_manifest_path=args.source_manifest,
        output_manifest_path=args.output_manifest,
        tokenized_dataset_path=args.tokenized_dataset,
        input_h5ad=args.input_h5ad,
        gene_vocabulary_path=args.gene_vocabulary,
        gp_library_path=args.gp_library,
        projection_gp_candidates_path=args.projection_gp_candidates,
        vendor_root=args.vendor_root,
        materialization_manifest_path=args.materialization_manifest,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Relocated and verified %s tokenized cells; wrote %s",
        manifest["shape"][0],
        args.output_manifest,
    )
    return 0


def _run_fold_input(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "build-fold-input",
                    "tokenization_manifest": str(args.tokenization_manifest),
                    "fold_table": str(args.fold_table),
                    "output": str(args.output),
                    "fold_id": args.fold_id,
                    "held_out_dataset": args.held_out_dataset,
                    "reference_design": args.reference_design,
                    "lineage": args.lineage,
                    "partition_column": args.partition_column,
                    "inner_validation_fold": args.inner_validation_fold,
                    "inner_fold_column": args.inner_fold_column,
                    "donor_scope_proof": "read from physical Hugging Face dataset",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Fold input already exists: {args.output}")
    manifest = build_fold_input_from_tokenization(
        tokenization_manifest_path=args.tokenization_manifest,
        fold_table_path=args.fold_table,
        output_path=args.output,
        fold_id=args.fold_id,
        held_out_dataset=args.held_out_dataset,
        lineage=args.lineage,
        partition_column=args.partition_column,
        sampler_manifest_path=args.sampler_manifest,
        reference_design=args.reference_design,
        inner_validation_fold=args.inner_validation_fold,
        inner_fold_column=args.inner_fold_column,
    )
    LOGGER.info(
        "Bound %s adaptation donors to %s",
        len(manifest["adaptation_biological_unit_ids"]),
        args.output,
    )
    return 0


def _run_query_input(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "build-query-input",
                    "tokenization_manifest": str(args.tokenization_manifest),
                    "model_manifest": str(args.model_manifest),
                    "output": str(args.output),
                    "adapt": False,
                    "frozen_checks": [
                        "gene vocabulary hash",
                        "GP library hash",
                        "token dictionary and preprocessing contract hash",
                        "training/query donor disjointness",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Query input already exists: {args.output}")
    manifest = build_query_input_from_tokenization(
        tokenization_manifest_path=args.tokenization_manifest,
        model_manifest_path=args.model_manifest,
        output_path=args.output,
        seed=args.seed,
        gp_allowlist_path=args.gp_allowlist,
        use_fold_bound_gp_candidates=args.use_fold_bound_gp_candidates,
        allow_all_gps=args.allow_all_gps,
        include_cell_token=args.include_cell_token,
        include_gene_encoder_cls=args.include_gene_encoder_cls,
        max_projected_bytes=args.max_projected_bytes,
        allow_oversized_projection=args.allow_oversized_projection,
    )
    LOGGER.info(
        "Bound frozen query input with %s biological units to %s",
        len(manifest["biological_unit_ids"]),
        args.output,
    )
    return 0


def _run_projection_input(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "build-projection-input",
                    "projection_role": args.role,
                    "tokenization_manifest": str(args.tokenization_manifest),
                    "model_manifest": str(args.model_manifest),
                    "output": str(args.output),
                    "adapt": False,
                    "optimizer_allowed": False,
                    "gp_allowlist": (
                        str(args.gp_allowlist) if args.gp_allowlist else None
                    ),
                    "allow_all_gps": args.allow_all_gps,
                    "max_projected_bytes": args.max_projected_bytes,
                    "allow_oversized_projection": args.allow_oversized_projection,
                    "physical_scope_revalidated": True,
                    "reference_scope": (
                        "exact adaptation donor and training-tokenization equality"
                        if args.role == "reference"
                        else (
                            "exact fixed inner-validation donor equality"
                            if args.role == "validation"
                            else "exact held-out donors, disjoint from training roles"
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Projection input already exists: {args.output}")
    manifest = build_projection_input_from_tokenization(
        tokenization_manifest_path=args.tokenization_manifest,
        model_manifest_path=args.model_manifest,
        output_path=args.output,
        role=args.role,
        seed=args.seed,
        gp_allowlist_path=args.gp_allowlist,
        use_fold_bound_gp_candidates=args.use_fold_bound_gp_candidates,
        allow_all_gps=args.allow_all_gps,
        include_cell_token=args.include_cell_token,
        include_gene_encoder_cls=args.include_gene_encoder_cls,
        max_projected_bytes=args.max_projected_bytes,
        allow_oversized_projection=args.allow_oversized_projection,
    )
    LOGGER.info(
        "Bound frozen %s projection with %s cells from %s biological units to %s",
        args.role,
        manifest["n_cells"],
        len(manifest["biological_unit_ids"]),
        args.output,
    )
    return 0


def _add_projection_gp_options(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--projection-gp-candidates",
        "--gp-allowlist",
        dest="gp_allowlist",
        type=Path,
        help="Fold-bound training-only GP candidate JSON manifest",
    )
    selection.add_argument(
        "--use-fold-bound-gp-candidates",
        action="store_true",
        help="Use the candidate manifest immutably recorded by the training fold",
    )
    selection.add_argument(
        "--allow-all-gps",
        action="store_true",
        help="Explicit bounded diagnostic only; persist every trained GP vector",
    )
    parser.add_argument("--include-cell-token", action="store_true")
    parser.add_argument("--include-gene-encoder-cls", action="store_true")
    parser.add_argument(
        "--max-projected-bytes", type=int, default=DEFAULT_MAX_PROJECTED_BYTES
    )
    parser.add_argument("--allow-oversized-projection", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    tokenize = subparsers.add_parser(
        "tokenize", help="Tokenize one exact all-cell H5AD without feature selection"
    )
    tokenize.add_argument("--input-h5ad", type=Path, required=True)
    tokenize.add_argument("--gene-vocabulary", type=Path, required=True)
    tokenize.add_argument("--gp-library", type=Path, required=True)
    tokenize.add_argument(
        "--projection-gp-candidates",
        type=Path,
        required=True,
        help="Training-only candidate JSON emitted by fold feature preparation",
    )
    tokenize.add_argument("--output-dir", type=Path, required=True)
    tokenize.add_argument(
        "--vendor-root",
        type=Path,
        default=REPOSITORY_ROOT / "tripso_code" / "tripso",
    )
    tokenize.add_argument(
        "--role", choices=("adaptation", "validation", "query"), required=True
    )
    tokenize.add_argument("--row-chunk-size", type=int, default=20_000)
    tokenize.add_argument("--nproc", type=int, default=4)
    tokenize.add_argument("--minimum-tokenizable-gp-genes", type=int, default=10)
    tokenize.add_argument("--keep-chunks", action="store_true")
    tokenize.add_argument("--overwrite", action="store_true")
    tokenize.add_argument("--dry-run", action="store_true")
    tokenize.set_defaults(handler=_run_tokenize)

    relocate = subparsers.add_parser(
        "relocate-tokenization",
        help=(
            "Verify an exact SFTP copy and rewrite only cluster-local absolute paths"
        ),
    )
    relocate.add_argument("--source-manifest", type=Path, required=True)
    relocate.add_argument("--output-manifest", type=Path, required=True)
    relocate.add_argument("--tokenized-dataset", type=Path, required=True)
    relocate.add_argument("--input-h5ad", type=Path, required=True)
    relocate.add_argument("--gene-vocabulary", type=Path, required=True)
    relocate.add_argument("--gp-library", type=Path, required=True)
    relocate.add_argument("--projection-gp-candidates", type=Path, required=True)
    relocate.add_argument(
        "--vendor-root",
        type=Path,
        default=REPOSITORY_ROOT / "tripso_code" / "tripso",
    )
    relocate.add_argument("--materialization-manifest", type=Path)
    relocate.add_argument("--overwrite", action="store_true")
    relocate.add_argument("--dry-run", action="store_true")
    relocate.set_defaults(handler=_run_relocate)

    fold = subparsers.add_parser(
        "build-fold-input",
        help="Bind physical adaptation tokens and donor scope to a training fold",
    )
    fold.add_argument("--tokenization-manifest", type=Path, required=True)
    fold.add_argument("--fold-table", type=Path, required=True)
    fold.add_argument("--output", type=Path, required=True)
    fold.add_argument("--fold-id", required=True)
    fold.add_argument("--held-out-dataset")
    fold.add_argument(
        "--reference-design", choices=("lodo", "all_healthy"), default="lodo"
    )
    fold.add_argument("--lineage", required=True)
    fold.add_argument("--partition-column", default="outer_role")
    fold.add_argument("--inner-validation-fold", type=int)
    fold.add_argument("--inner-fold-column", default="inner_fold")
    fold.add_argument("--sampler-manifest", type=Path)
    fold.add_argument("--overwrite", action="store_true")
    fold.add_argument("--dry-run", action="store_true")
    fold.set_defaults(handler=_run_fold_input)

    query = subparsers.add_parser(
        "build-query-input",
        help="Bind query tokens to a frozen trained model without adaptation",
    )
    query.add_argument("--tokenization-manifest", type=Path, required=True)
    query.add_argument("--model-manifest", type=Path, required=True)
    query.add_argument("--output", type=Path, required=True)
    query.add_argument("--seed", type=int)
    _add_projection_gp_options(query)
    query.add_argument("--overwrite", action="store_true")
    query.add_argument("--dry-run", action="store_true")
    query.set_defaults(handler=_run_query_input)

    projection = subparsers.add_parser(
        "build-projection-input",
        help=("Bind exact reference, validation, or query tokens to a frozen model"),
    )
    projection.add_argument(
        "--role", choices=("reference", "validation", "query"), required=True
    )
    projection.add_argument("--tokenization-manifest", type=Path, required=True)
    projection.add_argument("--model-manifest", type=Path, required=True)
    projection.add_argument("--output", type=Path, required=True)
    projection.add_argument("--seed", type=int)
    _add_projection_gp_options(projection)
    projection.add_argument("--overwrite", action="store_true")
    projection.add_argument("--dry-run", action="store_true")
    projection.set_defaults(handler=_run_projection_input)
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
