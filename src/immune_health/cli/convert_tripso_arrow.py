"""Convert selected TRIPSO Arrow embedding columns to aligned NPY/Parquet files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from immune_health.tripso_adapter.arrow_bridge import (
    DEFAULT_REQUIRED_METADATA,
    convert_tripso_arrow_embeddings,
    validate_projection_output_for_conversion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dataset", type=Path, required=True)
    parser.add_argument("--projection-output-manifest", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-column",
        action="append",
        required=True,
        help="Exact Arrow vector column to convert; repeat for multiple programs",
    )
    parser.add_argument("--cell-key-column", default="cell_key")
    parser.add_argument(
        "--required-metadata-column",
        action="append",
        help=(
            "Required joined metadata column. If omitted, require dataset, donor, "
            "sample, observation_id, lineage, and canonical fine_type."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    required = tuple(args.required_metadata_column or DEFAULT_REQUIRED_METADATA)
    projection_validation = validate_projection_output_for_conversion(
        args.projection_output_manifest,
        args.arrow_dataset,
        embedding_columns=args.embedding_column,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "arrow_dataset": str(args.arrow_dataset),
                    "projection_output": projection_validation,
                    "cell_metadata": str(args.cell_metadata),
                    "output_dir": str(args.output_dir),
                    "cell_key_column": args.cell_key_column,
                    "embedding_columns": args.embedding_column,
                    "required_metadata_columns": list(required),
                    "alignment": "one-to-one key join; row-count-only is forbidden",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = convert_tripso_arrow_embeddings(
        args.arrow_dataset,
        args.cell_metadata,
        args.output_dir,
        projection_output_manifest=args.projection_output_manifest,
        embedding_columns=args.embedding_column,
        cell_key_column=args.cell_key_column,
        required_metadata_columns=required,
        overwrite=args.overwrite,
    )
    logging.info(
        "Converted %s rows and %s embedding columns",
        payload["n_rows"],
        len(payload["embedding_outputs"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
