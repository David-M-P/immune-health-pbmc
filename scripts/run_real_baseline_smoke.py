#!/usr/bin/env python3
"""Run a bounded, read-only B-cell baseline smoke on the audited merged H5AD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from immune_health.smoke import run_real_baseline_smoke  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/data/reference.yaml",
        help="Audited reference-data configuration",
    )
    parser.add_argument(
        "--gene-program-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/gene_programs/default.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-dataset")
    parser.add_argument("--max-donors-per-dataset", type=int, default=2)
    parser.add_argument("--max-cells-per-donor", type=int, default=50)
    parser.add_argument("--n-pca-components", type=int, default=3)
    parser.add_argument("--maximum-gene-programs", type=int, default=3)
    parser.add_argument("--minimum-mapped-genes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the bounded run plan without reading H5AD counts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_type": "real_data_baseline_smoke",
                    "config": str(args.config.resolve()),
                    "gene_program_config": str(args.gene_program_config.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "heldout_dataset": (
                        args.heldout_dataset or "fifth configured dataset"
                    ),
                    "max_donors_per_dataset": args.max_donors_per_dataset,
                    "max_cells_per_donor": args.max_cells_per_donor,
                    "tripso_executed": False,
                    "production_fold_vocabulary_used": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = run_real_baseline_smoke(
        args.output_dir,
        data_config_path=args.config,
        gene_program_config_path=args.gene_program_config,
        heldout_dataset=args.heldout_dataset,
        max_donors_per_dataset=args.max_donors_per_dataset,
        max_cells_per_donor=args.max_cells_per_donor,
        n_pca_components=args.n_pca_components,
        maximum_gene_programs=args.maximum_gene_programs,
        minimum_mapped_genes=args.minimum_mapped_genes,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
