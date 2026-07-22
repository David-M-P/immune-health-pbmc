#!/usr/bin/env python3
"""Generate the read-only data audit required before modelling."""

from __future__ import annotations

import argparse
from pathlib import Path

from immune_health.data.audit import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Reserved for central CLI parity")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Read-only intermediate_data directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Audit is deterministic")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print intended outputs without reading H5AD metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    required = [args.data_root, args.provenance]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required audit inputs do not exist: {missing}")
    if args.dry_run:
        print(f"Would audit read-only data root: {args.data_root.resolve()}")
        print(f"Would read provenance: {args.provenance.resolve()}")
        print(f"Would write audit reports: {args.output_dir.resolve()}")
        return 0
    run_audit(
        data_root=args.data_root,
        output_dir=args.output_dir,
        provenance_path=args.provenance,
        repo_root=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
