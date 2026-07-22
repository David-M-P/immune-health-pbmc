#!/usr/bin/env python3
"""Validate the real Linux/Python-3.10 TRIPSO environment and optional smoke path."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from immune_health.tripso_adapter.contracts import atomic_write_json  # noqa: E402
from immune_health.tripso_adapter.environment import validate_environment  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=REPOSITORY_ROOT / "tripso_code" / "tripso",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--smoke-mode",
        choices=("none", "mock", "real"),
        default="none",
        help="Mock checks adapters only; real executes --real-smoke-command-json.",
    )
    parser.add_argument(
        "--real-smoke-command-json",
        help='JSON argv, e.g. ["python", "path/to/real_smoke.py"] (no shell).',
    )
    parser.add_argument("--real-smoke-cwd", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--geneformer-root",
        type=Path,
        help=(
            "Optional checkout root containing gf-12L-95M-i4096; enables full-model "
            "config, weight, and static-embedding alignment validation."
        ),
    )
    parser.add_argument("--geneformer-model-name", default="gf-12L-95M-i4096")
    parser.add_argument(
        "--geneformer-expected-hashes-json",
        help=(
            'Optional JSON mapping such as {"config.json":"...",'
            '"model.safetensors":"..."}.'
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    real_command = None
    if args.real_smoke_command_json:
        real_command = json.loads(args.real_smoke_command_json)
        if not isinstance(real_command, list) or not all(
            isinstance(item, str) for item in real_command
        ):
            raise ValueError("--real-smoke-command-json must be a JSON string list")
    expected_hashes = None
    if args.geneformer_expected_hashes_json:
        expected_hashes = json.loads(args.geneformer_expected_hashes_json)
        if not isinstance(expected_hashes, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in expected_hashes.items()
        ):
            raise ValueError(
                "--geneformer-expected-hashes-json must be a JSON string mapping"
            )

    report = validate_environment(
        vendor_root=args.vendor_root,
        smoke_mode=args.smoke_mode,
        real_smoke_command=real_command,
        real_smoke_cwd=args.real_smoke_cwd,
        geneformer_root=args.geneformer_root,
        geneformer_model_name=args.geneformer_model_name,
        geneformer_expected_hashes=expected_hashes,
    )
    if args.json_output:
        atomic_write_json(args.json_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["environment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
