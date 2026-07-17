"""Resolve the three configuration layers and record the result for one run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import yaml

from immune_health.config import (
    iter_placeholders,
    load_resolved_config,
    validate_required_sections,
)

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit paths for the three configuration layers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-config", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Local run root (default: runs)",
    )
    return parser.parse_args(argv)


def validate_run_id(run_id: str) -> None:
    """Keep each resolved file inside exactly one run directory."""
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single non-empty directory name")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve, validate, and save configuration for a development or full run."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_run_id(args.run_id)

    resolved = load_resolved_config(
        common_path=args.common_config,
        cluster_path=args.cluster_config,
        analysis_path=args.analysis_config,
    )
    validate_required_sections(resolved)
    resolved.setdefault("run", {})["id"] = args.run_id
    resolved["configuration_sources"] = {
        "common": str(args.common_config),
        "cluster": str(args.cluster_config),
        "analysis": str(args.analysis_config),
    }

    placeholders = list(iter_placeholders(resolved))
    for location, value in placeholders:
        LOGGER.warning("Placeholder remains at %s: %s", location, value)

    output_path = args.runs_dir / args.run_id / "resolved_config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)

    LOGGER.info("Wrote resolved configuration to %s", output_path)
    LOGGER.info("Found %d explicit placeholder(s)", len(placeholders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
