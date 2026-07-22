#!/usr/bin/env python3
"""Download or validate the exact full Geneformer checkpoint used by TRIPSO."""

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

from immune_health.provenance import atomic_write_json, sha256_file  # noqa: E402
from immune_health.tripso_adapter.geneformer import (  # noqa: E402
    VALIDATED_GENEFORMER_MODEL,
    VALIDATED_GENEFORMER_REVISION,
    validate_geneformer_root,
    validate_static_embedding_alignment,
)

GENEFORMER_REPOSITORY = "ctheodoris/Geneformer"
# This historical snapshot declares ``license: apache-2.0`` in the model-card
# front matter but does not contain a standalone LICENSE file.  Preserve the
# exact model card and request LICENSE as an optional pattern in case upstream
# later backfills it at the pinned revision.
UPSTREAM_METADATA_PATTERNS = ("README.md", "LICENSE")
REQUIRED_UPSTREAM_METADATA_FILES = ("README.md",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Directory that will contain gf-12L-95M-i4096/. Keep this outside "
            "Git and transfer this directory to Gefion."
        ),
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=REPOSITORY_ROOT / "tripso_code" / "tripso",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help=(
            "Small JSON report; defaults to OUTPUT_ROOT/geneformer_asset_manifest.json"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not access Hugging Face; validate files already under OUTPUT_ROOT",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Ask the Hugging Face client to use its cache without network access",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "repository": GENEFORMER_REPOSITORY,
        "revision": VALIDATED_GENEFORMER_REVISION,
        "model": VALIDATED_GENEFORMER_MODEL,
        "allow_patterns": [
            f"{VALIDATED_GENEFORMER_MODEL}/*",
            *UPSTREAM_METADATA_PATTERNS,
        ],
        "output_root": str(args.output_root.resolve()),
        "validate_only": bool(args.validate_only),
        "primary_analysis_requires_full_model": False,
        "purpose": "full-Geneformer sensitivity analysis",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    output_root = args.output_root.resolve()
    if not args.validate_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is missing; create the environment from "
                "environment.yml before downloading Geneformer"
            ) from exc
        snapshot_download(
            repo_id=GENEFORMER_REPOSITORY,
            revision=VALIDATED_GENEFORMER_REVISION,
            allow_patterns=[
                f"{VALIDATED_GENEFORMER_MODEL}/*",
                *UPSTREAM_METADATA_PATTERNS,
            ],
            local_dir=output_root,
            local_files_only=args.local_files_only,
            max_workers=4,
        )

    validation = validate_geneformer_root(output_root)
    static_embedding = (
        args.vendor_root.resolve()
        / "tripso"
        / "Utils"
        / "gf-12L-95M-i4096_word_embeddings_may2025.pt"
    )
    alignment = validate_static_embedding_alignment(
        geneformer_validation=validation,
        static_embedding_path=static_embedding,
    )
    if not alignment.get("passed"):
        raise RuntimeError(
            "The full checkpoint does not match TRIPSO's bundled static "
            f"initialization: {alignment}"
        )

    model_dir = output_root / VALIDATED_GENEFORMER_MODEL
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model_dir.iterdir())
        if path.is_file()
    }
    upstream_metadata = {
        name: {
            "size_bytes": (output_root / name).stat().st_size,
            "sha256": sha256_file(output_root / name),
        }
        for name in UPSTREAM_METADATA_PATTERNS
        if (output_root / name).is_file()
    }
    missing_metadata = sorted(
        set(REQUIRED_UPSTREAM_METADATA_FILES) - set(upstream_metadata)
    )
    if missing_metadata:
        raise RuntimeError(
            "The pinned Geneformer snapshot is missing required provenance files: "
            f"{missing_metadata}. Re-run without --validate-only."
        )
    manifest = {
        "schema_version": "immune-health-geneformer-asset/v1",
        **plan,
        "source_url": (
            "https://huggingface.co/ctheodoris/Geneformer/tree/"
            f"{VALIDATED_GENEFORMER_REVISION}/{VALIDATED_GENEFORMER_MODEL}"
        ),
        "license": "Apache-2.0 (upstream model repository)",
        "validation": validation,
        "static_embedding_alignment": alignment,
        "files": files,
        "upstream_metadata_files": upstream_metadata,
        "sftp_transfer_root": str(model_dir),
    }
    manifest_output = (
        args.manifest_output.resolve()
        if args.manifest_output is not None
        else output_root / "geneformer_asset_manifest.json"
    )
    atomic_write_json(manifest_output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
