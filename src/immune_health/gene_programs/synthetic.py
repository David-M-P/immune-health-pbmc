"""Tiny deterministic gene-program fixtures for tests and smoke runs only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from immune_health.gene_programs.io import GeneProgram


def synthetic_gene_programs() -> tuple[GeneProgram, ...]:
    """Return a small library explicitly marked as a non-production fixture."""

    metadata = {"test_fixture": True, "lineages": "B cells|CD4_like"}
    return (
        GeneProgram(
            "synthetic_age_up",
            ("GENE1", "GENE2", "GENE3"),
            source="synthetic_test_fixture",
            category="immune_age",
            direction="up",
            metadata=metadata,
        ),
        GeneProgram(
            "synthetic_age_down",
            ("GENE3", "GENE4", "GENE5"),
            source="synthetic_test_fixture",
            category="immune_age",
            direction="down",
            metadata=metadata,
        ),
        GeneProgram(
            "synthetic_control",
            ("GENE6", "GENE7"),
            source="synthetic_test_fixture",
            category="control",
            metadata={"test_fixture": True},
        ),
    )


def synthetic_gene_mapping() -> pd.DataFrame:
    """Return versioned Ensembl-to-symbol fixture rows."""

    return pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}.1" for i in range(1, 8)],
            "symbol": [f"GENE{i}" for i in range(1, 8)],
        }
    )


def write_synthetic_gp_library(
    directory: str | Path,
    *,
    format: str = "tsv",
) -> Path:
    """Write a tiny fixture, never a fallback for missing production resources."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    programs = synthetic_gene_programs()
    selected_format = format.lower()
    if selected_format == "gmt":
        path = output / "synthetic_gene_programs.gmt"
        lines = [
            "\t".join((program.program_id, program.category, *program.genes))
            for program in programs
        ]
        path.write_text("\n".join(lines) + "\n")
        return path
    if selected_format not in {"tsv", "csv"}:
        raise ValueError("Synthetic GP fixture format must be GMT, TSV, or CSV")
    path = output / f"synthetic_gene_programs.{selected_format}"
    rows = [
        {
            "program_id": program.program_id,
            "gene": gene,
            "source": program.source,
            "category": program.category,
            "direction": program.direction,
            "test_fixture": True,
            "lineages": program.metadata.get("lineages"),
        }
        for program in programs
        for gene in program.genes
    ]
    pd.DataFrame(rows).to_csv(
        path, sep="\t" if selected_format == "tsv" else ",", index=False
    )
    return path
