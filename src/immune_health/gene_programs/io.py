"""Generic, dependency-light gene-program resource loading and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

VALID_DIRECTIONS = {
    "undirected",
    "up",
    "down",
    "activation",
    "repression",
    "positive",
    "negative",
}


@dataclass(frozen=True)
class GeneProgram:
    """One named gene program plus provenance and applicability metadata."""

    program_id: str
    genes: tuple[str, ...]
    source: str
    category: str = "unspecified"
    direction: str = "undirected"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        program_id = str(self.program_id).strip()
        source = str(self.source).strip()
        genes = tuple(str(gene).strip() for gene in self.genes)
        if not program_id:
            raise ValueError("Gene-program ID must be nonempty")
        if not source:
            raise ValueError(f"Gene program {program_id!r} has no source")
        if not genes or any(not gene for gene in genes):
            raise ValueError(f"Gene program {program_id!r} has no valid genes")
        duplicated_genes = sorted(gene for gene in set(genes) if genes.count(gene) > 1)
        if duplicated_genes:
            raise ValueError(
                f"Gene program {program_id!r} repeats members: {duplicated_genes}"
            )
        direction = str(self.direction).strip().lower()
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Gene program {program_id!r} has unsupported direction {direction!r}"
            )
        object.__setattr__(self, "program_id", program_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "direction", direction)


def _load_gmt(path: Path, source: str | None) -> tuple[GeneProgram, ...]:
    programs: list[GeneProgram] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        fields = raw_line.rstrip("\n").split("\t")
        if len(fields) < 3:
            raise ValueError(
                f"GMT row {line_number} in {path} has fewer than three fields"
            )
        program_id, description, *genes = fields
        programs.append(
            GeneProgram(
                program_id=program_id,
                genes=tuple(genes),
                source=source or path.stem,
                category=description or "unspecified",
                metadata={"description": description, "resource_path": str(path)},
            )
        )
    return tuple(programs)


def _choose_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def _consistent_value(
    group: pd.DataFrame,
    column: str,
    default: Any,
    *,
    program_id: object,
) -> Any:
    if column not in group:
        return default
    values = group[column].dropna().astype(str).str.strip()
    values = values[values.ne("")].unique()
    if len(values) > 1:
        raise ValueError(
            f"Program {program_id!r} has inconsistent {column!r} metadata: "
            f"{values.tolist()}"
        )
    return values[0] if len(values) else default


def _load_delimited(
    path: Path, delimiter: str, source: str | None
) -> tuple[GeneProgram, ...]:
    frame = pd.read_csv(path, sep=delimiter, comment="#")
    if frame.empty:
        raise ValueError(f"Gene-program resource is empty: {path}")
    program_column = _choose_column(
        frame.columns,
        (
            "program_id",
            "program",
            "name",
            "geneset",
            "gene_set",
            "gs_name",
            "source",
        ),
    )
    gene_column = _choose_column(
        frame.columns,
        ("gene", "gene_symbol", "symbol", "ensembl_id", "target"),
    )
    genes_column = _choose_column(frame.columns, ("genes", "members"))
    if program_column is None or (gene_column is None and genes_column is None):
        raise ValueError(
            "TSV/CSV GP resources require a program_id/program/name column and "
            "either a gene/gene_symbol column or a delimited genes column"
        )
    source_column = _choose_column(frame.columns, ("source",))
    if source_column == program_column:
        # decoupler networks use ``source`` for the program/regulator name,
        # rather than for library provenance.
        source_column = None
    category_column = _choose_column(
        frame.columns,
        ("category", "collection", "gs_collection_name", "gs_collection"),
    )
    direction_column = _choose_column(frame.columns, ("direction",))
    reserved = {
        item
        for item in (
            program_column,
            gene_column,
            genes_column,
            source_column,
            category_column,
            direction_column,
        )
        if item is not None
    }

    programs: list[GeneProgram] = []
    for program_id, group in frame.groupby(program_column, sort=False, dropna=False):
        if pd.isna(program_id) or not str(program_id).strip():
            raise ValueError("GP resource contains an empty program identifier")
        genes: list[str] = []
        if gene_column is not None:
            genes.extend(group[gene_column].dropna().astype(str).tolist())
        if genes_column is not None:
            for value in group[genes_column].dropna().astype(str):
                genes.extend(item for item in re.split(r"[;,|\s]+", value) if item)
        normalized_genes = [str(gene).strip() for gene in genes if str(gene).strip()]
        unique_genes = tuple(dict.fromkeys(normalized_genes))
        metadata: dict[str, Any] = {
            "resource_path": str(path),
            "duplicate_gene_rows_removed": len(normalized_genes) - len(unique_genes),
        }
        for column in frame.columns:
            if column in reserved:
                continue
            # decoupler and MSigDB long tables legitimately contain per-gene
            # weights and identifiers. Only constants are program metadata;
            # row-varying values remain in the versioned source resource.
            values = group[column].dropna().astype(str).str.strip()
            values = values[values.ne("")].unique()
            if len(values) == 1:
                metadata[str(column)] = values[0]
        resource_source = (
            _consistent_value(group, source_column, path.stem, program_id=program_id)
            if source_column is not None
            else path.stem
        )
        program_source = source or resource_source
        category = (
            _consistent_value(
                group,
                category_column,
                "unspecified",
                program_id=program_id,
            )
            if category_column is not None
            else "unspecified"
        )
        direction = (
            _consistent_value(
                group,
                direction_column,
                "undirected",
                program_id=program_id,
            )
            if direction_column is not None
            else "undirected"
        )
        programs.append(
            GeneProgram(
                program_id=str(program_id),
                genes=unique_genes,
                source=program_source,
                category=category,
                direction=direction,
                metadata=metadata,
            )
        )
    return tuple(programs)


def load_gene_programs(
    path: str | Path,
    *,
    format: str | None = None,
    source: str | None = None,
) -> tuple[GeneProgram, ...]:
    """Load GMT or long/wide TSV/CSV gene programs with metadata."""

    resource = Path(path)
    if not resource.exists():
        raise FileNotFoundError(
            f"Gene-program resource not found: {resource}. Configure an explicit "
            "PROGENy, CollecTRI, Hallmark, Reactome, or project resource; test "
            "fixtures must not substitute for production data."
        )
    selected_format = (format or resource.suffix.lstrip(".")).lower()
    if selected_format == "gmt":
        programs = _load_gmt(resource, source)
    elif selected_format in {"tsv", "txt"}:
        programs = _load_delimited(resource, "\t", source)
    elif selected_format == "csv":
        programs = _load_delimited(resource, ",", source)
    else:
        raise ValueError(f"Unsupported GP format {selected_format!r}; use GMT/TSV/CSV")
    validate_gene_programs(programs)
    return programs


def validate_gene_programs(
    programs: Iterable[GeneProgram],
    *,
    minimum_size: int = 1,
    maximum_size: int | None = None,
) -> pd.DataFrame:
    """Validate names and membership, returning a reviewable summary."""

    programs = tuple(programs)
    if not programs:
        raise ValueError("Gene-program library is empty")
    identifiers = [program.program_id for program in programs]
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate gene-program IDs: {duplicates}")
    rows: list[dict[str, object]] = []
    for program in programs:
        size = len(program.genes)
        if size < minimum_size:
            raise ValueError(
                f"Gene program {program.program_id!r} has {size} genes; "
                f"minimum is {minimum_size}"
            )
        if maximum_size is not None and size > maximum_size:
            raise ValueError(
                f"Gene program {program.program_id!r} has {size} genes; "
                f"maximum is {maximum_size}"
            )
        rows.append(
            {
                "program_id": program.program_id,
                "source": program.source,
                "category": program.category,
                "direction": program.direction,
                "n_genes": size,
                "n_unique_genes": len(set(program.genes)),
            }
        )
    return pd.DataFrame(rows).sort_values("program_id").reset_index(drop=True)


def validate_gp_resource(
    path: str | Path,
    *,
    format: str | None = None,
    source: str | None = None,
    production: bool = True,
) -> tuple[GeneProgram, ...]:
    """Load a configured resource and refuse synthetic fixtures in production."""

    programs = load_gene_programs(path, format=format, source=source)
    if production:
        synthetic = [
            program.program_id
            for program in programs
            if str(program.metadata.get("test_fixture", "")).lower()
            in {"1", "true", "yes"}
            or "synthetic" in Path(path).name.lower()
        ]
        if synthetic:
            raise ValueError(
                f"Synthetic GP fixtures cannot be used in production: {synthetic[:5]}"
            )
    return programs
