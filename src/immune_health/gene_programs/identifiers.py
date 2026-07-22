"""Auditable mapping between versioned Ensembl IDs and approved symbols."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ENSEMBL_PATTERN = re.compile(r"^(ENSG\d+)(?:\.\d+)?$")


class AmbiguousGeneMappingError(ValueError):
    """Raised when a caller requests a unique mapping despite ambiguity."""


def strip_ensembl_version(identifier: object) -> str:
    """Strip one numeric Ensembl version suffix and validate the identifier."""

    value = str(identifier).strip()
    match = ENSEMBL_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Not a valid human Ensembl gene identifier: {value!r}")
    return match.group(1)


@dataclass(frozen=True)
class GeneMappingResult:
    """Complete mapping rows and explicit ambiguity/loss reports."""

    mapping: pd.DataFrame
    one_to_many: pd.DataFrame
    many_to_one: pd.DataFrame
    summary: dict[str, object]

    def require_unambiguous(
        self,
        *,
        allow_many_to_one: bool = False,
        allow_duplicate_queries_after_strip: bool = False,
    ) -> pd.DataFrame:
        """Return a one-row-per-query mapping or raise instead of dropping genes."""

        problems: list[str] = []
        if not self.one_to_many.empty:
            problems.append(f"{len(self.one_to_many)} one-to-many Ensembl mappings")
        if not allow_many_to_one and not self.many_to_one.empty:
            problems.append(f"{len(self.many_to_one)} many-to-one symbol mappings")
        duplicated = self.mapping.loc[
            self.mapping["duplicate_query_after_version_strip"],
            "stripped_ensembl_id",
        ].nunique()
        if duplicated and not allow_duplicate_queries_after_strip:
            problems.append(
                f"{duplicated} duplicated query IDs after version stripping"
            )
        unmapped = int(self.mapping["status"].eq("unmapped").sum())
        if unmapped:
            problems.append(f"{unmapped} unmapped query genes")
        if problems:
            raise AmbiguousGeneMappingError("; ".join(problems))
        result = self.mapping.loc[
            self.mapping["status"].isin({"mapped_unique", "mapped_many_to_one"}),
            ["query_index", "input_ensembl_id", "stripped_ensembl_id", "symbol"],
        ]
        return result.reset_index(drop=True)

    def write_reports(
        self, output_dir: str | Path, prefix: str = "gene_mapping"
    ) -> dict[str, Path]:
        """Write all mapping rows, ambiguity tables, and a versioned summary."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "mapping": output / f"{prefix}.tsv",
            "one_to_many": output / f"{prefix}.one_to_many.tsv",
            "many_to_one": output / f"{prefix}.many_to_one.tsv",
            "summary": output / f"{prefix}.summary.json",
        }
        self.mapping.to_csv(paths["mapping"], sep="\t", index=False)
        self.one_to_many.to_csv(paths["one_to_many"], sep="\t", index=False)
        self.many_to_one.to_csv(paths["many_to_one"], sep="\t", index=False)
        paths["summary"].write_text(
            json.dumps(self.summary, indent=2, sort_keys=True) + "\n"
        )
        return paths


def map_ensembl_to_symbols(
    query_ids: Iterable[object],
    mapping_resource: pd.DataFrame,
    *,
    resource_version: str,
    ensembl_column: str = "ensembl_id",
    symbol_column: str = "symbol",
) -> GeneMappingResult:
    """Map every query while retaining unmapped and ambiguous mapping rows.

    ``resource_version`` is mandatory so reports cannot be detached from the
    annotation release that created them.  Mapping-resource conflicts are
    represented in the result; no candidate symbol is selected implicitly.
    """

    if not str(resource_version).strip():
        raise ValueError("resource_version must be a nonempty release identifier")
    missing = {ensembl_column, symbol_column} - set(mapping_resource.columns)
    if missing:
        raise ValueError(f"Mapping resource lacks columns: {sorted(missing)}")
    resource = mapping_resource[[ensembl_column, symbol_column]].copy()
    n_resource_rows = len(resource)
    if resource.empty:
        raise ValueError("Mapping resource is empty")
    if resource.isna().any(axis=None):
        raise ValueError("Mapping resource contains missing Ensembl IDs or symbols")
    resource["stripped_ensembl_id"] = resource[ensembl_column].map(
        strip_ensembl_version
    )
    resource["symbol"] = resource[symbol_column].astype("string").str.strip()
    if resource["symbol"].eq("").any():
        raise ValueError("Mapping resource contains empty approved symbols")
    resource = resource[["stripped_ensembl_id", "symbol"]].drop_duplicates()
    n_unique_resource_pairs = len(resource)

    symbols_by_ensembl = (
        resource.groupby("stripped_ensembl_id", observed=True)["symbol"]
        .agg(lambda values: tuple(sorted(map(str, set(values)))))
        .to_dict()
    )
    ensembl_by_symbol = (
        resource.groupby("symbol", observed=True)["stripped_ensembl_id"]
        .agg(lambda values: tuple(sorted(map(str, set(values)))))
        .to_dict()
    )
    all_one_to_many = pd.DataFrame(
        [
            {
                "stripped_ensembl_id": ensembl,
                "candidate_symbols": "|".join(symbols),
                "n_candidate_symbols": len(symbols),
            }
            for ensembl, symbols in sorted(symbols_by_ensembl.items())
            if len(symbols) > 1
        ]
    )
    all_many_to_one = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "candidate_ensembl_ids": "|".join(ensembl_ids),
                "n_candidate_ensembl_ids": len(ensembl_ids),
            }
            for symbol, ensembl_ids in sorted(ensembl_by_symbol.items())
            if len(ensembl_ids) > 1
        ]
    )
    if all_one_to_many.empty:
        all_one_to_many = pd.DataFrame(
            columns=["stripped_ensembl_id", "candidate_symbols", "n_candidate_symbols"]
        )
    if all_many_to_one.empty:
        all_many_to_one = pd.DataFrame(
            columns=["symbol", "candidate_ensembl_ids", "n_candidate_ensembl_ids"]
        )

    queries = [str(value).strip() for value in query_ids]
    stripped = [strip_ensembl_version(value) for value in queries]
    query_ensembl = set(stripped)
    query_symbols = {
        symbol
        for ensembl in query_ensembl
        for symbol in symbols_by_ensembl.get(ensembl, tuple())
    }
    one_to_many = all_one_to_many.loc[
        all_one_to_many["stripped_ensembl_id"].isin(query_ensembl)
    ].reset_index(drop=True)
    many_to_one = all_many_to_one.loc[
        all_many_to_one["symbol"].isin(query_symbols)
    ].reset_index(drop=True)
    duplicated = pd.Series(stripped).duplicated(keep=False).to_numpy()
    rows: list[dict[str, object]] = []
    for index, (original, ensembl, duplicate) in enumerate(
        zip(queries, stripped, duplicated, strict=True)
    ):
        candidates = symbols_by_ensembl.get(ensembl, tuple())
        base = {
            "query_index": index,
            "input_ensembl_id": original,
            "stripped_ensembl_id": ensembl,
            "version_stripped": original != ensembl,
            "duplicate_query_after_version_strip": bool(duplicate),
            "mapping_resource_version": str(resource_version),
        }
        if not candidates:
            rows.append({**base, "symbol": pd.NA, "status": "unmapped"})
            continue
        if len(candidates) > 1:
            rows.extend(
                {**base, "symbol": symbol, "status": "ambiguous_one_to_many"}
                for symbol in candidates
            )
            continue
        symbol = candidates[0]
        status = (
            "mapped_many_to_one"
            if len(ensembl_by_symbol[symbol]) > 1
            else "mapped_unique"
        )
        rows.append({**base, "symbol": symbol, "status": status})
    mapping = pd.DataFrame(rows)
    mapped_query_indices = mapping.loc[
        mapping["status"].ne("unmapped"), "query_index"
    ].nunique()
    summary: dict[str, object] = {
        "mapping_resource_version": str(resource_version),
        "n_mapping_resource_rows": n_resource_rows,
        "n_unique_resource_pairs_after_strip": n_unique_resource_pairs,
        "n_duplicate_resource_pairs_after_strip": (
            n_resource_rows - n_unique_resource_pairs
        ),
        "n_query_ids": len(queries),
        "n_version_suffixes_stripped": sum(
            original != normalized for original, normalized in zip(queries, stripped)
        ),
        "n_unique_ids_after_strip": len(set(stripped)),
        "n_duplicate_query_ids_after_strip": len(stripped) - len(set(stripped)),
        "n_mapped_query_ids": int(mapped_query_indices),
        "n_unmapped_query_ids": int(
            mapping.loc[mapping["status"].eq("unmapped"), "query_index"].nunique()
        ),
        "n_one_to_many_resource_ids": len(all_one_to_many),
        "n_many_to_one_resource_symbols": len(all_many_to_one),
        "n_one_to_many_query_ids": len(one_to_many),
        "n_many_to_one_query_symbols": len(many_to_one),
    }
    return GeneMappingResult(
        mapping=mapping,
        one_to_many=one_to_many,
        many_to_one=many_to_one,
        summary=summary,
    )
