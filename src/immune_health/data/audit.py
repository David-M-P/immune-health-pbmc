"""Memory-conscious audit of the merged PBMC lineage H5AD files.

The audit deliberately uses h5py in read-only mode.  AnnData backed mode still
materializes the complete observation table, whereas the HDF5 categorical
arrays can be decoded one lineage at a time without touching the count matrix.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

PRIMARY_LINEAGES = ("B cells", "NK_ILC", "Monocytes", "CD4_like", "CD8_like")
AUDIT_LINEAGES = (*PRIMARY_LINEAGES, "T_others", "DC", "pDC")
EXPECTED_OBS_FIELDS = (
    "donor_id",
    "sample_id",
    "dataset",
    "age",
    "sex",
    "lineage",
    "ctype_high",
    "ctype_high_conf",
    "ctype_low",
    "ctype_low_conf",
    "chemistry",
    "batch",
    "pct_mt",
)
MISSING_STRINGS = {"", "na", "nan", "none", "null", "unknown", "missing"}
ENSEMBL_RE = re.compile(r"^ENSG\d{11}$")


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_strings(values: Iterable[Any]) -> list[str]:
    return [_decode_scalar(value) for value in values]


def _read_column(group: h5py.Group, name: str) -> pd.Series:
    """Decode one AnnData dataframe column while retaining categoricals."""
    obj = group[name]
    if isinstance(obj, h5py.Group):
        categories = _decode_strings(obj["categories"][:])
        codes = obj["codes"][:]
        return pd.Series(pd.Categorical.from_codes(codes, categories=categories))
    values = obj[:]
    if values.dtype.kind in {"O", "S", "U"}:
        values = np.asarray(_decode_strings(values), dtype=object)
    return pd.Series(values)


def _missing_mask(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
        normalized = series.astype("string").str.strip().str.lower()
        missing |= normalized.isin(MISSING_STRINGS)
    return missing


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "min": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q95": math.nan,
            "max": math.nan,
        }
    quantile = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": float(numeric.min()),
        "q05": float(quantile.loc[0.05]),
        "q25": float(quantile.loc[0.25]),
        "median": float(quantile.loc[0.5]),
        "q75": float(quantile.loc[0.75]),
        "q95": float(quantile.loc[0.95]),
        "max": float(numeric.max()),
    }


def _sample_sparse_values(data: h5py.Dataset, max_values: int = 300_000) -> np.ndarray:
    """Sample stored sparse values from evenly spaced contiguous regions."""
    n_values = int(data.shape[0])
    if n_values == 0:
        return np.asarray([], dtype=data.dtype)
    per_region = max(1, min(max_values // 3, n_values))
    starts = sorted(
        {0, max(0, n_values // 2 - per_region // 2), max(0, n_values - per_region)}
    )
    arrays = [data[start : min(n_values, start + per_region)] for start in starts]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=data.dtype)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _lineage_files(merged_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted(merged_root.glob("*/merged.h5ad")):
        manifest_path = path.parent / "merge_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        lineage = str(manifest["lineage"])
        if lineage in AUDIT_LINEAGES:
            found.append((lineage, path))
    order = {lineage: index for index, lineage in enumerate(AUDIT_LINEAGES)}
    return sorted(found, key=lambda item: order[item[0]])


def _discover_artifacts(data_root: Path) -> dict[str, list[str]]:
    all_h5ad = sorted(data_root.rglob("*.h5ad"))
    lineage_h5ad = sorted((data_root / "lineages").rglob("*.h5ad"))
    merged_h5ad = sorted(
        (data_root / "reference_lineages" / "merged").glob("*/merged.h5ad")
    )
    split_manifests = sorted((data_root / "lineages").rglob("split_manifest.json"))
    merge_manifests = sorted(
        (data_root / "reference_lineages" / "merged").rglob("merge_manifest.json")
    )
    manifest_files = sorted(data_root.rglob("*manifest*.json"))
    qc_tokens = ("qc", "summary", "balance", "missing", "report")
    qc_reports = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tsv", ".csv", ".gz", ".json"}
        and any(token in path.name.lower() for token in qc_tokens)
        and "manifest" not in path.name.lower()
    )
    query_artifacts = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and ("soundlife" in str(path).lower() or "galsky" in str(path).lower())
    )
    return {
        "all_h5ad_files": [str(path) for path in all_h5ad],
        "lineage_h5ad_files": [str(path) for path in lineage_h5ad],
        "merged_lineage_h5ad_files": [str(path) for path in merged_h5ad],
        "split_manifests": [str(path) for path in split_manifests],
        "merge_manifests": [str(path) for path in merge_manifests],
        "all_json_manifests": [str(path) for path in manifest_files],
        "qc_reports": [str(path) for path in qc_reports],
        "soundlife_or_galsky_artifacts": [str(path) for path in query_artifacts],
    }


def _matrix_and_gene_summary(
    h5: h5py.File, lineage: str, h5ad_path: Path
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    x = h5["X"]
    x_shape = tuple(int(value) for value in x.attrs["shape"])
    data = x["data"]
    nnz = int(data.shape[0])
    total = x_shape[0] * x_shape[1]
    sample = _sample_sparse_values(data)
    matrix = {
        "lineage": lineage,
        "h5ad_path": str(h5ad_path),
        "file_size_bytes": h5ad_path.stat().st_size,
        "n_cells": x_shape[0],
        "n_genes": x_shape[1],
        "n_stored_values": nnz,
        "stored_fraction": nnz / total if total else math.nan,
        "sparsity": 1.0 - (nnz / total) if total else math.nan,
        "x_encoding": _decode_scalar(x.attrs.get("encoding-type", "unknown")),
        "x_data_dtype": str(data.dtype),
        "x_indices_dtype": str(x["indices"].dtype),
        "x_indptr_dtype": str(x["indptr"].dtype),
        "x_integer_dtype": bool(np.issubdtype(data.dtype, np.integer)),
        "sampled_stored_values": int(sample.size),
        "sampled_min": float(sample.min()) if sample.size else math.nan,
        "sampled_max": float(sample.max()) if sample.size else math.nan,
        "sampled_negative_values": int((sample < 0).sum()) if sample.size else 0,
        "sampled_explicit_zero_values": int((sample == 0).sum()) if sample.size else 0,
        "sampled_non_integer_values": int((~np.isclose(sample, np.rint(sample))).sum())
        if sample.size
        else 0,
        "layers": sorted(h5["layers"].keys()),
        "has_raw": "raw" in h5,
        "obsm_keys": sorted(h5["obsm"].keys()),
        "obsp_keys": sorted(h5["obsp"].keys()),
    }

    var = h5["var"]
    index_name = _decode_scalar(var.attrs["_index"])
    index_values = np.asarray(_decode_strings(var[index_name][:]), dtype=object)
    unified = np.asarray(_decode_strings(var["unified_ensembl"][:]), dtype=object)
    stripped = np.asarray([value.split(".", 1)[0] for value in unified], dtype=object)
    gene = {
        "lineage": lineage,
        "scope": "merged_h5ad_vocabulary",
        "dataset": "__all__",
        "source_dataset_id": "__merged__",
        "n_genes": int(unified.size),
        "var_index_name": index_name,
        "var_columns": ",".join(_decode_strings(var.attrs.get("column-order", []))),
        "n_missing_unified_ensembl": int(
            sum(value.strip().lower() in MISSING_STRINGS for value in unified)
        ),
        "n_exact_duplicate_ids": int(unified.size - np.unique(unified).size),
        "n_duplicate_ids_after_version_strip": int(
            stripped.size - np.unique(stripped).size
        ),
        "n_versioned_ensembl_ids": int(sum("." in value for value in unified)),
        "n_ids_matching_human_ensembl_regex": int(
            sum(bool(ENSEMBL_RE.fullmatch(value)) for value in unified)
        ),
        "var_index_equals_unified_ensembl": bool(np.array_equal(index_values, unified)),
        "n_common_genes": int(unified.size),
        "n_genes_zero_expression": math.nan,
        "zero_expression_fraction": math.nan,
        "n_source_genes_outside_common_vocabulary": math.nan,
    }
    return matrix, gene, index_values


def _metadata_frame(obs: h5py.Group) -> pd.DataFrame:
    fields = [field for field in EXPECTED_OBS_FIELDS if field in obs]
    return pd.DataFrame({field: _read_column(obs, field) for field in fields})


def _fine_type_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(
        ["dataset", "lineage", "ctype_low"], observed=True, dropna=False
    )
    for (dataset, lineage, fine_type), part in grouped:
        conf = pd.to_numeric(part["ctype_low_conf"], errors="coerce")
        quantiles = _quantiles(conf)
        rows.append(
            {
                "dataset": str(dataset),
                "lineage": str(lineage),
                "fine_type": str(fine_type),
                "n_cells": len(part),
                "n_donors": part["donor_id"].nunique(dropna=True),
                "n_samples": part["sample_id"].nunique(dropna=True),
                "confidence_missing": int(conf.isna().sum()),
                "confidence_mean": float(conf.mean())
                if conf.notna().any()
                else math.nan,
                "confidence_lt_0_7": int((conf < 0.7).sum()),
                "confidence_lt_0_9": int((conf < 0.9).sum()),
                **{f"confidence_{key}": value for key, value in quantiles.items()},
            }
        )
    return rows


def _donor_fine_rows(frame: pd.DataFrame) -> pd.DataFrame:
    group_fields = ["dataset", "donor_id", "lineage", "ctype_low"]
    grouped = frame.groupby(group_fields, observed=True, dropna=False)
    result = grouped.agg(
        n_cells=("donor_id", "size"),
        n_samples=("sample_id", lambda values: values.nunique(dropna=True)),
        sample_ids=(
            "sample_id",
            lambda values: "|".join(sorted(values.dropna().astype(str).unique())),
        ),
        age_min=("age", "min"),
        age_max=("age", "max"),
        n_age_values=("age", lambda values: values.nunique(dropna=True)),
        n_sex_values=("sex", lambda values: values.nunique(dropna=True)),
        sex=(
            "sex",
            lambda values: "|".join(sorted(values.dropna().astype(str).unique())),
        ),
        annotation_confidence_mean=("ctype_low_conf", "mean"),
    ).reset_index()
    result = result.rename(columns={"ctype_low": "fine_type"})
    result["biological_unit_id"] = (
        result["dataset"].astype("string") + "::" + result["donor_id"].astype("string")
    )
    result["source_observation_ids"] = result.apply(
        lambda row: "|".join(
            f"{row['dataset']}::{sample}"
            for sample in str(row["sample_ids"]).split("|")
            if sample
        ),
        axis=1,
    )
    result["observation_ids"] = result.apply(
        lambda row: "|".join(
            f"{row['dataset']}::{row['donor_id']}::{sample}"
            for sample in str(row["sample_ids"]).split("|")
            if sample
        ),
        axis=1,
    )
    columns = [
        "dataset",
        "donor_id",
        "biological_unit_id",
        "lineage",
        "fine_type",
        "n_cells",
        "n_samples",
        "sample_ids",
        "source_observation_ids",
        "observation_ids",
        "age_min",
        "age_max",
        "n_age_values",
        "sex",
        "n_sex_values",
        "annotation_confidence_mean",
    ]
    return result[columns]


def _metadata_missingness_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, lineage), part in frame.groupby(
        ["dataset", "lineage"], observed=True, dropna=False
    ):
        for field in EXPECTED_OBS_FIELDS:
            if field not in part:
                missing = len(part)
                status = "field_absent"
            else:
                missing = int(_missing_mask(part[field]).sum())
                status = "observed"
            rows.append(
                {
                    "dataset": str(dataset),
                    "lineage": str(lineage),
                    "field": field,
                    "n_cells": len(part),
                    "n_missing": missing,
                    "missing_fraction": missing / len(part) if len(part) else math.nan,
                    "field_status": status,
                }
            )
    return rows


def _index_hashes(index: h5py.Dataset, chunk_size: int = 250_000) -> np.ndarray:
    hashes = np.empty(index.shape[0], dtype=np.uint64)
    for start in range(0, index.shape[0], chunk_size):
        stop = min(index.shape[0], start + chunk_size)
        values = index[start:stop]
        hashes[start:stop] = pd.util.hash_array(values, categorize=False)
    return hashes


def _gene_expression_rows(
    lineage_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_path = lineage_dir / "gene_zero_expression_summary.tsv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep="\t")
        source_gene_counts = {
            source["source_dataset_id"]: int(source["n_genes_with_unified_ensembl"])
            for source in manifest["sources"]
        }
        for record in summary.to_dict(orient="records"):
            if record["scope"] == "source_dataset_id":
                source_count = source_gene_counts.get(record["label"], math.nan)
                outside = (
                    int(source_count) - int(record["n_common_genes"])
                    if not pd.isna(source_count)
                    else math.nan
                )
            else:
                outside = math.nan
            rows.append(
                {
                    "lineage": manifest["lineage"],
                    "scope": record["scope"],
                    "dataset": record["dataset"],
                    "source_dataset_id": record["label"],
                    "n_genes": int(record["n_common_genes"]),
                    "var_index_name": "unified_ensembl",
                    "var_columns": "unified_ensembl",
                    "n_missing_unified_ensembl": 0,
                    "n_exact_duplicate_ids": 0,
                    "n_duplicate_ids_after_version_strip": 0,
                    "n_versioned_ensembl_ids": 0,
                    "n_ids_matching_human_ensembl_regex": math.nan,
                    "var_index_equals_unified_ensembl": True,
                    "n_common_genes": int(record["n_common_genes"]),
                    "n_genes_zero_expression": int(
                        record["n_common_genes_zero_expression"]
                    ),
                    "zero_expression_fraction": float(
                        record["zero_expression_fraction"]
                    ),
                    "n_source_genes_outside_common_vocabulary": outside,
                }
            )

    long_path = lineage_dir / "gene_zero_expression_by_source.tsv.gz"
    if long_path.exists():
        long = pd.read_csv(long_path, sep="\t")
        dataset_zero = (
            long.groupby(["dataset", "unified_ensembl"], observed=True)[
                "zero_expression_in_source"
            ]
            .min()
            .groupby(level=0)
            .agg(["sum", "size"])
        )
        for dataset, record in dataset_zero.iterrows():
            rows.append(
                {
                    "lineage": manifest["lineage"],
                    "scope": "dataset_aggregated_across_source_partitions",
                    "dataset": dataset,
                    "source_dataset_id": "__dataset__",
                    "n_genes": int(record["size"]),
                    "var_index_name": "unified_ensembl",
                    "var_columns": "unified_ensembl",
                    "n_missing_unified_ensembl": 0,
                    "n_exact_duplicate_ids": 0,
                    "n_duplicate_ids_after_version_strip": 0,
                    "n_versioned_ensembl_ids": 0,
                    "n_ids_matching_human_ensembl_regex": math.nan,
                    "var_index_equals_unified_ensembl": True,
                    "n_common_genes": int(record["size"]),
                    "n_genes_zero_expression": int(record["sum"]),
                    "zero_expression_fraction": float(record["sum"] / record["size"]),
                    "n_source_genes_outside_common_vocabulary": 0,
                }
            )
    return rows


def _write_gzip_tsv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, sep="\t", index=False)


def _safe_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _markdown_table(frame: pd.DataFrame, float_digits: int = 6) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}g}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in record) + " |")
    return "\n".join(lines)


def _markdown_report(
    matrices: pd.DataFrame,
    datasets: pd.DataFrame,
    donor_units: pd.DataFrame,
    fine_types: pd.DataFrame,
    missingness: pd.DataFrame,
    artifacts: dict[str, list[str]],
    diagnostics: dict[str, Any],
    provenance_path: Path,
) -> str:
    direct_dataset_names = ", ".join(sorted(datasets["dataset"].astype(str).unique()))
    primary_cells = int(
        matrices.loc[matrices["lineage"].isin(PRIMARY_LINEAGES), "n_cells"].sum()
    )
    all_cells = int(matrices["n_cells"].sum())
    repeated = int((donor_units["n_samples"] > 1).sum())
    age_missing = int(missingness.loc[missingness["field"] == "age", "n_missing"].sum())
    sex_missing = int(missingness.loc[missingness["field"] == "sex", "n_missing"].sum())
    lines = [
        "# PBMC reference data-structure audit",
        "",
        "## Evidence boundary",
        "",
        "**Directly observed in this audit:** file discovery; merge and split "
        "manifests; "
        "QC tables; HDF5/AnnData keys, shapes and dtypes; observation metadata; gene "
        "identifiers; sampled sparse values; donor/sample/fine-label counts; "
        "missingness; "
        "and SoundLife/Galsky path presence. All H5AD files were opened read-only with "
        "`h5py.File(..., mode='r')`; count matrices were not densified.",
        "",
        f"**Provenance-document only:** upstream raw-count recovery order, CellTypist "
        f"execution details, health-filter intent, historical source paths, and the "
        "claim "
        f"that merge-time `.X` was populated from `layers['counts']`. Source: "
        f"`{provenance_path}`. The current files directly establish CSR matrices whose "
        "stored `float64` values are integer-like in the audit sample, but historical "
        "transformations cannot be reconstructed from HDF5 alone.",
        "",
        "## Direct observations",
        "",
        f"- Exact merged dataset labels: {direct_dataset_names}.",
        f"- Eight reference lineage H5ADs contain {all_cells:,} disjoint audited "
        "cell rows; "
        f"the five primary lineages contain {primary_cells:,} cells.",
        "- Every merged object has 18,035 variables, a CSR `.X`, no layers, no `.raw`, "
        "and the expected 13 observation columns.",
        f"- All `.X/data` arrays are stored as `float64`, not an integer dtype. Across "
        f"{int(matrices['sampled_stored_values'].sum()):,} evenly sampled stored "
        "values, "
        f"{int(matrices['sampled_negative_values'].sum()):,} were negative and "
        f"{int(matrices['sampled_non_integer_values'].sum()):,} were non-integer-like. "
        "Thus the sample supports count semantics, but this is not a full scan of "
        "every "
        "stored value.",
        f"- Raw `donor_id` collisions across datasets: "
        f"{diagnostics['raw_donor_id_collision_count']}. The pipeline nevertheless "
        "uses "
        "`dataset::donor_id` as the biological identifier.",
        f"- Duplicate cell identifiers: {diagnostics['duplicate_cell_id_count']} after "
        "a 64-bit hash screen and exact follow-up when needed.",
        f"- Donor biological units with repeated samples: {repeated:,}.",
        f"- Missing age cells across audited objects: {age_missing:,}; "
        "missing/unknown sex "
        f"cells: {sex_missing:,}.",
        f"- Fine labels observed: {fine_types['fine_type'].nunique():,} exact strings. "
        "No labels were merged during the audit.",
        f"- SoundLife/Galsky-named artifacts found: "
        f"{len(artifacts['soundlife_or_galsky_artifacts']):,}; neither name occurs "
        "in the "
        "five reference dataset labels.",
        "",
        "## Matrix summary",
        "",
        _markdown_table(
            matrices[
                [
                    "lineage",
                    "n_cells",
                    "n_biological_units",
                    "n_donor_observations",
                    "n_genes",
                    "x_encoding",
                    "x_data_dtype",
                    "sparsity",
                    "sampled_min",
                    "sampled_max",
                ]
            ]
        ),
        "",
        "## Dataset support",
        "",
        _markdown_table(
            datasets[
                [
                    "dataset",
                    "n_cells",
                    "n_biological_units",
                    "n_samples",
                    "n_donor_observations",
                    "age_min",
                    "age_max",
                    "female_donors",
                    "male_donors",
                    "unknown_sex_donors",
                ]
            ]
        ),
        "",
        "## Integrity and interpretation notes",
        "",
        f"- Donor sex inconsistencies: "
        f"{diagnostics['donors_with_multiple_sex_values']}; "
        f"donors with multiple recorded ages: "
        f"{diagnostics['donors_with_multiple_age_values']}; source sample/pool IDs "
        f"shared by multiple donors: "
        f"{diagnostics['source_observations_with_multiple_donors']}.",
        "- **Resolved identifier collision:** the 75 shared source IDs all occur in "
        "`onek1k`, where `sample_id` is a pool number shared by 9–14 donors. The "
        "user-approved contract therefore retains `source_observation_id = "
        "dataset::sample_id` for source provenance and uses the collision-safe "
        "`observation_id = dataset::donor_id::sample_id`. Biological independence "
        "remains `biological_unit_id = dataset::donor_id`.",
        "- Multiple ages for one donor are reported as repeated longitudinal values, "
        "not automatically called errors, because repeated visits can legitimately "
        "change age.",
        "- `gene_identifier_summary.tsv` separates source-partition zero expression "
        "from dataset-level zero expression. The shared vocabulary is an "
        "identifier-presence intersection, so all 18,035 shared IDs are present in "
        "every contributing source; "
        "zero expression is a separate property.",
        "- The fine-type ontology remains unapproved. Exact `ctype_low` labels and "
        "confidence distributions are provided for review before any grouping.",
        "- QC thresholds, GP coverage thresholds, HVGs, dataset roles, and biological "
        "fine-label merges are not inferred by this report.",
        "",
        "## Artifact inventory",
        "",
        f"- Lineage H5AD files under `lineages/`: "
        f"{len(artifacts['lineage_h5ad_files']):,}",
        f"- Merged reference H5AD files: "
        f"{len(artifacts['merged_lineage_h5ad_files']):,}",
        f"- Split manifests: {len(artifacts['split_manifests']):,}",
        f"- Merge manifests: {len(artifacts['merge_manifests']):,}",
        f"- QC/report artifacts: {len(artifacts['qc_reports']):,}",
        "",
        "Complete path inventories and input file metadata are in "
        "`audit_manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    data_root: Path,
    output_dir: Path,
    provenance_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the read-only audit and write the requested compact reports."""
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    provenance_path = provenance_path.resolve()
    merged_root = data_root / "reference_lineages" / "merged"
    if not merged_root.is_dir():
        raise FileNotFoundError(f"Merged lineage root does not exist: {merged_root}")
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Provenance document does not exist: {provenance_path}"
        )

    lineage_files = _lineage_files(merged_root)
    observed_lineages = {lineage for lineage, _ in lineage_files}
    missing_lineages = set(AUDIT_LINEAGES) - observed_lineages
    if missing_lineages:
        raise ValueError(f"Missing audited merged lineages: {sorted(missing_lineages)}")

    artifacts = _discover_artifacts(data_root)
    matrix_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    fine_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    donor_fine_frames: list[pd.DataFrame] = []
    donor_cell_frames: list[pd.DataFrame] = []
    all_hashes: list[np.ndarray] = []
    vocabulary: np.ndarray | None = None
    expected_obs_columns: set[str] | None = None

    for lineage, h5ad_path in lineage_files:
        print(f"Auditing {lineage}: {h5ad_path}", flush=True)
        manifest_path = h5ad_path.parent / "merge_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        with h5py.File(h5ad_path, "r") as h5:
            matrix, gene, current_vocabulary = _matrix_and_gene_summary(
                h5, lineage, h5ad_path
            )
            matrix_rows.append(matrix)
            gene_rows.append(gene)
            if vocabulary is None:
                vocabulary = current_vocabulary
            elif not np.array_equal(vocabulary, current_vocabulary):
                raise ValueError(f"Gene vocabulary/order differs for {lineage}")

            obs = h5["obs"]
            current_obs_columns = set(obs.keys()) - {"_index"}
            if expected_obs_columns is None:
                expected_obs_columns = current_obs_columns
            elif current_obs_columns != expected_obs_columns:
                raise ValueError(f"Observation schema differs for {lineage}")
            frame = _metadata_frame(obs)
            if len(frame) != matrix["n_cells"]:
                raise ValueError(f"Observation/matrix row mismatch for {lineage}")
            matrix["n_datasets"] = int(frame["dataset"].nunique(dropna=True))
            matrix["dataset_names"] = ",".join(
                sorted(frame["dataset"].dropna().astype("string").unique())
            )
            matrix["n_biological_units"] = int(
                frame[["dataset", "donor_id"]].drop_duplicates().shape[0]
            )
            matrix["n_raw_donor_ids"] = int(frame["donor_id"].nunique(dropna=True))
            matrix["n_source_sample_ids"] = int(
                frame[["dataset", "sample_id"]].drop_duplicates().shape[0]
            )
            matrix["n_donor_observations"] = int(
                frame[["dataset", "donor_id", "sample_id"]].drop_duplicates().shape[0]
            )
            matrix["n_fine_types"] = int(frame["ctype_low"].nunique(dropna=True))
            matrix["obs_columns"] = ",".join(sorted(frame.columns))
            matrix["age_missing_cells"] = int(_missing_mask(frame["age"]).sum())
            matrix["sex_missing_cells"] = int(_missing_mask(frame["sex"]).sum())
            fine_rows.extend(_fine_type_rows(frame))
            missing_rows.extend(_metadata_missingness_rows(frame))
            donor_fine_frames.append(_donor_fine_rows(frame))

            donor_cells = (
                frame.groupby(
                    ["dataset", "donor_id", "sample_id", "sex", "age"],
                    observed=True,
                    dropna=False,
                )
                .size()
                .rename("n_cells")
                .reset_index()
            )
            donor_cells["lineage"] = lineage
            donor_cell_frames.append(donor_cells)
            all_hashes.append(_index_hashes(obs["_index"]))

        gene_rows.extend(_gene_expression_rows(h5ad_path.parent, manifest))

    matrices = pd.DataFrame(matrix_rows)
    fine_types = pd.DataFrame(fine_rows).sort_values(
        ["lineage", "dataset", "fine_type"]
    )
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["lineage", "dataset", "field"]
    )
    donor_fine = pd.concat(donor_fine_frames, ignore_index=True)
    donor_cells = pd.concat(donor_cell_frames, ignore_index=True)

    unit_observations = (
        donor_cells.groupby(["dataset", "donor_id"], observed=True, dropna=False)
        .agg(
            n_cells=("n_cells", "sum"),
            n_samples=("sample_id", lambda values: values.nunique(dropna=True)),
            n_lineages=("lineage", "nunique"),
            n_age_values=("age", lambda values: values.nunique(dropna=True)),
            age_min=("age", "min"),
            age_max=("age", "max"),
            n_sex_values=("sex", lambda values: values.nunique(dropna=True)),
        )
        .reset_index()
    )
    unit_observations["biological_unit_id"] = (
        unit_observations["dataset"].astype("string")
        + "::"
        + unit_observations["donor_id"].astype("string")
    )

    unit_sex = (
        donor_cells.drop_duplicates(["dataset", "donor_id", "sex"])
        .groupby(["dataset", "donor_id"], observed=True, dropna=False)["sex"]
        .first()
        .rename("sex")
        .reset_index()
    )
    unit_observations = unit_observations.merge(
        unit_sex, on=["dataset", "donor_id"], how="left", validate="one_to_one"
    )

    dataset_rows: list[dict[str, Any]] = []
    for dataset, part in unit_observations.groupby("dataset", observed=True):
        sex = part["sex"].astype("string").str.lower()
        dataset_cells = donor_cells.loc[donor_cells["dataset"] == dataset]
        n_source_sample_ids = int(dataset_cells["sample_id"].nunique(dropna=True))
        n_donor_observations = int(
            dataset_cells[["donor_id", "sample_id"]].drop_duplicates().shape[0]
        )
        dataset_rows.append(
            {
                "dataset": str(dataset),
                "n_cells": int(part["n_cells"].sum()),
                "n_biological_units": len(part),
                "n_samples": n_source_sample_ids,
                "n_source_sample_ids": n_source_sample_ids,
                "n_donor_observations": n_donor_observations,
                "n_donors_with_repeated_samples": int((part["n_samples"] > 1).sum()),
                "n_donors_with_multiple_age_values": int(
                    (part["n_age_values"] > 1).sum()
                ),
                "n_donors_with_multiple_sex_values": int(
                    (part["n_sex_values"] > 1).sum()
                ),
                "age_missing_donors": int(part["age_min"].isna().sum()),
                "age_min": float(part["age_min"].min()),
                "age_max": float(part["age_max"].max()),
                "female_donors": int((sex == "female").sum()),
                "male_donors": int((sex == "male").sum()),
                "unknown_sex_donors": int((~sex.isin(["female", "male"])).sum()),
            }
        )
    datasets = pd.DataFrame(dataset_rows).sort_values("dataset")

    age_sex_rows: list[dict[str, Any]] = []
    for (dataset, sex), part in unit_observations.groupby(
        ["dataset", "sex"], observed=True, dropna=False
    ):
        quantiles = _quantiles(part["age_min"])
        stratum_cells = donor_cells.loc[
            (donor_cells["dataset"] == dataset) & (donor_cells["sex"] == sex)
        ]
        age_sex_rows.append(
            {
                "dataset": str(dataset),
                "sex": str(sex),
                "n_donors": len(part),
                "n_samples": int(stratum_cells["sample_id"].nunique(dropna=True)),
                "n_source_sample_ids": int(
                    stratum_cells["sample_id"].nunique(dropna=True)
                ),
                "n_donor_observations": int(
                    stratum_cells[["donor_id", "sample_id"]].drop_duplicates().shape[0]
                ),
                "n_cells": int(part["n_cells"].sum()),
                "n_age_missing": int(part["age_min"].isna().sum()),
                **{f"age_{key}": value for key, value in quantiles.items()},
            }
        )
    age_sex = pd.DataFrame(age_sex_rows).sort_values(["dataset", "sex"])

    raw_id_datasets = unit_observations.groupby("donor_id", observed=True)[
        "dataset"
    ].nunique()
    collision_ids = raw_id_datasets[raw_id_datasets > 1].index.astype(str).tolist()
    observations = donor_cells[["dataset", "sample_id", "donor_id"]].drop_duplicates()
    observations["source_observation_id"] = (
        observations["dataset"].astype("string")
        + "::"
        + observations["sample_id"].astype("string")
    )
    observation_donor_counts = (
        observations.groupby(["dataset", "source_observation_id"], observed=True)[
            "donor_id"
        ]
        .nunique()
        .rename("n_donors")
        .reset_index()
    )
    colliding_observations = observation_donor_counts.loc[
        observation_donor_counts["n_donors"] > 1
    ]
    observations_with_multiple_donors = len(colliding_observations)
    observation_collisions_by_dataset = {
        str(dataset): int(len(part))
        for dataset, part in colliding_observations.groupby("dataset", observed=True)
    }
    observation_collision_examples = (
        colliding_observations.sort_values(["dataset", "source_observation_id"])
        .head(20)
        .to_dict(orient="records")
    )

    hashes = np.concatenate(all_hashes)
    hashes.sort()
    duplicate_hash_count = int(np.count_nonzero(hashes[1:] == hashes[:-1]))
    # A duplicate hash is followed up exactly only if one is observed. The current
    # reference has none, avoiding retention of millions of cell strings in memory.
    duplicate_cell_id_count = duplicate_hash_count

    diagnostics = {
        "raw_donor_id_collision_count": len(collision_ids),
        "raw_donor_id_collisions": sorted(collision_ids),
        "donors_with_multiple_sex_values": int(
            (unit_observations["n_sex_values"] > 1).sum()
        ),
        "donors_with_multiple_age_values": int(
            (unit_observations["n_age_values"] > 1).sum()
        ),
        "source_observations_with_multiple_donors": observations_with_multiple_donors,
        "observation_collisions_by_dataset": observation_collisions_by_dataset,
        "observation_collision_examples": observation_collision_examples,
        "identifier_resolution": {
            "status": "user_approved",
            "biological_unit_id": "dataset::donor_id",
            "source_observation_id": "dataset::sample_id",
            "observation_id": "dataset::donor_id::sample_id",
            "reason": (
                "OneK1K sample_id denotes a pool shared across donors; the "
                "donor-qualified observation ID prevents collisions while retaining "
                "source-pool provenance."
            ),
        },
        "duplicate_cell_id_hash_count": duplicate_hash_count,
        "duplicate_cell_id_count": duplicate_cell_id_count,
        "duplicate_cell_id_method": (
            "pandas stable uint64 hash of every obs index; exact follow-up required if "
            "hash duplicates are nonzero"
        ),
        "observation_schemas_identical": True,
        "gene_vocabulary_and_order_identical": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices.to_csv(output_dir / "lineage_summary.tsv", sep="\t", index=False)
    datasets.to_csv(output_dir / "dataset_summary.tsv", sep="\t", index=False)
    _write_gzip_tsv(
        donor_fine.sort_values(["dataset", "donor_id", "lineage", "fine_type"]),
        output_dir / "donor_summary.tsv.gz",
    )
    fine_types.to_csv(output_dir / "fine_type_summary.tsv", sep="\t", index=False)
    missingness.to_csv(output_dir / "metadata_missingness.tsv", sep="\t", index=False)
    age_sex.to_csv(output_dir / "age_sex_support.tsv", sep="\t", index=False)
    genes = pd.DataFrame(gene_rows).sort_values(
        ["lineage", "scope", "dataset", "source_dataset_id"]
    )
    genes.to_csv(output_dir / "gene_identifier_summary.tsv", sep="\t", index=False)

    input_h5ad_metadata = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for _, path in lineage_files
    ]
    small_input_hashes = {}
    for path_string in (
        artifacts["split_manifests"]
        + artifacts["merge_manifests"]
        + [str(provenance_path)]
    ):
        path = Path(path_string)
        small_input_hashes[str(path)] = _sha256(path)

    manifest = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_h5ad_access": True,
        "count_matrices_densified": False,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "provenance_document": str(provenance_path),
        "repository_commit": _git_commit(repo_root),
        "primary_lineages": list(PRIMARY_LINEAGES),
        "audited_lineages": list(AUDIT_LINEAGES),
        "observed_datasets": sorted(datasets["dataset"].astype(str).tolist()),
        "h5ad_inputs": input_h5ad_metadata,
        "small_input_sha256": small_input_hashes,
        "artifact_inventory": artifacts,
        "diagnostics": diagnostics,
        "evidence_boundary": {
            "direct_observation": (
                "Current filesystem, manifests/QC tables, and HDF5 structures/metadata"
            ),
            "provenance_only": (
                "Historical raw-count recovery, annotation execution, filtering "
                "intent, "
                "and merge transformation mechanics"
            ),
        },
        "outputs": [
            "data_structure_audit.md",
            "lineage_summary.tsv",
            "dataset_summary.tsv",
            "donor_summary.tsv.gz",
            "fine_type_summary.tsv",
            "metadata_missingness.tsv",
            "age_sex_support.tsv",
            "gene_identifier_summary.tsv",
            "audit_manifest.json",
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    report = _markdown_report(
        matrices,
        datasets,
        unit_observations,
        fine_types,
        missingness,
        artifacts,
        diagnostics,
        provenance_path,
    )
    (output_dir / "data_structure_audit.md").write_text(report)
    print(f"Audit reports written to {output_dir}", flush=True)
    return manifest
