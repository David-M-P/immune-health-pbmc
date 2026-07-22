"""Memory-bounded, fold-bound tokenization for the inspected TRIPSO vendor API.

The vendor convenience function mixes HVG selection, optional cell subsampling,
temporary file deletion, and tokenization.  Reference preparation has already made
those scientific choices, so this bridge invokes the inspected
``TranscriptomeTokenizer`` directly and records the resulting physical donor scope.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.data.h5ad import read_csr_rows
from immune_health.data.lineage_scope import validate_lineage_donor_scope

from .contracts import (
    TripsoContractError,
    atomic_write_json,
    canonical_json_hash,
    load_fold_input_manifest,
    prepare_fold_input_manifest,
    read_table,
    sha256_path,
    validate_fold_rows,
)
from .provenance import validate_checkpoint_manifest

TOKENIZATION_SCHEMA = "immune-health-tripso-tokenization/v1"
TOKENIZED_DATASET_INTEGRITY_SCHEMA = "immune-health-tokenized-dataset-file-inventory/v1"
TOKENIZATION_RELOCATION_SCHEMA = "immune-health-tokenization-relocation/v1"
PROJECTION_INPUT_SCHEMA = "immune-health-tripso-projection-input/v1"
# Kept as a public alias for callers that imported the old query-specific name.
QUERY_INPUT_SCHEMA = PROJECTION_INPUT_SCHEMA
PROJECTION_ROLES = frozenset({"query", "reference", "validation"})
TOKENIZATION_INPUT_SIZE = 4096
SPECIAL_TOKEN_COUNT = 2
MAX_RANKED_GENES = TOKENIZATION_INPUT_SIZE - SPECIAL_TOKEN_COUNT
DEFAULT_MAX_PROJECTED_BYTES = 250 * 1024**3

REQUIRED_METADATA_COLUMNS = (
    "cell_key",
    "dataset",
    "biological_unit_id",
    "observation_id",
    "fine_type",
    "fine_type_state_eligible",
    "fine_type_balance_eligible",
    "lineage",
)
OPTIONAL_METADATA_COLUMNS = (
    "donor_id",
    "sample_id",
    "source_observation_id",
    "source_cell_id",
    "age",
    "sex",
    "fine_type_confidence",
    "ctype_low",
    "ctype_low_conf",
    "fine_type_mapping_status",
    "annotation_confidence",
    "preparation_role",
    "outer_role",
    "fold_id",
)


def _ordered_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _directory_file_inventory(path: Path) -> dict[str, Any]:
    """Hash every regular file below a portable directory artifact.

    Hugging Face ``save_to_disk`` outputs are directories, so hashing only the
    JSON manifest cannot detect a partially copied or modified Arrow shard.  The
    inventory deliberately uses relative POSIX paths and content hashes: it is
    stable when an artifact is copied to another cluster while remaining strict
    about every byte that is consumed there.
    """

    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot inventory missing directory: {path}")
    records: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if candidate.is_symlink():
            raise TripsoContractError(
                f"Portable tokenized datasets may not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            records.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_path(candidate),
                }
            )
    if not records:
        raise TripsoContractError(f"Tokenized dataset directory is empty: {path}")
    return {
        "schema_version": TOKENIZED_DATASET_INTEGRITY_SCHEMA,
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
        "tree_sha256": canonical_json_hash(records),
    }


def _validate_directory_file_inventory(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if expected.get("schema_version") != TOKENIZED_DATASET_INTEGRITY_SCHEMA:
        raise TripsoContractError(
            "Tokenization manifest lacks a supported physical Arrow file inventory"
        )
    observed = _directory_file_inventory(path)
    if observed != dict(expected):
        raise TripsoContractError(
            "Physical tokenized dataset file inventory differs from its manifest; "
            "the SFTP copy may be incomplete or an Arrow file was modified"
        )
    return observed


def _read_gene_vocabulary(path: Path) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen gene vocabulary does not exist: {path}")
    genes = tuple(
        line.strip() for line in path.read_text().splitlines() if line.strip()
    )
    if not genes:
        raise TripsoContractError(f"Frozen gene vocabulary is empty: {path}")
    duplicates = [gene for gene, count in Counter(genes).items() if count > 1]
    if duplicates:
        raise TripsoContractError(
            f"Frozen gene vocabulary contains duplicates: {duplicates[:10]}"
        )
    non_ensembl = [
        gene for gene in genes if not re.fullmatch(r"ENSG\d+(?:\.\d+)?", gene)
    ]
    if non_ensembl:
        raise TripsoContractError(
            "TRIPSO production tokenization requires Ensembl gene IDs; examples: "
            f"{non_ensembl[:10]}"
        )
    return genes


def _projection_gp_selection(
    gp_library_path: Path,
    *,
    gp_allowlist_path: Path | None,
    allow_all_gps: bool,
    include_cell_token: bool,
    include_gene_encoder_cls: bool,
) -> dict[str, Any]:
    """Resolve an explicit, immutable set of GP columns to persist."""

    if (gp_allowlist_path is None) == (not allow_all_gps):
        raise TripsoContractError(
            "Choose exactly one of gp_allowlist_path or allow_all_gps=true"
        )
    gp_library_path = Path(gp_library_path)
    header = pd.read_csv(gp_library_path, nrows=0)
    available = tuple(map(str, header.columns))
    if not available:
        raise TripsoContractError("GP library has no program columns")
    if allow_all_gps:
        selected = available
        selection = {
            "mode": "all_gps_bounded_diagnostic",
            "allowlist_path": None,
            "allowlist_sha256": None,
        }
    else:
        assert gp_allowlist_path is not None
        gp_allowlist_path = Path(gp_allowlist_path).resolve()
        if not gp_allowlist_path.is_file():
            raise FileNotFoundError(
                f"GP projection candidate manifest is missing: {gp_allowlist_path}"
            )
        if gp_allowlist_path.suffix.lower() != ".json":
            raise TripsoContractError(
                "Production GP projection candidates must be the training-only JSON "
                "manifest, not an unbound text list"
            )
        with gp_allowlist_path.open(encoding="utf-8") as handle:
            candidate_manifest = json.load(handle)
        if candidate_manifest.get("schema_version") != (
            "immune-health-projection-gp-candidates/v1"
        ):
            raise TripsoContractError("Unsupported GP projection candidate manifest")
        if candidate_manifest.get("query_data_consulted") is not False:
            raise TripsoContractError(
                "GP projection candidates do not prove query-data exclusion"
            )
        if candidate_manifest.get("selection_level") != "donor_lineage_pseudobulk":
            raise TripsoContractError(
                "GP projection candidates use an unapproved selection level"
            )
        candidate_payload = dict(candidate_manifest)
        claimed_content_hash = candidate_payload.pop("manifest_content_sha256", None)
        if claimed_content_hash != canonical_json_hash(candidate_payload):
            raise TripsoContractError(
                "GP projection candidate manifest content hash is invalid"
            )
        selected_values = list(map(str, candidate_manifest.get("program_ids", [])))
        if not selected_values:
            raise TripsoContractError(
                "GP projection candidate manifest contains no programs"
            )
        duplicates = [
            value for value, count in Counter(selected_values).items() if count > 1
        ]
        if duplicates:
            raise TripsoContractError(
                f"GP projection allowlist contains duplicates: {duplicates[:10]}"
            )
        missing = [value for value in selected_values if value not in set(available)]
        if missing:
            raise TripsoContractError(
                "GP projection allowlist requests programs absent from the trained "
                f"fold library: {missing[:10]}"
            )
        expected_order = [
            program for program in available if program in set(selected_values)
        ]
        if selected_values != expected_order:
            raise TripsoContractError(
                "GP projection candidates are not in filtered GP-library order"
            )
        if candidate_manifest.get("program_ids_ordered_sha256") != canonical_json_hash(
            selected_values
        ):
            raise TripsoContractError(
                "GP projection candidate ordered-program hash is invalid"
            )
        candidate_gpdb_hash = candidate_manifest.get("binding", {}).get("gpdb_sha256")
        if candidate_gpdb_hash != sha256_path(gp_library_path):
            raise TripsoContractError(
                "GP projection candidates are bound to a different GP library"
            )
        selected = tuple(selected_values)
        selection = {
            "mode": "frozen_training_candidates",
            "allowlist_path": str(gp_allowlist_path),
            "allowlist_sha256": sha256_path(gp_allowlist_path),
            "candidate_manifest_content_sha256": claimed_content_hash,
            "candidate_manifest_schema": candidate_manifest["schema_version"],
        }
    return {
        **selection,
        "program_ids": list(selected),
        "program_ids_ordered_sha256": canonical_json_hash(list(selected)),
        "n_programs": len(selected),
        "available_program_count": len(available),
        "include_gp_gene_fraction": False,
        "include_cell_token": bool(include_cell_token),
        "include_gene_encoder_cls": bool(include_gene_encoder_cls),
    }


def _load_pickle_mapping(path: Path, name: str) -> Mapping[Any, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TRIPSO {name} does not exist: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TripsoContractError(f"TRIPSO {name} is not a mapping: {path}")
    return value


@contextmanager
def _vendor_import_path(vendor_root: Path) -> Iterator[None]:
    root = str(Path(vendor_root).resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted and root in sys.path:
            sys.path.remove(root)


def _load_vendor_surface(vendor_root: Path) -> tuple[type, Path, Path, Path]:
    vendor_root = Path(vendor_root).resolve()
    with _vendor_import_path(vendor_root):
        try:
            import tripso  # type: ignore
            from tripso.Preprocessing.geneformer_tokenizer import (  # type: ignore
                TranscriptomeTokenizer,
            )
        except Exception as exc:
            raise RuntimeError(
                "The inspected TRIPSO tokenizer cannot be imported. Validate the "
                "pinned environment before tokenization."
            ) from exc
    implementation = Path(inspect.getfile(TranscriptomeTokenizer)).resolve()
    try:
        implementation.relative_to(vendor_root)
    except ValueError as exc:
        raise RuntimeError(
            "Imported TRIPSO tokenizer does not come from the requested vendor root: "
            f"{implementation}"
        ) from exc
    return (
        TranscriptomeTokenizer,
        Path(tripso.TOKEN_DICTIONARY_FILE).resolve(),
        Path(tripso.GENE_MEDIAN_FILE).resolve(),
        implementation,
    )


def _materialization_manifest_path(input_h5ad: Path) -> Path:
    return Path(input_h5ad).with_suffix(".manifest.json")


def _validate_h5ad_contract(
    input_h5ad: Path,
    genes: Sequence[str],
    *,
    metadata_columns: Sequence[str],
    role: str,
) -> dict[str, Any]:
    input_h5ad = Path(input_h5ad).resolve()
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Materialized H5AD does not exist: {input_h5ad}")
    source = ad.read_h5ad(input_h5ad, backed="r")
    try:
        observed_genes = tuple(map(str, source.var_names))
        if observed_genes != tuple(genes):
            first_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(observed_genes, genes, strict=False)
                    )
                    if left != right
                ),
                min(len(observed_genes), len(genes)),
            )
            raise TripsoContractError(
                "Materialized H5AD gene order is not exactly the frozen vocabulary; "
                f"H5AD has {len(observed_genes)}, vocabulary has {len(genes)}, first "
                f"difference is position {first_difference}. Tokenization is refused."
            )
        if "ensembl_id" not in source.var:
            raise TripsoContractError(
                "Materialized H5AD lacks var['ensembl_id']; rerun the current "
                "materialization stage rather than guessing identifiers at "
                "tokenization."
            )
        ensembl = tuple(source.var["ensembl_id"].astype(str))
        if ensembl != tuple(genes):
            raise TripsoContractError(
                "var['ensembl_id'] must exactly equal the ordered frozen vocabulary"
            )
        missing = sorted(set(metadata_columns) - set(source.obs.columns))
        if missing:
            raise TripsoContractError(
                f"Materialized H5AD lacks required token metadata: {missing}"
            )
        for column in metadata_columns:
            values = source.obs[column]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise TripsoContractError(
                    f"Token metadata column {column!r} contains missing/empty values"
                )
        for column in (
            "fine_type_state_eligible",
            "fine_type_balance_eligible",
        ):
            normalized = source.obs[column].astype("string").str.strip().str.lower()
            mapped = normalized.map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            if mapped.isna().any():
                raise TripsoContractError(
                    f"Fine-type eligibility column {column!r} is not boolean"
                )
        state_eligible = (
            source.obs["fine_type_state_eligible"]
            .astype("string")
            .str.lower()
            .isin(("true", "1"))
        )
        balance_eligible = (
            source.obs["fine_type_balance_eligible"]
            .astype("string")
            .str.lower()
            .isin(("true", "1"))
        )
        if (state_eligible & ~balance_eligible).any():
            raise TripsoContractError(
                "State-eligible fine types must also be balance eligible"
            )
        special = (
            source.obs["fine_type"]
            .astype(str)
            .isin(("low_confidence", "other_confident"))
        )
        if (special & (state_eligible | balance_eligible)).any():
            raise TripsoContractError(
                "Special fine-type categories cannot be state/balance eligible"
            )
        if source.obs["cell_key"].astype(str).duplicated().any():
            raise TripsoContractError(
                "Materialized H5AD cell_key values are not unique"
            )
        if "n_counts" not in source.obs:
            raise TripsoContractError(
                "Materialized H5AD lacks full-universe n_counts. Rerun "
                "materialization; "
                "summing the selected 3k/9k genes would give the wrong denominator."
            )
        n_counts = pd.to_numeric(source.obs["n_counts"], errors="coerce").to_numpy()
        if not np.isfinite(n_counts).all() or (n_counts <= 0).any():
            raise TripsoContractError("n_counts must be finite and positive")
        if "filter_pass" in source.obs:
            retained = pd.to_numeric(source.obs["filter_pass"], errors="coerce")
            if retained.isna().any() or not retained.eq(1).all():
                raise TripsoContractError(
                    "filter_pass would remove cells, but this stage requires every "
                    "materialized cell to be tokenized"
                )
        if "preparation_role" in source.obs:
            observed_roles = set(source.obs["preparation_role"].astype(str))
            if observed_roles != {role}:
                raise TripsoContractError(
                    f"H5AD preparation roles {sorted(observed_roles)} do not equal "
                    f"requested role {role!r}"
                )
        donor_dataset_counts = source.obs.groupby("biological_unit_id", observed=True)[
            "dataset"
        ].nunique()
        if (donor_dataset_counts != 1).any():
            raise TripsoContractError(
                "A biological_unit_id maps to multiple datasets in the H5AD"
            )
        lineage_values = sorted(set(source.obs["lineage"].astype(str)))
        biological_unit_ids = sorted(set(source.obs["biological_unit_id"].astype(str)))
        return {
            "shape": [int(source.n_obs), int(source.n_vars)],
            "n_biological_units": len(biological_unit_ids),
            "biological_unit_ids": biological_unit_ids,
            "datasets": sorted(set(source.obs["dataset"].astype(str))),
            "lineages": lineage_values,
            "cell_key_ordered_sha256": _ordered_digest(source.obs["cell_key"]),
            "n_counts_source_required": (
                "full_source_h5ad_gene_universe_before_feature_subset"
            ),
        }
    finally:
        source.file.close()


def _gp_token_coverage(
    gp_library_path: Path,
    genes: Sequence[str],
    token_dictionary: Mapping[Any, Any],
) -> pd.DataFrame:
    gp_library_path = Path(gp_library_path)
    if not gp_library_path.is_file():
        raise FileNotFoundError(
            f"Filtered GP library does not exist: {gp_library_path}"
        )
    gpdb = pd.read_csv(gp_library_path)
    if not len(gpdb.columns):
        raise TripsoContractError("Filtered GP library has no program columns")
    vocabulary = set(genes)
    tokenizable = set(map(str, token_dictionary))
    rows: list[dict[str, Any]] = []
    for program in gpdb.columns:
        program_genes = tuple(
            dict.fromkeys(str(value).strip() for value in gpdb[program].dropna())
        )
        if not program_genes:
            raise TripsoContractError(f"Gene program {program!r} is empty")
        in_vocabulary = [gene for gene in program_genes if gene in vocabulary]
        in_tokens = [gene for gene in in_vocabulary if gene in tokenizable]
        rows.append(
            {
                "program_id": str(program),
                "n_library_genes": len(program_genes),
                "n_in_frozen_vocabulary": len(in_vocabulary),
                "n_tokenizable_genes": len(in_tokens),
                "frozen_vocabulary_coverage": len(in_vocabulary) / len(program_genes),
                "tokenizable_coverage": len(in_tokens) / len(program_genes),
                "missing_from_frozen_vocabulary": "|".join(
                    gene for gene in program_genes if gene not in vocabulary
                ),
                "missing_from_token_dictionary": "|".join(
                    gene for gene in in_vocabulary if gene not in tokenizable
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_raw_count_chunk(matrix: Any, start: int, stop: int) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).reshape(-1)
    if not np.isfinite(values).all() or (values < 0).any():
        raise TripsoContractError(
            f"H5AD rows {start}:{stop} contain non-finite or negative counts"
        )
    if values.size and not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise TripsoContractError(
            f"H5AD rows {start}:{stop} are not integer-like raw counts"
        )


def _length_histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(map(int, values))
    return {str(length): int(counts[length]) for length in sorted(counts)}


def _merge_histograms(histograms: Iterable[Mapping[str, Any]]) -> dict[int, int]:
    total: Counter[int] = Counter()
    for histogram in histograms:
        total.update({int(key): int(value) for key, value in histogram.items()})
    return dict(sorted(total.items()))


def _histogram_quantile(histogram: Mapping[int, int], quantile: float) -> float:
    total = sum(histogram.values())
    if total < 1:
        return float("nan")
    target = quantile * (total - 1)
    cumulative = 0
    for value, count in sorted(histogram.items()):
        if cumulative + count > target:
            return float(value)
        cumulative += count
    return float(max(histogram))


def _sequence_qc(histogram: Mapping[int, int]) -> dict[str, Any]:
    n_cells = int(sum(histogram.values()))
    n_truncated = int(
        sum(count for length, count in histogram.items() if length > MAX_RANKED_GENES)
    )
    # Multiple uncropped lengths can map to the same cropped length.
    cropped_histogram: Counter[int] = Counter()
    for length, count in histogram.items():
        cropped_histogram[min(length, MAX_RANKED_GENES) + SPECIAL_TOKEN_COUNT] += count
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        "n_cells": n_cells,
        "model_input_size": TOKENIZATION_INPUT_SIZE,
        "special_tokens_per_cell": SPECIAL_TOKEN_COUNT,
        "maximum_ranked_genes_per_cell": MAX_RANKED_GENES,
        "n_cells_truncated": n_truncated,
        "fraction_cells_truncated": n_truncated / n_cells if n_cells else float("nan"),
        "uncropped_ranked_gene_length_quantiles": {
            str(q): _histogram_quantile(histogram, q) for q in quantiles
        },
        "stored_sequence_length_quantiles": {
            str(q): _histogram_quantile(cropped_histogram, q) for q in quantiles
        },
        "uncropped_length_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
    }


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".parquet"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".tmp{suffix}", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif suffix == ".tsv":
            frame.to_csv(temporary, sep="\t", index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_remove_generated(path: Path, output_dir: Path) -> None:
    path = Path(path).resolve()
    output_dir = Path(output_dir).resolve()
    if path.parent != output_dir:
        raise RuntimeError(f"Refusing to remove non-child generated path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _physical_scope(dataset: Any) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    counts: Counter[tuple[str, str]] = Counter()
    digest = hashlib.sha256()
    donors: set[str] = set()
    selected = dataset.select_columns(["cell_key", "dataset", "biological_unit_id"])
    for batch in selected.iter(batch_size=100_000):
        keys = batch["cell_key"]
        datasets = batch["dataset"]
        biological_units = batch["biological_unit_id"]
        if not (len(keys) == len(datasets) == len(biological_units)):
            raise RuntimeError("Tokenized metadata columns have different lengths")
        for key, dataset_name, donor in zip(
            keys, datasets, biological_units, strict=True
        ):
            encoded = str(key).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            donor_text = str(donor)
            dataset_text = str(dataset_name)
            counts[(dataset_text, donor_text)] += 1
            donors.add(donor_text)
    scope = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "biological_unit_id": donor,
                "n_tokenized_cells": count,
            }
            for (dataset_name, donor), count in sorted(counts.items())
        ]
    )
    return scope, digest.hexdigest(), tuple(sorted(donors))


def tokenize_fold_h5ad(
    *,
    input_h5ad: Path,
    gene_vocabulary_path: Path,
    gp_library_path: Path,
    projection_gp_candidates_path: Path,
    output_dir: Path,
    vendor_root: Path,
    role: str,
    row_chunk_size: int = 20_000,
    nproc: int = 4,
    minimum_tokenizable_gp_genes: int = 10,
    keep_chunks: bool = False,
    overwrite: bool = False,
    _vendor_surface: tuple[type, Path, Path, Path] | None = None,
) -> dict[str, Any]:
    """Tokenize every cell in one exact fold/role H5AD and write audit manifests."""

    if role not in {"adaptation", "validation", "query"}:
        raise ValueError("role must be adaptation, validation, or query")
    if row_chunk_size < 1 or nproc < 1 or minimum_tokenizable_gp_genes < 1:
        raise ValueError("Chunk size, nproc, and minimum GP size must be positive")
    input_h5ad = Path(input_h5ad).resolve()
    gene_vocabulary_path = Path(gene_vocabulary_path).resolve()
    gp_library_path = Path(gp_library_path).resolve()
    projection_gp_candidates_path = Path(projection_gp_candidates_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dataset_path = output_dir / "tokenized.dataset"
    manifest_path = output_dir / "tokenization_manifest.json"
    chunks_dir = output_dir / "tokenized_chunks"
    scope_path = output_dir / "donor_scope.parquet"
    gp_coverage_path = output_dir / "gp_token_coverage.parquet"
    sequence_qc_path = output_dir / "sequence_qc.json"

    if manifest_path.exists() or final_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Tokenization output is already complete or materialized: {output_dir}"
            )
        for path in (
            manifest_path,
            final_dataset_path,
            chunks_dir,
            scope_path,
            gp_coverage_path,
            sequence_qc_path,
        ):
            if path.exists():
                _safe_remove_generated(path, output_dir)

    genes = _read_gene_vocabulary(gene_vocabulary_path)
    available_metadata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        metadata_columns = tuple(
            dict.fromkeys(
                (*REQUIRED_METADATA_COLUMNS, "idx", *OPTIONAL_METADATA_COLUMNS)
            )
        )
        # ``idx`` is generated from the approved cell key below, not required in
        # the materialized H5AD.
        input_metadata_columns = tuple(
            column
            for column in metadata_columns
            if column != "idx" and column in available_metadata.obs.columns
        )
    finally:
        available_metadata.file.close()
    missing_required = sorted(
        set(REQUIRED_METADATA_COLUMNS) - set(input_metadata_columns)
    )
    if missing_required:
        raise TripsoContractError(
            f"Materialized H5AD lacks required metadata: {missing_required}"
        )
    input_audit = _validate_h5ad_contract(
        input_h5ad,
        genes,
        metadata_columns=REQUIRED_METADATA_COLUMNS,
        role=role,
    )

    vendor_surface = _vendor_surface or _load_vendor_surface(vendor_root)
    tokenizer_class, token_dictionary_path, median_path, implementation_path = (
        vendor_surface
    )
    token_dictionary = _load_pickle_mapping(
        token_dictionary_path, "Geneformer token dictionary"
    )
    _load_pickle_mapping(median_path, "Geneformer median dictionary")
    tokenizable_genes = {gene for gene in genes if gene in token_dictionary}
    gp_coverage = _gp_token_coverage(gp_library_path, genes, token_dictionary)
    too_small = gp_coverage.loc[
        gp_coverage["n_tokenizable_genes"] < minimum_tokenizable_gp_genes,
        ["program_id", "n_tokenizable_genes"],
    ]
    if len(too_small):
        raise TripsoContractError(
            "Filtered gene programs fall below the tokenizable-gene minimum; "
            f"examples: {too_small.head(10).to_dict(orient='records')}"
        )
    candidate_selection = _projection_gp_selection(
        gp_library_path,
        gp_allowlist_path=projection_gp_candidates_path,
        allow_all_gps=False,
        include_cell_token=False,
        include_gene_encoder_cls=False,
    )
    projection_gp_candidates = {
        "path": candidate_selection["allowlist_path"],
        "sha256": candidate_selection["allowlist_sha256"],
        "manifest_content_sha256": candidate_selection[
            "candidate_manifest_content_sha256"
        ],
        "program_ids_ordered_sha256": candidate_selection["program_ids_ordered_sha256"],
        "program_ids": candidate_selection["program_ids"],
    }

    materialization_manifest = _materialization_manifest_path(input_h5ad)
    materialization_hash = (
        sha256_path(materialization_manifest)
        if materialization_manifest.is_file()
        else None
    )
    fine_type_ontology: dict[str, Any] | None = None
    lineage_donor_scope: dict[str, Any] | None = None
    if materialization_manifest.is_file():
        with materialization_manifest.open(encoding="utf-8") as handle:
            materialization_payload = json.load(handle)
        ontology_value = materialization_payload.get("fine_type_ontology")
        if not isinstance(ontology_value, Mapping):
            raise TripsoContractError(
                "Materialization manifest lacks approved fine-type provenance"
            )
        if (
            ontology_value.get("approval_status") != "approved"
            or not str(ontology_value.get("sha256", "")).strip()
        ):
            raise TripsoContractError(
                "Materialization manifest binds an unapproved fine-type ontology"
            )
        fine_type_ontology = dict(ontology_value)
        scope_value = materialization_payload.get("lineage_donor_scope")
        reference_design = str(materialization_payload.get("reference_design", ""))
        if reference_design in {"lodo", "all_healthy"}:
            if not isinstance(scope_value, Mapping):
                raise TripsoContractError(
                    "Reference materialization lacks its lineage-specific donor "
                    "scope; rerun feature preparation and materialization"
                )
            try:
                lineage_donor_scope = validate_lineage_donor_scope(
                    scope_value,
                    lineage=(
                        input_audit["lineages"][0]
                        if len(input_audit["lineages"]) == 1
                        else None
                    ),
                )
            except ValueError as exc:
                raise TripsoContractError(str(exc)) from exc
            expected_role_donors = lineage_donor_scope[
                "biological_unit_ids_by_preparation_role"
            ][role]
            if input_audit["biological_unit_ids"] != expected_role_donors:
                raise TripsoContractError(
                    "Materialized H5AD donors differ from the lineage-specific "
                    f"scope for role {role!r}"
                )
        elif scope_value is not None:
            try:
                lineage_donor_scope = validate_lineage_donor_scope(scope_value)
            except ValueError as exc:
                raise TripsoContractError(str(exc)) from exc
    input_h5ad_hash = sha256_path(input_h5ad)
    vendor_implementation_hash = sha256_path(implementation_path)
    try:
        vendor_implementation = str(
            implementation_path.relative_to(Path(vendor_root).resolve())
        )
    except ValueError:
        # Dependency-injected test surfaces need not live below a vendor checkout.
        vendor_implementation = implementation_path.name
    tokenizer_contract = {
        # A relative implementation identity keeps the immutable scientific
        # contract portable when the checkout moves between clusters.
        "vendor_implementation": vendor_implementation,
        "vendor_implementation_sha256": vendor_implementation_hash,
        "token_dictionary_sha256": sha256_path(token_dictionary_path),
        "median_dictionary_sha256": sha256_path(median_path),
        "model_input_size": TOKENIZATION_INPUT_SIZE,
        "special_tokens": True,
        "gene_mapping": "identity_only_gene_mapping_file_none",
        "rank_normalization": "Geneformer median-scaled counts-per-10000",
        "library_size_column": "n_counts",
        "library_size_source": ("full_source_h5ad_gene_universe_before_feature_subset"),
        "calculate_hvg": False,
        "cell_subsampling": False,
        "empty_cell_policy": "refuse_if_any_cell_is_dropped",
        "metadata_columns": ["idx", *input_metadata_columns],
    }
    run_fingerprint = canonical_json_hash(
        {
            "input_h5ad": str(input_h5ad),
            "input_h5ad_sha256": input_h5ad_hash,
            "materialization_manifest_sha256": materialization_hash,
            "gene_vocabulary_sha256": sha256_path(gene_vocabulary_path),
            "gp_library_sha256": sha256_path(gp_library_path),
            "projection_gp_candidates": projection_gp_candidates,
            "role": role,
            "row_chunk_size": row_chunk_size,
            "tokenizer_contract": tokenizer_contract,
        }
    )

    chunks_dir.mkdir(parents=True, exist_ok=True)
    custom_attributes = {column: column for column in input_metadata_columns}
    custom_attributes["idx"] = "idx"
    tokenizer = tokenizer_class(
        custom_attr_name_dict=custom_attributes,
        nproc=nproc,
        model_input_size=TOKENIZATION_INPUT_SIZE,
        special_token=True,
        collapse_gene_ids=True,
        keep_counts=False,
        gene_mapping_file=None,
        token_dictionary_file=token_dictionary_path,
        gene_median_file=median_path,
    )

    try:
        from datasets import concatenate_datasets, load_from_disk
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for tokenization"
        ) from exc

    source = ad.read_h5ad(input_h5ad, backed="r")
    chunk_datasets: list[Any] = []
    chunk_manifests: list[dict[str, Any]] = []
    try:
        for chunk_number, start in enumerate(range(0, source.n_obs, row_chunk_size)):
            stop = min(start + row_chunk_size, source.n_obs)
            chunk_path = chunks_dir / f"chunk_{chunk_number:06d}.dataset"
            chunk_manifest_path = chunks_dir / f"chunk_{chunk_number:06d}.json"
            expected_keys = tuple(source.obs["cell_key"].iloc[start:stop].astype(str))
            expected_key_hash = _ordered_digest(expected_keys)
            if chunk_path.is_dir() and chunk_manifest_path.is_file():
                chunk_manifest = json.loads(chunk_manifest_path.read_text())
                if (
                    chunk_manifest.get("run_fingerprint") != run_fingerprint
                    or chunk_manifest.get("row_interval") != [start, stop]
                    or chunk_manifest.get("cell_key_ordered_sha256")
                    != expected_key_hash
                ):
                    raise TripsoContractError(
                        f"Existing tokenization chunk is from another run: {chunk_path}"
                    )
                dataset = load_from_disk(str(chunk_path))
                if len(dataset) != stop - start:
                    raise TripsoContractError(
                        f"Existing tokenization chunk is incomplete: {chunk_path}"
                    )
            else:
                if chunk_path.exists():
                    shutil.rmtree(chunk_path)
                chunk_manifest_path.unlink(missing_ok=True)
                row_positions = np.arange(start, stop, dtype=np.int64)
                part = ad.AnnData(
                    X=read_csr_rows(input_h5ad, row_positions),
                    obs=source.obs.iloc[start:stop].copy(),
                    var=source.var.copy(),
                )
                _validate_raw_count_chunk(part.X, start, stop)
                part.obs["idx"] = part.obs["cell_key"].astype(str).to_numpy()
                for column in custom_attributes:
                    if column != "idx":
                        part.obs[column] = part.obs[column].astype(str).to_numpy()
                with tempfile.TemporaryDirectory(
                    prefix=f"tripso_tokenize_{chunk_number:06d}_", dir=output_dir
                ) as temporary_name:
                    temporary_dir = Path(temporary_name)
                    shard_path = temporary_dir / "cells.h5ad"
                    part.write_h5ad(shard_path)
                    tokenized_cells, cell_metadata, tokenized_counts = (
                        tokenizer.tokenize_files(temporary_dir, file_format="h5ad")
                    )
                if len(tokenized_cells) != stop - start:
                    raise TripsoContractError(
                        "The vendor tokenizer dropped cells (usually zero overlap with "
                        f"its dictionary) in rows {start}:{stop}; no silent cell loss "
                        "is permitted."
                    )
                raw_lengths = [len(sequence) for sequence in tokenized_cells]
                if not raw_lengths or min(raw_lengths) < 1:
                    raise TripsoContractError(
                        f"Rows {start}:{stop} contain an empty token sequence"
                    )
                dataset = tokenizer.create_dataset(
                    tokenized_cells,
                    cell_metadata,
                    tokenized_counts,
                    use_generator=False,
                    keep_uncropped_input_ids=False,
                )
                dataset = dataset.add_column("length_uncropped", raw_lengths)
                dataset = dataset.add_column(
                    "was_truncated",
                    [length > MAX_RANKED_GENES for length in raw_lengths],
                )
                observed_keys = tuple(map(str, dataset["cell_key"]))
                if observed_keys != expected_keys:
                    raise TripsoContractError(
                        f"Tokenized cell order changed in rows {start}:{stop}"
                    )
                dataset.save_to_disk(str(chunk_path))
                chunk_manifest = {
                    "schema_version": "immune-health-tripso-tokenization-chunk/v1",
                    "run_fingerprint": run_fingerprint,
                    "chunk_number": chunk_number,
                    "row_interval": [start, stop],
                    "n_cells": stop - start,
                    "cell_key_ordered_sha256": expected_key_hash,
                    "uncropped_length_histogram": _length_histogram(raw_lengths),
                    "huggingface_fingerprint": getattr(dataset, "_fingerprint", None),
                }
                atomic_write_json(chunk_manifest_path, chunk_manifest)
            chunk_datasets.append(dataset)
            chunk_manifests.append(chunk_manifest)
    finally:
        source.file.close()

    combined = (
        chunk_datasets[0]
        if len(chunk_datasets) == 1
        else concatenate_datasets(chunk_datasets)
    )
    if len(combined) != input_audit["shape"][0]:
        raise TripsoContractError(
            "Combined tokenized dataset does not contain every cell"
        )
    required_output_columns = {
        "input_ids",
        "length",
        "length_uncropped",
        "was_truncated",
        "idx",
        *REQUIRED_METADATA_COLUMNS,
    }
    missing_output = sorted(required_output_columns - set(combined.column_names))
    if missing_output:
        raise TripsoContractError(
            f"Tokenized dataset lacks required physical columns: {missing_output}"
        )

    partial_path = output_dir / f".tokenized.{os.getpid()}.partial.dataset"
    if partial_path.exists():
        shutil.rmtree(partial_path)
    combined.save_to_disk(str(partial_path))
    os.replace(partial_path, final_dataset_path)
    final_dataset = load_from_disk(str(final_dataset_path))
    tokenized_dataset_integrity = _directory_file_inventory(final_dataset_path)
    scope, physical_key_hash, physical_donors = _physical_scope(final_dataset)
    if physical_key_hash != input_audit["cell_key_ordered_sha256"]:
        raise TripsoContractError(
            "Physical tokenized cell keys differ from the materialized H5AD"
        )

    uncropped_histogram = _merge_histograms(
        chunk["uncropped_length_histogram"] for chunk in chunk_manifests
    )
    sequence_qc = _sequence_qc(uncropped_histogram)
    _write_table_atomic(scope, scope_path)
    _write_table_atomic(gp_coverage, gp_coverage_path)
    atomic_write_json(sequence_qc_path, sequence_qc)

    manifest: dict[str, Any] = {
        "schema_version": TOKENIZATION_SCHEMA,
        "role": role,
        "input_h5ad": str(input_h5ad),
        "input_materialization_manifest": (
            str(materialization_manifest)
            if materialization_manifest.is_file()
            else None
        ),
        "fine_type_ontology": fine_type_ontology,
        "lineage_donor_scope": lineage_donor_scope,
        "tokenized_dataset_path": str(final_dataset_path),
        "gene_vocabulary_path": str(gene_vocabulary_path),
        "gp_library_path": str(gp_library_path),
        "projection_gp_candidates": projection_gp_candidates,
        "vendor_root": str(Path(vendor_root).resolve()),
        "tokenizer_resources": {
            "token_dictionary_path": str(token_dictionary_path),
            "median_dictionary_path": str(median_path),
            "implementation_path": str(implementation_path),
        },
        "tokenized_dataset_integrity": tokenized_dataset_integrity,
        "shape": [len(final_dataset), len(genes)],
        "n_tokenizable_frozen_genes": len(tokenizable_genes),
        "n_frozen_genes_absent_from_token_dictionary": len(genes)
        - len(tokenizable_genes),
        "frozen_genes_absent_from_token_dictionary": [
            gene for gene in genes if gene not in token_dictionary
        ],
        "cell_downsampling_performed": False,
        "hvg_calculation_performed": False,
        "all_materialized_cells_tokenized": True,
        "metadata_columns": list(final_dataset.column_names),
        "datasets": sorted(scope["dataset"].astype(str).unique()),
        "lineages": input_audit["lineages"],
        "biological_unit_ids": list(physical_donors),
        "n_biological_units": len(physical_donors),
        "cell_key_ordered_sha256": physical_key_hash,
        "biological_unit_ids_sha256": canonical_json_hash(list(physical_donors)),
        "tokenizer_contract": tokenizer_contract,
        "tokenizer_contract_sha256": canonical_json_hash(tokenizer_contract),
        "run_fingerprint": run_fingerprint,
        "sequence_qc": sequence_qc,
        "gp_token_coverage_summary": {
            "minimum_required_tokenizable_genes": minimum_tokenizable_gp_genes,
            "minimum_observed_tokenizable_genes": int(
                gp_coverage["n_tokenizable_genes"].min()
            ),
            "median_tokenizable_coverage": float(
                gp_coverage["tokenizable_coverage"].median()
            ),
        },
        "files": {
            "donor_scope": str(scope_path),
            "gp_token_coverage": str(gp_coverage_path),
            "sequence_qc": str(sequence_qc_path),
        },
        "hashes": {
            "input_h5ad_sha256": input_h5ad_hash,
            "input_materialization_manifest_sha256": materialization_hash,
            "gene_vocabulary_sha256": sha256_path(gene_vocabulary_path),
            "gp_library_sha256": sha256_path(gp_library_path),
            "projection_gp_candidates_sha256": projection_gp_candidates["sha256"],
            "token_dictionary_sha256": sha256_path(token_dictionary_path),
            "median_dictionary_sha256": sha256_path(median_path),
            "vendor_implementation_sha256": vendor_implementation_hash,
            "donor_scope_sha256": sha256_path(scope_path),
            "gp_token_coverage_sha256": sha256_path(gp_coverage_path),
            "sequence_qc_sha256": sha256_path(sequence_qc_path),
        },
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(manifest_path, manifest)
    if not keep_chunks and chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    return manifest


def load_tokenization_manifest(
    path: Path, *, verify_paths: bool = True
) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != TOKENIZATION_SCHEMA:
        raise TripsoContractError(f"Unsupported tokenization manifest: {path}")
    expected_hash = manifest.get("manifest_sha256")
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    if expected_hash != canonical_json_hash(content):
        raise TripsoContractError(f"Tokenization manifest hash mismatch: {path}")
    if manifest.get("cell_downsampling_performed") is not False:
        raise TripsoContractError("Tokenization manifest permits cell subsampling")
    if manifest.get("hvg_calculation_performed") is not False:
        raise TripsoContractError("Tokenization manifest recalculates HVGs")
    if manifest.get("all_materialized_cells_tokenized") is not True:
        raise TripsoContractError("Tokenization manifest does not preserve all cells")
    candidate = manifest.get("projection_gp_candidates")
    required_candidate_fields = {
        "path",
        "sha256",
        "manifest_content_sha256",
        "program_ids_ordered_sha256",
        "program_ids",
    }
    if not isinstance(candidate, Mapping) or not required_candidate_fields <= set(
        candidate
    ):
        raise TripsoContractError(
            "Tokenization manifest lacks its training-only projection GP binding"
        )
    scope_value = manifest.get("lineage_donor_scope")
    if scope_value is not None:
        if not isinstance(scope_value, Mapping):
            raise TripsoContractError(
                "Tokenization lineage donor scope must be a mapping"
            )
        try:
            validate_lineage_donor_scope(
                scope_value,
                lineage=(
                    manifest["lineages"][0]
                    if len(manifest.get("lineages", [])) == 1
                    else None
                ),
            )
        except ValueError as exc:
            raise TripsoContractError(str(exc)) from exc
    if verify_paths:
        required = (
            "tokenized_dataset_path",
            "gene_vocabulary_path",
            "gp_library_path",
            "input_h5ad",
        )
        missing = [name for name in required if not Path(manifest[name]).exists()]
        if missing:
            raise FileNotFoundError(
                f"Tokenization manifest resources are missing: {missing}"
            )
        resource_paths = {
            "input_h5ad": "input_h5ad",
            "gene_vocabulary": "gene_vocabulary_path",
            "gp_library": "gp_library_path",
        }
        for name, manifest_key in resource_paths.items():
            observed = sha256_path(Path(manifest[manifest_key]))
            expected = manifest["hashes"][f"{name}_sha256"]
            if observed != expected:
                raise TripsoContractError(
                    f"Tokenization resource changed after manifest creation: {name}"
                )
        materialization_path = manifest.get("input_materialization_manifest")
        materialization_hash = manifest["hashes"].get(
            "input_materialization_manifest_sha256"
        )
        if (materialization_path is None) != (materialization_hash is None):
            raise TripsoContractError(
                "Tokenization materialization-manifest path/hash binding is incomplete"
            )
        if materialization_path is not None:
            materialization_path = Path(str(materialization_path))
            if not materialization_path.is_file():
                raise FileNotFoundError(
                    f"Input materialization manifest is missing: {materialization_path}"
                )
            if sha256_path(materialization_path) != materialization_hash:
                raise TripsoContractError(
                    "Input materialization manifest changed after tokenization"
                )
            with materialization_path.open(encoding="utf-8") as handle:
                materialization_payload = json.load(handle)
            ontology_record = manifest.get("fine_type_ontology")
            if not isinstance(ontology_record, Mapping):
                raise TripsoContractError(
                    "Tokenization lacks approved fine-type ontology provenance"
                )
            if (
                ontology_record.get("approval_status") != "approved"
                or not str(ontology_record.get("sha256", "")).strip()
            ):
                raise TripsoContractError(
                    "Tokenization fine-type ontology is not approved/hash-bound"
                )
            scope_value = manifest.get("lineage_donor_scope")
            reference_design = str(materialization_payload.get("reference_design", ""))
            if reference_design in {"lodo", "all_healthy"} and not isinstance(
                scope_value, Mapping
            ):
                raise TripsoContractError(
                    "Reference tokenization lacks its lineage-specific donor scope"
                )
            if scope_value is not None:
                try:
                    validated_scope = validate_lineage_donor_scope(
                        scope_value,
                        lineage=(
                            manifest["lineages"][0]
                            if len(manifest.get("lineages", [])) == 1
                            else None
                        ),
                    )
                except ValueError as exc:
                    raise TripsoContractError(str(exc)) from exc
                materialized_scope = materialization_payload.get("lineage_donor_scope")
                if materialized_scope != validated_scope:
                    raise TripsoContractError(
                        "Tokenization lineage donor scope differs from its "
                        "materialization manifest"
                    )
        tokenizer_resources = manifest.get("tokenizer_resources")
        if not isinstance(tokenizer_resources, Mapping):
            raise TripsoContractError(
                "Tokenization manifest lacks portable tokenizer-resource paths"
            )
        tokenizer_hashes = {
            "token_dictionary_path": "token_dictionary_sha256",
            "median_dictionary_path": "median_dictionary_sha256",
            "implementation_path": "vendor_implementation_sha256",
        }
        for path_key, hash_key in tokenizer_hashes.items():
            resource_path = Path(str(tokenizer_resources.get(path_key, "")))
            if not resource_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer resource is missing after transfer: {resource_path}"
                )
            if sha256_path(resource_path) != manifest["hashes"].get(hash_key):
                raise TripsoContractError(
                    f"Tokenizer resource changed after tokenization: {path_key}"
                )
        _validate_directory_file_inventory(
            Path(manifest["tokenized_dataset_path"]),
            manifest.get("tokenized_dataset_integrity", {}),
        )
        sidecars = manifest.get("files")
        if not isinstance(sidecars, Mapping):
            raise TripsoContractError("Tokenization manifest lacks sidecar files")
        for name in ("donor_scope", "gp_token_coverage", "sequence_qc"):
            sidecar_path = Path(str(sidecars.get(name, "")))
            if not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"Tokenization sidecar is missing after transfer: {sidecar_path}"
                )
            if sha256_path(sidecar_path) != manifest["hashes"].get(f"{name}_sha256"):
                raise TripsoContractError(
                    f"Tokenization sidecar changed after manifest creation: {name}"
                )
        candidate_path = Path(str(candidate["path"]))
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"Projection GP candidate resource is missing: {candidate_path}"
            )
        if sha256_path(candidate_path) != candidate["sha256"]:
            raise TripsoContractError(
                "Projection GP candidate resource changed after tokenization"
            )
        observed_candidate = _projection_gp_selection(
            Path(manifest["gp_library_path"]),
            gp_allowlist_path=candidate_path,
            allow_all_gps=False,
            include_cell_token=False,
            include_gene_encoder_cls=False,
        )
        comparisons = {
            "manifest_content_sha256": observed_candidate[
                "candidate_manifest_content_sha256"
            ],
            "program_ids_ordered_sha256": observed_candidate[
                "program_ids_ordered_sha256"
            ],
            "program_ids": observed_candidate["program_ids"],
        }
        mismatched = {
            key: (candidate[key], value)
            for key, value in comparisons.items()
            if candidate[key] != value
        }
        if mismatched:
            raise TripsoContractError(
                "Projection GP candidate binding differs from tokenization: "
                f"{mismatched}"
            )
    return manifest


def relocate_tokenization_manifest(
    *,
    source_manifest_path: Path,
    output_manifest_path: Path,
    tokenized_dataset_path: Path,
    input_h5ad: Path,
    gene_vocabulary_path: Path,
    gp_library_path: Path,
    projection_gp_candidates_path: Path,
    vendor_root: Path,
    materialization_manifest_path: Path | None = None,
    overwrite: bool = False,
    _vendor_surface: tuple[type, Path, Path, Path] | None = None,
) -> dict[str, Any]:
    """Rebind an exact SFTP copy without weakening its scientific contract.

    Only locations are changed. Every H5AD, Arrow shard, sidecar, vocabulary,
    GP database, candidate list, and tokenizer asset must be byte-identical to
    the source manifest. The source manifest itself is verified before any of its
    declarations are used and is retained by both file and canonical-content hash.
    """

    source_manifest_path = Path(source_manifest_path).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()
    tokenized_dataset_path = Path(tokenized_dataset_path).resolve()
    input_h5ad = Path(input_h5ad).resolve()
    gene_vocabulary_path = Path(gene_vocabulary_path).resolve()
    gp_library_path = Path(gp_library_path).resolve()
    projection_gp_candidates_path = Path(projection_gp_candidates_path).resolve()
    vendor_root = Path(vendor_root).resolve()
    if output_manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Relocated tokenization manifest already exists: {output_manifest_path}"
        )

    source_manifest_file_hash = sha256_path(source_manifest_path)
    # ``verify_paths=False`` is essential: the whole point of this operation is
    # that the source cluster's absolute locations no longer exist. It still
    # verifies the source manifest's canonical self-hash and safety declarations.
    source = load_tokenization_manifest(source_manifest_path, verify_paths=False)
    source_content_hash = str(source["manifest_sha256"])
    source_hashes = source.get("hashes")
    if not isinstance(source_hashes, Mapping):
        raise TripsoContractError("Source tokenization manifest lacks resource hashes")

    genes = _read_gene_vocabulary(gene_vocabulary_path)
    immutable_files = {
        "input_h5ad": (input_h5ad, "input_h5ad_sha256"),
        "gene_vocabulary": (gene_vocabulary_path, "gene_vocabulary_sha256"),
        "gp_library": (gp_library_path, "gp_library_sha256"),
    }
    for name, (resource_path, hash_key) in immutable_files.items():
        expected = source_hashes.get(hash_key)
        if not isinstance(expected, str) or sha256_path(resource_path) != expected:
            raise TripsoContractError(
                f"Relocated {name} is not byte-identical to the source artifact"
            )

    source_materialization = source.get("input_materialization_manifest")
    expected_materialization_hash = source_hashes.get(
        "input_materialization_manifest_sha256"
    )
    relocated_materialization: Path | None = None
    if source_materialization is not None:
        relocated_materialization = (
            Path(materialization_manifest_path).resolve()
            if materialization_manifest_path is not None
            else _materialization_manifest_path(input_h5ad)
        )
        if not relocated_materialization.is_file():
            raise FileNotFoundError(
                "The source tokenization used a materialization manifest, but its "
                f"transferred copy is missing: {relocated_materialization}"
            )
        if sha256_path(relocated_materialization) != expected_materialization_hash:
            raise TripsoContractError(
                "Relocated materialization manifest is not byte-identical to source"
            )
    elif materialization_manifest_path is not None:
        raise TripsoContractError(
            "A materialization manifest was supplied although the source contract "
            "did not bind one"
        )

    candidate_selection = _projection_gp_selection(
        gp_library_path,
        gp_allowlist_path=projection_gp_candidates_path,
        allow_all_gps=False,
        include_cell_token=False,
        include_gene_encoder_cls=False,
    )
    source_candidate = source.get("projection_gp_candidates")
    if not isinstance(source_candidate, Mapping):
        raise TripsoContractError("Source tokenization lacks its GP candidate binding")
    candidate_comparisons = {
        "sha256": candidate_selection["allowlist_sha256"],
        "manifest_content_sha256": candidate_selection[
            "candidate_manifest_content_sha256"
        ],
        "program_ids_ordered_sha256": candidate_selection["program_ids_ordered_sha256"],
        "program_ids": candidate_selection["program_ids"],
    }
    for key, observed in candidate_comparisons.items():
        if source_candidate.get(key) != observed:
            raise TripsoContractError(
                f"Relocated projection GP candidates differ from source binding: {key}"
            )

    vendor_surface = _vendor_surface or _load_vendor_surface(vendor_root)
    _, token_dictionary_path, median_path, implementation_path = vendor_surface
    token_dictionary_path = Path(token_dictionary_path).resolve()
    median_path = Path(median_path).resolve()
    implementation_path = Path(implementation_path).resolve()
    vendor_files = {
        "token dictionary": (token_dictionary_path, "token_dictionary_sha256"),
        "median dictionary": (median_path, "median_dictionary_sha256"),
        "tokenizer implementation": (
            implementation_path,
            "vendor_implementation_sha256",
        ),
    }
    for name, (resource_path, hash_key) in vendor_files.items():
        expected = source_hashes.get(hash_key)
        if not isinstance(expected, str) or sha256_path(resource_path) != expected:
            raise TripsoContractError(
                f"Relocated vendor {name} differs from the source tokenization"
            )
    try:
        implementation_identity = str(implementation_path.relative_to(vendor_root))
    except ValueError:
        implementation_identity = implementation_path.name
    if implementation_identity != source.get("tokenizer_contract", {}).get(
        "vendor_implementation"
    ):
        raise TripsoContractError(
            "Relocated tokenizer implementation identity differs from source"
        )

    _validate_directory_file_inventory(
        tokenized_dataset_path,
        source.get("tokenized_dataset_integrity", {}),
    )
    source_sidecars = source.get("files")
    if not isinstance(source_sidecars, Mapping):
        raise TripsoContractError("Source tokenization manifest lacks sidecar paths")
    relocated_sidecars: dict[str, str] = {}
    for name in ("donor_scope", "gp_token_coverage", "sequence_qc"):
        source_sidecar = source_sidecars.get(name)
        if not isinstance(source_sidecar, str):
            raise TripsoContractError(f"Source tokenization lacks sidecar {name}")
        relocated_sidecar = tokenized_dataset_path.parent / Path(source_sidecar).name
        if not relocated_sidecar.is_file():
            raise FileNotFoundError(
                f"Relocated tokenization sidecar is missing: {relocated_sidecar}"
            )
        if sha256_path(relocated_sidecar) != source_hashes.get(f"{name}_sha256"):
            raise TripsoContractError(
                f"Relocated tokenization sidecar differs from source: {name}"
            )
        relocated_sidecars[name] = str(relocated_sidecar)

    input_audit = _validate_h5ad_contract(
        input_h5ad,
        genes,
        metadata_columns=REQUIRED_METADATA_COLUMNS,
        role=str(source["role"]),
    )
    if input_audit["shape"] != source.get("shape"):
        raise TripsoContractError("Relocated H5AD shape differs from source contract")
    if input_audit["cell_key_ordered_sha256"] != source.get("cell_key_ordered_sha256"):
        raise TripsoContractError(
            "Relocated H5AD cell order differs from source contract"
        )

    # JSON round-tripping gives a deep, JSON-compatible copy without trusting or
    # mutating caller-owned mappings.
    relocated: dict[str, Any] = json.loads(json.dumps(source))
    relocated.pop("manifest_sha256", None)
    relocated.update(
        {
            "input_h5ad": str(input_h5ad),
            "input_materialization_manifest": (
                str(relocated_materialization)
                if relocated_materialization is not None
                else None
            ),
            "tokenized_dataset_path": str(tokenized_dataset_path),
            "gene_vocabulary_path": str(gene_vocabulary_path),
            "gp_library_path": str(gp_library_path),
            "vendor_root": str(vendor_root),
            "tokenizer_resources": {
                "token_dictionary_path": str(token_dictionary_path),
                "median_dictionary_path": str(median_path),
                "implementation_path": str(implementation_path),
            },
            "files": relocated_sidecars,
        }
    )
    relocated["projection_gp_candidates"] = {
        **dict(source_candidate),
        "path": str(projection_gp_candidates_path),
    }
    relocated["relocation"] = {
        "schema_version": TOKENIZATION_RELOCATION_SCHEMA,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_file_sha256": source_manifest_file_hash,
        "source_manifest_content_sha256": source_content_hash,
        "validation": {
            "source_manifest_self_hash": "passed",
            "immutable_resource_hashes": "passed",
            "tokenizer_asset_hashes": "passed",
            "arrow_file_inventory": "passed",
            "physical_scope": "passed",
        },
        "scientific_inputs_changed": False,
    }
    physical_scope = validate_physical_tokenization_scope(relocated)
    if (
        physical_scope["cell_key_ordered_sha256"]
        != input_audit["cell_key_ordered_sha256"]
    ):
        raise TripsoContractError(
            "Relocated Arrow rows do not match the relocated materialized H5AD"
        )
    relocated["manifest_sha256"] = canonical_json_hash(relocated)
    atomic_write_json(output_manifest_path, relocated)
    return load_tokenization_manifest(output_manifest_path, verify_paths=True)


def validate_physical_tokenization_scope(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read Arrow metadata and prove it still matches its tokenization manifest.

    This check is intentionally repeated when a model-bound projection input is
    built and again immediately before GPU projection.  A JSON declaration alone
    is not accepted as proof of the physical donor or cell scope.
    """

    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise RuntimeError("Hugging Face datasets is required") from exc
    dataset_path = Path(str(manifest["tokenized_dataset_path"]))
    physical = load_from_disk(str(dataset_path))
    missing = sorted(set(REQUIRED_METADATA_COLUMNS) - set(physical.column_names))
    if missing:
        raise TripsoContractError(
            f"Physical tokenized dataset lacks required columns: {missing}"
        )
    expected_shape = manifest.get("shape")
    if (
        not isinstance(expected_shape, list)
        or not expected_shape
        or int(expected_shape[0]) != len(physical)
    ):
        raise TripsoContractError(
            "Physical tokenized row count differs from tokenization manifest"
        )
    scope, cell_key_hash, physical_donors = _physical_scope(physical)
    declared_donors = tuple(sorted(map(str, manifest.get("biological_unit_ids", []))))
    if physical_donors != declared_donors:
        raise TripsoContractError(
            "Physical tokenized donor inventory differs from its manifest; "
            f"declared={list(declared_donors)[:5]}, "
            f"physical={list(physical_donors)[:5]}"
        )
    if canonical_json_hash(list(physical_donors)) != manifest.get(
        "biological_unit_ids_sha256"
    ):
        raise TripsoContractError(
            "Physical tokenized donor hash differs from its manifest"
        )
    if cell_key_hash != manifest.get("cell_key_ordered_sha256"):
        raise TripsoContractError(
            "Physical tokenized cell order/content differs from its manifest"
        )
    physical_datasets = tuple(sorted(scope["dataset"].astype(str).unique()))
    declared_datasets = tuple(sorted(map(str, manifest.get("datasets", []))))
    if physical_datasets != declared_datasets:
        raise TripsoContractError(
            "Physical tokenized dataset labels differ from its manifest"
        )

    lineages: set[str] = set()
    selected = physical.select_columns(["lineage"])
    for batch in selected.iter(batch_size=100_000):
        lineages.update(map(str, batch["lineage"]))
    physical_lineages = tuple(sorted(lineages))
    declared_lineages = tuple(sorted(map(str, manifest.get("lineages", []))))
    if physical_lineages != declared_lineages:
        raise TripsoContractError(
            "Physical tokenized lineage labels differ from its manifest"
        )
    return {
        "n_cells": len(physical),
        "biological_unit_ids": list(physical_donors),
        "biological_unit_ids_sha256": canonical_json_hash(list(physical_donors)),
        "cell_key_ordered_sha256": cell_key_hash,
        "datasets": list(physical_datasets),
        "lineages": list(physical_lineages),
    }


def _restrict_fold_rows_to_lineage_scope(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    lineage_donor_scope: Mapping[str, Any] | None,
    lineage: str,
    held_out_dataset: str | None,
    partition_column: str,
    reference_design: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Replace global-fold expectations with the proven physical lineage scope."""

    rows = [dict(row) for row in fold_rows]
    if lineage_donor_scope is None:
        return rows, None
    try:
        scope = validate_lineage_donor_scope(
            lineage_donor_scope,
            lineage=lineage,
        )
    except ValueError as exc:
        raise TripsoContractError(str(exc)) from exc

    global_fold = validate_fold_rows(
        rows,
        held_out_dataset,
        partition_column=partition_column,
        reference_design=reference_design,
    )
    global_by_role = {
        "adaptation": set(global_fold.adaptation_donors),
        "validation": set(global_fold.validation_donors),
        "query": set(global_fold.query_donors),
    }
    available_by_role = {
        role: set(
            map(
                str,
                scope["biological_unit_ids_by_preparation_role"][role],
            )
        )
        for role in ("adaptation", "validation", "query")
    }
    for role in ("adaptation", "validation", "query"):
        misplaced = sorted(available_by_role[role] - global_by_role[role])
        if misplaced:
            raise TripsoContractError(
                "Lineage donor scope disagrees with the current fold role for "
                f"{role!r}: {misplaced[:5]}"
            )

    global_donors = set().union(*global_by_role.values())
    available_donors = set().union(*available_by_role.values())
    excluded_donors = global_donors - available_donors
    recorded_excluded = set(
        map(
            str,
            scope["global_fold_biological_unit_ids_without_materialized_role_cells"],
        )
    )
    if excluded_donors != recorded_excluded:
        raise TripsoContractError(
            "Lineage donor scope does not describe the same global fold; "
            f"unrecorded={sorted(excluded_donors - recorded_excluded)[:5]}, "
            f"unexpected={sorted(recorded_excluded - excluded_donors)[:5]}"
        )

    restricted = []
    for row in rows:
        donor = f"{str(row['dataset']).strip()}::{str(row['donor_id']).strip()}"
        if donor in available_donors:
            restricted.append(row)
    if not restricted:
        raise TripsoContractError("Lineage donor scope leaves the fold empty")

    excluded_by_role = {
        role: sorted(global_by_role[role] - available_by_role[role])
        for role in ("adaptation", "validation", "query")
    }
    expected_by_role = {
        role: sorted(available_by_role[role])
        for role in ("adaptation", "validation", "query")
    }
    audit = {
        "status": "passed",
        "lineage": str(lineage),
        "scope_source": scope["scope_source"],
        "lineage_donor_scope_sha256": scope["scope_sha256"],
        "global_fold_donor_inventory_used": False,
        "expected_scope": (
            "physical_per_lineage_donors_after_fold_and_visit_selection"
        ),
        "n_expected_biological_units_by_role": {
            role: len(donors) for role, donors in expected_by_role.items()
        },
        "expected_biological_unit_ids_by_role_sha256": {
            role: canonical_json_hash(donors)
            for role, donors in expected_by_role.items()
        },
        "global_fold_biological_unit_ids_excluded_by_original_role": (excluded_by_role),
        "n_global_fold_biological_units_excluded_by_original_role": {
            role: len(donors) for role, donors in excluded_by_role.items()
        },
        "exclusion_reason": ("no_materialized_cells_in_lineage_after_visit_selection"),
        "n_global_fold_biological_units": len(global_donors),
        "global_fold_biological_unit_ids_sha256": canonical_json_hash(
            sorted(global_donors)
        ),
        "n_lineage_available_biological_units": len(available_donors),
    }
    return restricted, audit


def build_fold_input_from_tokenization(
    *,
    tokenization_manifest_path: Path,
    fold_table_path: Path,
    output_path: Path,
    fold_id: str,
    held_out_dataset: str | None,
    lineage: str,
    partition_column: str = "outer_role",
    sampler_manifest_path: Path | None = None,
    reference_design: str = "lodo",
    inner_validation_fold: int | None = None,
    inner_fold_column: str = "inner_fold",
) -> dict[str, Any]:
    """Bind a physical adaptation Arrow dataset to the donor fold contract."""

    tokenization = load_tokenization_manifest(tokenization_manifest_path)
    if tokenization["role"] != "adaptation":
        raise TripsoContractError(
            "Training fold input must be built from an adaptation tokenization"
        )
    if set(map(str, tokenization.get("lineages", ()))) != {str(lineage)}:
        raise TripsoContractError(
            "Training tokenization lineage differs from the requested fold lineage"
        )
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise RuntimeError("Hugging Face datasets is required") from exc
    physical = load_from_disk(tokenization["tokenized_dataset_path"])
    if "biological_unit_id" not in physical.column_names:
        raise TripsoContractError(
            "Physical tokenized dataset lacks biological_unit_id scope proof"
        )
    physical_donors = tuple(sorted(set(map(str, physical["biological_unit_id"]))))
    if canonical_json_hash(list(physical_donors)) != tokenization.get(
        "biological_unit_ids_sha256"
    ):
        raise TripsoContractError(
            "Physical tokenized donor scope differs from tokenization manifest"
        )
    candidates = tokenization.get("projection_gp_candidates")
    if not isinstance(candidates, Mapping) or not candidates.get("path"):
        raise TripsoContractError(
            "Adaptation tokenization lacks projection GP candidate provenance"
        )
    fold_rows = read_table(fold_table_path)
    if inner_validation_fold is not None:
        if reference_design != "lodo":
            raise TripsoContractError(
                "Fixed inner-validation selection is supported only for LODO folds"
            )
        if isinstance(inner_validation_fold, bool) or inner_validation_fold < 0:
            raise TripsoContractError(
                "inner_validation_fold must be a nonnegative integer"
            )
        if not inner_fold_column.strip():
            raise TripsoContractError("inner_fold_column must be nonempty")
        n_adaptation = 0
        n_validation = 0
        rebound_rows: list[dict[str, Any]] = []
        for index, original in enumerate(fold_rows, start=1):
            row = dict(original)
            dataset = str(row.get("dataset", "")).strip()
            outer_role = str(row.get("outer_role", "")).strip().lower()
            is_query = dataset == str(held_out_dataset) or outer_role == "query"
            if is_query:
                row["eligible_for_reference_fitting"] = False
                row["inner_selection_role"] = "outer_query_evaluation_only"
                rebound_rows.append(row)
                continue
            if outer_role not in {"reference", ""}:
                raise TripsoContractError(
                    f"LODO reference row {index} has invalid outer_role {outer_role!r}"
                )
            raw_fold = row.get(inner_fold_column)
            try:
                numeric_fold = float(str(raw_fold).strip())
                observed_fold = int(numeric_fold)
            except (TypeError, ValueError) as exc:
                raise TripsoContractError(
                    f"LODO reference row {index} lacks integer {inner_fold_column!r}"
                ) from exc
            if numeric_fold != observed_fold or observed_fold < 0:
                raise TripsoContractError(
                    f"LODO reference row {index} has invalid {inner_fold_column!r}"
                )
            is_validation = observed_fold == int(inner_validation_fold)
            row["eligible_for_reference_fitting"] = not is_validation
            row["inner_selection_role"] = (
                "validation" if is_validation else "adaptation"
            )
            n_validation += int(is_validation)
            n_adaptation += int(not is_validation)
            rebound_rows.append(row)
        if n_adaptation < 1 or n_validation < 1:
            raise TripsoContractError(
                "Fixed inner fold must yield both adaptation and validation donors"
            )
        fold_rows = rebound_rows
    fold_rows, lineage_scope_validation = _restrict_fold_rows_to_lineage_scope(
        fold_rows,
        lineage_donor_scope=tokenization.get("lineage_donor_scope"),
        lineage=lineage,
        held_out_dataset=held_out_dataset,
        partition_column=partition_column,
        reference_design=reference_design,
    )
    return prepare_fold_input_manifest(
        rows=fold_rows,
        output_path=output_path,
        fold_id=fold_id,
        held_out_dataset=held_out_dataset,
        lineage=lineage,
        tokenized_dataset_path=Path(tokenization["tokenized_dataset_path"]),
        gp_library_path=Path(tokenization["gp_library_path"]),
        gene_vocabulary_path=Path(tokenization["gene_vocabulary_path"]),
        projection_gp_candidates_path=Path(str(candidates["path"])),
        source_h5ad_path=Path(tokenization["input_h5ad"]),
        sampler_manifest_path=sampler_manifest_path,
        tokenization_manifest_path=Path(tokenization_manifest_path),
        tokenized_biological_unit_ids=physical_donors,
        partition_column=partition_column,
        reference_design=reference_design,
        inner_validation_fold=inner_validation_fold,
        inner_fold_column=(
            inner_fold_column if inner_validation_fold is not None else None
        ),
        lineage_donor_scope_validation=lineage_scope_validation,
    )


def build_query_input_from_tokenization(
    *,
    tokenization_manifest_path: Path,
    model_manifest_path: Path,
    output_path: Path,
    seed: int | None = None,
    gp_allowlist_path: Path | None = None,
    use_fold_bound_gp_candidates: bool = False,
    allow_all_gps: bool = False,
    include_cell_token: bool = False,
    include_gene_encoder_cls: bool = False,
    max_projected_bytes: int = DEFAULT_MAX_PROJECTED_BYTES,
    allow_oversized_projection: bool = False,
) -> dict[str, Any]:
    """Bind query tokens to the exact tokenizer, genes, GPs, and trained model."""

    return build_projection_input_from_tokenization(
        tokenization_manifest_path=tokenization_manifest_path,
        model_manifest_path=model_manifest_path,
        output_path=output_path,
        role="query",
        seed=seed,
        gp_allowlist_path=gp_allowlist_path,
        use_fold_bound_gp_candidates=use_fold_bound_gp_candidates,
        allow_all_gps=allow_all_gps,
        include_cell_token=include_cell_token,
        include_gene_encoder_cls=include_gene_encoder_cls,
        max_projected_bytes=max_projected_bytes,
        allow_oversized_projection=allow_oversized_projection,
    )


def build_reference_input_from_tokenization(
    *,
    tokenization_manifest_path: Path,
    model_manifest_path: Path,
    output_path: Path,
    seed: int | None = None,
    gp_allowlist_path: Path | None = None,
    use_fold_bound_gp_candidates: bool = False,
    allow_all_gps: bool = False,
    include_cell_token: bool = False,
    include_gene_encoder_cls: bool = False,
    max_projected_bytes: int = DEFAULT_MAX_PROJECTED_BYTES,
    allow_oversized_projection: bool = False,
) -> dict[str, Any]:
    """Bind every adaptation cell back to its newly frozen trained model."""

    return build_projection_input_from_tokenization(
        tokenization_manifest_path=tokenization_manifest_path,
        model_manifest_path=model_manifest_path,
        output_path=output_path,
        role="reference",
        seed=seed,
        gp_allowlist_path=gp_allowlist_path,
        use_fold_bound_gp_candidates=use_fold_bound_gp_candidates,
        allow_all_gps=allow_all_gps,
        include_cell_token=include_cell_token,
        include_gene_encoder_cls=include_gene_encoder_cls,
        max_projected_bytes=max_projected_bytes,
        allow_oversized_projection=allow_oversized_projection,
    )


def build_validation_input_from_tokenization(
    *,
    tokenization_manifest_path: Path,
    model_manifest_path: Path,
    output_path: Path,
    seed: int | None = None,
    gp_allowlist_path: Path | None = None,
    use_fold_bound_gp_candidates: bool = False,
    allow_all_gps: bool = False,
    include_cell_token: bool = False,
    include_gene_encoder_cls: bool = False,
    max_projected_bytes: int = DEFAULT_MAX_PROJECTED_BYTES,
    allow_oversized_projection: bool = False,
) -> dict[str, Any]:
    """Bind the exact fixed inner-validation donors to frozen model weights."""

    return build_projection_input_from_tokenization(
        tokenization_manifest_path=tokenization_manifest_path,
        model_manifest_path=model_manifest_path,
        output_path=output_path,
        role="validation",
        seed=seed,
        gp_allowlist_path=gp_allowlist_path,
        use_fold_bound_gp_candidates=use_fold_bound_gp_candidates,
        allow_all_gps=allow_all_gps,
        include_cell_token=include_cell_token,
        include_gene_encoder_cls=include_gene_encoder_cls,
        max_projected_bytes=max_projected_bytes,
        allow_oversized_projection=allow_oversized_projection,
    )


def build_projection_input_from_tokenization(
    *,
    tokenization_manifest_path: Path,
    model_manifest_path: Path,
    output_path: Path,
    role: str,
    seed: int | None = None,
    gp_allowlist_path: Path | None = None,
    use_fold_bound_gp_candidates: bool = False,
    allow_all_gps: bool = False,
    include_cell_token: bool = False,
    include_gene_encoder_cls: bool = False,
    max_projected_bytes: int = DEFAULT_MAX_PROJECTED_BYTES,
    allow_oversized_projection: bool = False,
) -> dict[str, Any]:
    """Bind an all-cell Arrow dataset to a model for inference-only projection.

    ``reference`` is the exact adaptation dataset used to fit the checkpoint;
    ``validation`` is the exact fixed inner donor fold; and ``query`` is the
    untouched outer cohort. All roles use frozen weights.
    """

    if role not in PROJECTION_ROLES:
        raise TripsoContractError(
            f"Projection role must be one of {sorted(PROJECTION_ROLES)}"
        )

    tokenization = load_tokenization_manifest(tokenization_manifest_path)
    expected_tokenization_role = {
        "reference": "adaptation",
        "validation": "validation",
        "query": "query",
    }[role]
    if tokenization["role"] != expected_tokenization_role:
        raise TripsoContractError(
            f"{role.title()} projection must use a "
            f"{expected_tokenization_role} tokenization"
        )
    physical_scope = validate_physical_tokenization_scope(tokenization)
    model = validate_checkpoint_manifest(model_manifest_path)
    fold = load_fold_input_manifest(Path(model["paths"]["fold_input_manifest"]))
    observed_fold_hash = sha256_path(Path(model["paths"]["fold_input_manifest"]))
    if observed_fold_hash != model["hashes"].get("input_manifest_sha256"):
        raise TripsoContractError(
            "Model fold input changed after checkpoint provenance was recorded"
        )
    training_tokenization_path = fold.get("inputs", {}).get(
        "tokenization_manifest_path"
    )
    if not training_tokenization_path:
        raise TripsoContractError(
            "Model fold input predates the production tokenization contract; its "
            "query tokenizer cannot be proven identical"
        )
    training_tokenization = load_tokenization_manifest(
        Path(training_tokenization_path), verify_paths=False
    )
    comparisons = {
        "gp_library_sha256": (
            model["hashes"]["gp_library_sha256"],
            tokenization["hashes"]["gp_library_sha256"],
        ),
        "gene_vocabulary_sha256": (
            model["hashes"]["gene_vocabulary_sha256"],
            tokenization["hashes"]["gene_vocabulary_sha256"],
        ),
        "tokenizer_contract_sha256": (
            training_tokenization["tokenizer_contract_sha256"],
            tokenization["tokenizer_contract_sha256"],
        ),
        "token_dictionary_sha256": (
            training_tokenization["hashes"]["token_dictionary_sha256"],
            tokenization["hashes"]["token_dictionary_sha256"],
        ),
        "median_dictionary_sha256": (
            training_tokenization["hashes"]["median_dictionary_sha256"],
            tokenization["hashes"]["median_dictionary_sha256"],
        ),
    }
    training_ontology = training_tokenization.get("fine_type_ontology")
    projection_ontology = tokenization.get("fine_type_ontology")
    if training_ontology is not None or projection_ontology is not None:
        comparisons["fine_type_ontology_sha256"] = (
            (training_ontology or {}).get("sha256"),
            (projection_ontology or {}).get("sha256"),
        )
    training_scope = training_tokenization.get("lineage_donor_scope")
    projection_scope = tokenization.get("lineage_donor_scope")
    if (
        fold.get("reference_design") == "lodo" or role in {"reference", "validation"}
    ) and (training_scope is not None or projection_scope is not None):
        comparisons["lineage_donor_scope_sha256"] = (
            (training_scope or {}).get("scope_sha256"),
            (projection_scope or {}).get("scope_sha256"),
        )
    mismatches = {
        name: values for name, values in comparisons.items() if values[0] != values[1]
    }
    if mismatches:
        raise TripsoContractError(
            f"Frozen {role} tokenization differs from training: {mismatches}"
        )
    adaptation_donors = set(map(str, fold["adaptation_biological_unit_ids"]))
    validation_donors = set(map(str, fold["validation_biological_unit_ids"]))
    query_donors = set(map(str, fold["query_biological_unit_ids"]))
    projection_donors = set(map(str, physical_scope["biological_unit_ids"]))
    if role == "reference":
        missing = sorted(adaptation_donors - projection_donors)
        unexpected = sorted(projection_donors - adaptation_donors)
        if missing or unexpected:
            raise TripsoContractError(
                "Reference projection must contain exactly all adaptation donors; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        if (
            Path(tokenization_manifest_path).resolve()
            != Path(training_tokenization_path).resolve()
        ):
            raise TripsoContractError(
                "Reference projection must reuse the exact training tokenization "
                "manifest path recorded by the fold input"
            )
        expected_training_hash = fold.get("hashes", {}).get(
            "tokenization_manifest_sha256"
        )
        observed_training_hash = sha256_path(Path(tokenization_manifest_path))
        if (
            not expected_training_hash
            or observed_training_hash != expected_training_hash
        ):
            raise TripsoContractError(
                "Reference projection tokenization is not the immutable training "
                "tokenization recorded by the fold"
            )
    elif role == "validation":
        missing = sorted(validation_donors - projection_donors)
        unexpected = sorted(projection_donors - validation_donors)
        if missing or unexpected:
            raise TripsoContractError(
                "Validation projection must contain exactly the fixed inner-fold "
                f"donors; missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        if not validation_donors:
            raise TripsoContractError(
                "The trained fold does not declare inner-validation donors"
            )
    else:
        overlap = sorted((adaptation_donors | validation_donors) & projection_donors)
        if overlap:
            raise TripsoContractError(
                "Query tokenization contains adaptation/validation biological "
                f"units: {overlap[:10]}"
            )
        if fold.get("reference_design") == "lodo":
            missing = sorted(query_donors - projection_donors)
            unexpected = sorted(projection_donors - query_donors)
            if missing or unexpected:
                raise TripsoContractError(
                    "Outer-query projection must contain exactly the held-out fold "
                    f"donors; missing={missing[:10]}, unexpected={unexpected[:10]}"
                )
    projection_lineages = tokenization.get("lineages") or []
    if set(map(str, projection_lineages)) != {str(model["lineage"])}:
        raise TripsoContractError(
            f"{role.title()} tokenization lineage differs from the trained model"
        )
    selection_count = sum(
        (
            gp_allowlist_path is not None,
            bool(use_fold_bound_gp_candidates),
            bool(allow_all_gps),
        )
    )
    if selection_count != 1:
        raise TripsoContractError(
            "Choose exactly one explicit GP projection policy: fold-bound "
            "candidates, a matching candidate path, or allow_all_gps"
        )
    bound_candidate_path = fold.get("inputs", {}).get("projection_gp_candidates_path")
    bound_candidate_hash = fold.get("hashes", {}).get("projection_gp_candidates_sha256")
    bound_program_hash = fold.get("hashes", {}).get(
        "projection_gp_program_ids_ordered_sha256"
    )
    if not allow_all_gps:
        if (
            not bound_candidate_path
            or not bound_candidate_hash
            or not bound_program_hash
        ):
            raise TripsoContractError(
                "Training fold lacks an immutable projection-GP candidate binding"
            )
        if use_fold_bound_gp_candidates:
            gp_allowlist_path = Path(bound_candidate_path)
        assert gp_allowlist_path is not None
        if Path(gp_allowlist_path).resolve() != Path(bound_candidate_path).resolve():
            raise TripsoContractError(
                "Projection candidate path differs from the model-bound training fold"
            )
        if sha256_path(Path(gp_allowlist_path)) != bound_candidate_hash:
            raise TripsoContractError(
                "Projection candidate manifest differs from the model-bound hash"
            )
        for name, candidate_binding in (
            ("training", training_tokenization.get("projection_gp_candidates")),
            (role, tokenization.get("projection_gp_candidates")),
        ):
            if not isinstance(candidate_binding, Mapping):
                raise TripsoContractError(
                    f"{name.title()} tokenization lacks projection-GP binding"
                )
            expected_binding = {
                "path": str(Path(bound_candidate_path).resolve()),
                "sha256": bound_candidate_hash,
                "program_ids_ordered_sha256": bound_program_hash,
            }
            observed_binding = {
                key: candidate_binding.get(key) for key in expected_binding
            }
            if observed_binding != expected_binding:
                raise TripsoContractError(
                    f"{name.title()} tokenization projection-GP binding differs "
                    "from the trained fold"
                )
    gp_projection = _projection_gp_selection(
        Path(tokenization["gp_library_path"]),
        gp_allowlist_path=gp_allowlist_path,
        allow_all_gps=allow_all_gps,
        include_cell_token=include_cell_token,
        include_gene_encoder_cls=include_gene_encoder_cls,
    )
    if (
        not allow_all_gps
        and gp_projection["program_ids_ordered_sha256"] != bound_program_hash
    ):
        raise TripsoContractError(
            "Projection candidate programs differ from the model-bound fold"
        )
    if isinstance(max_projected_bytes, bool) or int(max_projected_bytes) < 1:
        raise TripsoContractError("max_projected_bytes must be a positive integer")
    embedding_dimension = model.get("model_configuration", {}).get(
        "gp_latent_dimension",
        model.get("model_configuration", {}).get("embedding_dimension"),
    )
    if isinstance(embedding_dimension, bool) or not isinstance(
        embedding_dimension, (int, float)
    ):
        raise TripsoContractError(
            "Model manifest lacks a numeric GP embedding dimension"
        )
    embedding_dimension = int(embedding_dimension)
    if embedding_dimension < 1:
        raise TripsoContractError("GP embedding dimension must be positive")
    estimated_gp_vector_bytes = (
        int(physical_scope["n_cells"])
        * int(gp_projection["n_programs"])
        * embedding_dimension
        * 4
    )
    if (
        estimated_gp_vector_bytes > int(max_projected_bytes)
        and not allow_oversized_projection
    ):
        raise TripsoContractError(
            "Projected GP payload exceeds the configured byte guard: "
            f"estimated={estimated_gp_vector_bytes}, "
            f"maximum={int(max_projected_bytes)}. Reduce the frozen GP allowlist "
            "or explicitly allow an oversized diagnostic."
        )
    gp_projection["embedding_dimension"] = embedding_dimension
    gp_projection["float_bytes_per_value"] = 4
    gp_projection["estimated_gp_vector_bytes"] = estimated_gp_vector_bytes
    gp_projection["maximum_projected_bytes"] = int(max_projected_bytes)
    gp_projection["oversized_projection_override"] = bool(allow_oversized_projection)
    projection_metadata_columns = [
        column
        for column in tokenization.get("metadata_columns", [])
        if column not in {"input_ids", "counts"}
    ]
    payload: dict[str, Any] = {
        "schema_version": PROJECTION_INPUT_SCHEMA,
        "projection_role": role,
        "adapt": False,
        "optimizer_allowed": False,
        "all_tokenized_cells_required": True,
        "model_manifest": str(Path(model_manifest_path).resolve()),
        "tokenization_manifest": str(Path(tokenization_manifest_path).resolve()),
        "tokenized_dataset_path": tokenization["tokenized_dataset_path"],
        "gp_library_path": tokenization["gp_library_path"],
        "gene_vocabulary_path": tokenization["gene_vocabulary_path"],
        "lineage": model["lineage"],
        "datasets": tokenization["datasets"],
        "biological_unit_ids": physical_scope["biological_unit_ids"],
        "n_cells": physical_scope["n_cells"],
        "cell_key_ordered_sha256": physical_scope["cell_key_ordered_sha256"],
        "projection_metadata_columns": projection_metadata_columns,
        "gp_projection": gp_projection,
        "seed": int(model["seed"] if seed is None else seed),
        "hashes": {
            "gp_library_sha256": tokenization["hashes"]["gp_library_sha256"],
            "gene_vocabulary_sha256": tokenization["hashes"]["gene_vocabulary_sha256"],
            "tokenizer_contract_sha256": tokenization["tokenizer_contract_sha256"],
            "token_dictionary_sha256": tokenization["hashes"][
                "token_dictionary_sha256"
            ],
            "median_dictionary_sha256": tokenization["hashes"][
                "median_dictionary_sha256"
            ],
            "tokenization_manifest_sha256": sha256_path(
                Path(tokenization_manifest_path)
            ),
            "model_manifest_sha256": sha256_path(Path(model_manifest_path)),
        },
        "model_configuration": model["model_configuration"],
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    atomic_write_json(output_path, payload)
    return payload
