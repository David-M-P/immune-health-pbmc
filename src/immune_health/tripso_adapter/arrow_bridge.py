"""Key-safe conversion of TRIPSO HuggingFace Arrow embeddings.

TRIPSO saves one vector column per gene program in a HuggingFace Dataset.  The
downstream aggregation code consumes dense NumPy arrays plus rectangular cell
metadata.  This bridge verifies alignment through the safe string ``cell_key``
before writing one memory-mapped float32 NPY per requested embedding column.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from immune_health.provenance import atomic_write_json, sha256_file, stable_hash

ARROW_BRIDGE_SCHEMA = "immune-health-tripso-arrow-bridge/v1"
PROJECTION_OUTPUT_SCHEMA = "immune-health-tripso-projection-output/v1"
DEFAULT_REQUIRED_METADATA = (
    "dataset",
    "donor_id",
    "sample_id",
    "observation_id",
    "lineage",
    "fine_type",
)


def _ordered_key_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _arrow_files(path: Path) -> tuple[Path, ...]:
    path = Path(path).resolve()
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"Arrow dataset does not exist: {path}")
    state_path = path / "state.json"
    files: list[Path] = []
    if state_path.is_file():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        for entry in state.get("_data_files", []):
            filename = entry.get("filename") if isinstance(entry, dict) else None
            if filename:
                files.append(path / filename)
    if not files:
        files = sorted(path.glob("*.arrow"))
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(f"No complete Arrow shards found in {path}")
    return tuple(files)


def _record_batches(
    files: Sequence[Path], columns: Sequence[str] | None = None
) -> Iterator[pa.RecordBatch]:
    requested = None if columns is None else tuple(columns)
    for path in files:
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_stream(source)
                batches = reader
            except pa.ArrowInvalid:
                source.seek(0)
                file_reader = ipc.open_file(source)
                batches = (
                    file_reader.get_batch(index)
                    for index in range(file_reader.num_record_batches)
                )
            for batch in batches:
                if requested is not None:
                    missing = [
                        column
                        for column in requested
                        if column not in batch.schema.names
                    ]
                    if missing:
                        raise ValueError(f"Arrow shard {path} lacks columns: {missing}")
                    batch = batch.select(requested)
                yield batch


def _read_metadata(path: Path) -> pd.DataFrame:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if ".parquet" in suffixes:
        return pd.read_parquet(path)
    if ".tsv" in suffixes or ".txt" in suffixes:
        return pd.read_csv(path, sep="\t")
    if ".csv" in suffixes:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported cell metadata format: {path}")


def _array_to_matrix(array: pa.Array | pa.ChunkedArray, column: str) -> np.ndarray:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        raise ValueError(f"Embedding column {column!r} contains null vectors")
    if pa.types.is_fixed_size_list(array.type):
        width = int(array.type.list_size)
        values = np.asarray(array.values.to_numpy(zero_copy_only=False))
        matrix = values.reshape(len(array), width)
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        values = array.to_pylist()
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise ValueError(f"Embedding column {column!r} has ragged vectors")
        matrix = np.asarray(values)
    else:
        matrix = np.asarray(array.to_numpy(zero_copy_only=False))
        if matrix.ndim == 1:
            matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"Embedding column {column!r} is not a vector column")
    try:
        result = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Embedding column {column!r} is not numeric") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"Embedding column {column!r} contains nonfinite values")
    return result


def _output_stem(column: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", column).strip("._") or "embedding"
    suffix = hashlib.sha256(column.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{suffix}"


def _float32_payload_digest(array: np.ndarray, *, row_chunk: int = 65_536) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(array), row_chunk):
        values = np.ascontiguousarray(array[start : start + row_chunk])
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def validate_projection_output_for_conversion(
    manifest_path: Path,
    arrow_dataset: Path,
    *,
    embedding_columns: Sequence[str],
) -> dict[str, object]:
    """Verify an immutable frozen-projection output before Arrow conversion."""

    manifest_path = Path(manifest_path).resolve()
    arrow_dataset = Path(arrow_dataset).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Projection output manifest is missing: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != PROJECTION_OUTPUT_SCHEMA:
        raise ValueError(f"Unsupported projection output manifest: {manifest_path}")
    content = dict(manifest)
    claimed_hash = content.pop("manifest_sha256", None)
    if claimed_hash != stable_hash(content):
        raise ValueError("Projection output manifest content hash does not match")
    expected_arrow = (
        manifest_path.parent / str(manifest.get("arrow_dataset", ""))
    ).resolve()
    if arrow_dataset != expected_arrow:
        raise ValueError(
            "Arrow dataset is not the output bound to the projection manifest: "
            f"expected {expected_arrow}, observed {arrow_dataset}"
        )
    if not arrow_dataset.is_dir():
        raise FileNotFoundError(f"Projected Arrow dataset is missing: {arrow_dataset}")

    expected_files = manifest.get("arrow_files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("Projection output manifest lacks Arrow shard hashes")
    observed_files = [
        {
            "path": path.relative_to(arrow_dataset).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(arrow_dataset.rglob("*"))
        if path.is_file()
    ]
    if observed_files != expected_files:
        raise ValueError("Projected Arrow tree differs from its output manifest")
    expected_tree_hash = manifest.get("hashes", {}).get("arrow_tree_sha256")
    if stable_hash(observed_files) != expected_tree_hash:
        raise ValueError("Projected Arrow tree hash is inconsistent")

    gp_projection = manifest.get("gp_projection")
    if not isinstance(gp_projection, dict):
        raise ValueError("Projection output manifest lacks GP selection provenance")
    program_ids = gp_projection.get("program_ids")
    if not isinstance(program_ids, list) or not program_ids:
        raise ValueError("Projection output GP program list is invalid")
    program_ids = list(map(str, program_ids))
    program_digest = stable_hash(program_ids)
    if (
        len(program_ids) != len(set(program_ids))
        or gp_projection.get("program_ids_ordered_sha256") != program_digest
        or manifest.get("hashes", {}).get("gp_program_ids_ordered_sha256")
        != program_digest
    ):
        raise ValueError("Projection output GP allowlist order/hash is invalid")
    if int(gp_projection.get("n_programs", len(program_ids))) != len(program_ids):
        raise ValueError("Projection output GP count is inconsistent")
    requested = tuple(dict.fromkeys(map(str, embedding_columns)))
    missing = sorted(set(requested) - set(program_ids))
    if missing:
        raise ValueError(
            "Requested embedding columns are outside the frozen GP allowlist: "
            f"{missing}"
        )
    role = str(manifest.get("projection_role", ""))
    if role not in {"reference", "validation", "query"}:
        raise ValueError("Projection output role is invalid")
    eligible_for_model_selection = manifest.get("eligible_for_model_selection")
    outer_query_evaluation_only = manifest.get("outer_query_evaluation_only")
    if eligible_for_model_selection is not (role == "validation"):
        raise ValueError(
            "Projection output model-selection eligibility disagrees with its role"
        )
    if outer_query_evaluation_only is not (role == "query"):
        raise ValueError(
            "Projection output outer-query evaluation flag disagrees with its role"
        )
    inner_model_selection = manifest.get("inner_model_selection")
    if role == "validation" and (
        not isinstance(inner_model_selection, dict)
        or inner_model_selection.get("enabled") is not True
        or inner_model_selection.get("outer_query_used_for_model_selection")
        is not False
    ):
        raise ValueError(
            "Validation projection lacks its fixed inner-selection provenance"
        )
    if (
        manifest.get("adapt") is not False
        or manifest.get("optimizer_used") is not False
        or manifest.get("all_tokenized_cells_projected") is not True
    ):
        raise ValueError("Projection output does not prove frozen all-cell inference")
    if int(manifest.get("n_cells", -1)) < 1 or not manifest.get(
        "cell_key_ordered_sha256"
    ):
        raise ValueError("Projection output physical cell scope is invalid")
    biological_units = list(map(str, manifest.get("biological_unit_ids", [])))
    if (
        not biological_units
        or biological_units != sorted(set(biological_units))
        or manifest.get("biological_unit_ids_sha256") != stable_hash(biological_units)
    ):
        raise ValueError("Projection output biological-unit scope is invalid")
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": claimed_hash,
        "projection_role": role,
        "eligible_for_model_selection": eligible_for_model_selection,
        "outer_query_evaluation_only": outer_query_evaluation_only,
        "inner_model_selection": inner_model_selection,
        "reference_design": manifest.get("reference_design"),
        "heldout_dataset": manifest.get("heldout_dataset"),
        "fold_id": manifest.get("fold_id"),
        "lineage": manifest.get("lineage"),
        "seed": manifest.get("seed"),
        "datasets": list(map(str, manifest.get("datasets", []))),
        "biological_unit_ids": biological_units,
        "n_cells": int(manifest.get("n_cells", -1)),
        "cell_key_ordered_sha256": manifest.get("cell_key_ordered_sha256"),
        "arrow_dataset": str(arrow_dataset),
        "arrow_tree_sha256": expected_tree_hash,
        "model_manifest": manifest.get("model_manifest"),
        "model_manifest_sha256": manifest.get("hashes", {}).get(
            "model_manifest_sha256"
        ),
        "checkpoint_sha256": manifest.get("hashes", {}).get("checkpoint_sha256"),
        "gp_program_ids": program_ids,
        "gp_program_ids_ordered_sha256": manifest.get("hashes", {}).get(
            "gp_program_ids_ordered_sha256"
        ),
    }


def validate_arrow_conversion_for_aggregation(
    manifest_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    *,
    embedding_column: str,
) -> dict[str, object]:
    """Bind one aggregation input pair to a key-checked Arrow conversion.

    Equal row counts cannot prove cell alignment.  This validator requires the
    exact NPY and Parquet outputs named by the conversion manifest, rehashes both,
    and recomputes the ordered ``cell_key`` digest before downstream grouping.
    """

    manifest_path = Path(manifest_path).resolve()
    embeddings_path = Path(embeddings_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Arrow conversion manifest is missing: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != ARROW_BRIDGE_SCHEMA:
        raise ValueError(f"Unsupported Arrow conversion manifest: {manifest_path}")
    content = dict(manifest)
    claimed_hash = content.pop("manifest_sha256", None)
    if claimed_hash != stable_hash(content):
        raise ValueError("Arrow conversion manifest content hash does not match")

    projection_record = manifest.get("projection_output")
    if not isinstance(projection_record, dict):
        raise ValueError("Arrow conversion lacks projection-output provenance")
    projection_manifest_path = Path(str(projection_record.get("manifest_path", "")))
    projection_validation = validate_projection_output_for_conversion(
        projection_manifest_path,
        Path(str(manifest.get("arrow_dataset", ""))),
        embedding_columns=[embedding_column],
    )
    if projection_validation["manifest_sha256"] != projection_record.get(
        "manifest_sha256"
    ):
        raise ValueError("Projection output manifest changed after Arrow conversion")

    outputs = manifest.get("embedding_outputs")
    if not isinstance(outputs, dict) or embedding_column not in outputs:
        raise ValueError(
            f"Arrow conversion manifest has no embedding column {embedding_column!r}"
        )
    embedding = outputs[embedding_column]
    if not isinstance(embedding, dict):
        raise ValueError("Arrow conversion embedding record is invalid")
    expected_embeddings = (
        manifest_path.parent / str(embedding.get("path", ""))
    ).resolve()
    expected_metadata = (
        manifest_path.parent / str(manifest.get("metadata_output", ""))
    ).resolve()
    if embeddings_path != expected_embeddings:
        raise ValueError(
            "Embedding path is not the output bound to the Arrow conversion "
            f"manifest: expected {expected_embeddings}, observed {embeddings_path}"
        )
    if metadata_path != expected_metadata:
        raise ValueError(
            "Metadata path is not the output bound to the Arrow conversion "
            f"manifest: expected {expected_metadata}, observed {metadata_path}"
        )
    if not embeddings_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Arrow conversion NPY or metadata output is missing")
    if sha256_file(metadata_path) != manifest.get("metadata_sha256"):
        raise ValueError("Arrow conversion metadata SHA-256 does not match manifest")

    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    expected_shape = embedding.get("shape")
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 2
        or list(values.shape) != expected_shape
    ):
        raise ValueError(
            "Arrow conversion embedding shape does not match manifest: "
            f"expected {expected_shape}, observed {list(values.shape)}"
        )
    if values.dtype != np.dtype("float32") or embedding.get("dtype") != "float32":
        raise ValueError(
            "Arrow conversion embeddings must be float32 in both file and manifest"
        )
    if int(manifest.get("n_rows", -1)) != len(values):
        raise ValueError("Arrow conversion row count does not match embedding array")
    payload_digest = _float32_payload_digest(values)
    if payload_digest != embedding.get("float32_payload_sha256"):
        raise ValueError("Arrow conversion embedding payload SHA-256 does not match")

    metadata = _read_metadata(metadata_path)
    key_column = manifest.get("cell_key_column")
    if not isinstance(key_column, str) or key_column not in metadata:
        raise ValueError("Converted metadata lacks the manifest cell-key column")
    if len(metadata) != len(values):
        raise ValueError("Converted metadata and embedding row counts differ")
    if metadata[key_column].isna().any() or metadata[key_column].duplicated().any():
        raise ValueError("Converted metadata cell keys must be unique and nonnull")
    if "embedding_row" not in metadata or not np.array_equal(
        pd.to_numeric(metadata["embedding_row"], errors="coerce").to_numpy(),
        np.arange(len(metadata)),
    ):
        raise ValueError("Converted metadata embedding_row order is invalid")
    key_digest = _ordered_key_digest(metadata[key_column].astype(str))
    if key_digest != manifest.get("cell_key_ordered_sha256"):
        raise ValueError("Converted metadata ordered cell-key digest does not match")
    if key_digest != projection_validation.get("cell_key_ordered_sha256"):
        raise ValueError("Converted cell order differs from frozen projection output")
    if len(values) != projection_validation.get("n_cells"):
        raise ValueError("Converted row count differs from frozen projection output")

    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "embedding_column": embedding_column,
        "embedding_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "n_rows": len(values),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "float32_payload_sha256": payload_digest,
        "metadata_sha256": manifest["metadata_sha256"],
        "cell_key_ordered_sha256": key_digest,
        "alignment_method": "manifest-bound ordered cell_key digest",
        "projection_output": projection_validation,
    }


def convert_tripso_arrow_embeddings(
    arrow_dataset: Path,
    cell_metadata_path: Path,
    output_dir: Path,
    *,
    projection_output_manifest: Path,
    embedding_columns: Sequence[str],
    cell_key_column: str = "cell_key",
    required_metadata_columns: Sequence[str] = DEFAULT_REQUIRED_METADATA,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert requested Arrow vector columns after a one-to-one key join."""

    columns = tuple(dict.fromkeys(map(str, embedding_columns)))
    if not columns:
        raise ValueError("At least one explicit embedding column is required")
    projection_validation = validate_projection_output_for_conversion(
        projection_output_manifest,
        arrow_dataset,
        embedding_columns=columns,
    )
    if cell_key_column.endswith("_id"):
        raise ValueError(
            "The Arrow row key must not end in '_id'; TRIPSO casts such metadata "
            "to integer tensors. Use the safe name 'cell_key'."
        )
    if cell_key_column in columns:
        raise ValueError("cell_key cannot also be an embedding column")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "arrow_conversion_manifest.json"
    metadata_output = output_dir / "cell_metadata.parquet"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Arrow conversion is already complete: {manifest_path}")
    output_paths = {
        column: output_dir / f"{_output_stem(column)}.npy" for column in columns
    }
    if not overwrite:
        existing = [
            path for path in (metadata_output, *output_paths.values()) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite conversion outputs: {existing}"
            )

    files = _arrow_files(arrow_dataset)
    requested = (cell_key_column, *columns)
    key_parts: list[np.ndarray] = []
    dimensions: dict[str, int] = {}
    n_rows = 0
    for batch in _record_batches(files, requested):
        keys = np.asarray(batch.column(cell_key_column).to_pylist(), dtype=object)
        if any(value is None or not str(value).strip() for value in keys):
            raise ValueError("Arrow cell_key contains null or empty values")
        key_parts.append(keys.astype(str))
        n_rows += len(batch)
        for column in columns:
            matrix = _array_to_matrix(batch.column(column), column)
            observed = int(matrix.shape[1])
            previous = dimensions.setdefault(column, observed)
            if previous != observed:
                raise ValueError(
                    f"Embedding dimension changed for {column!r}: "
                    f"{previous} versus {observed}"
                )
    if n_rows == 0:
        raise ValueError("Arrow dataset contains no rows")
    arrow_keys = np.concatenate(key_parts)
    if pd.Index(arrow_keys).duplicated().any():
        duplicated = pd.Index(arrow_keys)[pd.Index(arrow_keys).duplicated()][0]
        raise ValueError(f"Arrow cell_key is not unique; example {duplicated!r}")

    metadata = _read_metadata(Path(cell_metadata_path))
    required = {cell_key_column, *required_metadata_columns}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"External cell metadata lacks columns: {missing}")
    if (
        metadata[cell_key_column].isna().any()
        or metadata[cell_key_column].duplicated().any()
    ):
        raise ValueError(
            "External cell metadata must have unique, nonnull cell_key values"
        )
    order = pd.DataFrame(
        {cell_key_column: arrow_keys.astype(str), "embedding_row": np.arange(n_rows)}
    )
    metadata = metadata.copy()
    metadata[cell_key_column] = metadata[cell_key_column].astype(str)
    aligned = order.merge(
        metadata,
        on=cell_key_column,
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    missing_metadata = aligned["_merge"].ne("both")
    if missing_metadata.any():
        examples = aligned.loc[missing_metadata, cell_key_column].head().tolist()
        raise ValueError(f"Arrow cells are absent from external metadata: {examples}")
    aligned = aligned.drop(columns="_merge").sort_values("embedding_row")
    if not np.array_equal(aligned[cell_key_column].to_numpy(), arrow_keys):
        raise AssertionError("Key join changed Arrow row order")
    if any(aligned[column].isna().any() for column in required_metadata_columns):
        bad = [
            column
            for column in required_metadata_columns
            if aligned[column].isna().any()
        ]
        raise ValueError(f"Required metadata columns contain missing values: {bad}")
    key_digest = _ordered_key_digest(arrow_keys)
    if n_rows != projection_validation["n_cells"]:
        raise ValueError("Arrow row count differs from projection output manifest")
    if key_digest != projection_validation["cell_key_ordered_sha256"]:
        raise ValueError("Arrow cell order differs from projection output manifest")
    observed_datasets = sorted(aligned["dataset"].astype(str).unique().tolist())
    if observed_datasets != sorted(projection_validation["datasets"]):
        raise ValueError("Converted metadata datasets differ from projection output")
    observed_lineages = sorted(aligned["lineage"].astype(str).unique().tolist())
    if observed_lineages != [str(projection_validation["lineage"])]:
        raise ValueError("Converted metadata lineage differs from projection output")
    observed_units = sorted(
        (
            aligned["dataset"].astype(str) + "::" + aligned["donor_id"].astype(str)
        ).unique()
    )
    if observed_units != projection_validation["biological_unit_ids"]:
        raise ValueError(
            "Converted metadata biological-unit scope differs from projection output"
        )
    metadata_temporary = output_dir / ".cell_metadata.partial.parquet"
    temporary_paths = {
        column: output_dir / f".{_output_stem(column)}.partial.npy"
        for column in columns
    }
    for temporary in (metadata_temporary, *temporary_paths.values()):
        if temporary.exists():
            temporary.unlink()
    aligned.to_parquet(metadata_temporary, index=False)
    arrays = {
        column: np.lib.format.open_memmap(
            temporary_paths[column],
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, dimensions[column]),
        )
        for column in columns
    }
    payload_hashes = {column: hashlib.sha256() for column in columns}
    try:
        cursor = 0
        for batch in _record_batches(files, requested):
            observed_keys = np.asarray(
                batch.column(cell_key_column).to_pylist(), dtype=str
            )
            expected_keys = arrow_keys[cursor : cursor + len(batch)]
            if not np.array_equal(observed_keys, expected_keys):
                raise RuntimeError(
                    "Arrow source row order changed between conversion passes"
                )
            for column in columns:
                matrix = _array_to_matrix(batch.column(column), column)
                arrays[column][cursor : cursor + len(batch)] = matrix
                payload_hashes[column].update(matrix.tobytes(order="C"))
            cursor += len(batch)
        if cursor != n_rows:
            raise AssertionError("Arrow conversion wrote an incomplete row count")
        for array in arrays.values():
            array.flush()
        arrays.clear()
        for column in columns:
            os.replace(temporary_paths[column], output_paths[column])
        os.replace(metadata_temporary, metadata_output)
    except BaseException:
        arrays.clear()
        for temporary in (metadata_temporary, *temporary_paths.values()):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise

    payload: dict[str, object] = {
        "schema_version": ARROW_BRIDGE_SCHEMA,
        "arrow_dataset": str(Path(arrow_dataset).resolve()),
        "arrow_files": [
            {"path": str(path), "size_bytes": path.stat().st_size} for path in files
        ],
        "source_cell_metadata": str(Path(cell_metadata_path).resolve()),
        "cell_key_column": cell_key_column,
        "cell_key_ordered_sha256": key_digest,
        "n_rows": n_rows,
        "metadata_output": metadata_output.name,
        "metadata_sha256": sha256_file(metadata_output),
        "required_metadata_columns": list(required_metadata_columns),
        "embedding_outputs": {
            column: {
                "path": output_paths[column].name,
                "shape": [n_rows, dimensions[column]],
                "dtype": "float32",
                "float32_payload_sha256": payload_hashes[column].hexdigest(),
            }
            for column in columns
        },
        "alignment_validation": {
            "method": "one-to-one cell_key join preserving Arrow order",
            "passed": True,
            "row_count_only_alignment": False,
        },
        "projection_output": projection_validation,
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return payload
