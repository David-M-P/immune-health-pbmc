"""Frozen all-cell reference/validation/query projection with state guards."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .contracts import (
    REQUIRED_VENDOR_ASSETS,
    TripsoContractError,
    atomic_write_json,
    canonical_json_hash,
    load_fold_input_manifest,
    sha256_path,
)
from .geneformer import (
    geneformer_runtime_compatibility,
    resolve_geneformer_root,
    validate_geneformer_root,
)
from .provenance import validate_checkpoint_manifest
from .tokenization import (
    PROJECTION_INPUT_SCHEMA,
    load_tokenization_manifest,
    validate_physical_tokenization_scope,
)

PROJECTION_OUTPUT_SCHEMA = "immune-health-tripso-projection-output/v1"


class FrozenProjectionError(RuntimeError):
    """Raised if a projection requests or causes model adaptation."""


def _value_bytes(value: Any) -> bytes:
    detached = value
    if hasattr(detached, "detach"):
        detached = detached.detach()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    if hasattr(detached, "contiguous"):
        detached = detached.contiguous()
    if hasattr(detached, "numpy"):
        array = detached.numpy()
        return (
            str(getattr(array, "dtype", "")).encode()
            + repr(getattr(array, "shape", None)).encode()
            + array.tobytes()
        )
    try:
        return json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
    except TypeError:
        return repr(value).encode("utf-8")


def model_state_hash(model: Any) -> str:
    """Hash all named parameters and buffers exposed by ``state_dict``."""
    if not hasattr(model, "state_dict"):
        raise FrozenProjectionError("Frozen projection model must expose state_dict()")
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_value_bytes(state[name]))
        digest.update(b"\0")
    return digest.hexdigest()


def _parameters(model: Any) -> list[Any]:
    if not hasattr(model, "parameters"):
        return []
    return list(model.parameters())


@contextmanager
def frozen_projection_guard(model: Any) -> Iterator[str]:
    """Set evaluation/frozen mode and prove that state is unchanged afterward."""
    before = model_state_hash(model)
    parameters = _parameters(model)
    prior_requires_grad = [
        getattr(parameter, "requires_grad", None) for parameter in parameters
    ]
    if hasattr(model, "eval"):
        model.eval()
    for parameter in parameters:
        if hasattr(parameter, "requires_grad_"):
            parameter.requires_grad_(False)
        elif hasattr(parameter, "requires_grad"):
            parameter.requires_grad = False
    try:
        yield before
    finally:
        after = model_state_hash(model)
        # Keep the model frozen after query scoring; do not restore train mode.
        for parameter, prior in zip(parameters, prior_requires_grad):
            if (
                prior is not None
                and getattr(parameter, "requires_grad", None) is not False
            ):
                raise FrozenProjectionError(
                    "A model parameter re-enabled gradients during projection"
                )
        if after != before:
            raise FrozenProjectionError(
                "Model state (parameters or buffers) changed during frozen projection"
            )


def _inference_context() -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return nullcontext()
    return torch.inference_mode()


def project_frozen(
    model: Any,
    batches: Iterable[Any],
    *,
    forward: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
    adapt: bool = False,
    fit: bool = False,
) -> list[Any]:
    """Project batches without gradients, optimizer steps, or state updates."""
    if optimizer is not None:
        raise FrozenProjectionError("An optimizer is forbidden during projection")
    if adapt or fit:
        raise FrozenProjectionError("Projection cannot adapt or fit the model")
    operation = forward or (lambda active_model, batch: active_model(batch))
    outputs: list[Any] = []
    with frozen_projection_guard(model):
        with _inference_context():
            for batch in batches:
                outputs.append(operation(model, batch))
    return outputs


def validate_frozen_query_resources(
    *,
    model_manifest_path: Path,
    query_manifest: Mapping[str, Any],
    vendor_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a model-bound, physical, inference-only projection input.

    The historical function name remains for compatibility.  It now validates
    reference, fixed inner-validation, and outer-query roles and never trusts
    donor declarations without re-reading the Arrow dataset.
    """

    if query_manifest.get("schema_version") != PROJECTION_INPUT_SCHEMA:
        raise TripsoContractError("Unsupported frozen projection-input schema")
    claimed_manifest_hash = query_manifest.get("manifest_sha256")
    payload = dict(query_manifest)
    payload.pop("manifest_sha256", None)
    if claimed_manifest_hash != canonical_json_hash(payload):
        raise TripsoContractError("Projection-input manifest hash does not match")
    role = query_manifest.get("projection_role")
    if role not in {"query", "reference", "validation"}:
        raise TripsoContractError(
            "Projection role must be reference, validation, or query"
        )
    if query_manifest.get("adapt") is not False:
        raise TripsoContractError("Projection manifest must set adapt=false")
    if query_manifest.get("optimizer_allowed") is not False:
        raise TripsoContractError("Projection manifest must forbid an optimizer")
    if query_manifest.get("all_tokenized_cells_required") is not True:
        raise TripsoContractError("Projection manifest must require every cell")

    model_manifest = validate_checkpoint_manifest(model_manifest_path)
    declared_model_path = Path(str(query_manifest.get("model_manifest", "")))
    if declared_model_path.resolve() != Path(model_manifest_path).resolve():
        raise TripsoContractError(
            "Projection input is bound to a different model manifest"
        )
    if query_manifest.get("hashes", {}).get("model_manifest_sha256") != sha256_path(
        Path(model_manifest_path)
    ):
        raise TripsoContractError("Model manifest changed after projection binding")

    fold_path = Path(model_manifest["paths"]["fold_input_manifest"])
    if sha256_path(fold_path) != model_manifest["hashes"].get("input_manifest_sha256"):
        raise TripsoContractError("Model-bound fold manifest changed after training")
    fold = load_fold_input_manifest(fold_path)
    if canonical_json_hash(fold) != model_manifest.get(
        "fold_input_manifest_content_hash"
    ):
        raise TripsoContractError("Model fold content differs from its provenance")
    if model_manifest.get("inner_model_selection", {}) != fold.get(
        "inner_model_selection", {}
    ):
        raise TripsoContractError(
            "Model inner-selection provenance differs from its bound fold"
        )

    tokenization_path = Path(str(query_manifest.get("tokenization_manifest", "")))
    if query_manifest.get("hashes", {}).get(
        "tokenization_manifest_sha256"
    ) != sha256_path(tokenization_path):
        raise TripsoContractError(
            "Tokenization manifest changed after projection binding"
        )
    tokenization = load_tokenization_manifest(tokenization_path)
    for field in (
        "tokenized_dataset_path",
        "gp_library_path",
        "gene_vocabulary_path",
    ):
        if (
            Path(str(query_manifest.get(field, ""))).resolve()
            != Path(str(tokenization.get(field, ""))).resolve()
        ):
            raise TripsoContractError(
                f"Projection {field} differs from its bound tokenization manifest"
            )
    physical = validate_physical_tokenization_scope(tokenization)
    if (
        list(query_manifest.get("biological_unit_ids", []))
        != physical["biological_unit_ids"]
    ):
        raise TripsoContractError(
            "Projection donor declaration differs from physical Arrow donors"
        )
    if int(query_manifest.get("n_cells", -1)) != physical["n_cells"]:
        raise TripsoContractError(
            "Projection cell count differs from the physical Arrow dataset"
        )
    if (
        query_manifest.get("cell_key_ordered_sha256")
        != physical["cell_key_ordered_sha256"]
    ):
        raise TripsoContractError(
            "Projection cell keys differ from the physical Arrow dataset"
        )

    training_tokenization_path = fold.get("inputs", {}).get(
        "tokenization_manifest_path"
    )
    if not training_tokenization_path:
        raise TripsoContractError(
            "The trained fold lacks a bound training tokenization manifest"
        )
    training_tokenization_path = Path(training_tokenization_path)
    training_tokenization = load_tokenization_manifest(
        training_tokenization_path, verify_paths=False
    )
    required_hashes = (
        "gp_library_sha256",
        "gene_vocabulary_sha256",
        "tokenizer_contract_sha256",
        "token_dictionary_sha256",
        "median_dictionary_sha256",
    )
    for name in required_hashes:
        if name in {"gp_library_sha256", "gene_vocabulary_sha256"}:
            expected = model_manifest["hashes"].get(name)
        elif name == "tokenizer_contract_sha256":
            expected = training_tokenization.get(name)
        else:
            expected = training_tokenization.get("hashes", {}).get(name)
        observed = query_manifest.get("hashes", {}).get(name)
        if not observed or observed != expected:
            raise TripsoContractError(
                f"Frozen projection resource mismatch for {name}: expected {expected}, "
                f"observed {observed}"
            )
        token_observed = (
            tokenization.get(name)
            if name == "tokenizer_contract_sha256"
            else tokenization.get("hashes", {}).get(name)
        )
        if token_observed != observed:
            raise TripsoContractError(
                f"Physical tokenization resource mismatch for {name}"
            )

    actual_resource_hashes = {
        "gp_library_sha256": sha256_path(Path(query_manifest["gp_library_path"])),
        "gene_vocabulary_sha256": sha256_path(
            Path(query_manifest["gene_vocabulary_path"])
        ),
    }
    for name, actual in actual_resource_hashes.items():
        if actual != query_manifest["hashes"][name]:
            raise TripsoContractError(
                f"Frozen projection resource file changed for {name}"
            )

    physical_donors = set(map(str, physical["biological_unit_ids"]))
    adaptation = set(map(str, fold["adaptation_biological_unit_ids"]))
    validation = set(map(str, fold["validation_biological_unit_ids"]))
    outer_query = set(map(str, fold["query_biological_unit_ids"]))
    if role == "reference":
        if physical_donors != adaptation:
            raise TripsoContractError(
                "Reference projection donors are not exactly adaptation donors"
            )
        if tokenization.get("role") != "adaptation":
            raise TripsoContractError(
                "Reference projection does not use adaptation tokenization"
            )
        if tokenization_path.resolve() != training_tokenization_path.resolve():
            raise TripsoContractError(
                "Reference projection is not the exact training tokenization"
            )
        if sha256_path(tokenization_path) != fold.get("hashes", {}).get(
            "tokenization_manifest_sha256"
        ):
            raise TripsoContractError(
                "Reference tokenization differs from the fold-bound training input"
            )
    elif role == "validation":
        if physical_donors != validation:
            raise TripsoContractError(
                "Validation projection donors are not exactly the fixed inner-fold "
                "validation donors"
            )
        if tokenization.get("role") != "validation":
            raise TripsoContractError(
                "Validation projection does not use validation tokenization"
            )
        if not validation:
            raise TripsoContractError(
                "The trained fold does not declare inner-validation donors"
            )
    else:
        if tokenization.get("role") != "query":
            raise TripsoContractError(
                "Query projection does not use query tokenization"
            )
        overlap = sorted(physical_donors & (adaptation | validation))
        if overlap:
            raise TripsoContractError(
                "Query projection overlaps adaptation/validation donors: "
                f"{overlap[:10]}"
            )
        if fold.get("reference_design") == "lodo" and physical_donors != outer_query:
            raise TripsoContractError(
                "Outer-query projection donors are not exactly the held-out fold donors"
            )

    if set(map(str, physical["lineages"])) != {str(model_manifest["lineage"])}:
        raise TripsoContractError(
            "Projection lineage differs from the trained model lineage"
        )
    if query_manifest.get("lineage") != model_manifest["lineage"]:
        raise TripsoContractError("Projection manifest lineage differs from model")
    gp_projection = query_manifest.get("gp_projection")
    if not isinstance(gp_projection, Mapping):
        raise TripsoContractError("Projection manifest lacks GP storage selection")
    program_ids = gp_projection.get("program_ids")
    if (
        not isinstance(program_ids, list)
        or not program_ids
        or len(program_ids) != len(set(map(str, program_ids)))
    ):
        raise TripsoContractError(
            "Projection GP program_ids must be a nonempty unique list"
        )
    program_ids = list(map(str, program_ids))
    if gp_projection.get("program_ids_ordered_sha256") != canonical_json_hash(
        program_ids
    ):
        raise TripsoContractError("Projection GP ordered-program hash is invalid")
    with Path(query_manifest["gp_library_path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        available_programs = next(csv.reader(handle), [])
    if not available_programs:
        raise TripsoContractError("Projection GP library has no columns")
    missing_programs = [
        program for program in program_ids if program not in set(available_programs)
    ]
    if missing_programs:
        raise TripsoContractError(
            "Projection requests GP columns absent from the library: "
            f"{missing_programs[:10]}"
        )
    selection_mode = gp_projection.get("mode")
    if selection_mode == "frozen_training_candidates":
        allowlist_path = Path(str(gp_projection.get("allowlist_path", "")))
        if gp_projection.get("allowlist_sha256") != sha256_path(allowlist_path):
            raise TripsoContractError("Frozen GP projection allowlist changed")
        expected_candidate_path = fold.get("inputs", {}).get(
            "projection_gp_candidates_path"
        )
        expected_candidate_hash = fold.get("hashes", {}).get(
            "projection_gp_candidates_sha256"
        )
        expected_program_hash = fold.get("hashes", {}).get(
            "projection_gp_program_ids_ordered_sha256"
        )
        model_candidate_path = model_manifest.get("paths", {}).get(
            "projection_gp_candidates"
        )
        model_candidate_hash = model_manifest.get("hashes", {}).get(
            "projection_gp_candidates_sha256"
        )
        model_program_hash = model_manifest.get("hashes", {}).get(
            "projection_gp_program_ids_ordered_sha256"
        )
        if (
            not expected_candidate_path
            or allowlist_path.resolve() != Path(expected_candidate_path).resolve()
            or gp_projection.get("allowlist_sha256") != expected_candidate_hash
            or gp_projection.get("program_ids_ordered_sha256") != expected_program_hash
        ):
            raise TripsoContractError(
                "Projection GP candidates differ from the model-bound fold"
            )
        if (
            not model_candidate_path
            or Path(str(model_candidate_path)).resolve()
            != Path(str(expected_candidate_path)).resolve()
            or model_candidate_hash != expected_candidate_hash
            or model_program_hash != expected_program_hash
        ):
            raise TripsoContractError(
                "Model artifact lacks the exact fold-bound GP candidate identity"
            )
        for label, source in (
            ("training", training_tokenization),
            (role, tokenization),
        ):
            binding = source.get("projection_gp_candidates")
            if not isinstance(binding, Mapping) or (
                binding.get("sha256") != expected_candidate_hash
                or binding.get("program_ids_ordered_sha256") != expected_program_hash
            ):
                raise TripsoContractError(
                    f"{label.title()} tokenization has a different GP candidate binding"
                )
    elif selection_mode == "all_gps_bounded_diagnostic":
        if program_ids != available_programs:
            raise TripsoContractError(
                "All-GP diagnostic must retain every GP-library column in order"
            )
    else:
        raise TripsoContractError("Projection GP selection mode is invalid")
    if int(gp_projection.get("n_programs", -1)) != len(program_ids):
        raise TripsoContractError("Projection GP program count is invalid")
    embedding_dimension = int(gp_projection.get("embedding_dimension", 0))
    estimated_bytes = physical["n_cells"] * len(program_ids) * embedding_dimension * 4
    if estimated_bytes != int(gp_projection.get("estimated_gp_vector_bytes", -1)):
        raise TripsoContractError("Projection byte estimate is inconsistent")
    maximum_bytes = int(gp_projection.get("maximum_projected_bytes", 0))
    if maximum_bytes < 1:
        raise TripsoContractError("Projection maximum byte guard is invalid")
    if estimated_bytes > maximum_bytes and not gp_projection.get(
        "oversized_projection_override", False
    ):
        raise TripsoContractError("Projection exceeds its frozen maximum byte guard")
    if gp_projection.get("include_cell_token") and model_manifest.get(
        "model_type"
    ) not in {"Global", "Global_LoRA"}:
        raise TripsoContractError(
            "cell_token projection is available only for Global model types"
        )
    expected_config = model_manifest.get("model_configuration", {})
    if query_manifest.get("model_configuration") != expected_config:
        raise TripsoContractError("Frozen projection changes model configuration")
    for name in (
        "tokenizer",
        "preprocessing",
        "embedding_dimension",
        "model_type",
    ):
        if (
            name in expected_config
            and query_manifest.get("model_configuration", {}).get(name)
            != expected_config[name]
        ):
            raise TripsoContractError(f"Frozen query changes model setting {name!r}")

    if vendor_root is not None:
        for relative_name in REQUIRED_VENDOR_ASSETS:
            key = f"vendor_asset:{relative_name}"
            expected = model_manifest.get("hashes", {}).get(key)
            if expected is None:
                raise TripsoContractError(
                    "Model artifact lacks required vendored asset hash: "
                    f"{relative_name}"
                )
            observed = sha256_path(Path(vendor_root) / relative_name)
            if observed != expected:
                raise TripsoContractError(
                    f"Vendored projection asset changed: {relative_name}"
                )
    return model_manifest


def make_all_cell_projection_datamodule_class(vendor_datamodule_class: type) -> type:
    """Expose every projection row and preserve string-valued ``*_id`` metadata.

    The inspected vendor collator assumes every metadata column ending in ``_id``
    is an integer class encoding.  The project identifier contract deliberately
    uses collision-safe strings, so the local subclass temporarily removes those
    fields from the vendor conversion and restores them as Python strings.
    """

    class AllCellProjectionDataModule(  # type: ignore[misc, valid-type]
        vendor_datamodule_class
    ):
        _immune_health_all_cell_projection = True

        def setup(self, stage: str | None = None) -> None:
            super().setup(stage=stage)
            self.test_dataset = self.dataset
            gdata = self.dataset.tk_dataset.gdata
            metadata_columns = tuple(getattr(self, "metadata", ()))
            preserved: set[str] = set()
            if len(gdata):
                first = gdata[0]
                preserved.update(
                    column
                    for column in metadata_columns
                    if column.endswith("_id") and isinstance(first.get(column), str)
                )
            self._immune_health_preserved_string_ids = tuple(sorted(preserved))

        def custom_collate(self, batch: list[Mapping[str, Any]]) -> Any:
            preserved = getattr(self, "_immune_health_preserved_string_ids", ())
            original_metadata = self.metadata
            self.metadata = [
                column for column in original_metadata if column not in preserved
            ]
            try:
                output = super().custom_collate(batch)
            finally:
                self.metadata = original_metadata
            if getattr(self, "return_tuple", False):
                return output
            for column in preserved:
                output[column] = [item["tk"][column] for item in batch]
            return output

    AllCellProjectionDataModule.__name__ = "AllCellProjectionDataModule"
    AllCellProjectionDataModule.__qualname__ = "AllCellProjectionDataModule"
    return AllCellProjectionDataModule


def make_query_only_datamodule_class(vendor_datamodule_class: type) -> type:
    """Backward-compatible alias for the role-neutral all-cell datamodule."""

    return make_all_cell_projection_datamodule_class(vendor_datamodule_class)


@contextmanager
def _selected_projection_columns(
    lightning_module: Any,
    *,
    program_ids: Iterable[str],
    metadata_columns: Iterable[str],
    include_cell_token: bool,
    include_gene_encoder_cls: bool,
) -> Iterator[dict[str, Any]]:
    """Filter each vendor batch before it enters the accumulated Arrow dataset.

    The inspected vendor test step creates one Hugging Face ``Dataset`` per batch
    and then concatenates it. Temporarily replacing only that function's local
    ``Dataset.from_dict`` surface prevents unselected GP vectors from ever entering
    the CPU accumulator or the saved Arrow shards. Vendored source is unchanged.
    """

    test_step = getattr(lightning_module, "test_step", None)
    function = getattr(test_step, "__func__", test_step)
    globals_dict = getattr(function, "__globals__", None)
    if not isinstance(globals_dict, dict) or "Dataset" not in globals_dict:
        raise RuntimeError(
            "Cannot install selected-GP projection filter; the inspected vendor "
            "test_step no longer exposes its Hugging Face Dataset surface"
        )
    original_dataset = globals_dict["Dataset"]
    selected = tuple(map(str, program_ids))
    metadata = tuple(dict.fromkeys(map(str, metadata_columns)))
    endpoint_columns = tuple(
        name
        for name, include in (
            ("gene_encoder_cls", include_gene_encoder_cls),
            ("cell_token", include_cell_token),
        )
        if include
    )
    audit: dict[str, Any] = {"batches_filtered": 0, "column_order": None}

    class SelectedDataset:
        @staticmethod
        def from_dict(values: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            missing_programs = [name for name in selected if name not in values]
            if missing_programs:
                raise TripsoContractError(
                    "Trained model output lacks requested GP columns: "
                    f"{missing_programs[:10]}"
                )
            missing_endpoints = [
                name for name in endpoint_columns if name not in values
            ]
            if missing_endpoints:
                raise TripsoContractError(
                    "Trained model output lacks requested endpoint columns: "
                    f"{missing_endpoints}"
                )
            kept_order = [*selected, *endpoint_columns]
            kept_order.extend(name for name in metadata if name in values)
            filtered = {name: values[name] for name in kept_order}
            if audit["column_order"] is None:
                audit["column_order"] = kept_order
            elif audit["column_order"] != kept_order:
                raise TripsoContractError(
                    "Projected Arrow column inventory changed between batches"
                )
            audit["batches_filtered"] = int(audit["batches_filtered"]) + 1
            return original_dataset.from_dict(filtered, *args, **kwargs)

    globals_dict["Dataset"] = SelectedDataset
    try:
        yield audit
    finally:
        globals_dict["Dataset"] = original_dataset


def _hash_projection_tree(path: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        records.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_path(file_path),
            }
        )
    if not records:
        raise RuntimeError(f"Projected Arrow directory contains no files: {path}")
    return records, canonical_json_hash(records)


def _validate_projected_arrow_scope(
    arrow_path: Path,
    *,
    projection_manifest: Mapping[str, Any],
    expected_columns: Iterable[str],
) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise RuntimeError("Hugging Face datasets is required") from exc
    dataset = load_from_disk(str(arrow_path))
    column_order = list(map(str, dataset.column_names))
    expected_order = list(map(str, expected_columns))
    if column_order != expected_order:
        raise TripsoContractError(
            "Projected Arrow columns differ from the selected inventory: "
            f"expected={expected_order}, observed={column_order}"
        )
    if len(dataset) != int(projection_manifest["n_cells"]):
        raise TripsoContractError(
            "Projected Arrow row count differs from the all-cell input"
        )
    required = {
        "cell_key",
        "dataset",
        "biological_unit_id",
        "observation_id",
        "fine_type",
        "lineage",
    }
    missing = sorted(required - set(column_order))
    if missing:
        raise TripsoContractError(
            f"Projected Arrow lacks required biological metadata: {missing}"
        )
    digest = hashlib.sha256()
    donors: set[str] = set()
    datasets: set[str] = set()
    lineages: set[str] = set()
    selected = dataset.select_columns(
        ["cell_key", "dataset", "biological_unit_id", "lineage"]
    )
    for batch in selected.iter(batch_size=100_000):
        for key in batch["cell_key"]:
            encoded = str(key).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        donors.update(map(str, batch["biological_unit_id"]))
        datasets.update(map(str, batch["dataset"]))
        lineages.update(map(str, batch["lineage"]))
    key_hash = digest.hexdigest()
    if key_hash != projection_manifest["cell_key_ordered_sha256"]:
        raise TripsoContractError(
            "Projected Arrow cell keys/order differ from the frozen input"
        )
    donor_order = sorted(donors)
    if donor_order != list(projection_manifest["biological_unit_ids"]):
        raise TripsoContractError(
            "Projected Arrow donor scope differs from the frozen input"
        )
    return {
        "n_cells": len(dataset),
        "column_order": column_order,
        "cell_key_ordered_sha256": key_hash,
        "biological_unit_ids": donor_order,
        "biological_unit_ids_sha256": canonical_json_hash(donor_order),
        "datasets": sorted(datasets),
        "lineages": sorted(lineages),
        "huggingface_fingerprint": getattr(dataset, "_fingerprint", None),
    }


@contextmanager
def _geneformer_projection_compatibility(
    model_manifest: Mapping[str, Any],
    tripso_module: Any,
) -> Iterator[Mapping[str, Any] | None]:
    """Install the reviewed full-Geneformer fixes while a checkpoint is loaded.

    Loading a Lightning checkpoint reconstructs the vendored model before any
    query batch is evaluated.  Full-Geneformer checkpoints therefore need the
    same temporary path and forward-signature compatibility used during training.
    The primary static-embedding ``from_scratch`` model does not need this context.
    """

    model_configuration = model_manifest.get("model_configuration", {})
    if not isinstance(model_configuration, Mapping):
        raise TripsoContractError("Model manifest has invalid model_configuration")
    vendor_call = model_configuration.get("vendor_call", {})
    if not isinstance(vendor_call, Mapping):
        raise TripsoContractError(
            "Model manifest has invalid model_configuration.vendor_call"
        )
    if vendor_call.get("fm_encoder_pkg") != "geneformer":
        yield None
        return

    model_name = vendor_call.get("fm_encoder_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TripsoContractError(
            "Full-Geneformer projection requires fm_encoder_name in the model "
            "manifest's vendor_call"
        )
    geneformer_root = resolve_geneformer_root(None)
    validation = validate_geneformer_root(
        geneformer_root,
        model_name=model_name,
    )
    recorded_identity = model_configuration.get("geneformer_identity")
    if not isinstance(recorded_identity, Mapping):
        raise TripsoContractError(
            "Full-Geneformer model manifest lacks its pinned model identity"
        )
    expected_identity = {
        "model_name": validation["model_name"],
        "source_revision": validation["source_revision"],
        "config": validation["config"],
        "hashes": validation["hashes"],
        "hashes_pinned": validation["hashes_pinned"],
    }
    observed_identity = {
        name: recorded_identity.get(name) for name in expected_identity
    }
    if observed_identity != expected_identity:
        raise TripsoContractError(
            "Full-Geneformer assets differ from the identity recorded at training"
        )
    train_fn = getattr(tripso_module, "train", None)
    if not callable(train_fn):
        raise RuntimeError(
            "Cannot install full-Geneformer projection compatibility because the "
            "inspected tripso.train entry point is unavailable"
        )
    with geneformer_runtime_compatibility(
        train_fn,
        geneformer_root=geneformer_root,
    ):
        yield validation


def run_vendor_frozen_projection(
    *,
    model_manifest_path: Path,
    query_manifest: Mapping[str, Any],
    output_dir: Path,
    batch_size: int = 128,
    precision: int | str = 32,
    vendor_root: Path | None = None,
    projection_manifest_path: Path | None = None,
) -> Path:
    """Project every tokenized reference, validation, or query cell.

    The upstream public helper projects an 80/10/10 split. This adapter subclasses
    its datamodule locally and assigns the complete query dataset to the test loader.
    Lightning's ``test`` loop does not configure or step an optimizer.
    """
    model_manifest = validate_frozen_query_resources(
        model_manifest_path=model_manifest_path,
        query_manifest=query_manifest,
        vendor_root=vendor_root,
    )
    # The validator guarantees this field.  The default keeps narrow test doubles
    # and callers that monkeypatch validation backward compatible.
    role = str(query_manifest.get("projection_role", "query"))
    try:
        import pytorch_lightning as pl  # type: ignore
        import tripso  # type: ignore
        from tripso.Datamodules.datamodule import txDataModule  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Real TRIPSO projection dependencies are unavailable; run the environment "
            "validator. No mock output was substituted."
        ) from exc

    checkpoint = Path(model_manifest["paths"]["checkpoint"])
    model_type = model_manifest["model_type"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_audit: dict[str, Any] = {}
    frozen_state_sha256: str | None = None
    with _geneformer_projection_compatibility(model_manifest, tripso):
        AllCellProjectionDataModule = make_all_cell_projection_datamodule_class(
            txDataModule
        )
        evaluator = tripso.gpEval(
            dataset_path=str(query_manifest["tokenized_dataset_path"]),
            gpdb_path=str(query_manifest["gp_library_path"]),
            output_dir=str(output_dir),
            model_type=model_type,
            path_to_trained_model=str(checkpoint.parent.parent),
            seed=int(query_manifest.get("seed", model_manifest["seed"])),
            batch_size=batch_size,
        )
        lightning_module = evaluator._init_trainer(  # inspected vendored API
            save_emb=True,
            split_label=role,
            hparam_save=evaluator.hparam_save,
        )
        data_module = AllCellProjectionDataModule(
            folder=str(query_manifest["tokenized_dataset_path"]),
            batch_size=batch_size,
            data_split_to_pass_to_test_step=role,
            seed=int(query_manifest.get("seed", model_manifest["seed"])),
            fm_encoder_name=evaluator.fm_encoder_name,
            model_input_size=evaluator.max_len,
        )
        trainer = pl.Trainer(
            max_epochs=1,
            devices=1,
            accelerator="auto",
            precision=precision,
            logger=False,
            enable_checkpointing=False,
        )
        gp_projection = query_manifest["gp_projection"]
        with _selected_projection_columns(
            lightning_module,
            program_ids=gp_projection["program_ids"],
            metadata_columns=query_manifest["projection_metadata_columns"],
            include_cell_token=bool(gp_projection["include_cell_token"]),
            include_gene_encoder_cls=bool(gp_projection["include_gene_encoder_cls"]),
        ) as filter_audit:
            with frozen_projection_guard(lightning_module) as frozen_state_sha256:
                with _inference_context():
                    trainer.test(lightning_module, data_module)
    embeddings_path = output_dir / "embeddings" / f"{role}_set"
    if not embeddings_path.is_dir():
        raise RuntimeError(
            "TRIPSO projection finished without its chunked Arrow dataset: "
            f"{embeddings_path}"
        )
    column_order = filter_audit.get("column_order")
    if not isinstance(column_order, list) or not filter_audit.get("batches_filtered"):
        raise RuntimeError("Projection completed without filtered Arrow batches")
    arrow_scope = _validate_projected_arrow_scope(
        embeddings_path,
        projection_manifest=query_manifest,
        expected_columns=column_order,
    )
    shard_records, arrow_tree_sha256 = _hash_projection_tree(embeddings_path)
    fold = load_fold_input_manifest(
        Path(model_manifest["paths"]["fold_input_manifest"])
    )
    projection_input_file_sha256 = (
        sha256_path(projection_manifest_path)
        if projection_manifest_path is not None
        else None
    )
    output_manifest: dict[str, Any] = {
        "schema_version": PROJECTION_OUTPUT_SCHEMA,
        "projection_role": role,
        "eligible_for_model_selection": role == "validation",
        "outer_query_evaluation_only": role == "query",
        "inner_model_selection": dict(fold.get("inner_model_selection", {})),
        "reference_design": fold["reference_design"],
        "heldout_dataset": fold["held_out_dataset"],
        "fold_id": fold["fold_id"],
        "lineage": model_manifest["lineage"],
        "model_type": model_manifest["model_type"],
        "seed": int(query_manifest["seed"]),
        "adapt": False,
        "optimizer_used": False,
        "all_tokenized_cells_projected": True,
        "projection_input_manifest": (
            str(Path(projection_manifest_path).resolve())
            if projection_manifest_path is not None
            else None
        ),
        "model_manifest": str(Path(model_manifest_path).resolve()),
        "fold_input_manifest": str(
            Path(model_manifest["paths"]["fold_input_manifest"]).resolve()
        ),
        "arrow_dataset": f"embeddings/{role}_set",
        "n_cells": arrow_scope["n_cells"],
        "datasets": arrow_scope["datasets"],
        "biological_unit_ids": arrow_scope["biological_unit_ids"],
        "biological_unit_ids_sha256": arrow_scope["biological_unit_ids_sha256"],
        "cell_key_ordered_sha256": arrow_scope["cell_key_ordered_sha256"],
        "lineages": arrow_scope["lineages"],
        "gp_projection": dict(query_manifest["gp_projection"]),
        "embedding_dimension": int(
            query_manifest["gp_projection"]["embedding_dimension"]
        ),
        "endpoint_columns": [
            name for name in ("gene_encoder_cls", "cell_token") if name in column_order
        ],
        "expected_column_inventory_order": column_order,
        "metadata_columns": [
            name
            for name in column_order
            if name not in set(query_manifest["gp_projection"]["program_ids"])
            and name not in {"gene_encoder_cls", "cell_token"}
        ],
        "arrow_files": shard_records,
        "hashes": {
            "arrow_tree_sha256": arrow_tree_sha256,
            "model_manifest_sha256": sha256_path(model_manifest_path),
            "checkpoint_sha256": model_manifest["hashes"]["checkpoint_sha256"],
            "fold_input_manifest_sha256": model_manifest["hashes"][
                "input_manifest_sha256"
            ],
            "projection_input_manifest_sha256": projection_input_file_sha256,
            "projection_input_content_sha256": query_manifest["manifest_sha256"],
            "gp_library_sha256": query_manifest["hashes"]["gp_library_sha256"],
            "gene_vocabulary_sha256": query_manifest["hashes"][
                "gene_vocabulary_sha256"
            ],
            "gp_program_ids_ordered_sha256": query_manifest["gp_projection"][
                "program_ids_ordered_sha256"
            ],
            "model_state_before_and_after_sha256": frozen_state_sha256,
        },
        "huggingface_fingerprint": arrow_scope["huggingface_fingerprint"],
        "vendor_batch_filter": {
            "installed_without_vendor_source_edits": True,
            "batches_filtered": int(filter_audit["batches_filtered"]),
            "unselected_gp_vectors_persisted": False,
        },
    }
    output_manifest["manifest_sha256"] = canonical_json_hash(output_manifest)
    atomic_write_json(output_dir / "projection_output_manifest.json", output_manifest)
    return embeddings_path


def run_mock_projection_smoke() -> dict[str, Any]:
    """Exercise adapter guards without claiming a real TRIPSO smoke test."""

    class Parameter:
        def __init__(self, value: int) -> None:
            self.value = value
            self.requires_grad = True

        def requires_grad_(self, value: bool) -> "Parameter":
            self.requires_grad = value
            return self

    class Model:
        def __init__(self) -> None:
            self.weight = Parameter(3)
            self.training = True

        def state_dict(self) -> dict[str, int]:
            return {"weight": self.weight.value}

        def parameters(self) -> list[Parameter]:
            return [self.weight]

        def eval(self) -> "Model":
            self.training = False
            return self

        def __call__(self, batch: int) -> int:
            return self.weight.value * batch

    model = Model()
    before = model_state_hash(model)
    outputs = project_frozen(model, [1, 2, 3])
    after = model_state_hash(model)
    return {
        "mock_adapter_smoke_passed": outputs == [3, 6, 9] and before == after,
        "real_tripso_import_tested": False,
        "real_tripso_training_smoke_passed": False,
        "note": (
            "Mock smoke validates adapter guards only; it is not TRIPSO validation."
        ),
    }
