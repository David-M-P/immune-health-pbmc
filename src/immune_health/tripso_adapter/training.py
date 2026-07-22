"""Conservative wrappers around the inspected vendored ``tripso.train`` API."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .contracts import (
    TripsoContractError,
    atomic_write_json,
    load_fold_input_manifest,
)
from .dynamic_sampling import (
    DEFAULT_STRING_ID_COLUMNS,
    DynamicSamplerConfig,
    make_dynamic_datamodule_class,
    make_identifier_safe_datamodule_class,
    normalize_dynamic_sampler_config,
)
from .geneformer import (
    VALIDATED_GENEFORMER_REVISION,
    geneformer_runtime_compatibility,
    resolve_geneformer_root,
    validate_geneformer_root,
)
from .local_logging import (
    collect_local_training_metrics,
    local_csv_logging_context,
    local_tracking_plan,
)

FORBIDDEN_BOOLEAN_OPTIONS = {
    "adapt_query",
    "adapt_held_out",
    "fit_on_query",
    "random_cell_split_for_biological_evaluation",
    "use_query_for_feature_selection",
    "use_query_for_gp_filtering",
    "use_query_for_model_selection",
}

PROJECT_SAMPLER_PARAMETER = "project_sampler"
ALL_GENES_FROM_VOCABULARY_PARAMETER = "all_genes_from_fold_vocabulary"


@dataclass(frozen=True)
class TripsoTrainingSpec:
    """One base/global invocation tied to a validated donor fold."""

    fold_input_manifest_path: Path
    output_dir: Path
    model_type: str
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False


def _validate_vendor_signature(train_fn: Callable[..., Any]) -> None:
    signature = inspect.signature(train_fn)
    required = {"dataset_path", "gpdb_path", "output_dir", "model_type", "seed"}
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    if not required.issubset(signature.parameters):
        missing = sorted(required - set(signature.parameters))
        raise RuntimeError(
            "The vendored tripso.train API no longer matches the inspected adapter; "
            f"missing parameters: {missing}"
        )


def _load_fold_gene_vocabulary(path: Path) -> list[str]:
    """Load the fold-bound ordered vocabulary without guessing gene identities."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not all(
            isinstance(item, str) and item.strip() for item in payload
        ):
            raise TripsoContractError(
                "JSON gene vocabulary must be a non-empty string list"
            )
        genes = [item.strip() for item in payload]
    else:
        genes = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                # A fold vocabulary is intentionally a one-column resource.  A
                # delimiter is accepted only to tolerate a conventional header.
                genes.append(value.split("\t", 1)[0].split(",", 1)[0].strip())
        if genes and genes[0].lower() in {
            "gene",
            "genes",
            "gene_id",
            "ensembl_id",
            "feature_id",
        }:
            genes = genes[1:]
    if not genes:
        raise TripsoContractError(f"Fold gene vocabulary is empty: {path}")
    duplicates = sorted(gene for gene, count in Counter(genes).items() if count > 1)
    if duplicates:
        raise TripsoContractError(
            f"Fold gene vocabulary contains duplicate identifiers: {duplicates[:10]}"
        )
    return genes


def _materialize_fold_vocabulary(
    call: dict[str, Any], inputs: Mapping[str, Any]
) -> None:
    requested = call.pop(ALL_GENES_FROM_VOCABULARY_PARAMETER, False)
    if requested not in {True, False, None}:
        raise TripsoContractError(
            f"{ALL_GENES_FROM_VOCABULARY_PARAMETER} must be boolean"
        )
    if not requested:
        return
    if "all_genes" in call:
        raise TripsoContractError(
            "Specify either all_genes or all_genes_from_fold_vocabulary, not both"
        )
    genes = _load_fold_gene_vocabulary(Path(inputs["gene_vocabulary_path"]))
    call["all_genes"] = genes
    bert = dict(call.get("bert_config") or {})
    configured_length = bert.get("max_seq_len")
    if configured_length is not None and int(configured_length) != len(genes):
        raise TripsoContractError(
            "bert_config.max_seq_len differs from the fold gene vocabulary: "
            f"{configured_length} != {len(genes)}"
        )
    bert["max_seq_len"] = len(genes)
    call["bert_config"] = bert


def _sequence_and_identifier_contract(call: Mapping[str, Any]) -> dict[str, Any]:
    genes = call.get("all_genes")
    gene_list = list(genes) if isinstance(genes, (list, tuple)) else []
    ensembl_count = sum(
        bool(re.fullmatch(r"ENSG\d+(?:\.\d+)?", str(gene))) for gene in gene_list
    )
    obvious_ensembl = bool(gene_list) and ensembl_count / len(gene_list) >= 0.8
    gene_format = str(call.get("gene_format", "symbol"))
    if obvious_ensembl and gene_format != "ensembl":
        raise TripsoContractError(
            "Fold vocabulary is predominantly ENSG identifiers, but "
            f"gene_format={gene_format!r}. Use gene_format='ensembl' so TRIPSO "
            "does not attempt symbol-to-Ensembl conversion on Ensembl IDs."
        )

    bert = call.get("bert_config") or {}
    per_cell_limit = (
        int(bert["tokenization_input_size"])
        if call.get("fm_encoder_pkg") == "from_scratch"
        and bert.get("tokenization_input_size") is not None
        else None
    )
    if (
        per_cell_limit is None
        and call.get("fm_encoder_pkg") == "geneformer"
        and call.get("fm_encoder_name") == "gf-12L-95M-i4096"
    ):
        per_cell_limit = 4096
    n_genes = len(gene_list) if gene_list else None
    exceeds = bool(
        n_genes is not None and per_cell_limit is not None and n_genes > per_cell_limit
    )
    return {
        "gene_identifier_format": gene_format,
        "n_fold_gene_universe": n_genes,
        "n_ensembl_like_fold_genes": ensembl_count if gene_list else None,
        "per_cell_rank_token_limit": per_cell_limit,
        "gene_encoder_max_seq_len": bert.get("max_seq_len"),
        "fold_gene_universe_exceeds_per_cell_limit": exceeds,
        "per_cell_truncation_expected": exceeds,
        "interpretation": (
            "Each cell retains at most the top-ranked expressed genes; the fold "
            "gene universe may be larger and defines eligible/modelled genes."
            if per_cell_limit is not None
            else "Sequence length is read from the validated full-model config."
        ),
    }


def _validate_training_parameters(parameters: Mapping[str, Any]) -> None:
    split_unit = str(parameters.get("biological_split_unit", "donor")).lower()
    if split_unit != "donor":
        raise TripsoContractError(
            f"Biological split unit must be donor, not {split_unit!r}"
        )
    for name in FORBIDDEN_BOOLEAN_OPTIONS:
        if parameters.get(name) not in {None, False}:
            raise TripsoContractError(
                f"Forbidden TRIPSO training option enabled: {name}"
            )
    if str(parameters.get("split_unit", "donor")).lower() == "cell":
        raise TripsoContractError("Random cell-level biological splits are forbidden")
    if parameters.get("sampler") == "weighted":
        raise TripsoContractError(
            "The vendor weighted cell sampler is not donor-hierarchical. Prepare a "
            "fold-specific hierarchical sampling manifest and pass sampler=None."
        )
    raw_sampler = parameters.get(PROJECT_SAMPLER_PARAMETER)
    if raw_sampler is not None and not isinstance(raw_sampler, Mapping):
        raise TripsoContractError("project_sampler must be a JSON mapping")
    if (
        isinstance(raw_sampler, Mapping)
        and raw_sampler.get("enabled", True)
        and parameters.get("sampler") is not None
    ):
        raise TripsoContractError(
            "project_sampler cannot be combined with a vendor sampler; set sampler=null"
        )


def build_training_call(
    spec: TripsoTrainingSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve and validate one vendor training call without importing TRIPSO."""
    fold = load_fold_input_manifest(spec.fold_input_manifest_path)
    _validate_training_parameters(spec.parameters)
    requested_model_type = spec.model_type.strip()
    if requested_model_type not in {"Base", "Global", "Global_LoRA"}:
        raise TripsoContractError(
            f"Unsupported trainable TRIPSO model_type {requested_model_type!r}"
        )

    inputs = fold["inputs"]
    call = dict(spec.parameters)
    raw_project_sampler = call.pop(PROJECT_SAMPLER_PARAMETER, None)
    _materialize_fold_vocabulary(call, inputs)
    geneformer_root_value = call.pop("geneformer_root", None)
    geneformer_expected_hashes = call.pop("geneformer_expected_hashes", None)
    geneformer_revision = call.pop("geneformer_revision", None)
    for adapter_only in (
        "biological_split_unit",
        "split_unit",
        *FORBIDDEN_BOOLEAN_OPTIONS,
    ):
        call.pop(adapter_only, None)
    protected = {
        "dataset_path": inputs["tokenized_dataset_path"],
        "gpdb_path": inputs["gp_library_path"],
        "output_dir": str(Path(spec.output_dir).resolve()),
        "model_type": requested_model_type,
        "seed": int(spec.seed),
    }
    conflicting = {
        key: (call[key], value)
        for key, value in protected.items()
        if key in call and call[key] != value
    }
    if conflicting:
        raise TripsoContractError(
            f"Training parameters cannot override fold-bound values: {conflicting}"
        )
    call.update(protected)
    sequence_contract = _sequence_and_identifier_contract(call)

    project_sampler: DynamicSamplerConfig | None = None
    if raw_project_sampler is not None:
        assert isinstance(raw_project_sampler, Mapping)
        project_sampler = normalize_dynamic_sampler_config(
            raw_project_sampler,
            batch_size=int(call.get("batch_size", 32)),
            seed=int(spec.seed),
            lineage=str(fold["lineage"]),
        )

    effective_encoder_package = str(call.get("fm_encoder_pkg", "geneformer"))
    geneformer_validation: dict[str, Any] | None = None
    if effective_encoder_package == "geneformer":
        if geneformer_revision not in {None, VALIDATED_GENEFORMER_REVISION}:
            raise TripsoContractError(
                "geneformer_revision must match the reviewed historical revision "
                f"{VALIDATED_GENEFORMER_REVISION}"
            )
        configured_root = geneformer_root_value
        if configured_root in {None, ""}:
            configured_root = None
        if configured_root is not None or os.environ.get("TRIPSO_GENEFORMER_ROOT"):
            root = resolve_geneformer_root(configured_root)
            if geneformer_expected_hashes is not None and not isinstance(
                geneformer_expected_hashes, Mapping
            ):
                raise TripsoContractError(
                    "geneformer_expected_hashes must be a JSON mapping"
                )
            geneformer_validation = validate_geneformer_root(
                root,
                model_name=str(call.get("fm_encoder_name", "gf-6L-30M-i2048")),
                expected_hashes=geneformer_expected_hashes,
            )
            geneformer_validation["requested_source_revision"] = (
                geneformer_revision or VALIDATED_GENEFORMER_REVISION
            )
            # The frozen full-model wrapper returns contextual embeddings but no
            # gene-MLM logits. Vendor defaults would therefore KeyError in
            # compute_gene_loss, and a nonzero gene-only warmup would have no
            # differentiable loss. Bind the only reviewed Base-compatible setup.
            call.setdefault("calc_gene_loss", False)
            call.setdefault("calc_gp_loss", True)
            call.setdefault("warmup", 0)
            if call["calc_gene_loss"] is not False:
                raise TripsoContractError(
                    "Full Geneformer requires calc_gene_loss=false because its "
                    "frozen wrapper does not return gene-MLM logits"
                )
            if requested_model_type == "Base" and call["calc_gp_loss"] is not True:
                raise TripsoContractError(
                    "Full Geneformer Base training requires calc_gp_loss=true"
                )
            if call["warmup"] != 0:
                raise TripsoContractError(
                    "Full Geneformer requires warmup=0; gene-only warmup has no "
                    "trainable loss for the frozen wrapper"
                )
        else:
            geneformer_validation = {
                "passed": False,
                "required_before_real_training": True,
                "model_name": str(call.get("fm_encoder_name", "gf-6L-30M-i2048")),
                "configuration": (
                    "Set adapter parameter geneformer_root or environment variable "
                    "TRIPSO_GENEFORMER_ROOT"
                ),
                "source_revision": (
                    geneformer_revision or VALIDATED_GENEFORMER_REVISION
                ),
            }

    if requested_model_type in {"Global", "Global_LoRA"}:
        base_model = call.get("path_to_base_model")
        if not base_model:
            raise TripsoContractError("Global training requires path_to_base_model")
        base_checkpoint = Path(base_model) / "checkpoints" / "last.ckpt"
        if not base_checkpoint.is_file():
            raise FileNotFoundError(
                "Validated Base checkpoint is required before Global training: "
                f"{base_checkpoint}"
            )

    invocation = {
        "schema_version": "immune-health-tripso-training-call/v1",
        "fold_input_manifest": str(Path(spec.fold_input_manifest_path).resolve()),
        "fold_id": fold["fold_id"],
        "reference_design": fold.get("reference_design", "lodo"),
        "held_out_dataset": fold["held_out_dataset"],
        "lineage": fold["lineage"],
        "adaptation_biological_unit_ids": fold["adaptation_biological_unit_ids"],
        "biological_evaluation_split": "donor",
        "vendor_internal_cell_split_scope": "adaptation_donors_only",
        "tokenized_dataset_scope_validation": fold.get(
            "tokenized_dataset_scope_validation",
            {"status": "not_performed", "required_before_real_training": True},
        ),
        "project_sampler": (
            project_sampler.manifest() if project_sampler is not None else None
        ),
        "training_sampler_backend": (
            "immune_health.sampling.HierarchicalCellSampler"
            if project_sampler is not None and project_sampler.enabled
            else "vendor"
        ),
        "metadata_collation_backend": (
            "immune_health.tripso_adapter.IdentifierSafeDataModule"
        ),
        "vendor_optimizer_split_preserved": not (
            project_sampler is not None and project_sampler.enabled
        ),
        "experiment_tracking": local_tracking_plan(spec.output_dir),
        "geneformer_validation": geneformer_validation,
        "sequence_and_identifier_contract": sequence_contract,
        "vendor_call": call,
    }
    return call, invocation


@contextmanager
def _dynamic_datamodule_context(
    train_fn: Callable[..., Any], config: DynamicSamplerConfig | None
) -> Iterator[None]:
    globals_dict = getattr(train_fn, "__globals__", None)
    if not isinstance(globals_dict, dict) or "txDataModule" not in globals_dict:
        raise RuntimeError(
            "Cannot inject identifier-safe data loading: the TRIPSO train function "
            "no longer exposes the inspected txDataModule global"
        )
    original = globals_dict["txDataModule"]
    if config is not None and config.enabled:
        replacement = make_dynamic_datamodule_class(original, config)
    else:
        preserve = (
            config.preserve_string_id_columns
            if config is not None
            else DEFAULT_STRING_ID_COLUMNS
        )
        replacement = make_identifier_safe_datamodule_class(original, preserve)
    globals_dict["txDataModule"] = replacement
    try:
        yield
    finally:
        globals_dict["txDataModule"] = original


def run_tripso_training(
    spec: TripsoTrainingSpec,
    *,
    train_fn: Callable[..., Any] | None = None,
) -> Any:
    """Call TRIPSO only after writing a donor-safe, auditable invocation."""
    call, invocation = build_training_call(spec)
    if (
        not spec.dry_run
        and invocation["tokenized_dataset_scope_validation"].get("status") != "passed"
    ):
        raise TripsoContractError(
            "Real TRIPSO training requires donor IDs extracted from the physical "
            "tokenized dataset to prove it contains adaptation donors only"
        )
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_path = output_dir / "tripso_training_invocation.json"
    atomic_write_json(invocation_path, invocation)
    if spec.dry_run:
        return invocation

    if train_fn is None:
        try:
            import tripso  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TRIPSO import failed. Run scripts/validate_tripso_environment.py; "
                "the training call was not started."
            ) from exc
        train_fn = tripso.train
    _validate_vendor_signature(train_fn)

    project_sampler: DynamicSamplerConfig | None = None
    raw_project_sampler = spec.parameters.get(PROJECT_SAMPLER_PARAMETER)
    if raw_project_sampler is not None:
        assert isinstance(raw_project_sampler, Mapping)
        project_sampler = normalize_dynamic_sampler_config(
            raw_project_sampler,
            batch_size=int(call.get("batch_size", 32)),
            seed=int(spec.seed),
            lineage=str(invocation["lineage"]),
        )

    effective_encoder_package = str(call.get("fm_encoder_pkg", "geneformer"))
    local_logging_runtime: dict[str, Any] = {}
    with ExitStack() as contexts:
        contexts.enter_context(_dynamic_datamodule_context(train_fn, project_sampler))
        local_logging_runtime = contexts.enter_context(
            local_csv_logging_context(train_fn, output_dir=output_dir)
        )
        if effective_encoder_package == "geneformer":
            validation = invocation.get("geneformer_validation") or {}
            if not validation.get("passed"):
                # Resolve again to produce a direct, actionable error when a dry
                # build was created before assets were transferred.
                root = resolve_geneformer_root(spec.parameters.get("geneformer_root"))
                validation = validate_geneformer_root(
                    root,
                    model_name=str(call.get("fm_encoder_name", "gf-6L-30M-i2048")),
                    expected_hashes=spec.parameters.get("geneformer_expected_hashes"),
                )
            contexts.enter_context(
                geneformer_runtime_compatibility(
                    train_fn,
                    geneformer_root=Path(validation["root"]),
                )
            )
        result = train_fn(**call)

    checkpoint = output_dir / "checkpoints" / "last.ckpt"
    local_metrics = collect_local_training_metrics(output_dir)
    completion: dict[str, Any] = {
        **invocation,
        "experiment_tracking": {
            **invocation["experiment_tracking"],
            **local_logging_runtime,
            "metrics": local_metrics,
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.is_file() and checkpoint.stat().st_size > 0,
    }
    atomic_write_json(output_dir / "tripso_training_result.json", completion)
    if not completion["checkpoint_exists"]:
        raise RuntimeError(
            "TRIPSO returned without a non-empty checkpoints/last.ckpt artifact"
        )
    return result


def train_base_model(
    fold_input_manifest_path: Path,
    output_dir: Path,
    *,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
    train_fn: Callable[..., Any] | None = None,
    dry_run: bool = False,
) -> Any:
    """Train the inspected vendor Base model on adaptation donors only."""
    return run_tripso_training(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_input_manifest_path,
            output_dir=output_dir,
            model_type="Base",
            seed=seed,
            parameters=parameters or {},
            dry_run=dry_run,
        ),
        train_fn=train_fn,
    )


def train_global_model(
    fold_input_manifest_path: Path,
    output_dir: Path,
    base_model_dir: Path,
    *,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
    train_fn: Callable[..., Any] | None = None,
    dry_run: bool = False,
) -> Any:
    """Train the inspected vendor Global model from a validated Base model."""
    resolved_parameters = dict(parameters or {})
    resolved_parameters["path_to_base_model"] = str(Path(base_model_dir).resolve())
    resolved_parameters.setdefault("global_training", "sequential")
    return run_tripso_training(
        TripsoTrainingSpec(
            fold_input_manifest_path=fold_input_manifest_path,
            output_dir=output_dir,
            model_type="Global",
            seed=seed,
            parameters=resolved_parameters,
            dry_run=dry_run,
        ),
        train_fn=train_fn,
    )
