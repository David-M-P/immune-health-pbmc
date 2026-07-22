"""Validation and runtime compatibility for an external full Geneformer model."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .contracts import TripsoContractError, sha256_path

VALIDATED_GENEFORMER_MODEL = "gf-12L-95M-i4096"
VALIDATED_GENEFORMER_REVISION = "94d98d1774fc21920d2f01dbd23bc22d232a6ec2"
EXPECTED_GENEFORMER_CONFIG = {
    "vocab_size": 20275,
    "hidden_size": 512,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "max_position_embeddings": 4096,
}
EXPECTED_GENEFORMER_HASHES = {
    "config.json": "f56780389d8c89c1b6c4084e2e6ee1f736558e4b3bb8ce7473159e83465de401",
    "model.safetensors": (
        "4365ba23e393fcfa0e65a94ac64a0983cd788bd23a8d4914f4ab66f85cfe043c"
    ),
}
GENEFORMER_ROOT_ENV = "TRIPSO_GENEFORMER_ROOT"


def resolve_geneformer_root(value: str | os.PathLike[str] | None) -> Path:
    """Resolve an explicit root, falling back to a cluster-safe environment var."""

    raw = value if value not in {None, ""} else os.environ.get(GENEFORMER_ROOT_ENV)
    if raw in {None, ""}:
        raise TripsoContractError(
            "Full Geneformer mode requires adapter parameter 'geneformer_root' or "
            f"environment variable {GENEFORMER_ROOT_ENV}."
        )
    return Path(str(raw)).expanduser().resolve()


def validate_geneformer_root(
    root: Path,
    *,
    model_name: str = VALIDATED_GENEFORMER_MODEL,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the exact model layout consumed by the vendored TRIPSO code."""

    root = Path(root).expanduser().resolve()
    if model_name != VALIDATED_GENEFORMER_MODEL:
        raise TripsoContractError(
            f"Only {VALIDATED_GENEFORMER_MODEL!r} has a validated full-Geneformer "
            f"contract; received {model_name!r}."
        )
    model_dir = root / model_name
    config_path = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    missing = []
    if not root.is_dir():
        missing.append(str(root))
    if not config_path.is_file() or config_path.stat().st_size == 0:
        missing.append(str(config_path))
    if not weights.is_file() or weights.stat().st_size == 0:
        missing.append(str(weights))
    if missing:
        raise FileNotFoundError(
            "Full Geneformer assets are missing/non-empty-file validation failed: "
            + ", ".join(missing)
        )

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in EXPECTED_GENEFORMER_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise TripsoContractError(
            f"Unexpected {model_name} config; refusing an unreviewed architecture: "
            f"{mismatches}"
        )
    observed_hashes = {
        "config.json": sha256_path(config_path),
        weights.name: sha256_path(weights),
    }
    pinned_hashes = dict(EXPECTED_GENEFORMER_HASHES)
    for name, expected in (expected_hashes or {}).items():
        if name in pinned_hashes and expected != pinned_hashes[name]:
            raise TripsoContractError(
                f"Configured Geneformer hash for {name} conflicts with the reviewed "
                f"revision {VALIDATED_GENEFORMER_REVISION}"
            )
        pinned_hashes[name] = expected
    for name, expected in pinned_hashes.items():
        if name not in observed_hashes:
            raise TripsoContractError(
                f"Unknown Geneformer expected-hash asset {name!r}; observed "
                f"{sorted(observed_hashes)}"
            )
        if observed_hashes[name] != expected:
            raise TripsoContractError(
                f"Geneformer hash mismatch for {name}: expected {expected}, "
                f"observed {observed_hashes[name]}"
            )
    return {
        "passed": True,
        "root": str(root),
        "model_name": model_name,
        "model_directory": str(model_dir),
        "config": {key: config[key] for key in EXPECTED_GENEFORMER_CONFIG},
        "weights_file": str(weights),
        "hashes": observed_hashes,
        "hashes_pinned": True,
        "source_revision": VALIDATED_GENEFORMER_REVISION,
    }


def validate_static_embedding_alignment(
    *,
    geneformer_validation: Mapping[str, Any],
    static_embedding_path: Path,
) -> dict[str, Any]:
    """Compare full-model input embeddings with TRIPSO's May-2025 tensor.

    This deliberately runs only in the environment validator: it loads roughly
    two 40-MB tensors and is unnecessary on every training launch.
    """

    weights_path = Path(str(geneformer_validation["weights_file"]))
    if weights_path.name != "model.safetensors":
        return {
            "passed": False,
            "checked": False,
            "reason": "Exact alignment check requires model.safetensors",
        }
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment-specific dependency
        return {
            "passed": False,
            "checked": False,
            "reason": f"Cannot import torch/safetensors: {type(exc).__name__}: {exc}",
        }
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            key = "bert.embeddings.word_embeddings.weight"
            if key not in handle.keys():
                return {
                    "passed": False,
                    "checked": True,
                    "reason": f"Full model lacks tensor {key!r}",
                }
            full = handle.get_tensor(key)
        try:
            static = torch.load(
                static_embedding_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # older torch
            static = torch.load(static_embedding_path, map_location="cpu")
        same = full.shape == static.shape and torch.equal(full, static)
        return {
            "passed": bool(same),
            "checked": True,
            "full_shape": list(full.shape),
            "static_shape": list(static.shape),
            "full_dtype": str(full.dtype),
            "static_dtype": str(static.dtype),
            "exact_tensor_equality": bool(same),
            "static_embedding_sha256": sha256_path(static_embedding_path),
        }
    except Exception as exc:  # pragma: no cover - reports corrupt real assets
        return {
            "passed": False,
            "checked": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }


@contextmanager
def geneformer_runtime_compatibility(
    train_fn: Callable[..., Any],
    *,
    geneformer_root: Path,
) -> Iterator[None]:
    """Temporarily repair two non-portable assumptions in vendored TRIPSO.

    TRIPSO imports ``get_gf_repo`` into multiple module globals and currently
    hard-codes the original author's NFS path.  Its Base forward also passes a
    keyword supported by the from-scratch wrapper but absent from ``gfWrapper``.
    Both are patched only for the duration of the call and restored afterward;
    the vendored source tree remains byte-for-byte unchanged.
    """

    root_text = str(Path(geneformer_root).resolve())

    def configured_root() -> str:
        return root_text

    globals_dict = getattr(train_fn, "__globals__", None)
    if not isinstance(globals_dict, dict) or "get_gf_repo" not in globals_dict:
        raise RuntimeError(
            "Cannot install Geneformer path compatibility: train function no "
            "longer exposes the inspected module globals"
        )

    patched: list[tuple[Any, str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def patch(container: Any, name: str, value: Any) -> None:
        identity = (id(container), name)
        if identity in seen:
            return
        if isinstance(container, dict):
            if name not in container:
                return
            old = container[name]
            container[name] = value
        else:
            if not hasattr(container, name):
                return
            old = getattr(container, name)
            setattr(container, name, value)
        patched.append((container, name, old))
        seen.add(identity)

    patch(globals_dict, "get_gf_repo", configured_root)
    module_names = {
        getattr(train_fn, "__module__", ""),
        "tripso.Utils.geneformer_utils",
        "tripso.Models.gp_model",
        "tripso.Preprocessing.preprocess",
    }
    for name in sorted(module_names):
        if not name:
            continue
        try:
            module = sys.modules.get(name) or importlib.import_module(name)
        except ImportError:
            continue
        patch(module, "get_gf_repo", configured_root)
        if name.endswith("Preprocessing.preprocess"):
            patch(module, "geneformer_repo_path", root_text)

    try:
        base_class = globals_dict.get("gpTransformerBase")
        gp_module = (
            importlib.import_module(base_class.__module__)
            if base_class is not None
            else sys.modules.get("tripso.Models.gp_model")
        )
        wrapper_class = getattr(gp_module, "gfWrapper", None)
        if wrapper_class is None:
            raise RuntimeError("Vendored gfWrapper class could not be located")
        original_forward = wrapper_class.forward
        signature = inspect.signature(original_forward)
        accepts_kwargs = any(
            value.kind is inspect.Parameter.VAR_KEYWORD
            for value in signature.parameters.values()
        )
        if "return_mean_non_padding" not in signature.parameters and not accepts_kwargs:

            @functools.wraps(original_forward)
            def compatible_forward(
                instance: Any,
                input_dataset: Any,
                masking: Any,
                *args: Any,
                return_mean_non_padding: bool = False,
                **kwargs: Any,
            ) -> Any:
                del return_mean_non_padding
                return original_forward(
                    instance, input_dataset, masking, *args, **kwargs
                )

            patch(wrapper_class, "forward", compatible_forward)
        yield
    finally:
        for container, name, old in reversed(patched):
            if isinstance(container, dict):
                container[name] = old
            else:
                setattr(container, name, old)
