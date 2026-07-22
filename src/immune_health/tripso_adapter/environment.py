"""Validation of the real vendored TRIPSO runtime and immutable assets."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import platform
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .contracts import REQUIRED_VENDOR_ASSETS, sha256_path
from .geneformer import (
    VALIDATED_GENEFORMER_MODEL,
    validate_geneformer_root,
    validate_static_embedding_alignment,
)
from .projection import run_mock_projection_smoke

DIST_ALIASES = {
    "requests": "requests",
    "scikit_learn": "scikit-learn",
    "pytorch_lightning": "pytorch-lightning",
}

REQUIRED_IMPORTS_BEYOND_REQUIREMENTS = (
    "torch",
    "torchmetrics",
    "torchvision",
)


def _requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    stripped = stripped.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)", stripped)
    return match.group(1) if match else None


def _requirement_pin(line: str) -> str | None:
    if "==" not in line:
        return None
    return line.split("==", 1)[1].split(";", 1)[0].strip()


def validate_dependencies(requirements_path: Path) -> list[dict[str, Any]]:
    """Compare installed distributions with every uncommented vendor requirement."""
    results: list[dict[str, Any]] = []
    with Path(requirements_path).open(encoding="utf-8") as handle:
        lines = list(handle)
    for line in lines:
        name = _requirement_name(line)
        if name is None:
            continue
        distribution = DIST_ALIASES.get(name.lower(), name.replace("_", "-"))
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        expected = _requirement_pin(line)
        results.append(
            {
                "requirement": line.strip(),
                "distribution": distribution,
                "installed_version": installed,
                "present": installed is not None,
                "pin_matches": expected is None or installed == expected,
            }
        )
    for name in REQUIRED_IMPORTS_BEYOND_REQUIREMENTS:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        results.append(
            {
                "requirement": f"{name} (imported by TRIPSO; vendor pin commented out)",
                "distribution": name,
                "installed_version": installed,
                "present": installed is not None,
                "pin_matches": None,
            }
        )
    return results


@contextmanager
def _vendor_on_path(vendor_root: Path) -> Iterator[None]:
    vendor_text = str(Path(vendor_root).resolve())
    inserted = vendor_text not in sys.path
    if inserted:
        sys.path.insert(0, vendor_text)
    try:
        yield
    finally:
        if inserted and vendor_text in sys.path:
            sys.path.remove(vendor_text)


def validate_tripso_import(vendor_root: Path) -> dict[str, Any]:
    """Actually import TRIPSO and verify the inspected public/private surfaces."""
    for name in list(sys.modules):
        if name == "tripso" or name.startswith("tripso."):
            del sys.modules[name]
    try:
        with _vendor_on_path(vendor_root):
            tripso = importlib.import_module("tripso")
            datamodule = importlib.import_module("tripso.Datamodules.datamodule")
            gp_model = importlib.import_module("tripso.Models.gp_model")
            training_module = importlib.import_module("tripso.Train.training")
            train_parameters = set(inspect.signature(tripso.train).parameters)
            eval_parameters = set(inspect.signature(tripso.gpEval).parameters)
            gf_forward_parameters = set(
                inspect.signature(gp_model.gfWrapper.forward).parameters
            )
            required_train = {
                "dataset_path",
                "gpdb_path",
                "output_dir",
                "model_type",
                "seed",
            }
            required_eval = {
                "dataset_path",
                "gpdb_path",
                "output_dir",
                "path_to_trained_model",
            }
            tracking_globals = {
                "configure_save_id",
                "configure_wandb",
                "configure_logger",
                "rank_zero_only",
            }
            tracking_surface_ok = (
                tracking_globals <= tripso.train.__globals__.keys()
                and hasattr(getattr(training_module.pl, "loggers", None), "CSVLogger")
            )
            api_ok = (
                required_train <= train_parameters
                and required_eval <= eval_parameters
                and hasattr(datamodule, "txDataModule")
                and hasattr(gp_model, "gfWrapper")
                and "txDataModule" in tripso.train.__globals__
                and "get_gf_repo" in tripso.train.__globals__
                and tracking_surface_ok
            )
            return {
                "passed": api_ok,
                "module_path": str(Path(tripso.__file__).resolve()),
                "train_parameters": sorted(train_parameters),
                "gp_eval_parameters": sorted(eval_parameters),
                "full_geneformer_forward_parameters": sorted(gf_forward_parameters),
                "full_geneformer_forward_adapter_required": (
                    "return_mean_non_padding" not in gf_forward_parameters
                ),
                "vendor_geneformer_root": training_module.get_gf_repo(),
                "geneformer_root_adapter_required": True,
                "local_csv_tracking_adapter": {
                    "required_vendor_globals": sorted(tracking_globals),
                    "surface_passed": tracking_surface_ok,
                    "network_required": False,
                },
                "query_adapter_private_surface": "gpEval._init_trainer",
                "error": None
                if api_ok
                else "Imported TRIPSO API differs from the inspected adapter contract",
            }
    except BaseException as exc:
        return {
            "passed": False,
            "module_path": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-12000:],
        }


def validate_vendor_assets(vendor_root: Path) -> list[dict[str, Any]]:
    """Verify all assets used by the May-2025/4096-token vendor configuration."""
    results = []
    for relative_name in REQUIRED_VENDOR_ASSETS:
        path = Path(vendor_root) / relative_name
        present = path.is_file() and path.stat().st_size > 0
        results.append(
            {
                "relative_path": relative_name,
                "path": str(path.resolve()),
                "present_nonempty": present,
                "size_bytes": path.stat().st_size if present else None,
                "sha256": sha256_path(path) if present else None,
            }
        )
    return results


def run_real_smoke_command(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    """Run an explicitly supplied no-shell real smoke driver and capture its result."""
    if not command:
        raise ValueError("A real smoke command must contain at least one argument")
    completed = subprocess.run(
        list(command),
        cwd=Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "offline")},
    )
    return {
        "mode": "real-command",
        "command": list(command),
        "returncode": completed.returncode,
        "real_tripso_training_smoke_passed": completed.returncode == 0,
        "output": completed.stdout[-20000:],
    }


def validate_environment(
    *,
    vendor_root: Path,
    smoke_mode: str = "none",
    real_smoke_command: Sequence[str] | None = None,
    real_smoke_cwd: Path | None = None,
    geneformer_root: Path | None = None,
    geneformer_model_name: str = VALIDATED_GENEFORMER_MODEL,
    geneformer_expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable Linux/Python-3.10/TRIPSO validation report."""
    vendor_root = Path(vendor_root).resolve()
    requirements = vendor_root / "requirements.txt"
    dependency_results = (
        validate_dependencies(requirements) if requirements.is_file() else []
    )
    assets = validate_vendor_assets(vendor_root) if vendor_root.is_dir() else []
    platform_check = platform.system() == "Linux"
    python_check = sys.version_info[:2] == (3, 10)
    dependency_check = bool(dependency_results) and all(
        result["present"] and result["pin_matches"] is not False
        for result in dependency_results
    )
    asset_check = bool(assets) and all(item["present_nonempty"] for item in assets)
    import_result = validate_tripso_import(vendor_root)

    if geneformer_root is None:
        geneformer: dict[str, Any] = {
            "passed": True,
            "requested": False,
            "note": (
                "Full Geneformer was not requested. The primary from_scratch "
                "TRIPSO configuration uses the vendored static embedding tensor."
            ),
        }
    else:
        try:
            geneformer = {
                "requested": True,
                **validate_geneformer_root(
                    geneformer_root,
                    model_name=geneformer_model_name,
                    expected_hashes=geneformer_expected_hashes,
                ),
            }
            alignment = validate_static_embedding_alignment(
                geneformer_validation=geneformer,
                static_embedding_path=(
                    vendor_root
                    / "tripso/Utils/gf-12L-95M-i4096_word_embeddings_may2025.pt"
                ),
            )
            geneformer["static_embedding_alignment"] = alignment
            geneformer["passed"] = bool(geneformer["passed"] and alignment["passed"])
        except Exception as exc:
            geneformer = {
                "passed": False,
                "requested": True,
                "root": str(Path(geneformer_root).resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }

    if smoke_mode == "none":
        smoke: dict[str, Any] = {
            "mode": "none",
            "mock_adapter_smoke_passed": False,
            "real_tripso_training_smoke_passed": False,
            "note": "No smoke test requested.",
        }
    elif smoke_mode == "mock":
        smoke = {"mode": "mock", **run_mock_projection_smoke()}
    elif smoke_mode == "real":
        if real_smoke_command is None:
            raise ValueError("smoke_mode='real' requires real_smoke_command")
        smoke = run_real_smoke_command(real_smoke_command, real_smoke_cwd or Path.cwd())
    else:
        raise ValueError(f"Unknown smoke mode: {smoke_mode}")

    environment_passed = (
        platform_check
        and python_check
        and dependency_check
        and asset_check
        and bool(import_result.get("passed"))
        and bool(geneformer.get("passed"))
    )
    return {
        "schema_version": "immune-health-tripso-environment/v1",
        "environment_passed": environment_passed,
        "real_end_to_end_passed": environment_passed
        and bool(smoke.get("real_tripso_training_smoke_passed")),
        "checks": {
            "linux": {
                "passed": platform_check,
                "observed": platform.system(),
                "required": "Linux",
            },
            "python_3_10": {
                "passed": python_check,
                "observed": platform.python_version(),
                "required": "3.10.x",
            },
            "dependencies": {
                "passed": dependency_check,
                "results": dependency_results,
            },
            "vendor_assets": {"passed": asset_check, "results": assets},
            "tripso_import_and_api": import_result,
            "full_geneformer": geneformer,
            "smoke": smoke,
        },
    }
