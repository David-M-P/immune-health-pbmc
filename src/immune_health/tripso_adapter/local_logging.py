"""Network-free experiment logging around the inspected TRIPSO train wrapper.

The vendored ``run_training`` function couples model training to Weights &
Biases in three places: login, logger construction, and an API readback after
``Trainer.fit``.  An offline W&B run cannot satisfy that API readback.  This
module changes only those observability hooks at runtime and leaves the vendor
datamodule, models, Lightning modules, callbacks, losses, and trainer in place.
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

from .contracts import sha256_path

TRACKING_GLOBALS = frozenset(
    {
        "configure_save_id",
        "configure_wandb",
        "configure_logger",
        "rank_zero_only",
    }
)
LOCAL_LOG_DIRECTORY = "local_csv"
LOCAL_LOG_VERSION = "training"
CANONICAL_METRICS_FILENAME = "training_metrics.csv"


def local_tracking_plan(output_dir: Path) -> dict[str, Any]:
    """Describe the project-owned tracking contract before training starts."""

    output_dir = Path(output_dir).resolve()
    return {
        "backend": "pytorch_lightning_csv",
        "network_required": False,
        "wandb_login_called": False,
        "wandb_api_readback_called": False,
        "vendor_source_modified": False,
        "runtime_patch_scope": sorted(TRACKING_GLOBALS),
        "logger_directory": str(output_dir / LOCAL_LOG_DIRECTORY / LOCAL_LOG_VERSION),
        "canonical_metrics_path": str(output_dir / CANONICAL_METRICS_FILENAME),
    }


def _safe_save_id(args: Mapping[str, Any]) -> str:
    """Return a stable callback/checkpoint label without an external login."""

    model_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.get("model_type", "model")))
    seed = int(args.get("seed", 0))
    return f"immune_health_{model_type}_seed_{seed}"


@contextmanager
def local_csv_logging_context(
    train_fn: Callable[..., Any], *, output_dir: Path
) -> Iterator[dict[str, Any]]:
    """Temporarily replace only TRIPSO's W&B-facing helper globals.

    A custom test/training callable without the inspected vendor globals is
    allowed through unchanged.  A partial vendor-like surface is rejected so a
    future upstream refactor cannot silently re-enable a network dependency.
    """

    globals_dict = getattr(train_fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        yield {"surface_status": "custom_callable_unmodified"}
        return

    present = TRACKING_GLOBALS & globals_dict.keys()
    if not present:
        yield {"surface_status": "custom_callable_unmodified"}
        return
    if present != TRACKING_GLOBALS:
        missing = sorted(TRACKING_GLOBALS - present)
        raise RuntimeError(
            "TRIPSO tracking surface differs from the inspected adapter; missing "
            f"globals: {missing}"
        )
    pl_module = globals_dict.get("pl")
    csv_logger_class = getattr(getattr(pl_module, "loggers", None), "CSVLogger", None)
    if csv_logger_class is None:
        raise RuntimeError(
            "TRIPSO's imported PyTorch Lightning module does not expose CSVLogger"
        )

    resolved_output = Path(output_dir).resolve()

    def configure_save_id(args: Mapping[str, Any]) -> str:
        return _safe_save_id(args)

    def configure_wandb(args: Mapping[str, Any], save_id: str) -> None:
        del args, save_id

    def configure_logger(args: Mapping[str, Any]) -> Any:
        configured_output = Path(str(args["output_dir"])).resolve()
        if configured_output != resolved_output:
            raise RuntimeError(
                "Vendor logger output differs from the fold-bound output directory: "
                f"{configured_output} != {resolved_output}"
            )
        return csv_logger_class(
            save_dir=str(resolved_output),
            name=LOCAL_LOG_DIRECTORY,
            version=LOCAL_LOG_VERSION,
        )

    originals = {name: globals_dict[name] for name in TRACKING_GLOBALS}
    globals_dict.update(
        {
            "configure_save_id": configure_save_id,
            "configure_wandb": configure_wandb,
            "configure_logger": configure_logger,
            # This name is used only by the final W&B API readback condition in
            # the inspected function body.  It is a module-local proxy, not
            # Lightning's global distributed rank state.
            "rank_zero_only": SimpleNamespace(rank=1),
        }
    )
    try:
        yield {
            "surface_status": "inspected_vendor_tracking_replaced",
            "save_id": None,
        }
    finally:
        globals_dict.update(originals)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with (
            source.open("rb") as input_handle,
            os.fdopen(descriptor, "wb") as output_handle,
        ):
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _numeric_value(value: str) -> int | float | None:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def collect_local_training_metrics(output_dir: Path) -> dict[str, Any]:
    """Create one canonical CSV and a small JSON-safe final-value summary."""

    output_dir = Path(output_dir).resolve()
    logger_root = output_dir / LOCAL_LOG_DIRECTORY / LOCAL_LOG_VERSION
    candidates = (
        sorted(logger_root.rglob("metrics.csv")) if logger_root.exists() else []
    )
    if len(candidates) > 1:
        raise RuntimeError(
            "Local TRIPSO logging emitted multiple metric tables for one job: "
            + ", ".join(map(str, candidates))
        )
    if not candidates:
        return {
            "status": "not_emitted",
            "backend": "pytorch_lightning_csv",
            "canonical_path": str(output_dir / CANONICAL_METRICS_FILENAME),
            "n_rows": 0,
            "last_logged_values": {},
        }

    source = candidates[0]
    canonical = output_dir / CANONICAL_METRICS_FILENAME
    _copy_atomic(source, canonical)
    n_rows = 0
    last_values: dict[str, int | float] = {}
    with canonical.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            for name, raw_value in row.items():
                if raw_value is None:
                    continue
                numeric = _numeric_value(raw_value)
                if numeric is not None:
                    last_values[name] = numeric
    return {
        "status": "written",
        "backend": "pytorch_lightning_csv",
        "source_path": str(source.resolve()),
        "canonical_path": str(canonical),
        "sha256": sha256_path(canonical),
        "n_rows": n_rows,
        "last_logged_values": last_values,
    }
