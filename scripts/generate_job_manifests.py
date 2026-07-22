#!/usr/bin/env python3
"""Generate, but never submit, staged TRIPSO JSONL array manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot create a job slug from {value!r}")
    return slug


def _replace(template: str, values: Mapping[str, Any]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", str(value))
    return rendered


def _is_pending(value: str) -> bool:
    return "<" in value and ">" in value


def load_experiment(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {path}")
    if config.get("schema_version") != "immune-health-tripso-experiment/v1":
        raise ValueError(f"Unsupported experiment config schema: {path}")
    datasets = config.get("healthy_datasets", [])
    lineages = config.get("lineages", [])
    if len(datasets) != 5 or len(set(datasets)) != 5:
        raise ValueError("TRIPSO LODO requires exactly five unique healthy datasets")
    if len(lineages) != 5 or len(set(lineages)) != 5:
        raise ValueError(
            "The pilot matrix requires exactly five unique primary lineages"
        )
    hvg_sizes = config.get("hvg_sizes")
    if hvg_sizes != [3000, 9000]:
        raise ValueError(
            "The primary TRIPSO comparison requires hvg_sizes: [3000, 9000]"
        )
    selection = config.get("model_selection", {})
    if selection != {
        "role": "validation",
        "inner_validation_fold": 0,
        "inner_fold_column": "inner_fold",
        "outer_query_evaluation_only": True,
        "outer_query_used_for_selection": False,
    }:
        raise ValueError(
            "model_selection must fix inner_fold=0 validation and seal outer query"
        )
    samplers = config.get("samplers")
    if not isinstance(samplers, dict):
        raise ValueError("samplers must be a YAML mapping")
    primary = {
        name
        for name, settings in samplers.items()
        if isinstance(settings, dict) and settings.get("primary_screen") is True
    }
    expected_primary = {"native_all_cells", "donor_uniform_observed", "hybrid"}
    if primary != expected_primary:
        raise ValueError(
            f"Primary sampler set must be exactly {sorted(expected_primary)}"
        )
    if (
        "fully_balanced" not in samplers
        or samplers["fully_balanced"].get("primary_screen") is not False
    ):
        raise ValueError("fully_balanced must remain an optional diagnostic")
    valid_project_modes = {"observed_proportions", "fully_balanced", "hybrid"}
    for name, settings in samplers.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Sampler {name!r} must be a YAML mapping")
        enabled = settings.get("project_sampler_enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"Sampler {name!r} must declare project_sampler_enabled")
        backend = settings.get("backend")
        expected_backend = "hierarchical" if enabled else "vendor"
        if backend != expected_backend:
            raise ValueError(f"Sampler {name!r} backend must be {expected_backend!r}")
        if enabled:
            if settings.get("project_sampler_mode") not in valid_project_modes:
                raise ValueError(f"Sampler {name!r} has an invalid project mode")
            for field in ("alpha", "fine_type_lambda"):
                value = settings.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"Sampler {name!r} {field} must be numeric")
                if not 0 <= float(value) <= 1:
                    raise ValueError(f"Sampler {name!r} {field} must be in [0, 1]")
        elif any(
            settings.get(field) is not None
            for field in ("project_sampler_mode", "alpha", "fine_type_lambda")
        ):
            raise ValueError(
                f"Native vendor sampler {name!r} must use null alpha/lambda/mode"
            )
    return config


def _selected_configurations(
    config: Mapping[str, Any], stage_name: str, lineage: str
) -> list[dict[str, Any]]:
    stage = config["stages"][stage_name]
    selection_mode = stage.get("configuration_selection")
    if selection_mode in {"all_primary", "all_configured"}:
        return [
            {"sampler": sampler, "hvg_size": hvg_size, "selection_index": index}
            for index, (sampler, hvg_size) in enumerate(
                (
                    (sampler, hvg_size)
                    for sampler, settings in config["samplers"].items()
                    if settings["primary_screen"] or selection_mode == "all_configured"
                    for hvg_size in config["hvg_sizes"]
                )
            )
        ]
    selected = stage.get("configuration_selection_by_lineage", {}).get(lineage)
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{stage_name} has no configuration for {lineage!r}")
    configurations: list[dict[str, Any]] = []
    for index, value in enumerate(selected):
        if not isinstance(value, dict) or set(value) != {"sampler", "hvg_size"}:
            raise ValueError(
                f"{stage_name} configuration {index + 1} for {lineage!r} must "
                "contain only sampler and hvg_size"
            )
        sampler = value["sampler"]
        hvg_size = value["hvg_size"]
        if not isinstance(sampler, str) or not sampler:
            raise ValueError("Selected sampler must be a non-empty string")
        if not _is_pending(sampler) and sampler not in config["samplers"]:
            raise ValueError(f"Unknown sampler {sampler!r} for {lineage!r}")
        hvg_pending = isinstance(hvg_size, str) and _is_pending(hvg_size)
        if not hvg_pending and hvg_size not in config["hvg_sizes"]:
            raise ValueError(f"Unknown HVG size {hvg_size!r} for {lineage!r}")
        configurations.append(
            {
                "sampler": sampler,
                "hvg_size": hvg_size,
                "selection_index": index,
            }
        )
    return configurations


def generate_jobs(
    config: Mapping[str, Any], stage_name: str, *, base_seed: int
) -> list[dict[str, Any]]:
    """Expand one stage; unresolved scientific selections become non-runnable rows."""
    if stage_name not in {"stage1", "stage2", "stage3"}:
        raise ValueError(f"Unknown stage: {stage_name}")
    stage = config["stages"][stage_name]
    offsets = stage.get("seed_offsets")
    if (
        not isinstance(offsets, list)
        or not offsets
        or not all(isinstance(value, int) and value >= 0 for value in offsets)
    ):
        raise ValueError(f"{stage_name}.seed_offsets must be non-negative integers")
    reused_offsets = stage.get("reused_seed_offsets", [])
    if not isinstance(reused_offsets, list) or not all(
        isinstance(value, int) and value >= 0 for value in reused_offsets
    ):
        raise ValueError(
            f"{stage_name}.reused_seed_offsets must be non-negative integers"
        )
    overlap = sorted(set(offsets) & set(reused_offsets))
    if overlap:
        raise ValueError(f"{stage_name} would retrain reused seed offsets: {overlap}")

    jobs: list[dict[str, Any]] = []
    datasets = list(config["healthy_datasets"])
    fold_values: list[str | None]
    if stage["design"] == "lodo":
        fold_values = datasets
    elif stage["design"] == "final_all_healthy":
        fold_values = [None]
    else:
        raise ValueError(f"Unknown design in {stage_name}: {stage['design']!r}")

    for lineage in config["lineages"]:
        lineage_slug = _slug(lineage)
        for heldout in fold_values:
            fold_slug = f"lodo_{_slug(heldout)}" if heldout else "all_healthy"
            configurations = _selected_configurations(config, stage_name, lineage)
            for configuration in configurations:
                sampler = configuration["sampler"]
                hvg_size = configuration["hvg_size"]
                sampler_pending = _is_pending(sampler)
                hvg_pending = isinstance(hvg_size, str) and _is_pending(hvg_size)
                pending = sampler_pending or hvg_pending
                selection_number = int(configuration["selection_index"]) + 1
                sampler_slug = (
                    f"pending_sampler_{selection_number}"
                    if sampler_pending
                    else sampler
                )
                hvg_slug = (
                    f"pending_hvg_{selection_number}"
                    if hvg_pending
                    else f"hvg{int(hvg_size)}"
                )
                feature_set = None if hvg_pending else f"hvg{int(hvg_size)}_plus_gp"
                configuration_slug = (
                    f"pending_configuration_{selection_number}"
                    if pending
                    else f"{feature_set}__{sampler_slug}"
                )
                sampler_config = config["samplers"].get(sampler, {})
                for offset in offsets:
                    seed = int(base_seed) + offset
                    values = {
                        "stage": stage_name,
                        "lineage": lineage,
                        "lineage_slug": lineage_slug,
                        "heldout_dataset": heldout or "all_healthy",
                        "fold_slug": fold_slug,
                        "sampler": sampler_slug,
                        "hvg_size": hvg_size,
                        "hvg_slug": hvg_slug,
                        "feature_set": feature_set,
                        "configuration_slug": configuration_slug,
                        "seed": seed,
                    }
                    identity = "|".join(
                        str(values[name])
                        for name in (
                            "stage",
                            "lineage_slug",
                            "fold_slug",
                            "configuration_slug",
                            "seed",
                        )
                    )
                    suffix = hashlib.sha256(identity.encode()).hexdigest()[:10]
                    job_id = (
                        f"{stage_name}-{lineage_slug}-{fold_slug}-"
                        f"{configuration_slug}-s{seed}-{suffix}"
                    )
                    values["job_id"] = job_id
                    output_dir = _replace(config["paths"]["output_template"], values)
                    job_spec_path = f"{output_dir}/job_spec.json"
                    values["job_spec_path"] = job_spec_path
                    if heldout is None:
                        fold_input = _replace(
                            config["paths"]["final_input_template"], values
                        )
                        fold_manifest = None
                    else:
                        fold_input = _replace(
                            config["paths"]["fold_input_template"], values
                        )
                        fold_manifest = _replace(
                            config["paths"]["fold_manifest_template"], values
                        )

                    upstream = [
                        {
                            "path": config["paths"]["environment_report"],
                            "json_require": {"environment_passed": True},
                        },
                        {
                            "path": fold_input,
                            "json_require": {
                                "schema_version": "immune-health-tripso-fold-input/v1"
                            },
                        },
                    ]
                    if fold_manifest:
                        upstream.append({"path": fold_manifest})
                    command = [
                        _replace(str(part), values) for part in config["worker_command"]
                    ]
                    jobs.append(
                        {
                            "schema_version": "immune-health-slurm-job/v1",
                            "job_id": job_id,
                            "experiment_id": config["experiment_id"],
                            "stage": stage_name,
                            "design": stage["design"],
                            "lineage": lineage,
                            "heldout_dataset": heldout,
                            "reference_datasets": [
                                dataset for dataset in datasets if dataset != heldout
                            ],
                            "hvg_size": None if hvg_pending else int(hvg_size),
                            "feature_set": feature_set,
                            "includes_all_retained_gp_genes": True,
                            "sampler_mode": None if pending else sampler,
                            "sampling_backend": sampler_config.get("backend"),
                            "project_sampler_enabled": sampler_config.get(
                                "project_sampler_enabled"
                            ),
                            "project_sampler_mode": sampler_config.get(
                                "project_sampler_mode"
                            ),
                            "selection_status": "pending" if pending else "configured",
                            "model_selection_role": (
                                "validation" if stage["design"] == "lodo" else None
                            ),
                            "inner_validation_fold": (
                                config["model_selection"]["inner_validation_fold"]
                                if stage["design"] == "lodo"
                                else None
                            ),
                            "outer_query_evaluation_only": (stage["design"] == "lodo"),
                            "outer_query_used_for_selection": False,
                            "runnable": not pending,
                            "alpha": sampler_config.get("alpha"),
                            "fine_type_lambda": sampler_config.get("fine_type_lambda"),
                            "extends_stage": stage.get("extends_stage"),
                            "reused_seed_offsets": list(reused_offsets),
                            "seed": seed,
                            "output_dir": output_dir,
                            "upstream_artifacts": upstream,
                            "expected_outputs": [
                                f"{output_dir}/checkpoints/last.ckpt",
                                f"{output_dir}/model_manifest.json",
                            ],
                            "command": command if not pending else [],
                            "working_directory": "${PROJECT_ROOT}",
                            "job_spec_path": job_spec_path,
                        }
                    )
    return jobs


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_manifest(path: Path, jobs: Iterable[Mapping[str, Any]]) -> int:
    rows = list(jobs)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write_text(path, content)
    return len(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiments" / "tripso_lodo.yaml",
    )
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "stage3", "all"), default="all"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "slurm" / "manifests"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    config = load_experiment(args.config)
    stage_names = (
        ("stage1", "stage2", "stage3") if args.stage == "all" else (args.stage,)
    )
    summary = {
        "schema_version": "immune-health-slurm-manifest-summary/v1",
        "experiment_config": str(args.config.resolve()),
        "base_seed": args.seed,
        "hvg_sizes": list(config["hvg_sizes"]),
        "primary_samplers": [
            name
            for name, settings in config["samplers"].items()
            if settings["primary_screen"]
        ],
        "model_selection": dict(config["model_selection"]),
        "stages": {},
        "submitted": False,
    }
    for stage_name in stage_names:
        jobs = generate_jobs(config, stage_name, base_seed=args.seed)
        runnable = sum(bool(job["runnable"]) for job in jobs)
        summary["stages"][stage_name] = {
            "jobs": len(jobs),
            "runnable_jobs": runnable,
            "pending_selection_jobs": len(jobs) - runnable,
            "seed_offsets": list(config["stages"][stage_name]["seed_offsets"]),
            "reused_seed_offsets": list(
                config["stages"][stage_name].get("reused_seed_offsets", [])
            ),
        }
        LOGGER.info(
            "%s: %d rows (%d runnable, %d pending scientific selection)",
            stage_name,
            len(jobs),
            runnable,
            len(jobs) - runnable,
        )
        if not args.dry_run:
            write_manifest(args.output_dir / f"{stage_name}.jsonl", jobs)
    if not args.dry_run:
        _atomic_write_text(
            args.output_dir / "manifest_summary.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
        LOGGER.info("Generated manifests only; no jobs were submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
