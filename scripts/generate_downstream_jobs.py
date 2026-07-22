#!/usr/bin/env python3
"""Generate strict, restartable downstream CPU manifests without submitting jobs.

Pass 1 expands completed reference/validation projections through the frozen-GP
selector.  Pass 2 is a separate outer-evaluation boundary: it requires both the
validated selector artifact and an explicit, self-hashed query allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from immune_health.gene_programs.tripso_selection import (
    validate_tripso_gp_selection_manifest,
)
from immune_health.tripso_adapter.arrow_bridge import (
    validate_projection_output_for_conversion,
)

JOB_SCHEMA = "immune-health-slurm-job/v1"
PLAN_SCHEMA = "immune-health-downstream-candidate-plan/v1"
ALLOWLIST_SCHEMA = "immune-health-outer-query-evaluation-allowlist/v1"
PROJECTION_SCHEMA = "immune-health-tripso-projection-output/v1"
CONVERSION_SCHEMA = "immune-health-tripso-arrow-bridge/v1"
ENDPOINT_SCHEMA = "immune-health-donor-gp-endpoint/v1"
REFERENCE_SCHEMA = "immune-health-frozen-healthy-reference/v1"
SELECTION_SCHEMA = "immune-health-tripso-gp-selection/v1"
EMPIRICAL_INDEX_SCHEMA = "immune-health-empirical-row-index/v1"
EMPIRICAL_SCORE_SCHEMA = "immune-health-empirical-matched-depth-score/v1"
WEIGHTING_SCHEMES = ("donor_pooled", "cohort_balanced")
ROLES = ("reference", "validation", "query")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"
    return f"{base}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def _embedding_stem(gp_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", gp_id).strip("._") or "embedding"
    return f"{base}-{hashlib.sha256(gp_id.encode()).hexdigest()[:10]}"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_self_hashed_json(
    path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    value = _read_json(path, label)
    if value.get("schema_version") != schema:
        raise ValueError(f"Unsupported {label} schema: {path}")
    content = dict(value)
    claimed = content.pop("manifest_sha256", None)
    if claimed != _canonical_hash(content):
        raise ValueError(f"{label} canonical self-hash does not match: {path}")
    return value, _file_hash(path)


def _expand_path(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "${" in expanded or ("<" in expanded and ">" in expanded):
        raise ValueError(f"Unresolved path placeholder: {value}")
    return Path(expanded).resolve()


def _validate_bound_file(record: Any, label: str) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a path/SHA256 object")
    configured = record.get("path")
    expected = record.get("sha256")
    if (
        not isinstance(configured, str)
        or not configured
        or not isinstance(expected, str)
    ):
        raise ValueError(f"{label} requires non-empty path and sha256")
    path = _expand_path(configured)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _file_hash(path)
    if observed != expected:
        raise ValueError(f"{label} hash differs: {path}")
    return configured, expected


def _read_project_jobs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != JOB_SCHEMA:
                    raise ValueError(f"Unsupported job at {path}:{line_number}")
                if not str(row.get("stage", "")).startswith("posttrain_project_"):
                    raise ValueError(
                        f"Expected post-training projection job at {path}:{line_number}"
                    )
                role = row.get("projection_role")
                if role not in ROLES:
                    raise ValueError(f"Unsupported projection role: {role!r}")
                rows.append(row)
    return rows


def _project_index(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for job in jobs:
        key = (str(job.get("parent_training_job_id", "")), str(job["projection_role"]))
        if not key[0] or key in result:
            raise ValueError(f"Duplicate or empty projection identity: {key}")
        result[key] = job
    return result


def _projection_state(
    job: Mapping[str, Any], gp_ids: Sequence[str]
) -> tuple[dict[str, Any] | None, str | None]:
    if not job.get("runnable", False):
        return None, "projection_job_is_non_runnable"
    data_dir = str(job["projection_data_dir"])
    manifest_configured = f"{data_dir}/projection_output_manifest.json"
    try:
        manifest_path = _expand_path(manifest_configured)
    except ValueError as exc:
        return None, str(exc)
    if not manifest_path.is_file():
        return None, f"projection_output_manifest_missing:{manifest_path}"
    payload = _read_json(manifest_path, "projection output manifest")
    if payload.get("schema_version") != PROJECTION_SCHEMA:
        raise ValueError(f"Unsupported projection output: {manifest_path}")
    if payload.get("projection_role") != job.get("projection_role"):
        raise ValueError("Projection job role differs from completed output")
    arrow_relative = payload.get("arrow_dataset")
    if not isinstance(arrow_relative, str) or not arrow_relative:
        raise ValueError("Projection output lacks its Arrow dataset path")
    arrow_configured = f"{data_dir}/{arrow_relative}"
    arrow_path = _expand_path(arrow_configured)
    validation = validate_projection_output_for_conversion(
        manifest_path, arrow_path, embedding_columns=gp_ids
    )
    return {
        "job": job,
        "payload": payload,
        "validation": validation,
        "manifest": manifest_configured,
        "manifest_file_sha256": _file_hash(manifest_path),
        "arrow_dataset": arrow_configured,
    }, None


def _plan_models(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = plan.get("models")
    if not isinstance(raw, list):
        raise ValueError("Candidate plan models must be a list")
    result: dict[str, dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("Candidate plan model rows must be objects")
        parent = str(value.get("parent_training_job_id", ""))
        if not parent or parent in result:
            raise ValueError(f"Duplicate or empty candidate-plan model: {parent!r}")
        status = value.get("model_selection_status")
        if status not in {"selected", "pending", "not_selected"}:
            raise ValueError(f"Invalid model_selection_status for {parent}")
        result[parent] = value
    return result


def _candidate_endpoints(model: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = model.get("candidate_endpoints")
    if not isinstance(raw, list) or not raw:
        return []
    values: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("candidate_endpoints rows must be objects")
        gp_id = str(item.get("gp_id", "")).strip()
        fine_type = str(item.get("fine_type", "")).strip()
        if not gp_id or not fine_type:
            raise ValueError("candidate endpoint GP/fine type cannot be empty")
        values.append((gp_id, fine_type))
    if len(values) != len(set(values)):
        raise ValueError("candidate_endpoints contains duplicates")
    return values


def _minimum_exact_sex_donors(plan: Mapping[str, Any]) -> int:
    """Resolve the self-hash-bound exact-sex support threshold."""

    reference = plan.get("healthy_reference")
    if not isinstance(reference, Mapping):
        raise ValueError(
            "Candidate plan must contain a healthy_reference settings object"
        )
    value = reference.get("minimum_exact_sex_donors")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "Candidate plan healthy_reference.minimum_exact_sex_donors must be "
            "a positive integer"
        )
    return value


def _job(
    *,
    job_id: str,
    stage: str,
    parent: str,
    model: Mapping[str, Any],
    output_dir: str,
    seed: int,
    command: Sequence[str],
    upstream: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
    depends_on: Sequence[str],
    runnable: bool = True,
    pending_reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if runnable and (not command or not expected):
        raise ValueError("Runnable downstream jobs require command and outputs")
    value: dict[str, Any] = {
        "schema_version": JOB_SCHEMA,
        "job_id": job_id,
        "stage": stage,
        "parent_training_job_id": parent,
        "selection_group_id": model.get("selection_group_id", parent),
        "lineage": model.get("lineage"),
        "seed": int(seed),
        "compute_class": "cpu",
        "runnable": bool(runnable),
        "pending_reason": pending_reason,
        "output_dir": output_dir,
        "job_spec_path": f"{output_dir}/job_spec.json",
        "working_directory": "${PROJECT_ROOT}",
        "upstream_artifacts": list(upstream),
        "depends_on_job_ids": list(dict.fromkeys(depends_on)),
        "expected_outputs": list(expected) if runnable else [],
        "command": list(command) if runnable else [],
        "outer_query_results_used_for_selection": False,
        "empirical_sliced_wasserstein": "not_computed_no_empirical_scorer_cli",
    }
    value.update(dict(extra or {}))
    return value


def _pending(
    *,
    parent: str,
    model: Mapping[str, Any],
    base: str,
    seed: int,
    reason: str,
    label: str,
) -> dict[str, Any]:
    token = _slug(f"{parent}|{label}")
    return _job(
        job_id=f"downstream-pending-{token}",
        stage=label,
        parent=parent,
        model=model,
        output_dir=f"{base}/pending/{token}",
        seed=seed,
        command=[],
        upstream=[],
        expected=[],
        depends_on=[],
        runnable=False,
        pending_reason=reason,
    )


def _base(job: Mapping[str, Any]) -> str:
    projection_dir = PurePosixPath(str(job["projection_data_dir"]))
    return f"{projection_dir.parent.parent}/downstream"


def _plan_artifact(path: Path, file_sha256: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256,
        "json_require": {
            "schema_version": PLAN_SCHEMA,
            "outer_query_results_used_for_selection": False,
        },
    }


def _static_artifact(path: str, sha256: str) -> dict[str, Any]:
    return {"path": path, "sha256": sha256}


def _role_metadata(model: Mapping[str, Any], role: str) -> tuple[str, str]:
    metadata = model.get("cell_metadata")
    if not isinstance(metadata, Mapping) or role not in metadata:
        raise ValueError(f"Missing exact cell_metadata binding for role={role}")
    return _validate_bound_file(metadata[role], f"{role} cell metadata")


def _scoring_resources(
    model: Mapping[str, Any], role: str, gp_id: str
) -> tuple[str, str, str, str, float]:
    resources = model.get("scoring_resources")
    if not isinstance(resources, Mapping) or not isinstance(
        resources.get(role), Mapping
    ):
        raise ValueError(f"Missing scoring_resources for role={role}")
    role_resources = resources[role]
    query_genes, query_genes_sha = _validate_bound_file(
        role_resources.get("query_genes"), f"{role} query genes"
    )
    vocabulary, vocabulary_sha = _validate_bound_file(
        role_resources.get("frozen_vocabulary"), f"{role} frozen vocabulary"
    )
    coverage = role_resources.get("gp_coverage")
    if isinstance(coverage, Mapping):
        coverage = coverage.get(gp_id)
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ValueError(f"Missing numeric GP coverage for {role}/{gp_id}")
    if not 0 <= float(coverage) <= 1:
        raise ValueError(f"GP coverage is outside [0,1] for {role}/{gp_id}")
    return query_genes, query_genes_sha, vocabulary, vocabulary_sha, float(coverage)


def _conversion_and_endpoints(
    *,
    output: dict[str, list[dict[str, Any]]],
    parent: str,
    model: Mapping[str, Any],
    state: Mapping[str, Any],
    role: str,
    endpoints: Sequence[tuple[str, str]],
    ontology: tuple[str, str],
    plan_upstream: Mapping[str, Any],
) -> dict[str, Any]:
    project = state["job"]
    seed = int(project["seed"])
    base = _base(project)
    metadata, metadata_sha = _role_metadata(model, role)
    gp_ids = list(dict.fromkeys(gp_id for gp_id, _ in endpoints))
    conversion_dir = f"{base}/converted/{role}"
    conversion_id = f"downstream-convert-{role}-{_slug(parent)}"
    converted = {
        gp_id: f"{conversion_dir}/{_embedding_stem(gp_id)}.npy" for gp_id in gp_ids
    }
    conversion_manifest = f"{conversion_dir}/arrow_conversion_manifest.json"
    converted_metadata = f"{conversion_dir}/cell_metadata.parquet"
    command = [
        "python",
        "-m",
        "immune_health.cli.convert_tripso_arrow",
        "--arrow-dataset",
        str(state["arrow_dataset"]),
        "--projection-output-manifest",
        str(state["manifest"]),
        "--cell-metadata",
        metadata,
        "--output-dir",
        conversion_dir,
    ]
    for gp_id in gp_ids:
        command.extend(["--embedding-column", gp_id])
    conversion = _job(
        job_id=conversion_id,
        stage=f"downstream_convert_{role}",
        parent=parent,
        model=model,
        output_dir=conversion_dir,
        seed=seed,
        command=command,
        upstream=[
            plan_upstream,
            {
                "path": str(state["manifest"]),
                "sha256": str(state["manifest_file_sha256"]),
                "json_require": {"schema_version": PROJECTION_SCHEMA},
            },
            {"path": str(state["arrow_dataset"])},
            _static_artifact(metadata, metadata_sha),
        ],
        expected=[conversion_manifest, converted_metadata, *converted.values()],
        depends_on=[str(project["job_id"])],
        extra={"projection_role": role, "candidate_gp_ids": gp_ids},
    )
    output[f"convert_{role}"].append(conversion)

    aggregate_jobs: dict[str, dict[str, Any]] = {}
    endpoint_jobs: dict[tuple[str, str], dict[str, Any]] = {}
    for gp_id in gp_ids:
        gp_token = _slug(gp_id)
        aggregate_dir = f"{base}/aggregation/{role}/{gp_token}"
        aggregate_id = f"downstream-aggregate-{role}-{gp_token}-{_slug(parent)}"
        aggregation_manifest = (
            f"{aggregate_dir}/donor_distribution_aggregation_manifest.json"
        )
        aggregate = _job(
            job_id=aggregate_id,
            stage=f"downstream_aggregate_{role}",
            parent=parent,
            model=model,
            output_dir=aggregate_dir,
            seed=seed,
            command=[
                "python",
                "-m",
                "immune_health.cli",
                "aggregate-donor-distributions",
                "--embeddings",
                converted[gp_id],
                "--metadata",
                converted_metadata,
                "--arrow-conversion-manifest",
                conversion_manifest,
                "--gp-id",
                gp_id,
                "--fine-type-universe",
                ontology[0],
                "--output-dir",
                aggregate_dir,
            ],
            upstream=[
                plan_upstream,
                {
                    "path": conversion_manifest,
                    "json_require": {"schema_version": CONVERSION_SCHEMA},
                },
                {"path": converted[gp_id]},
                {"path": converted_metadata},
                _static_artifact(*ontology),
            ],
            expected=[
                f"{aggregate_dir}/fine_type_distributions.parquet",
                f"{aggregate_dir}/empirical_distribution_groups.parquet",
                f"{aggregate_dir}/empirical_distribution_rows.npy",
                f"{aggregate_dir}/empirical_distribution_manifest.json",
                aggregation_manifest,
            ],
            depends_on=[conversion_id],
            extra={
                "projection_role": role,
                "gp_id": gp_id,
                "fine_type_universe_frozen": True,
            },
        )
        output[f"aggregate_{role}"].append(aggregate)
        aggregate_jobs[gp_id] = aggregate

        bootstrap_dir = f"{base}/bootstrap/cell/{role}/{gp_token}"
        bootstrap = _job(
            job_id=f"downstream-bootstrap-cell-{role}-{gp_token}-{_slug(parent)}",
            stage=f"downstream_bootstrap_cell_embedding_mean_{role}",
            parent=parent,
            model=model,
            output_dir=bootstrap_dir,
            seed=seed,
            command=[
                "python",
                "-m",
                "immune_health.cli",
                "bootstrap-scores",
                "--uncertainty-layer",
                "cell",
                "--metadata",
                converted_metadata,
                "--features",
                converted[gp_id],
                "--n-bootstrap",
                str(int(model.get("n_cell_bootstrap", 100))),
                "--output",
                f"{bootstrap_dir}/cell_bootstrap.parquet",
            ],
            upstream=[
                {"path": conversion_manifest},
                {"path": converted_metadata},
                {"path": converted[gp_id]},
            ],
            expected=[f"{bootstrap_dir}/cell_bootstrap.parquet"],
            depends_on=[conversion_id],
            extra={
                "uncertainty_layer": "cell",
                "projection_role": role,
                "uncertainty_estimand": (
                    "within-observation fine-type-stratified embedding mean; "
                    "not a healthy-reference score standard error"
                ),
            },
        )
        output["bootstrap_cell"].append(bootstrap)

    for gp_id, fine_type in endpoints:
        endpoint_token = _slug(f"{gp_id}|{fine_type}")
        aggregate_dir = str(aggregate_jobs[gp_id]["output_dir"])
        endpoint_dir = f"{base}/endpoints/{role}/{endpoint_token}"
        endpoint_id = f"downstream-endpoint-{role}-{endpoint_token}-{_slug(parent)}"
        endpoint = _job(
            job_id=endpoint_id,
            stage=f"downstream_endpoint_{role}",
            parent=parent,
            model=model,
            output_dir=endpoint_dir,
            seed=seed,
            command=[
                "python",
                "-m",
                "immune_health.cli",
                "assemble-donor-gp-endpoint",
                "--aggregate-table",
                f"{aggregate_dir}/fine_type_distributions.parquet",
                "--aggregation-manifest",
                f"{aggregate_dir}/donor_distribution_aggregation_manifest.json",
                "--projection-output-manifest",
                str(state["manifest"]),
                "--lineage",
                str(model["lineage"]),
                "--fine-type",
                fine_type,
                "--gp-id",
                gp_id,
                "--output-dir",
                endpoint_dir,
            ],
            upstream=[
                {
                    "path": (
                        f"{aggregate_dir}/donor_distribution_aggregation_manifest.json"
                    ),
                    "json_require": {
                        "status": "complete",
                        "stage": "donor_distribution_aggregation",
                    },
                },
                {"path": f"{aggregate_dir}/fine_type_distributions.parquet"},
                {
                    "path": str(state["manifest"]),
                    "json_require": {"schema_version": PROJECTION_SCHEMA},
                },
            ],
            expected=[
                f"{endpoint_dir}/endpoint_metadata.parquet",
                f"{endpoint_dir}/endpoint_locations.npy",
                f"{endpoint_dir}/endpoint_covariances.npy",
                f"{endpoint_dir}/endpoint_manifest.json",
            ],
            depends_on=[str(aggregate_jobs[gp_id]["job_id"])],
            extra={
                "projection_role": role,
                "gp_id": gp_id,
                "fine_type": fine_type,
            },
        )
        output[f"endpoint_{role}"].append(endpoint)
        endpoint_jobs[(gp_id, fine_type)] = endpoint
    return {
        "base": base,
        "seed": seed,
        "state": state,
        "endpoints": endpoint_jobs,
        "aggregates": aggregate_jobs,
    }


def _fit_references(
    *,
    output: dict[str, list[dict[str, Any]]],
    parent: str,
    model: Mapping[str, Any],
    reference_artifacts: Mapping[str, Any],
    minimum_exact_sex_donors: int,
    scientific_settings_artifact: Mapping[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    fits: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (gp_id, fine_type), endpoint in reference_artifacts["endpoints"].items():
        endpoint_dir = str(endpoint["output_dir"])
        for weighting in WEIGHTING_SCHEMES:
            fit_dir = (
                f"{reference_artifacts['base']}/references/"
                f"{_slug(f'{gp_id}|{fine_type}')}/{weighting}"
            )
            fit_id = (
                f"downstream-fit-{weighting}-{_slug(f'{gp_id}|{fine_type}')}-"
                f"{_slug(parent)}"
            )
            command = [
                "python",
                "-m",
                "immune_health.cli",
                "fit-healthy-reference",
                "--metadata",
                f"{endpoint_dir}/endpoint_metadata.parquet",
                "--features",
                f"{endpoint_dir}/endpoint_locations.npy",
                "--endpoint-manifest",
                f"{endpoint_dir}/endpoint_manifest.json",
                "--weighting-scheme",
                weighting,
                "--age-kernel-minimum-exact-sex-donors",
                str(minimum_exact_sex_donors),
                "--output-dir",
                fit_dir,
            ]
            if (
                reference_artifacts["state"]["payload"].get("reference_design")
                == "all_healthy"
            ):
                command.append("--final-all-healthy")
            fit = _job(
                job_id=fit_id,
                stage="downstream_fit_reference",
                parent=parent,
                model=model,
                output_dir=fit_dir,
                seed=int(reference_artifacts["seed"]),
                command=command,
                upstream=[
                    {
                        "path": f"{endpoint_dir}/endpoint_manifest.json",
                        "json_require": {"schema_version": ENDPOINT_SCHEMA},
                    },
                    {"path": f"{endpoint_dir}/endpoint_metadata.parquet"},
                    {"path": f"{endpoint_dir}/endpoint_locations.npy"},
                    {"path": f"{endpoint_dir}/endpoint_covariances.npy"},
                    scientific_settings_artifact,
                ],
                expected=[
                    f"{fit_dir}/healthy_reference.json",
                    f"{fit_dir}/healthy_reference_arrays.npz",
                    f"{fit_dir}/age_kernel_reference.json",
                    f"{fit_dir}/training_crossfit_scores.parquet",
                    f"{fit_dir}/age_support_grid.parquet",
                    f"{fit_dir}/cohort_age_slope_diagnostics.parquet",
                ],
                depends_on=[str(endpoint["job_id"])],
                extra={
                    "gp_id": gp_id,
                    "fine_type": fine_type,
                    "weighting_scheme": weighting,
                    "age_kernel_covariance_storage": "reuse_endpoint_npy_by_hash",
                    "minimum_exact_sex_donors": minimum_exact_sex_donors,
                    "minimum_exact_sex_donors_source": (
                        "candidate_plan.healthy_reference.minimum_exact_sex_donors"
                    ),
                },
            )
            output["fit_reference"].append(fit)
            fits[(gp_id, fine_type, weighting)] = fit
    return fits


def _emit_empirical_scores(
    *,
    output: dict[str, list[dict[str, Any]]],
    role: str,
    parent: str,
    model: Mapping[str, Any],
    reference_artifacts: Mapping[str, Any],
    target_artifacts: Mapping[str, Any],
    minimum_exact_sex_donors: int,
    scientific_settings_artifact: Mapping[str, Any],
) -> None:
    """Emit matched-depth empirical sensitivities for exact role-paired artifacts."""

    for identity, target_endpoint in target_artifacts["endpoints"].items():
        gp_id, fine_type = identity
        reference_endpoint = reference_artifacts["endpoints"][identity]
        reference_index = (
            f"{reference_artifacts['aggregates'][gp_id]['output_dir']}/"
            "empirical_distribution_manifest.json"
        )
        target_index = (
            f"{target_artifacts['aggregates'][gp_id]['output_dir']}/"
            "empirical_distribution_manifest.json"
        )
        for weighting in WEIGHTING_SCHEMES:
            empirical_dir = (
                f"{target_artifacts['base']}/empirical_scores/{role}/"
                f"{_slug(f'{gp_id}|{fine_type}')}/{weighting}"
            )
            empirical_id = (
                f"downstream-empirical-{role}-{weighting}-"
                f"{_slug(f'{gp_id}|{fine_type}')}-{_slug(parent)}"
            )
            empirical = _job(
                job_id=empirical_id,
                stage=f"downstream_empirical_score_{role}",
                parent=parent,
                model=model,
                output_dir=empirical_dir,
                seed=int(target_artifacts["seed"]),
                command=[
                    "python",
                    "-m",
                    "immune_health.cli",
                    "score-empirical-endpoint",
                    "--reference-endpoint-manifest",
                    f"{reference_endpoint['output_dir']}/endpoint_manifest.json",
                    "--query-endpoint-manifest",
                    f"{target_endpoint['output_dir']}/endpoint_manifest.json",
                    "--reference-empirical-index",
                    reference_index,
                    "--query-empirical-index",
                    target_index,
                    "--weighting-scheme",
                    weighting,
                    "--minimum-exact-sex-donors",
                    str(minimum_exact_sex_donors),
                    "--output-dir",
                    empirical_dir,
                ],
                upstream=[
                    {
                        "path": (
                            f"{reference_endpoint['output_dir']}/endpoint_manifest.json"
                        ),
                        "json_require": {"schema_version": ENDPOINT_SCHEMA},
                    },
                    {
                        "path": (
                            f"{target_endpoint['output_dir']}/endpoint_manifest.json"
                        ),
                        "json_require": {"schema_version": ENDPOINT_SCHEMA},
                    },
                    {
                        "path": reference_index,
                        "json_require": {"schema_version": EMPIRICAL_INDEX_SCHEMA},
                    },
                    {
                        "path": target_index,
                        "json_require": {"schema_version": EMPIRICAL_INDEX_SCHEMA},
                    },
                    scientific_settings_artifact,
                ],
                expected=[
                    f"{empirical_dir}/empirical_matched_depth_scores.parquet",
                    f"{empirical_dir}/empirical_matched_depth_replicates.parquet",
                    f"{empirical_dir}/empirical_reliability.parquet",
                    f"{empirical_dir}/empirical_scoring_manifest.json",
                ],
                depends_on=[
                    str(reference_endpoint["job_id"]),
                    str(target_endpoint["job_id"]),
                    str(reference_artifacts["aggregates"][gp_id]["job_id"]),
                    str(target_artifacts["aggregates"][gp_id]["job_id"]),
                ],
                extra={
                    "target_role": role,
                    "gp_id": gp_id,
                    "fine_type": fine_type,
                    "weighting_scheme": weighting,
                    "empirical_sliced_wasserstein": (
                        "computed_by_score-empirical-endpoint_on_success"
                    ),
                    "empirical_output_schema": EMPIRICAL_SCORE_SCHEMA,
                    "minimum_exact_sex_donors": minimum_exact_sex_donors,
                    "minimum_exact_sex_donors_source": (
                        "candidate_plan.healthy_reference.minimum_exact_sex_donors"
                    ),
                },
            )
            output["empirical_scoring"].append(empirical)


def _score_role(
    *,
    output: dict[str, list[dict[str, Any]]],
    role: str,
    parent: str,
    model: Mapping[str, Any],
    target_artifacts: Mapping[str, Any],
    fits: Mapping[tuple[str, str, str], Mapping[str, Any]],
    selected_artifact: Mapping[str, Any] | None = None,
    allowlist_artifact: Mapping[str, Any] | None = None,
) -> None:
    for (gp_id, fine_type), endpoint in target_artifacts["endpoints"].items():
        endpoint_dir = str(endpoint["output_dir"])
        query_genes, genes_sha, vocabulary, vocabulary_sha, coverage = (
            _scoring_resources(model, role, gp_id)
        )
        state = target_artifacts["state"]
        payload = state["payload"]
        for weighting in WEIGHTING_SCHEMES:
            fit = fits[(gp_id, fine_type, weighting)]
            score_dir = (
                f"{target_artifacts['base']}/scores/{role}/"
                f"{_slug(f'{gp_id}|{fine_type}')}/{weighting}"
            )
            score_id = (
                f"downstream-score-{role}-{weighting}-"
                f"{_slug(f'{gp_id}|{fine_type}')}-{_slug(parent)}"
            )
            upstream: list[Mapping[str, Any]] = [
                {
                    "path": f"{fit['output_dir']}/healthy_reference.json",
                    "json_require": {"schema_version": REFERENCE_SCHEMA},
                },
                {
                    "path": f"{endpoint_dir}/endpoint_manifest.json",
                    "json_require": {"schema_version": ENDPOINT_SCHEMA},
                },
                {"path": f"{endpoint_dir}/endpoint_metadata.parquet"},
                {"path": f"{endpoint_dir}/endpoint_locations.npy"},
                _static_artifact(query_genes, genes_sha),
                _static_artifact(vocabulary, vocabulary_sha),
            ]
            if selected_artifact is not None:
                upstream.append(selected_artifact)
            if allowlist_artifact is not None:
                upstream.append(allowlist_artifact)
            score = _job(
                job_id=score_id,
                stage=f"downstream_score_{role}",
                parent=parent,
                model=model,
                output_dir=score_dir,
                seed=int(target_artifacts["seed"]),
                command=[
                    "python",
                    "-m",
                    "immune_health.cli",
                    "score-query",
                    "--reference-manifest",
                    f"{fit['output_dir']}/healthy_reference.json",
                    "--query-metadata",
                    f"{endpoint_dir}/endpoint_metadata.parquet",
                    "--features",
                    f"{endpoint_dir}/endpoint_locations.npy",
                    "--endpoint-manifest",
                    f"{endpoint_dir}/endpoint_manifest.json",
                    "--query-genes",
                    query_genes,
                    "--frozen-vocabulary",
                    vocabulary,
                    "--gp-coverage",
                    str(coverage),
                    "--model-manifest",
                    str(payload["model_manifest"]),
                    "--query-manifest",
                    str(payload["projection_input_manifest"]),
                    "--output",
                    f"{score_dir}/{role}_scores.parquet",
                    "--report",
                    f"{score_dir}/{role}_report.json",
                ],
                upstream=upstream,
                expected=[
                    f"{score_dir}/{role}_scores.parquet",
                    f"{score_dir}/{role}_report.json",
                ],
                depends_on=[str(endpoint["job_id"]), str(fit["job_id"])],
                extra={
                    "scoring_role": role,
                    "gp_id": gp_id,
                    "fine_type": fine_type,
                    "weighting_scheme": weighting,
                    "outer_query_evaluation_only": role == "query",
                },
            )
            output[f"score_{role}"].append(score)


def _empty_output() -> dict[str, list[dict[str, Any]]]:
    return {
        **{f"convert_{role}": [] for role in ROLES},
        **{f"aggregate_{role}": [] for role in ROLES},
        **{f"endpoint_{role}": [] for role in ROLES},
        "fit_reference": [],
        "score_validation": [],
        "score_query": [],
        "select_transferable": [],
        "bootstrap_cell": [],
        "bootstrap_reference": [],
        "evaluate": [],
        "empirical_scoring": [],
        "pending": [],
    }


def generate_pass1(
    jobs: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Generate reference/validation work; query projection rows are ignored."""

    output = _empty_output()
    projects = _project_index(jobs)
    models = _plan_models(plan)
    minimum_exact_sex_donors = _minimum_exact_sex_donors(plan)
    ontology = _validate_bound_file(
        plan.get("fine_type_universe"), "frozen fine-type universe"
    )
    plan_upstream = _plan_artifact(plan_path, plan_file_sha256)
    resolved: dict[str, dict[str, Any]] = {}
    for parent, model in models.items():
        ref_job = projects.get((parent, "reference"))
        base = _base(ref_job) if ref_job is not None else "${OUTPUT_ROOT}/downstream"
        seed = int(ref_job.get("seed", 0)) if ref_job is not None else 0
        if model["model_selection_status"] != "selected":
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason=f"model_selection_status={model['model_selection_status']}",
                    label="downstream_model_selection_pending",
                )
            )
            continue
        endpoints = _candidate_endpoints(model)
        if not endpoints:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason="candidate fine-type/GP endpoints unresolved",
                    label="downstream_endpoint_selection_pending",
                )
            )
            continue
        if ref_job is None:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason="reference projection job row is missing",
                    label="downstream_reference_projection_pending",
                )
            )
            continue
        gp_ids = list(dict.fromkeys(gp for gp, _ in endpoints))
        reference_state, reason = _projection_state(ref_job, gp_ids)
        if reference_state is None:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason=str(reason),
                    label="downstream_reference_projection_pending",
                )
            )
            continue
        if str(model.get("lineage")) != str(reference_state["payload"]["lineage"]):
            raise ValueError(f"Candidate-plan lineage differs for {parent}")
        reference = _conversion_and_endpoints(
            output=output,
            parent=parent,
            model=model,
            state=reference_state,
            role="reference",
            endpoints=endpoints,
            ontology=ontology,
            plan_upstream=plan_upstream,
        )
        fits = _fit_references(
            output=output,
            parent=parent,
            model=model,
            reference_artifacts=reference,
            minimum_exact_sex_donors=minimum_exact_sex_donors,
            scientific_settings_artifact=plan_upstream,
        )
        state: dict[str, Any] = {
            "model": model,
            "reference": reference,
            "fits": fits,
            "endpoints": endpoints,
            "reference_state": reference_state,
        }
        validation_job = projects.get((parent, "validation"))
        if validation_job is None:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason="inner-validation projection job row is missing",
                    label="downstream_validation_scoring_pending",
                )
            )
        else:
            validation_state, validation_reason = _projection_state(
                validation_job, gp_ids
            )
            if validation_state is None:
                output["pending"].append(
                    _pending(
                        parent=parent,
                        model=model,
                        base=base,
                        seed=seed,
                        reason=str(validation_reason),
                        label="downstream_validation_scoring_pending",
                    )
                )
            else:
                validation = _conversion_and_endpoints(
                    output=output,
                    parent=parent,
                    model=model,
                    state=validation_state,
                    role="validation",
                    endpoints=endpoints,
                    ontology=ontology,
                    plan_upstream=plan_upstream,
                )
                _score_role(
                    output=output,
                    role="validation",
                    parent=parent,
                    model=model,
                    target_artifacts=validation,
                    fits=fits,
                )
                _emit_empirical_scores(
                    output=output,
                    role="validation",
                    parent=parent,
                    model=model,
                    reference_artifacts=reference,
                    target_artifacts=validation,
                    minimum_exact_sex_donors=minimum_exact_sex_donors,
                    scientific_settings_artifact=plan_upstream,
                )
                state["validation"] = validation
        resolved[parent] = state

    groups: dict[str, list[str]] = {}
    for parent, state in resolved.items():
        group = str(state["model"].get("selection_group_id", parent))
        groups.setdefault(group, []).append(parent)
    for group, parents in sorted(groups.items()):
        states = [resolved[parent] for parent in sorted(parents)]
        first = states[0]
        payloads = [value["reference_state"]["payload"] for value in states]
        lineages = {str(value["lineage"]) for value in payloads}
        fold_ids = {str(value["fold_id"]) for value in payloads}
        heldouts = {str(value.get("heldout_dataset") or "") for value in payloads}
        dataset_sets = {
            tuple(sorted(map(str, value["datasets"]))) for value in payloads
        }
        seeds = sorted({int(value["seed"]) for value in payloads})
        base = str(first["reference"]["base"])
        group_model = first["model"]
        if (
            len(lineages) != 1
            or len(fold_ids) != 1
            or len(heldouts) != 1
            or len(dataset_sets) != 1
            or not next(iter(heldouts))
            or len(seeds) < 2
        ):
            output["select_transferable"].append(
                _pending(
                    parent=parents[0],
                    model=group_model,
                    base=base,
                    seed=seeds[0],
                    reason=(
                        "selector requires one LODO lineage/fold/training-cohort "
                        "scope and at least two completed seeds"
                    ),
                    label="downstream_transferable_gp_selection_pending",
                )
            )
            continue
        references = [
            fit
            for state in states
            for (gp, fine, weighting), fit in state["fits"].items()
            if weighting == "donor_pooled"
        ]
        selection_dir = f"{base}/selection/{_slug(group)}"
        command = [
            "python",
            "-m",
            "immune_health.cli",
            "select-transferable-tripso-gps",
        ]
        for fit in references:
            command.extend(
                ["--reference-manifest", f"{fit['output_dir']}/healthy_reference.json"]
            )
        command.extend(
            [
                "--lineage",
                next(iter(lineages)),
                "--fold-id",
                next(iter(fold_ids)),
                "--heldout-dataset",
                next(iter(heldouts)),
            ]
        )
        for dataset in next(iter(dataset_sets)):
            command.extend(["--required-training-dataset", dataset])
        for seed in seeds:
            command.extend(["--required-seed", str(seed)])
        command.extend(
            ["--weighting-scheme", "donor_pooled", "--output-dir", selection_dir]
        )
        selector = _job(
            job_id=f"downstream-select-{_slug(group)}",
            stage="downstream_select_transferable_gps",
            parent=parents[0],
            model=group_model,
            output_dir=selection_dir,
            seed=seeds[0],
            command=command,
            upstream=[
                plan_upstream,
                *[
                    {
                        "path": f"{fit['output_dir']}/healthy_reference.json",
                        "json_require": {"schema_version": REFERENCE_SCHEMA},
                    }
                    for fit in references
                ],
            ],
            expected=[
                f"{selection_dir}/tripso_gp_cohort_seed_effects.parquet",
                f"{selection_dir}/tripso_gp_selection.parquet",
                f"{selection_dir}/selected_tripso_gps.json",
            ],
            depends_on=[str(fit["job_id"]) for fit in references],
            extra={
                "selector_input_role": "reference",
                "query_artifacts_allowed": False,
                "weighting_scheme": "donor_pooled",
            },
        )
        output["select_transferable"].append(selector)

    output["bootstrap_reference"].append(
        _pending(
            parent="global",
            model={"selection_group_id": "global", "lineage": None},
            base="${OUTPUT_ROOT}/downstream",
            seed=0,
            reason=(
                "reference bootstrap CLI requires exact one-row query artifacts; "
                "no row-slicing artifact was supplied"
            ),
            label="downstream_reference_bootstrap_pending",
        )
    )
    output["evaluate"].append(
        _pending(
            parent="global",
            model={"selection_group_id": "global", "lineage": None},
            base="${OUTPUT_ROOT}/downstream",
            seed=0,
            reason="evaluate-lodo requires an exact preassembled fold prediction table",
            label="downstream_evaluation_pending",
        )
    )
    if not output["empirical_scoring"]:
        output["empirical_scoring"].append(
            _pending(
                parent="global",
                model={"selection_group_id": "global", "lineage": None},
                base="${OUTPUT_ROOT}/downstream",
                seed=0,
                reason=(
                    "exact paired reference/validation endpoint and empirical-index "
                    "artifacts are not yet available"
                ),
                label="downstream_empirical_scoring_pending",
            )
        )
    return output


def generate_pass2(
    jobs: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
    selected_path: Path,
    allowlist: Mapping[str, Any],
    allowlist_path: Path,
    allowlist_file_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Generate locked outer-query work for selector-retained endpoints only."""

    selected = validate_tripso_gp_selection_manifest(selected_path)
    selected_file_sha = _file_hash(selected_path)
    if allowlist.get("selection_manifest_sha256") != selected_file_sha:
        raise ValueError("Query allowlist binds a different selector artifact")
    if (
        allowlist.get("selection_basis") != "inner_validation_only"
        or allowlist.get("outer_query_data_consulted_for_selection") is not False
        or allowlist.get("outer_query_evaluation_only") is not True
        or allowlist.get("outer_query_results_used_for_selection") is not False
        or allowlist.get("query_derived_evidence_used_for_selection") is not False
    ):
        raise ValueError("Query allowlist does not prove evaluation-only scope")
    selected_training_ids = allowlist.get("selected_training_job_ids")
    allowed = allowlist.get("allowed_parent_training_job_ids", selected_training_ids)
    if allowed != selected_training_ids:
        raise ValueError(
            "Downstream allowed_parent_training_job_ids must exactly equal the "
            "post-training selected_training_job_ids"
        )
    if (
        not isinstance(allowed, list)
        or not allowed
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError("Query allowlist parent IDs must be a non-empty unique list")
    output = _empty_output()
    projects = _project_index(jobs)
    models = _plan_models(plan)
    minimum_exact_sex_donors = _minimum_exact_sex_donors(plan)
    ontology = _validate_bound_file(
        plan.get("fine_type_universe"), "frozen fine-type universe"
    )
    plan_upstream = _plan_artifact(plan_path, plan_file_sha256)
    selection_artifact = {
        "path": str(selected_path.resolve()),
        "sha256": selected_file_sha,
        "json_require": {
            "schema_version": SELECTION_SCHEMA,
            "query_data_consulted": False,
            "reference_role_required": "reference",
        },
    }
    allowlist_artifact = {
        "path": str(allowlist_path.resolve()),
        "sha256": allowlist_file_sha256,
        "json_require": {
            "schema_version": ALLOWLIST_SCHEMA,
            "selection_basis": "inner_validation_only",
            "outer_query_data_consulted_for_selection": False,
            "outer_query_evaluation_only": True,
            "outer_query_results_used_for_selection": False,
        },
    }
    selected_endpoints = {
        (str(value["gp_id"]), str(value["fine_type"]))
        for value in selected["selected_endpoints"]
    }
    for parent in allowed:
        if parent not in models:
            raise ValueError(f"Allowlisted query model is absent from plan: {parent}")
        model = models[parent]
        if model.get("model_selection_status") != "selected":
            raise ValueError(f"Allowlisted query model is not selected: {parent}")
        query_job = projects.get((parent, "query"))
        base = (
            _base(query_job) if query_job is not None else "${OUTPUT_ROOT}/downstream"
        )
        seed = int(query_job.get("seed", 0)) if query_job is not None else 0
        if str(selected.get("lineage")) != str(model.get("lineage")):
            raise ValueError(f"Selector lineage differs from model {parent}")
        candidates = set(_candidate_endpoints(model))
        endpoints = sorted(candidates & selected_endpoints)
        if not endpoints:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason="selector retained no candidate endpoint for this model",
                    label="downstream_query_endpoint_pending",
                )
            )
            continue
        if query_job is None:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason="explicitly allowlisted query projection row is missing",
                    label="downstream_query_projection_pending",
                )
            )
            continue
        gp_ids = list(dict.fromkeys(gp for gp, _ in endpoints))
        query_state, reason = _projection_state(query_job, gp_ids)
        if query_state is None:
            output["pending"].append(
                _pending(
                    parent=parent,
                    model=model,
                    base=base,
                    seed=seed,
                    reason=str(reason),
                    label="downstream_query_projection_pending",
                )
            )
            continue
        query_payload = query_state["payload"]
        if (
            query_payload.get("projection_role") != "query"
            or query_payload.get("outer_query_evaluation_only") is not True
            or query_payload.get("reference_design") != "lodo"
            or str(query_payload.get("heldout_dataset"))
            != str(selected.get("heldout_dataset"))
            or str(query_payload.get("fold_id")) != str(selected.get("fold_id"))
            or int(query_payload.get("seed"))
            not in set(map(int, selected.get("required_seeds", ())))
        ):
            raise ValueError(
                f"Query projection scope differs from selector artifact: {parent}"
            )
        query = _conversion_and_endpoints(
            output=output,
            parent=parent,
            model=model,
            state=query_state,
            role="query",
            endpoints=endpoints,
            ontology=ontology,
            plan_upstream=plan_upstream,
        )
        fits: dict[tuple[str, str, str], dict[str, Any]] = {}
        for gp_id, fine_type in endpoints:
            for weighting in WEIGHTING_SCHEMES:
                endpoint_token = _slug(f"{gp_id}|{fine_type}")
                fit_dir = f"{query['base']}/references/{endpoint_token}/{weighting}"
                fits[(gp_id, fine_type, weighting)] = {
                    "job_id": (
                        f"downstream-fit-{weighting}-"
                        f"{_slug(f'{gp_id}|{fine_type}')}-{_slug(parent)}"
                    ),
                    "output_dir": fit_dir,
                }
        _score_role(
            output=output,
            role="query",
            parent=parent,
            model=model,
            target_artifacts=query,
            fits=fits,
            selected_artifact=selection_artifact,
            allowlist_artifact=allowlist_artifact,
        )
        # The same selected endpoints have exact reference/query empirical indices.
        reference_stub: dict[str, Any] = {
            "base": query["base"],
            "seed": query["seed"],
            "endpoints": {},
            "aggregates": {},
        }
        for gp_id, fine_type in endpoints:
            identity = (gp_id, fine_type)
            endpoint_token = _slug(f"{gp_id}|{fine_type}")
            reference_stub["endpoints"][identity] = {
                "job_id": (
                    f"downstream-endpoint-reference-{endpoint_token}-{_slug(parent)}"
                ),
                "output_dir": (f"{query['base']}/endpoints/reference/{endpoint_token}"),
            }
            gp_token = _slug(gp_id)
            reference_stub["aggregates"][gp_id] = {
                "job_id": (
                    f"downstream-aggregate-reference-{gp_token}-{_slug(parent)}"
                ),
                "output_dir": f"{query['base']}/aggregation/reference/{gp_token}",
            }
        _emit_empirical_scores(
            output=output,
            role="query",
            parent=parent,
            model=model,
            reference_artifacts=reference_stub,
            target_artifacts=query,
            minimum_exact_sex_donors=minimum_exact_sex_donors,
            scientific_settings_artifact=plan_upstream,
        )
    exact_evaluations = allowlist.get("evaluation_inputs", [])
    if exact_evaluations:
        if not isinstance(exact_evaluations, list):
            raise ValueError("evaluation_inputs must be a list")
        for index, value in enumerate(exact_evaluations):
            predictions, predictions_sha = _validate_bound_file(
                value.get("predictions") if isinstance(value, Mapping) else None,
                f"evaluation input {index}",
            )
            evaluation_dir = f"${{OUTPUT_ROOT}}/downstream/evaluation/{index}"
            output["evaluate"].append(
                _job(
                    job_id=f"downstream-evaluate-{index}",
                    stage="downstream_evaluate_lodo",
                    parent="global",
                    model={"selection_group_id": "global", "lineage": None},
                    output_dir=evaluation_dir,
                    seed=0,
                    command=[
                        "python",
                        "-m",
                        "immune_health.cli",
                        "evaluate-lodo",
                        "--predictions",
                        predictions,
                        "--output",
                        f"{evaluation_dir}/lodo_metrics.parquet",
                        "--wide-output",
                        f"{evaluation_dir}/lodo_metrics_wide.parquet",
                    ],
                    upstream=[
                        _static_artifact(predictions, predictions_sha),
                        selection_artifact,
                        allowlist_artifact,
                    ],
                    expected=[
                        f"{evaluation_dir}/lodo_metrics.parquet",
                        f"{evaluation_dir}/lodo_metrics_wide.parquet",
                    ],
                    depends_on=[],
                    extra={"outer_query_evaluation_only": True},
                )
            )
    else:
        output["evaluate"].append(
            _pending(
                parent="global",
                model={"selection_group_id": "global", "lineage": None},
                base="${OUTPUT_ROOT}/downstream",
                seed=0,
                reason="no exact preassembled fold prediction table was allowlisted",
                label="downstream_evaluation_pending",
            )
        )
    output["bootstrap_reference"].append(
        _pending(
            parent="global",
            model={"selection_group_id": "global", "lineage": None},
            base="${OUTPUT_ROOT}/downstream",
            seed=0,
            reason=(
                "no exact one-row query metadata/feature artifact was allowlisted "
                "for the current reference-bootstrap CLI"
            ),
            label="downstream_reference_bootstrap_pending",
        )
    )
    if not output["empirical_scoring"]:
        output["empirical_scoring"].append(
            _pending(
                parent="global",
                model={"selection_group_id": "global", "lineage": None},
                base="${OUTPUT_ROOT}/downstream",
                seed=0,
                reason=(
                    "exact selected reference/query endpoint and empirical-index "
                    "artifacts are not yet available"
                ),
                label="downstream_empirical_scoring_pending",
            )
        )
    return output


def _atomic_write(path: Path, content: str) -> None:
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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = list(rows)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in values
    )
    _atomic_write(path, content)
    return len(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="pass_number", choices=("1", "2"), required=True)
    parser.add_argument(
        "--projection-job-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--selected-gps", type=Path)
    parser.add_argument("--query-allowlist", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("slurm/manifests/downstream")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, plan_file_sha = _validate_self_hashed_json(
        args.candidate_plan, schema=PLAN_SCHEMA, label="candidate plan"
    )
    minimum_exact_sex_donors = _minimum_exact_sex_donors(plan)
    if (
        plan.get("selection_scope") != "training_or_inner_validation_only"
        or plan.get("outer_query_results_used_for_selection") is not False
    ):
        raise ValueError("Candidate plan does not prove training/inner-only selection")
    jobs = _read_project_jobs(args.projection_job_manifest)
    if args.pass_number == "1":
        if args.selected_gps is not None or args.query_allowlist is not None:
            raise ValueError("Pass 1 forbids query selection/allowlist inputs")
        generated = generate_pass1(
            jobs,
            plan=plan,
            plan_path=args.candidate_plan,
            plan_file_sha256=plan_file_sha,
        )
    else:
        if args.selected_gps is None or args.query_allowlist is None:
            raise ValueError("Pass 2 requires --selected-gps and --query-allowlist")
        allowlist, allowlist_file_sha = _validate_self_hashed_json(
            args.query_allowlist,
            schema=ALLOWLIST_SCHEMA,
            label="query allowlist",
        )
        generated = generate_pass2(
            jobs,
            plan=plan,
            plan_path=args.candidate_plan,
            plan_file_sha256=plan_file_sha,
            selected_path=args.selected_gps,
            allowlist=allowlist,
            allowlist_path=args.query_allowlist,
            allowlist_file_sha256=allowlist_file_sha,
        )
    summary = {
        "schema_version": "immune-health-downstream-job-summary/v1",
        "pass": int(args.pass_number),
        "candidate_plan": str(args.candidate_plan.resolve()),
        "candidate_plan_file_sha256": plan_file_sha,
        "minimum_exact_sex_donors": minimum_exact_sex_donors,
        "minimum_exact_sex_donors_source": {
            "path": str(args.candidate_plan.resolve()),
            "file_sha256": plan_file_sha,
            "json_pointer": "/healthy_reference/minimum_exact_sex_donors",
        },
        "counts": {name: len(rows) for name, rows in generated.items()},
        "runnable_counts": {
            name: sum(bool(row["runnable"]) for row in rows)
            for name, rows in generated.items()
        },
        "outer_query_work_emitted": bool(generated["score_query"]),
        "outer_query_evaluation_only": int(args.pass_number) == 2,
        "submitted": False,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    target = args.output_dir / f"pass{args.pass_number}"
    for name, rows in generated.items():
        _write_jsonl(target / f"{name}.jsonl", rows)
    _atomic_write(
        target / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
