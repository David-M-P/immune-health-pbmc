"""Safe, explicit serialization for frozen healthy trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from immune_health.healthy_reference.endpoint import validate_endpoint_inputs
from immune_health.healthy_reference.kernel import AgeKernelReference
from immune_health.healthy_reference.trajectory import HealthyTrajectory
from immune_health.provenance import atomic_write_json, sha256_file, stable_hash

REFERENCE_SCHEMA = "immune-health-frozen-healthy-reference/v1"
AGE_KERNEL_REFERENCE_SCHEMA = "immune-health-frozen-age-kernel-reference/v1"


def write_age_kernel_reference(
    model: AgeKernelReference,
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Serialize configuration while reusing the immutable endpoint arrays."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_arrays_path = output_dir / "age_kernel_reference_arrays.npz"
    if legacy_arrays_path.exists():
        raise FileExistsError(
            "Refusing to publish AgeKernelReference beside a legacy copied "
            f"covariance archive: {legacy_arrays_path}"
        )
    manifest_path = output_dir / "age_kernel_reference.json"
    endpoint = metadata.get("endpoint_artifact")
    if not isinstance(endpoint, Mapping):
        raise ValueError("AgeKernelReference requires a manifest-bound endpoint")
    required_endpoint_fields = {
        "manifest_path",
        "manifest_sha256",
        "covariances_npy_sha256",
        "shape",
        "covariance_shape",
        "feature_ids",
        "observation_id_ordered_sha256",
    }
    if not required_endpoint_fields.issubset(endpoint):
        raise ValueError("AgeKernelReference endpoint binding is incomplete")
    if list(model.locations_.shape) != list(endpoint["shape"]) or list(
        model.covariances_.shape
    ) != list(endpoint["covariance_shape"]):
        raise ValueError("AgeKernelReference arrays differ from endpoint dimensions")
    payload: dict[str, Any] = {
        "schema_version": AGE_KERNEL_REFERENCE_SCHEMA,
        "model_class": "AgeKernelReference",
        "hyperparameters": {
            "bandwidth": model.bandwidth,
            "minimum_exact_sex_donors": model.minimum_exact_sex_donors,
            "weighting_scheme": model.weighting_scheme,
            "age_grid_size": model.age_grid_size,
        },
        "fitted_state": {
            "shape": list(model.locations_.shape),
            "covariance_shape": list(model.covariances_.shape),
            "age_range": list(model.age_range_),
            "datasets": sorted(np.unique(model.datasets_).tolist()),
            "n_biological_units": len(model.training_biological_units_),
        },
        "covariance_semantics": {
            "query": "within-cell donor GP covariance from endpoint_covariances.npy",
            "reference": (
                "age-kernel mixture of within-cell donor GP covariances plus "
                "between-donor location covariance"
            ),
            "spline_residual_covariance_used": False,
        },
        "storage_contract": {
            "locations": "reuse_endpoint_locations_npy_by_hash",
            "covariances": "reuse_endpoint_covariances_npy_by_hash",
            "copied_covariance_archive_written": False,
        },
        **dict(metadata),
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_write_json(manifest_path, payload)
    return manifest_path


def load_age_kernel_reference(
    manifest_path: Path,
) -> tuple[AgeKernelReference, dict[str, Any]]:
    """Load and fully revalidate a frozen donor-distribution reference."""

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != AGE_KERNEL_REFERENCE_SCHEMA:
        raise ValueError(f"Unsupported age-kernel manifest: {manifest_path}")
    content = dict(manifest)
    claimed = content.pop("manifest_sha256", None)
    if claimed != stable_hash(content):
        raise ValueError("Age-kernel manifest content hash does not match")
    endpoint = manifest.get("endpoint_artifact")
    if not isinstance(endpoint, Mapping):
        raise ValueError("Age-kernel manifest lacks its immutable endpoint binding")
    endpoint_manifest_path = Path(str(endpoint.get("manifest_path", "")))
    if not endpoint_manifest_path.is_file():
        raise FileNotFoundError(
            f"Age-kernel endpoint manifest is missing: {endpoint_manifest_path}"
        )
    with endpoint_manifest_path.open(encoding="utf-8") as handle:
        endpoint_manifest = json.load(handle)
    metadata_path = endpoint_manifest_path.parent / str(
        endpoint_manifest.get("metadata_path", "")
    )
    locations_path = endpoint_manifest_path.parent / str(
        endpoint_manifest.get("features_path", "")
    )
    endpoint_validation = validate_endpoint_inputs(
        endpoint_manifest_path,
        metadata_path,
        locations_path,
        expected_role="reference",
    )
    for key in (
        "manifest_sha256",
        "covariances_npy_sha256",
        "observation_id_ordered_sha256",
    ):
        if endpoint_validation.get(key) != endpoint.get(key):
            raise ValueError(f"Age-kernel endpoint binding changed: {key}")
    endpoint_metadata = pd.read_parquet(metadata_path)
    locations = np.load(locations_path, mmap_mode="r", allow_pickle=False)
    covariances = np.load(
        endpoint_validation["covariances_path"], mmap_mode="r", allow_pickle=False
    )
    model = AgeKernelReference(**manifest["hyperparameters"]).fit(
        locations,
        covariances,
        endpoint_metadata["age"],
        endpoint_metadata["sex"],
        endpoint_metadata["biological_unit_id"],
        datasets=endpoint_metadata["dataset"],
    )
    if list(model.locations_.shape) != manifest.get("fitted_state", {}).get("shape"):
        raise ValueError("Age-kernel fitted shape differs from its manifest")
    return model, manifest


def write_reference(
    model: HealthyTrajectory,
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Serialize numeric state to NPZ and declarative metadata to JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "healthy_reference_arrays.npz"
    manifest_path = output_dir / "healthy_reference.json"
    np.savez_compressed(
        arrays_path,
        coefficients=np.asarray(model.coefficients_, dtype=float),
        residual_covariance=np.asarray(model.residual_covariance_, dtype=float),
        age_direction=np.asarray(model.age_direction_, dtype=float),
        training_biological_units=np.asarray(
            sorted(model.training_biological_units_), dtype=str
        ),
        training_biological_unit_rows=np.asarray(
            model.training_biological_unit_rows_, dtype=str
        ),
        training_ages=np.asarray(model.training_ages_, dtype=float),
        training_sexes=np.asarray(model.training_sexes_, dtype=str),
        training_datasets=(
            np.asarray([], dtype=str)
            if model.training_datasets_ is None
            else np.asarray(model.training_datasets_, dtype=str)
        ),
        training_row_weights=np.asarray(model.training_row_weights_, dtype=float),
    )
    payload = {
        "schema_version": REFERENCE_SCHEMA,
        "arrays_path": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "model_class": "HealthyTrajectory",
        "hyperparameters": {
            "n_spline_knots": model.n_spline_knots,
            "ridge": model.ridge,
            "age_grid_size": model.age_grid_size,
            "covariance_regularization": model.covariance_regularization,
            "weighting_scheme": model.weighting_scheme,
        },
        "fitted_state": {
            "age_mean": model.age_mean_,
            "age_scale": model.age_scale_,
            "age_range": list(model.age_range_),
            "knots": list(model.knots_),
            "sex_levels": list(model.sex_levels_),
            "dataset_levels": list(model.dataset_levels_),
            "dataset_proportions": dict(model.dataset_proportions_),
            "training_weight_summary": dict(model.training_weight_summary_),
            "n_features": model.n_features_in_,
        },
        **dict(metadata),
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path, arrays_path


def load_reference(manifest_path: Path) -> tuple[HealthyTrajectory, dict[str, Any]]:
    """Load a frozen trajectory without executing a pickle payload."""

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError(f"Unsupported healthy-reference manifest: {manifest_path}")
    arrays_path = manifest_path.parent / manifest["arrays_path"]
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Healthy-reference arrays are missing: {arrays_path}")
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError("Healthy-reference array hash does not match its manifest")
    archive = np.load(arrays_path, allow_pickle=False)
    hyperparameters = manifest["hyperparameters"]
    state = manifest["fitted_state"]
    model = HealthyTrajectory(**hyperparameters)
    model.age_mean_ = float(state["age_mean"])
    model.age_scale_ = float(state["age_scale"])
    model.age_range_ = tuple(map(float, state["age_range"]))
    model.knots_ = tuple(map(float, state["knots"]))
    model.sex_levels_ = tuple(map(str, state["sex_levels"]))
    model.dataset_levels_ = tuple(map(str, state["dataset_levels"]))
    model.dataset_proportions_ = {
        str(key): float(value) for key, value in state["dataset_proportions"].items()
    }
    model.training_weight_summary_ = {
        str(key): float(value)
        for key, value in state.get("training_weight_summary", {}).items()
    }
    model.n_features_in_ = int(state["n_features"])
    model.coefficients_ = np.asarray(archive["coefficients"], dtype=float)
    model.residual_covariance_ = np.asarray(archive["residual_covariance"], dtype=float)
    model.age_direction_ = np.asarray(archive["age_direction"], dtype=float)
    model.training_biological_units_ = frozenset(
        archive["training_biological_units"].astype(str).tolist()
    )
    support_arrays = {
        "training_biological_unit_rows",
        "training_ages",
        "training_sexes",
        "training_datasets",
        "training_row_weights",
    }
    if support_arrays.issubset(archive.files):
        model.training_biological_unit_rows_ = np.asarray(
            archive["training_biological_unit_rows"], dtype=str
        )
        model.training_ages_ = np.asarray(archive["training_ages"], dtype=float)
        model.training_sexes_ = np.asarray(archive["training_sexes"], dtype=str)
        stored_datasets = np.asarray(archive["training_datasets"], dtype=str)
        model.training_datasets_ = stored_datasets if len(stored_datasets) else None
        model.training_row_weights_ = np.asarray(
            archive["training_row_weights"], dtype=float
        )
    else:
        model.training_biological_unit_rows_ = None
        model.training_ages_ = None
        model.training_sexes_ = None
        model.training_datasets_ = None
        model.training_row_weights_ = None
    return model, manifest
