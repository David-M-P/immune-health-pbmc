"""Load and combine the project's YAML configuration layers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

Config = dict[str, Any]

REQUIRED_SECTIONS = (
    "project",
    "annotation",
    "lineages",
    "aggregation",
    "healthy_reference",
    "validation",
    "cluster",
    "paths",
    "environment",
    "slurm",
    "run",
    "datasets",
    "sampling",
    "models",
    "reports",
)


def load_yaml(path: Path) -> Config:
    """Load one YAML mapping with a clear error for invalid top-level content."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return loaded


def _expand_environment(value: Any) -> Any:
    """Expand environment variables recursively without interpreting placeholders."""
    if isinstance(value, Mapping):
        return {key: _expand_environment(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_environment(child) for child in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_path_config(path: Path) -> Config:
    """Load a standalone config and remember its base directory for path resolution."""
    resolved_path = path.resolve()
    config = _expand_environment(load_yaml(resolved_path))
    config["_config_path"] = str(resolved_path)
    config["_config_dir"] = str(resolved_path.parent)
    return config


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a configured path relative to the declaring config file."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(str(config.get("_config_dir", Path.cwd())))
    return (config_dir / path).resolve()


def validate_reference_data_config(config: Mapping[str, Any]) -> None:
    """Validate the audited data contract, including the approved IDs."""
    required = {
        "paths",
        "identifiers",
        "metadata_fields",
        "counts",
        "genes",
        "datasets",
        "lineages",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Reference data configuration is missing sections: {missing}")
    identifiers = config["identifiers"]
    expected = {
        "biological_unit_id": "dataset::donor_id",
        "source_observation_id": "dataset::sample_id",
        "observation_id": "dataset::donor_id::sample_id",
    }
    mismatched = {
        key: (identifiers.get(key), value)
        for key, value in expected.items()
        if identifiers.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "Identifier contract differs from the approved audit decision: "
            f"{mismatched}"
        )
    primary = {
        name
        for name, entry in config["lineages"].items()
        if entry.get("role") == "primary"
    }
    expected_primary = {"B cells", "NK_ILC", "Monocytes", "CD4_like", "CD8_like"}
    if primary != expected_primary:
        raise ValueError(
            f"Primary lineage set differs from audited target set: {sorted(primary)}"
        )


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge mappings; later scalar and list values replace earlier ones."""
    merged: Config = deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = merge_configs(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_resolved_config(
    common_path: Path,
    cluster_path: Path,
    analysis_path: Path,
) -> Config:
    """Load common, cluster, and analysis YAML files in precedence order."""
    resolved = load_yaml(common_path)
    resolved = merge_configs(resolved, load_yaml(cluster_path))
    return merge_configs(resolved, load_yaml(analysis_path))


def validate_required_sections(config: Mapping[str, Any]) -> None:
    """Raise an error if a required configuration section is absent."""
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Resolved configuration is missing sections: {missing_text}")


def iter_placeholders(
    value: Any,
    location: tuple[str, ...] = (),
) -> Iterator[tuple[str, str]]:
    """Yield dotted locations and values for explicit angle-bracket placeholders."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_placeholders(child, (*location, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_placeholders(child, (*location, str(index)))
    elif isinstance(value, str) and "<" in value and ">" in value:
        yield ".".join(location), value
