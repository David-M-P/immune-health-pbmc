"""Load and combine the project's YAML configuration layers."""

from __future__ import annotations

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
