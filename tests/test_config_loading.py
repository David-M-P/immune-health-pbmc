from pathlib import Path

import pytest

from immune_health.config import (
    iter_placeholders,
    load_resolved_config,
    load_yaml,
    merge_configs,
    validate_required_sections,
)

REPOSITORY_ROOT = Path(__file__).parents[1]

REQUESTED_CONFIGS = (
    "configs/common.yaml",
    "configs/clusters/gefion.yaml",
    "configs/clusters/genomedk.yaml",
    "configs/analyses/development.yaml",
    "configs/analyses/full_healthy_reference.yaml",
    "configs/annotation/celltypist_v1.yaml",
    "configs/datasets/example_dataset.yaml",
)

SCIENTIFIC_AND_ANALYSIS_SECTIONS = (
    "project",
    "annotation",
    "lineages",
    "aggregation",
    "healthy_reference",
    "validation",
    "run",
    "datasets",
    "sampling",
    "models",
    "reports",
)

ANNOTATION_COLUMNS = {
    "cell_id",
    "analysis_lineage",
    "fine_cell_type",
    "annotation_method",
    "annotation_version",
    "confidence",
    "second_best_label",
    "decision_margin",
    "annotation_status",
}

DATASET_METADATA_FIELDS = {
    "cell_id",
    "library_id",
    "sample_id",
    "donor_id",
    "age",
    "sex",
    "healthy_status",
}


@pytest.mark.parametrize("relative_path", REQUESTED_CONFIGS)
def test_requested_config_is_a_yaml_mapping(relative_path: str) -> None:
    config = load_yaml(REPOSITORY_ROOT / relative_path)
    assert config


def test_nested_configuration_values_merge_explicitly() -> None:
    base = {"run": {"mode": "development", "seed": 42}, "datasets": ["a"]}
    override = {"run": {"seed": 7}, "datasets": ["b"]}

    assert merge_configs(base, override) == {
        "run": {"mode": "development", "seed": 7},
        "datasets": ["b"],
    }
    assert base["run"]["seed"] == 42


@pytest.mark.parametrize("cluster_name", ("gefion", "genomedk"))
def test_development_configuration_resolves_for_each_cluster(
    cluster_name: str,
) -> None:
    resolved = load_resolved_config(
        common_path=REPOSITORY_ROOT / "configs/common.yaml",
        cluster_path=REPOSITORY_ROOT / f"configs/clusters/{cluster_name}.yaml",
        analysis_path=REPOSITORY_ROOT / "configs/analyses/development.yaml",
    )

    validate_required_sections(resolved)
    assert resolved["cluster"]["name"] == cluster_name
    assert resolved["annotation"]["version"] == "lineage_v1_celltypist"
    assert resolved["aggregation"]["biological_unit"] == "donor"


def test_cluster_placeholders_are_visible_and_gefion_account_is_known() -> None:
    gefion = load_yaml(REPOSITORY_ROOT / "configs/clusters/gefion.yaml")
    genomedk = load_yaml(REPOSITORY_ROOT / "configs/clusters/genomedk.yaml")

    assert gefion["cluster"]["project_account"] == "cu_0071"
    assert list(iter_placeholders(gefion))
    assert list(iter_placeholders(genomedk))


def test_clusters_resolve_to_the_same_scientific_configuration() -> None:
    resolved_by_cluster = {
        cluster_name: load_resolved_config(
            common_path=REPOSITORY_ROOT / "configs/common.yaml",
            cluster_path=REPOSITORY_ROOT / f"configs/clusters/{cluster_name}.yaml",
            analysis_path=REPOSITORY_ROOT / "configs/analyses/development.yaml",
        )
        for cluster_name in ("gefion", "genomedk")
    }

    gefion_science = {
        key: resolved_by_cluster["gefion"][key]
        for key in SCIENTIFIC_AND_ANALYSIS_SECTIONS
    }
    genomedk_science = {
        key: resolved_by_cluster["genomedk"][key]
        for key in SCIENTIFIC_AND_ANALYSIS_SECTIONS
    }
    assert gefion_science == genomedk_science


def test_celltypist_profile_declares_the_annotation_contract() -> None:
    config = load_yaml(REPOSITORY_ROOT / "configs/annotation/celltypist_v1.yaml")

    assert config["annotation"]["method"] == "celltypist"
    assert set(config["output_columns"]) == ANNOTATION_COLUMNS
    assert "unknown" in config["allowed_statuses"]


def test_example_dataset_declares_required_metadata_mappings() -> None:
    config = load_yaml(REPOSITORY_ROOT / "configs/datasets/example_dataset.yaml")

    assert set(config["metadata_columns"]) == DATASET_METADATA_FIELDS
    assert config["counts"]["source"] == "<COUNTS_LAYER_OR_X>"
