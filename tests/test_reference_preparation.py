from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from immune_health.data.reference_preparation import (
    ReferenceFeatureConfig,
    annotate_fold_cell_roles,
    build_all_healthy_reference_fold,
    build_terekhova_one_visit_manifest,
    materialize_fold_h5ad,
    materialize_frozen_query_h5ad,
    prepare_fold_features,
)

APPROVED_FINE_TYPE_ONTOLOGY = (
    Path(__file__).parents[1] / "configs" / "data" / "fine_type_ontology.approved.yaml"
)


def _selection_hash(seed: int, biological: str, observation: str) -> str:
    return hashlib.sha256(f"{seed}::{biological}::{observation}".encode()).hexdigest()


def test_terekhova_visit_choice_is_seeded_hash_not_age_or_row_order() -> None:
    donors = pd.DataFrame(
        {
            "dataset": ["terekhova", "terekhova"],
            "donor_id": ["d1", "d2"],
            "biological_unit_id": ["terekhova::d1", "terekhova::d2"],
            "sample_ids": ["late|early|middle", "only"],
            "observation_ids": [
                "terekhova::d1::late|terekhova::d1::early|terekhova::d1::middle",
                "terekhova::d2::only",
            ],
            # These ages must not enter the selection algorithm.
            "age_min": [20, 80],
            "age_max": [22, 80],
        }
    )
    first = build_terekhova_one_visit_manifest(donors, seed=17)
    second = build_terekhova_one_visit_manifest(donors.iloc[::-1], seed=17)
    selected_first = first.loc[
        first["selected_for_reference"], "observation_id"
    ].tolist()
    selected_second = second.loc[
        second["selected_for_reference"], "observation_id"
    ].tolist()
    assert selected_first == selected_second
    expected_d1 = min(
        ("terekhova::d1::late", "terekhova::d1::early", "terekhova::d1::middle"),
        key=lambda value: (_selection_hash(17, "terekhova::d1", value), value),
    )
    assert expected_d1 in selected_first
    assert (
        first.groupby("biological_unit_id")["selected_for_reference"].sum().eq(1).all()
    )


def test_fold_roles_trim_terekhova_and_allow_explicit_longitudinal_query() -> None:
    obs = pd.DataFrame(
        {
            "dataset": ["terekhova", "terekhova", "aidav2"],
            "donor_id": ["d1", "d1", "a1"],
            "sample_id": ["v1", "v2", "s1"],
            "age": [40, 41, 50],
            "sex": ["female", "female", "male"],
        },
        index=["cell-a", "cell-b", "cell-c"],
    )
    visits = build_terekhova_one_visit_manifest(obs, seed=3)
    reference_fold = pd.DataFrame(
        {
            "fold_id": ["lodo_aidav2"] * 2,
            "heldout_dataset": ["aidav2"] * 2,
            "dataset": ["terekhova", "aidav2"],
            "biological_unit_id": ["terekhova::d1", "aidav2::a1"],
            "outer_role": ["reference", "query"],
            "inner_fold": [0, pd.NA],
        }
    )
    annotated = annotate_fold_cell_roles(obs, obs.index, reference_fold, visits)
    terekhova_roles = annotated.loc[
        annotated["dataset"].eq("terekhova"), "preparation_role"
    ]
    assert terekhova_roles.eq("adaptation").sum() == 1
    assert terekhova_roles.eq("excluded_nonselected_visit").sum() == 1
    assert (
        annotated.loc[annotated["dataset"].eq("aidav2"), "preparation_role"]
        .eq("query")
        .all()
    )

    query_fold = reference_fold.copy()
    query_fold["heldout_dataset"] = "terekhova"
    query_fold["outer_role"] = ["query", "reference"]
    annotated_query = annotate_fold_cell_roles(obs, obs.index, query_fold, visits)
    # Production keeps one visit even for a LODO query so donors are not
    # replicated. Retaining all visits is an explicit longitudinal sensitivity.
    assert (
        annotated_query.loc[
            annotated_query["dataset"].eq("terekhova"), "preparation_role"
        ]
        .eq("query")
        .sum()
        == 1
    )
    annotated_longitudinal = annotate_fold_cell_roles(
        obs,
        obs.index,
        query_fold,
        visits,
        global_one_visit_query=False,
    )
    assert (
        annotated_longitudinal.loc[
            annotated_longitudinal["dataset"].eq("terekhova"), "preparation_role"
        ]
        .eq("query")
        .all()
    )


def test_all_healthy_fold_has_no_query_and_supports_optional_inner_validation() -> None:
    datasets = ("a", "b", "c", "d", "terekhova")
    donors = pd.DataFrame(
        {
            "dataset": list(datasets),
            "donor_id": [f"d{index}" for index in range(5)],
            "biological_unit_id": [
                f"{dataset}::d{index}" for index, dataset in enumerate(datasets)
            ],
            "global_inner_fold": [0, 1, 0, 1, 0],
        }
    )
    final = build_all_healthy_reference_fold(donors, healthy_datasets=datasets)
    assert final["heldout_dataset"].isna().all()
    assert set(final["outer_role"]) == {"reference"}
    assert set(final["reference_partition"]) == {"adaptation"}
    assert final["eligible_for_reference_fitting"].all()

    selected = build_all_healthy_reference_fold(
        donors,
        healthy_datasets=datasets,
        inner_validation_fold=1,
    )
    assert set(selected["reference_partition"]) == {"adaptation", "validation"}
    assert (
        selected.loc[selected["inner_fold"].eq(1), "eligible_for_reference_fitting"]
        .eq(False)
        .all()
    )
    obs = pd.DataFrame(
        {
            "dataset": list(datasets),
            "donor_id": donors["donor_id"].tolist(),
            "sample_id": ["s1"] * 5,
            "age": [30, 40, 50, 60, 70],
            "sex": ["female", "male", "female", "male", "female"],
        },
        index=[f"cell-{index}" for index in range(5)],
    )
    visits = build_terekhova_one_visit_manifest(obs, seed=42)
    roles = annotate_fold_cell_roles(
        obs,
        obs.index,
        selected,
        visits,
        reference_design="all_healthy",
    )
    expected_validation = set(
        selected.loc[
            selected["reference_partition"].eq("validation"), "biological_unit_id"
        ]
    )
    observed_validation = set(
        roles.loc[roles["preparation_role"].eq("validation"), "biological_unit_id"]
    )
    assert observed_validation == expected_validation
    assert not roles["preparation_role"].eq("query").any()


def _tiny_fold_input(path: Path, query_multiplier: int) -> tuple[Path, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    matrices: list[np.ndarray] = []
    cell_number = 0
    for dataset, role in (
        ("train_a", "reference"),
        ("train_b", "reference"),
        ("held", "query"),
    ):
        for donor_number in range(2):
            donor = f"{dataset}_d{donor_number}"
            for replicate in range(2):
                rows.append(
                    {
                        "dataset": dataset,
                        "donor_id": donor,
                        "sample_id": "visit1",
                        "age": 30 + donor_number,
                        "sex": "female" if donor_number == 0 else "male",
                        "lineage": "B cells",
                        "ctype_low": "Naive B cells",
                        "ctype_low_conf": 0.95,
                    }
                )
                base = np.array(
                    [
                        3 + donor_number + replicate,
                        1 + (dataset == "train_b") * 3,
                        8 * donor_number + replicate,
                        2,
                        1 + replicate,
                        4 + (dataset == "train_a") * replicate,
                    ],
                    dtype=float,
                )
                if role == "query":
                    base = np.array([1, query_multiplier, query_multiplier, 1, 1, 1])
                matrices.append(base)
                cell_number += 1
    obs = pd.DataFrame(rows, index=[f"cell-{index}" for index in range(cell_number)])
    genes = [f"ENSG{index}" for index in range(6)]
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.vstack(matrices)),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    adata.write_h5ad(path)
    fold = pd.DataFrame(
        {
            "fold_id": ["lodo_held"] * 6,
            "heldout_dataset": ["held"] * 6,
            "dataset": ["train_a", "train_a", "train_b", "train_b", "held", "held"],
            "biological_unit_id": [
                "train_a::train_a_d0",
                "train_a::train_a_d1",
                "train_b::train_b_d0",
                "train_b::train_b_d1",
                "held::held_d0",
                "held::held_d1",
            ],
            "outer_role": ["reference"] * 4 + ["query"] * 2,
            "inner_fold": [0, 1, 0, 1, pd.NA, pd.NA],
        }
    )
    return path, fold


def test_fold_features_ignore_query_and_union_program_genes(tmp_path: Path) -> None:
    first_h5ad, fold = _tiny_fold_input(tmp_path / "first.h5ad", 10)
    second_h5ad, _ = _tiny_fold_input(tmp_path / "second.h5ad", 100_000)
    # The global split can legitimately contain a donor with zero cells in this
    # lineage. It must remain audited without becoming an expected H5AD/Arrow donor.
    fold = pd.concat(
        [
            fold,
            pd.DataFrame(
                [
                    {
                        "fold_id": "lodo_held",
                        "heldout_dataset": "held",
                        "dataset": "train_a",
                        "biological_unit_id": "train_a::lineage_absent",
                        "outer_role": "reference",
                        "inner_fold": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    programs = tmp_path / "programs.gmt"
    programs.write_text(
        "GP_A\ttest\tENSG0\tENSG4\nGP_B\ttest\tENSG1\tENSG5\n",
        encoding="utf-8",
    )
    empty_visits = pd.DataFrame(columns=["observation_id", "selected_for_reference"])
    config = ReferenceFeatureConfig(
        hvg_sizes=(2, 4),
        hvg_mean_bins=2,
        hvg_minimum_donor_fraction=0.0,
        hvg_minimum_dataset_fraction=1.0,
        gp_minimum_mapped_genes=1,
        gp_maximum_program_size=10,
        gp_minimum_expression_coverage=0.0,
        gp_minimum_donor_coverage=0.0,
        gp_minimum_dataset_fraction=1.0,
        gp_redundancy_jaccard_threshold=1.0,
        gp_projection_control_ids=("GP_A",),
    )
    outputs = []
    for number, h5ad in enumerate((first_h5ad, second_h5ad)):
        output = tmp_path / f"features-{number}"
        prepare_fold_features(
            h5ad,
            fold,
            empty_visits,
            programs,
            output,
            lineage="B cells",
            fine_type_ontology_path=APPROVED_FINE_TYPE_ONTOLOGY,
            config=config,
            chunk_size=3,
            inner_validation_fold=0,
        )
        outputs.append(output)
    assert (outputs[0] / "hvg4.txt").read_text() == (
        outputs[1] / "hvg4.txt"
    ).read_text()
    hvg2 = (outputs[0] / "hvg2.txt").read_text().splitlines()
    hvg4 = (outputs[0] / "hvg4.txt").read_text().splitlines()
    assert hvg2 == hvg4[:2]
    model_genes = set((outputs[0] / "model_genes_hvg4.txt").read_text().splitlines())
    assert {"ENSG0", "ENSG1", "ENSG4", "ENSG5"} <= model_genes
    manifest = json.loads((outputs[0] / "feature_manifest.json").read_text())
    assert manifest["cell_downsampling_performed"] is False
    assert manifest["inner_validation_fold"] == 0
    assert manifest["cell_counts_by_preparation_role"]["adaptation"] == 4
    assert manifest["cell_counts_by_preparation_role"]["validation"] == 4
    assert manifest["cell_counts_by_preparation_role"]["query"] == 4
    lineage_scope = manifest["lineage_donor_scope"]
    assert lineage_scope[
        "global_fold_biological_unit_ids_without_materialized_role_cells"
    ] == ["train_a::lineage_absent"]
    assert lineage_scope["n_biological_units_by_preparation_role"] == {
        "adaptation": 2,
        "validation": 2,
        "query": 2,
    }
    candidates = json.loads((outputs[0] / "projection_gp_candidates.json").read_text())
    assert candidates["selection_level"] == "donor_lineage_pseudobulk"
    assert candidates["purpose"] == "training_only_projection_storage_gate"
    assert candidates["query_data_consulted"] is False
    assert candidates["program_ids"] == ["GP_A"]
    assert candidates["zero_candidate_policy"] == "fail_closed"
    assert (outputs[0] / "projection_gp_candidates.tsv").read_bytes() == (
        outputs[1] / "projection_gp_candidates.tsv"
    ).read_bytes()

    materialized_results: dict[int, ad.AnnData] = {}
    for hvg_size in (2, 4):
        materialized = tmp_path / f"adaptation_hvg{hvg_size}.h5ad"
        materialize_fold_h5ad(
            first_h5ad,
            outputs[0],
            materialized,
            role="adaptation",
            hvg_size=hvg_size,
            row_chunk_size=3,
            max_loaded_elements=100,
        )
        materialization_manifest = json.loads(
            materialized.with_suffix(".manifest.json").read_text()
        )
        assert materialization_manifest["lineage_donor_scope"] == lineage_scope
        result = ad.read_h5ad(materialized)
        materialized_results[hvg_size] = result
        assert result.n_obs == 4
        expected_genes = (
            (outputs[0] / f"model_genes_hvg{hvg_size}.txt").read_text().splitlines()
        )
        assert result.var_names.tolist() == expected_genes
        assert result.obs["cell_key"].is_unique
        expected_keys = (
            result.obs["dataset"].astype(str)
            + "::"
            + result.obs["source_cell_id"].astype(str)
        )
        assert (result.obs["cell_key"] == expected_keys).all()
        assert (result.obs["fine_type"] == result.obs["ctype_low"]).all()
        assert set(result.obs["preparation_role"]) == {"adaptation"}
        assert result.var["ensembl_id"].tolist() == expected_genes
        source = ad.read_h5ad(first_h5ad)
        expected_full_library_sizes = np.asarray(
            source[result.obs["source_cell_id"].astype(str).tolist()].X.sum(axis=1)
        ).reshape(-1)
        np.testing.assert_array_equal(
            result.obs["n_counts"].to_numpy(), expected_full_library_sizes
        )
        assert sparse.issparse(result.X)

    validation_path = tmp_path / "validation_hvg2.h5ad"
    materialize_fold_h5ad(
        first_h5ad,
        outputs[0],
        validation_path,
        role="validation",
        hvg_size=2,
        row_chunk_size=3,
        max_loaded_elements=100,
    )
    validation = ad.read_h5ad(validation_path)
    assert validation.n_obs == 4
    assert set(validation.obs["biological_unit_id"].astype(str)) == {
        "train_a::train_a_d0",
        "train_b::train_b_d0",
    }
    assert materialized_results[2].obs_names.tolist() == (
        materialized_results[4].obs_names.tolist()
    )


def test_projection_candidate_prefilter_fails_closed_without_candidates(
    tmp_path: Path,
) -> None:
    input_h5ad, fold = _tiny_fold_input(tmp_path / "zero_candidates.h5ad", 10)
    programs = tmp_path / "programs.gmt"
    programs.write_text("GP_A\ttest\tENSG0\tENSG4\n", encoding="utf-8")
    config = ReferenceFeatureConfig(
        hvg_sizes=(2,),
        hvg_mean_bins=2,
        hvg_minimum_donor_fraction=0.0,
        hvg_minimum_dataset_fraction=1.0,
        gp_minimum_mapped_genes=1,
        gp_maximum_program_size=10,
        gp_minimum_expression_coverage=0.0,
        gp_minimum_donor_coverage=0.0,
        gp_minimum_dataset_fraction=1.0,
        gp_redundancy_jaccard_threshold=1.0,
        # Defaults require evidence from three cohorts; this fixture has two.
        gp_projection_control_ids=(),
    )
    with pytest.raises(ValueError, match="retained zero programs"):
        prepare_fold_features(
            input_h5ad,
            fold,
            pd.DataFrame(columns=["observation_id", "selected_for_reference"]),
            programs,
            tmp_path / "blocked_features",
            lineage="B cells",
            fine_type_ontology_path=APPROVED_FINE_TYPE_ONTOLOGY,
            config=config,
            chunk_size=3,
        )


def test_all_healthy_features_use_every_reference_donor_without_heldout(
    tmp_path: Path,
) -> None:
    input_h5ad, fold = _tiny_fold_input(tmp_path / "all_healthy.h5ad", 7)
    fold["reference_design"] = "all_healthy"
    fold["fold_id"] = "all_healthy"
    fold["heldout_dataset"] = pd.NA
    fold["outer_role"] = "reference"
    fold["eligible_for_reference_fitting"] = True
    programs = tmp_path / "all_healthy_programs.gmt"
    programs.write_text(
        "GP_A\ttest\tENSG0\tENSG4\nGP_B\ttest\tENSG1\tENSG5\n",
        encoding="utf-8",
    )
    config = ReferenceFeatureConfig(
        hvg_sizes=(2, 4),
        hvg_mean_bins=2,
        hvg_minimum_donor_fraction=0.0,
        hvg_minimum_dataset_fraction=1.0,
        gp_minimum_mapped_genes=1,
        gp_maximum_program_size=10,
        gp_minimum_expression_coverage=0.0,
        gp_minimum_donor_coverage=0.0,
        gp_minimum_dataset_fraction=1.0,
        gp_redundancy_jaccard_threshold=1.0,
        gp_projection_control_ids=("GP_A",),
    )
    output = tmp_path / "final_features"
    manifest = prepare_fold_features(
        input_h5ad,
        fold,
        pd.DataFrame(columns=["observation_id", "selected_for_reference"]),
        programs,
        output,
        lineage="B cells",
        fine_type_ontology_path=APPROVED_FINE_TYPE_ONTOLOGY,
        reference_design="all_healthy",
        config=config,
        chunk_size=3,
    )
    assert manifest["reference_design"] == "all_healthy"
    assert manifest["heldout_dataset"] is None
    assert manifest["training_datasets"] == ["held", "train_a", "train_b"]
    assert manifest["cell_counts_by_preparation_role"] == {"adaptation": 12}

    materialized = tmp_path / "all_healthy_adaptation.h5ad"
    result = materialize_fold_h5ad(
        input_h5ad,
        output,
        materialized,
        role="adaptation",
        hvg_size=4,
        row_chunk_size=3,
        max_loaded_elements=100,
    )
    assert result["reference_design"] == "all_healthy"
    assert result["heldout_dataset"] is None
    assert result["shape"][0] == 12

    frozen_genes = (output / "model_genes_hvg4.txt").read_text().splitlines()
    query_genes = frozen_genes[:-1]
    query = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, len(query_genes)), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "dataset": ["future", "future"],
                "donor_id": ["q1", "q2"],
                "sample_id": ["s1", "s1"],
                "age": [44, 55],
                "sex": ["female", "male"],
                "lineage": ["B cells", "B cells"],
                "ctype_low": ["Naive B cells", "Naive B cells"],
                "ctype_low_conf": [0.95, 0.60],
            },
            index=["query-cell-1", "query-cell-2"],
        ),
        var=pd.DataFrame(index=query_genes),
    )
    query_path = tmp_path / "future_query.h5ad"
    query.write_h5ad(query_path)
    query_output = tmp_path / "future_query_frozen.h5ad"
    query_manifest = materialize_frozen_query_h5ad(
        query_path,
        output,
        query_output,
        hvg_size=4,
        lineage="B cells",
        minimum_gene_coverage=0.5,
        row_chunk_size=1,
        max_loaded_elements=100,
    )
    mapped = ad.read_h5ad(query_output)
    assert mapped.var_names.tolist() == frozen_genes
    assert mapped.obs["cell_key"].tolist() == [
        "future::query-cell-1",
        "future::query-cell-2",
    ]
    assert mapped.obs["fine_type"].astype(str).tolist() == [
        "Naive B cells",
        "low_confidence",
    ]
    assert mapped.obs["fine_type_state_eligible"].tolist() == [True, False]
    assert np.asarray(mapped.X[:, -1].todense()).sum() == 0
    assert query_manifest["feature_selection_on_query_performed"] is False
    assert query_manifest["n_frozen_genes_missing_from_query"] == 1
