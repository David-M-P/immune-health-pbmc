from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from immune_health.data.ontology import (
    apply_fine_type_ontology,
    generate_candidate_ontology,
    load_fine_type_ontology,
    summarize_fine_type_labels,
    write_candidate_ontology,
)
from immune_health.gene_programs import (
    AmbiguousGeneMappingError,
    GeneProgram,
    GPFilterConfig,
    compute_training_gene_statistics,
    filter_gene_programs_training_only,
    load_gene_programs,
    map_ensembl_to_symbols,
    select_hvgs_training_only,
    synthetic_gene_programs,
    validate_gp_resource,
    write_synthetic_gp_library,
)
from immune_health.sampling import HierarchicalCellSampler
from immune_health.splits import (
    REFERENCE_DATASETS,
    add_stable_identifiers,
    assert_lodo_integrity,
    assert_partition_disjoint,
    build_global_donor_manifest,
    build_lodo_tables,
    write_lodo_manifests,
)


def _split_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(REFERENCE_DATASETS):
        for donor_index in range(8):
            donor = f"d{donor_index}"
            samples = (
                (f"visit{donor_index}_a", f"visit{donor_index}_b")
                if dataset == "terekhova" and donor_index < 2
                else ("shared_pool",)
                if dataset == "onek1k"
                else (f"sample{donor_index}",)
            )
            for sample in samples:
                for lineage in ("B cells", "CD4_like"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "donor_id": donor,
                            "sample_id": sample,
                            "lineage": lineage,
                            "age": 20 + 8 * donor_index + dataset_index,
                            "sex": "female" if donor_index % 2 == 0 else "male",
                        }
                    )
    return pd.DataFrame(rows)


def test_approved_observation_id_retains_shared_source_pool() -> None:
    records = pd.DataFrame(
        {
            "dataset": ["onek1k", "onek1k", "terekhova", "terekhova"],
            "donor_id": ["d1", "d2", "d3", "d3"],
            "sample_id": ["pool61", "pool61", "v1", "v2"],
        }
    )
    identified = add_stable_identifiers(records)

    assert (
        identified.loc[0, "source_observation_id"]
        == identified.loc[1, "source_observation_id"]
    )
    assert identified.loc[0, "observation_id"] != identified.loc[1, "observation_id"]
    assert (
        identified.loc[2, "biological_unit_id"]
        == identified.loc[3, "biological_unit_id"]
    )
    assert identified.loc[2, "observation_id"] != identified.loc[3, "observation_id"]

    corrupted = identified.copy()
    corrupted.loc[0, "observation_id"] = "wrong"
    with pytest.raises(ValueError, match="conflicts"):
        add_stable_identifiers(corrupted)


def test_global_lodo_assignments_are_donor_grouped_and_lineage_independent(
    tmp_path: Path,
) -> None:
    cells = _split_metadata()
    manifest = build_global_donor_manifest(cells, n_inner_folds=3, seed=17)
    shuffled = build_global_donor_manifest(
        cells.sample(frac=1, random_state=9), n_inner_folds=3, seed=17
    )
    assignment = manifest.set_index("biological_unit_id")["global_inner_fold"]
    shuffled_assignment = shuffled.set_index("biological_unit_id")["global_inner_fold"]
    pd.testing.assert_series_equal(
        assignment.sort_index(), shuffled_assignment.sort_index()
    )
    assert len(manifest) == 5 * 8
    assert manifest["biological_unit_id"].is_unique
    assert manifest.groupby("dataset")["global_inner_fold"].nunique().eq(3).all()

    folds = build_lodo_tables(manifest)
    for heldout, table in folds.items():
        query_roles = table.loc[table["dataset"].eq(heldout), "outer_role"]
        assert query_roles.eq("query").all()
        assert table.loc[table["dataset"].eq(heldout), "inner_fold"].isna().all()
        assert not table.loc[
            table["dataset"].eq(heldout), "eligible_for_reference_fitting"
        ].any()
        assert_lodo_integrity(table, heldout)

    # Every repeated sample and lineage joins to exactly one donor assignment.
    joined = add_stable_identifiers(cells).merge(
        manifest[["biological_unit_id", "global_inner_fold"]],
        on="biological_unit_id",
        validate="many_to_one",
    )
    donor_fold_counts = joined.groupby("biological_unit_id")[
        "global_inner_fold"
    ].nunique()
    assert donor_fold_counts.eq(1).all()
    assert joined.groupby("observation_id")["global_inner_fold"].nunique().eq(1).all()

    paths = write_lodo_manifests(manifest, tmp_path)
    written = json.loads(paths["manifest"].read_text())
    assert written["identifier_definitions"]["observation_id"] == (
        "dataset::donor_id::sample_id"
    )
    assert written["source_observation_may_span_donors"] is True


def test_lodo_integrity_checks_reject_reference_leakage() -> None:
    manifest = build_global_donor_manifest(_split_metadata(), seed=2)
    table = build_lodo_tables(manifest)["onek1k"]
    leaked = table.copy()
    leaked.loc[leaked["dataset"].eq("onek1k"), "eligible_for_reference_fitting"] = True
    with pytest.raises(AssertionError, match="Query donor"):
        assert_lodo_integrity(leaked, "onek1k")
    with pytest.raises(AssertionError, match="overlap"):
        assert_partition_disjoint({"train": ["a", "b"], "query": ["b"]})


def test_global_manifest_recovers_samples_from_aggregated_audit_rows() -> None:
    records = pd.DataFrame(
        {
            "dataset": ["a", "a", "a", "b", "b", "b"],
            "donor_id": ["d1", "d1", "d2", "d3", "d3", "d4"],
            "sample_ids": ["v1|v2", "v2", "pool", "s1", "s1|s2", "s3"],
            "age_min": [30, 30, 40, 50, 50, 60],
            "age_max": [31, 31, 40, 51, 51, 60],
            "sex": ["female", "female", "male", "female", "female", "male"],
        }
    )
    manifest = build_global_donor_manifest(
        records, datasets=("a", "b"), n_inner_folds=2, seed=5
    ).set_index("biological_unit_id")

    assert manifest.loc["a::d1", "sample_ids"] == "v1|v2"
    assert manifest.loc["a::d1", "source_observation_ids"] == "a::v1|a::v2"
    assert manifest.loc["a::d1", "observation_ids"] == "a::d1::v1|a::d1::v2"
    assert manifest.loc["a::d1", "n_source_observations"] == 2


def _sampler_metadata(dataset_donor_counts: tuple[int, ...] = (1, 4)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, n_donors in enumerate(dataset_donor_counts):
        dataset = f"s{dataset_index}"
        for donor_index in range(n_donors):
            donor = f"{dataset}::d{donor_index}"
            for fine_type, count in (("major", 9), ("minor", 1)):
                for cell_index in range(count):
                    rows.append(
                        {
                            "cell_id": f"{donor}:{fine_type}:{cell_index}",
                            "dataset": dataset,
                            "biological_unit_id": donor,
                            "fine_type": fine_type,
                            "lineage": "B cells",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    ((0.0, (0.5, 0.5)), (0.5, (1 / 3, 2 / 3)), (1.0, (0.2, 0.8))),
)
def test_sampler_dataset_probabilities_follow_eligible_donor_counts(
    alpha: float, expected: tuple[float, float]
) -> None:
    sampler = HierarchicalCellSampler(_sampler_metadata(), alpha=alpha)
    dataset_rows = sampler.intended_distribution.query("level == 'dataset'")
    probabilities = dataset_rows.set_index("dataset")["intended_probability"]
    assert probabilities["s0"] == pytest.approx(expected[0])
    assert probabilities["s1"] == pytest.approx(expected[1])


@pytest.mark.parametrize(
    ("mode", "hybrid_lambda", "expected_major"),
    (
        ("observed_proportions", 0.7, 0.9),
        ("fully_balanced", 0.7, 0.5),
        ("hybrid", 0.7, 0.78),
    ),
)
def test_sampler_fine_type_targets_and_empirical_draws(
    mode: str, hybrid_lambda: float, expected_major: float
) -> None:
    sampler = HierarchicalCellSampler(
        _sampler_metadata((1,)),
        mode=mode,
        hybrid_lambda=hybrid_lambda,
        alpha=0.5,
        seed=123,
    )
    fine_rows = sampler.intended_distribution.query("level == 'fine_type'")
    target = fine_rows.set_index("fine_type")["conditional_probability"]
    assert target["major"] == pytest.approx(expected_major)

    result = sampler.sample_epoch(20_000, epoch=3, batch_size=1)
    realized = result.distribution_log.query("level == 'fine_type'").set_index(
        "fine_type"
    )["realized_conditional_probability"]
    assert realized["major"] == pytest.approx(expected_major, abs=0.015)
    assert result.duplicate_cell_draws_within_batches == 0


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("observed_proportions", {"major": 0.6, "minor": 0.2, "low_confidence": 0.2}),
        ("hybrid", {"major": 0.57, "minor": 0.29, "low_confidence": 0.14}),
        ("fully_balanced", {"major": 0.5, "minor": 0.5, "low_confidence": 0.0}),
    ),
)
def test_sampler_uniform_uplift_only_applies_to_balance_eligible_fine_types(
    mode: str, expected: dict[str, float]
) -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": [f"cell-{index}" for index in range(10)],
            "dataset": ["cohort"] * 10,
            "biological_unit_id": ["cohort::donor"] * 10,
            "fine_type": ["major"] * 6 + ["minor"] * 2 + ["low_confidence"] * 2,
            "fine_type_balance_eligible": [True] * 8 + [False] * 2,
            "lineage": ["B cells"] * 10,
        }
    )
    sampler = HierarchicalCellSampler(
        metadata,
        mode=mode,
        hybrid_lambda=0.7,
        seed=61,
    )
    targets = sampler.intended_distribution.query("level == 'fine_type'").set_index(
        "fine_type"
    )

    assert targets["conditional_probability"].to_dict() == pytest.approx(expected)
    assert targets.loc["low_confidence", "uniform_fine_type_probability"] == 0.0
    assert not bool(targets.loc["low_confidence", "fine_type_balance_eligible"])
    assert targets["conditional_probability"].sum() == pytest.approx(1.0)

    result = sampler.sample_epoch(20_000, epoch=4, batch_size=1)
    realized = result.distribution_log.query("level == 'fine_type'").set_index(
        "fine_type"
    )
    assert realized["realized_conditional_probability"].to_dict() == pytest.approx(
        expected, abs=0.015
    )
    if mode == "fully_balanced":
        ineligible_positions = set(
            metadata.index[metadata["fine_type"].eq("low_confidence")]
        )
        assert not (set(result.cell_positions) & ineligible_positions)
        assert result.summary()["balance_ineligible_draws"] == 0
        assert result.summary()["n_zero_probability_fine_type_strata"] == 1


def test_sampler_defaults_missing_balance_eligibility_to_all_eligible() -> None:
    sampler = HierarchicalCellSampler(
        _sampler_metadata((1,)), mode="hybrid", hybrid_lambda=0.7
    )

    fine_rows = sampler.intended_distribution.query("level == 'fine_type'")
    assert fine_rows["fine_type_balance_eligible"].all()
    assert set(fine_rows["balance_eligibility_source"]) == {
        "default_all_eligible_missing_column"
    }
    assert fine_rows.set_index("fine_type").loc[
        "major", "conditional_probability"
    ] == pytest.approx(0.78)


def test_sampler_rejects_inconsistent_flags_and_audits_all_ineligible_fallback() -> (
    None
):
    inconsistent = _sampler_metadata((1,))
    inconsistent["fine_type_balance_eligible"] = True
    inconsistent.loc[inconsistent.index[0], "fine_type_balance_eligible"] = False
    with pytest.raises(ValueError, match="must be constant"):
        HierarchicalCellSampler(inconsistent)

    all_ineligible = _sampler_metadata((1,))
    all_ineligible["fine_type_balance_eligible"] = False
    hybrid = HierarchicalCellSampler(all_ineligible, mode="hybrid")
    hybrid_rows = hybrid.intended_distribution.query("level == 'fine_type'")
    assert hybrid_rows["conditional_probability"].sum() == pytest.approx(1.0)
    assert hybrid_rows["effective_fine_type_lambda"].eq(1.0).all()
    assert hybrid_rows["balance_fallback_to_observed"].all()
    hybrid_result = hybrid.sample_epoch(1_000, batch_size=1)
    assert (
        hybrid_result.summary()[
            "n_donors_fallback_to_observed_no_balance_eligible_type"
        ]
        == 1
    )

    # Lambda=1 already has no uniform component and needs no fallback.
    observed = HierarchicalCellSampler(
        all_ineligible, mode="observed_proportions"
    ).intended_distribution.query("level == 'fine_type'")
    assert observed["conditional_probability"].sum() == pytest.approx(1.0)
    assert not observed["balance_fallback_to_observed"].any()


def test_sampler_avoids_cloning_until_unavoidable_and_is_rank_deterministic() -> None:
    metadata = _sampler_metadata((1,))
    sampler = HierarchicalCellSampler(metadata, mode="fully_balanced", seed=44)
    unique_batch = sampler.sample_epoch(10, batch_size=10)
    assert len(np.unique(unique_batch.cell_positions)) == 10
    assert unique_batch.duplicate_cell_draws_within_batches == 0

    oversized = sampler.sample_epoch(12, batch_size=12)
    assert oversized.duplicate_cell_draws_within_batches == 2
    assert oversized.forced_replacement_cycles == 1
    repeat = sampler.sample_epoch(12, batch_size=12)
    np.testing.assert_array_equal(oversized.cell_positions, repeat.cell_positions)

    other_rank = HierarchicalCellSampler(
        metadata, mode="fully_balanced", seed=44, rank=1, world_size=2
    ).sample_epoch(12, batch_size=12)
    rank_zero = HierarchicalCellSampler(
        metadata, mode="fully_balanced", seed=44, rank=0, world_size=2
    ).sample_epoch(12, batch_size=12)
    assert not np.array_equal(rank_zero.cell_positions, other_rank.cell_positions)


def test_generated_ontology_is_exact_reviewable_and_uncertainty_aware(
    tmp_path: Path,
) -> None:
    audited = pd.DataFrame(
        {
            "dataset": ["a", "b", "a"],
            "lineage": ["B cells", "B cells", "B cells"],
            "fine_type": ["Memory B", "Memory B", "Age-associated B"],
            "n_cells": [100, 50, 3],
            "n_donors": [20, 10, 2],
            "confidence_mean": [0.97, 0.95, 0.70],
            "confidence_lt_0_9": [2, 2, 2],
        }
    )
    summary = summarize_fine_type_labels(audited, poor_donor_coverage_below=5)
    rare = summary.loc[summary["fine_type"].eq("Age-associated B")].iloc[0]
    assert rare["found_in_one_dataset"]
    assert rare["poor_donor_coverage"]
    assert rare["marker_expression_sanity"] == "not_evaluated"

    ontology = generate_candidate_ontology(audited, minimum_cells_for_state=30)
    assert ontology["generated"] and ontology["requires_approval"]
    mappings = ontology["lineages"]["B cells"]["mappings"]
    assert {item["original_label"] for item in mappings} == {
        "Memory B",
        "Age-associated B",
    }
    assert all(
        item["original_label"] == item["canonical_fine_type"] for item in mappings
    )
    path = write_candidate_ontology(ontology, tmp_path / "candidate.yaml")
    assert "requires_approval: true" in path.read_text()

    cells = pd.DataFrame(
        {
            "lineage": ["B cells"] * 3,
            "fine_type": ["Memory B", "Memory B", "unseen"],
            "annotation_confidence": [0.99, 0.2, 0.99],
        }
    )
    with pytest.raises(ValueError, match="requires scientific approval"):
        apply_fine_type_ontology(cells, ontology)
    mapped = apply_fine_type_ontology(cells, ontology, allow_unapproved=True)
    assert mapped["canonical_fine_type"].tolist() == [
        "Memory B",
        "low_confidence",
        "other_confident",
    ]


def test_approved_ontology_quarantines_without_dropping_cells() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "data"
        / "fine_type_ontology.approved.yaml"
    )
    ontology = load_fine_type_ontology(path, require_approved=True)
    cells = pd.DataFrame(
        {
            "lineage": ["B cells"] * 4,
            "ctype_low": [
                "Naive B cells",
                "Naive B cells",
                "MAIT cells",
                "unexpected label",
            ],
            "ctype_low_conf": [0.95, 0.60, 0.95, 0.95],
        }
    )
    mapped = apply_fine_type_ontology(
        cells,
        ontology,
        fine_type_column="ctype_low",
        confidence_column="ctype_low_conf",
        output_column="fine_type",
    )
    assert len(mapped) == len(cells)
    assert mapped["fine_type"].tolist() == [
        "Naive B cells",
        "low_confidence",
        "other_confident",
        "other_confident",
    ]
    assert mapped["fine_type_state_eligible"].tolist() == [True, False, False, False]
    assert mapped["fine_type_balance_eligible"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="not scientifically approved"):
        apply_fine_type_ontology(
            pd.DataFrame(
                {
                    "lineage": ["DC"],
                    "ctype_low": ["DC2"],
                    "ctype_low_conf": [0.99],
                }
            ),
            ontology,
            fine_type_column="ctype_low",
            confidence_column="ctype_low_conf",
        )


def test_strict_ensembl_mapping_reports_all_loss_and_ambiguity() -> None:
    resource = pd.DataFrame(
        {
            "ensembl_id": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000002",
                "ENSG00000000003",
            ],
            "symbol": ["A", "B", "B_ALT", "A"],
        }
    )
    result = map_ensembl_to_symbols(
        [
            "ENSG00000000001.4",
            "ENSG00000000001.5",
            "ENSG00000000002",
            "ENSG00000000009",
        ],
        resource,
        resource_version="Ensembl-test-v1",
    )
    assert result.summary["n_version_suffixes_stripped"] == 2
    assert result.summary["n_one_to_many_resource_ids"] == 1
    assert result.summary["n_many_to_one_resource_symbols"] == 1
    assert set(result.mapping["status"]) == {
        "mapped_many_to_one",
        "ambiguous_one_to_many",
        "unmapped",
    }
    assert result.mapping["duplicate_query_after_version_strip"].sum() == 2
    with pytest.raises(AmbiguousGeneMappingError):
        result.require_unambiguous()

    # Ambiguities elsewhere in a full mapping resource do not invalidate an
    # unrelated, uniquely mapped query, but remain counted in the summary.
    extended = pd.concat(
        [
            resource,
            pd.DataFrame({"ensembl_id": ["ENSG00000000009"], "symbol": ["Z"]}),
        ],
        ignore_index=True,
    )
    clean = map_ensembl_to_symbols(
        ["ENSG00000000009"], extended, resource_version="Ensembl-test-v1"
    )
    assert clean.require_unambiguous().loc[0, "symbol"] == "Z"
    assert clean.summary["n_one_to_many_resource_ids"] == 1


def test_generic_gp_loaders_and_production_fixture_guard(tmp_path: Path) -> None:
    gmt = tmp_path / "library.gmt"
    gmt.write_text("p1\tHallmark\tA\tB\n")
    programs = load_gene_programs(gmt, source="Hallmark")
    assert programs[0].genes == ("A", "B")
    assert programs[0].category == "Hallmark"

    fixture = write_synthetic_gp_library(tmp_path, format="tsv")
    loaded = validate_gp_resource(fixture, production=False)
    assert len(loaded) == 3
    assert loaded[0].metadata["test_fixture"] == "True"
    with pytest.raises(ValueError, match="Synthetic"):
        validate_gp_resource(fixture, production=True)
    with pytest.raises(FileNotFoundError, match="Configure an explicit"):
        load_gene_programs(tmp_path / "missing.gmt")

    inconsistent = tmp_path / "inconsistent.tsv"
    pd.DataFrame(
        {
            "program_id": ["p", "p"],
            "gene": ["A", "B"],
            "direction": ["up", "down"],
        }
    ).to_csv(inconsistent, sep="\t", index=False)
    with pytest.raises(ValueError, match="inconsistent 'direction'"):
        load_gene_programs(inconsistent)
    with pytest.raises(ValueError, match="repeats members"):
        GeneProgram("duplicates", ("A", "A"), "test")


def test_generic_gp_loader_supports_decoupler_and_msigdb_exports(
    tmp_path: Path,
) -> None:
    decoupler = tmp_path / "progeny.tsv"
    pd.DataFrame(
        {
            "source": ["TNFa", "TNFa", "JAK-STAT"],
            "target": ["A", "B", "C"],
            "weight": [1.0, -0.5, 0.2],
        }
    ).to_csv(decoupler, sep="\t", index=False)
    programs = load_gene_programs(decoupler, source="PROGENy")
    assert {program.program_id for program in programs} == {"TNFa", "JAK-STAT"}
    assert all(program.source == "PROGENy" for program in programs)

    msigdb = tmp_path / "hallmark.tsv"
    pd.DataFrame(
        {
            "gs_name": ["HALLMARK_X", "HALLMARK_X", "HALLMARK_X"],
            "gene_symbol": ["A", "A", "B"],
            "gs_collection_name": ["Hallmark", "Hallmark", "Hallmark"],
            "source_gene": ["a1", "a2", "b1"],
        }
    ).to_csv(msigdb, sep="\t", index=False)
    hallmark = load_gene_programs(msigdb, source="MSigDB")
    assert hallmark[0].genes == ("A", "B")
    assert hallmark[0].category == "Hallmark"
    assert hallmark[0].metadata["duplicate_gene_rows_removed"] == 1


def _gene_stats(heldout_value: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("train_a", "train_b", "heldout"):
        for gene in ("A", "B", "C", "D", "X", "Y"):
            if dataset == "heldout":
                coverage = heldout_value if gene in {"X", "Y"} else 0.0
            else:
                coverage = 0.9 if gene in {"A", "B", "C", "D"} else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "gene": gene,
                    "expression_coverage": coverage,
                    "donor_coverage": coverage,
                }
            )
    return pd.DataFrame(rows)


def test_gp_filtering_is_training_only_and_reports_redundancy() -> None:
    programs = (
        GeneProgram("small", ("A", "B", "C"), "test"),
        GeneProgram("large", ("A", "B", "C", "D"), "test"),
        GeneProgram("heldout_only", ("X", "Y"), "test"),
        GeneProgram(
            "wrong_lineage",
            ("A", "D"),
            "test",
            metadata={"lineages": "Monocytes"},
        ),
    )
    config = GPFilterConfig(
        minimum_mapped_genes=2,
        maximum_program_size=10,
        minimum_expression_coverage=0.5,
        minimum_donor_coverage=0.5,
        redundancy_jaccard_threshold=0.7,
    )
    low = filter_gene_programs_training_only(
        programs,
        _gene_stats(0.0),
        training_datasets=("train_a", "train_b"),
        heldout_dataset="heldout",
        config=config,
        lineage="B cells",
    )
    high = filter_gene_programs_training_only(
        programs,
        _gene_stats(1.0),
        training_datasets=("train_a", "train_b"),
        heldout_dataset="heldout",
        config=config,
        lineage="B cells",
    )
    assert [program.program_id for program in low.programs] == ["large"]
    assert [program.program_id for program in high.programs] == ["large"]
    pd.testing.assert_frame_equal(low.report, high.report)
    report = low.report.set_index("program_id")
    assert report.loc["small", "reason"] == "redundant_jaccard"
    assert report.loc["small", "redundant_with"] == "large"
    assert report.loc["heldout_only", "reason"] == ("too_few_training_supported_genes")
    assert "not_applicable_to_lineage" in report.loc["wrong_lineage", "reason"]
    assert low.ignored_nontraining_rows == 6


def test_sparse_hvg_and_coverage_statistics_never_use_heldout_cells() -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["train", "train", "train", "heldout"],
            "biological_unit_id": ["train::d1", "train::d1", "train::d2", "heldout::q"],
        }
    )
    genes = ("variable", "query_only", "constant")
    base = sparse.csr_matrix(
        np.asarray(
            [
                [0, 0, 1],
                [1, 0, 1],
                [5, 0, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
    )
    changed = base.copy().tolil()
    changed[3, 1] = 1_000_000
    changed = changed.tocsr()
    first = select_hvgs_training_only(
        base,
        metadata,
        genes,
        training_datasets=("train",),
        heldout_dataset="heldout",
        n_top_genes=1,
    )
    second = select_hvgs_training_only(
        changed,
        metadata,
        genes,
        training_datasets=("train",),
        heldout_dataset="heldout",
        n_top_genes=1,
    )
    assert first.selected_genes == second.selected_genes == ("variable",)
    stats = compute_training_gene_statistics(
        changed,
        metadata,
        genes,
        training_datasets=("train",),
        heldout_dataset="heldout",
    ).set_index("gene")
    assert stats.loc["query_only", "expression_coverage"] == 0
    assert stats.loc["variable", "donor_coverage"] == 1


def test_synthetic_gp_helper_is_explicitly_test_only() -> None:
    programs = synthetic_gene_programs()
    assert programs
    assert all(program.metadata.get("test_fixture") for program in programs)
