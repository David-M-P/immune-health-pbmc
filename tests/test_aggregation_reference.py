"""Deterministic donor-level scientific vertical-slice tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.aggregation import (
    aggregate_fine_type_distributions,
    covariance_decomposition,
    fine_type_stratified_bootstrap,
    gaussian_wasserstein_distance,
    lineage_state_scores,
    shrinkage_covariance,
    sliced_wasserstein_distance,
    stratified_bootstrap_indices,
)
from immune_health.baselines import (
    AgeSexCompositionModel,
    ElasticNetAgeModel,
    TrainOnlyPCA,
    aitchison_distance,
    build_composition_table,
    build_pseudobulk,
    composition_matrix,
    ensure_donor_observation_ids,
    score_gene_programs,
)
from immune_health.evaluation import (
    age_prediction_metrics,
    finalize_fine_type_output,
    finalize_lineage_output,
    matched_depth_sensitivity,
    reliability_curve,
    validate_fine_type_gp_schema,
    validate_lineage_gp_schema,
)
from immune_health.healthy_reference import (
    HealthyTrajectory,
    age_kernel_weights,
    bootstrap_healthy_reference_scores,
    cross_fit_trajectory,
)


def _cell_fixture(seed: int = 12) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    counts: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    donor_number = 0
    for dataset, n_donors in (("train_a", 6), ("train_b", 6), ("query", 4)):
        for donor_in_dataset in range(n_donors):
            donor_number += 1
            donor = f"d{donor_in_dataset}"
            age = 20.0 + 4.0 * donor_number
            sex = "F" if donor_in_dataset % 2 == 0 else "M"
            for visit in range(2 if donor_in_dataset == 0 else 1):
                sample = "shared_pool" if dataset == "train_a" else f"v{visit}"
                naive_cells = 7 if age < 50 else 4
                fine_type_counts = (
                    ("naive", naive_cells, 0.0),
                    ("memory", 10 - naive_cells, 1.0),
                )
                for fine_type, n_cells, offset in fine_type_counts:
                    for _ in range(n_cells):
                        expression = rng.poisson(2.0, size=8)
                        expression[0] += int(age // 20)
                        expression[1] += int(offset * 2)
                        counts.append(expression)
                        noise_scale = 0.04 + age / 1_000.0
                        embeddings.append(
                            np.asarray(
                                [
                                    age / 20.0 + rng.normal(0, noise_scale),
                                    offset + rng.normal(0, noise_scale),
                                ]
                            )
                        )
                        records.append(
                            {
                                "dataset": dataset,
                                "donor_id": donor,
                                "sample_id": sample,
                                "age": age,
                                "sex": sex,
                                "lineage": "B cells",
                                "fine_type": fine_type,
                                "confidence": 0.95,
                            }
                        )
    return (
        sparse.csr_matrix(np.asarray(counts)),
        pd.DataFrame(records),
        np.vstack(embeddings),
    )


class IdentifierAndBaselineTests(unittest.TestCase):
    def test_collision_safe_observation_id_and_sparse_pseudobulk(self) -> None:
        obs = pd.DataFrame(
            {
                "dataset": ["onek1k", "onek1k", "onek1k", "onek1k"],
                "donor_id": ["d1", "d1", "d2", "d2"],
                "sample_id": ["pool7", "pool7", "pool7", "pool7"],
                "lineage": ["B cells"] * 4,
                "fine_type": ["naive", "naive", "naive", "memory"],
                "age": [30, 30, 50, 50],
                "sex": ["F", "F", "M", "M"],
            }
        )
        raw = sparse.csr_matrix([[1, 0, 2], [3, 1, 0], [5, 0, 1], [0, 2, 4]])
        identified = ensure_donor_observation_ids(obs)
        self.assertEqual(identified["biological_unit_id"].nunique(), 2)
        self.assertEqual(identified["observation_id"].nunique(), 2)
        self.assertEqual(identified["source_observation_id"].nunique(), 1)
        self.assertEqual(identified.loc[0, "source_observation_id"], "onek1k::pool7")
        self.assertEqual(identified.loc[0, "observation_id"], "onek1k::d1::pool7")

        result = build_pseudobulk(raw, obs, ["g1", "g2", "g3"])
        self.assertTrue(sparse.isspmatrix_csr(result.counts))
        self.assertEqual(result.counts.shape, (3, 3))
        np.testing.assert_array_equal(result.counts[0].toarray(), [[4, 1, 2]])
        self.assertEqual(result.metadata.loc[0, "library_size"], 7)
        np.testing.assert_array_equal(
            raw.toarray(), [[1, 0, 2], [3, 1, 0], [5, 0, 1], [0, 2, 4]]
        )

    def test_composition_model_and_gp_scores(self) -> None:
        counts, obs, _ = _cell_fixture()
        table = build_composition_table(
            obs, fine_type_universe=("naive", "memory", "rare")
        )
        metadata, matrix, labels = composition_matrix(table)
        self.assertEqual(labels, ("memory", "naive", "rare"))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        self.assertTrue((table.loc[table["fine_type"] == "rare", "n_cells"] == 0).all())
        self.assertAlmostEqual(aitchison_distance(matrix[0], matrix[0]), 0.0)

        model = AgeSexCompositionModel().fit(
            matrix,
            metadata["age"],
            metadata["sex"],
            metadata["biological_unit_id"],
            fine_types=labels,
        )
        expected = model.predict(metadata["age"], metadata["sex"])
        np.testing.assert_allclose(expected.sum(axis=1), 1.0)
        distances = model.distance(matrix, metadata["age"], metadata["sex"])
        self.assertTrue(np.isfinite(distances).all())

        pseudobulk = build_pseudobulk(counts, obs, [f"g{i}" for i in range(8)])
        scores = score_gene_programs(
            pseudobulk.counts,
            pseudobulk.gene_ids,
            {"age_gp": ["g0", "g2"], "missing": ["x"]},
            minimum_genes=2,
        )
        self.assertEqual(len(scores), 2 * len(pseudobulk.metadata))
        missing_scores = scores.loc[scores["gp_id"] == "missing", "gp_score"]
        self.assertTrue(missing_scores.isna().all())

    def test_training_only_pca_and_donor_grouped_elastic_net(self) -> None:
        rng = np.random.default_rng(7)
        groups = np.asarray([f"train::d{i}" for i in range(24)])
        age = np.linspace(20, 80, len(groups))
        counts = sparse.csr_matrix(
            rng.poisson(2, size=(len(groups), 12)) + np.floor(age[:, None] / 20)
        )
        pca = TrainOnlyPCA(n_components=4)
        training_latent = pca.fit_transform(
            counts,
            feature_ids=[f"g{i}" for i in range(12)],
            training_biological_units=groups,
        )
        query_counts = sparse.csr_matrix(rng.poisson(3, size=(4, 12)))
        projected = pca.transform(
            query_counts,
            feature_ids=[f"g{i}" for i in range(12)],
            query_biological_units=[f"query::d{i}" for i in range(4)],
        )
        self.assertEqual(projected.shape, (4, 4))
        with self.assertRaises(ValueError):
            pca.transform(
                query_counts[:1],
                feature_ids=[f"g{i}" for i in range(12)],
                query_biological_units=[groups[0]],
            )

        age_model = ElasticNetAgeModel(
            alphas=(0.001, 0.01), l1_ratios=(0.1, 0.5), n_splits=4
        ).fit(training_latent, age, groups)
        prediction = age_model.predict(
            projected,
            query_biological_units=[f"query::d{i}" for i in range(4)],
        )
        self.assertEqual(prediction.shape, (4,))
        self.assertEqual(len(age_model.cv_results_), 4)


class DistributionAndUncertaintyTests(unittest.TestCase):
    def test_shrinkage_distances_and_decomposition(self) -> None:
        rng = np.random.default_rng(4)
        first = rng.normal(size=(40, 5))
        shifted = first + np.asarray([1, 0, 0, 0, 0])
        covariance = shrinkage_covariance(first)
        self.assertTrue((np.linalg.eigvalsh(covariance) > 0).all())
        self.assertAlmostEqual(sliced_wasserstein_distance(first, first, seed=3), 0.0)
        distance = sliced_wasserstein_distance(first, shifted, seed=3)
        self.assertGreater(distance, 0.2)
        self.assertEqual(distance, sliced_wasserstein_distance(first, shifted, seed=3))
        self.assertAlmostEqual(
            gaussian_wasserstein_distance(
                first.mean(axis=0), covariance, first.mean(axis=0), covariance
            ),
            0.0,
            places=6,
        )

        groups = {
            "a": np.asarray([[0.0], [2.0]]),
            "b": np.asarray([[4.0], [6.0]]),
        }
        result = covariance_decomposition(groups)
        self.assertAlmostEqual(result.within_trace, 1.0)
        self.assertAlmostEqual(result.between_trace, 4.0)
        self.assertAlmostEqual(result.total_trace, 5.0)
        scores = lineage_state_scores(
            groups,
            {label: values + 0.5 for label, values in groups.items()},
            {"a": 0.7, "b": 0.3},
            {"a": 0.5, "b": 0.5},
        )
        self.assertAlmostEqual(
            scores["total_lineage_heterogeneity"],
            scores["within_fine_type_heterogeneity"]
            + scores["between_fine_type_heterogeneity"],
        )
        self.assertGreater(scores["composition_only_score"], 0)

    def test_rare_state_is_missing_and_bootstrap_stays_stratified(self) -> None:
        _, obs, embeddings = _cell_fixture()
        rare_row = obs.iloc[[0]].copy()
        rare_row["fine_type"] = "rare"
        obs = pd.concat([obs, rare_row], ignore_index=True)
        embeddings = np.vstack([embeddings, embeddings[0]])
        result = aggregate_fine_type_distributions(
            embeddings,
            obs,
            gp_id="gp1",
            fine_type_universe=("naive", "memory", "rare", "never_seen"),
            min_state_cells=3,
            annotation_confidence_col="confidence",
        )
        rare = result.table[result.table["fine_type"] == "rare"].iloc[0]
        self.assertEqual(rare["n_cells"], 1)
        self.assertFalse(rare["state_available"])
        self.assertTrue(pd.isna(rare["location_summary"]))
        self.assertTrue(np.isnan(rare["covariance_trace"]))
        never_seen = result.table[result.table["fine_type"] == "never_seen"]
        self.assertTrue((never_seen["n_cells"] == 0).all())
        self.assertTrue(never_seen["location_summary"].isna().all())

        labels = np.asarray(["a"] * 7 + ["b"] * 3)
        draws = list(stratified_bootstrap_indices(labels, n_bootstrap=4, seed=19))
        draws_again = list(stratified_bootstrap_indices(labels, n_bootstrap=4, seed=19))
        for first, second in zip(draws, draws_again):
            np.testing.assert_array_equal(first, second)
            self.assertEqual((labels[first] == "a").sum(), 7)
            self.assertEqual((labels[first] == "b").sum(), 3)
        values = np.arange(10.0)[:, None]
        estimates = fine_type_stratified_bootstrap(
            values,
            labels,
            lambda sample, strata: np.asarray(
                [sample[strata == label].mean() for label in ("a", "b")]
            ),
            n_bootstrap=5,
            seed=2,
        )
        self.assertEqual(estimates.shape, (5, 2))

    def test_ontology_ineligible_fine_type_is_composition_only(self) -> None:
        obs = pd.DataFrame(
            {
                "dataset": ["a"] * 6,
                "donor_id": ["d1"] * 6,
                "sample_id": ["s1"] * 6,
                "lineage": ["B cells"] * 6,
                "fine_type": ["naive"] * 3 + ["low_confidence"] * 3,
                "fine_type_state_eligible": [True] * 3 + [False] * 3,
                "age": [40.0] * 6,
                "sex": ["female"] * 6,
            }
        )
        embeddings = np.arange(12, dtype=float).reshape(6, 2)
        result = aggregate_fine_type_distributions(
            embeddings,
            obs,
            gp_id="gp1",
            fine_type_universe={"B cells": ["naive"]},
            min_state_cells=2,
        )
        special = result.table.loc[result.table["fine_type"].eq("low_confidence")].iloc[
            0
        ]
        self.assertEqual(special["fine_type_fraction"], 0.5)
        self.assertFalse(special["fine_type_state_eligible"])
        self.assertFalse(special["state_available"])
        self.assertEqual(special["state_quality"], "ineligible_fine_type")
        self.assertTrue(pd.isna(special["location_summary"]))


class HealthyReferenceAndEvaluationTests(unittest.TestCase):
    def _donor_features(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(31)
        age = np.linspace(20, 80, 30)
        sex = np.where(np.arange(len(age)) % 2, "M", "F")
        dataset = np.where(np.arange(len(age)) % 3, "a", "b")
        donors = np.asarray([f"{dataset[i]}::d{i}" for i in range(len(age))])
        features = np.column_stack(
            [
                age / 20 + 0.15 * (sex == "M") + rng.normal(0, 0.03, len(age)),
                np.sin(age / 18) + rng.normal(0, 0.03, len(age)),
            ]
        )
        return features, age, sex, dataset, donors

    def test_frozen_trajectory_cross_fit_kernel_and_reference_bootstrap(self) -> None:
        features, age, sex, dataset, donors = self._donor_features()
        model = HealthyTrajectory(n_spline_knots=2).fit(
            features, age, sex, donors, datasets=dataset
        )
        query = model.predict([55.0], ["F"])[0] + np.asarray([0.1, 0.0])
        score = model.score(query, 55.0, "F")
        self.assertTrue(20 <= score["predicted_gp_age"] <= 80)
        self.assertAlmostEqual(
            score["gp_age_acceleration"], score["predicted_gp_age"] - 55.0
        )
        self.assertLessEqual(
            score["off_trajectory_distance"], score["age_matched_distance"]
        )
        with self.assertRaises(ValueError):
            model.predict([55.0], ["F"], datasets=["held_out"])

        cross_fit = cross_fit_trajectory(
            features,
            age,
            sex,
            donors,
            datasets=dataset,
            n_splits=5,
            model_kwargs={"n_spline_knots": 2},
        )
        self.assertTrue(np.isfinite(cross_fit.expected_locations).all())
        self.assertEqual(cross_fit.scores["row_index"].nunique(), len(features))

        weights = age_kernel_weights(
            age,
            50,
            sexes=sex,
            target_sex="F",
            biological_unit_ids=donors,
            exclude_biological_units=[donors[0]],
            minimum_exact_sex_donors=4,
        )
        self.assertAlmostEqual(weights.weights.sum(), 1.0)
        self.assertEqual(weights.weights[0], 0.0)
        self.assertTrue(weights.exact_sex_used)

        boot = bootstrap_healthy_reference_scores(
            features,
            age,
            sex,
            donors,
            query,
            55,
            "F",
            datasets=dataset,
            n_bootstrap=4,
            seed=9,
            model_kwargs={"n_spline_knots": 1},
        )
        self.assertEqual(len(boot), 4)
        self.assertTrue(np.isfinite(boot["age_matched_distance"]).all())

    def test_schema_matched_depth_and_end_to_end_slice(self) -> None:
        counts, obs, embeddings = _cell_fixture()
        train = obs["dataset"] != "query"
        query = ~train
        aggregation = aggregate_fine_type_distributions(
            embeddings[query],
            obs.loc[query].reset_index(drop=True),
            gp_id="synthetic_gp",
            min_state_cells=3,
            provenance={"annotation_version": "synthetic_v1"},
        )
        trajectory = HealthyTrajectory(n_spline_knots=1).fit(
            embeddings[train],
            obs.loc[train, "age"],
            obs.loc[train, "sex"],
            ensure_donor_observation_ids(obs.loc[train])["biological_unit_id"],
            datasets=obs.loc[train, "dataset"],
        )
        table = aggregation.table.copy()
        for row_index, row in table.iterrows():
            key = (
                row["observation_id"],
                row["lineage"],
                row["fine_type"],
                row["gp_id"],
            )
            estimate = aggregation.estimates[key]
            if estimate.state_available:
                score = trajectory.score(
                    estimate.location,
                    float(row["age"]),
                    str(row["sex"]),
                )
                for name, value in score.items():
                    table.loc[row_index, name] = value
        output = finalize_fine_type_output(
            table,
            {
                "model_id": "tiny_baseline",
                "fold_id": "heldout_query",
                "seed": 12,
                "annotation_version": "synthetic_v1",
                "gp_library_version": "synthetic_v1",
                "reference_version": "synthetic_v1",
            },
        )
        validate_fine_type_gp_schema(output)
        self.assertEqual(output["dataset"].unique().tolist(), ["query"])

        first = output.iloc[0]
        lineage_output = finalize_lineage_output(
            pd.DataFrame(
                [
                    {
                        column: first[column]
                        for column in (
                            "dataset",
                            "donor_id",
                            "biological_unit_id",
                            "sample_id",
                            "source_observation_id",
                            "observation_id",
                            "age",
                            "sex",
                            "lineage",
                            "gp_id",
                        )
                    }
                    | {
                        "within_fine_type_heterogeneity": 1.0,
                        "between_fine_type_heterogeneity": 2.0,
                        "total_lineage_heterogeneity": 3.0,
                    }
                ]
            ),
            {
                "model_id": "tiny_baseline",
                "fold_id": "heldout_query",
                "seed": 12,
                "annotation_version": "synthetic_v1",
                "gp_library_version": "synthetic_v1",
                "reference_version": "synthetic_v1",
            },
        )
        validate_lineage_gp_schema(lineage_output)

        per_observation = (
            output.groupby("observation_id", observed=True).first().reset_index()
        )
        metrics = age_prediction_metrics(
            per_observation["age"], per_observation["predicted_gp_age"]
        )
        self.assertEqual(
            metrics["n_observations"],
            ensure_donor_observation_ids(obs.loc[query])["observation_id"].nunique(),
        )

        sensitivity = matched_depth_sensitivity(
            embeddings[:40],
            embeddings[40:90],
            depths=(10, 25, 100),
            n_replicates=3,
            seed=5,
        )
        curve = reliability_curve(sensitivity)
        self.assertEqual(curve["cell_depth"].tolist(), [10, 25])
        insufficient = sensitivity.loc[sensitivity["cell_depth"] == 100, "status"].iloc[
            0
        ]
        self.assertEqual(insufficient, "insufficient_depth")

        pseudobulk = build_pseudobulk(counts, obs, [f"g{i}" for i in range(8)])
        self.assertGreater(pseudobulk.counts.nnz, 0)


if __name__ == "__main__":
    unittest.main()
