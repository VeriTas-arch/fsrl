import unittest
from itertools import combinations

import numpy as np

from fsrl.analysis.posterior import ExactRankingPosterior
from fsrl.experiments.assembly.trajectory import (
    build_complete_graph_geometry,
    classified_effects,
    gradient_energy_fraction,
    hodge_potentials,
    load_json,
    normalize_potentials,
    potential_alignment,
    validate_registered_sources,
    vector_gradient_energy_fraction,
)
from fsrl.infrastructure.study_registry import resolve_record
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol


class AssemblyTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        self.geometry = build_complete_graph_geometry(self.protocol)

    def test_scalar_and_vector_additive_fields_have_unit_gradient_fraction(self):
        rng = np.random.default_rng(4)
        potentials = rng.normal(size=(3, self.protocol.n_items))
        scalar = potentials @ self.geometry.incidence.T
        vector_potentials = rng.normal(size=(3, self.protocol.n_items, 5))
        vector = np.einsum("ei,bid->bed", self.geometry.incidence, vector_potentials)
        np.testing.assert_allclose(gradient_energy_fraction(scalar, self.geometry), 1.0)
        np.testing.assert_allclose(
            vector_gradient_energy_fraction(vector, self.geometry), 1.0
        )

    def test_hodge_potential_and_expected_rank_are_same_complete_graph_target(self):
        protocol = RankingProtocol(
            protocol_id="test",
            item_labels=("a", "b", "c", "d"),
            true_order_high_to_low=(0, 1, 2, 3),
            support_pairs_higher_lower=((0, 1),),
            support_blocks=1,
            query_blocks=1,
            human_targets={},
        )
        geometry = build_complete_graph_geometry(protocol)
        exact = ExactRankingPosterior(protocol.n_items)
        rng = np.random.default_rng(8)
        probabilities = rng.dirichlet(np.ones(exact.n_hypotheses), size=5)
        fields = np.asarray(
            [
                [
                    2.0
                    * np.sum(
                        row[exact.positions[:, first] < exact.positions[:, second]]
                    )
                    - 1.0
                    for first, second in geometry.pairs
                ]
                for row in probabilities
            ]
        )
        hodge = normalize_potentials(hodge_potentials(fields, geometry))
        expected_rank = normalize_potentials(-(probabilities @ exact.positions))
        np.testing.assert_allclose(hodge, expected_rank, atol=1e-12)

    def test_numerical_zero_potential_has_no_spurious_direction(self):
        tiny = np.linspace(-1e-15, 1e-15, self.protocol.n_items)
        normalized = normalize_potentials(tiny)
        np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
        alignment = potential_alignment(tiny, self.geometry.true_potential)
        self.assertTrue(np.isnan(alignment["cosine"]))
        self.assertTrue(np.isnan(alignment["kendall_tau"]))

    def test_alignment_reports_direction_and_rank_order(self):
        first = np.asarray([3.0, 2.0, 1.0, 0.0])
        aligned = potential_alignment(first, first)
        reversed_alignment = potential_alignment(first, -first)
        self.assertAlmostEqual(float(aligned["cosine"]), 1.0)
        self.assertAlmostEqual(float(aligned["kendall_tau"]), 1.0)
        self.assertAlmostEqual(float(reversed_alignment["cosine"]), -1.0)
        self.assertAlmostEqual(float(reversed_alignment["kendall_tau"]), -1.0)

    def test_effect_classes_partition_all_complete_graph_edges(self):
        relation = (0, 1)
        fields = np.ones((1, len(self.geometry.pairs)))
        effects = classified_effects(fields, (relation,), self.geometry)
        self.assertEqual(len(tuple(combinations(range(8), 2))), 28)
        for name in ("direct", "endpoint_sharing", "remote"):
            self.assertAlmostEqual(effects["mean_absolute"][name][0], 1.0)
        self.assertEqual(
            sum(
                int(np.sum(mask))
                for mask in (
                    np.asarray(
                        [
                            len(set(relation).intersection(pair)) == overlap
                            for pair in self.geometry.pairs
                        ]
                    )
                    for overlap in (2, 1, 0)
                )
            ),
            28,
        )

    def test_registered_result_contains_both_unfiltered_pilot_seeds(self):
        specification = load_json(
            resolve_record("benchmarks/assembly_trajectory_v1.json")
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)
        result = load_json(resolve_record("results/assembly_trajectory_v1.json"))
        self.assertEqual(set(result["pilot_seeds"]), {"1901", "1902"})
        for row in result["pilot_seeds"].values():
            self.assertIn("raw_subject_level", row)
            self.assertEqual(
                row["matched_zero_evidence_branches"]["stable_omitted_max_abs_effect"],
                0.0,
            )
            self.assertEqual(
                row["leave_one_relation_out"]["stable_omitted_max_abs_pair_influence"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
