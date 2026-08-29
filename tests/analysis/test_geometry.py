import unittest

import numpy as np

from fsrl.analysis.geometry import (
    analyze_item_geometry,
    antisymmetric_hodge_item_representations,
    context_averaged_item_representations,
    evaluate_geometry_gate,
    rank_positions,
    ranking_rdm,
    rdm_spearman,
    representation_rdm,
)
from fsrl.tasks.protocol_catalog import load_registered_protocol


class GeometryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_registered_protocol("liu_v1")

    def test_context_average_requires_both_pair_orientations(self):
        hidden = {}
        for first in range(8):
            for second in range(8):
                if first != second:
                    hidden[(first, second)] = np.asarray([first, second], dtype=float)
        representations = context_averaged_item_representations(hidden, 8)
        self.assertEqual(representations.shape, (8, 2))

    def test_antisymmetric_hodge_recovers_centered_item_vectors(self):
        rng = np.random.default_rng(91)
        item_vectors = rng.normal(size=(8, 3))
        hidden = {}
        for first in range(8):
            for second in range(8):
                if first != second:
                    hidden[(first, second)] = item_vectors[first] - item_vectors[second]
        recovered = antisymmetric_hodge_item_representations(hidden, 8)
        np.testing.assert_allclose(
            recovered,
            item_vectors - np.mean(item_vectors, axis=0, keepdims=True),
            atol=1e-12,
        )

    def test_subjective_rank_geometry_beats_true_for_wrong_subjects(self):
        orders = [
            [0, 2, 1, 3, 4, 5, 6, 7],
            [0, 1, 3, 2, 4, 5, 6, 7],
            [0, 1, 2, 4, 3, 5, 6, 7],
        ]
        behavior = {
            "subjects": [
                {
                    "subject": index,
                    "ranking_class": "self_consistent_incorrect",
                    "overall_accuracy": 0.9,
                    "subjective_order_high_to_low": order,
                }
                for index, order in enumerate(orders)
            ]
        }
        representations = tuple(
            -rank_positions(order)[:, None].astype(float) for order in orders
        )
        result = analyze_item_geometry(self.protocol, representations, behavior)
        self.assertEqual(result["group"]["subjects"], 3)
        self.assertGreater(result["group"]["mean_subjective_minus_true"], 0.0)

    def test_geometry_rank_reuse_preserves_pairwise_spearman_values_exactly(self):
        rng = np.random.default_rng(92)
        orders = []
        for subject in range(12):
            order = list(range(8))
            first = 1 + subject % 6
            order[first], order[first + 1] = order[first + 1], order[first]
            orders.append(order)
        behavior = {
            "subjects": [
                {
                    "subject": subject,
                    "ranking_class": "self_consistent_incorrect",
                    "overall_accuracy": 0.7,
                    "subjective_order_high_to_low": order,
                }
                for subject, order in enumerate(orders)
            ]
        }
        representations = tuple(rng.normal(size=(8, 20)) for _ in orders)
        result = analyze_item_geometry(self.protocol, representations, behavior)
        true_rdm = ranking_rdm(rank_positions(self.protocol.true_order_high_to_low))
        subjective_rdms = tuple(ranking_rdm(rank_positions(order)) for order in orders)
        for subject, row in enumerate(result["subjects"]):
            neural_rdm = representation_rdm(representations[subject])
            self.assertEqual(
                row["rho_neural_subjective"],
                rdm_spearman(neural_rdm, subjective_rdms[subject]),
            )
            self.assertEqual(row["rho_neural_true"], rdm_spearman(neural_rdm, true_rdm))
            self.assertEqual(
                row["rho_neural_other_subjective_mean"],
                float(
                    np.mean(
                        [
                            rdm_spearman(neural_rdm, subjective_rdms[other])
                            for other in range(len(orders))
                            if other != subject
                        ]
                    )
                ),
            )

    def test_gate_requires_subject_count_direction_and_sign_test(self):
        result = {
            "group": {
                "subjects": 12,
                "mean_subjective_minus_true": 0.1,
                "mean_subjective_minus_other": 0.08,
                "one_sided_sign_test_p": 0.01,
            }
        }
        specification = {
            "geometry_gate_id": "test",
            "registration_status": "exploratory-test",
            "minimum_subjects": 10,
            "minimum_mean_subjective_minus_true_spearman": 0.0,
            "minimum_mean_subjective_minus_other_spearman": 0.0,
            "maximum_one_sided_sign_test_p": 0.05,
        }
        report = evaluate_geometry_gate(result, specification)
        self.assertTrue(report["passed"])
        self.assertEqual(report["registration_status"], "exploratory-test")
        result["group"]["mean_subjective_minus_true"] = -0.01
        self.assertFalse(evaluate_geometry_gate(result, specification)["passed"])
