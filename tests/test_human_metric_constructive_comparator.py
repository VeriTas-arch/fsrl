import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fsrl.human_metric_constructive_comparator import (
    DEFAULT_SPECIFICATION_PATH,
    behavioral_components,
    bootstrap_counts,
    decide,
    distribution_and_field,
    efficient_inter_subject_tau,
    model_arrays,
    pair_metadata,
    subject_log_likelihood_matrix,
    write_json_exclusive,
)
from fsrl.ranking_protocol import load_ranking_protocol


class HumanMetricConstructiveComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = json.loads(DEFAULT_SPECIFICATION_PATH.read_text())
        protocol_path = (
            DEFAULT_SPECIFICATION_PATH.parents[1]
            / cls.specification["registered_sources"]["liu_protocol"]["path"]
        )
        cls.protocol = load_ranking_protocol(protocol_path)
        cls.metadata = pair_metadata(cls.protocol)
        cls.arrays = model_arrays(cls.protocol, cls.specification)

    def test_task_geometry_preserves_metric_support_and_twenty_pair_contract(self):
        self.assertEqual(self.arrays["orders"].shape, (40320, 8))
        self.assertEqual(self.arrays["masks"].shape, (256, 8))
        self.assertEqual(self.arrays["energies"].shape, (256, 40320))
        self.assertEqual(int(np.sum(self.metadata["selected"])), 20)
        self.assertAlmostEqual(float(np.mean(self.metadata["selected_distances"])), 2.8)
        self.assertAlmostEqual(
            float(
                np.sum(
                    (self.metadata["selected_distances"] - 2.8) ** 2
                )
            ),
            57.2,
        )
        np.testing.assert_allclose(
            self.arrays["support_magnitudes"],
            np.asarray([5, 1, 3, 4, 2, 3, 3, 7]) / 7.0,
            atol=0.0,
            rtol=0.0,
        )

    def test_stable_access_masks_have_exact_energy_endpoints(self):
        np.testing.assert_allclose(self.arrays["energies"][0], 0.0)
        np.testing.assert_allclose(
            self.arrays["energies"][-1],
            np.sum(self.arrays["relation_residual_sq"], axis=1),
            atol=5e-15,
            rtol=0.0,
        )
        self.assertEqual(len(np.unique(self.arrays["masks"], axis=0)), 256)

    def test_marginal_order_distribution_and_pair_field_are_probabilities(self):
        order, field, identities = distribution_and_field(
            np.asarray([0.5, 0.1, 0.05]), self.arrays, device="cpu"
        )
        self.assertEqual(order.shape, (40320,))
        self.assertEqual(field.shape, (28,))
        self.assertLessEqual(identities["order_probability_sum_abs_error"], 1e-12)
        self.assertGreaterEqual(float(np.min(order)), 0.0)
        self.assertTrue(np.all((field >= 0.0) & (field <= 1.0)))

    def test_subject_likelihood_matrix_matches_direct_binomial_calculation(self):
        counts = np.asarray([[8.0, 3.0]])
        orientations = np.asarray([[True, False], [False, True]])
        epsilon = 0.1
        observed = subject_log_likelihood_matrix(counts, orientations, epsilon)[0]
        expected = []
        for row in orientations:
            value = 0.0
            for count, correct in zip(counts[0], row, strict=True):
                probability = 1.0 - epsilon if correct else epsilon
                value += math.log(math.comb(10, int(count)))
                value += count * math.log(probability)
                value += (10.0 - count) * math.log1p(-probability)
            expected.append(value)
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0.0)

    def test_committed_order_mixture_is_not_product_of_pair_marginals(self):
        # Two equally likely orders make both pairs marginally 0.5, but only
        # the coherent all-correct/all-wrong patterns can occur without lapse.
        order_correct = np.asarray([[True, True], [False, False]])
        counts = np.asarray([[10.0, 10.0]])
        log_matrix = subject_log_likelihood_matrix(counts, order_correct, 0.01)[0]
        joint = float(np.sum(0.5 * np.exp(log_matrix)))
        independent_marginal = 0.5**2
        self.assertGreater(joint, independent_marginal)

    def test_behavioral_components_recover_one_perfect_coherent_rank(self):
        counts = np.full((3, 28), 10.0)
        result = behavioral_components(counts, self.arrays)
        self.assertEqual(result["metrics"]["eligible_subjects"], 3)
        self.assertEqual(result["metrics"]["analysis_subjects"], 0)
        self.assertEqual(result["metrics"]["ranking_class_counts"]["correct"], 3)
        self.assertEqual(
            result["metrics"]["mean_self_consistency_coefficient"], 1.0
        )

    def test_efficient_tau_matches_explicit_pairwise_rank_comparison(self):
        positions = np.asarray(
            [
                [0, 1, 2, 3],
                [0, 2, 1, 3],
                [3, 2, 1, 0],
            ]
        )
        observed = efficient_inter_subject_tau(positions)
        explicit = []
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                products = []
                for left in range(4):
                    for right in range(left + 1, 4):
                        products.append(
                            np.sign(positions[first, left] - positions[first, right])
                            * np.sign(
                                positions[second, left] - positions[second, right]
                            )
                        )
                explicit.append(np.mean(products))
        self.assertAlmostEqual(observed, float(np.mean(explicit)))

    def test_bootstrap_counts_are_deterministic_and_complete(self):
        first = bootstrap_counts(17, 37, 123)
        second = bootstrap_counts(17, 37, 123)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sum(first, axis=1), np.full(17, 37))

    def test_decision_tree_reserves_pass_for_field_and_individual_conjunction(self):
        self.assertEqual(
            decide(True, True, True, True)["outcome"],
            "metric_constructive_comparator_externally_adequate",
        )
        self.assertEqual(
            decide(True, True, False, True)["outcome"],
            "field_adequate_individual_qualification_failed",
        )
        self.assertEqual(
            decide(False, False, True, True)["outcome"],
            "metric_constructive_comparator_externally_inadequate",
        )
        self.assertEqual(decide(True, True, True, False)["outcome"], "noninterpretable")

    def test_exclusive_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json_exclusive(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"status": "second"})


if __name__ == "__main__":
    unittest.main()
