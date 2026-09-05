import unittest

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from tools.provenance.main_model_evaluation_v2 import (
    HUMAN_COMPOSITION,
    internal_inversion_counts,
    inversion_bins,
    ranking_total_variation,
    sampled_classes,
)


class MainModelEvaluationV2Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = RankingProtocol(
            protocol_id="synthetic",
            item_labels=tuple(str(i) for i in range(8)),
            true_order_high_to_low=tuple(range(8)),
            support_pairs_higher_lower=tuple((i, i + 1) for i in range(7)),
            support_blocks=1,
            query_blocks=3,
            human_targets={},
        )

    def test_internal_inversions_use_all_28_strict_pair_relations(self):
        codes = np.broadcast_to(
            np.eye(8, 15, dtype=np.float64)[None, :, :], (77, 8, 15)
        ).copy()
        weights = np.zeros((2, 77, 15), dtype=np.float64)
        weights[0, :, :8] = np.arange(8, 0, -1)
        weights[1, :, :8] = np.arange(8, 0, -1)
        weights[1, :, [0, 1]] = weights[1, :, [1, 0]]
        counts = internal_inversion_counts(weights, codes, self.protocol)
        np.testing.assert_array_equal(counts[0], 0)
        np.testing.assert_array_equal(counts[1], 1)
        np.testing.assert_array_equal(inversion_bins(counts)[0], 0)
        np.testing.assert_array_equal(inversion_bins(counts)[1], 1)

    def test_ranking_total_variation_is_joint_composition_distance(self):
        self.assertEqual(ranking_total_variation(HUMAN_COMPOSITION), 0.0)
        actual = ranking_total_variation(np.asarray([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(actual, 1.0 - HUMAN_COMPOSITION[0])
        with self.assertRaises(ValueError):
            ranking_total_variation(np.asarray([0.5, 0.5]))

    def test_narrow_sampled_classifier_matches_registered_authority(self):
        rng = np.random.default_rng(910_001)
        margins = rng.normal(size=(1, 77, 56))
        classes, eligible = sampled_classes(
            margins,
            self.protocol,
            choice_seed=920_001,
            temperature=0.25,
        )
        schedules = (ordered_pairs(8),) * 77
        behavior = analyze_sampled_query_policy(
            self.protocol,
            bundle_logits({"logits": margins[0]}, schedules),
            seed=920_001,
            temperature=0.25,
        )
        expected_classes = np.asarray(
            [
                ("correct", "self_consistent_incorrect", "self_inconsistent").index(
                    row["ranking_class"]
                )
                for row in behavior["subjects"]
            ]
        )
        expected_eligible = np.asarray(
            [row["overall_accuracy"] >= 0.5 for row in behavior["subjects"]]
        )
        np.testing.assert_array_equal(classes[0], expected_classes)
        np.testing.assert_array_equal(eligible[0], expected_eligible)


if __name__ == "__main__":
    unittest.main()
