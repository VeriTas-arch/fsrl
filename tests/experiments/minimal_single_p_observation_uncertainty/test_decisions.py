from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.minimal_single_p_observation_uncertainty.decisions import (
    expected_direction_supported,
    paired_contrast,
    study_outcome,
)


class DecisionTests(unittest.TestCase):
    def test_paired_directions(self):
        high = np.arange(20, dtype=float) + 1
        low = np.arange(20, dtype=float)
        positive = paired_contrast(high, low, seed=1, samples=100)
        negative = paired_contrast(low, high, seed=2, samples=100)
        self.assertTrue(
            expected_direction_supported("stable_within_subject_errors", positive)
        )
        self.assertTrue(
            expected_direction_supported("inter_subject_ranking_diversity", negative)
        )

    def test_nonfinite_is_not_dropped(self):
        result = paired_contrast(
            np.asarray([1.0, np.nan]),
            np.asarray([0.0, 0.0]),
            seed=1,
            samples=10,
        )
        self.assertFalse(result["defined"])

    def test_outcome_precedence(self):
        self.assertEqual(study_outcome(0, True, True), "no_acute_completion")
        self.assertEqual(
            study_outcome(1, True, True), "direction_specific_acute_completion"
        )
        self.assertEqual(
            study_outcome(1, True, False),
            "acute_completion_without_direction_specificity",
        )


if __name__ == "__main__":
    unittest.main()
