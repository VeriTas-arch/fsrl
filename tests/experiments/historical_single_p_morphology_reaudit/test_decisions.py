from __future__ import annotations

import unittest

from fsrl.experiments.historical_single_p_morphology_reaudit.decisions import (
    constrained_unit,
    error_inflation,
    family_replicated,
    outcome,
)


class DecisionTests(unittest.TestCase):
    def test_constrained_threshold_is_inclusive_and_rejects_bad_pair(self):
        common = {
            "competent": True,
            "binding": True,
            "nine": True,
            "latent_bimodal": 15,
            "sampled_bimodal": 15,
        }
        self.assertTrue(constrained_unit(**common, bad_pair=False))
        self.assertFalse(constrained_unit(**common, bad_pair=True))
        self.assertFalse(
            constrained_unit(**{**common, "latent_bimodal": 14}, bad_pair=False)
        )

    def test_error_inflation_requires_increase_and_preservation_failure(self):
        self.assertTrue(
            error_inflation(
                sampled_bimodal=16,
                control_sampled_bimodal=15,
                nine=False,
                bad_pair=False,
            )
        )
        self.assertFalse(
            error_inflation(
                sampled_bimodal=16,
                control_sampled_bimodal=15,
                nine=True,
                bad_pair=False,
            )
        )
        self.assertFalse(
            error_inflation(
                sampled_bimodal=15,
                control_sampled_bimodal=15,
                nine=False,
                bad_pair=True,
            )
        )

    def test_family_replication_and_outcomes(self):
        units = [{"seed": seed, "constrained_morphology": True} for seed in (1, 2, 3)]
        self.assertTrue(family_replicated(units, [1, 2, 3]))
        self.assertEqual(outcome(0, 0), "no_current_standard_precedent")
        self.assertEqual(outcome(1, 0), "isolated_current_standard_precedent")
        self.assertEqual(outcome(3, 1), "single_family_current_standard_precedent")
        self.assertEqual(outcome(6, 2), "cross_family_current_standard_precedent")


if __name__ == "__main__":
    unittest.main()
