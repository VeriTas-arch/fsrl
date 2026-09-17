"""Exhaustive outcome precedence and endpoint-boundary tests."""

from __future__ import annotations

import unittest

from fsrl.experiments.single_p_anytime.decisions import classify, interval_gate


class AnytimeDecisionTests(unittest.TestCase):
    def test_outcome_precedence(self):
        base = {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": True,
        }
        self.assertEqual(classify(**base), "anytime_admitted")
        self.assertEqual(
            classify(**{**base, "short_improved": False}), "horizon_tradeoff"
        )
        self.assertEqual(
            classify(**{**base, "short_improved": False, "long_improved": False}),
            "no_horizon_benefit",
        )
        self.assertEqual(
            classify(**{**base, "anytime_competent": False}),
            "anytime_competence_failure",
        )
        self.assertEqual(
            classify(**{**base, "historical_preserved": False}),
            "historical_task_degradation",
        )
        self.assertEqual(
            classify(**{**base, "fresh_fixed_valid": False}),
            "training_recipe_failure",
        )
        self.assertEqual(classify(**{**base, "integrity": False}), "noninterpretable")

    def test_ce_is_strict_and_probability_margin_is_inclusive(self):
        self.assertTrue(interval_gate(ce_upper=-1e-9, probability_lower=-0.02))
        self.assertFalse(interval_gate(ce_upper=0.0, probability_lower=-0.02))
        self.assertFalse(interval_gate(ce_upper=-1e-9, probability_lower=-0.020001))


if __name__ == "__main__":
    unittest.main()
