"""Known coupled budgets and independent per-phase write accounting."""

import unittest

import numpy as np

from fsrl.experiments.write_budget_match.budgets import means, solve
from fsrl.experiments.write_budget_match.qualification import numerical_checks
from fsrl.experiments.write_budget_match.reporting import budget_effect, paired_effect


class BudgetTests(unittest.TestCase):
    def test_paired_effect_and_weighted_budget_ratio(self):
        old = np.array([[0, 0, 1, 4], [0, 0, 3, 2]], dtype=float)
        point, draws = budget_effect(old * 1.03, old, np.array([1, 0.5]), 823)
        np.testing.assert_allclose(point, 1.03)
        np.testing.assert_allclose(draws, 1.03)
        point, draws = paired_effect(np.array([0.3, 0.5]), np.array([0.1, 0.3]), 823)
        self.assertAlmostEqual(point, 0.2)
        np.testing.assert_allclose(draws, 0.2)

    def test_phase_l1_and_clipping_use_old_eligibility(self):
        self.assertTrue(all(v["passed"] for v in numerical_checks().values()))

    def test_solver_recovers_coupled_gains_without_outcome_labels(self):
        config = {
            "difference_step": 0.02,
            "gain_limit": 64,
            "max_nfev": 24,
            "tolerance": 1e-5,
            "budget_tolerance": 0.05,
        }

        def evaluate(g):
            return np.tile([g[0], g[0] * g[1]], (2, 1))

        result = solve(evaluate, np.tile([2, 6], (2, 1)), config)
        self.assertTrue(result["matched"])
        np.testing.assert_allclose(result["gain"], [2, 3], rtol=1e-4)

    def test_pooled_match_cannot_hide_opposite_observation_errors(self):
        config = {
            "difference_step": 0.02,
            "gain_limit": 64,
            "max_nfev": 24,
            "tolerance": 1e-5,
            "budget_tolerance": 0.05,
        }
        result = solve(lambda g: np.array([g, 2 * g]), np.ones((2, 2)), config)
        self.assertFalse(result["matched"])

    def test_budget_weights_archives_equally_and_preserves_observations(self):
        a = {"writes": np.ones((2, 4, 1))}
        b = {"writes": np.full((2, 4, 3), 3)}
        value = means({"clean/a": a, "clean/b": b, "noisy/a": a})
        np.testing.assert_array_equal(value, [[4, 4], [2, 2]])


if __name__ == "__main__":
    unittest.main()
