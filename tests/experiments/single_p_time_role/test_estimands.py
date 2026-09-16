import unittest

import numpy as np

from fsrl.experiments.single_p_time_role.analysis import generic_endpoints
from fsrl.experiments.single_p_time_role.estimands import (
    canonical_field,
    derangement,
    fixed_effect_slope,
    hodge,
    positive_scale,
)


class TimeRoleEstimandTests(unittest.TestCase):
    def test_canonical_field_antisymmetrizes_both_orders(self):
        pairs = np.asarray([(0, 1), (1, 0)])
        margins = np.asarray([3.0, -1.0])
        with self.assertRaises(ValueError):
            canonical_field(margins, pairs, n_items=3)

    def test_hodge_reconstructs_field(self):
        field = np.arange(28, dtype=float)[None, :]
        result = hodge(field)
        np.testing.assert_allclose(result["gradient"] + result["residual"], field)

    def test_positive_scale_is_single_nonnegative_degree_of_freedom(self):
        control = np.asarray([1.0, -2.0, 3.0])
        self.assertAlmostEqual(positive_scale(0.25 * control, control), 4.0)
        self.assertEqual(positive_scale(-control, control), 0.0)

    def test_derangement_is_deterministic_and_has_no_fixed_point(self):
        first = derangement(28, 17)
        np.testing.assert_array_equal(first, derangement(28, 17))
        self.assertFalse(np.any(first == np.arange(28)))

    def test_fixed_effect_slope_removes_prefix_means(self):
        prefix = np.tile(np.arange(3), 5)
        maturity = np.arange(15, dtype=float) % 5 + prefix * 10
        outcome = -1.5 * maturity + prefix * 100
        self.assertAlmostEqual(fixed_effect_slope(outcome, maturity, prefix), -1.5)

    def test_generic_endpoints_accept_subject_by_query_targets(self):
        margins = np.asarray([[1.0, -1.0], [2.0, -2.0]])
        targets = np.asarray([[1, 0], [1, 0]])
        learned = np.asarray([[True, False], [True, False]])
        result = generic_endpoints(margins, targets, learned)
        self.assertTrue(np.all(result["generic_learned"] > 0.5))
        self.assertTrue(np.all(result["generic_nonlearned"] > 0.5))


if __name__ == "__main__":
    unittest.main()
