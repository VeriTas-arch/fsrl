import unittest

import numpy as np

from fsrl.experiments.single_p_time_role.analysis import (
    generic_endpoints,
    paired_interval,
)
from fsrl.experiments.single_p_time_role.estimands import (
    canonical_field,
    derangement,
    fixed_effect_slope,
    hodge,
    positive_scale,
)
from fsrl.experiments.single_p_time_role.trajectory import _canonical_correct_signs
from fsrl.experiments.training_strategy.batches import EpisodeBatch


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

    def test_generic_correct_signs_follow_random_query_orientations(self):
        canonical = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        pairs = np.asarray(canonical, dtype=np.int64)[:, None, :]
        pairs[::2] = pairs[::2, :, ::-1]
        canonical_signs = np.where(np.arange(28) % 3 == 0, -1.0, 1.0)
        oriented_signs = canonical_signs.copy()
        oriented_signs[::2] *= -1.0
        targets = ((oriented_signs + 1.0) / 2.0).astype(np.int64)
        batch = EpisodeBatch(
            {
                "item_codes": np.zeros((1, 8, 15), dtype=np.float32),
                "query_pairs": pairs,
                "targets": targets,
            }
        )
        np.testing.assert_array_equal(
            _canonical_correct_signs(batch), canonical_signs[None, :]
        )

    def test_paired_interval_uses_complete_cases_within_panel(self):
        result = paired_interval(
            [np.asarray([np.nan, 1.0]), np.asarray([3.0, np.nan])],
            seed=1,
            draws=20,
        )
        self.assertEqual(result["point"], 2.0)
        self.assertTrue(np.all(np.isfinite(tuple(result["interval"].values()))))


if __name__ == "__main__":
    unittest.main()
