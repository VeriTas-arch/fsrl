import unittest

import numpy as np

from fsrl.analysis.statistics import bootstrap_mean_interval, correlation_or_zero


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_mean_interval_is_seed_deterministic(self):
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        first = bootstrap_mean_interval(values, 100, 17)
        second = bootstrap_mean_interval(values, 100, 17)
        self.assertEqual(first, second)
        self.assertEqual(first["point"], 3.75)

    def test_correlation_or_zero_handles_constant_and_varying_inputs(self):
        self.assertEqual(correlation_or_zero(np.ones(3), np.arange(3)), 0.0)
        self.assertAlmostEqual(correlation_or_zero(np.arange(3), np.arange(3)), 1.0)
