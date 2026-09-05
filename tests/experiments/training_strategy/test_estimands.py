import unittest

import numpy as np

from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts, summarize_subjects
from fsrl.experiments.training_strategy.estimands import (
    estimate,
    paired_estimate,
    query_endpoints,
    subject_means,
)


class RegisteredEstimandTests(unittest.TestCase):
    def setUp(self):
        self.statistics = {"samples": 1000, "interval": 0.95}

    def test_sigmoid_precedes_orientation_average(self):
        margins = np.asarray([[[10.0, 1.0], [0.0, 0.0]]])
        groups = {"overall": np.ones(2, dtype=bool)}
        result = query_endpoints(margins, [1, -1], groups, temperature=0.25)
        expected = exact_probability(margins * [1, -1], 0.25).mean()
        self.assertAlmostEqual(result["probability"]["overall"][0], expected)
        wrong = exact_probability((margins * [1, -1]).mean(2), 0.25).mean()
        self.assertGreater(abs(expected - wrong), 0.1)
        # One correct, one incorrect, and two exact ties contribute equally.
        self.assertEqual(result["exact_decision"]["overall"][0], 0.5)

    def test_participants_have_equal_weight_despite_unequal_group_sizes(self):
        values = np.asarray([[1.0, 0.0, 0.0], [0.3, 0.6, 0.9], [0.0, 0.0, 0.0]])
        mask = np.asarray([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
        rows = subject_means(values, mask)
        np.testing.assert_allclose(rows, [1.0, 0.6, np.nan], equal_nan=True)
        result = estimate(rows, seed=12, statistics=self.statistics)
        self.assertAlmostEqual(result["mean"], 0.8)
        self.assertEqual(result["subjects"], 2)
        self.assertEqual(result["total_subjects"], 3)
        self.assertEqual(result["excluded_subject_indices"], [2])

    def test_complete_case_subset_is_selected_before_paired_resampling(self):
        first = np.asarray([0.4, np.nan, 0.8, 0.7, 0.2])
        second = np.asarray([0.1, 0.9, 0.2, np.nan, 0.9])
        result = paired_estimate(first, second, seed=51, statistics=self.statistics)
        delta = np.asarray([0.3, 0.6, -0.7])
        counts = bootstrap_counts(np.random.default_rng(51), 1000, 3)
        expected = summarize_subjects(delta, counts, interval=0.95)
        for name in ("mean", "lower", "upper"):
            self.assertAlmostEqual(
                result["bootstrap"][name], expected["bootstrap"][name]
            )
        self.assertEqual(result["excluded_subject_indices"], [1, 3])
        repeated = paired_estimate(first, second, seed=51, statistics=self.statistics)
        self.assertEqual(result, repeated)

    def test_empty_endpoint_is_explicit_and_undefined(self):
        result = estimate(np.full(3, np.nan), seed=1, statistics=self.statistics)
        self.assertEqual(result["subjects"], 0)
        self.assertIsNone(result["bootstrap"]["lower"])
        self.assertEqual(result["excluded_subject_indices"], [0, 1, 2])

    def test_invalid_axes_signs_and_infinite_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            query_endpoints(np.zeros((2, 3)), 1, {}, temperature=1.0)
        with self.assertRaises(ValueError):
            query_endpoints(np.zeros((2, 3, 2)), 0, {}, temperature=1.0)
        with self.assertRaises(ValueError):
            estimate(np.asarray([np.inf]), seed=1, statistics=self.statistics)
        with self.assertRaises(ValueError):
            paired_estimate(np.ones(2), np.ones(3), seed=1, statistics=self.statistics)
