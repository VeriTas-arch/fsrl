"""Changed update timing and generic-only schedule estimators."""

import unittest

import numpy as np

from fsrl.experiments.modulation_schedule.audit import experience_features, fit_schedule
from fsrl.experiments.modulation_schedule.qualification import numerical_checks
from fsrl.experiments.modulation_schedule.reporting import write_effects
from fsrl.experiments.training_strategy.batches import EpisodeBatch


class ScheduleTests(unittest.TestCase):
    def test_write_change_uses_paired_subjects_and_float64(self):
        old = np.array([1, 2, 4], dtype=np.float32)
        new = old * 2
        a = {"A0": {task: {"total_write": new} for task in ("generic", "liu")}}
        b = {"A0": {task: {"total_write": old} for task in ("generic", "liu")}}
        points, draws = write_effects(a, b, 7123, 100)
        self.assertEqual(points["A0/generic/write_ratio"], 2)
        np.testing.assert_array_equal(draws["A0/generic/write_ratio"], 2)
        np.testing.assert_array_equal(
            draws["A0/generic/write_difference"], draws["A0/liu/write_difference"]
        )

    def test_old_eligibility_and_intact_computation(self):
        self.assertTrue(all(v["passed"] for v in numerical_checks().values()))

    def test_history_uses_only_preceding_oriented_observations(self):
        cpu = EpisodeBatch(
            {
                "local_evidence": np.array([[0.2], [-0.4], [0.9]]),
                "support_pairs": np.array([[[0, 1]], [[1, 0]], [[0, 1]]]),
            }
        )
        _, count, gap = experience_features(cpu)
        np.testing.assert_array_equal(count[:, 0], [0, 1, 2])
        np.testing.assert_allclose(gap[:, 0], [0, 0.2, 0.6])

    def test_phase_calibration_recovers_distinct_signed_constants(self):
        batches = {}
        for arm in ("clean", "noisy"):
            for name in ("test-28", "test-32", "test-36", "test-40"):
                raw = {
                    "trial": np.array([0, 0, 1, 1]),
                    "phase": np.array([2, 3, 2, 3]),
                    "m": np.array([[-2.0], [3.0], [-2.0], [3.0]]),
                    "potential": np.ones((4, 1)),
                }
                raw.update(
                    {
                        key: np.zeros((4, 1))
                        for key in ("q", "time", "repetitions", "discrepancy")
                    }
                )
                batches[arm + "/" + name] = raw
        result = fit_schedule(batches)
        np.testing.assert_array_equal(result["phase"], [0, 0, -2, 3])
        np.testing.assert_array_equal(result["constant"], [0.5] * 4)


if __name__ == "__main__":
    unittest.main()
