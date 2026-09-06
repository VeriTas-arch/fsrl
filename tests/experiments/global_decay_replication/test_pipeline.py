import unittest

import numpy as np

from fsrl.experiments.global_decay_replication.cohorts import summarize_points
from fsrl.experiments.global_decay_replication.generic import MODES, summarize
from fsrl.experiments.global_decay_replication.inputs import cohort_indices
from fsrl.experiments.global_decay_replication.protocol import COHORTS, SEEDS


class GlobalDecayReplicationPipelineTests(unittest.TestCase):
    def test_cohort_shards_are_complete_and_bounded(self):
        with self.assertRaises(ValueError):
            cohort_indices(1)
        self.assertEqual(cohort_indices(380), list(range(380, 400)))

    def test_generic_gate_requires_every_seed_and_both_conditions(self):
        shape = (len(SEEDS), len(MODES), 256)
        arrays: dict[str, np.ndarray] = {
            name: np.full(shape, 0.8)
            for name in ("learned_accuracy", "nonlearned_accuracy")
        }
        for name in (
            "loss",
            "sampled_mean_signed_margin",
            "conditional_mean_signed_margin",
            "mean_margin_variance",
            "mean_signal_to_variance",
        ):
            arrays[name] = np.ones(shape)
        arrays["h_late"] = np.full(shape, 0.6)
        arrays["h_late"][:, MODES.index("adaptive_eta_resampled")] = 0.2
        arrays["occurrence_l"] = np.ones((*shape, 4))
        arrays["max_reference_error"] = np.zeros((len(SEEDS), len(MODES)))
        result = summarize(arrays)
        self.assertTrue(result["generic_gate_passed"])
        arrays["h_late"][1, MODES.index("adaptive_eta_resampled")] = 0.9
        self.assertFalse(summarize(arrays)["generic_gate_passed"])

    def test_liu_summary_requires_all_registered_cohorts(self):
        with self.assertRaises(RuntimeError):
            summarize_points([])
        self.assertEqual(COHORTS, 400)


if __name__ == "__main__":
    unittest.main()
