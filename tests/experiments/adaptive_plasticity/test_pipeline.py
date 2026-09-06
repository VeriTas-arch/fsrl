import unittest

import numpy as np

from fsrl.experiments.adaptive_plasticity.generic_selection import (
    MODES,
    SCHEDULES,
    summarize,
)
from fsrl.experiments.adaptive_plasticity.inputs import cohort_indices
from fsrl.experiments.adaptive_plasticity.protocol import (
    CONDITIONS,
    SEEDS,
    cohort_specification,
    resolved_specification,
    specification,
)


class AdaptivePlasticityPipelineTests(unittest.TestCase):
    def test_frozen_authorities_and_matrix(self):
        contract = specification()
        self.assertEqual(
            contract["candidate"]["slow_parameters"], ["raw_eta0", "raw_global_gain"]
        )
        self.assertEqual(SEEDS, (2117, 2118, 2119))
        self.assertEqual(CONDITIONS, ("fixed_eta_resampled", "adaptive_eta_resampled"))
        spec = resolved_specification()
        self.assertEqual(spec["optimization"]["total_steps"], 1500)
        self.assertEqual(spec["task"]["support_blocks"], 4)

    def test_cohort_seed_ranges_are_disjoint(self):
        used = set()
        for index in range(400):
            settings = cohort_specification(index)["evaluation"]["liu"]
            groups = [
                [settings["cue_seed"]],
                range(settings["support_seed"], settings["support_seed"] + 77),
                [settings["subject_encoding_seed"]],
                [settings["encoding_seed"]],
                range(settings["choice_seed"], settings["choice_seed"] + 154),
            ]
            for group in groups:
                self.assertFalse(used.intersection(group))
                used.update(group)
        self.assertGreater(min(used), 3_000_000)
        self.assertLess(max(used), 4_800_000)

    def test_unregistered_cohorts_and_shards_are_rejected(self):
        for index in (-1, 400):
            with self.assertRaises(ValueError):
                cohort_specification(index)
        for start in (-1, 1, 400):
            with self.assertRaises(ValueError):
                cohort_indices(start)
        np.testing.assert_array_equal(cohort_indices(380), np.arange(380, 400))

    def test_generic_selection_prefers_relation_only_with_both_controls(self):
        shape = (3, 2, 3, 256)
        arrays = {
            name: np.full(shape, 0.8)
            for name in ("learned_accuracy", "nonlearned_accuracy")
        }
        arrays["loss"] = np.full(shape, 0.5)
        arrays["h_late"] = np.full(shape, 0.5)
        for name in (
            "conditional_mean_signed_margin",
            "mean_margin_variance",
            "mean_signal_to_variance",
        ):
            arrays[name] = np.ones(shape)
        for schedule in range(len(SCHEDULES)):
            arrays["h_late"][:, schedule, MODES.index("fixed")] = 0.6
            arrays["h_late"][:, schedule, MODES.index("adaptive_relation")] = 0.2
            arrays["h_late"][:, schedule, MODES.index("adaptive_global")] = 0.3
        clustered = SCHEDULES.index("clustered")
        arrays["loss"][:, clustered, MODES.index("adaptive_relation")] = 0.2
        arrays["loss"][:, clustered, MODES.index("adaptive_global")] = 0.3
        result = summarize(arrays)
        self.assertEqual(result["selected_scheduler"], "relation")
        self.assertTrue(result["relation_specific_supported"])
        self.assertTrue(result["generic_gate_passed"])


if __name__ == "__main__":
    unittest.main()
