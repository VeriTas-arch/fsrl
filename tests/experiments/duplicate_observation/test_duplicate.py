"""Same-q duplication preserves observations; summed columns preserve computation."""

import unittest

import numpy as np

from fsrl.experiments.duplicate_observation.inputs import (
    duplicate_observation,
    observed,
)
from fsrl.experiments.duplicate_observation.protocol import parents, recipe
from fsrl.experiments.duplicate_observation.qualification import numerical_checks
from fsrl.experiments.duplicate_observation.reporting import (
    decision,
    improvement_decision,
)
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.observation_uncertainty.inputs import attach_noise
from fsrl.experiments.training_strategy.batches import sample_episodes


class DuplicateTests(unittest.TestCase):
    def test_only_hint_changes_in_both_observation_conditions(self):
        spec = recipe()
        cpu = attach_noise(
            prepare_shared(
                sample_episodes(generator(spec), np.random.default_rng(95101), 3)
            ),
            41,
        )
        original_hash = cpu.fingerprint()
        for arm in ("clean", "noisy"):
            full = observed(cpu, arm, spec, keep_hint=True)
            candidate = observed(cpu, arm, spec)
            np.testing.assert_array_equal(
                candidate.arrays["support_inputs"][..., 34],
                candidate.arrays["support_inputs"][..., 37],
            )
            expected = full.arrays["support_inputs"].copy()
            expected[..., 34] = expected[..., 37]
            for name, actual in candidate.arrays.items():
                np.testing.assert_array_equal(
                    actual, expected if name == "support_inputs" else full.arrays[name]
                )
            self.assertEqual(
                duplicate_observation(candidate).fingerprint(), candidate.fingerprint()
            )
        self.assertEqual(cpu.fingerprint(), original_hash)

    def test_duplicate_forward_equal_gradients_update_and_column_sum_mapping(self):
        self.assertTrue(all(row["passed"] for row in numerical_checks().values()))

    def test_decision_matches_archived_support_rule(self):
        for row in parents()["result"]["pairs"].values():
            panels = {
                key: {"candidate": value["single"]}
                for key, value in row["panels"].items()
            }
            actual = decision(
                panels,
                row["single_equal_panel_mean"],
                row["noninferiority_equal_panel_mean"],
            )
            expected = {
                key.replace("all_single", "all_candidate"): value
                for key, value in row["decision"].items()
            }
            self.assertEqual(actual, expected)

    def test_noninferiority_does_not_imply_direct_improvement(self):
        summary = {
            "uncertain": {"interval": {"lower": -0.003, "upper": 0.004}},
            "worse": {"interval": {"lower": -0.01, "upper": -0.002}},
            "better": {"interval": {"lower": 0.001, "upper": 0.02}},
        }
        actual = improvement_decision(summary)
        self.assertEqual(actual["uncertain"], {"improved": False, "worsened": False})
        self.assertEqual(actual["worse"], {"improved": False, "worsened": True})
        self.assertEqual(actual["better"], {"improved": True, "worsened": False})
