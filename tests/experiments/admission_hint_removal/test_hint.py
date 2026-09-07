"""Hint masking preserves physical observations and compact mapping preserves computation."""

import unittest

import numpy as np

from fsrl.experiments.admission_hint_removal.inputs import observed, remove_hint
from fsrl.experiments.admission_hint_removal.protocol import parents, recipe
from fsrl.experiments.admission_hint_removal.qualification import numerical_checks
from fsrl.experiments.admission_hint_removal.reporting import decision
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.observation_uncertainty.inputs import attach_noise
from fsrl.experiments.training_strategy.batches import sample_episodes


class HintTests(unittest.TestCase):
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
            self.assertFalse(np.any(candidate.arrays["support_inputs"][..., 34]))
            expected = full.arrays["support_inputs"].copy()
            expected[..., 34] = 0
            for name, actual in candidate.arrays.items():
                np.testing.assert_array_equal(
                    actual, expected if name == "support_inputs" else full.arrays[name]
                )
            self.assertEqual(
                remove_hint(candidate).fingerprint(), candidate.fingerprint()
            )
        self.assertEqual(cpu.fingerprint(), original_hash)

    def test_zero_hint_forward_gradients_update_and_column_mapping(self):
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
