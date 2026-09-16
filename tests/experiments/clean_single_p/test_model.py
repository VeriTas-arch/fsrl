import unittest

import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.clean_single_p.model import (
    AffineSinglePSequence,
    expand_shadow_inputs,
    map_shadow,
    shared_parameters,
)
from fsrl.experiments.linear_modulation.model import LinearModulationRNN


class CleanSinglePModelTests(unittest.TestCase):
    def shadow(self, hidden_size=9):
        torch.manual_seed(931001)
        return LinearModulationRNN(
            RetroModelConfig(38, hidden_size, 2, 3), device="cpu"
        )

    def test_mapping_and_parameter_counts(self):
        shadow = self.shadow(200)
        control = map_shadow(shadow, "time_retained_control")
        candidate = map_shadow(shadow, "clean_no_time")
        self.assertEqual(sum(p.numel() for p in control.parameters()), 87203)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 87003)
        self.assertIn("time_weight", dict(control.named_parameters()))
        self.assertNotIn("time_weight", dict(candidate.named_parameters()))
        self.assertEqual(list(control.named_buffers()), [])
        self.assertEqual(list(candidate.named_buffers()), [])
        for name, value in shared_parameters(control).items():
            self.assertTrue(torch.equal(value, shared_parameters(candidate)[name]))

    def test_time_control_matches_duplicate_q_shadow(self):
        shadow = self.shadow()
        control = map_shadow(shadow, "time_retained_control")
        task = torch.randn(3, 32)
        time = torch.rand(3, 1) * (2 / 3)
        legacy = expand_shadow_inputs(task, time, 15)
        hidden = torch.randn(3, 9)
        eligibility = torch.randn(3, 9, 9)
        weights = torch.randn(3, 9, 9)
        old = shadow(legacy, hidden, eligibility, weights)
        new = control.step(task, hidden, eligibility, weights, time_values=time)
        expected = (old[0][:, 1:2] - old[0][:, 0:1], old[2], *old[3:])
        for observed, target in zip(new, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=2e-6, rtol=0)

    def test_no_time_has_no_time_api_or_effect(self):
        candidate = map_shadow(self.shadow(), "clean_no_time")
        sequence = AffineSinglePSequence(candidate)
        task = torch.randn(4, 3, 32)
        state = (
            candidate.initial_hidden(3),
            candidate.initial_eligibility(3),
            candidate.initial_fast_weights(3),
        )
        first = sequence(task, *state, True)
        second = sequence(task, *state, True)
        for observed, target in zip(first, second, strict=True):
            self.assertTrue(torch.equal(observed, target))
        with self.assertRaisesRegex(ValueError, "does not accept time"):
            candidate.step(task[0], *state, time_values=torch.zeros(3, 1))


if __name__ == "__main__":
    unittest.main()
