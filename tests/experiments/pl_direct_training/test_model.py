import unittest

import torch

from fsrl.core.factorized_plastic_rnn import FactorizedPlasticRNN
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.pl_direct_training.model import (
    NoTimePlasticRNN,
    map_shadow_model,
    shared_parameter_tensors,
)


class DirectModelTests(unittest.TestCase):
    def shadow(self, hidden_size: int = 9) -> RetroModulRNN:
        torch.manual_seed(3001)
        return RetroModulRNN(
            RetroModelConfig(37, hidden_size, 2, 3),
            device="cpu",
        )

    def test_paired_mapping_has_equal_shared_parameters_and_no_buffers(self):
        shadow = self.shadow()
        control = map_shadow_model(shadow, "time_retained_control")
        no_time = map_shadow_model(shadow, "no_time_candidate")
        self.assertIsInstance(control, FactorizedPlasticRNN)
        self.assertIsInstance(no_time, NoTimePlasticRNN)
        self.assertIn("time_weight", dict(control.named_parameters()))
        self.assertNotIn("time_weight", dict(no_time.named_parameters()))
        self.assertNotIn("time_weight", no_time.state_dict())
        self.assertFalse(control.uses_legacy_numerics)
        self.assertFalse(no_time.uses_legacy_numerics)
        for name, value in shared_parameter_tensors(control).items():
            self.assertTrue(torch.equal(value, shared_parameter_tensors(no_time)[name]))
        self.assertEqual(list(control.named_buffers()), [])
        self.assertEqual(list(no_time.named_buffers()), [])

    def test_registered_parameter_counts(self):
        shadow = self.shadow(hidden_size=200)
        control = map_shadow_model(shadow, "time_retained_control")
        no_time = map_shadow_model(shadow, "no_time_candidate")
        self.assertEqual(sum(value.numel() for value in control.parameters()), 87405)
        self.assertEqual(sum(value.numel() for value in no_time.parameters()), 87205)

    def test_clean_time_control_matches_shadow_active_computation(self):
        shadow = self.shadow()
        control = map_shadow_model(shadow, "time_retained_control")
        task = torch.randn(3, 32)
        time = torch.rand(3, 1) * (2.0 / 3.0)
        legacy = torch.zeros(3, 37)
        legacy[:, :31] = task[:, :31]
        legacy[:, 31] = 1.0
        legacy[:, 32:33] = time
        legacy[:, 34] = task[:, 31]
        hidden = torch.randn(3, 9)
        eligibility = torch.randn(3, 9, 9)
        fast_weights = torch.randn(3, 9, 9)
        old = shadow(legacy, hidden, eligibility, fast_weights)
        new = control(task, time, hidden, eligibility, fast_weights)
        expected = (old[0][:, 1:2] - old[0][:, 0:1], *old[2:])
        for observed, target in zip(new, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=2e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
