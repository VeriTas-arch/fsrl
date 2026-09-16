import unittest

import torch

from fsrl.core.factorized_plastic_rnn import (
    FactorizedPlasticRNN,
    FactorizedPlasticRNNConfig,
    FactorizedRecurrentSequence,
    factorize_legacy_inputs,
)
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.core.sequence import RecurrentSequence


class FactorizedPlasticRNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3101)
        self.batch_size = 3
        self.hidden_size = 7
        self.cue_size = 15
        self.legacy = RetroModulRNN(
            RetroModelConfig(
                input_size=37,
                hidden_size=self.hidden_size,
                output_size=2,
                batch_size=self.batch_size,
            ),
            device="cpu",
        )
        self.factorized = FactorizedPlasticRNN.from_legacy(self.legacy)

    def active_inputs(self, steps: int = 1) -> torch.Tensor:
        inputs = torch.randn(steps, self.batch_size, 37)
        inputs[..., 31] = 1.0
        inputs[..., 33] = 0.0
        inputs[..., 35:] = 0.0
        return inputs

    def test_configuration_has_32_external_channels(self):
        config = FactorizedPlasticRNNConfig()
        self.assertEqual(config.input_size, 32)
        self.assertEqual(config.response_index, 30)
        self.assertEqual(config.evidence_index, 31)
        fresh = FactorizedPlasticRNN(config, device="cpu")
        self.assertFalse(fresh.uses_legacy_numerics)
        self.assertNotIn("legacy_input_bias", fresh.state_dict())

    def test_checkpoint_conversion_enables_nontrainable_numerical_adapter(self):
        self.assertTrue(self.factorized.uses_legacy_numerics)
        parameters = dict(self.factorized.named_parameters())
        buffers = dict(self.factorized.named_buffers())
        for name in (
            "legacy_constant_weight",
            "legacy_input_bias",
            "legacy_output_weight",
            "legacy_output_bias",
        ):
            self.assertNotIn(name, parameters)
            self.assertIn(name, buffers)

        restored = FactorizedPlasticRNN(
            self.factorized.model_config,
            device="cpu",
            legacy_numerical_compatibility=True,
        )
        restored.load_state_dict(self.factorized.state_dict())
        self.assertTrue(restored.uses_legacy_numerics)

    def test_active_input_factorization_is_lossless(self):
        legacy = self.active_inputs(4)
        external, times = factorize_legacy_inputs(legacy, self.cue_size)
        self.assertEqual(external.shape, (4, self.batch_size, 32))
        self.assertTrue(torch.equal(external[..., :31], legacy[..., :31]))
        self.assertTrue(torch.equal(external[..., 31], legacy[..., 34]))
        self.assertTrue(torch.equal(times[..., 0], legacy[..., 32]))

    def test_factorization_rejects_nonpassive_legacy_channels(self):
        for index in (31, 33, 35, 36):
            with self.subTest(index=index):
                legacy = self.active_inputs()
                legacy[..., index] = 0.5
                with self.assertRaises(ValueError):
                    factorize_legacy_inputs(legacy, self.cue_size)

    def test_single_step_matches_legacy_state_and_outputs(self):
        legacy_inputs = self.active_inputs()[0]
        external, times = factorize_legacy_inputs(legacy_inputs, self.cue_size)
        hidden = torch.randn(self.batch_size, self.hidden_size)
        eligibility = torch.randn(self.batch_size, self.hidden_size, self.hidden_size)
        fast_weights = torch.randn(self.batch_size, self.hidden_size, self.hidden_size)
        legacy = self.legacy(legacy_inputs, hidden, eligibility, fast_weights)
        factorized = self.factorized(external, times, hidden, eligibility, fast_weights)
        expected = (legacy[0][:, 1:2] - legacy[0][:, 0:1], *legacy[2:])
        for observed, target in zip(factorized, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=1e-6, rtol=0.0)

    def test_four_step_sequence_matches_legacy(self):
        legacy_inputs = self.active_inputs(4)
        external, times = factorize_legacy_inputs(legacy_inputs, self.cue_size)
        hidden = self.legacy.initial_hidden(self.batch_size)
        eligibility = self.legacy.initial_eligibility(self.batch_size)
        fast_weights = self.legacy.initial_fast_weights(self.batch_size)
        legacy = RecurrentSequence(self.legacy)(
            legacy_inputs, hidden, eligibility, fast_weights, True
        )
        factorized = FactorizedRecurrentSequence(self.factorized)(
            external, times, hidden, eligibility, fast_weights, True
        )
        expected = (legacy[0][:, 1:2] - legacy[0][:, 0:1], legacy[2], *legacy[3:])
        for observed, target in zip(factorized, expected, strict=True):
            torch.testing.assert_close(observed, target, atol=1e-6, rtol=0.0)

    def test_two_blank_steps_leave_fast_weights_unchanged(self):
        blank = torch.zeros(2, self.batch_size, 37)
        fast_weights = torch.randn(self.batch_size, self.hidden_size, self.hidden_size)
        result = RecurrentSequence(self.legacy)(
            blank,
            self.legacy.initial_hidden(self.batch_size),
            self.legacy.initial_eligibility(self.batch_size),
            fast_weights,
            True,
        )
        self.assertTrue(torch.equal(result[-1], fast_weights))


if __name__ == "__main__":
    unittest.main()
