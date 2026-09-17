import unittest

import torch

from fsrl.experiments.minimal_single_p.model import (
    MinimalSinglePSequence,
    level_settings,
    make_model,
)
from fsrl.experiments.minimal_single_p.optimization import (
    forward_batch,
    make_optimizer,
    training_step,
)
from fsrl.experiments.minimal_single_p.protocol import LEVELS
from fsrl.experiments.minimal_single_p.qualification import _fixture
from fsrl.infra.provenance import tensor_hashes


class MinimalSinglePModelTests(unittest.TestCase):
    def test_registered_structure_and_direct_initialization(self):
        models = {level: make_model(level, 41) for level in LEVELS}
        self.assertEqual(tensor_hashes(models["C0"]), tensor_hashes(models["M1"]))
        for level, model in models.items():
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                level_settings(level)["parameter_count"],
            )
            self.assertEqual(model.model_config.input_size, 32)
            self.assertFalse(list(model.named_buffers()))
            self.assertEqual(model.alpha is not None, level in {"C0", "M1"})
            self.assertEqual(model.etaet is not None, level in {"C0", "M1", "M2"})
        for level in LEVELS[2:]:
            self.assertEqual(
                int(torch.count_nonzero(models[level].h2modulation.weight)), 0
            )
            self.assertEqual(
                int(torch.count_nonzero(models[level].h2modulation.bias)), 0
            )

    def test_p_before_e_and_fixed_eta(self):
        model = make_model("M3", 42, hidden_size=5)
        inputs = torch.randn(2, 32)
        hidden = torch.randn(2, 5)
        eligibility = torch.randn(2, 5, 5)
        weights = torch.randn(2, 5, 5)
        _, modulation, next_hidden, next_e, next_p = model.step(
            inputs, hidden, eligibility, weights
        )
        expected_p = torch.clamp(
            weights + modulation[:, None] * eligibility, -50.0, 50.0
        )
        expected_e = torch.tanh(next_hidden[:, :, None] * hidden[:, None, :])
        torch.testing.assert_close(next_p, expected_p)
        torch.testing.assert_close(next_e, expected_e)

    def test_zero_modulation_still_receives_query_gradient(self):
        cpu = _fixture(43, batch_size=2)
        batch, _ = cpu.to("cpu")
        model = make_model("M2", 43, hidden_size=7)
        optimizer = make_optimizer(model, 1e-4)
        result, norm = training_step(
            model,
            MinimalSinglePSequence(model),
            batch,
            optimizer,
            penalty=0.0,
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(norm), 0.0)
        self.assertGreater(float(model.h2modulation.weight.detach().abs().max()), 0.0)

    def test_c0_penalty_is_the_only_c0_m1_loss_difference(self):
        cpu = _fixture(44, batch_size=2)
        batch, _ = cpu.to("cpu")
        c0 = make_model("C0", 44, hidden_size=7)
        m1 = make_model("M1", 44, hidden_size=7)
        first = forward_batch(c0, MinimalSinglePSequence(c0), batch, penalty=1e-4)
        second = forward_batch(m1, MinimalSinglePSequence(m1), batch, penalty=0.0)
        torch.testing.assert_close(first.margins, second.margins, atol=0, rtol=0)
        torch.testing.assert_close(
            first.loss - second.loss,
            1e-4 * first.fast_weights.square().mean(),
        )


if __name__ == "__main__":
    unittest.main()
