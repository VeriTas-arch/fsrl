import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import inverse_softplus
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.local_fidelity.curvature_gate import (
    CurvatureGateTransition,
    curvature_risk,
)


class CurvatureGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = TrainConfig(bs=3, hs=8, cs=4)
        self.backbone = RetroModulRNN(self.config.to_model_dict())
        self.gate = CurvatureGateTransition(
            self.backbone, epsilon=1e-8, initial_beta=1.0
        )

    def test_inverse_softplus_recovers_positive_value(self):
        raw = torch.tensor(inverse_softplus(1.25))
        self.assertAlmostEqual(float(torch.nn.functional.softplus(raw)), 1.25)

    def test_risk_matches_registered_equation(self):
        baseline = torch.tensor([[0.2, -0.4], [0.8, 0.1]])
        drive = torch.tensor([[0.5, -0.3], [-0.2, 0.7]])
        hidden = torch.tanh(baseline)
        derivative = 1.0 - hidden.square()
        expected = torch.linalg.vector_norm(
            -hidden * derivative * drive.square(), dim=1, keepdim=True
        ) / (torch.linalg.vector_norm(derivative * drive, dim=1, keepdim=True) + 1e-8)
        self.assertTrue(torch.allclose(curvature_risk(baseline, drive, 1e-8), expected))

    def test_gamma_one_exactly_reproduces_backbone_transition(self):
        device = self.backbone.w.device
        inputs = torch.randn(self.config.bs, self.config.inputsize, device=device)
        hidden = torch.randn(self.config.bs, self.config.hs, device=device)
        eligibility = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        expected = self.backbone(inputs, hidden, eligibility, fast_weights)
        observed = self.gate(
            inputs,
            hidden,
            eligibility,
            fast_weights,
            torch.ones(self.config.bs, 1, device=device),
        )
        for expected_value, observed_value in zip(expected, observed[:6]):
            self.assertTrue(torch.equal(expected_value, observed_value))

    def test_only_beta_is_trainable_and_conditioned_gamma_is_bounded(self):
        trainable = [
            name for name, value in self.gate.named_parameters() if value.requires_grad
        ]
        self.assertEqual(trainable, ["raw_beta"])
        device = self.backbone.w.device
        inputs = torch.randn(self.config.bs, self.config.inputsize, device=device)
        hidden = torch.randn(self.config.bs, self.config.hs, device=device)
        eligibility = torch.zeros(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        output = self.gate(inputs, hidden, eligibility, fast_weights)
        risk, gamma = output[6:8]
        self.assertTrue(torch.all(risk >= 0.0))
        self.assertTrue(torch.all(gamma > 0.0))
        self.assertTrue(torch.all(gamma <= 1.0))
        output[0].sum().backward()
        self.assertIsNotNone(self.gate.raw_beta.grad)
        self.assertTrue(np.isfinite(float(self.gate.raw_beta.grad)))


if __name__ == "__main__":
    unittest.main()
