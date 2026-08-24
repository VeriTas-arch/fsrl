import unittest

import numpy as np
import torch

from fsrl.config import TrainConfig
from fsrl.model import RetroModulRNN
from fsrl.policy_opposition_gate import (
    PolicyOppositionGateTransition,
    policy_opposition_statistics,
)


class PolicyOppositionGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.config = TrainConfig(bs=3, hs=8, cs=4)
        self.backbone = RetroModulRNN(self.config.to_model_dict())
        self.gate = PolicyOppositionGateTransition(
            self.backbone, tau=0.01, epsilon=1e-8, initial_beta=1.0
        )

    def test_statistics_match_registered_relative_scale_equation(self):
        baseline = torch.tensor([[0.2, -0.4], [0.8, 0.1]])
        drive = torch.tensor([[0.5, -0.3], [-0.2, 0.7]])
        margin = torch.tensor([0.6, -0.9])
        hidden = torch.tanh(baseline)
        derivative = 1.0 - hidden.square()
        jacobian = derivative * drive
        quadratic = -hidden * derivative * drive.square()
        j = torch.sum(margin * jacobian, dim=1, keepdim=True)
        k = torch.sum(margin * quadratic, dim=1, keepdim=True)
        scale = torch.sum(margin.square()) * torch.sum(
            jacobian.square(), dim=1, keepdim=True
        )
        denominator = j.square() + 0.01 * scale + 1e-8
        observed = policy_opposition_statistics(
            baseline, drive, margin, tau=0.01, epsilon=1e-8
        )
        expected = (
            torch.relu(-j * k) / denominator,
            torch.relu(j * k) / denominator,
            j,
            k,
            scale,
            denominator,
        )
        for actual, target in zip(observed, expected):
            self.assertTrue(torch.allclose(actual, target))

    def test_opposition_and_sign_reversed_risks_select_opposite_signs(self):
        baseline = torch.tensor([[0.5], [0.5]])
        drive = torch.tensor([[1.0], [-1.0]])
        margin = torch.ones(1)
        opposition, support, j, k, _, _ = policy_opposition_statistics(
            baseline, drive, margin, tau=0.01, epsilon=1e-8
        )
        product = j * k
        self.assertLess(float(product[0]), 0.0)
        self.assertGreater(float(opposition[0]), 0.0)
        self.assertEqual(float(support[0]), 0.0)
        self.assertGreater(float(product[1]), 0.0)
        self.assertEqual(float(opposition[1]), 0.0)
        self.assertGreater(float(support[1]), 0.0)

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

    def test_only_beta_is_trainable_and_both_gammas_are_bounded(self):
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
        opposition = self.gate(inputs, hidden, eligibility, fast_weights)
        support = self.gate(
            inputs, hidden, eligibility, fast_weights, use_support_risk=True
        )
        for output in (opposition, support):
            risk, gamma = output[6:8]
            self.assertTrue(torch.all(risk >= 0.0))
            self.assertTrue(torch.all(gamma > 0.0))
            self.assertTrue(torch.all(gamma <= 1.0))
        (opposition[0].sum() + support[0].sum()).backward()
        self.assertIsNotNone(self.gate.raw_beta.grad)
        self.assertTrue(np.isfinite(float(self.gate.raw_beta.grad)))


if __name__ == "__main__":
    unittest.main()
