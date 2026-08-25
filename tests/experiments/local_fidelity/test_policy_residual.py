import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.local_fidelity.policy_residual import (
    PolicyResidualTransition,
    inverse_sigmoid,
    policy_residual_statistics,
)


class PolicyResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.config = TrainConfig(bs=3, hs=8, cs=4)
        self.backbone = RetroModulRNN(self.config.to_model_dict())
        self.residual = PolicyResidualTransition(self.backbone, initial_eta=0.5)

    def test_inverse_sigmoid_recovers_bounded_value(self):
        raw = torch.tensor(inverse_sigmoid(0.3))
        self.assertAlmostEqual(float(torch.sigmoid(raw)), 0.3)

    def test_statistics_match_registered_equation(self):
        baseline = torch.tensor([[0.2, -0.4], [0.8, 0.1]])
        drive = torch.tensor([[0.5, -0.3], [-0.2, 0.7]])
        margin = torch.tensor([0.6, -0.9])
        baseline_hidden = torch.tanh(baseline)
        exact = torch.tanh(baseline + drive) - baseline_hidden
        linear = (1.0 - baseline_hidden.square()) * drive
        hidden_residual = linear - exact
        expected = (
            torch.sum(margin * hidden_residual, dim=1, keepdim=True),
            torch.sum(margin * exact, dim=1, keepdim=True),
            torch.sum(margin * linear, dim=1, keepdim=True),
            torch.linalg.vector_norm(hidden_residual, dim=1, keepdim=True),
        )
        observed = policy_residual_statistics(baseline, drive, margin)
        for actual, target in zip(observed, expected):
            self.assertTrue(torch.allclose(actual, target))

    def test_eta_zero_exactly_reproduces_every_backbone_output(self):
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
        observed = self.residual(
            inputs,
            hidden,
            eligibility,
            fast_weights,
            torch.zeros(self.config.bs, 1, device=device),
        )
        for expected_value, observed_value in zip(expected, observed[:6]):
            self.assertTrue(torch.equal(expected_value, observed_value))

    def test_eta_one_restores_first_order_policy_increment(self):
        device = self.backbone.w.device
        inputs = torch.randn(self.config.bs, self.config.inputsize, device=device)
        hidden = torch.randn(self.config.bs, self.config.hs, device=device)
        eligibility = torch.zeros(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        original = self.backbone(inputs, hidden, eligibility, fast_weights)
        statistics = self.residual.statistics(inputs, hidden, fast_weights)
        observed = self.residual(
            inputs,
            hidden,
            eligibility,
            fast_weights,
            torch.ones(self.config.bs, 1, device=device),
        )
        original_margin = original[0][:, 1] - original[0][:, 0]
        observed_margin = observed[0][:, 1] - observed[0][:, 0]
        self.assertTrue(
            torch.allclose(
                observed_margin - original_margin,
                statistics[0][:, 0],
                rtol=0.0,
                atol=1e-6,
            )
        )
        for expected_value, observed_value in zip(original[1:], observed[1:6]):
            self.assertTrue(torch.equal(expected_value, observed_value))

    def test_only_eta_is_trainable_and_gradient_is_finite(self):
        trainable = [
            name
            for name, value in self.residual.named_parameters()
            if value.requires_grad
        ]
        self.assertEqual(trainable, ["raw_eta"])
        device = self.backbone.w.device
        inputs = torch.randn(self.config.bs, self.config.inputsize, device=device)
        hidden = torch.randn(self.config.bs, self.config.hs, device=device)
        eligibility = torch.zeros(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=device
        )
        output = self.residual(inputs, hidden, eligibility, fast_weights)
        self.assertTrue(torch.all(output[7] > 0.0))
        self.assertTrue(torch.all(output[7] < 1.0))
        output[0].sum().backward()
        self.assertIsNotNone(self.residual.raw_eta.grad)
        self.assertTrue(np.isfinite(float(self.residual.raw_eta.grad)))
