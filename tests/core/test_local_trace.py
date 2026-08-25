import unittest

import numpy as np
import torch

from fsrl.core.local_trace import (
    ConjunctiveLocalTrace,
    antisymmetric_conjunctive_key,
    inverse_softplus,
)


class ConjunctiveLocalTraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)
        self.cue_size = 5
        self.trace = ConjunctiveLocalTrace(self.cue_size, initial_gain=0.3)
        self.left = torch.randn(3, self.cue_size, device=self.trace.raw_gain.device)
        self.right = torch.randn(3, self.cue_size, device=self.trace.raw_gain.device)
        self.forward = torch.cat((self.left, self.right), dim=1)
        self.reverse = torch.cat((self.right, self.left), dim=1)

    def test_inverse_softplus_recovers_positive_gain(self):
        raw = torch.tensor(inverse_softplus(0.3))
        self.assertAlmostEqual(float(torch.nn.functional.softplus(raw)), 0.3)

    def test_key_and_query_orientation_are_antisymmetric(self):
        forward = antisymmetric_conjunctive_key(self.forward, self.cue_size)
        reverse = antisymmetric_conjunctive_key(self.reverse, self.cue_size)
        self.assertTrue(torch.allclose(reverse, -forward, atol=1e-7, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(forward, dim=1),
                torch.ones(3, device=forward.device),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_support_presentation_reversal_leaves_write_unchanged(self):
        state = self.trace.initial_state(3)
        values = torch.tensor([0.2, -0.4, 0.7], device=state.device)
        forward = self.trace.write(state, self.forward, values)
        reverse = self.trace.write(state, self.reverse, -values)
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-7, rtol=0.0))

    def test_direct_read_is_positive_and_reverse_is_negative(self):
        state = self.trace.initial_state(3)
        values = torch.tensor([0.2, 0.4, 0.7], device=state.device)
        state = self.trace.write(state, self.forward, values)
        direct, _ = self.trace.read(state, self.forward)
        reverse, _ = self.trace.read(state, self.reverse)
        self.assertTrue(torch.allclose(direct[:, 0], values, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(reverse, -direct, atol=1e-6, rtol=0.0))

    def test_zero_gain_reproduces_logits_and_gain_is_only_parameter(self):
        state = self.trace.write(
            self.trace.initial_state(3),
            self.forward,
            torch.ones(3, device=self.trace.raw_gain.device),
        )
        logits = torch.randn(3, 2, device=state.device)
        zero = torch.zeros(3, 1, device=state.device)
        corrected, _raw, _gain, correction = self.trace(
            logits, state, self.forward, zero
        )
        self.assertTrue(torch.equal(corrected, logits))
        self.assertTrue(torch.equal(correction, zero))
        trainable = [
            name for name, value in self.trace.named_parameters() if value.requires_grad
        ]
        self.assertEqual(trainable, ["raw_gain"])

    def test_common_mode_and_gradient(self):
        state = self.trace.write(
            self.trace.initial_state(3),
            self.forward,
            torch.ones(3, device=self.trace.raw_gain.device),
        )
        logits = torch.randn(3, 2, device=state.device)
        corrected, _raw, gain, _correction = self.trace(logits, state, self.forward)
        self.assertTrue(torch.all(gain > 0.0))
        self.assertTrue(
            torch.allclose(
                torch.mean(corrected, dim=1),
                torch.mean(logits, dim=1),
                atol=1e-7,
                rtol=0.0,
            )
        )
        corrected.sum().backward()
        self.assertIsNotNone(self.trace.raw_gain.grad)
        self.assertTrue(np.isfinite(float(self.trace.raw_gain.grad)))
