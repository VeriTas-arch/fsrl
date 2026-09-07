"""Affine equations, common initialization, clipping and write-weighted audit."""

import unittest

import numpy as np
import torch

from fsrl.experiments.linear_modulation.audit import stats
from fsrl.experiments.linear_modulation.model import make_model
from fsrl.experiments.linear_modulation.protocol import recipe
from fsrl.experiments.linear_modulation.qualification import (
    initialization_checks,
    numerical_checks,
)


class ModulationTests(unittest.TestCase):
    def test_affine_forward_gradient_adam_and_input_merge(self):
        self.assertTrue(all(v["passed"] for v in numerical_checks().values()))

    def test_all_seed_common_initialization_and_rng(self):
        spec = recipe()
        for seed in (2531, 2532, 2533):
            self.assertTrue(initialization_checks(spec, seed, "cpu")["passed"])

    def test_old_eligibility_and_original_clipping(self):
        spec = recipe()
        spec["architecture"]["hidden_size"] = 8
        net, _ = make_model(spec, 925003, device="cpu")
        with torch.no_grad():
            net.h2DA.weight.zero_()
            net.h2DA.bias.fill_(2)
            h, e, p = (
                net.initial_hidden(2),
                torch.ones(2, 8, 8),
                torch.full((2, 8, 8), 49.0),
            )
            out = net(torch.zeros(2, 38), h, e, p)
            torch.testing.assert_close(out[5], torch.full_like(p, 50.0))
            torch.testing.assert_close(out[4], (1 - net.etaet) * e)
            no_old = net(torch.zeros(2, 38), h, torch.zeros_like(e), p)
            torch.testing.assert_close(no_old[5], p)
            self.assertEqual(net.h2DA.out_features, 1)
            self.assertFalse(hasattr(net, "DAmult"))

    def test_zero_without_write_opportunity_is_not_counted_as_suppression(self):
        a, b = np.array([0.0, 3.0, 0.1]), np.array([0.0, 3.0, -0.1])
        m = np.tanh(a) - np.tanh(b)
        result = stats(a, b, m, np.array([0.0, 2.0, 1.0]), np.abs(m), np.zeros(3), 1.0)
        self.assertEqual(result["eligibility_active"], 2)
        self.assertAlmostEqual(result["fraction"]["near_zero_m"], 2 / 3)
        self.assertAlmostEqual(
            result["eligibility_active_fraction"]["near_zero_m"], 0.5
        )
        self.assertAlmostEqual(
            result["potential_write_weighted_fraction"]["near_zero_m"], 0.8
        )
