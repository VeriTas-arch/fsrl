"""Small non-Liu scientific and numerical invariants on CPU."""

import unittest

import numpy as np
import torch

from fsrl.experiments.score_circuit.circuit import (
    coefficients,
    derivative,
    initial_state,
    integrate_support,
    integration_chunk,
    query_read,
)
from fsrl.experiments.score_circuit.qualification import fixture
from fsrl.experiments.score_circuit.reference import affine_support, differences


class CircuitTests(unittest.TestCase):
    def test_pooling_and_positive_input(self):
        x = differences(fixture()["support_cues"])
        np.testing.assert_array_equal(2 * np.abs(x).sum(-1), (x * x).sum(-1))
        self.assertGreaterEqual(float((2 + x).min()), 0)

    def test_initial_signal_is_not_external_target_error(self):
        y = initial_state(1, 15, "cpu")
        args = (
            y.new_full((1, 15), 4),
            y.new_ones(1),
            y.new_ones(1),
            y.new_ones(1),
            y.new_tensor(1),
            coefficients(0.989, 1, 4096, "cpu"),
        )
        dy = derivative(y, *args)
        torch.testing.assert_close(dy[:, :30], torch.zeros_like(dy[:, :30]))
        self.assertGreater(float(dy[0, 32]), 0)
        self.assertLess(float(dy[0, 33]), 0)

    def test_error_gain_is_dynamic_shunting(self):
        y = initial_state(1, 15, "cpu")
        y[0, 32:34] = torch.tensor([1.0, -1.0])
        y[0, 34] = 1 / (60 + 1e-8)
        y[0, 35] = -1 / (60 + 1e-8)
        args = (
            y.new_full((1, 15), 4),
            y.new_ones(1),
            y.new_ones(1),
            y.new_ones(1),
            y.new_tensor(1),
            coefficients(0.989, 1, 4096, "cpu"),
        )
        dy = derivative(y, *args)
        torch.testing.assert_close(
            dy[:, 34:], torch.zeros_like(dy[:, 34:]), atol=1e-12, rtol=0
        )
        self.assertGreater(float(dy[0, 0]), 0)
        self.assertLess(float(dy[0, 15]), 0)

    def test_no_write_and_zero_error(self):
        inputs = fixture(trials=1, subjects=1)
        for control in ("teacher_off", "mismatch_clamp"):
            actual = integrate_support(
                inputs,
                0.989,
                1,
                256,
                integration_chunk,
                device="cpu",
                control=control,
                duration=0.02,
            )
            np.testing.assert_array_equal(actual["trajectory"][..., :30], 1)
        inputs["signed"][:] = 0
        actual = integrate_support(
            inputs, 0.989, 1, 256, integration_chunk, device="cpu", duration=0.02
        )
        np.testing.assert_allclose(actual["trajectory"][..., :30], 1, atol=1e-12)
        inputs["signed"][:] = 0.8
        inputs["retention"][:] = 0
        actual = integrate_support(
            inputs, 0.989, 1, 256, integration_chunk, device="cpu", duration=0.02
        )
        np.testing.assert_array_equal(actual["trajectory"][..., :30], 1)

    def test_affine_reversal_pairing_and_readonly(self):
        inputs = fixture(trials=1, subjects=2)
        output = integrate_support(
            inputs, 0.989, 1, 256, integration_chunk, device="cpu", duration=0.02
        )
        expected = affine_support(inputs, 0.989, 1, duration=0.02)
        np.testing.assert_allclose(output["trajectory"], expected, atol=1e-6, rtol=0)
        reverse = {
            **inputs,
            "support_cues": np.concatenate(
                (inputs["support_cues"][..., 15:], inputs["support_cues"][..., :15]),
                axis=-1,
            ),
            "signed": -inputs["signed"],
        }
        other = integrate_support(
            reverse, 0.989, 1, 256, integration_chunk, device="cpu", duration=0.02
        )
        np.testing.assert_allclose(
            output["trajectory"][..., :30], other["trajectory"][..., :30], atol=1e-12
        )
        state = output["trajectory"][:, -1]
        np.testing.assert_allclose(state[:, :15] + state[:, 15:30], 2, atol=1e-12)
        before = state.copy()
        margin = query_read(state, inputs["query_cues"], 7.2, 0.002)
        opposite = query_read(state, inputs["query_cues"], 7.2, 0.002, reverse=True)
        np.testing.assert_allclose(margin, opposite, atol=1e-9)
        np.testing.assert_array_equal(state, before)


if __name__ == "__main__":
    unittest.main()
