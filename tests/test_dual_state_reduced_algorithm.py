import unittest
from types import SimpleNamespace

import numpy as np

from fsrl.dual_state_reduced_algorithm import (
    ReducedParameters,
    accumulator_fit,
    complete_geometry,
    hodge_potential,
    local_edge_compression,
    reduced_step,
    rollout,
    unconstrained_bilinear_fit,
    unconstrained_bilinear_predict,
)


class DualStateReducedAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.geometry = complete_geometry()

    def test_hodge_round_trip_uses_centered_item_state(self):
        rng = np.random.default_rng(1)
        potential = rng.normal(size=(5, 8))
        potential -= np.mean(potential, axis=1, keepdims=True)
        field = potential @ self.geometry.incidence.T
        np.testing.assert_allclose(
            hodge_potential(field, self.geometry), potential, atol=1e-12
        )

    def test_edge_kernel_exactly_reassembles_tensor_trace(self):
        rng = np.random.default_rng(2)
        codes = rng.normal(size=(8, 5))
        trials = []
        values = []
        for index, (first, second) in enumerate(self.geometry.pairs[:9]):
            if index % 2:
                first, second = second, first
            trials.append(SimpleNamespace(left_item=first, right_item=second))
            values.append((-1.0) ** index * (index + 1) / 10.0)
        result = local_edge_compression(
            codes, tuple(trials), np.asarray(values), self.geometry
        )
        self.assertLessEqual(result["max_abs_error"], 1e-12)
        np.testing.assert_allclose(result["direct"], result["compressed"], atol=1e-12)

    def test_accumulator_recovers_linear_centered_update(self):
        rng = np.random.default_rng(3)
        evidence = rng.normal(size=(500, 8))
        evidence -= np.mean(evidence, axis=1, keepdims=True)
        matrix = rng.normal(size=(8, 8))
        delta = evidence @ matrix.T
        delta -= np.mean(delta, axis=1, keepdims=True)
        fitted = accumulator_fit(evidence, delta, ridge=1e-10)
        prediction = reduced_step(
            np.zeros_like(evidence), evidence, fitted
        )
        np.testing.assert_allclose(prediction, delta, atol=1e-8)

    def test_rank_two_step_has_exact_zero_evidence_identity(self):
        rng = np.random.default_rng(4)
        states = rng.normal(size=(7, 8))
        states -= np.mean(states, axis=1, keepdims=True)
        parameters = ReducedParameters(
            A=rng.normal(size=(8, 8)),
            U=rng.normal(size=(8, 2)),
            V=rng.normal(size=(8, 2)),
            W=rng.normal(size=(8, 2)),
        )
        np.testing.assert_allclose(
            reduced_step(states, np.zeros_like(states), parameters), states, atol=1e-12
        )

    def test_unconstrained_bilinear_fit_recovers_synthetic_transition(self):
        rng = np.random.default_rng(5)
        states = rng.normal(size=(2000, 8))
        states -= np.mean(states, axis=1, keepdims=True)
        evidence = rng.normal(size=(2000, 8))
        evidence -= np.mean(evidence, axis=1, keepdims=True)
        coefficients = rng.normal(scale=0.1, size=(72, 8))
        target_delta = unconstrained_bilinear_predict(
            coefficients, states, evidence
        )
        fitted = unconstrained_bilinear_fit(
            states, evidence, target_delta, ridge=1e-10
        )
        predicted = unconstrained_bilinear_predict(fitted, states, evidence)
        np.testing.assert_allclose(predicted, target_delta, atol=2e-7)

    def test_rollout_starts_from_exchangeable_zero_state(self):
        evidence = np.zeros((3, 8), dtype=np.float64)
        evidence[:, 0] = 1.0
        evidence[:, 1] = -1.0
        parameters = ReducedParameters(A=np.eye(8))
        values = rollout(evidence, parameters)
        self.assertEqual(values.shape, (4, 8))
        np.testing.assert_allclose(values[0], 0.0)
        np.testing.assert_allclose(values[-1], 3.0 * evidence[0])


if __name__ == "__main__":
    unittest.main()
