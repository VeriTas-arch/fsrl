import unittest

import numpy as np

from fsrl.dual_state_reduced_algorithm import EpisodeTrajectory
from fsrl.dual_state_reduced_algorithm_v2 import (
    ScalarHistoryParameters,
    fit_scalar_history,
    history_before,
    scalar_history_rollout,
    scalar_history_step,
)


class ScalarHistoryReducedAlgorithmTests(unittest.TestCase):
    def test_history_is_prior_cumulative_effective_evidence_energy(self):
        evidence = np.zeros((3, 8), dtype=np.float64)
        evidence[0, [0, 1]] = [0.5, -0.5]
        evidence[1, [2, 3]] = [1.0, -1.0]
        np.testing.assert_allclose(history_before(evidence), [0.0, 0.25, 1.25])

    def test_zero_evidence_changes_neither_state_nor_history(self):
        rng = np.random.default_rng(1)
        state = rng.normal(size=(4, 8))
        state -= np.mean(state, axis=1, keepdims=True)
        parameters = ScalarHistoryParameters(
            A=rng.normal(size=(8, 8)), B=rng.normal(size=(8, 8))
        )
        observed = scalar_history_step(
            state, np.zeros_like(state), np.arange(4.0), parameters
        )
        np.testing.assert_allclose(observed, state, atol=1e-12)

    def test_fit_recovers_scalar_history_transition(self):
        rng = np.random.default_rng(2)
        true = ScalarHistoryParameters(
            A=rng.normal(scale=0.1, size=(8, 8)),
            B=rng.normal(scale=0.05, size=(8, 8)),
        )
        records = []
        for _ in range(80):
            evidence = rng.normal(scale=0.3, size=(12, 8))
            evidence -= np.mean(evidence, axis=1, keepdims=True)
            potentials = scalar_history_rollout(evidence, true)
            records.append(
                EpisodeTrajectory(
                    potentials=potentials,
                    fields=np.empty((13, 28)),
                    evidence=evidence,
                    loo_potential=np.empty(8),
                    loo_field=np.empty(28),
                    loo_relation=(0, 1),
                    local_exact_error=0.0,
                    local_identity_raw=np.empty(0),
                    local_kernel_raw=np.empty(0),
                )
            )
        fitted = fit_scalar_history(records, ridge=1e-10)
        test_evidence = rng.normal(scale=0.3, size=(10, 8))
        test_evidence -= np.mean(test_evidence, axis=1, keepdims=True)
        np.testing.assert_allclose(
            scalar_history_rollout(test_evidence, fitted),
            scalar_history_rollout(test_evidence, true),
            atol=2e-9,
        )

    def test_rollout_uses_history_before_current_write(self):
        evidence = np.zeros((2, 8), dtype=np.float64)
        evidence[:, [0, 1]] = [1.0, -1.0]
        parameters = ScalarHistoryParameters(A=np.zeros((8, 8)), B=np.eye(8))
        values = scalar_history_rollout(evidence, parameters)
        np.testing.assert_allclose(values[1], 0.0)
        np.testing.assert_allclose(values[2], evidence[1])


if __name__ == "__main__":
    unittest.main()
