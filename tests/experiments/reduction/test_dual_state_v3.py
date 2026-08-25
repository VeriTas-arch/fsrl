import unittest

import numpy as np

from fsrl.experiments.reduction import dual_state_v2 as v2
from fsrl.experiments.reduction.dual_state_v1 import EpisodeTrajectory
from fsrl.experiments.reduction.dual_state_v3 import (
    ItemHistoryParameters,
    bind_item_rollout,
    fit_item_history,
    item_history_before,
    item_history_rollout,
    item_history_step,
)


class ItemHistoryReducedAlgorithmTests(unittest.TestCase):
    def test_history_retains_itemwise_evidence_energy(self):
        evidence = np.zeros((3, 8), dtype=np.float64)
        evidence[0, [0, 1]] = [0.5, -0.5]
        evidence[1, [1, 2]] = [1.0, -1.0]
        expected = np.zeros((3, 8))
        expected[1, [0, 1]] = 0.25
        expected[2, [0, 1, 2]] = [0.25, 1.25, 1.0]
        np.testing.assert_allclose(item_history_before(evidence), expected)

    def test_zero_evidence_is_exact_identity(self):
        rng = np.random.default_rng(1)
        state = rng.normal(size=(4, 8))
        state -= np.mean(state, axis=1, keepdims=True)
        parameters = ItemHistoryParameters(
            A=rng.normal(size=(8, 8)), B=rng.normal(size=(8, 8))
        )
        np.testing.assert_allclose(
            item_history_step(
                state, np.zeros_like(state), rng.random((4, 8)), parameters
            ),
            state,
            atol=1e-12,
        )

    def test_fit_recovers_item_history_transition(self):
        rng = np.random.default_rng(2)
        true = ItemHistoryParameters(
            A=rng.normal(scale=0.1, size=(8, 8)),
            B=rng.normal(scale=0.05, size=(8, 8)),
        )
        records = []
        for _ in range(80):
            evidence = rng.normal(scale=0.3, size=(12, 8))
            evidence -= np.mean(evidence, axis=1, keepdims=True)
            records.append(
                EpisodeTrajectory(
                    potentials=item_history_rollout(evidence, true),
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
        fitted = fit_item_history(records, ridge=1e-10)
        test = rng.normal(scale=0.3, size=(10, 8))
        test -= np.mean(test, axis=1, keepdims=True)
        np.testing.assert_allclose(
            item_history_rollout(test, fitted),
            item_history_rollout(test, true),
            atol=2e-9,
        )

    def test_preservation_rollout_binding_is_scoped(self):
        original = v2.scalar_history_rollout_batch
        parameters = ItemHistoryParameters(A=np.eye(8), B=np.eye(8))
        with bind_item_rollout(parameters):
            self.assertIsNot(v2.scalar_history_rollout_batch, original)
        self.assertIs(v2.scalar_history_rollout_batch, original)


if __name__ == "__main__":
    unittest.main()
