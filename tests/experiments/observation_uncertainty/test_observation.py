"""Scientific observation, cue, comparator and optimization invariants."""

import unittest

import numpy as np

from fsrl.experiments.observation_uncertainty.comparator import solve
from fsrl.experiments.observation_uncertainty.protocol import specification
from fsrl.experiments.observation_uncertainty.qualification import input_checks, qualify
from fsrl.experiments.training_strategy.batches import EpisodeBatch


class ObservationTests(unittest.TestCase):
    def test_independent_encoding_and_information_matched_history(self):
        self.assertTrue(
            all(row["passed"] for row in input_checks(specification()).values())
        )

    def test_observer_recovers_a_known_chain_without_query_targets(self):
        cpu = EpisodeBatch(
            {
                "support_pairs": np.asarray([[[0, 1]], [[1, 2]]]),
                "local_evidence": np.asarray([[1.0], [2.0]]),
                "item_codes": np.eye(3)[None],
                "query_pairs": np.asarray([[0, 2], [2, 0]]),
            }
        )
        margins, ranks = solve(cpu, 1.0)
        np.testing.assert_allclose(margins, [[3.0, -3.0]], atol=1e-12)
        np.testing.assert_array_equal(ranks, [2])
        cpu.arrays["local_evidence"][1] = 0
        margins, ranks = solve(cpu, 1.0)
        np.testing.assert_allclose(margins, [[0.5, -0.5]], atol=1e-12)
        np.testing.assert_array_equal(ranks, [1])

    def test_reference_forward_and_joint_update(self):
        self.assertTrue(all(row["passed"] for row in qualify(specification()).values()))


if __name__ == "__main__":
    unittest.main()
