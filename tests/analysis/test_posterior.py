import math
import unittest

import numpy as np

from fsrl.analysis.posterior import (
    ExactRankingPosterior,
    RelationEvidence,
    evidence_from_protocol,
)
from fsrl.tasks.registered_protocol import load_ranking_protocol


class ExactRankingPosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_ranking_protocol()
        cls.model = ExactRankingPosterior(8, temperature=0.02)
        cls.evidence = evidence_from_protocol(cls.protocol)

    def test_enumerates_all_40320_orders(self):
        self.assertEqual(self.model.n_hypotheses, math.factorial(8))
        self.assertEqual(self.model.orders.shape, (40320, 8))

    def test_registered_evidence_recovers_true_map_order(self):
        state = self.model.fit(self.evidence)
        self.assertAlmostEqual(float(np.sum(state.probabilities)), 1.0)
        self.assertEqual(
            self.model.map_order(state), self.protocol.true_order_high_to_low
        )

    def test_evidence_order_does_not_change_posterior(self):
        forward = self.model.fit(self.evidence)
        reverse = self.model.fit(tuple(reversed(self.evidence)))
        np.testing.assert_allclose(
            forward.probabilities, reverse.probabilities, atol=1e-14
        )

    def test_zero_reliability_removes_one_relation(self):
        full = self.model.fit(self.evidence)
        weakened = list(self.evidence)
        target = weakened[0]
        weakened[0] = RelationEvidence(
            target.higher_item,
            target.lower_item,
            target.magnitude,
            reliability=0.0,
        )
        reduced = self.model.fit(weakened)
        self.assertGreater(self.model.posterior_entropy(reduced), 0.0)
        self.assertFalse(np.array_equal(full.probabilities, reduced.probabilities))

    def test_committed_readout_is_stable_and_transitive(self):
        state = self.model.fit(self.evidence)
        order = self.model.map_order(state)
        for high, middle, low in ((0, 1, 2), (2, 4, 7)):
            self.assertGreater(
                self.model.committed_pair_probability(order, high, middle), 0.5
            )
            self.assertGreater(
                self.model.committed_pair_probability(order, middle, low), 0.5
            )
            self.assertGreater(
                self.model.committed_pair_probability(order, high, low), 0.5
            )

    def test_pair_posterior_is_antisymmetric(self):
        state = self.model.fit(self.evidence)
        probability = self.model.pair_probability(state, 1, 6)
        reverse = self.model.pair_probability(state, 6, 1)
        self.assertAlmostEqual(probability + reverse, 1.0)


if __name__ == "__main__":
    unittest.main()
