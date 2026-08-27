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

    def test_cached_fit_preserves_reference_accumulation_bitwise(self):
        state = self.model.fit(self.evidence)
        energy = np.zeros(self.model.n_hypotheses, dtype=np.float64)
        for observation in self.evidence:
            predicted = (
                self.model.positions[:, observation.lower_item]
                - self.model.positions[:, observation.higher_item]
            ) / float(self.model.n_items - 1)
            residual = predicted - observation.magnitude
            energy += observation.reliability * residual * residual
        log_weights = -energy / self.model.temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        probabilities = weights / np.sum(weights)
        np.testing.assert_array_equal(state.energy, energy)
        np.testing.assert_array_equal(state.probabilities, probabilities)

    def test_hypothesis_space_is_shared_read_only_and_indexed_exactly(self):
        second = ExactRankingPosterior(8, temperature=0.1)
        self.assertIs(self.model.orders, second.orders)
        self.assertIs(self.model.positions, second.positions)
        self.assertFalse(self.model.orders.flags.writeable)
        self.assertFalse(self.model.positions.flags.writeable)
        indices = np.random.default_rng(93).integers(
            0, self.model.n_hypotheses, size=100
        )
        for index in indices:
            order = [int(item) for item in self.model.orders[index]]
            self.assertEqual(self.model.order_index(order), index)

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

        reference = float(
            np.sum(
                state.probabilities[
                    self.model.positions[:, 1] < self.model.positions[:, 6]
                ]
            )
        )
        self.assertEqual(probability, reference)
