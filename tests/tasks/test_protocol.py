import json
import unittest

import numpy as np

from fsrl.tasks.registered_protocol import DEFAULT_PROTOCOL_PATH, load_ranking_protocol


class RankingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_ranking_protocol()

    def test_registered_protocol_counts_and_graph(self):
        self.assertEqual(self.protocol.n_items, 8)
        self.assertEqual(self.protocol.support_trials, 32)
        self.assertEqual(self.protocol.query_trials, 280)
        self.assertEqual(len(self.protocol.support_pairs_higher_lower), 8)
        self.assertEqual(len(self.protocol.learned_pairs), 8)

    def test_support_schedule_is_passive_evidence_with_signed_magnitude(self):
        schedule = self.protocol.support_schedule(np.random.default_rng(7))
        self.assertEqual(len(schedule), 32)
        counts = {pair: 0 for pair in self.protocol.support_pairs_higher_lower}
        rank = {
            item: position
            for position, item in enumerate(self.protocol.true_order_high_to_low)
        }
        for trial in schedule:
            counts[(trial.higher_item, trial.lower_item)] += 1
            expected = (rank[trial.lower_item] - rank[trial.higher_item]) / 7.0
            if trial.left_item == trial.higher_item:
                self.assertAlmostEqual(trial.signed_magnitude, expected)
            else:
                self.assertAlmostEqual(trial.signed_magnitude, -expected)
        self.assertEqual(set(counts.values()), {4})

    def test_query_schedule_covers_each_pair_ten_times_without_distance_input(self):
        schedule = self.protocol.query_schedule(np.random.default_rng(11))
        pair_counts = {}
        rank = {
            item: position
            for position, item in enumerate(self.protocol.true_order_high_to_low)
        }
        for trial in schedule:
            pair = tuple(sorted((trial.left_item, trial.right_item)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            self.assertEqual(
                trial.correct_action,
                1 if rank[trial.left_item] < rank[trial.right_item] else 0,
            )
        self.assertEqual(len(pair_counts), 28)
        self.assertEqual(set(pair_counts.values()), {10})

    def test_json_declares_strict_frozen_readout_contract(self):
        with DEFAULT_PROTOCOL_PATH.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        query = raw["query"]
        self.assertFalse(query["feedback"])
        self.assertFalse(query["distance_input"])
        self.assertTrue(query["reset_hidden_each_trial"])
        self.assertTrue(query["reset_eligibility_each_trial"])
        self.assertEqual(query["fast_weights"], "freeze_after_support")
        self.assertEqual(query["time_channel"], "constant_at_support_query_boundary")
