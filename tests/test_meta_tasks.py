import unittest

import numpy as np

from fsrl.meta_tasks import (
    GenericRankingTaskGenerator,
    graph_is_connected,
    held_out_liu_graph_signatures,
    liu_graph_signature,
)


class GenericRankingTaskTests(unittest.TestCase):
    def setUp(self):
        self.generator = GenericRankingTaskGenerator(cue_size=8)

    def test_samples_connected_sparse_graphs_and_holds_out_liu(self):
        rng = np.random.default_rng(31)
        held_out = held_out_liu_graph_signatures()
        self.assertEqual(len(held_out), 2)
        self.assertIn(liu_graph_signature(), held_out)
        observed_sizes = set()
        for _ in range(100):
            episode = self.generator.sample(rng)
            observed_sizes.add(len(episode.graph_rank_pairs))
            self.assertTrue(graph_is_connected(8, episode.graph_rank_pairs))
            self.assertNotIn(episode.graph_rank_pairs, held_out)
            self.assertGreaterEqual(
                len({abs(a - b) for a, b in episode.graph_rank_pairs}), 2
            )
        self.assertEqual(observed_sizes, {7, 8, 9, 10})

    def test_episode_has_random_codes_permutation_and_all_queries(self):
        episode = self.generator.sample(np.random.default_rng(32), n_edges=9)
        self.assertEqual(episode.item_codes.shape, (8, 8))
        self.assertEqual(set(episode.true_order_high_to_low), set(range(8)))
        self.assertEqual(len(episode.support_trials), 36)
        self.assertEqual(len(episode.query_trials), 28)
        self.assertEqual(len(episode.subject_encoding.item_salience), 8)
        self.assertEqual(
            len(
                {
                    tuple(sorted((trial.left_item, trial.right_item)))
                    for trial in episode.query_trials
                }
            ),
            28,
        )

    def test_each_support_edge_appears_once_per_block_with_correct_sign(self):
        episode = self.generator.sample(np.random.default_rng(33), n_edges=8)
        rank = {
            item: position
            for position, item in enumerate(episode.true_order_high_to_low)
        }
        counts = {}
        for trial in episode.support_trials:
            pair = (trial.higher_item, trial.lower_item)
            counts[pair] = counts.get(pair, 0) + 1
            magnitude = (rank[trial.lower_item] - rank[trial.higher_item]) / 7.0
            expected = magnitude if trial.left_item == trial.higher_item else -magnitude
            self.assertAlmostEqual(trial.signed_magnitude, expected)
        self.assertEqual(len(counts), 8)
        self.assertEqual(set(counts.values()), {4})

    def test_seed_reproduces_full_episode(self):
        first = self.generator.sample(np.random.default_rng(34), n_edges=7)
        second = self.generator.sample(np.random.default_rng(34), n_edges=7)
        np.testing.assert_array_equal(first.item_codes, second.item_codes)
        self.assertEqual(first.true_order_high_to_low, second.true_order_high_to_low)
        self.assertEqual(first.graph_rank_pairs, second.graph_rank_pairs)
        self.assertEqual(first.support_trials, second.support_trials)
        self.assertEqual(first.query_trials, second.query_trials)


if __name__ == "__main__":
    unittest.main()
