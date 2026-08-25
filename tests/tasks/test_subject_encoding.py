import unittest

import numpy as np

from fsrl.tasks.meta_tasks import GenericRankingTaskGenerator
from fsrl.tasks.subject_encoding import (
    SubjectEncodingConfig,
    sample_subject_encoding_states,
)


class SubjectEncodingTests(unittest.TestCase):
    def test_seed_reproduces_latent_state(self):
        first = sample_subject_encoding_states(np.random.default_rng(61), 3, 8)
        second = sample_subject_encoding_states(np.random.default_rng(61), 3, 8)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], first[1])

    def test_reliability_is_bounded_and_contains_no_rank_label(self):
        state = sample_subject_encoding_states(
            np.random.default_rng(62), 1, 8, SubjectEncodingConfig()
        )[0]
        reliability = state.relation_reliability(1, 6, 3)
        self.assertGreaterEqual(reliability, state.minimum_reliability)
        self.assertLess(reliability, 1.0)
        self.assertFalse(hasattr(state, "true_order"))
        self.assertFalse(hasattr(state, "subjective_order"))

    def test_one_relation_keeps_one_reliability_across_support_blocks(self):
        episode = GenericRankingTaskGenerator(cue_size=8).sample(
            np.random.default_rng(63), n_edges=8
        )
        by_relation = {}
        for trial in episode.support_trials:
            relation = (trial.higher_item, trial.lower_item)
            by_relation.setdefault(relation, set()).add(trial.encoding_reliability)
        self.assertEqual(len(by_relation), 8)
        self.assertTrue(all(len(values) == 1 for values in by_relation.values()))
        realized = {next(iter(values)) for values in by_relation.values()}
        self.assertTrue(realized <= {0.0, 1.0})
        self.assertEqual(realized, {0.0, 1.0})
