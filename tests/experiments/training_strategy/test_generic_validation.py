import unittest
from dataclasses import replace

import numpy as np

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.training_strategy.batches import graph_bucket, prepare_batch
from fsrl.experiments.training_strategy.generic_validation import (
    _group_rollout,
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.protocol import (
    load_specification,
    training_config,
)
from fsrl.training.backbone import make_model_and_tasks, registered_excluded_signatures


class GenericValidationTests(unittest.TestCase):
    def test_heldout_stream_and_grouping_preserve_episode_identity(self):
        spec = load_specification()
        spec["evaluation"]["generic"]["episodes"] = 12
        episodes = validation_episodes(spec)
        repeated = validation_episodes(spec)
        groups = validation_groups(episodes)
        self.assertEqual(
            sorted(index for indices in groups.values() for index in indices),
            list(range(12)),
        )
        for episode, duplicate in zip(episodes, repeated, strict=True):
            self.assertEqual(graph_bucket(episode.graph_rank_pairs), 0)
            self.assertNotIn(episode.graph_rank_pairs, registered_excluded_signatures())
            self.assertEqual(
                prepare_batch((episode,)).fingerprint(),
                prepare_batch((duplicate,)).fingerprint(),
            )
        config = replace(training_config(spec, 910007), hidden_size=8, batch_size=2)
        _, backbone, _ = make_model_and_tasks(config, device="cpu")
        local = ConjunctiveLocalTrace(15, device="cpu")
        indices = next(iter(groups.values()))
        selected = tuple(episodes[index] for index in indices)
        result = _group_rollout(backbone, local, RecurrentSequence(backbone), selected)
        self.assertEqual(result["margins"]["intact"].shape, (len(indices), 28))
        np.testing.assert_array_equal(
            result["learned"].sum(axis=1), len(selected[0].graph_rank_pairs)
        )
        np.testing.assert_array_equal(
            result["correct_signs"],
            (2 * prepare_batch(selected).arrays["targets"] - 1)
            .reshape(28, len(indices))
            .T,
        )
