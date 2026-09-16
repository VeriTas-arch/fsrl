import unittest

import numpy as np
import torch

from fsrl.experiments.pl_direct_training.batches import (
    expand_legacy_inputs,
    prepare_batch,
    sample_episodes,
)
from fsrl.experiments.pl_direct_training.protocol import load_specification
from fsrl.experiments.pl_direct_training.task import make_task_generator


class DirectBatchTests(unittest.TestCase):
    def test_registered_horizon_and_factorization(self):
        specification = load_specification()
        episodes = sample_episodes(
            make_task_generator(specification),
            np.random.default_rng(103001),
            4,
            validation=False,
        )
        cpu = prepare_batch(episodes)
        batch = cpu.to("cpu")
        support_trials = len(episodes[0].support_trials)
        self.assertIn(support_trials, range(28, 41, 4))
        self.assertEqual(batch.support_inputs.shape, (support_trials, 4, 4, 32))
        self.assertEqual(batch.support_times.shape, (support_trials, 4, 4, 1))
        self.assertEqual(batch.query_inputs.shape, (2, 112, 32))
        self.assertEqual(4 * support_trials + 56, 4 * support_trials + 2 * 28)
        self.assertIn(2 * support_trials, range(56, 81, 8))
        legacy = expand_legacy_inputs(
            batch.support_inputs, batch.support_times, cue_size=15
        )
        self.assertEqual(legacy.shape[-1], 37)
        self.assertTrue(torch.equal(legacy[..., 31], torch.ones_like(legacy[..., 31])))
        self.assertTrue(torch.equal(legacy[..., 32:33], batch.support_times))
        self.assertTrue(torch.equal(legacy[..., 34], batch.support_inputs[..., 31]))
        self.assertEqual(int(torch.count_nonzero(legacy[..., 33])), 0)
        self.assertEqual(int(torch.count_nonzero(legacy[..., 35:])), 0)


if __name__ == "__main__":
    unittest.main()
