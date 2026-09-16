import unittest

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.clean_single_p.evaluation import legacy_input_record
from fsrl.experiments.clean_single_p.model import (
    AffineSinglePSequence,
    map_shadow,
)
from fsrl.experiments.clean_single_p.optimization import (
    forward_batch,
    make_optimizer,
    training_step,
)
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.experiments.memory_structure.inputs import generator
from fsrl.experiments.pl_direct_training.protocol import load_specification
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes
from fsrl.infra.provenance import load_json
from fsrl.paths import STUDIES_ROOT


class CleanSinglePTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(
            STUDIES_ROOT / "clean_single_p/records/benchmarks/clean_single_p_v1.json"
        )
        parent = load_specification()
        cls.task_spec = {
            "task": parent["task"],
            "optimization": {
                "batch_size": 3,
            },
        }

    def batch(self, arm="clean"):
        episodes = sample_episodes(
            generator(self.task_spec), np.random.default_rng(931001), 3
        )
        return prepare_single_p(episodes, arm, observation_seed=931002).to("cpu")

    def model(self, condition):
        torch.manual_seed(931001)
        shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cpu")
        model = map_shadow(shadow, condition)
        return model, AffineSinglePSequence(model)

    def test_clean_batch_has_one_observation_channel(self):
        batch, _ = self.batch("noisy")
        self.assertEqual(batch.support_inputs.shape[-1], 32)
        self.assertEqual(batch.query_inputs.shape[-1], 32)
        self.assertTrue(torch.count_nonzero(batch.query_inputs[..., -1]) == 0)

    def test_forward_and_update_both_conditions(self):
        batch, times = self.batch()
        for condition in ("time_retained_control", "clean_no_time"):
            with self.subTest(condition=condition):
                model, sequence = self.model(condition)
                result = forward_batch(
                    condition, model, sequence, batch, times, penalty=1e-4
                )
                self.assertEqual(result.margins.shape, (84, 1))
                self.assertTrue(torch.isfinite(result.loss))
                before = model.w.detach().clone()
                training_step(
                    condition,
                    model,
                    sequence,
                    batch,
                    times,
                    make_optimizer(model, self.protocol),
                    self.protocol,
                )
                self.assertFalse(torch.equal(before, model.w))

    def test_direct_input_reference_is_adapted_for_legacy_loader(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        arrays = {"values": np.arange(6, dtype=np.float32).reshape(2, 3)}
        direct = {"path": "ignored.npz", "sha256": "0" * 64, "bytes": 0}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, **arrays)
            with patch(
                "fsrl.experiments.clean_single_p.evaluation.verify_reference",
                return_value=path,
            ):
                adapted = legacy_input_record(direct)

        self.assertIs(adapted["file"], direct)
        self.assertEqual(adapted["fingerprint"], EpisodeBatch(arrays).fingerprint())


if __name__ == "__main__":
    unittest.main()
