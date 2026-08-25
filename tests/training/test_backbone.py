import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fsrl.training.backbone import (
    COMPILED_TRAINING_EXECUTION,
    MetaTrainConfig,
    build_meta_input_sequence,
    build_meta_inputs,
    compile_meta_model,
    make_model_and_tasks,
    run_meta_batch,
    save_meta_checkpoint,
)


class MetaTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.training_config = MetaTrainConfig(
            seed=41,
            outer_steps=1,
            batch_size=2,
            hidden_size=8,
            cue_size=8,
            min_edges=7,
            max_edges=7,
            support_blocks=1,
            save_every=1,
        )
        self.model_config, self.net, self.generator = make_model_and_tasks(
            self.training_config
        )

    def test_input_builder_has_no_query_target_argument(self):
        parameters = inspect.signature(build_meta_inputs).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("correct_action", parameters)
        self.assertNotIn("label", parameters)

    def test_batched_input_sequence_matches_stepwise_builder(self):
        rng = np.random.default_rng(51)
        episodes = tuple(
            self.generator.sample(rng, n_edges=7) for _ in range(self.model_config.bs)
        )
        trials = [episode.support_trials[0] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [trial.signed_magnitude for trial in trials], dtype=np.float32
        )
        sequence = build_meta_input_sequence(
            self.model_config,
            episodes,
            left,
            right,
            signed,
            num_steps=self.model_config.triallen,
            time_value=0.25,
            support_trial=True,
        )
        for numstep, observed in enumerate(sequence.unbind()):
            expected = build_meta_inputs(
                self.model_config,
                episodes,
                left,
                right,
                signed,
                numstep=numstep,
                time_value=0.25,
                support_trial=True,
            )
            self.assertTrue(torch.equal(observed, expected))

    def test_one_meta_batch_backpropagates_through_passive_support(self):
        stats = run_meta_batch(
            self.training_config,
            self.model_config,
            self.net,
            self.generator,
            np.random.default_rng(41),
        )
        self.assertTrue(torch.isfinite(stats.loss))
        self.assertTrue(0.0 <= stats.query_accuracy <= 1.0)
        self.assertGreater(stats.mean_abs_fast_weight, 0.0)
        stats.loss.backward()
        self.assertIsNotNone(self.net.alpha.grad)
        self.assertGreater(float(torch.sum(torch.abs(self.net.alpha.grad))), 0.0)

    def test_compiler_uses_fullgraph_default_mode(self):
        with patch(
            "fsrl.training.backbone.torch.compile", return_value=self.net
        ) as compiler:
            self.assertIs(compile_meta_model(self.net), self.net)
        compiler.assert_called_once_with(
            self.net,
            backend="inductor",
            fullgraph=True,
            mode="default",
        )

    def test_saved_config_is_valid_json_and_registers_held_out_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_meta_checkpoint(
                output_dir,
                self.net,
                self.training_config,
                step=0,
                execution=COMPILED_TRAINING_EXECUTION,
            )
            with (output_dir / "config.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertTrue(metadata["task_distribution"]["liu_graph_held_out"])
            self.assertFalse(
                metadata["task_distribution"]["query_labels_enter_episode_inputs"]
            )
            subject_encoding = metadata["task_distribution"]["subject_encoding"]
            self.assertEqual(
                subject_encoding["state_scope"], "fixed_for_entire_episode"
            )
            self.assertEqual(subject_encoding["mode"], "stable_omission")
            self.assertFalse(subject_encoding["contains_rank_label"])
            self.assertEqual(len(metadata["checkpoint"]["sha256"]), 64)
            self.assertEqual(metadata["execution"], COMPILED_TRAINING_EXECUTION)


if __name__ == "__main__":
    unittest.main()
