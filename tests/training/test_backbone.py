import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fsrl.infra.runtime import DEFAULT_COMPILED_PROFILE
from fsrl.training.backbone import (
    COMPILED_TRAINING_EXECUTION,
    OPTIMIZED_COMPILED_TRAINING_EXECUTION,
    OPTIMIZED_TRAINING_PROFILE,
    MetaTrainConfig,
    RecurrentSequence,
    build_meta_input_sequence,
    build_meta_inputs,
    compile_meta_model,
    compile_meta_sequence,
    compiled_execution_record,
    make_model_and_tasks,
    optimized_compiled_execution_record,
    run_meta_batch,
    run_optimized_meta_batch,
    save_meta_checkpoint,
)
from fsrl.training.backbone import main as training_main


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
        for support_trial, num_steps in (
            (True, self.model_config.triallen),
            (False, 2),
        ):
            with self.subTest(support_trial=support_trial):
                sequence = build_meta_input_sequence(
                    self.model_config,
                    episodes,
                    left,
                    right,
                    signed,
                    num_steps=num_steps,
                    time_value=0.25,
                    support_trial=support_trial,
                )
                expected = torch.stack(
                    [
                        build_meta_inputs(
                            self.model_config,
                            episodes,
                            left,
                            right,
                            signed,
                            numstep=numstep,
                            time_value=0.25,
                            support_trial=support_trial,
                        )
                        for numstep in range(num_steps)
                    ]
                )
                self.assertTrue(torch.equal(sequence, expected))

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

    def test_sequence_runner_matches_stepwise_values_and_gradients(self):
        device = next(self.net.parameters()).device
        input_sequence = torch.linspace(
            -0.5,
            0.5,
            steps=4 * self.model_config.bs * self.model_config.inputsize,
            device=device,
        ).reshape(4, self.model_config.bs, self.model_config.inputsize)
        for update_fast_weights in (False, True):
            with self.subTest(update_fast_weights=update_fast_weights):
                stepwise_net = copy.deepcopy(self.net)
                sequence_net = copy.deepcopy(self.net)
                hidden = stepwise_net.initialZeroState(self.model_config.bs)
                eligibility = stepwise_net.initialZeroET(self.model_config.bs)
                fast_weights = stepwise_net.initialZeroPlasticWeights(
                    self.model_config.bs
                )
                stepwise_outputs = None
                for inputs in input_sequence.unbind(0):
                    stepwise_outputs = stepwise_net(
                        inputs, hidden, eligibility, fast_weights
                    )
                    _, _, _, hidden, eligibility, proposed_fast_weights = (
                        stepwise_outputs
                    )
                    if update_fast_weights:
                        fast_weights = proposed_fast_weights
                assert stepwise_outputs is not None
                stepwise_outputs = (*stepwise_outputs[:5], fast_weights)

                sequence_outputs = RecurrentSequence(sequence_net)(
                    input_sequence,
                    sequence_net.initialZeroState(self.model_config.bs),
                    sequence_net.initialZeroET(self.model_config.bs),
                    sequence_net.initialZeroPlasticWeights(self.model_config.bs),
                    update_fast_weights,
                )
                for observed, expected in zip(
                    sequence_outputs, stepwise_outputs, strict=True
                ):
                    self.assertTrue(torch.equal(observed, expected))

                sum(tensor.sum() for tensor in stepwise_outputs).backward()
                sum(tensor.sum() for tensor in sequence_outputs).backward()
                for stepwise_parameter, sequence_parameter in zip(
                    stepwise_net.parameters(), sequence_net.parameters(), strict=True
                ):
                    if stepwise_parameter.grad is None:
                        self.assertIsNone(sequence_parameter.grad)
                    else:
                        self.assertTrue(
                            torch.equal(
                                stepwise_parameter.grad, sequence_parameter.grad
                            )
                        )

    def test_optimized_batch_preserves_loss_accuracy_and_gradients(self):
        stepwise_net = copy.deepcopy(self.net)
        optimized_net = copy.deepcopy(self.net)
        stepwise_stats = run_meta_batch(
            self.training_config,
            self.model_config,
            stepwise_net,
            self.generator,
            np.random.default_rng(97),
        )
        optimized_stats = run_optimized_meta_batch(
            self.training_config,
            self.model_config,
            optimized_net,
            RecurrentSequence(optimized_net),
            self.generator,
            np.random.default_rng(97),
        )
        self.assertEqual(stepwise_stats.n_edges, optimized_stats.n_edges)
        self.assertEqual(stepwise_stats.query_accuracy, optimized_stats.query_accuracy)
        self.assertAlmostEqual(
            stepwise_stats.query_cross_entropy,
            optimized_stats.query_cross_entropy,
            places=6,
        )
        self.assertEqual(
            stepwise_stats.mean_abs_fast_weight,
            optimized_stats.mean_abs_fast_weight,
        )
        torch.testing.assert_close(
            stepwise_stats.loss.detach(), optimized_stats.loss.detach()
        )

        stepwise_stats.loss.backward()
        optimized_stats.loss.backward()
        for stepwise_parameter, optimized_parameter in zip(
            stepwise_net.parameters(), optimized_net.parameters(), strict=True
        ):
            if stepwise_parameter.grad is None:
                self.assertIsNone(optimized_parameter.grad)
            else:
                torch.testing.assert_close(
                    stepwise_parameter.grad,
                    optimized_parameter.grad,
                    rtol=1e-5,
                    atol=1e-7,
                )

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

    def test_sequence_compiler_uses_fullgraph_low_overhead_mode(self):
        compiled_runner = object()
        with patch(
            "fsrl.training.backbone.torch.compile", return_value=compiled_runner
        ) as compiler:
            self.assertIs(compile_meta_sequence(self.net), compiled_runner)
        sequence_runner = compiler.call_args.args[0]
        self.assertIsInstance(sequence_runner, RecurrentSequence)
        self.assertIs(sequence_runner.cell, self.net)
        compiler.assert_called_once_with(
            sequence_runner,
            backend="inductor",
            fullgraph=True,
            mode="reduce-overhead",
        )

    def test_optimized_execution_record_is_versioned(self):
        self.assertEqual(
            optimized_compiled_execution_record(OPTIMIZED_TRAINING_PROFILE),
            OPTIMIZED_COMPILED_TRAINING_EXECUTION,
        )
        self.assertEqual(
            OPTIMIZED_COMPILED_TRAINING_EXECUTION["execution_schema_version"], 3
        )
        self.assertEqual(
            OPTIMIZED_COMPILED_TRAINING_EXECUTION["cuda_graph_iteration_boundary"],
            "explicit_outer_step",
        )
        self.assertEqual(
            OPTIMIZED_COMPILED_TRAINING_EXECUTION["item_code_sampling"],
            "sequential_candidates_vectorized_similarity_check",
        )
        self.assertEqual(
            OPTIMIZED_COMPILED_TRAINING_EXECUTION["host_trial_sequence_assembly"],
            "single_preallocated_numpy_array",
        )
        self.assertEqual(
            compiled_execution_record(DEFAULT_COMPILED_PROFILE),
            COMPILED_TRAINING_EXECUTION,
        )

    def test_cli_keeps_legacy_mode_and_selects_prospective_cuda_mode(self):
        cases = (
            (["--compile-model"], "default", False),
            (["--optimized-execution"], "reduce-overhead", True),
            (
                [
                    "--optimized-execution",
                    "--compile-mode",
                    "max-autotune-no-cudagraphs",
                ],
                "max-autotune-no-cudagraphs",
                True,
            ),
        )
        for compile_args, expected_mode, optimized_execution in cases:
            with (
                self.subTest(compile_args=compile_args),
                patch("fsrl.training.backbone.train_meta_model") as trainer,
            ):
                training_main(
                    [
                        "--output-dir",
                        "/tmp/fsrl-training-cli-test",
                        "--device",
                        "cuda",
                        *compile_args,
                    ]
                )
                profile = trainer.call_args.kwargs["execution_profile"]
                self.assertEqual(profile.compile_mode, expected_mode)
                self.assertEqual(
                    trainer.call_args.kwargs["optimized_execution"],
                    optimized_execution,
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

    def test_prospective_checkpoint_records_observed_runtime(self):
        runtime = {
            "execution_schema_version": 2,
            "blas_thread_limit": 1,
            "float32_matmul_precision": "highest",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_meta_checkpoint(
                output_dir,
                self.net,
                self.training_config,
                step=0,
                execution=OPTIMIZED_COMPILED_TRAINING_EXECUTION,
                runtime=runtime,
            )
            metadata = json.loads((output_dir / "config.json").read_text())
        self.assertEqual(metadata["runtime"], runtime)
        self.assertEqual(metadata["execution"]["runtime_profile"]["blas_threads"], 1)
