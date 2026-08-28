import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenEvaluationBackend,
    FrozenFastWeightEvaluator,
    deterministic_cue_codes,
    load_frozen_retro_checkpoint,
    run_causal_suite,
)
from fsrl.infra.runtime import ExecutionProfile
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class FrozenFastWeightEvaluatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict(), device="cpu")
        self.net.eval()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            load_ranking_protocol(),
            cue_seed=5,
            support_seed=7,
        )

    def _batched_evaluator(self):
        return FrozenFastWeightEvaluator(
            self.net,
            self.config,
            load_ranking_protocol(),
            cue_seed=5,
            support_seed=7,
            backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
            execution_profile=ExecutionProfile(
                device="cpu",
                compile=False,
                require_cuda=False,
            ),
        )

    def test_protocol_only_mode_requires_an_explicit_non_neural_contract(self):
        protocol = load_ranking_protocol()
        with self.assertRaisesRegex(ValueError, "neural network is required"):
            FrozenFastWeightEvaluator(None, self.config, protocol)

        evaluator = FrozenFastWeightEvaluator(
            None,
            self.config,
            protocol,
            cue_seed=5,
            support_seed=7,
            protocol_only=True,
        )
        evidence = evaluator.realized_support_evidence()
        self.assertEqual(len(evidence), self.config.bs)
        self.assertTrue(all(len(rows) == protocol.support_trials for rows in evidence))
        with self.assertRaisesRegex(RuntimeError, "cannot execute neural rollouts"):
            evaluator.initialize_fast_weights()

    def test_batched_input_sequence_matches_legacy_step_builder(self):
        batched = self._batched_evaluator()
        trials = [schedule[0] for schedule in batched.support_schedules]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [trial.signed_magnitude for trial in trials], dtype=np.float32
        )
        sequence = batched._batched_input_sequence(
            left,
            right,
            signed,
            num_steps=self.config.triallen,
            time_value=0.25,
            support_trial=True,
        )
        for step, observed in enumerate(sequence.unbind(0)):
            expected = batched._step_inputs(
                left,
                right,
                signed,
                numstep=step,
                time_value=0.25,
                support_trial=True,
            )
            self.assertTrue(torch.equal(observed, expected))

    def test_batched_backend_preserves_fast_weights_and_query_logits(self):
        batched = self._batched_evaluator()
        pairs = tuple(
            oriented
            for first, second in combinations(range(self.evaluator.protocol.n_items), 2)
            for oriented in ((first, second), (second, first))
        )
        schedules = tuple(pairs for _ in range(self.config.bs))
        for intervention in FastWeightIntervention:
            with self.subTest(intervention=intervention.value):
                legacy_weights = self.evaluator.learn_fast_weights(intervention)
                batched_weights = batched.learn_fast_weights(intervention)
                torch.testing.assert_close(
                    batched_weights, legacy_weights, rtol=0.0, atol=0.0
                )
                legacy_logits = self.evaluator.readout_logits(
                    legacy_weights,
                    schedules,
                    alpha_zero=intervention == FastWeightIntervention.ALPHA_ZERO,
                )
                batched_logits = batched.readout_logits(
                    batched_weights,
                    schedules,
                    alpha_zero=intervention == FastWeightIntervention.ALPHA_ZERO,
                )
                for legacy_subject, batched_subject in zip(
                    legacy_logits, batched_logits, strict=True
                ):
                    self.assertEqual(set(legacy_subject), set(batched_subject))
                    np.testing.assert_allclose(
                        list(batched_subject.values()),
                        list(legacy_subject.values()),
                        rtol=1e-6,
                        atol=1e-7,
                    )

    def test_batched_backend_preserves_hidden_and_logit_trajectories(self):
        batched = self._batched_evaluator()
        pairs = tuple(combinations(range(self.evaluator.protocol.n_items), 2))
        schedules = tuple(pairs for _ in range(self.config.bs))
        fast_weights = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        for alpha_zero in (False, True):
            with self.subTest(alpha_zero=alpha_zero):
                legacy_hidden, legacy_logits = (
                    self.evaluator.readout_hidden_and_logit_trajectories(
                        fast_weights, schedules, alpha_zero=alpha_zero
                    )
                )
                batched_hidden, batched_logits = (
                    batched.readout_hidden_and_logit_trajectories(
                        fast_weights, schedules, alpha_zero=alpha_zero
                    )
                )
                for subject in range(self.config.bs):
                    self.assertEqual(
                        set(legacy_hidden[subject]), set(batched_hidden[subject])
                    )
                    self.assertEqual(
                        set(legacy_logits[subject]), set(batched_logits[subject])
                    )
                    for pair in pairs:
                        np.testing.assert_allclose(
                            batched_hidden[subject][pair],
                            legacy_hidden[subject][pair],
                            rtol=1e-6,
                            atol=1e-7,
                        )
                        np.testing.assert_allclose(
                            batched_logits[subject][pair],
                            legacy_logits[subject][pair],
                            rtol=1e-6,
                            atol=1e-7,
                        )

    def test_batched_backend_records_explicit_prospective_execution(self):
        record = self._batched_evaluator().evaluation_execution_record()
        self.assertEqual(record["execution_schema_version"], 2)
        self.assertEqual(record["backend"], "batched_sequence")
        self.assertFalse(record["profile"]["compile"])
        self.assertEqual(record["query_batching"], "all_query_pairs_by_subject")
        self.assertEqual(
            record["trajectory_transfer"],
            "one_hidden_and_one_logit_batched_device_to_cpu_transfer",
        )

    def test_batched_causal_suite_records_observed_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "net.dat"
            torch.save(self.net.state_dict(), checkpoint)
            result = run_causal_suite(
                checkpoint,
                batch_size=self.config.bs,
                cue_seed=5,
                support_seed=7,
                order_seed=11,
                order_schedules=2,
                cue_mode="shared",
                subject_encoding_mode="none",
                subject_encoding_seed=0,
                evaluation_backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
                execution_profile=ExecutionProfile(
                    device="cpu",
                    compile=False,
                    require_cuda=False,
                ),
            )
        execution = result["evaluation_execution"]
        self.assertEqual(execution["execution_schema_version"], 2)
        self.assertEqual(execution["runtime"]["execution_schema_version"], 2)
        self.assertEqual(execution["runtime"]["blas_thread_limit"], 1)

    def test_write_off_and_reset_remove_fast_state(self):
        intact = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        write_off = self.evaluator.learn_fast_weights(FastWeightIntervention.WRITE_OFF)
        reset = self.evaluator.learn_fast_weights(FastWeightIntervention.RESET)
        self.assertGreater(float(torch.mean(torch.abs(intact))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(write_off))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(reset))), 0.0)

    def test_incremental_support_endpoint_matches_intact_evaluation(self):
        fast_weights = self.evaluator.initialize_fast_weights()
        for trial_index in range(self.evaluator.protocol.support_trials):
            fast_weights = self.evaluator.advance_support_trial(
                fast_weights, trial_index
            )
        expected = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        torch.testing.assert_close(fast_weights, expected)

    def test_zero_evidence_preserves_trial_but_removes_magnitude(self):
        fast_weights = self.evaluator.initialize_fast_weights()
        explicit_zero = self.evaluator.advance_support_trial(
            fast_weights, 0, zero_evidence=True
        )
        relation_zero = self.evaluator.advance_support_trial(
            fast_weights,
            0,
            zero_relations=frozenset(
                self.evaluator.protocol.support_pairs_higher_lower
            ),
        )
        natural = self.evaluator.advance_support_trial(fast_weights, 0)
        torch.testing.assert_close(explicit_zero, relation_zero)
        self.assertGreater(float(torch.max(torch.abs(natural - explicit_zero))), 0.0)

    def test_shuffle_rotates_subject_fast_states(self):
        intact = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        shuffled = self.evaluator.learn_fast_weights(FastWeightIntervention.SHUFFLE)
        torch.testing.assert_close(shuffled, torch.roll(intact, shifts=1, dims=0))

    def test_strict_frozen_readout_is_test_order_invariant(self):
        intact = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        metrics = self.evaluator.order_invariance(intact, schedules=4, seed=13)
        self.assertEqual(metrics.pairs, 56)
        self.assertLessEqual(metrics.max_abs_logit_delta, 1e-7)

    def test_hidden_readout_is_query_order_invariant(self):
        fast_weights = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        forward = tuple(combinations(range(8), 2))
        reverse = tuple(reversed(forward))
        first = self.evaluator.readout_hidden_states(
            fast_weights, tuple(forward for _ in range(self.config.bs))
        )
        second = self.evaluator.readout_hidden_states(
            fast_weights, tuple(reverse for _ in range(self.config.bs))
        )
        for subject in range(self.config.bs):
            for pair in forward:
                np.testing.assert_allclose(
                    first[subject][pair], second[subject][pair], atol=1e-7
                )

    def test_hidden_trajectory_contains_registered_response_state(self):
        fast_weights = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        pairs = tuple(combinations(range(8), 2))
        schedules = tuple(pairs for _ in range(self.config.bs))
        response = self.evaluator.readout_hidden_states(fast_weights, schedules)
        trajectories, logit_trajectories = (
            self.evaluator.readout_hidden_and_logit_trajectories(
                fast_weights, schedules
            )
        )
        response_logits = self.evaluator.readout_logits(fast_weights, schedules)
        for subject in range(self.config.bs):
            for pair in pairs:
                self.assertEqual(
                    trajectories[subject][pair].shape,
                    (self.config.triallen, self.config.hs),
                )
                np.testing.assert_allclose(
                    trajectories[subject][pair][1], response[subject][pair]
                )
                self.assertAlmostEqual(
                    logit_trajectories[subject][pair][1],
                    response_logits[subject][pair],
                    places=6,
                )

    def test_alpha_is_restored_after_alpha_zero_control(self):
        before = self.net.alpha.detach().clone()
        self.evaluator.learn_fast_weights(FastWeightIntervention.ALPHA_ZERO)
        torch.testing.assert_close(self.net.alpha, before)

    def test_condition_accuracy_balances_both_pair_orientations(self):
        original_readout = self.evaluator.readout_logits

        def always_choose_left(_fast_weights, pair_schedules, **_kwargs):
            return tuple(
                {pair: 1.0 for pair in schedule} for schedule in pair_schedules
            )

        self.evaluator.readout_logits = always_choose_left
        try:
            metrics = self.evaluator.condition_metrics(FastWeightIntervention.INTACT)
        finally:
            self.evaluator.readout_logits = original_readout
        self.assertEqual(metrics.overall_accuracy, 0.5)
        self.assertEqual(metrics.learned_accuracy, 0.5)
        self.assertEqual(metrics.nonlearned_accuracy, 0.5)

    def test_checkpoint_loader_registers_shape_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "net.dat"
            torch.save(self.net.state_dict(), path)
            loaded, config, info = load_frozen_retro_checkpoint(path, batch_size=2)
        self.assertEqual(config.bs, 2)
        self.assertEqual(info.hidden_size, self.config.hs)
        self.assertEqual(info.cue_size, self.config.cs)
        self.assertEqual(len(info.sha256), 64)
        self.assertEqual(loaded.i2h.in_features, self.config.inputsize)

    def test_permuted_shared_codes_preserve_codebook_but_change_item_mapping(self):
        codes = deterministic_cue_codes(3, 8, 8, 17, mode="permuted_shared")
        reference = {tuple(code) for code in codes[0]}
        self.assertEqual(set(codes.ravel()), {-1.0, 1.0})
        self.assertEqual({tuple(code) for code in codes[1]}, reference)
        self.assertFalse(
            torch.equal(torch.from_numpy(codes[0]), torch.from_numpy(codes[1]))
        )

    def test_pathological_six_bit_seed_completes_with_valid_codebook(self):
        codes = deterministic_cue_codes(3, 8, 6, 5)
        self.assertEqual(codes.shape, (3, 8, 6))
        for first, second in combinations(codes[0], 2):
            self.assertLessEqual(float(np.mean(first == second)), 0.66)

    def test_impossible_codebook_fails_after_bounded_search(self):
        with self.assertRaisesRegex(ValueError, "Could not construct 3"):
            deterministic_cue_codes(1, 3, 1, 5)

    def test_stable_omission_is_binary_and_fixed_across_blocks(self):
        evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            load_ranking_protocol(),
            cue_seed=5,
            support_seed=7,
            subject_encoding_mode="stable_omission",
            subject_encoding_seed=19,
        )
        realized = {
            gain
            for subject_gains in evaluator.subject_relation_gains or ()
            for gain in subject_gains.values()
        }
        self.assertTrue(realized <= {0.0, 1.0})
        self.assertEqual(realized, {0.0, 1.0})
        schedule = evaluator.support_schedules[0]
        by_relation = {}
        for trial_index, trial in enumerate(schedule):
            relation = (trial.higher_item, trial.lower_item)
            by_relation.setdefault(relation, set()).add(
                evaluator._encoding_reliability(0, trial_index)
            )
        self.assertTrue(all(len(values) == 1 for values in by_relation.values()))

    def test_temporal_omission_controls_are_binary_and_resampled(self):
        for mode in ("presentationwise_omission", "blockwise_omission"):
            evaluator = FrozenFastWeightEvaluator(
                self.net,
                self.config,
                load_ranking_protocol(),
                cue_seed=5,
                support_seed=7,
                subject_encoding_mode=mode,
                subject_encoding_seed=19,
            )
            realized = {
                gain
                for subject_gains in evaluator.subject_trial_gains or ()
                for gain in subject_gains
            }
            self.assertTrue(realized <= {0.0, 1.0})
            self.assertEqual(realized, {0.0, 1.0})

    def test_uniform_control_has_no_subject_or_relation_bottleneck(self):
        evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            load_ranking_protocol(),
            cue_seed=5,
            support_seed=7,
            subject_encoding_mode="uniform_no_bottleneck",
            subject_encoding_seed=19,
        )
        realized = {
            gain
            for subject_gains in evaluator.subject_trial_gains or ()
            for gain in subject_gains
        }
        self.assertEqual(len(realized), 1)
        self.assertTrue(0.0 < next(iter(realized)) < 1.0)

    def test_source_corrected_protocol_uses_registered_true_order(self):
        protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            protocol,
            cue_seed=5,
            support_seed=7,
        )
        original_readout = evaluator.readout_logits

        def correct_source_order(_fast_weights, pair_schedules, **_kwargs):
            return tuple(
                {
                    pair: (
                        1.0
                        if evaluator.item_rank[pair[0]] < evaluator.item_rank[pair[1]]
                        else -1.0
                    )
                    for pair in schedule
                }
                for schedule in pair_schedules
            )

        evaluator.readout_logits = correct_source_order
        try:
            metrics = evaluator.condition_metrics(FastWeightIntervention.INTACT)
        finally:
            evaluator.readout_logits = original_readout
        self.assertEqual(metrics.overall_accuracy, 1.0)
        self.assertEqual(metrics.learned_accuracy, 1.0)
        self.assertEqual(metrics.nonlearned_accuracy, 1.0)
