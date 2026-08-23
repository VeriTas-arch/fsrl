import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.config import TrainConfig
from fsrl.liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    deterministic_cue_codes,
    load_retro_checkpoint,
)
from fsrl.model import RetroModulRNN
from fsrl.ranking_protocol import load_ranking_protocol


class LiuEvaluatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.net.eval()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            load_ranking_protocol(),
            cue_seed=5,
            support_seed=7,
        )

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
            loaded, config, info = load_retro_checkpoint(path, batch_size=2)
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
        protocol = load_ranking_protocol("benchmarks/liu_v2.json")
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
                        if evaluator.item_rank[pair[0]]
                        < evaluator.item_rank[pair[1]]
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


if __name__ == "__main__":
    unittest.main()
