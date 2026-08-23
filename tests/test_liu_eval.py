import tempfile
import unittest
from pathlib import Path

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

    def test_shuffle_rotates_subject_fast_states(self):
        intact = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        shuffled = self.evaluator.learn_fast_weights(FastWeightIntervention.SHUFFLE)
        torch.testing.assert_close(shuffled, torch.roll(intact, shifts=1, dims=0))

    def test_strict_frozen_readout_is_test_order_invariant(self):
        intact = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        metrics = self.evaluator.order_invariance(intact, schedules=4, seed=13)
        self.assertEqual(metrics.pairs, 56)
        self.assertLessEqual(metrics.max_abs_logit_delta, 1e-7)

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


if __name__ == "__main__":
    unittest.main()
