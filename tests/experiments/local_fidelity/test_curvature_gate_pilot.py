import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.experiments.local_fidelity.curvature_gate import CurvatureGateTransition
from fsrl.experiments.local_fidelity.curvature_gate_pilot import (
    _ordered_pairs,
    margin_fields,
    query_binding_summary,
    query_bundle,
)
from fsrl.tasks.registered_protocol import load_ranking_protocol


class CurvatureGatePilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.protocol = load_ranking_protocol()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=19,
            support_seed=23,
        )
        self.gate = CurvatureGateTransition(self.net)
        self.fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=self.net.w.device
        )
        pairs = _ordered_pairs(self.protocol.n_items)
        self.schedules = tuple(pairs for _ in range(self.config.bs))

    def test_original_gate_bundle_matches_frozen_evaluator(self):
        bundle = query_bundle(
            self.evaluator,
            self.gate,
            self.fast_weights,
            self.schedules,
            condition="original_v1",
            gamma_global=0.7,
            shuffle_seed=31,
        )
        expected = self.evaluator.readout_logits(self.fast_weights, self.schedules)
        for subject, schedule in enumerate(self.schedules):
            for index, pair in enumerate(schedule):
                self.assertAlmostEqual(
                    bundle["logits"][subject, index], expected[subject][pair], places=6
                )
        self.assertTrue(
            np.array_equal(
                bundle["applied_gammas"], np.ones_like(bundle["applied_gammas"])
            )
        )

    def test_shuffled_gate_preserves_each_subject_gamma_multiset(self):
        conditioned = query_bundle(
            self.evaluator,
            self.gate,
            self.fast_weights,
            self.schedules,
            condition="conditioned_gate",
            gamma_global=0.7,
            shuffle_seed=37,
        )
        shuffled = query_bundle(
            self.evaluator,
            self.gate,
            self.fast_weights,
            self.schedules,
            condition="shuffled_gate",
            gamma_global=0.7,
            shuffle_seed=37,
        )
        self.assertTrue(
            np.allclose(
                np.sort(conditioned["conditioned_gammas"], axis=1),
                np.sort(shuffled["applied_gammas"], axis=1),
            )
        )

    def test_margin_fields_antisymmetrize_adjacent_orientations(self):
        logits = np.arange(self.config.bs * 56, dtype=np.float64).reshape(
            self.config.bs, 56
        )
        observed = margin_fields({"logits": logits}, self.protocol.n_items)
        expected = 0.5 * (logits[:, 0::2] - logits[:, 1::2])
        self.assertTrue(np.array_equal(observed, expected))

    def test_query_binding_reduces_mismatch_queries_within_subject(self):
        intact = torch.randn(
            self.config.bs,
            self.config.hs,
            self.config.hs,
            device=self.net.w.device,
        )
        loo = intact[None].repeat(8, 1, 1, 1) - 0.01 * torch.randn(
            8,
            self.config.bs,
            self.config.hs,
            self.config.hs,
            device=self.net.w.device,
        )
        retained = np.ones((8, self.config.bs), dtype=bool)
        counts = np.eye(self.config.bs, dtype=np.float64)
        result = query_binding_summary(
            self.evaluator, intact, loo, retained, counts, 0.95
        )
        self.assertEqual(
            len(result["raw_subject_level"]["matched_minus_shared_endpoint"]),
            self.config.bs,
        )
        self.assertEqual(result["conditioned_minus_original_max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
