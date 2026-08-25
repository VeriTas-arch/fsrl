import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.experiments.local_fidelity.policy_residual import PolicyResidualTransition
from fsrl.experiments.local_fidelity.policy_residual_pilot import (
    balanced_magnitude_signs,
    decision_summary,
    query_bundle,
)
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.registered_protocol import load_ranking_protocol


class PolicyResidualPilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.protocol = load_ranking_protocol()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=73,
            support_seed=79,
        )
        self.residual = PolicyResidualTransition(self.net)
        self.fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=self.net.w.device
        )
        pairs = ordered_pairs(self.protocol.n_items)
        self.schedules = tuple(pairs for _ in range(self.config.bs))

    def _bundle(self, condition, shuffle_seed=83, null_seed=89):
        return query_bundle(
            self.evaluator,
            self.residual,
            self.fast_weights,
            self.schedules,
            condition=condition,
            shuffle_seed=shuffle_seed,
            magnitude_null_seed=null_seed,
        )

    def test_original_bundle_matches_frozen_backbone(self):
        bundle = self._bundle("original_v1")
        expected = self.evaluator.readout_logits(self.fast_weights, self.schedules)
        for subject, schedule in enumerate(self.schedules):
            for index, pair in enumerate(schedule):
                self.assertAlmostEqual(
                    bundle["logits"][subject, index],
                    expected[subject][pair],
                    places=6,
                )
        self.assertTrue(
            np.array_equal(
                bundle["applied_corrections"],
                np.zeros_like(bundle["applied_corrections"]),
            )
        )

    def test_matched_magnitude_null_is_balanced_and_preserves_magnitude(self):
        null = self._bundle("matched_magnitude_null", null_seed=97)
        self.assertTrue(
            np.allclose(
                np.abs(null["applied_residual_bases"]),
                np.abs(null["policy_residuals"]),
            )
        )
        signs = balanced_magnitude_signs(self.config.bs, 56, 97)
        self.assertTrue(np.all(np.sum(signs > 0.0, axis=1) == 28))
        self.assertTrue(np.all(np.sum(signs < 0.0, axis=1) == 28))
        self.assertTrue(np.array_equal(signs, balanced_magnitude_signs(3, 56, 97)))

    def test_shuffled_residual_preserves_each_subject_multiset(self):
        natural = self._bundle("policy_residual", shuffle_seed=101)
        shuffled = self._bundle("shuffled_residual", shuffle_seed=101)
        self.assertTrue(
            np.allclose(
                np.sort(natural["policy_residuals"], axis=1),
                np.sort(shuffled["applied_residual_bases"], axis=1),
            )
        )

    def test_decision_requires_both_control_superiority_rules(self):
        counts = np.eye(3, dtype=np.float64)

        def row(local, h_value, other):
            values = lambda value: np.full(3, value, dtype=np.float64)
            return {
                "subject_level": {
                    "retained_relation_mean_direct_correctness": values(local),
                    "H_greater_A_direct_correctness": values(h_value),
                    "other_relation_mean_direct_correctness": values(other),
                },
                "summary": {
                    "H_greater_A_direct_correctness": {"bootstrap": {"upper": h_value}},
                    "remote_absolute": {
                        "mean": 0.2,
                        "bootstrap": {"lower": 0.1},
                    },
                    "gauge_invariant_R_third_rel": {
                        "mean": 0.2,
                        "bootstrap": {"lower": 0.1},
                    },
                },
            }

        local = {
            "counts": counts,
            "interval": 0.95,
            "original_v1": row(0.10, -0.10, 0.195),
            "policy_residual": row(0.30, 0.20, 0.20),
            "matched_magnitude_null": row(0.15, 0.0, 0.19),
            "shuffled_residual": row(0.14, 0.0, 0.19),
        }
        qualification = {
            "passed": True,
            "causal_result": {"conditions": {"intact": {"nonlearned_accuracy": 0.8}}},
        }
        residual_qualification = {"passed": True}
        causal = {"conditions": {"intact": {"nonlearned_accuracy": 0.8}}}
        binding = {
            "policy_residual_minus_original_max_abs": 0.0,
            "policy_residual": {
                "matched_minus_shared_endpoint": {"mean": 0.1},
                "matched_minus_disjoint": {"mean": 0.1},
            },
        }
        terminal = {"summary": {"policy_residual": {"bootstrap": {"lower": 0.1}}}}
        specification = {"primary_decision_rules": {}}
        passing = decision_summary(
            specification,
            qualification,
            residual_qualification,
            causal,
            local,
            binding,
            terminal,
        )
        self.assertTrue(passing["all_primary_rules_pass"])
        local["shuffled_residual"] = row(0.31, 0.0, 0.19)
        failing = decision_summary(
            specification,
            qualification,
            residual_qualification,
            causal,
            local,
            binding,
            terminal,
        )
        self.assertFalse(failing["flags"]["control_specificity"])
        self.assertEqual(failing["outcome"], "valid_local_or_specificity_failure")
