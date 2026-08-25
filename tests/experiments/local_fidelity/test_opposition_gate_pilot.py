import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.experiments.local_fidelity.opposition_gate import (
    PolicyOppositionGateTransition,
)
from fsrl.experiments.local_fidelity.opposition_gate_pilot import (
    decision_summary,
    query_bundle,
)
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.registered_protocol import load_ranking_protocol


class PolicyOppositionGatePilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.protocol = load_ranking_protocol()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=47,
            support_seed=53,
        )
        self.gate = PolicyOppositionGateTransition(self.net)
        self.fast_weights = torch.randn(
            self.config.bs, self.config.hs, self.config.hs, device=self.net.w.device
        )
        pairs = ordered_pairs(self.protocol.n_items)
        self.schedules = tuple(pairs for _ in range(self.config.bs))

    def _bundle(self, condition, shuffle_seed=59):
        return query_bundle(
            self.evaluator,
            self.gate,
            self.fast_weights,
            self.schedules,
            condition=condition,
            gamma_global=0.7,
            shuffle_seed=shuffle_seed,
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
                bundle["applied_gammas"], np.ones_like(bundle["applied_gammas"])
            )
        )

    def test_sign_reversed_uses_support_risk_with_same_beta(self):
        opposition = self._bundle("opposition_gate")
        support = self._bundle("sign_reversed_support_gate")
        self.assertTrue(
            np.array_equal(
                opposition["first_order_values"], support["first_order_values"]
            )
        )
        self.assertTrue(
            np.array_equal(opposition["quadratic_values"], support["quadratic_values"])
        )
        expected = 1.0 / (
            1.0 + float(self.gate.beta.detach()) * support["support_risks"]
        )
        self.assertTrue(np.allclose(support["conditioned_gammas"], expected))
        product = opposition["first_order_values"] * opposition["quadratic_values"]
        self.assertTrue(
            np.array_equal(
                opposition["opposition_risks"][product > 0.0],
                np.zeros(np.sum(product > 0.0)),
            )
        )
        self.assertTrue(
            np.array_equal(
                support["support_risks"][product < 0.0],
                np.zeros(np.sum(product < 0.0)),
            )
        )

    def test_shuffled_opposition_preserves_each_subject_gamma_multiset(self):
        opposition = self._bundle("opposition_gate", shuffle_seed=61)
        shuffled = self._bundle("shuffled_opposition_gate", shuffle_seed=61)
        self.assertTrue(
            np.allclose(
                np.sort(opposition["conditioned_gammas"], axis=1),
                np.sort(shuffled["applied_gammas"], axis=1),
            )
        )

    def test_decision_requires_superiority_to_every_control(self):
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
            "opposition_gate": row(0.30, 0.20, 0.20),
            "matched_global_scalar": row(0.15, 0.0, 0.19),
            "shuffled_opposition_gate": row(0.14, 0.0, 0.19),
            "sign_reversed_support_gate": row(0.13, 0.0, 0.19),
        }
        qualification = {
            "passed": True,
            "causal_result": {"conditions": {"intact": {"nonlearned_accuracy": 0.8}}},
        }
        opposition_qualification = {"passed": True}
        causal = {"conditions": {"intact": {"nonlearned_accuracy": 0.8}}}
        binding = {
            "opposition_minus_original_max_abs": 0.0,
            "opposition_gate": {
                "matched_minus_shared_endpoint": {"mean": 0.1},
                "matched_minus_disjoint": {"mean": 0.1},
            },
        }
        terminal = {"summary": {"opposition_gate": {"bootstrap": {"lower": 0.1}}}}
        specification = {"primary_decision_rules": {}}
        passing = decision_summary(
            specification,
            qualification,
            opposition_qualification,
            causal,
            local,
            binding,
            terminal,
        )
        self.assertTrue(passing["all_primary_rules_pass"])
        local["sign_reversed_support_gate"] = row(0.31, 0.0, 0.19)
        failing = decision_summary(
            specification,
            qualification,
            opposition_qualification,
            causal,
            local,
            binding,
            terminal,
        )
        self.assertFalse(failing["flags"]["control_specificity"])
        self.assertEqual(failing["outcome"], "valid_local_or_specificity_failure")


if __name__ == "__main__":
    unittest.main()
