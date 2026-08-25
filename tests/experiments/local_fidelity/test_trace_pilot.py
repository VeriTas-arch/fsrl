import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.experiments.local_fidelity.trace_pilot import (
    build_local_trace,
    canonical_derangements,
    decision_summary,
    query_bundle,
    shuffled_pair_indices,
)
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.registered_protocol import load_ranking_protocol


class ConjunctiveLocalTracePilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107)
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.protocol = load_ranking_protocol()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=109,
            support_seed=113,
            subject_encoding_mode="stable_omission",
            subject_encoding_seed=127,
        )
        self.local = ConjunctiveLocalTrace(self.config.cs, initial_gain=0.2)
        self.fast_weights = self.evaluator.learn_fast_weights(
            FastWeightIntervention.INTACT
        )
        self.local_state = build_local_trace(self.evaluator, self.local)
        pairs = ordered_pairs(self.protocol.n_items)
        self.schedules = tuple(pairs for _ in range(self.config.bs))

    def _bundle(self, condition):
        return query_bundle(
            self.evaluator,
            self.local,
            self.fast_weights,
            self.local_state,
            self.schedules,
            condition=condition,
            shuffle_seed=131,
        )

    def test_local_off_matches_frozen_backbone(self):
        observed = self._bundle("original_v1_local_off")
        expected = self.evaluator.readout_logits(self.fast_weights, self.schedules)
        for subject, schedule in enumerate(self.schedules):
            for index, pair in enumerate(schedule):
                self.assertAlmostEqual(
                    observed["logits"][subject, index],
                    expected[subject][pair],
                    places=6,
                )
        self.assertTrue(
            np.array_equal(
                observed["applied_local_margins"],
                np.zeros_like(observed["applied_local_margins"]),
            )
        )

    def test_dual_margin_is_exactly_global_plus_local(self):
        observed = self._bundle("dual_intact")
        self.assertTrue(
            np.allclose(
                observed["logits"],
                observed["global_logits"] + observed["applied_local_margins"],
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_shuffle_is_a_canonical_derangement_with_orientation(self):
        deranged = canonical_derangements(3, self.protocol.n_items, 137)
        identity = np.arange(deranged.shape[1])
        self.assertTrue(np.all(deranged != identity[None]))
        self.assertTrue(
            np.array_equal(
                deranged,
                canonical_derangements(3, self.protocol.n_items, 137),
            )
        )
        mapped = shuffled_pair_indices(3, self.protocol.n_items, 137)
        self.assertTrue(np.all(mapped[:, 0::2] // 2 == mapped[:, 1::2] // 2))
        self.assertTrue(np.all(mapped[:, 0::2] % 2 == 0))
        self.assertTrue(np.all(mapped[:, 1::2] % 2 == 1))

    def test_stable_omitted_relation_has_zero_local_effect(self):
        relation = self.protocol.support_pairs_higher_lower[0]
        omitted = build_local_trace(
            self.evaluator, self.local, zero_relations=frozenset((relation,))
        )
        retained = np.asarray(
            [
                self.evaluator.subject_relation_gains[subject][relation] > 0.0
                for subject in range(self.config.bs)
            ]
        )
        natural = self.local_state.detach().cpu().numpy()
        omitted_values = omitted.detach().cpu().numpy()
        self.assertTrue(np.allclose(natural[~retained], omitted_values[~retained]))

    def test_decision_requires_double_dissociation(self):
        counts = np.eye(3, dtype=np.float64)

        def field(local, h_value, remote=0.2, third=0.2):
            values = lambda value: np.full(3, value, dtype=np.float64)
            return {
                "subject_level": {
                    "retained_relation_mean_direct_correctness": values(local),
                    "H_greater_A_direct_correctness": values(h_value),
                    "remote_absolute": values(remote),
                    "gauge_invariant_R_third_rel": values(third),
                },
                "summary": {
                    "H_greater_A_direct_correctness": {"bootstrap": {"upper": h_value}},
                    "remote_absolute": {
                        "mean": remote,
                        "bootstrap": {"lower": max(0.01, remote / 2)},
                    },
                    "gauge_invariant_R_third_rel": {
                        "mean": third,
                        "bootstrap": {"lower": max(0.01, third / 2)},
                    },
                },
            }

        local = {
            "counts": counts,
            "interval": 0.95,
            "original_v1_local_off": field(0.1, -0.1, remote=0.4),
            "dual_intact": field(0.3, 0.2, remote=0.4),
            "local_query_key_shuffle": field(0.15, 0.0, remote=0.4),
            "global_P_off_local_intact": field(0.25, 0.2, remote=0.05),
        }
        specificity = {
            "summary": {"direct_minus_three_remote": {"bootstrap": {"lower": 0.1}}}
        }

        def behavior(learned, nonlearned):
            return {
                "subjects": [
                    {
                        "overall_accuracy": (learned + nonlearned) / 2,
                        "learned_accuracy": learned,
                        "nonlearned_accuracy": nonlearned,
                    }
                    for _ in range(3)
                ]
            }

        behaviors = {
            "original_v1_local_off": behavior(0.7, 0.8),
            "dual_intact": behavior(0.9, 0.8),
            "local_query_key_shuffle": behavior(0.7, 0.8),
            "global_P_off_local_intact": behavior(0.8, 0.5),
        }
        qualification = {"passed": True}
        binding = {
            "conditioned_minus_original_max_abs": 0.0,
            "original_v1": {
                "matched_minus_shared_endpoint": {"mean": 0.1},
                "matched_minus_disjoint": {"mean": 0.1},
            },
        }
        terminal = {"summary": {"original_v1": {"bootstrap": {"lower": 0.1}}}}
        result = decision_summary(
            {"primary_decision_rules": {}},
            qualification,
            local,
            specificity,
            behaviors,
            binding,
            terminal,
        )
        self.assertTrue(result["all_primary_rules_pass"])
        specificity["summary"]["direct_minus_three_remote"]["bootstrap"]["lower"] = -0.1
        failed = decision_summary(
            {"primary_decision_rules": {}},
            qualification,
            local,
            specificity,
            behaviors,
            binding,
            terminal,
        )
        self.assertFalse(failed["flags"]["local_direct_specificity"])
        self.assertEqual(failed["outcome"], "local_rescue_without_double_dissociation")
