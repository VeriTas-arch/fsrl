import unittest

from fsrl.experiments.local_fidelity.trace_replication import (
    backbone_training_config,
    cross_seed_decision,
    local_adaptation_config,
    seed_specification,
    within_seed_decision,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record


class ConjunctiveLocalTraceReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_json(
            resolve_record("benchmarks/conjunctive_local_trace_replication_v2_3.json")
        )

    def test_seed_configs_are_frozen_and_independent(self):
        self.assertEqual(backbone_training_config(self.specification, 2102).seed, 2102)
        self.assertEqual(backbone_training_config(self.specification, 2103).seed, 2103)
        self.assertEqual(local_adaptation_config(self.specification, 2102).seed, 12202)
        self.assertEqual(local_adaptation_config(self.specification, 2103).seed, 12203)
        self.assertEqual(
            seed_specification(self.specification, 2102)["liu_evaluation"][
                "bootstrap_seed"
            ],
            32202,
        )

    @staticmethod
    def _summary(mean=0.2, lower=0.1, upper=0.3):
        return {"mean": mean, "bootstrap": {"lower": lower, "upper": upper}}

    def _pilot(self):
        raw = [0.2, 0.3]
        local = {
            condition: {
                "raw_subject_level": {
                    "retained_relation_mean_direct_correctness": raw,
                    "remote_absolute": [0.04, 0.04],
                }
            }
            for condition in (
                "original_v1_local_off",
                "dual_intact",
                "local_query_key_shuffle",
                "global_P_off_local_intact",
            )
        }
        local["dual_intact"]["raw_subject_level"][
            "retained_relation_mean_direct_correctness"
        ] = [0.5, 0.6]
        local["local_query_key_shuffle"]["raw_subject_level"][
            "retained_relation_mean_direct_correctness"
        ] = [0.25, 0.35]
        local["original_v1_local_off"]["raw_subject_level"]["remote_absolute"] = [
            1.0,
            1.0,
        ]
        behavior = {
            "subjects": [
                {"learned_accuracy": 0.8, "nonlearned_accuracy": 0.4},
                {"learned_accuracy": 0.9, "nonlearned_accuracy": 0.5},
            ]
        }
        return {
            "local_fidelity": local,
            "behavior": {"global_P_off_local_intact": behavior},
            "integrity": {
                "local_off_v1_logit_max_abs_error": 0.0,
                "local_margin_identity_max_abs_error": 0.0,
                "stable_omitted_max_abs_pair_influence": 0.0,
            },
            "original_v1_qualification": {"passed": True},
            "decision": {"flags": {"global_branch_preservation": True}},
        }

    def _attribution(self):
        return {
            "boundary_and_probability": {
                "delta_probability": {"retained": {"summary": self._summary()}}
            },
            "self_cross": {
                "retained_signed_self": self._summary(),
                "retained_absolute_cross_to_self_ratio": 0.2,
            },
            "local_only": {
                "P_off_local_intact": {
                    "retained_minus_omitted_exact_probability": self._summary(),
                    "retained_exact_probability": self._summary(mean=0.67),
                }
            },
            "integrity": {
                "dual_margin_identity_max_abs_error": 0.0,
                "self_plus_cross_identity_max_abs_error": 0.0,
                "stable_omitted_self_max_abs": 0.0,
                "slope_additive_identity_max_abs_error": 0.0,
            },
        }

    def test_within_seed_decision_keeps_four_links_independent(self):
        specification = {
            **self.specification,
            "liu_evaluation": {
                **self.specification["liu_evaluation"],
                "subjects": 2,
                "bootstrap_samples": 2,
            },
        }
        decision = within_seed_decision(
            specification, self._pilot(), self._attribution(), 2102
        )
        self.assertTrue(decision["all_four_primary_links_pass"])
        failed = self._attribution()
        failed["boundary_and_probability"]["delta_probability"]["retained"]["summary"][
            "bootstrap"
        ]["lower"] = -0.1
        mixed = within_seed_decision(specification, self._pilot(), failed, 2102)
        self.assertFalse(mixed["flags"]["retained_exact_probability_rescue"])
        self.assertTrue(mixed["flags"]["causal_direct_rescue"])

    def test_cross_seed_decision_does_not_pool_a_heterogeneous_link(self):
        effect = self._summary()
        effects = {
            name: effect
            for name in (
                "retained_dual_minus_v1_exact_probability",
                "dual_minus_v1_direct_correctness",
                "dual_minus_query_shuffle_direct_correctness",
                "retained_correct_signed_self_contribution",
                "P_off_retained_minus_omitted_exact_probability",
                "P_off_learned_minus_nonlearned_sampled_accuracy",
                "P_off_remote_minus_quarter_v1_remote",
            )
        }
        flags = {name: True for name in self.specification["primary_links"]}
        seed_results = {
            "2102": {
                "decision": {
                    "interpretable": True,
                    "flags": flags,
                    "primary_effects": effects,
                }
            },
            "2103": {
                "decision": {
                    "interpretable": True,
                    "flags": {**flags, "causal_direct_rescue": False},
                    "primary_effects": effects,
                }
            },
        }
        decision = cross_seed_decision(self.specification, seed_results)
        self.assertEqual(
            decision["links"]["causal_direct_rescue"]["status"],
            "heterogeneous_or_unresolved",
        )
        self.assertEqual(decision["outcome"], "heterogeneous_or_unresolved")
        self.assertEqual(decision["network_population_inference"], "not_performed")
