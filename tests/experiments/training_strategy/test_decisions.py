import copy
import unittest

from fsrl.experiments.training_strategy.decisions import (
    behavior_preservation,
    competence,
    cost_comparison,
    criterion,
    mechanism,
    noninferiority,
    outcome,
    study_outcome,
)
from fsrl.experiments.training_strategy.protocol import load_specification


def summary(value):
    return {
        "mean": value,
        "subjects": 77,
        "bootstrap": {"lower": value, "upper": value},
    }


class RegisteredDecisionTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_specification()
        self.conditions = {
            name: {
                key: {"passed": True} for key in ("competence", "mechanism", "behavior")
            }
            for name in ("matched_staged", "joint")
        }

    def test_boundary_inclusion_strictness_and_missing_endpoints(self):
        for op, expected in ((">=", True), (">", False), ("<=", True)):
            with self.subTest(operator=op):
                self.assertEqual(
                    criterion(summary(0.1), 0.1, statistic="lower", operator=op)[
                        "passed"
                    ],
                    expected,
                )
        for invalid in (None, float("nan"), float("inf")):
            self.assertFalse(
                criterion(summary(invalid), 0.0, statistic="mean", operator=">=")[
                    "passed"
                ]
            )

    def test_all_six_noninferiority_endpoints_required(self):
        contract = self.spec["decision_contract"]["paired_noninferiority"]
        paired = {
            domain: {group: summary(-0.02) for group in contract[f"{domain}_groups"]}
            for domain in ("generic", "liu")
        }
        self.assertTrue(noninferiority(paired, self.spec)["passed"])
        self.assertEqual(len(noninferiority(paired, self.spec)["checks"]), 6)
        paired["liu"]["omitted"] = summary(-0.02001)
        self.assertFalse(noninferiority(paired, self.spec)["passed"])

    def test_competence_uses_combined_exact_decisions_not_probabilities(self):
        contract = self.spec["decision_contract"]["per_condition_competence"]
        data = {
            domain: {
                "intact": {
                    "exact_decision": {
                        key: summary(value)
                        for key, value in contract[
                            f"{domain}_intact_exact_decision_mean_minimum"
                        ].items()
                    }
                }
            }
            for domain in ("generic", "liu")
        }
        data["constructive"] = {"intact_transitive_triplet_fraction": summary(0.95)}
        self.assertTrue(competence(data, self.spec)["passed"])
        data["liu"]["intact"]["exact_decision"]["nonlearned"] = summary(0.69)
        self.assertFalse(competence(data, self.spec)["passed"])

    def test_each_causal_link_and_local_only_boundary_is_required(self):
        values = {
            "intact_minus_P_off_nonlearned": 0.1,
            "P_off_nonlearned": 0.55,
            "global_remote_absolute": 0.011,
            "global_third_party_relational": 0.051,
            "intact_minus_local_off_retained": 0.01,
            "intact_minus_local_off_omitted": 0.01,
            "intact_minus_query_shuffle_learned": 0.01,
            "intact_minus_evidence_shuffle_learned": 0.01,
            "P_off_learned": 0.501,
            "local_remote_minus_quarter_combined": 0.0,
        }
        effects = {key: summary(value) for key, value in values.items()}
        self.assertTrue(mechanism(effects)["passed"])
        for key in values:
            altered = {**effects, key: summary(None)}
            with self.subTest(endpoint=key):
                self.assertFalse(mechanism(altered)["passed"])

    def test_named_six_behavior_rows_not_any_six(self):
        names = self.spec["decision_contract"]["behavior"][
            "historically_quantitative_rows"
        ]
        extra = [
            "symbolic_distance_effect",
            "serial_position_effect",
            "self_consistent_vs_inconsistent_errors",
        ]
        flags = {
            key: {"qualitative": True, "calibration": key in names}
            for key in names + extra
        }
        self.assertTrue(behavior_preservation({"flags": flags}, self.spec)["passed"])
        flags[names[0]]["calibration"] = False
        flags[extra[0]]["calibration"] = True
        self.assertEqual(sum(row["calibration"] for row in flags.values()), 6)
        self.assertFalse(behavior_preservation({"flags": flags}, self.spec)["passed"])
        del flags[extra[-1]]
        with self.assertRaises(ValueError):
            behavior_preservation({"flags": flags}, self.spec)

    def test_outcome_tree_preserves_distinct_failure_levels(self):
        self.assertEqual(
            outcome(self.conditions, {"passed": True}),
            "single_stage_preserves_mechanism_and_behavior",
        )
        cases = (
            ("joint", "competence", "joint_recipe_insufficient"),
            ("matched_staged", "competence", "matched_comparator_insufficient"),
            ("joint", "mechanism", "alternative_computational_solution"),
            ("matched_staged", "mechanism", "alternative_computational_solution"),
            ("joint", "behavior", "mechanism_preserved_behavior_incomplete"),
        )
        for condition, layer, expected in cases:
            altered = copy.deepcopy(self.conditions)
            altered[condition][layer]["passed"] = False
            with self.subTest(condition=condition, layer=layer):
                self.assertEqual(outcome(altered, {"passed": True}), expected)
        self.assertEqual(
            outcome(self.conditions, {"passed": False}), "competent_but_not_noninferior"
        )

    def test_no_network_omission_or_majority_vote(self):
        results = {
            str(seed): {"outcome": "single_stage_preserves_mechanism_and_behavior"}
            for seed in (2108, 2109, 2110)
        }
        results["2110"]["outcome"] = "mechanism_preserved_behavior_incomplete"
        self.assertEqual(
            study_outcome(results, self.spec), "mechanism_preserved_behavior_incomplete"
        )
        del results["2110"]
        with self.assertRaises(ValueError):
            study_outcome(results, self.spec)

    def test_fewer_stages_does_not_establish_compute_advantage(self):
        staged = {"warm_training_seconds": 100.0, "peak_allocated_bytes": 1000}
        self.assertTrue(cost_comparison(staged, staged)["efficiency_advantage"])
        joint = {"warm_training_seconds": 99.0, "peak_allocated_bytes": 1001}
        self.assertFalse(cost_comparison(staged, joint)["efficiency_advantage"])
