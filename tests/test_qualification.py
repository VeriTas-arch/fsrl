import copy
import unittest

from fsrl.qualification import (
    DEFAULT_QUALIFICATION_PATH,
    evaluate_qualification,
    load_json,
)


def passing_result():
    intact = {
        "overall_accuracy": 0.8,
        "nonlearned_accuracy": 0.75,
        "mean_transitive_triplet_fraction": 0.98,
    }
    ablated = {
        "overall_accuracy": 0.51,
        "nonlearned_accuracy": 0.5,
        "mean_transitive_triplet_fraction": 0.78,
        "mean_pair_decision_agreement_to_intact": 0.5,
    }
    return {
        "cue_mode": "permuted_shared",
        "subject_encoding": {"mode": "stable_omission"},
        "training_provenance": {
            "present": True,
            "checkpoint_sha_matches": True,
            "task_distribution": {"liu_graph_held_out": True},
        },
        "conditions": {
            "intact": intact,
            "write_off": dict(ablated),
            "alpha_zero": dict(ablated),
            "reset": dict(ablated),
            "shuffle": dict(ablated),
        },
        "order_invariance": {"max_abs_logit_delta": 1e-8},
    }


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.specification = load_json(DEFAULT_QUALIFICATION_PATH)

    def test_all_registered_checks_can_pass(self):
        report = evaluate_qualification(passing_result(), self.specification)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["registration_status"],
            "developmental_after_v1_ablation_diagnostics",
        )
        self.assertTrue(all(check["passed"] for check in report["checks"]))

    def test_missing_plasticity_effect_is_a_no_go(self):
        result = passing_result()
        result["conditions"]["reset"]["nonlearned_accuracy"] = 0.74
        report = evaluate_qualification(result, self.specification)
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("reset.nonlearned_accuracy", failed)

    def test_position_dependence_is_a_no_go(self):
        result = copy.deepcopy(passing_result())
        result["order_invariance"]["max_abs_logit_delta"] = 0.01
        report = evaluate_qualification(result, self.specification)
        self.assertFalse(report["passed"])

    def test_unregistered_training_distribution_is_a_no_go(self):
        result = passing_result()
        result["training_provenance"] = {"present": False}
        report = evaluate_qualification(result, self.specification)
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("training_provenance.liu_graph_held_out", failed)

    def test_ablated_policy_that_preserves_intact_ranking_is_a_no_go(self):
        result = passing_result()
        result["conditions"]["reset"]["mean_pair_decision_agreement_to_intact"] = 0.9
        report = evaluate_qualification(result, self.specification)
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("reset.mean_pair_decision_agreement_to_intact", failed)


if __name__ == "__main__":
    unittest.main()
