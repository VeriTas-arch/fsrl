import unittest

from fsrl.experiments.pl_direct_training.evaluation import seed_outcome
from fsrl.experiments.pl_direct_training.reporting import cohort_outcome


def condition(*, competence=True, mechanism=True, behavior=True):
    return {
        "decisions": {
            "competence": {"passed": competence},
            "mechanism": {"passed": mechanism},
            "behavior": {"passed": behavior},
        }
    }


class DirectDecisionTests(unittest.TestCase):
    def test_outcome_order_distinguishes_control_candidate_and_behavior(self):
        rows = {
            "time_retained_control": condition(),
            "no_time_candidate": condition(),
        }
        self.assertEqual(seed_outcome(rows, {"passed": True}), "development_admitted")
        rows["time_retained_control"] = condition(mechanism=False)
        self.assertEqual(
            seed_outcome(rows, {"passed": True}),
            "training_parameterization_failure",
        )
        rows["time_retained_control"] = condition()
        rows["no_time_candidate"] = condition(competence=False)
        self.assertEqual(seed_outcome(rows, {"passed": True}), "no_time_recipe_failure")
        rows["no_time_candidate"] = condition()
        self.assertEqual(
            seed_outcome(rows, {"passed": False}),
            "competent_but_time_noninferior_failure",
        )
        rows["no_time_candidate"] = condition(mechanism=False)
        self.assertEqual(
            seed_outcome(rows, {"passed": True}),
            "alternative_no_time_organization",
        )
        rows["no_time_candidate"] = condition(behavior=False)
        self.assertEqual(seed_outcome(rows, {"passed": True}), "behavior_incomplete")

    def test_all_seed_rule_does_not_use_majority_vote(self):
        rows = {
            "3001": {"admitted": True, "outcome": "development_admitted"},
            "3002": {"admitted": True, "outcome": "development_admitted"},
            "3003": {"admitted": False, "outcome": "behavior_incomplete"},
        }
        self.assertEqual(cohort_outcome("development", rows), "behavior_incomplete")


if __name__ == "__main__":
    unittest.main()
