import unittest

import numpy as np

from fsrl.local_behavior_attribution import (
    _ratio_summary,
    decision_summary,
    exact_probability,
    slope_decomposition,
)


class LocalBehaviorAttributionTests(unittest.TestCase):
    def test_exact_probability_uses_correct_signed_margin_and_temperature(self):
        margins = np.asarray([-1.0, 0.0, 1.0])
        observed = exact_probability(margins, 0.25)
        expected = 1.0 / (1.0 + np.exp(-margins / 0.25))
        self.assertTrue(np.allclose(observed, expected))
        self.assertAlmostEqual(observed[1], 0.5)

    def test_ratio_summary_bootstraps_subject_totals(self):
        numerator = np.asarray([1.0, 3.0])
        denominator = np.asarray([2.0, 6.0])
        counts = np.asarray([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        result = _ratio_summary(numerator, denominator, counts, 0.95)
        self.assertAlmostEqual(result["point"], 0.5)
        self.assertAlmostEqual(result["bootstrap"]["mean"], 0.5)

    def test_slope_groups_sum_to_total(self):
        class Protocol:
            n_items = 4
            true_order_high_to_low = (0, 1, 2, 3)
            support_pairs_higher_lower = ((0, 1), (2, 3))

        class Config:
            bs = 2

        class Evaluator:
            protocol = Protocol()
            config = Config()

        probabilities = np.asarray(
            [
                [0.6, 0.7, 0.8, 0.65, 0.75, 0.9],
                [0.55, 0.72, 0.82, 0.62, 0.79, 0.88],
            ]
        )
        retained = np.asarray([[True, False], [False, True]])
        counts = np.eye(2)
        result = slope_decomposition(
            Evaluator(),
            {
                "original_v1_local_off": probabilities,
                "dual_intact": probabilities + 0.01,
            },
            retained,
            counts,
            0.95,
        )
        for row in result["conditions"].values():
            self.assertLessEqual(row["additive_identity_max_abs_error"], 1e-12)

    def test_decision_prefers_dual_evidence_access(self):
        error_mass = {
            "omitted_error_mass_fraction": {
                "point": 0.7,
                "bootstrap": {"lower": 0.6},
            }
        }
        probability = {
            "delta_probability": {
                "retained": {"summary": {"bootstrap": {"lower": 0.01}}}
            }
        }
        self_cross = {
            "retained_signed_self": {"bootstrap": {"lower": 0.1}},
            "retained_absolute_cross_to_self_ratio": 0.2,
        }
        local_only = {
            "P_off_local_intact": {
                "retained_exact_probability": {"bootstrap": {"lower": 0.7}},
                "omitted_exact_probability": {"bootstrap": {"upper": 0.52}},
                "retained_minus_omitted_exact_probability": {
                    "bootstrap": {"lower": 0.15}
                },
            }
        }
        slope = {"largest_positive_original_contributor": "nonlearned"}
        sampled = {
            "frozen_sampled_learned_accuracy_contrast": {
                "bootstrap": {"lower": -0.01, "upper": 0.01}
            }
        }
        specification = {
            "decision_rules": {},
            "outcome_tree": {
                "dual_evidence_access": "dual",
                "shared_value_transform": "value",
                "confirmation_estimand_sensitivity": "endpoint",
                "mixed_or_unresolved": "mixed",
            },
        }
        result = decision_summary(
            specification,
            error_mass,
            probability,
            self_cross,
            local_only,
            slope,
            sampled,
        )
        self.assertEqual(result["outcome"], "dual_evidence_access")
        self.assertTrue(result["flags"]["omission_dominant"])
        self.assertTrue(result["flags"]["retained_local_sufficient"])


if __name__ == "__main__":
    unittest.main()
