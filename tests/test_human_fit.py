import unittest

from fsrl.human_fit import select_global_temperature

SECONDARY = [
    "learned_accuracy",
    "nonlearned_accuracy",
    "symbolic_distance_slope",
]


def result(temperature, accuracy, *, checkpoint="abc", secondary=0.0):
    return {
        "protocol_id": "test-protocol",
        "checkpoint": {"sha256": checkpoint},
        "cue_seed": 1,
        "support_seed": 2,
        "subject_encoding_seed": 3,
        "subject_encoding_mode": "stable_omission",
        "sampling": {
            "seed": 4,
            "temperature": temperature,
            "test_blocks": 10,
            "trials_per_subject": 280,
        },
        "summary": {
            "overall_accuracy": accuracy,
            "learned_accuracy": secondary,
            "nonlearned_accuracy": secondary,
            "symbolic_distance_slope": secondary,
        },
    }


class HumanFitTests(unittest.TestCase):
    def setUp(self):
        self.specification = {
            "fit_id": "test-fit",
            "registration_status": "developmental-test",
            "protocol_id": "test-protocol",
            "temperature_grid": [1.0, 0.5],
            "selection_metric": "absolute_overall_accuracy_error",
            "target_overall_accuracy": 0.87,
            "secondary_metrics_are_out_of_sample": SECONDARY,
        }

    def test_selection_uses_overall_accuracy_only(self):
        report = select_global_temperature(
            [
                result(1.0, 0.80, secondary=0.99),
                result(0.5, 0.86, secondary=0.01),
            ],
            self.specification,
        )
        self.assertEqual(report["selected_temperature"], 0.5)
        self.assertEqual(report["registration_status"], "developmental-test")
        self.assertEqual(report["status"], "descriptive_fit_only")

    def test_requires_the_registered_grid(self):
        with self.assertRaisesRegex(ValueError, "registered grid"):
            select_global_temperature([result(1.0, 0.80)], self.specification)

    def test_rejects_mixed_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "same cohort"):
            select_global_temperature(
                [result(1.0, 0.80), result(0.5, 0.86, checkpoint="different")],
                self.specification,
            )


if __name__ == "__main__":
    unittest.main()
