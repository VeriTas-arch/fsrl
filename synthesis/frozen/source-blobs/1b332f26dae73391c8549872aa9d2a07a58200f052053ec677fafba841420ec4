import tempfile
import unittest
from pathlib import Path

from fsrl.confirmation import (
    aggregate_confirmation,
    compare_behavior_to_human,
    load_json,
    validate_confirmation_contract,
)


class ConfirmationTests(unittest.TestCase):
    def test_frozen_contract_hashes_and_human_gate_pass(self):
        result = validate_confirmation_contract()
        self.assertTrue(result["passed"])

    def test_pilot_and_formal_seeds_are_disjoint(self):
        pilot_path = Path("benchmarks/pilot_v1.json")
        self.assertTrue(validate_confirmation_contract(pilot_path)["passed"])
        pilot = load_json(pilot_path)
        formal = load_json(Path("benchmarks/confirmation_v1.json"))
        self.assertTrue(
            set(pilot["training"]["seeds"]).isdisjoint(
                formal["training"]["seeds"]
            )
        )

    def test_behavior_comparison_uses_registered_bootstrap_interval(self):
        behavior = {
            "summary": {
                "eligible_subjects": 10,
                "overall_accuracy": 0.8,
                "learned_accuracy": 0.8,
                "nonlearned_accuracy": 0.8,
                "symbolic_distance_slope": {"mean": 0.04},
                "ranking_class_counts": {
                    "correct": 1,
                    "self_consistent_incorrect": 8,
                    "self_inconsistent": 1,
                },
                "stable_error_subject_prevalence": {
                    "80": {"analysis": 0.9},
                    "100": {"analysis": 0.8},
                },
                "beta_pair_class_counts_analysis": {},
            }
        }
        target = {
            name: {
                "mean": value,
                "standard_deviation": 0.1,
                "lower": value - 0.1,
                "upper": value + 0.1,
            }
            for name, value in {
                "overall_accuracy": 0.8,
                "learned_accuracy": 0.8,
                "nonlearned_accuracy": 0.8,
                "symbolic_distance_slope": 0.04,
                "correct_ranker_proportion": 0.1,
                "self_consistent_incorrect_proportion": 0.8,
                "self_inconsistent_proportion": 0.1,
                "stable_error_80_analysis_proportion": 0.9,
                "stable_error_100_analysis_proportion": 0.8,
            }.items()
        }
        human = {
            "bootstrap": {"metrics": target},
            "combined": {
                "published_figure_checks": {"beta_pair_class_counts": {}}
            },
        }
        result = compare_behavior_to_human(behavior, human, list(target))
        self.assertTrue(result["passed"])

    def test_aggregate_refuses_partial_seed_reporting(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(RuntimeError, "all registered seeds"),
        ):
            aggregate_confirmation(output_root=Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
