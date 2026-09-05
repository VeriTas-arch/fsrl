import copy
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fsrl.experiments.cohort_diagnostic.execution import validate_shard
from fsrl.experiments.cohort_diagnostic.inputs import read_arrays, shard_indices
from fsrl.experiments.cohort_diagnostic.protocol import (
    cohort_settings,
    load_parameters,
    specification,
)
from fsrl.experiments.cohort_diagnostic.qualification import parent_point_parity
from fsrl.experiments.cohort_diagnostic.statistics import (
    cohort_record,
    continuous_summary,
    interval_classification,
    reference_intervals,
    summarize_fit,
    wilson,
)
from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.evaluation import write_arrays


def synthetic_behavior() -> dict:
    return {
        "subjects": [
            {
                "overall_accuracy": accuracy,
                "learned_accuracy": learned,
                "nonlearned_accuracy": nonlearned,
                "subjective_order_high_to_low": list(range(8)),
            }
            for accuracy, learned, nonlearned in (
                (0.4, 0.1, 0.2),
                (0.8, 0.9, 0.7),
                (0.8, 0.9, 0.9),
            )
        ],
        "pairs": [
            {"pair": list(pair), "mean_accuracy_all": 0.8, "learned": False}
            for pair in combinations(range(8), 2)
        ],
        "summary": {
            "ranking_class_counts": {
                "correct": 1,
                "self_consistent_incorrect": 0,
                "self_inconsistent": 1,
            },
            "symbolic_distance_slope": {
                "mean": 0.04,
                "p_vs_zero": 0.001,
                "t_vs_zero": 4,
            },
            "stable_error_subject_prevalence": {"80": {"analysis": 1.0}},
            "mean_inter_subject_kendall_tau": None,
            "beta_pair_class_counts_analysis": {
                "bimodal": 15,
                "ordinary_unimodal": 0,
                "low_accuracy": 0,
                "high_accuracy": 13,
                "boundary": 0,
                "not_fit": 0,
            },
            "analysis_subjects_excluding_correct_rankers": 1,
        },
    }


class CohortDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specification()
        cls.references = human_references(resolved_specification())

    def rows(self):
        point = cohort_record(synthetic_behavior(), self.references)
        intervals = reference_intervals(self.references)
        point["values"] = {
            name: (ref["lower"] + ref["upper"]) / 2 for name, ref in intervals.items()
        }
        return [{"cohort": index, **copy.deepcopy(point)} for index in range(400)]

    def test_registered_matrix_has_no_new_training(self):
        self.assertEqual(self.spec["fits"], [2114, 2115, 2116])
        self.assertEqual(self.spec["cohorts"]["count"], 400)
        self.assertEqual(self.spec["condition"], "resampled")
        self.assertEqual(set(load_parameters()), {"2114", "2115", "2116"})

    def test_seed_ranges_are_disjoint(self):
        used = set()
        for index in range(400):
            seeds = cohort_settings(index)["evaluation"]["liu"]
            groups = [
                [seeds["cue_seed"]],
                list(range(seeds["support_seed"], seeds["support_seed"] + 77)),
                [seeds["subject_encoding_seed"]],
                [seeds["encoding_seed"]],
                list(range(seeds["choice_seed"], seeds["choice_seed"] + 154)),
            ]
            for group in groups:
                self.assertFalse(used.intersection(group))
                used.update(group)
        self.assertTrue(min(used) > 971200)
        self.assertTrue(max(used) < 2980000)

    def test_unregistered_indices_are_rejected(self):
        for index in (-1, 400):
            with self.assertRaises(ValueError):
                cohort_settings(index)
        for index in (-1, 1, 400):
            with self.assertRaises(ValueError):
                shard_indices(index)
        self.assertEqual(shard_indices(380), list(range(380, 400)))

    def test_all_nine_exposed_parent_records_have_exact_points(self):
        errors = parent_point_parity()
        self.assertEqual(len(errors), 9)
        self.assertLessEqual(max(errors.values()), 1e-12)

    def test_all_subject_accuracy_is_not_eligible_only(self):
        record = cohort_record(synthetic_behavior(), self.references)
        self.assertAlmostEqual(record["values"]["learned_accuracy"], 1.9 / 3)
        self.assertAlmostEqual(record["values"]["nonlearned_accuracy"], 1.8 / 3)
        self.assertEqual(record["values"]["correct_ranker"], 0.5)
        self.assertEqual(record["eligible_subjects"], 2)
        self.assertEqual(record["analysis_subjects"], 1)

    def test_undefined_never_passes(self):
        behavior = synthetic_behavior()
        for subject in behavior["subjects"]:
            subject["overall_accuracy"] = 0.4
        behavior["summary"]["ranking_class_counts"] = dict.fromkeys(
            behavior["summary"]["ranking_class_counts"], 0
        )
        behavior["summary"]["analysis_subjects_excluding_correct_rankers"] = 0
        behavior["summary"]["stable_error_subject_prevalence"]["80"]["analysis"] = None
        result = cohort_record(behavior, self.references)
        for name in (
            "stable_within_subject_errors",
            "inter_subject_ranking_diversity",
            "hodge_reconstructed_subjective_ranking",
            "self_consistent_vs_inconsistent_errors",
        ):
            self.assertFalse(result["flags"][name]["qualitative"])
            self.assertFalse(result["flags"][name]["calibration"])

    def test_classification_uses_whole_interval_and_strict_excess(self):
        reference = {"lower": 1, "upper": 2}
        for interval, expected in (
            (None, "unresolved"),
            ({"lower": 2.01, "upper": 3}, "sustained_above_reference"),
            ({"lower": 0, "upper": 0.99}, "sustained_below_reference"),
            ({"lower": 1, "upper": 2}, "mean_within_reference"),
            ({"lower": 2, "upper": 3}, "boundary_unresolved"),
            ({"lower": 0.99, "upper": 1.5}, "boundary_unresolved"),
        ):
            self.assertEqual(interval_classification(interval, reference), expected)

    def test_complete_cohort_inventory_is_required(self):
        rows = self.rows()
        for incomplete in (rows[:-1], rows[::-1], rows[:399] + [rows[0]]):
            with self.assertRaises(RuntimeError):
                summarize_fit(incomplete, 2114, self.spec, self.references)

    def test_constant_cohort_summary_and_no_promotion(self):
        result = summarize_fit(self.rows(), 2114, self.spec, self.references)
        self.assertEqual(result["outcome"], "mean_within_reference")
        self.assertFalse(result["main_model_promoted"])
        self.assertFalse(result["parent_outcome_changed"])
        self.assertEqual(sum(row["cohorts"] for row in result["morphology_joint"]), 400)

    def test_undefined_cohort_is_not_dropped(self):
        rows = self.rows()
        rows[31]["values"]["symbolic_distance_effect"] = None
        result = continuous_summary(rows, 2114, self.spec, self.references)[
            "symbolic_distance_effect"
        ]
        self.assertEqual(result["undefined_cohorts"], [31])
        self.assertIsNone(result["mean"])
        self.assertEqual(result["classification"], "unresolved")

    def test_bootstrap_resamples_whole_cohorts(self):
        rows = self.rows()
        values = np.linspace(0.7, 0.9, len(rows))
        for row, value in zip(rows, values, strict=True):
            row["values"]["learned_accuracy"] = float(value)
        result = continuous_summary(rows, 2114, self.spec, self.references)[
            "learned_accuracy"
        ]
        counts = np.random.default_rng(2980000 + 2114).multinomial(
            400, np.full(400, 1 / 400), size=10000
        )
        low, high = np.quantile(counts @ values / 400, [0.025, 0.975])
        self.assertAlmostEqual(result["mean"], values.mean(), places=12)
        self.assertAlmostEqual(result["interval"]["lower"], low, places=12)
        self.assertAlmostEqual(result["interval"]["upper"], high, places=12)

    def test_wilson_boundaries(self):
        self.assertEqual(wilson([False] * 400)["lower"], 0)
        self.assertAlmostEqual(wilson([True] * 400)["upper"], 1)
        value = wilson([False, True] * 200)
        self.assertAlmostEqual(value["rate"], 0.5)
        self.assertAlmostEqual(value["lower"] + value["upper"], 1)

    def test_shard_source_guard_precedes_array_access(self):
        with (
            patch(
                "fsrl.experiments.cohort_diagnostic.execution.reference",
                return_value={"locked": True},
            ),
            self.assertRaises(RuntimeError),
        ):
            validate_shard({"execution_lock": {}}, {}, 0)

    def test_numeric_storage_round_trip(self):
        arrays = {
            "cohort_indices": np.arange(20),
            "margins": np.arange(120).reshape(20, 3, 2).astype(float),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.npz"
            write_arrays(path, arrays)
            with patch(
                "fsrl.experiments.cohort_diagnostic.inputs.verify_reference",
                return_value=path,
            ):
                loaded = read_arrays({})
        for key, value in arrays.items():
            np.testing.assert_array_equal(loaded[key], value)
