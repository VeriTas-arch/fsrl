import copy
import unittest

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.experiments.confirmation.reproduction_map import model_record
from fsrl.experiments.training_strategy.behavior import (
    behavior_metrics,
    classify_rows,
    human_references,
)
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.protocol import load_specification
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs


class BehaviorMapParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_specification()
        cls.references = human_references(cls.specification)
        protocol = RankingProtocol(
            "synthetic-behavior",
            tuple(str(i) for i in range(8)),
            tuple(range(8)),
            tuple((i, i + 1) for i in range(7)),
            2,
            20,
            {},
        )
        rng = np.random.default_rng(910030)
        potentials = -np.arange(8)[None, :] * 0.2 + rng.normal(size=(24, 8))
        logits = tuple(
            {(i, j): float(p[i] - p[j]) for i, j in ordered_pairs(8)}
            for p in potentials
        )
        cls.behavior = analyze_sampled_query_policy(
            protocol, logits, seed=910031, temperature=1.0
        )

    def test_all_nine_point_classifiers_match_frozen_map(self):
        statistics = {"samples": 100, "interval": 0.95}
        current = behavior_metrics(self.behavior, 910032, statistics)
        flags = classify_rows(current["metrics"], self.references)
        old_behavior = copy.deepcopy(self.behavior)
        old_behavior["participant_bootstrap"] = {
            name: {"bootstrap": current["metrics"][name]["interval"]}
            for name in ("learned_accuracy", "nonlearned_accuracy")
        }
        legacy = model_record(
            {"seeds": {"1": {"behavior": {"dual_access_matched": old_behavior}}}},
            "1",
            self.references["intervals"],
            self.references["serial"],
            self.references["tau"],
            {"bootstrap": {"samples": 100, "human_new_estimands_seed": 910032}},
        )
        self.assertEqual(flags, legacy["flags"])
        self.assertEqual(len(flags), 9)
        self.assertEqual(current["eligible_subjects"], legacy["eligible_subjects"])
        self.assertAlmostEqual(
            current["metrics"]["inter_subject_ranking_diversity"]["point"],
            legacy["metrics"]["inter_subject_ranking_diversity"]["point"],
        )

    def test_absent_analysis_cohort_is_reported_without_dropping_other_rows(self):
        behavior = copy.deepcopy(self.behavior)
        for subject in behavior["subjects"]:
            subject["ranking_class"] = "correct"
        current = behavior_metrics(behavior, 910033, {"samples": 100, "interval": 0.95})
        flags = classify_rows(current["metrics"], self.references)
        self.assertEqual(current["analysis_subjects_excluding_correct_rankers"], 0)
        self.assertIsNone(
            current["metrics"]["inter_subject_ranking_diversity"]["point"]
        )
        self.assertFalse(flags["stable_within_subject_errors"]["qualitative"])
        self.assertFalse(flags["inter_subject_ranking_diversity"]["calibration"])
        self.assertTrue(flags["learned_accuracy"]["qualitative"])
        self.assertEqual(len(flags), 9)
        self.assertEqual(json_ready(current), current)

    def test_undefined_distance_test_cannot_be_counted_as_positive(self):
        current = behavior_metrics(
            self.behavior, 910034, {"samples": 100, "interval": 0.95}
        )
        current["metrics"]["symbolic_distance_effect"] = {
            "mean": 0.1,
            "p_vs_zero": None,
        }
        flags = classify_rows(current["metrics"], self.references)
        self.assertFalse(flags["symbolic_distance_effect"]["qualitative"])
