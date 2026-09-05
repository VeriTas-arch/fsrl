"""Sorted JSON must not weaken the frozen main-model v2 report audit."""

import json
import unittest

from tools.provenance.main_model_evaluation_v2 import render_report
from tools.provenance.verify_main_model_evaluation_v2 import (
    check_summary_and_report,
)


def _statistic(mean: float) -> dict:
    return {"mean": mean, "interval": {"lower": mean - 0.01, "upper": mean + 0.01}}


class MainModelEvaluationV2PublicationTests(unittest.TestCase):
    def setUp(self):
        internal = {
            "classification": "direction_unresolved",
            "transition": {},
            "inversion_bins": {},
            "internal_strict_correct": _statistic(0.03),
            "sampled_correct_all_subjects": _statistic(0.03),
            "loss_flow": _statistic(0.01),
            "rescue_flow": _statistic(0.01),
            "net_sampling_shift": _statistic(0.0),
        }
        interval = {
            "mean": 0.9,
            "interval": {"lower": 0.89, "upper": 0.91},
        }
        self.rebuilt = {
            "outcome": "retrospective_core_behavior_supported",
            "fits": {
                "2114": {
                    "retrospective_core_behavior_supported": True,
                    "core_stability_threshold_prospective_for_this_fit": False,
                    "legacy_full_quantitative_fidelity": False,
                    "legacy_pilot_rows": {
                        "qualitative": 9,
                        "quantitative": 8,
                        "total": 9,
                    },
                    "legacy_joint_qualitative": {"rate": 0.97},
                    "legacy_joint_quantitative": {"rate": 0.02},
                    "difficulty_guardrails": {
                        "learned_accuracy": interval,
                        "nonlearned_accuracy": interval,
                    },
                    "ranking_composition": {},
                    "ranking_composition_total_variation": _statistic(0.12),
                    "internal_to_sampled": internal,
                }
            },
            "sampling_localization": "direction_unresolved",
            "labels": {},
            "next_model": {},
        }
        original = {**self.rebuilt, "contract_sha256": "x"}
        self.saved = render_report(original)
        self.loaded = json.loads(json.dumps(original, sort_keys=True))

    def test_sorted_roundtrip_restores_original_report_order(self):
        self.assertNotEqual(self.saved, render_report(self.loaded))
        check_summary_and_report(self.loaded, self.rebuilt, self.saved)

    def test_changed_summary_is_rejected(self):
        self.loaded["outcome"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "summary differs"):
            check_summary_and_report(self.loaded, self.rebuilt, self.saved)

    def test_changed_report_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "report differs"):
            check_summary_and_report(self.loaded, self.rebuilt, self.saved + " ")


if __name__ == "__main__":
    unittest.main()
