"""Sorted JSON must not invalidate or weaken frozen publication verification."""

import json
import unittest

from fsrl.experiments.cohort_diagnostic.reporting import render_report
from tools.provenance.verify_resampled_cohort_v1 import check_summary_and_report


class ResampledReportAuditTests(unittest.TestCase):
    def setUp(self):
        interval = {"lower": 0.1, "upper": 0.2}
        rate = {"successes": 2, "cohorts": 4, "rate": 0.5, **interval}
        endpoint = {
            "mean": 0.15,
            "interval": interval,
            "reference": interval,
            "classification": "mean_within_reference",
            "undefined_cohorts": [],
        }
        self.rebuilt = {
            "fits": {
                "2114": {
                    "continuous": {
                        "symbolic_distance_effect": endpoint,
                        "learned_accuracy": endpoint,
                    },
                    "outcome": "mean_within_reference",
                    "all_nine": {"joint": rate},
                    "pass_rates": {
                        "learned_accuracy": {"qualitative": rate, "calibration": rate}
                    },
                }
            },
            "outcome": "mean_within_reference",
        }
        original = {**self.rebuilt, "stop_rule": "stop"}
        self.saved = render_report(original)
        self.loaded = json.loads(json.dumps(original, sort_keys=True))

    def test_sorted_json_roundtrip_preserves_exact_original_report(self):
        self.assertNotEqual(self.saved, render_report(self.loaded))
        check_summary_and_report(self.loaded, self.rebuilt, self.saved)

    def test_changed_statistic_is_rejected(self):
        self.loaded["outcome"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "statistics"):
            check_summary_and_report(self.loaded, self.rebuilt, self.saved)

    def test_changed_report_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "report differs"):
            check_summary_and_report(self.loaded, self.rebuilt, self.saved + " ")
