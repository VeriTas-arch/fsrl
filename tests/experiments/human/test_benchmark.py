import unittest

from fsrl.experiments.human.benchmark import (
    DEFAULT_PREREGISTERED_PATH,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_REPLICATION_PATH,
    build_human_benchmark,
)
from fsrl.tasks.registered_protocol import load_ranking_protocol


class HumanBenchmarkTests(unittest.TestCase):
    def test_v2_protocol_matches_source_order_and_pairs(self):
        protocol = load_ranking_protocol(DEFAULT_PROTOCOL_PATH)
        labels = protocol.item_labels
        self.assertEqual(
            tuple(labels[item] for item in protocol.true_order_high_to_low),
            tuple(reversed(labels)),
        )
        observed = {
            frozenset((labels[higher], labels[lower]))
            for higher, lower in protocol.support_pairs_higher_lower
        }
        expected = {
            frozenset(pair)
            for pair in (
                ("A", "F"),
                ("B", "C"),
                ("B", "E"),
                ("C", "G"),
                ("D", "F"),
                ("D", "G"),
                ("E", "H"),
                ("A", "H"),
            )
        }
        self.assertEqual(observed, expected)

    def test_source_reanalysis_reproduces_paper_checks(self):
        result = build_human_benchmark(
            DEFAULT_PREREGISTERED_PATH,
            DEFAULT_REPLICATION_PATH,
            DEFAULT_PROTOCOL_PATH,
            bootstrap_seed=17,
            bootstrap_samples=10,
        )
        summary = result["combined"]
        self.assertEqual(
            result["status"], "source_recomputed_and_paper_checks_reproduced"
        )
        self.assertEqual(summary["eligible_subjects"], 77)
        self.assertEqual(
            summary["ranking_class_counts"],
            {
                "correct": 8,
                "self_consistent_incorrect": 64,
                "self_inconsistent": 5,
            },
        )
        self.assertAlmostEqual(summary["overall_accuracy"], 0.8523654916512059)
        self.assertAlmostEqual(summary["learned_accuracy"], 0.913961038961039)
        self.assertAlmostEqual(summary["nonlearned_accuracy"], 0.8277272727272728)
