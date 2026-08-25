import unittest
from itertools import combinations

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.tasks.registered_protocol import load_ranking_protocol


def logits_for_orders(orders, magnitude=20.0):
    outputs = []
    for order in orders:
        rank = {item: position for position, item in enumerate(order)}
        subject = {}
        for first, second in combinations(range(len(order)), 2):
            subject[(first, second)] = (
                magnitude if rank[first] < rank[second] else -magnitude
            )
            subject[(second, first)] = -subject[(first, second)]
        outputs.append(subject)
    return tuple(outputs)


class BehavioralAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_ranking_protocol()

    def test_perfect_policy_produces_correct_rankers(self):
        logits = logits_for_orders([tuple(range(8))] * 4)
        result = analyze_sampled_query_policy(self.protocol, logits, seed=81)
        summary = result["summary"]
        self.assertEqual(summary["overall_accuracy"], 1.0)
        self.assertEqual(summary["ranking_class_counts"]["correct"], 4)
        self.assertEqual(summary["excluded_below_chance"], 0)

    def test_coherent_wrong_order_is_not_called_correct(self):
        wrong_order = (0, 2, 1, 3, 4, 5, 6, 7)
        logits = logits_for_orders([wrong_order] * 3)
        result = analyze_sampled_query_policy(self.protocol, logits, seed=82)
        summary = result["summary"]
        self.assertEqual(summary["ranking_class_counts"]["correct"], 0)
        self.assertEqual(
            summary["ranking_class_counts"]["self_consistent_incorrect"], 3
        )
        self.assertEqual(summary["mean_self_consistency_coefficient"], 1.0)

    def test_randomized_orientation_prevents_always_left_inflation(self):
        subject = {
            oriented: 20.0
            for pair in combinations(range(8), 2)
            for oriented in (pair, (pair[1], pair[0]))
        }
        result = analyze_sampled_query_policy(self.protocol, (subject,) * 40, seed=83)
        accuracy = result["summary"]["overall_accuracy"]
        self.assertGreater(accuracy, 0.45)
        self.assertLess(accuracy, 0.55)

    def test_subjective_orders_are_recorded_per_subject(self):
        orders = [tuple(range(8)), tuple(reversed(range(8)))]
        result = analyze_sampled_query_policy(
            self.protocol, logits_for_orders(orders), seed=84
        )
        self.assertEqual(
            result["subjects"][0]["subjective_order_high_to_low"], list(orders[0])
        )
        self.assertEqual(
            result["subjects"][1]["subjective_order_high_to_low"], list(orders[1])
        )


if __name__ == "__main__":
    unittest.main()
