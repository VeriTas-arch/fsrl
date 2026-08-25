import unittest
from itertools import combinations

from fsrl.analysis.algorithmic import compare_neural_policy_to_exact_posterior
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class AlgorithmicComparisonTests(unittest.TestCase):
    def test_true_order_logits_match_exact_map_with_full_evidence(self):
        protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        rank = {
            item: position
            for position, item in enumerate(protocol.true_order_high_to_low)
        }
        evidence = tuple(
            {
                "higher_item": higher,
                "lower_item": lower,
                "magnitude": (rank[lower] - rank[higher]) / 7.0,
                "reliability": 1.0,
            }
            for higher, lower in protocol.support_pairs_higher_lower
            for _ in range(protocol.support_blocks)
        )
        logits = {}
        for first, second in combinations(range(protocol.n_items), 2):
            margin = float(rank[second] - rank[first])
            logits[(first, second)] = margin
            logits[(second, first)] = -margin
        behavior = {
            "subjects": [
                {
                    "ranking_class": "correct",
                    "subjective_order_high_to_low": list(
                        protocol.true_order_high_to_low
                    ),
                }
            ]
        }
        result = compare_neural_policy_to_exact_posterior(
            protocol,
            (evidence,),
            (logits,),
            behavior,
            posterior_temperature=0.05,
            readout_temperature=0.25,
        )
        self.assertTrue(result["subjects"][0]["neural_is_map"])
        self.assertEqual(result["group"]["neural_map_proportion"], 1.0)
        self.assertEqual(result["subjects"][0]["closest_map_kendall_tau"], 1.0)


if __name__ == "__main__":
    unittest.main()
