import unittest

from fsrl.evaluation.metrics import count_circular_triads, maximum_circular_triads


class EvaluationMetricTests(unittest.TestCase):
    def test_circular_triad_count_distinguishes_cycle_from_total_order(self):
        cycle = {(0, 1): 0, (0, 2): 2, (1, 2): 1}
        ordered = {(0, 1): 0, (0, 2): 0, (1, 2): 1}
        self.assertEqual(count_circular_triads(cycle, 3), 1)
        self.assertEqual(count_circular_triads(ordered, 3), 0)

    def test_maximum_circular_triads_supports_odd_and_even_item_counts(self):
        self.assertEqual(maximum_circular_triads(3), 1)
        self.assertEqual(maximum_circular_triads(8), 20)
