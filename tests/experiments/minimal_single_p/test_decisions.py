import unittest

from fsrl.experiments.minimal_single_p.decisions import (
    classify_level,
    panel_passed,
    successor,
)


def endpoint(lower):
    return {"bootstrap": {"lower": lower}}


class MinimalSinglePDecisionTests(unittest.TestCase):
    def test_strict_core_gates(self):
        panel = {
            "competence": {
                "learned": endpoint(0.6),
                "nonlearned": endpoint(0.6),
            },
            "P_dependence": {
                "learned": endpoint(0.1),
                "nonlearned": endpoint(0.1),
            },
            "coherence": endpoint(0.96),
        }
        self.assertTrue(panel_passed(panel))
        panel["coherence"] = endpoint(0.95)
        self.assertFalse(panel_passed(panel))

    def test_outcomes_and_stop_transitions(self):
        self.assertEqual(
            classify_level({"a": True, "b": True, "c": True}),
            "clear_continue",
        )
        self.assertEqual(
            classify_level({"a": True, "b": False, "c": False}),
            "mixed_boundary",
        )
        self.assertEqual(
            classify_level({"a": False, "b": False, "c": False}),
            "structural_collapse",
        )
        self.assertEqual(successor("C0", "clear_continue"), "M1")
        self.assertIsNone(successor("C0", "mixed_boundary"))
        self.assertIsNone(successor("M5_H50", "clear_continue"))


if __name__ == "__main__":
    unittest.main()
