"""Actual absent-memory computation and paired noninferiority boundaries."""

import unittest

from fsrl.experiments.local_memory_removal.qualification import numerical_checks
from fsrl.experiments.local_memory_removal.statistics import (
    noninferiority,
    qualification_checks,
)


class RemovalTests(unittest.TestCase):
    def test_no_local_forward_gradient_and_update(self):
        self.assertTrue(all(row["passed"] for row in numerical_checks().values()))

    def test_complete_case_matched_probability(self):
        self.assertTrue(all(qualification_checks().values()))

    def test_uncertain_is_not_noninferior_or_inferior(self):
        rows = noninferiority(
            {
                "supported": {"interval": {"lower": -0.02, "upper": 0.01}},
                "uncertain": {"interval": {"lower": -0.03, "upper": 0.01}},
                "inferior": {"interval": {"lower": -0.04, "upper": -0.021}},
            }
        )
        self.assertTrue(rows["supported"]["noninferior"])
        self.assertEqual(
            rows["uncertain"], {"noninferior": False, "materially_inferior": False}
        )
        self.assertTrue(rows["inferior"]["materially_inferior"])


if __name__ == "__main__":
    unittest.main()
