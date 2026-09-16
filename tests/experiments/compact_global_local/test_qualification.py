from __future__ import annotations

import unittest

from fsrl.experiments.compact_global_local.protocol import load_specification
from fsrl.experiments.compact_global_local.qualification import (
    cpu_qualification,
    gradient_qualification,
)


class CompactQualificationTests(unittest.TestCase):
    def test_cpu_equation_and_gradient_gates_pass(self):
        checks = {
            **cpu_qualification(),
            **gradient_qualification(load_specification()),
        }
        self.assertTrue(all(row["passed"] for row in checks.values()), checks)


if __name__ == "__main__":
    unittest.main()
