import unittest

from tools.quality.complexity_budget import audit_complexity_budget


class QualityGateTests(unittest.TestCase):
    def test_complexity_debt_cannot_expand_or_worsen(self):
        result = audit_complexity_budget()
        self.assertTrue(result["passed"], result)
