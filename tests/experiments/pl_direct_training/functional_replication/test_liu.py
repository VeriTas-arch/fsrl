import unittest

import numpy as np

from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    blockwise_derangements,
)
from fsrl.experiments.pl_direct_training.functional_replication.liu import (
    FunctionalLiuEvaluator,
    admitted_local_evidence,
)


class FunctionalAccessTests(unittest.TestCase):
    def test_shared_and_dual_admission_restore_historical_equations(self):
        self.assertEqual(admitted_local_evidence(0.7, 1.0, 0.2, "shared"), 0.7)
        self.assertEqual(admitted_local_evidence(0.7, 1.0, 0.2, "dual"), 0.7)
        self.assertEqual(admitted_local_evidence(-0.4, 0.0, 0.3, "shared"), 0.0)
        self.assertAlmostEqual(admitted_local_evidence(-0.4, 0.0, 0.3, "dual"), -0.12)

    def test_evidence_routing_is_a_blockwise_multiset_derangement(self):
        values = np.arange(5 * 4 * 8, dtype=np.float32).reshape(5, 32)
        maps = blockwise_derangements(5, 4, 8, 97)
        routed = FunctionalLiuEvaluator.route_evidence(values, maps)
        self.assertTrue(np.all(maps != np.arange(8)[None, None, :]))
        for subject in range(5):
            for block in range(4):
                start = block * 8
                stop = start + 8
                np.testing.assert_array_equal(
                    np.sort(routed[subject, start:stop]),
                    np.sort(values[subject, start:stop]),
                )


if __name__ == "__main__":
    unittest.main()
