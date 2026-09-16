import unittest

from fsrl.experiments.pl_direct_training.protocol import load_specification
from fsrl.experiments.pl_direct_training.qualification import cpu_qualification


class DirectQualificationTests(unittest.TestCase):
    def test_cpu_qualification_covers_registered_gates(self):
        checks = cpu_qualification(load_specification())
        self.assertEqual(
            set(checks),
            {
                "initialization_and_fresh_boundary",
                "no_time_isolation",
                "optimizer_forward_loss",
                "optimizer_effective_parameters",
                "optimizer_mapped_moments",
                "packed_local",
                "timestep_identity",
                "gradient_paths",
            },
        )
        self.assertTrue(all(row["passed"] for row in checks.values()))
        self.assertEqual(
            checks["timestep_identity"]["liu_v2_literal_active_microsteps"], 688
        )


if __name__ == "__main__":
    unittest.main()
