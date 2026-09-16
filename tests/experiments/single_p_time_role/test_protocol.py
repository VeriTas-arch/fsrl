import unittest

from fsrl.experiments.single_p_time_role.protocol import specification


class TimeRoleProtocolTests(unittest.TestCase):
    def test_protocol_fixes_unconditional_stages_and_primary_cell(self):
        spec = specification()
        self.assertEqual(spec["design"]["primary_cell"], "Ce")
        self.assertIn("unconditional", spec["design"]["execution_rule"])
        self.assertEqual(
            spec["stage_2_frozen_time_intervention"]["query_conditions"],
            [0.0, 1 / 3, 2 / 3],
        )
        self.assertEqual(spec["statistics"]["bootstrap_draws"], 2000)
        self.assertEqual(
            spec["active_repair"]["repair"]["time_retained_control_probe_time"],
            1 / 3,
        )
        self.assertEqual(
            spec["active_qualification_fix"]["repair"]["qualification_authority"],
            "records/benchmarks/qualification_v3.json",
        )
        self.assertEqual(
            spec["active_execution_repair"]["repair"]["source_lock_authority"],
            "records/benchmarks/source_lock_v2.json",
        )
        self.assertEqual(
            spec["active_integrity_repair"]["repair"]["source_lock_authority"],
            "records/benchmarks/source_lock_v3.json",
        )
        self.assertEqual(
            spec["active_arithmetic_repair"]["repair"]["source_lock_authority"],
            "records/benchmarks/source_lock_v4.json",
        )


if __name__ == "__main__":
    unittest.main()
