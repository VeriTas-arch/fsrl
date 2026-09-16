import unittest

from fsrl.experiments.pl_crosstalk_decomposition.protocol import (
    PROTOCOL_SHA256,
    load_specification,
    registered_seeds,
)


class CrossTalkProtocolTests(unittest.TestCase):
    def test_contract_freezes_read_only_three_seed_diagnostic(self):
        specification = load_specification()
        self.assertEqual(
            PROTOCOL_SHA256,
            "0a8aa21c8798811817d980c77701193ceedc9c75c7644eb4dd32577c54dd27ba",
        )
        self.assertEqual(registered_seeds(specification), (3004, 3005, 3006))
        self.assertEqual(
            specification["active_repair"]["repair_id"],
            "pl-crosstalk-decomposition-v1-implementation-repair1",
        )
        self.assertEqual(
            specification["design"]["new_training_or_adaptation"], "forbidden"
        )
        self.assertIn(
            "shared total correct-signed margin",
            specification["estimands"]["exact_baseline_operating_point"],
        )
        self.assertEqual(specification["statistics"]["network_pooling"], "forbidden")

    def test_raw_inputs_are_exactly_declared_for_every_seed(self):
        frozen = load_specification()["design"]["frozen_inputs"]["raw_arrays"]
        self.assertEqual(set(frozen), {"3004", "3005", "3006"})
        for seed, record in frozen.items():
            self.assertIn(f"seed-{seed}/raw.npz", record["path"])
            self.assertEqual(len(record["sha256"]), 64)
            self.assertGreater(record["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
