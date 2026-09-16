import unittest

from fsrl.experiments.pl_direct_training.protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    registered_conditions,
    registered_seeds,
)


class DirectProtocolTests(unittest.TestCase):
    def test_frozen_contract_and_repair_resolve_complete_recipe(self):
        specification = load_specification()
        self.assertEqual(
            PROTOCOL_SHA256,
            "49b02c9275cdf3707a3fa7508f75ea0ec2c8daa257150e2e671c1baa940ff2ad",
        )
        self.assertEqual(
            REPAIR_SHA256,
            "3cbd18ac36ce40b190c7759ef9ef7d3b7f951d26dfb42ae7f1c0da139d75d261",
        )
        self.assertEqual(specification["optimization"]["initial_local_gain"], 0.1)
        self.assertEqual(
            specification["optimization"]["base_backbone_learning_rate"], 1e-4
        )
        self.assertEqual(specification["optimization"]["local_learning_rate"], 0.01)
        self.assertEqual(
            registered_conditions(specification),
            ("time_retained_control", "no_time_candidate"),
        )
        self.assertEqual(
            registered_seeds(specification, "development"), (3001, 3002, 3003)
        )
        self.assertEqual(
            registered_seeds(specification, "confirmation"), (3004, 3005, 3006)
        )


if __name__ == "__main__":
    unittest.main()
