import unittest

from fsrl.experiments.pl_direct_training.functional_replication.protocol import (
    EXECUTION_REPAIR_SHA256,
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    candidate_specification,
    load_execution_repair,
    load_specification,
    registered_conditions,
    registered_seeds,
)


class FunctionalProtocolTests(unittest.TestCase):
    def test_frozen_contract_resolves_unchanged_no_time_candidate(self):
        specification = load_specification()
        candidate = candidate_specification()
        self.assertEqual(
            PROTOCOL_SHA256,
            "dc146c7723ad147a5fd964f8af5c66cad4b182c034fa20fc3c5391bb4a2e16d0",
        )
        self.assertEqual(
            REPAIR_SHA256,
            "f8fb383ee947a44ba784195af9f87b380136dcae22dbccac688eff704cc087ee",
        )
        self.assertEqual(
            EXECUTION_REPAIR_SHA256,
            "5707bcf608434700d12517dcfa75c47a38fe4912f918cb4287b53774a456d4ce",
        )
        self.assertFalse(
            load_execution_repair()["authorized_repair"]["scientific_change"]
        )
        self.assertEqual(registered_seeds(specification), (3004, 3005, 3006))
        self.assertEqual(
            registered_conditions(specification),
            (
                "dual_access",
                "shared_access",
                "dual_evidence_shuffle",
                "dual_query_shuffle",
                "local_off",
                "P_off_dual",
                "P_off_shared",
            ),
        )
        self.assertEqual(
            specification["active_repair"]["remote_reference_condition"],
            "shared_access",
        )
        self.assertEqual(
            candidate["architecture"]["no_time_candidate"]["backbone_parameters"],
            87205,
        )
        self.assertEqual(candidate["optimization"]["total_steps"], 1500)
        self.assertEqual(candidate["architecture"]["common"]["support_trial_steps"], 4)
        self.assertEqual(candidate["architecture"]["common"]["query_trial_steps"], 2)


if __name__ == "__main__":
    unittest.main()
