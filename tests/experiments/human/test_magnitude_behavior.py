import copy
import json
import tempfile
import unittest
from pathlib import Path

from fsrl.experiments.human.magnitude_behavior import (
    DEFAULT_SPECIFICATION_PATH,
    planned_power,
    run_validation,
    validate_specification,
)


class MagnitudePlacementBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = json.loads(DEFAULT_SPECIFICATION_PATH.read_text())

    def test_frozen_assignment_and_analysis_contract_passes(self):
        result = validate_specification(self.specification)

        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "synthetic_validation_passed")
        self.assertEqual(result["data_files_opened"], [])
        self.assertTrue(
            all(value for group in result["gates"].values() for value in group.values())
        )
        geometry = result["derived"]["assignment_geometry"]
        self.assertEqual(len(geometry["learned_pairs"]), 8)
        self.assertEqual(len(geometry["nonlearned_order_flip_pairs"]), 7)
        self.assertEqual(len(geometry["nonlearned_same_direction_pairs"]), 13)

    def test_assignment_b_is_exact_and_not_an_arbitrary_gap_shuffle(self):
        changed = copy.deepcopy(self.specification)
        changed["frozen_assignments"]["assignment_B"]["support_gaps"]["F>A"] = 3

        result = validate_specification(changed)

        self.assertFalse(result["passed"])
        self.assertFalse(
            result["gates"]["assignment"]["assignment_B_registered_gaps_match_levels"]
        )

    def test_pair_partition_cannot_be_changed_after_registration(self):
        changed = copy.deepcopy(self.specification)
        changed["frozen_assignments"]["pair_partition"]["nonlearned_order_flip"][0] = (
            "A-C"
        )

        result = validate_specification(changed)

        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["assignment"]["order_flip_pair_set"])
        self.assertFalse(result["gates"]["assignment"]["pair_partition_complete"])

    def test_planned_n_is_first_registered_multiple_that_passes(self):
        result = validate_specification(self.specification)
        power_100 = result["derived"]["power_at_n_100"]
        power_96 = result["derived"]["power_at_n_96"]

        self.assertGreaterEqual(
            min(value for axis in power_100.values() for value in axis.values()),
            0.90,
        )
        self.assertLess(
            min(value for axis in power_96.values() for value in axis.values()),
            0.90,
        )
        observed = planned_power(100, 0.0598490073222847, 0.02)
        self.assertAlmostEqual(observed["equivalence_at_zero"], 0.9057117958156491)

    def test_validation_does_not_open_registered_participant_sources(self):
        changed = copy.deepcopy(self.specification)
        for source in changed["registered_sources"].values():
            source["path"] = "forbidden/nonexistent-participant-source"

        result = validate_specification(changed)

        self.assertTrue(result["passed"])
        self.assertEqual(result["data_files_opened"], [])

    def test_result_writer_is_exclusive_and_records_pure_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validation.json"
            first = run_validation(DEFAULT_SPECIFICATION_PATH, output)
            written = json.loads(output.read_text())

            self.assertTrue(first["passed"])
            self.assertEqual(written["data_files_opened"], [])
            self.assertEqual(
                written["provenance"]["registered_source_files_opened"], []
            )
            with self.assertRaises(FileExistsError):
                run_validation(DEFAULT_SPECIFICATION_PATH, output)
