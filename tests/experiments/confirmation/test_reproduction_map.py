import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fsrl.experiments.confirmation.reproduction_map import (
    IMPLEMENTATION_LOCK_PATH,
    SPECIFICATION_PATH,
    bootstrap_mean_interval,
    build_map,
    classify_network_flags,
    endpoint_statistics,
    position_profile,
    validate_sources,
)
from fsrl.infra.provenance import load_json, write_json_exclusive


class ModelBehaviorReproductionMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build_map()

    def test_registered_sources_and_implementation_are_locked(self):
        validation = validate_sources()

        self.assertTrue(validation["passed"])
        self.assertTrue(all(check["passed"] for check in validation["checks"]))

    def test_source_validation_fails_closed_on_changed_specification(self):
        specification = load_json(SPECIFICATION_PATH)
        changed = copy.deepcopy(specification)
        changed["scientific_question"] += " changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                validate_sources(path, IMPLEMENTATION_LOCK_PATH)

    def test_position_profile_and_endpoint_rule_use_incident_pairs(self):
        rows = []
        for first in range(8):
            for second in range(first + 1, 8):
                rows.append(
                    {
                        "pair": [first, second],
                        "learned": False,
                        "value": float(first in {0, 7} or second in {0, 7}),
                    }
                )

        profile = position_profile(rows, "value")
        endpoints = endpoint_statistics(profile)

        self.assertEqual(profile.shape, (8,))
        self.assertTrue(endpoints["both_endpoints_above_interior"])
        self.assertGreater(endpoints["mean_endpoint_contrast"], 0.0)

    def test_bootstrap_is_deterministic_and_finite(self):
        first = bootstrap_mean_interval(
            np.asarray([0.0, 1.0, 1.0]), samples=1000, seed=5
        )
        second = bootstrap_mean_interval(
            np.asarray([0.0, 1.0, 1.0]), samples=1000, seed=5
        )

        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first["point"]))
        self.assertLessEqual(first["interval"]["lower"], first["point"])
        self.assertGreaterEqual(first["interval"]["upper"], first["point"])

    def test_status_requires_both_networks_without_pooling(self):
        reproduced = {
            "2104": {"qualitative": True, "calibration": True},
            "2105": {"qualitative": True, "calibration": True},
        }
        mismatch = copy.deepcopy(reproduced)
        mismatch["2105"]["calibration"] = False
        failed = copy.deepcopy(reproduced)
        failed["2104"]["qualitative"] = False

        self.assertEqual(classify_network_flags(reproduced), "reproduced")
        self.assertEqual(
            classify_network_flags(mismatch),
            "qualitatively_reproduced_quantitatively_mismatched",
        )
        self.assertEqual(classify_network_flags(failed), "not_reproduced")

    def test_complete_map_has_nine_rows_and_no_checkpoint_execution(self):
        result = self.result

        self.assertTrue(all(result["identity_gates"].values()))
        self.assertEqual(set(result["networks"]), {"2104", "2105"})
        self.assertEqual(result["summary"]["rows"], 9)
        self.assertEqual(sum(result["summary"]["status_counts"].values()), 9)
        self.assertEqual(
            result["execution_mode"],
            "read_only_existing_artifacts_no_checkpoint_load",
        )
        self.assertNotIn("pooled", json.dumps(result["networks"]))

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_exclusive(path, self.result)

            self.assertEqual(load_json(path), self.result)
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, self.result)


if __name__ == "__main__":
    unittest.main()
