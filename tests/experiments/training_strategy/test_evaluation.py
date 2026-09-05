import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fsrl.experiments.training_strategy import evaluation, reporting
from fsrl.experiments.training_strategy.estimands import paired_estimate


class EvaluationAdmissionAndStorageTests(unittest.TestCase):
    def test_evaluation_cannot_reach_runtime_or_data_before_all_artifact_lock(self):
        with (
            patch.object(
                evaluation,
                "validate_artifact_lock",
                side_effect=RuntimeError("not locked"),
            ),
            patch.object(evaluation, "configure_execution") as runtime,
            patch.object(evaluation, "evaluate_one") as one,
        ):
            with self.assertRaisesRegex(RuntimeError, "not locked"):
                evaluation.evaluate_all()
            runtime.assert_not_called()
            one.assert_not_called()

    def test_report_cannot_run_before_committed_artifact_lock(self):
        with (
            patch.object(
                reporting,
                "validate_artifact_lock",
                side_effect=RuntimeError("not locked"),
            ),
            patch.object(reporting, "validate_evaluation") as read,
        ):
            with self.assertRaisesRegex(RuntimeError, "not locked"):
                reporting.write_report()
            read.assert_not_called()

    def test_numeric_arrays_roundtrip_without_pickle_and_do_not_overwrite(self):
        arrays = evaluation.flatten_arrays(
            {
                "liu": {
                    "margin": np.asarray([[0.5, np.nan]]),
                    "retained": np.asarray([True, False]),
                },
                "metadata": "not an array",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            evaluation.write_arrays(path, arrays)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"liu__margin", "liu__retained"})
                self.assertEqual(archive["liu__retained"].dtype, np.dtype(bool))
                self.assertTrue(np.isnan(archive["liu__margin"][0, 1]))
            with self.assertRaises(FileExistsError):
                evaluation.write_arrays(path, arrays)
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            evaluation.flatten_arrays({"unsafe": np.asarray([{}], dtype=object)})

    def test_json_missing_endpoints_roundtrip_into_paired_complete_cases(self):
        first = json.loads(
            json.dumps(
                evaluation.json_ready(np.asarray([0.7, np.nan, 0.8])), allow_nan=False
            )
        )
        second = [0.6, 0.2, 0.6]
        result = paired_estimate(
            first, second, seed=910041, statistics={"samples": 100, "interval": 0.95}
        )
        self.assertEqual(result["subjects"], 2)
        self.assertAlmostEqual(result["mean"], 0.15)

    def test_pairing_audit_rejects_routing_changes(self):
        keys = (
            "retention",
            "probabilities",
            "cue_codes",
            "support_pairs",
            "observed_signed_evidence",
            "natural_local_evidence",
            "shuffled_local_evidence",
            "evidence_routing",
            "query_routing",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {f"liu__{key}": np.ones((2, 2)) for key in keys}
            evaluation.write_arrays(root / "a.npz", arrays)
            arrays["liu__query_routing"] = np.zeros((2, 2))
            evaluation.write_arrays(root / "b.npz", arrays)
            first = {
                "generic_stream_fingerprints": {"1": "same"},
                "raw_arrays": {"path": "a.npz"},
            }
            second = {**first, "raw_arrays": {"path": "b.npz"}}
            with (
                patch.object(reporting, "REPO_ROOT", root),
                self.assertRaises(AssertionError),
            ):
                reporting.verify_matched_evaluation(first, second)
