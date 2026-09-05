import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fsrl.experiments.minimal_learner import locks, training
from fsrl.experiments.minimal_learner.protocol import PROTOCOL_SHA256, specification


class MinimalLockTests(unittest.TestCase):
    def test_no_training_can_begin_before_source_lock(self):
        with (
            patch.object(
                training, "validate_source", side_effect=RuntimeError("source lock")
            ),
            patch.object(training, "runtime") as runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "source lock"):
                training.train_all()
            runtime.assert_not_called()

    def test_empty_smoke_checks_cannot_become_execution_authority(self):
        with (
            patch.object(locks, "sources", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "required"),
        ):
            locks.validate_smoke(
                {
                    "passed": True,
                    "seed": 910101,
                    "liu_evaluated": False,
                    "sources": [],
                    "checks": {},
                }
            )

    def test_complete_log_chain_and_actual_scalar_update_counts(self):
        spec = specification()
        spec["optimization"]["total_steps"] = 2
        spec["optimization"]["total_episode_exposures"] = 64
        batch_hash = hashlib.sha256(b"fixture").hexdigest()
        digest = hashlib.sha256()
        logs = []
        for step in range(2):
            digest.update(bytes.fromhex(batch_hash))
            logs.append(
                {
                    "step": step,
                    "batch_sha256": batch_hash,
                    "stream_sha256": digest.hexdigest(),
                }
            )
        config = {
            "seed": 2111,
            "condition": "score_only",
            "protocol_sha256": PROTOCOL_SHA256,
            "optimization": spec["optimization"],
            "episodes": 64,
            "stream_sha256": digest.hexdigest(),
            "optimizer_steps": {"raw_eta": 2, "raw_global_gain": 2},
            "checkpoint": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "train_log.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in logs) + "\n")
            with (
                patch.object(locks, "run_directory", return_value=directory),
                patch.object(locks, "validate_complete"),
                patch.object(locks, "specification", return_value=spec),
                patch.object(locks, "verify_reference"),
                patch.object(locks, "load_json", return_value=config),
            ):
                self.assertEqual(locks.validate_training(2111, "score_only"), config)
                broken = copy.deepcopy(config)
                broken["optimizer_steps"]["raw_eta"] = 1
                with (
                    patch.object(locks, "load_json", return_value=broken),
                    self.assertRaisesRegex(RuntimeError, "Adam"),
                ):
                    locks.validate_training(2111, "score_only")
                path.write_text(json.dumps(logs[0]) + "\n")
                with self.assertRaisesRegex(RuntimeError, "step"):
                    locks.validate_training(2111, "score_only")


if __name__ == "__main__":
    unittest.main()
