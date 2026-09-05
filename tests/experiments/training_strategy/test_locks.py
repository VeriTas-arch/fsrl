import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from fsrl.experiments.training_strategy import locks
from fsrl.experiments.training_strategy.protocol import (
    PROTOCOL_SHA256,
    load_specification,
    phase_for_step,
    training_config,
)


class TrainingLockTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_specification()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "seed-2108" / "matched_staged"
        self.directory.mkdir(parents=True)
        (self.directory / "net.pth").write_bytes(b"fixture")
        (self.directory / "local.pth").write_bytes(b"fixture")
        digest = hashlib.sha256()
        self.rows = []
        for step in range(1500):
            fingerprint = hashlib.sha256(str(step).encode()).hexdigest()
            digest.update(bytes.fromhex(fingerprint))
            self.rows.append(
                {
                    "step": step,
                    "phase": phase_for_step(self.spec, "matched_staged", step),
                    "batch_fingerprint": fingerprint,
                    "stream_fingerprint": digest.hexdigest(),
                }
            )
        self.metadata = {
            "seed": 2108,
            "condition": "matched_staged",
            "protocol_sha256": PROTOCOL_SHA256,
            "training": asdict(training_config(self.spec, 2108)),
            "episode_exposures": 48000,
            "stream_fingerprint": digest.hexdigest(),
            "backbone_updates": 1000,
            "local_updates": 500,
            "optimizer_parameter_steps": {
                "backbone.h2DA.weight": 1000,
                "backbone.h2v.weight": 0,
                "local.raw_gain": 500,
            },
            "stage_boundary_backbone": {"w": "unchanged"},
            "final_backbone": {"w": "unchanged"},
            "checkpoint": {},
        }
        self.manifest = {"lifecycle_state": "complete"}
        self.addCleanup(patch.stopall)
        patch.object(
            locks, "validate_run_manifest", return_value={"passed": True}
        ).start()
        patch.object(
            locks, "verify_reference", return_value=self.directory / "net.pth"
        ).start()

    def validate(self):
        (self.directory / "config.json").write_text(json.dumps(self.metadata))
        (self.directory / "run.json").write_text(json.dumps(self.manifest))
        (self.directory / "train_log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        return locks.validate_training_run(self.directory, self.spec)

    def test_every_registered_step_and_actual_optimizer_counter_are_checked(self):
        self.assertEqual(self.validate(), self.metadata)
        self.metadata["optimizer_parameter_steps"]["local.raw_gain"] = 499
        with self.assertRaisesRegex(RuntimeError, "Adam counter"):
            self.validate()

    def test_incomplete_run_cannot_be_reused(self):
        self.manifest["lifecycle_state"] = "running"
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            self.validate()

    def test_missing_step_wrong_phase_or_data_chain_blocks_admission(self):
        original = copy.deepcopy(self.rows)
        mutations = (
            ("step", 5),
            ("phase", "joint"),
            ("stream_fingerprint", "0" * 64),
            ("batch_fingerprint", "0" * 64),
        )
        for key, value in mutations:
            self.rows = copy.deepcopy(original)
            self.rows[1001][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate()
        self.rows = original[:-1]
        with self.assertRaisesRegex(RuntimeError, "every registered update"):
            self.validate()

    def test_wrong_seed_config_exposure_and_frozen_backbone_are_rejected(self):
        original = copy.deepcopy(self.metadata)
        for key, value in (
            ("seed", 2109),
            ("episode_exposures", 47968),
            ("final_backbone", {"w": "changed"}),
            ("training", {}),
        ):
            self.metadata = {**original, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate()

    def test_locked_artifact_set_cannot_omit_a_seed_or_condition(self):
        incomplete = {"source_lock": {}, "protocol_sha256": PROTOCOL_SHA256, "runs": {}}
        with (
            patch.object(locks, "validate_source_lock"),
            patch.object(locks, "git_text", return_value="witness"),
            patch.object(locks, "reference", return_value={}),
            patch.object(locks, "load_json", return_value=incomplete),
            self.assertRaisesRegex(RuntimeError, "all six"),
        ):
            locks.validate_artifact_lock()

    def test_source_lock_rejects_stale_smoke_source_set_and_profile(self):
        # Reuse only the shape of a development integrity record, never its pass status.
        smoke = {
            "passed": True,
            "seed": 910001,
            "liu_evaluated": False,
            "sources": [{"path": "fixture"}],
            "runtime": {
                "profile": {
                    "device": "cuda",
                    "compile": True,
                    "compile_fullgraph": True,
                    "compile_mode": "default",
                    "compile_backend": "inductor",
                    "cpu_threads": 1,
                    "blas_threads": 1,
                    "require_cuda": True,
                },
                "cuda_available": True,
                "compiler_threads": 1,
                "torch_intraop_threads": 1,
                "torch_interop_threads": 1,
                "blas_thread_limit": 1,
                "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "checks": {
                name: {"passed": True}
                for name in (
                    "P_T",
                    "logits",
                    "loss",
                    "updated_raw_gain",
                    "updated_backbone_h2DA.weight",
                    "gradient_0",
                    "gradient_1",
                    "gradient_2",
                    "gradient_3",
                )
            },
        }
        with patch.object(
            locks, "implementation_sources", return_value=smoke["sources"]
        ):
            locks._validate_smoke(smoke)
            broken = copy.deepcopy(smoke)
            broken["runtime"]["compiler_threads"] = 32
            with self.assertRaisesRegex(RuntimeError, "execution profile"):
                locks._validate_smoke(broken)
            broken = copy.deepcopy(smoke)
            del broken["checks"]["gradient_1"]
            with self.assertRaisesRegex(RuntimeError, "gradient/update"):
                locks._validate_smoke(broken)
        with (
            patch.object(locks, "implementation_sources", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "source set"),
        ):
            locks._validate_smoke(smoke)
