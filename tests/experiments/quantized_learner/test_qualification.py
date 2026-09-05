import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.quantized_learner import qualification
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.quantized_learner.protocol import (
    make_model,
    resolved_specification,
    specification,
)
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra import formal_runtime
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.run_manifest import ProspectiveRun


class QualificationTests(unittest.TestCase):
    def test_cpu_transcript_allows_run_manifest_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "qualification"
            with ProspectiveRun.start(
                directory,
                workflow_id="fixture",
                execution_id="fixture",
                producer={},
                resolved_config={},
            ):
                (directory / qualification.CPU_TEST_LOG).write_text(
                    "fixture tests passed\n"
                )
            result = validate_run_manifest(directory / "run.json")
            self.assertTrue(result["passed"], result)

    def test_scratch_optimizer_checks_every_state_and_parameter(self):
        spec = resolved_specification()
        rng = np.random.default_rng(973001)
        batch = generic_batch(sample_episodes(task_generator(), rng, 3))
        batch, _ = encode_batch(
            batch,
            "persistent",
            rng.random(batch.arrays["signed"].shape),
            specification()["encoding"]["codebook"],
        )
        eager = make_model(spec)
        candidate = copy.deepcopy(eager)
        result = qualification.parity_checks(
            eager, candidate, candidate, batch, spec, {"atol": 0, "rtol": 0}
        )
        for step in range(3):
            for name in ("raw_eta", "raw_global_gain"):
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertIn(f"step-{step}/adam-{name}-{field}", result)
                self.assertTrue(result[f"step-{step}/actual-update-{name}"]["passed"])
            for index in range(5):
                self.assertIn(f"step-{step}/output-{index}", result)
        self.assertTrue(all(row["passed"] for row in result.values()))
        reference = qualification.reference_checks(
            batch, spec, {"atol": 1e-9, "rtol": 1e-7}
        )
        self.assertTrue(all(row["passed"] for row in reference.values()))

    def test_comparison_does_not_accept_nan_or_broadcasted_shape(self):
        for first, second in (([np.nan], [np.nan]), ([1], [[1]]), ([1.0], [1.1])):
            with self.subTest(first=first):
                self.assertFalse(
                    qualification.comparison(first, second, atol=0, rtol=0)["passed"]
                )
        self.assertTrue(qualification.comparison([], [], atol=0, rtol=0)["passed"])

    def test_qualification_requires_committed_pushed_source_before_runtime(self):
        with (
            patch.object(
                qualification, "require_pushed_clean", side_effect=RuntimeError("dirty")
            ),
            patch.object(qualification, "runtime") as runtime,
            self.assertRaisesRegex(RuntimeError, "dirty"),
        ):
            qualification.qualify(1)
        runtime.assert_not_called()
        with self.assertRaises(ValueError):
            qualification.qualify(0)

    def test_registered_dispatch_configures_bounded_runtime_before_qualification(self):
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch(
                "fsrl.experiments.quantized_learner.__main__.main", return_value=37
            ) as run,
        ):
            result = formal_runtime.main(
                ["quantized-relational-learner", "qualify", "--attempt", "2"]
            )
        configure.assert_called_once_with()
        run.assert_called_once_with(["qualify", "--attempt", "2"])
        self.assertEqual(result, 37)


if __name__ == "__main__":
    unittest.main()
