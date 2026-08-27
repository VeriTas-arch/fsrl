import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from fsrl.infra.runtime import (
    ExecutionProfile,
    begin_compiled_iteration,
    configure_runtime,
    uses_cuda_graphs,
)


class RuntimeProfileTests(unittest.TestCase):
    def test_only_inductor_cuda_graph_profiles_mark_iteration_boundaries(self):
        cuda_graph_profile = ExecutionProfile(
            device="cuda",
            compile_mode="reduce-overhead",
            require_cuda=False,
        )
        self.assertTrue(uses_cuda_graphs(cuda_graph_profile))
        with patch("torch.compiler.cudagraph_mark_step_begin") as marker:
            begin_compiled_iteration(cuda_graph_profile)
            marker.assert_called_once_with()

        for profile in (
            ExecutionProfile(
                device="cuda",
                compile_mode="default",
                require_cuda=False,
            ),
            ExecutionProfile(
                device="cuda",
                compile_mode="max-autotune-no-cudagraphs",
                require_cuda=False,
            ),
            ExecutionProfile(
                device="cpu",
                compile_mode="reduce-overhead",
                require_cuda=False,
            ),
            ExecutionProfile(
                device="cuda",
                compile=False,
                compile_mode="reduce-overhead",
                require_cuda=False,
            ),
        ):
            with self.subTest(profile=profile):
                self.assertFalse(uses_cuda_graphs(profile))
                with patch("torch.compiler.cudagraph_mark_step_begin") as marker:
                    begin_compiled_iteration(profile)
                    marker.assert_not_called()

    def test_invalid_thread_limits_fail_before_configuration(self):
        with self.assertRaisesRegex(ValueError, "cpu_threads"):
            configure_runtime(
                ExecutionProfile(
                    device="cpu",
                    cpu_threads=0,
                    compile=False,
                    require_cuda=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "blas_threads"):
            configure_runtime(
                ExecutionProfile(
                    device="cpu",
                    blas_threads=0,
                    compile=False,
                    require_cuda=False,
                )
            )

    def test_fresh_process_bounds_loaded_blas_and_records_runtime_policy(self):
        code = """
import json
import numpy as np
from fsrl.infra.runtime import ExecutionProfile, configure_runtime

np.ones((64, 64)) @ np.ones((64, 64))
profile = ExecutionProfile(
    device="cpu",
    cpu_threads=1,
    blas_threads=1,
    compile=False,
    require_cuda=False,
)
print(json.dumps(configure_runtime(profile)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["execution_schema_version"], 2)
        self.assertEqual(result["profile"]["blas_threads"], 1)
        self.assertEqual(result["blas_thread_limit"], 1)
        self.assertTrue(result["blas_threadpools"])
        self.assertTrue(
            all(pool["num_threads"] == 1 for pool in result["blas_threadpools"])
        )
        self.assertIn(result["float32_matmul_precision"], {"highest", "high", "medium"})
        self.assertIsInstance(result["deterministic_algorithms"], bool)

    def test_process_rejects_incompatible_interop_reconfiguration(self):
        code = """
import json
import torch
from fsrl.infra.runtime import ExecutionProfile, configure_runtime

first = ExecutionProfile(
    device="cpu",
    cpu_threads=1,
    blas_threads=1,
    compile=False,
    require_cuda=False,
)
second = ExecutionProfile(
    device="cpu",
    cpu_threads=2,
    blas_threads=2,
    compile=False,
    require_cuda=False,
)
configure_runtime(first)
try:
    configure_runtime(second)
except RuntimeError as error:
    print(json.dumps({
        "error": str(error),
        "interop_threads": torch.get_num_interop_threads(),
    }))
else:
    raise AssertionError("incompatible runtime reconfiguration was accepted")
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertIn("process-global", result["error"])
        self.assertEqual(result["interop_threads"], 1)

    def test_late_interop_configuration_has_a_clear_error(self):
        code = """
import json
import torch
from fsrl.infra.runtime import ExecutionProfile, configure_runtime

torch.set_num_interop_threads(1)
profile = ExecutionProfile(
    device="cpu",
    cpu_threads=2,
    blas_threads=1,
    compile=False,
    require_cuda=False,
)
try:
    configure_runtime(profile)
except RuntimeError as error:
    print(json.dumps({"error": str(error)}))
else:
    raise AssertionError("late inter-op configuration was accepted")
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertIn("before parallel work starts", result["error"])
