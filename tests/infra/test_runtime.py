import json
import subprocess
import sys
import unittest

from fsrl.infra.runtime import ExecutionProfile, configure_runtime


class RuntimeProfileTests(unittest.TestCase):
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
