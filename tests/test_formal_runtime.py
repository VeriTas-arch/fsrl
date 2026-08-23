import copy
import json
import subprocess
import sys
import unittest

from fsrl.confirmation import (
    _formal_runtime_source_registration,
    _formal_training_source_registration,
    _validate_formal_runtime_record,
    _validate_formal_training_execution,
)
from fsrl.meta_train import COMPILED_TRAINING_EXECUTION


class FormalRuntimeTests(unittest.TestCase):
    def test_fresh_process_applies_single_thread_policy(self):
        command = (
            "from fsrl.formal_runtime import configure_formal_runtime; "
            "import json; print(json.dumps(configure_formal_runtime()))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["active"])
        self.assertEqual(result["cpu_thread_limit"], 1)
        self.assertEqual(result["torch_intraop_threads"], 1)
        self.assertEqual(result["torch_interop_threads"], 1)

        recorded_runtime = dict(result)
        recorded_runtime["cuda_available"] = True
        recorded_runtime["device"] = "cuda"
        record = {
            "execution_runtime": recorded_runtime,
            "execution_runtime_source": _formal_runtime_source_registration(),
        }
        _validate_formal_runtime_record(record)

        record["execution_runtime"]["torch_interop_threads"] = 2
        with self.assertRaisesRegex(RuntimeError, "bounded GPU runtime"):
            _validate_formal_runtime_record(record)

    def test_compiled_training_record_is_source_locked(self):
        record = {
            "training_execution": copy.deepcopy(COMPILED_TRAINING_EXECUTION),
            "training_execution_source": _formal_training_source_registration(),
        }
        _validate_formal_training_execution(record)

        record["training_execution"]["torch_compile"]["fullgraph"] = False
        with self.assertRaisesRegex(RuntimeError, "registered compiled training"):
            _validate_formal_training_execution(record)


if __name__ == "__main__":
    unittest.main()
