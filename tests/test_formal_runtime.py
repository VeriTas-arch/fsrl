import copy
import json
import subprocess
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from fsrl import formal_runtime
from fsrl.confirmation import (
    _formal_runtime_source_registration,
    _formal_training_source_registration,
    _validate_formal_runtime_record,
    _validate_formal_training_execution,
)
from fsrl.meta_train import COMPILED_TRAINING_EXECUTION


class FormalRuntimeTests(unittest.TestCase):
    def test_support_topology_transport_dispatch_uses_registered_entry_point(self):
        module_name = "fsrl.support_topology_transport"
        module = ModuleType(module_name)
        workflow = Mock(return_value=41)
        module.main = workflow
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch.dict(sys.modules, {module_name: module}),
        ):
            result = formal_runtime.main(
                ["liu-support-topology-transport", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 41)

    def test_human_metric_constructive_comparator_dispatch_uses_registered_entry_point(
        self,
    ):
        module_name = "fsrl.human_metric_constructive_comparator"
        module = ModuleType(module_name)
        workflow = Mock(return_value=37)
        module.main = workflow
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch.dict(sys.modules, {module_name: module}),
        ):
            result = formal_runtime.main(
                ["human-metric-constructive-comparator", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 37)

    def test_global_policy_comparator_adequacy_dispatch_uses_registered_entry_point(
        self,
    ):
        module_name = "fsrl.global_policy_comparator_adequacy"
        module = ModuleType(module_name)
        workflow = Mock(return_value=31)
        module.main = workflow
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch.dict(sys.modules, {module_name: module}),
        ):
            result = formal_runtime.main(
                ["global-policy-comparator-adequacy", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 31)

    def test_global_policy_allocation_audit_dispatch_uses_registered_entry_point(self):
        module_name = "fsrl.global_policy_allocation_audit"
        module = ModuleType(module_name)
        workflow = Mock(return_value=29)
        module.main = workflow
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch.dict(sys.modules, {module_name: module}),
        ):
            result = formal_runtime.main(
                ["global-policy-allocation-audit", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 29)

    def test_field_fingerprint_replication_dispatch_uses_registered_entry_point(self):
        module_name = "fsrl.global_policy_field_fingerprint_replication"
        module = ModuleType(module_name)
        workflow = Mock(return_value=23)
        module.main = workflow
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch.dict(sys.modules, {module_name: module}),
        ):
            result = formal_runtime.main(
                ["global-policy-field-fingerprint-replication", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 23)

    def test_field_reassembly_dispatch_uses_registered_entry_point(self):
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch(
                "fsrl.global_policy_field_reassembly.main", return_value=19
            ) as workflow,
        ):
            result = formal_runtime.main(
                ["global-policy-field-reassembly", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 19)

    def test_amplitude_provenance_dispatch_uses_registered_entry_point(self):
        with (
            patch.object(formal_runtime, "configure_formal_runtime") as configure,
            patch(
                "fsrl.global_policy_amplitude_provenance.main", return_value=17
            ) as workflow,
        ):
            result = formal_runtime.main(
                ["global-policy-amplitude-provenance", "--sentinel"]
            )
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 17)

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
        recorded_runtime["cuda_version"] = (
            recorded_runtime.get("cuda_version") or "test-cuda"
        )
        recorded_runtime["device"] = "cuda"
        recorded_runtime["device_name"] = (
            recorded_runtime.get("device_name") or "test-gpu"
        )
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
