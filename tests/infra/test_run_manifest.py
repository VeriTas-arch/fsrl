import json
import tempfile
import unittest
from pathlib import Path

from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun


class ProspectiveRunTests(unittest.TestCase):
    def _start(self, output_dir: Path) -> ProspectiveRun:
        return ProspectiveRun.start(
            output_dir,
            workflow_id="test-workflow",
            execution_id="execution-1",
            producer={"module": "tests.infra.test_run_manifest"},
            resolved_config={"seed": 7},
        )

    def test_complete_run_records_every_owned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution-1"
            run = self._start(output_dir)
            write_json_exclusive(output_dir / "metrics.json", {"score": 1.0})
            run.complete()

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["lifecycle_state"], "complete")
            self.assertEqual(manifest["file_count"], 1)
            self.assertEqual(manifest["files"][0]["path"], "metrics.json")
            self.assertTrue(validate_run_manifest(run.manifest_path)["passed"])

    def test_existing_execution_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution-1"
            self._start(output_dir)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self._start(output_dir)

    def test_context_records_failure_and_preserves_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution-1"
            with (
                self.assertRaisesRegex(RuntimeError, "synthetic failure"),
                self._start(output_dir),
            ):
                write_json_exclusive(output_dir / "partial.json", {"step": 1})
                raise RuntimeError("synthetic failure")

            manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["lifecycle_state"], "failed")
            self.assertEqual(manifest["error"]["type"], "RuntimeError")
            self.assertEqual(manifest["files"][0]["path"], "partial.json")
            self.assertTrue(validate_run_manifest(output_dir / "run.json")["passed"])

    def test_unknown_lifecycle_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._start(Path(directory) / "execution-1")
            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            manifest["lifecycle_state"] = "mystery"
            run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_run_manifest(run.manifest_path)
        self.assertFalse(result["passed"])
        self.assertIn(
            "prospective run manifest has invalid lifecycle_state", result["errors"]
        )

    def test_complete_manifest_rejects_undeclared_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution-1"
            with self._start(output_dir):
                write_json_exclusive(output_dir / "metrics.json", {"score": 1.0})
            (output_dir / "late-file.txt").write_text("unowned\n", encoding="utf-8")

            result = validate_run_manifest(output_dir / "run.json")

            self.assertFalse(result["passed"])
            self.assertIn("run file is not declared: late-file.txt", result["errors"])


if __name__ == "__main__":
    unittest.main()
