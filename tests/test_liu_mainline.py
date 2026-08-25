import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fsrl.liu_mainline import (
    ARTIFACTS_PATH,
    EXPECTED_DAG,
    MANIFEST_PATH,
    REPORT_VIEW_PATH,
    canonical_manifest_payload_sha256,
    doctor_mainline,
    load_json,
    restore_test_artifacts,
    summarize_mainline,
    validate_manifest_structure,
    verify_artifact_bundle,
    verify_evidence_files,
    verify_freeze_attestation,
    verify_historical_executions,
    verify_mainline,
    verify_replay_contracts,
    verify_report_view,
)
from fsrl.study_registry import resolve_record

ROOT = Path(__file__).resolve().parents[1]


class LiuMainlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST_PATH)

    def test_claim_dag_is_frozen_and_non_linear(self):
        validation = validate_manifest_structure(self.manifest)
        self.assertTrue(validation["passed"])
        self.assertEqual(
            {
                name: node["depends_on"]
                for name, node in self.manifest["claim_nodes"].items()
            },
            EXPECTED_DAG,
        )
        self.assertEqual(
            self.manifest["claim_nodes"]["global_reassembly"]["depends_on"],
            ["task_fidelity"],
        )
        self.assertEqual(
            self.manifest["claim_nodes"]["local_direct_fidelity"]["depends_on"],
            ["task_fidelity"],
        )

    def test_manifest_lifecycle_attestation_is_valid(self):
        freeze = self.manifest["freeze"]
        self.assertEqual(freeze["freeze_ref"], "refs/tags/liu-mainline-v1")
        self.assertRegex(
            canonical_manifest_payload_sha256(self.manifest), r"^[0-9a-f]{64}$"
        )
        if self.manifest["status"] == "draft":
            self.assertIsNone(freeze["validated_candidate_commit"])
            self.assertIsNone(freeze["canonical_payload_sha256"])
        else:
            self.assertRegex(freeze["validated_candidate_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(
                freeze["canonical_payload_sha256"],
                canonical_manifest_payload_sha256(self.manifest),
            )
        self.assertTrue(verify_freeze_attestation(self.manifest)["passed"])

    def test_frozen_manifest_requires_its_canonical_payload_hash(self):
        frozen = json.loads(json.dumps(self.manifest))
        frozen["status"] = "frozen"
        frozen["freeze"]["validated_candidate_commit"] = "a" * 40
        frozen["freeze"]["canonical_payload_sha256"] = (
            canonical_manifest_payload_sha256(frozen)
        )
        self.assertTrue(validate_manifest_structure(frozen)["passed"])
        frozen["scope"] += " changed"
        with self.assertRaisesRegex(RuntimeError, "canonical payload hash mismatch"):
            validate_manifest_structure(frozen)

    def test_research_lineage_is_not_claim_dependency(self):
        serialized = json.dumps(self.manifest)
        self.assertNotIn('"active_lock"', serialized)
        for node in self.manifest["claim_nodes"].values():
            self.assertTrue(set(node["motivated_by"]).isdisjoint(node["depends_on"]))
        task_dependencies = json.dumps(
            self.manifest["claim_nodes"]["task_fidelity"], sort_keys=True
        )
        for name in ("Miconi", "Lippl", "Nelli"):
            self.assertNotIn(name, task_dependencies)
            self.assertTrue(
                any(key.startswith(name) for key in self.manifest["reference_context"])
            )

    def test_evidence_and_historical_execution_hashes_pass(self):
        evidence = verify_evidence_files(self.manifest)
        historical = verify_historical_executions(self.manifest)
        self.assertTrue(evidence["passed"])
        self.assertGreaterEqual(len(evidence["checks"]), 54)
        self.assertTrue(historical["passed"])
        self.assertEqual(len(historical["executions"]), 8)
        for execution in historical["executions"]:
            self.assertTrue(execution["historical_files"])
            self.assertTrue(execution["source_checks"])

    def test_artifact_bundle_is_complete_and_content_addressed(self):
        artifacts = load_json(ARTIFACTS_PATH)
        validation = verify_artifact_bundle(artifacts)
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["members"], 33)
        self.assertEqual(validation["uncompressed_member_bytes"], 4093182)
        bundle = artifacts["bundle"]
        self.assertIn(bundle["sha256"], bundle["path"])
        self.assertEqual(bundle["storage_backend"], "repository_bundle")
        logical_names = [member["logical_name"] for member in artifacts["members"]]
        self.assertEqual(len(logical_names), len(set(logical_names)))

    def test_cpu_test_artifact_profile_is_explicit_and_hash_checked(self):
        restored = restore_test_artifacts()
        self.assertTrue(restored["passed"])
        self.assertEqual(restored["profile"], "cpu_test_suite")
        self.assertEqual(len(restored["restored_artifacts"]), 6)

    def test_exact_and_semantic_replay_contracts_are_distinct(self):
        validation = verify_replay_contracts(self.manifest)
        self.assertTrue(validation["passed"])
        self.assertEqual(len(validation["records"]), 8)
        for record in self.manifest["execution_records"].values():
            policy = record["replay_policy"]
            self.assertIn("expected_sha256", policy["exact"])
            self.assertTrue(policy["semantic"]["assertions"])

    def test_report_view_has_four_figures_and_resolved_pointers(self):
        view = load_json(REPORT_VIEW_PATH)
        validation = verify_report_view(self.manifest, view)
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["figures"], 4)
        self.assertEqual(validation["metrics"], 26)
        self.assertEqual(
            [figure["id"] for figure in view["figures"]], view["table_order"]
        )

    def test_summarize_is_deterministic_and_preserves_provenance(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            first_dir = Path(first) / "summary"
            second_dir = Path(second) / "summary"
            first_result = summarize_mainline(first_dir)
            summarize_mainline(second_dir)
            self.assertEqual(first_result["figures"], 4)
            self.assertEqual(first_result["metrics"], 26)
            first_files = sorted(path for path in first_dir.iterdir())
            second_files = sorted(path for path in second_dir.iterdir())
            self.assertEqual(
                [path.name for path in first_files],
                [path.name for path in second_files],
            )
            for first_path, second_path in zip(first_files, second_files, strict=True):
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            summary = json.loads((first_dir / "liu_mainline_summary.json").read_text())
            for figure in summary["figures"]:
                for metric in figure["metrics"]:
                    self.assertTrue(metric["source_sha256"])
                    self.assertTrue(metric["json_pointer"].startswith("/"))

    def test_summarize_process_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary"
            code = (
                "import sys; "
                "from pathlib import Path; "
                "from fsrl.liu_mainline import summarize_mainline; "
                f"summarize_mainline(Path({str(output)!r})); "
                "assert 'torch' not in sys.modules"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_summarize_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary"
            summarize_mainline(output)
            with self.assertRaisesRegex(RuntimeError, "refuses to overwrite"):
                summarize_mainline(output)

    def test_replay_requires_one_explicit_registered_stage(self):
        stages = self.manifest["replay_stages"]
        self.assertNotIn("all", stages)
        self.assertEqual(stages["global_reassembly"], "dual_access_confirmation")
        self.assertEqual(stages["local_direct_fidelity"], "dual_access_confirmation")
        completed = subprocess.run(
            [sys.executable, "-m", "fsrl.liu_mainline", "replay", "--all"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--stage", completed.stderr)

    def test_doctor_keeps_cpu_readiness_separate_from_gpu_visibility(self):
        result = doctor_mainline()
        self.assertTrue(result["core_ready"])
        self.assertFalse(result["gpu_required"])
        self.assertTrue(result["passed"])

    def test_presentation_v2_defines_exact_L_and_a_equivalence(self):
        presentation = resolve_record("docs/liu_presentation_package_v2.md").read_text()
        self.assertIn("L_{t+1}=L_t+s_t^L k_{r_t}", presentation)
        self.assertIn("a_{t+1}=a_t+s_t^L e_{r_t}", presentation)
        self.assertIn("L_T=\\sum_r a_{T,r}k_r", presentation)
        self.assertIn("(K a_T)_q", presentation)
        self.assertIn("K_{qr}=k_q^\\top k_r", presentation)

    def test_complete_overlay_verification_passes(self):
        validation = verify_mainline()
        self.assertTrue(validation["passed"])


if __name__ == "__main__":
    unittest.main()
