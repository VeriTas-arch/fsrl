import unittest
from pathlib import Path

from fsrl.infra.study_registry import (
    canonical_file_sha256,
    historical_file_registration,
    historical_registered_file_sha256,
    materialized_file_registration,
    materialized_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT


class RecordIdentityBoundaryTests(unittest.TestCase):
    def test_locator_rewrite_exposes_distinct_historical_and_current_identities(self):
        path = resolve_record("benchmarks/assembly_diagnostics_v1.json")
        historical = historical_file_registration(path)
        current = materialized_file_registration(path)
        self.assertEqual(
            current["path"],
            "studies/assembly_diagnostics/records/benchmarks/"
            "assembly_diagnostics_v1.json",
        )
        self.assertEqual(historical["path"], "benchmarks/assembly_diagnostics_v1.json")
        self.assertNotEqual(historical["sha256"], current["sha256"])
        self.assertEqual(current["sha256"], materialized_file_sha256(path))
        self.assertEqual(historical["sha256"], historical_registered_file_sha256(path))

    def test_compatibility_alias_is_explicitly_historical(self):
        path = resolve_record("benchmarks/assembly_diagnostics_v1.json")
        self.assertEqual(
            canonical_file_sha256(path), historical_registered_file_sha256(path)
        )

    def test_unrewritten_record_has_same_bytes_under_both_identities(self):
        path = resolve_record("docs/liu_presentation_package_v2.md")
        self.assertEqual(
            materialized_file_sha256(path), historical_registered_file_sha256(path)
        )

    def test_current_registration_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            materialized_file_registration(Path("/tmp/non-repository-record.json"))

    def test_current_registration_uses_repository_root_not_caller_cwd(self):
        path = REPO_ROOT / "studies" / "registry.toml"
        registration = materialized_file_registration(path)
        self.assertEqual(registration["path"], "studies/registry.toml")
