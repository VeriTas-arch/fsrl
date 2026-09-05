import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fsrl.infra import study_registry
from tools.provenance.index_source_provenance_v1 import _owner_record_paths


class NativeRecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "studies" / "prospective" / "records" / "contract.json"
        self.path.parent.mkdir(parents=True)
        payload = b'{"registration_status":"prospective"}\n'
        self.path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.record = {
            "path": "records/contract.json",
            "legacy_path": "studies/prospective/records/contract.json",
            "origin": "native",
            "role": "registered_contract",
            "sha256": digest,
            "bytes": len(payload),
            "source_ref": f"sha256:{digest}",
        }

    def validate_record(self, record):
        errors = []
        with (
            patch.object(study_registry, "ROOT", self.root),
            patch.object(study_registry, "STUDIES_ROOT", self.root / "studies"),
        ):
            pair = study_registry._validate_record(
                owner_kind="study",
                owner={"id": "prospective"},
                record=record,
                errors=errors,
            )
        return pair, errors

    def test_native_record_validates_its_current_bytes_without_a_fake_migration(self):
        pair, errors = self.validate_record(self.record)
        self.assertEqual(errors, [])
        self.assertEqual(pair, (self.record["legacy_path"], self.record["legacy_path"]))

    def test_native_record_rejects_alias_origin_source_and_byte_tampering(self):
        for field, value, message in (
            ("legacy_path", "benchmarks/invented.json", "invent a legacy alias"),
            ("origin", "ignored", "unknown record origin"),
            ("source_ref", "refs/heads/dev", "content source_ref"),
            ("sha256", "invalid", "invalid SHA-256"),
            ("bytes", 1, "byte count changed"),
        ):
            with self.subTest(field=field):
                _, errors = self.validate_record({**self.record, field: value})
                self.assertTrue(any(message in error for error in errors), errors)
        self.path.write_bytes(b"tampered")
        _, errors = self.validate_record(self.record)
        self.assertTrue(any("hash changed" in error for error in errors), errors)

    def test_native_label_cannot_remove_a_historical_migration_obligation(self):
        registry = study_registry.load_registry()
        studies = copy.deepcopy(study_registry.load_studies(registry))
        record = studies["task_fidelity"]["records"][0]
        record["origin"] = "native"
        record["source_ref"] = f"sha256:{record['sha256']}"
        chapters = {chapter["id"]: chapter for chapter in registry["chapters"]}
        with patch.object(
            study_registry,
            "_load_and_validate_studies",
            return_value=(chapters, studies),
        ):
            validation = study_registry.validate_registry(registry)
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any("migration has unowned records" in e for e in validation["errors"]),
            validation["errors"],
        )

    def test_v1_source_inventory_excludes_only_explicit_native_records(self):
        manifest = self.path.parent.parent / "study.toml"
        manifest.write_text(
            '[[records]]\npath = "records/contract.json"\norigin = "native"\n',
            encoding="utf-8",
        )
        historical, native = _owner_record_paths(manifest)
        self.assertEqual(historical, [])
        self.assertEqual(native, {self.path, manifest})
        manifest.write_text(
            '[[records]]\npath = "records/contract.json"\norigin = "native"\n'
            '[[records]]\npath = "records/old.json"\n',
            encoding="utf-8",
        )
        historical, native = _owner_record_paths(manifest)
        self.assertEqual(historical, [self.path.parent / "old.json"])
        self.assertEqual(native, {self.path})
