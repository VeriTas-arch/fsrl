import copy
import re
import unittest
from pathlib import Path

from fsrl.liu_mainline import MANIFEST_PATH, load_json, verify_evidence_files
from fsrl.study_registry import (
    GENERATED_PATHS,
    MIGRATION_PATH,
    ROOT,
    check_navigation,
    file_sha256,
    load_migration,
    load_registry,
    load_source_provenance,
    load_studies,
    load_synthesis,
    render_navigation,
    resolve_record,
    validate_registry,
    verify_source_lock,
)
from tools.provenance.index_source_provenance_v1 import run as check_source_provenance


class StudyRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.synthesis = load_synthesis()
        cls.migration = load_migration()
        cls.studies = load_studies(cls.registry)
        cls.validation = validate_registry(cls.registry, cls.synthesis, cls.migration)

    def test_registry_owns_the_complete_migrated_record(self):
        self.assertTrue(self.validation["passed"], self.validation["errors"])
        self.assertEqual(self.validation["studies"], 42)
        self.assertEqual(self.validation["chapters"], 9)
        self.assertEqual(self.validation["records"], 211)
        self.assertEqual(self.validation["study_records"], 191)
        self.assertEqual(self.validation["synthesis_records"], 20)
        self.assertEqual(self.validation["source_provenance"], 127)

    def test_old_flat_roots_are_not_the_active_layout(self):
        for directory in ("docs", "benchmarks", "results", "research", "mainlines"):
            self.assertFalse((ROOT / directory).exists(), directory)
        for record in self.migration["records"]:
            self.assertFalse((ROOT / record["legacy_path"]).exists())
            self.assertTrue((ROOT / record["path"]).is_file())

    def test_migration_is_bijective_and_legacy_paths_resolve(self):
        legacy = [record["legacy_path"] for record in self.migration["records"]]
        current = [record["path"] for record in self.migration["records"]]
        self.assertEqual(len(legacy), len(set(legacy)))
        self.assertEqual(len(current), len(set(current)))
        self.assertTrue(MIGRATION_PATH.is_file())
        for record in self.migration["records"]:
            self.assertEqual(
                resolve_record(record["legacy_path"]), ROOT / record["path"]
            )

    def test_historical_source_locks_resolve_through_git_provenance(self):
        provenance = load_source_provenance()
        sources = provenance["sources"]
        self.assertEqual(len(sources), 127)
        self.assertEqual(provenance["source_reference_occurrences"], 631)
        self.assertFalse((ROOT / "synthesis" / "frozen" / "source").exists())
        self.assertFalse((ROOT / "synthesis" / "frozen" / "source-blobs").exists())
        pairs = {(source["path"], source["sha256"]) for source in sources}
        self.assertEqual(len(pairs), len(sources))

        historical = next(
            source
            for source in sources
            if file_sha256(ROOT / source["path"]) != source["sha256"]
        )
        self.assertEqual(resolve_record(historical["path"]), ROOT / historical["path"])
        verification = verify_source_lock(historical["path"], historical["sha256"])
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["observed_sha256"], historical["sha256"])

    def test_source_provenance_rejects_a_tampered_git_registration(self):
        altered = copy.deepcopy(load_source_provenance())
        altered["sources"][0]["sha256"] = "0" * 64
        validation = validate_registry(
            self.registry, self.synthesis, self.migration, altered
        )
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any(
                "source provenance verification failed" in error
                for error in validation["errors"]
            )
        )

    def test_source_provenance_index_matches_all_structured_evidence(self):
        result = check_source_provenance(apply=False)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["source_versions"], 127)
        self.assertEqual(result["source_reference_occurrences"], 631)

    def test_migration_provenance_must_match_the_local_authority(self):
        altered = copy.deepcopy(self.migration)
        altered["records"][0]["sha256"] = "0" * 64
        validation = validate_registry(self.registry, self.synthesis, altered)
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any(
                "migration provenance disagrees" in error
                for error in validation["errors"]
            )
        )

    def test_failed_executions_remain_visibly_noncanonical(self):
        attempts = [
            record
            for record in self.migration["records"]
            if "noninterpretable" in Path(record["legacy_path"]).name
        ]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            {record["role"] for record in attempts}, {"noninterpretable_attempt"}
        )
        topology = next(
            record
            for record in self.migration["records"]
            if record["legacy_path"]
            == "results/liu_support_topology_transport_v1.attempt1.json"
        )
        self.assertEqual(topology["role"], "superseded_repair_source")

    def test_generated_human_views_are_current_and_deterministic(self):
        check = check_navigation()
        self.assertTrue(check["passed"], check)
        self.assertEqual(check["stale_generated_files"], [])
        self.assertEqual(check["unexpected_study_readmes"], [])
        first = render_navigation(self.registry, self.synthesis)
        second = render_navigation(self.registry, self.synthesis)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(GENERATED_PATHS) + 42)

    def test_every_study_has_a_human_capsule_and_local_authority(self):
        for study_id, study in self.studies.items():
            directory = ROOT / "studies" / study_id
            self.assertTrue((directory / "study.toml").is_file())
            content = (directory / "README.md").read_text(encoding="utf-8")
            self.assertIn(study["question"], content)
            self.assertIn(study["finding"], content)
            self.assertIn(study["boundary"], content)
            self.assertEqual(study["review_state"], "indexed")

    def test_every_generated_local_link_resolves(self):
        rendered = render_navigation(self.registry, self.synthesis)
        for relative, content in rendered.items():
            source = ROOT / relative
            for raw_target in re.findall(r"\[[^]]+\]\(([^)]+)\)", content):
                target = raw_target.strip("<>").split("#", 1)[0]
                if not target or "://" in target:
                    continue
                resolved = (source.parent / target).resolve()
                self.assertTrue(resolved.exists(), f"{relative}: {raw_target}")

    def test_frozen_mainline_evidence_hashes_survive_the_path_migration(self):
        manifest = load_json(MANIFEST_PATH)
        evidence = verify_evidence_files(manifest)
        self.assertTrue(evidence["passed"], evidence)


if __name__ == "__main__":
    unittest.main()
