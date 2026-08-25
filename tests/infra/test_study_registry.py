import copy
import importlib.util
import re
import unittest
from pathlib import Path

from fsrl.infra.study_registry import (
    GENERATED_PATHS,
    MIGRATION_PATH,
    ROOT,
    SYNTHESIS_SNAPSHOT_MIGRATION_PATH,
    check_navigation,
    file_sha256,
    load_migration,
    load_migrations,
    load_registry,
    load_runtime_locator_migration,
    load_source_provenance,
    load_studies,
    load_synthesis,
    render_navigation,
    resolve_record,
    validate_registry,
    verify_source_lock,
)
from fsrl.workflows.frozen_evidence import (
    MANIFEST_PATH,
    load_json,
    verify_evidence_files,
)
from tools.provenance.index_source_provenance_v1 import run as check_source_provenance
from tools.provenance.rewrite_runtime_locators_v1 import audit as audit_runtime_locators


class StudyRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.synthesis = load_synthesis()
        cls.migration = load_migration()
        cls.migrations = load_migrations(cls.registry)
        cls.studies = load_studies(cls.registry)
        cls.validation = validate_registry(cls.registry, cls.synthesis, cls.migration)

    def test_registry_owns_the_complete_migrated_record(self):
        self.assertTrue(self.validation["passed"], self.validation["errors"])
        self.assertEqual(self.validation["studies"], 42)
        self.assertEqual(self.validation["chapters"], 9)
        self.assertEqual(self.validation["records"], 211)
        self.assertEqual(self.validation["migration_steps"], 231)
        self.assertEqual(self.validation["study_records"], 191)
        self.assertEqual(self.validation["synthesis_records"], 20)
        self.assertEqual(self.validation["retired_assets"], 2)
        self.assertEqual(self.validation["runtime_locator_rewrites"], 76)
        self.assertEqual(self.validation["source_provenance"], 127)

    def test_old_flat_roots_are_not_the_active_layout(self):
        for directory in (
            "docs",
            "benchmarks",
            "results",
            "research",
            "mainlines",
            "checkpoints",
            "output",
            "figures",
        ):
            self.assertFalse((ROOT / directory).exists(), directory)
        for record in self.migration["records"]:
            self.assertFalse((ROOT / record["legacy_path"]).exists())
            self.assertTrue(resolve_record(record["legacy_path"]).is_file())
        self.assertFalse((ROOT / "synthesis" / "records").exists())
        self.assertFalse((ROOT / "synthesis" / "frozen").exists())

    def test_active_python_defaults_do_not_recreate_legacy_output_root(self):
        legacy_root = re.compile(r"\bROOT\s*/\s*['\"]output['\"]")
        offenders = []
        for path in sorted((ROOT / "fsrl").rglob("*.py")):
            if legacy_root.search(path.read_text(encoding="utf-8")):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_runtime_locator_rewrite_is_provenance_locked(self):
        migration = load_runtime_locator_migration()
        self.assertEqual(migration["record_count"], 76)
        self.assertEqual(migration["replacement_count"], 784)
        validation = audit_runtime_locators()
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertEqual(validation["dependent_updates"], 3)

    def test_data_root_contains_only_external_inputs_and_its_contract(self):
        self.assertEqual(
            sorted(path.name for path in (ROOT / "data").iterdir()),
            ["README.md", "external"],
        )

    def test_migration_is_bijective_and_legacy_paths_resolve(self):
        records = [
            record for migration in self.migrations for record in migration["records"]
        ]
        legacy = [record["legacy_path"] for record in records]
        current = [record["path"] for record in records]
        self.assertEqual(len(legacy), len(set(legacy)))
        self.assertEqual(len(current), len(set(current)))
        self.assertTrue(MIGRATION_PATH.is_file())
        self.assertTrue(SYNTHESIS_SNAPSHOT_MIGRATION_PATH.is_file())
        for record in records:
            self.assertTrue(resolve_record(record["legacy_path"]).is_file())

    def test_historical_source_locks_resolve_through_git_provenance(self):
        provenance = load_source_provenance()
        sources = provenance["sources"]
        self.assertEqual(len(sources), 127)
        self.assertEqual(provenance["source_reference_occurrences"], 631)
        snapshot = ROOT / "synthesis" / "snapshots" / "reporting_v1"
        self.assertFalse((snapshot / "source").exists())
        self.assertFalse((snapshot / "source-blobs").exists())
        pairs = {(source["path"], source["sha256"]) for source in sources}
        self.assertEqual(len(pairs), len(sources))

        historical = next(
            source
            for source in sources
            if not (ROOT / source["path"]).is_file()
            or file_sha256(ROOT / source["path"]) != source["sha256"]
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

    def test_retired_development_checkpoint_is_git_backed_not_worktree_state(self):
        assets = self.studies["development_qualification"]["retired_assets"]
        self.assertEqual(len(assets), 2)
        for asset in assets:
            self.assertFalse((ROOT / asset["path"]).exists())
            self.assertEqual(asset["source_ref"], "refs/tags/liu-mainline-v1")

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

    def test_active_human_docs_have_live_local_links_and_python_modules(self):
        historical_roots = (
            ROOT / "studies",
            ROOT / "synthesis" / "snapshots" / "reporting_v1",
        )
        failures = []
        for path in sorted(ROOT.rglob("*.md")):
            if ".git" in path.parts or ".cache" in path.parts:
                continue
            if historical_roots[0] in path.parents and "records" in path.parts:
                continue
            if historical_roots[1] in path.parents:
                continue
            content = path.read_text(encoding="utf-8")
            for raw_target in re.findall(r"(?<!!)\[[^]]*\]\(([^)]+)\)", content):
                target = raw_target.strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    failures.append(
                        f"missing link: {path.relative_to(ROOT)} -> {raw_target}"
                    )
            for module in re.findall(
                r"python\s+-m\s+(fsrl(?:\.[A-Za-z0-9_]+)+)", content
            ):
                if importlib.util.find_spec(module) is None:
                    failures.append(
                        f"missing module: {path.relative_to(ROOT)} -> {module}"
                    )
        self.assertEqual(failures, [])

    def test_frozen_mainline_evidence_hashes_survive_the_path_migration(self):
        manifest = load_json(MANIFEST_PATH)
        evidence = verify_evidence_files(manifest)
        self.assertTrue(evidence["passed"], evidence)
