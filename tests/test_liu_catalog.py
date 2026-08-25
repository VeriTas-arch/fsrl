import re
import unittest
from pathlib import Path

from fsrl.liu_catalog import (
    GENERATED_PATHS,
    ROOT,
    check_indexes,
    discover_inventory,
    file_role,
    load_catalog,
    render_indexes,
    validate_catalog,
)
from fsrl.liu_mainline import MANIFEST_PATH, load_json, verify_evidence_files


class LiuCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.validation = validate_catalog(cls.catalog)

    def test_catalog_uniquely_covers_the_historical_inventory(self):
        self.assertTrue(self.validation["passed"], self.validation["errors"])
        self.assertEqual(self.validation["inventory_files"], 203)
        self.assertEqual(self.validation["studies"], 43)
        self.assertEqual(self.validation["chapters"], 9)
        self.assertEqual(self.validation["unassigned"], [])
        self.assertEqual(self.validation["ambiguous"], {})

    def test_failed_executions_are_never_classified_as_frozen_results(self):
        attempts = [
            path
            for path in discover_inventory()
            if path.parts[0] == "results" and "noninterpretable" in path.name
        ]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            {file_role(path) for path in attempts}, {"noninterpretable_attempt"}
        )
        topology_attempt = Path(
            "results/liu_support_topology_transport_v1.attempt1.json"
        )
        self.assertEqual(file_role(topology_attempt), "superseded_repair_source")

    def test_generated_human_views_are_current_and_deterministic(self):
        check = check_indexes()
        self.assertTrue(check["passed"], check)
        self.assertEqual(check["stale_generated_files"], [])
        self.assertEqual(check["unexpected_study_capsules"], [])
        first = render_indexes(self.catalog, self.validation)
        second = render_indexes(self.catalog, self.validation)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(GENERATED_PATHS) + 43)

    def test_every_study_has_a_physical_human_capsule(self):
        for study in self.catalog["studies"]:
            capsule = ROOT / "research" / "liu" / "studies" / study["id"] / "README.md"
            self.assertTrue(capsule.is_file(), study["id"])
            content = capsule.read_text(encoding="utf-8")
            self.assertIn(study["question"], content)
            self.assertIn(study["finding"], content)
            self.assertIn(study["boundary"], content)

    def test_every_generated_local_link_resolves(self):
        rendered = render_indexes(self.catalog, self.validation)
        for relative, content in rendered.items():
            source = ROOT / relative
            for raw_target in re.findall(r"\[[^]]+\]\(([^)]+)\)", content):
                target = raw_target.strip("<>").split("#", 1)[0]
                if not target or "://" in target:
                    continue
                resolved = (source.parent / target).resolve()
                self.assertTrue(resolved.exists(), f"{relative}: {raw_target}")

    def test_legacy_paths_are_pinned_but_future_capsules_may_be_native(self):
        policy = self.catalog["canonical_path_policy"]
        self.assertEqual(policy["mode"], "legacy_pinned_capsule_native")
        self.assertIn("versioned provenance migration", policy["migration_boundary"])
        self.assertEqual(
            policy["migration_audit"]["explicit_legacy_path_reference_occurrences"],
            1198,
        )
        for relative in discover_inventory():
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_frozen_mainline_evidence_hashes_still_pass(self):
        manifest = load_json(MANIFEST_PATH)
        evidence = verify_evidence_files(manifest)
        self.assertTrue(evidence["passed"], evidence)


if __name__ == "__main__":
    unittest.main()
