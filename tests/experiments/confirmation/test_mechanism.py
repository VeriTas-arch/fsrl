import unittest

from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import registered_file_sha256, resolve_record
from fsrl.infra.study_registry import resolve_registered_path as resolve_path


class MechanismConfirmationContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_json(
            resolve_record("benchmarks/mechanism_confirmation_v1.json")
        )

    def test_every_registered_source_is_immutable(self):
        for registration in self.contract["registered_sources"].values():
            self.assertEqual(
                registered_file_sha256(
                    registration["path"],
                    registration["sha256"],
                    resolved_path=resolve_path(registration["path"]),
                ),
                registration["sha256"],
            )

    def test_formal_population_matches_older_frozen_contract(self):
        older = load_json(resolve_record("benchmarks/confirmation_v1.json"))
        self.assertEqual(
            self.contract["formal_artifact_contract"]["seeds"],
            older["training"]["seeds"],
        )
        self.assertEqual(
            self.contract["formal_artifact_contract"]["checkpoint_selection"],
            older["training"]["checkpoint_selection"],
        )

    def test_only_development_supported_links_are_primary(self):
        primary = {row["id"] for row in self.contract["primary_mechanism_links"]}
        self.assertIn("history_dependent_expression", primary)
        self.assertIn("history_matched_nonlinearity", primary)
        self.assertNotIn("history_factor_generation", primary)
        self.assertTrue(
            any(
                "total factor-generation" in diagnostic
                for diagnostic in self.contract["registered_nonprimary_diagnostics"]
            )
        )


if __name__ == "__main__":
    unittest.main()
