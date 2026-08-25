import tempfile
import unittest
from pathlib import Path

from fsrl.experiments.confirmation.mechanism import (
    COMPONENT_RESULTS,
    COMPONENT_SOURCES,
    DEVELOPMENT_TRAINING_PATH,
    aggregate_mechanism_confirmation,
    reproduction_gate,
    seed_estimands,
    threshold_status,
    validate_mechanism_contract,
    write_adapters,
)
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


class MechanismConfirmationRunnerTests(unittest.TestCase):
    def test_frozen_contract_and_population_validate(self):
        self.assertTrue(validate_mechanism_contract()["passed"])

    def test_adapters_change_artifacts_without_changing_estimand_contracts(self):
        assembly = load_json(COMPONENT_SOURCES["assembly"])
        artifacts = assembly["registered_sources"]["pilot_artifacts"]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_adapters(
                Path(temp_dir),
                training_specification_path=DEVELOPMENT_TRAINING_PATH,
                artifact_registrations=artifacts,
            )
            for component, path in paths.items():
                source = load_json(COMPONENT_SOURCES[component])
                adapter = load_json(path)
                self.assertEqual(
                    adapter["registered_sources"]["pilot_artifacts"], artifacts
                )
                self.assertEqual(adapter["bootstrap"], source["bootstrap"])

    def test_development_results_map_to_all_primary_seed_estimands(self):
        components = {
            name: load_json(path)["pilot_seeds"]["1901"]
            for name, path in COMPONENT_RESULTS.items()
        }
        primary, diagnostics = seed_estimands(components)
        self.assertEqual(len(primary), 11)
        self.assertGreater(primary["eligibility_donor_identity_advantage"], 0.0)
        self.assertIn("history_factor_generation", diagnostics)
        self.assertTrue(reproduction_gate(components, 3.814697265625e-6)["passed"])

    def test_threshold_status_keeps_unresolved_distinct_from_contrary(self):
        self.assertEqual(
            threshold_status({"bootstrap": {"lower": -0.1, "upper": 0.2}}),
            "unresolved",
        )
        self.assertEqual(
            threshold_status({"bootstrap": {"lower": -0.2, "upper": -0.1}}),
            "directionally_contrary",
        )

    def test_aggregate_refuses_partial_formal_seed_reporting(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(RuntimeError, "all registered formal seeds"),
        ):
            aggregate_mechanism_confirmation(output_root=Path(temp_dir))
