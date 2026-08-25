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
                self.assertEqual(
                    adapter["bootstrap"],
                    source["bootstrap"],
                )

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


if __name__ == "__main__":
    unittest.main()
