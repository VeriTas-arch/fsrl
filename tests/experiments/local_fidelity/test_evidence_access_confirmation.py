import unittest
from pathlib import Path

import fsrl.experiments.local_fidelity.evidence_access_pilot as dual_access
from fsrl.experiments.local_fidelity.evidence_access_confirmation import (
    bind_fresh_artifacts,
    confirmation_decision,
    fresh_seeds,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record


class DualEvidenceAccessConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_json(
            resolve_record("benchmarks/dual_evidence_access_confirmation_v2_4.json")
        )

    def test_fresh_seed_contract_is_exact(self):
        self.assertEqual(fresh_seeds(self.specification), (2104, 2105))

    def test_fresh_seed_contract_rejects_replacement(self):
        changed = {
            **self.specification,
            "network_seed_contract": {
                **self.specification["network_seed_contract"],
                "mandatory_seeds": [2104, 2106],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "2104 and 2105"):
            fresh_seeds(changed)

    def test_confirmation_decision_never_pools_heterogeneous_link(self):
        flags = {name: True for name in self.specification["primary_links"]}
        seeds = {
            "2104": {"decision": {"interpretable": True, "flags": flags}},
            "2105": {
                "decision": {
                    "interpretable": True,
                    "flags": {**flags, "retained_fidelity_preservation": False},
                }
            },
        }
        result = confirmation_decision(self.specification, seeds)
        self.assertEqual(result["outcome"], "heterogeneous_or_unresolved")
        self.assertEqual(
            result["links"]["retained_fidelity_preservation"]["status"],
            "heterogeneous_or_unresolved",
        )
        self.assertEqual(result["network_population_inference"], "not_performed")

    def test_confirmation_pass_has_fresh_backbone_label(self):
        flags = {name: True for name in self.specification["primary_links"]}
        seeds = {
            str(seed): {"decision": {"interpretable": True, "flags": flags}}
            for seed in (2104, 2105)
        }
        result = confirmation_decision(self.specification, seeds)
        self.assertEqual(result["v2_4_rule_outcome"], "all_links_pass")
        self.assertEqual(result["outcome"], "fresh_backbone_confirmation_pass")

    def test_fresh_binding_restores_frozen_module_paths(self):
        original_specification = dual_access.V2_3_SPECIFICATION_PATH
        original_output = dual_access.V2_3_OUTPUT_ROOT
        specification = Path("/tmp/fresh-specification.json")
        output = Path("/tmp/fresh-output")
        with bind_fresh_artifacts(specification, output):
            self.assertEqual(dual_access.V2_3_SPECIFICATION_PATH, specification)
            self.assertEqual(dual_access.V2_3_OUTPUT_ROOT, output)
        self.assertEqual(dual_access.V2_3_SPECIFICATION_PATH, original_specification)
        self.assertEqual(dual_access.V2_3_OUTPUT_ROOT, original_output)


if __name__ == "__main__":
    unittest.main()
