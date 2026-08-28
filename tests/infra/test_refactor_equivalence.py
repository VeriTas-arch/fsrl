import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from fsrl.evaluation.frozen_fast_weight import (
    FrozenEvaluationBackend,
    run_causal_suite,
)
from tools.provenance.audit_refactor_equivalence_v1 import (
    DEFAULT_CONTRACT,
    load_contract,
)


class RefactorEquivalenceAuditTests(unittest.TestCase):
    def test_contract_is_pinned_scoped_and_resolvable(self):
        contract = load_contract()
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(len(contract["baseline_commit"]), 40)
        self.assertEqual(len(contract["candidate_commit"]), 40)
        for broad_path in (
            "fsrl",
            "studies",
            "synthesis",
            "tests",
            "tools",
            "workflows",
        ):
            self.assertNotIn(broad_path, contract["audited_paths"])

        for group in (*contract["exact_checks"], *contract["bounded_checks"]):
            for selector in group["selectors"]:
                parts = selector.split(".")
                module = importlib.import_module(".".join(parts[:3]))
                test_case = getattr(module, parts[3])
                self.assertTrue(callable(getattr(test_case, parts[4])))

    def test_contract_rejects_a_negative_tolerance(self):
        contract = load_contract()
        contract["bounded_checks"][0]["atol"] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-negative tolerances"):
                load_contract(path)

    def test_root_readme_matches_the_current_evaluation_default(self):
        readme = DEFAULT_CONTRACT.parents[2] / "README.md"
        text = readme.read_text(encoding="utf-8")
        self.assertIn("current high-level default is `batched_sequence`", text)
        self.assertNotIn("The default remains `legacy_stepwise`", text)
        self.assertIn(
            "python -m tools.provenance.audit_refactor_equivalence_v1",
            text,
        )
        default = (
            inspect.signature(run_causal_suite).parameters["evaluation_backend"].default
        )
        self.assertEqual(
            default,
            FrozenEvaluationBackend.BATCHED_SEQUENCE,
        )


if __name__ == "__main__":
    unittest.main()
