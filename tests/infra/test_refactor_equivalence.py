import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from fsrl.evaluation.causal_suite import run_causal_suite
from fsrl.evaluation.frozen_fast_weight import FrozenEvaluationBackend
from tools.provenance.audit_refactor_equivalence_v1 import (
    DEFAULT_CONTRACT,
    load_contract,
)
from tools.provenance.audit_refactor_equivalence_v2 import (
    DEFAULT_CONTRACT as V2_CONTRACT,
)
from tools.provenance.audit_refactor_equivalence_v2 import (
    load_contract as load_v2_contract,
)
from tools.provenance.audit_refactor_equivalence_v3 import (
    DEFAULT_CONTRACT as V3_CONTRACT,
)
from tools.provenance.audit_refactor_equivalence_v3 import (
    load_contract as load_v3_contract,
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
                module = importlib.import_module(".".join(parts[:-2]))
                test_case = getattr(module, parts[-2])
                self.assertTrue(callable(getattr(test_case, parts[-1])))

    def test_contract_rejects_a_negative_tolerance(self):
        contract = load_contract()
        contract["bounded_checks"][0]["atol"] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-negative tolerances"):
                load_contract(path)

    def test_v2_contract_pins_cross_commit_snapshot_and_current_candidate(self):
        contract = load_v2_contract()
        self.assertEqual(
            contract["audit_id"], "relational-model-refactor-equivalence-v2"
        )
        self.assertEqual(
            contract["candidate_commit"],
            "c9319b69043ea3521f768aae887dc876408032d5",
        )
        self.assertEqual(
            contract["cross_commit_checks"],
            [
                {
                    "id": "encoding-and-relational-query-snapshot",
                    "script": ("tools/provenance/refactor_equivalence_snapshot_v2.py"),
                }
            ],
        )
        self.assertTrue(V2_CONTRACT.is_file())

    def test_contract_rejects_an_escaping_cross_commit_script(self):
        contract = load_v2_contract()
        contract["cross_commit_checks"][0]["script"] = "../outside.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative Python paths"):
                load_v2_contract(path)

    def test_v3_contract_pins_closing_fix_candidate(self):
        contract = load_v3_contract()
        self.assertEqual(
            contract["audit_id"], "relational-model-refactor-equivalence-v3"
        )
        self.assertEqual(
            contract["baseline_commit"],
            "c4f6ee14044b296181693a0dcd62e985f63636f9",
        )
        self.assertEqual(
            contract["candidate_commit"],
            "b864f2542cd49613dd320aebff35f98e53b0c8dc",
        )
        self.assertEqual(len(contract["cross_commit_checks"]), 2)
        self.assertTrue(V3_CONTRACT.is_file())
        for group in (*contract["exact_checks"], *contract["bounded_checks"]):
            for selector in group["selectors"]:
                parts = selector.split(".")
                module = importlib.import_module(".".join(parts[:-2]))
                test_case = getattr(module, parts[-2])
                self.assertTrue(callable(getattr(test_case, parts[-1])))

    def test_evaluation_readme_matches_the_current_default(self):
        root = DEFAULT_CONTRACT.parents[2]
        readme = root / "fsrl" / "evaluation" / "README.md"
        text = readme.read_text(encoding="utf-8")
        self.assertIn("current high-level default is `batched_sequence`", text)
        self.assertNotIn("The default remains `legacy_stepwise`", text)
        default = (
            inspect.signature(run_causal_suite).parameters["evaluation_backend"].default
        )
        self.assertEqual(
            default,
            FrozenEvaluationBackend.BATCHED_SEQUENCE,
        )

    def test_tools_readme_lists_all_refactor_equivalence_audits(self):
        root = DEFAULT_CONTRACT.parents[2]
        text = (root / "tools" / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "python -m tools.provenance.audit_refactor_equivalence_v1",
            text,
        )
        self.assertIn(
            "python -m tools.provenance.audit_refactor_equivalence_v2",
            text,
        )
        self.assertIn(
            "python -m tools.provenance.audit_refactor_equivalence_v3",
            text,
        )


if __name__ == "__main__":
    unittest.main()
