import copy
import json
import unittest
from pathlib import Path

from fsrl.infra.semantic_contract import (
    evaluate_assertion,
    evaluate_semantic_contract,
    json_pointer,
    load_semantic_contract,
    validate_contract_source,
    validate_semantic_contract,
)
from fsrl.paths import REPO_ROOT

CONTRACT_ROOT = REPO_ROOT / "fsrl" / "experiments" / "transport" / "contracts"


class SemanticContractTests(unittest.TestCase):
    def test_json_pointer_handles_arrays_and_escaped_tokens(self):
        document = {"a/b": {"~value": [3, 5]}}
        self.assertEqual(json_pointer(document, "/a~1b/~0value/1"), 5)
        with self.assertRaisesRegex(KeyError, "does not resolve"):
            json_pointer(document, "/a~1b/missing")

    def test_assertion_operators_preserve_open_and_closed_boundaries(self):
        document = {"value": 0.5, "ready": True}
        self.assertFalse(
            evaluate_assertion(
                document,
                {
                    "json_pointer": "/value",
                    "operator": "greater_than",
                    "expected": 0.5,
                },
            )["passed"]
        )
        self.assertTrue(
            evaluate_assertion(
                document,
                {
                    "json_pointer": "/value",
                    "operator": "greater_equal",
                    "expected": 0.5,
                },
            )["passed"]
        )
        self.assertTrue(
            evaluate_assertion(
                document, {"json_pointer": "/ready", "operator": "is_true"}
            )["passed"]
        )

    def test_invalid_contract_fails_closed(self):
        contract = {
            "document_type": "fsrl.semantic_contract",
            "schema_version": 1,
            "contract_id": "example",
            "criteria": {"gate": []},
            "required_criteria": ["missing"],
            "integrity_assertions": [],
        }
        validation = validate_semantic_contract(contract)
        self.assertFalse(validation["passed"])
        self.assertGreaterEqual(len(validation["errors"]), 3)

    def test_checked_in_contracts_bind_to_registered_sources(self):
        for path in sorted(CONTRACT_ROOT.glob("*.json")):
            with self.subTest(path=path.name):
                contract = load_semantic_contract(path)
                self.assertTrue(validate_contract_source(contract)["passed"])

    def test_integrity_failure_masks_scientific_flags(self):
        path = CONTRACT_ROOT / "topology_within_cell_v1.json"
        contract = load_semantic_contract(path)
        document = self._passing_document(contract)
        document["integrity"]["all_passed"] = False
        result = evaluate_semantic_contract(document, contract)
        self.assertFalse(result["interpretable"])
        self.assertFalse(result["passed"])
        self.assertFalse(any(result["flags"].values()))

    def test_contract_threshold_change_is_data_not_runner_logic(self):
        path = CONTRACT_ROOT / "topology_within_cell_v1.json"
        contract = load_semantic_contract(path)
        document = self._passing_document(contract)
        baseline = evaluate_semantic_contract(document, contract)
        self.assertTrue(baseline["passed"])
        changed = copy.deepcopy(contract)
        criterion = changed["criteria"]["intact_competence"][0]
        criterion["expected"] = json_pointer(document, criterion["json_pointer"])
        self.assertFalse(evaluate_semantic_contract(document, changed)["passed"])

    @staticmethod
    def _passing_document(contract: dict) -> dict:
        document = {"metrics": {}, "integrity": {"all_passed": True}}
        for assertions in contract["criteria"].values():
            for assertion in assertions:
                pointer = assertion["json_pointer"]
                expected = assertion.get("expected")
                operator = assertion.get("operator", "equals")
                if operator in {"greater_than", "greater_equal"}:
                    value = expected + 1.0
                elif operator in {"less_than", "less_equal"}:
                    value = expected - 1.0
                elif operator == "is_true":
                    value = True
                elif operator == "is_false":
                    value = False
                else:
                    value = expected
                target = document
                tokens = pointer[1:].split("/")
                for token in tokens[:-1]:
                    target = target.setdefault(token, {})
                target[tokens[-1]] = value
        return document


class RegisteredDecisionParityTests(unittest.TestCase):
    def test_topology_contract_replays_all_frozen_cell_decisions(self):
        self._assert_registered_parity(
            CONTRACT_ROOT / "topology_within_cell_v1.json",
            REPO_ROOT / "studies/support_topology_transport/records/results/"
            "liu_support_topology_transport_v1.json",
            "graphs",
        )

    def test_item_count_contract_replays_all_frozen_cell_decisions(self):
        self._assert_registered_parity(
            CONTRACT_ROOT / "item_count_within_cell_v1.json",
            REPO_ROOT / "studies/item_count_transport/records/results/"
            "liu_item_count_transport_v1.json",
            "sizes",
        )

    def _assert_registered_parity(
        self, contract_path: Path, result_path: Path, cell_key: str
    ) -> None:
        contract = load_semantic_contract(contract_path)
        frozen = json.loads(result_path.read_text(encoding="utf-8"))
        cells = 0
        for seed in frozen["seeds"].values():
            for cell in seed[cell_key].values():
                observed = evaluate_semantic_contract(
                    {"metrics": cell["metrics"], "integrity": cell["integrity"]},
                    contract,
                )
                expected = {
                    "interpretable": observed["interpretable"],
                    **observed["decision_outputs"],
                    "flags": observed["flags"],
                }
                self.assertEqual(expected, cell["decision"])
                cells += 1
        self.assertEqual(cells, 9)
