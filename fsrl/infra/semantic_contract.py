"""Versioned, data-driven assertions for scientific equivalence contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fsrl.infra.provenance import file_sha256
from fsrl.infra.record_catalog import record_reference, resolve_record_id

SUPPORTED_OPERATORS = frozenset(
    {
        "equals",
        "not_equals",
        "is_true",
        "is_false",
        "less_than",
        "less_equal",
        "greater_than",
        "greater_equal",
    }
)


def json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer without interpreting scientific values."""

    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer!r}")
    value = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            try:
                value = value[int(token)]
            except (IndexError, ValueError) as error:
                raise KeyError(f"JSON pointer does not resolve: {pointer}") from error
        elif isinstance(value, dict):
            try:
                value = value[token]
            except KeyError as error:
                raise KeyError(f"JSON pointer does not resolve: {pointer}") from error
        else:
            raise TypeError(f"JSON pointer crosses a scalar: {pointer}")
    return value


def evaluate_assertion(document: Any, assertion: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one explicit assertion and return an auditable observation."""

    pointer = assertion.get("json_pointer")
    if not isinstance(pointer, str):
        raise TypeError("semantic assertion requires a JSON pointer")
    operator = assertion.get("operator", "equals")
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported semantic assertion operator: {operator}")
    observed = json_pointer(document, pointer)
    expected = assertion.get("expected")
    if operator == "equals":
        passed = observed == expected
    elif operator == "not_equals":
        passed = observed != expected
    elif operator == "is_true":
        expected = True
        passed = observed is True
    elif operator == "is_false":
        expected = False
        passed = observed is False
    elif operator == "less_than":
        passed = observed < expected
    elif operator == "less_equal":
        passed = observed <= expected
    elif operator == "greater_than":
        passed = observed > expected
    else:
        passed = observed >= expected
    return {
        "json_pointer": pointer,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def evaluate_assertions(
    document: Any, assertions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evaluate a non-empty conjunction of semantic assertions."""

    if not isinstance(assertions, list) or not assertions:
        raise ValueError("semantic assertion group must be a non-empty list")
    return [evaluate_assertion(document, assertion) for assertion in assertions]


def validate_semantic_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the small versioned decision-contract schema."""

    errors: list[str] = []
    if contract.get("document_type") != "fsrl.semantic_contract":
        errors.append("invalid semantic-contract document_type")
    if contract.get("schema_version") != 1:
        errors.append("semantic-contract schema_version must be 1")
    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        errors.append("semantic contract requires contract_id")
    criteria = contract.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        errors.append("semantic contract requires criteria")
        criteria = {}
    for name, assertions in criteria.items():
        if not isinstance(name, str) or not name:
            errors.append("semantic criterion names must be non-empty strings")
        if not isinstance(assertions, list) or not assertions:
            errors.append(f"semantic criterion must contain assertions: {name}")
            continue
        for assertion in assertions:
            if not isinstance(assertion, dict):
                errors.append(f"semantic assertion must be an object: {name}")
                continue
            if assertion.get("operator", "equals") not in SUPPORTED_OPERATORS:
                errors.append(f"unsupported semantic assertion operator: {name}")
            if not isinstance(assertion.get("json_pointer"), str):
                errors.append(f"semantic assertion requires JSON pointer: {name}")
    required = contract.get("required_criteria")
    if (
        not isinstance(required, list)
        or not required
        or any(name not in criteria for name in required)
        or len(required) != len(set(required))
    ):
        errors.append("required_criteria must select unique declared criteria")
    integrity = contract.get("integrity_assertions")
    if not isinstance(integrity, list) or not integrity:
        errors.append("semantic contract requires integrity_assertions")
    outputs = contract.get("decision_outputs", {})
    if not isinstance(outputs, dict):
        errors.append("decision_outputs must be an object")
        outputs = {}
    for name, selector in outputs.items():
        if not isinstance(name, str) or not isinstance(selector, dict):
            errors.append("decision output selectors must be named objects")
            continue
        keys = set(selector)
        if keys == {"criterion"}:
            if selector["criterion"] not in criteria:
                errors.append(f"decision output selects unknown criterion: {name}")
        elif keys != {"all_required"} or selector["all_required"] is not True:
            errors.append(f"invalid decision output selector: {name}")
    return {
        "passed": not errors,
        "errors": errors,
        "contract_id": contract_id,
        "criteria": len(criteria),
    }


def validate_contract_source(contract: dict[str, Any]) -> dict[str, Any]:
    """Bind a maintained semantic contract to one immutable registered source."""

    source = contract.get("source", {})
    record_id = source.get("record_id")
    expected_sha256 = source.get("materialized_sha256")
    criteria_pointer = source.get("criteria_pointer")
    if not all(
        isinstance(value, str) and value for value in (record_id, expected_sha256)
    ):
        raise ValueError("semantic contract source requires record ID and SHA-256")
    reference = record_reference(record_id)
    path = resolve_record_id(record_id)
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"semantic contract source identity mismatch: {record_id}")
    source_document = json.loads(path.read_text(encoding="utf-8"))
    source_criteria = json_pointer(source_document, criteria_pointer)
    missing = [name for name in contract["criteria"] if name not in source_criteria]
    if missing:
        raise RuntimeError(f"semantic contract source omits criteria: {missing}")
    return {
        "passed": True,
        "record_id": record_id,
        "repository_path": reference["repository_path"],
        "sha256": observed_sha256,
        "criteria": len(contract["criteria"]),
    }


def load_semantic_contract(path: Path | str) -> dict[str, Any]:
    """Load and validate one checked-in semantic contract."""

    source = Path(path)
    contract = json.loads(source.read_text(encoding="utf-8"))
    validation = validate_semantic_contract(contract)
    if not validation["passed"]:
        raise ValueError(f"invalid semantic contract {source}: {validation['errors']}")
    validate_contract_source(contract)
    return contract


def evaluate_semantic_contract(
    document: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    """Interpret one contract without embedding thresholds in runner code."""

    validation = validate_semantic_contract(contract)
    if not validation["passed"]:
        raise ValueError(f"invalid semantic contract: {validation['errors']}")
    integrity_assertions = evaluate_assertions(
        document, contract["integrity_assertions"]
    )
    interpretable = all(assertion["passed"] for assertion in integrity_assertions)
    criteria = {}
    for name, assertions in contract["criteria"].items():
        observations = evaluate_assertions(document, assertions)
        criteria[name] = {
            "passed": bool(
                interpretable and all(row["passed"] for row in observations)
            ),
            "assertions": observations,
        }
    required = contract["required_criteria"]
    all_required = all(criteria[name]["passed"] for name in required)
    outputs = {}
    for name, selector in contract.get("decision_outputs", {}).items():
        if "criterion" in selector:
            outputs[name] = criteria[selector["criterion"]]["passed"]
        else:
            outputs[name] = all_required
    return {
        "contract_id": contract["contract_id"],
        "passed": bool(interpretable and all_required),
        "interpretable": bool(interpretable),
        "integrity_assertions": integrity_assertions,
        "required_criteria": required,
        "criteria": criteria,
        "flags": {name: value["passed"] for name, value in criteria.items()},
        "decision_outputs": outputs,
    }
