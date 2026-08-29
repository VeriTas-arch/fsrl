"""Registered within-cell decision contract shared by topology transports."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fsrl.infra.semantic_contract import (
    evaluate_semantic_contract,
    load_semantic_contract,
)

DECISION_CONTRACT_PATH = (
    Path(__file__).with_name("contracts") / "topology_within_cell_v1.json"
)


@lru_cache(maxsize=1)
def _decision_contract() -> dict:
    return load_semantic_contract(DECISION_CONTRACT_PATH)


def within_cell_decision(metrics: dict, integrity: dict) -> dict:
    evaluation = evaluate_semantic_contract(
        {"metrics": metrics, "integrity": integrity}, _decision_contract()
    )
    return {
        "interpretable": evaluation["interpretable"],
        **evaluation["decision_outputs"],
        "flags": evaluation["flags"],
    }
