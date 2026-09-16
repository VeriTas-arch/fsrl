"""Frozen authorities for direct P/L training."""

from __future__ import annotations

from copy import deepcopy

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "pl_direct_training"
    / "records"
    / "benchmarks"
    / "pl_direct_training_v1.json"
)
PROTOCOL_SHA256 = "49b02c9275cdf3707a3fa7508f75ea0ec2c8daa257150e2e671c1baa940ff2ad"
PROTOCOL_COMMIT = "3a9e498a6a7b2af9080cde135090d1c51517952f"
REPAIR_PATH = PROTOCOL_PATH.with_name("pl_direct_training_v1.repair1.json")
REPAIR_SHA256 = "3cbd18ac36ce40b190c7759ef9ef7d3b7f951d26dfb42ae7f1c0da139d75d261"
REPAIR_COMMIT = "09629a47412b1eb081107ef42720e77adec3435f"


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen direct-training contract has changed")
    if file_sha256(REPAIR_PATH) != REPAIR_SHA256:
        raise RuntimeError("the frozen direct-training repair has changed")
    specification = deepcopy(load_json(PROTOCOL_PATH))
    repair = load_json(REPAIR_PATH)
    optimization = specification["optimization"]
    reparameterization = specification["training_reparameterization"]
    optimization["initial_local_gain"] = repair["resolution"]["initial_local_gain"]
    optimization["base_backbone_learning_rate"] = reparameterization[
        "base_backbone_learning_rate"
    ]
    optimization["local_learning_rate"] = 0.01
    specification["active_repair"] = {
        "repair_id": repair["repair_id"],
        "path": REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": REPAIR_SHA256,
        "commit": REPAIR_COMMIT,
    }
    return specification


def registered_seeds(specification: dict, cohort: str) -> tuple[int, ...]:
    key = {"development": "development", "confirmation": "reserved_confirmation"}.get(
        cohort
    )
    if key is None:
        raise ValueError(f"unknown cohort: {cohort}")
    return tuple(int(seed) for seed in specification["seeds"][key])


def registered_conditions(specification: dict) -> tuple[str, ...]:
    return tuple(str(value) for value in specification["design"]["conditions"])
