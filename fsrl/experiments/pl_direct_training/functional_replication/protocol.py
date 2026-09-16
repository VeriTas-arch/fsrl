"""Frozen authorities for clean no-time P/L functional replication."""

from __future__ import annotations

from copy import deepcopy

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

from ..protocol import (
    PROTOCOL_SHA256 as PARENT_PROTOCOL_SHA256,
)
from ..protocol import (
    REPAIR_SHA256 as PARENT_REPAIR_SHA256,
)
from ..protocol import load_specification as load_parent_specification

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "pl_functional_replication"
    / "records"
    / "benchmarks"
    / "pl_functional_replication_v1.json"
)
PROTOCOL_SHA256 = "dc146c7723ad147a5fd964f8af5c66cad4b182c034fa20fc3c5391bb4a2e16d0"
PROTOCOL_COMMIT = "976fb8ec5a645f532650bafe074790748fc02259"
REPAIR_PATH = PROTOCOL_PATH.with_name("pl_functional_replication_v1.repair1.json")
REPAIR_SHA256 = "f8fb383ee947a44ba784195af9f87b380136dcae22dbccac688eff704cc087ee"
REPAIR_COMMIT = "f91d02252df73eb0e5586787a0cf8841fef9354f"
EXECUTION_REPAIR_PATH = PROTOCOL_PATH.with_name(
    "pl_functional_replication_v1.execution_repair1.json"
)
EXECUTION_REPAIR_SHA256 = (
    "5707bcf608434700d12517dcfa75c47a38fe4912f918cb4287b53774a456d4ce"
)
CONDITION = "no_time_candidate"


def load_execution_repair() -> dict:
    if file_sha256(EXECUTION_REPAIR_PATH) != EXECUTION_REPAIR_SHA256:
        raise RuntimeError("the frozen functional execution repair has changed")
    return load_json(EXECUTION_REPAIR_PATH)


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen functional-replication contract has changed")
    if file_sha256(REPAIR_PATH) != REPAIR_SHA256:
        raise RuntimeError("the frozen functional-replication repair has changed")
    specification = deepcopy(load_json(PROTOCOL_PATH))
    repair = load_json(REPAIR_PATH)
    parent = specification["candidate_identity"]
    if parent["parent_contract"]["sha256"] != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("functional replication cites a different parent contract")
    if parent["parent_repair"]["sha256"] != PARENT_REPAIR_SHA256:
        raise RuntimeError("functional replication cites a different parent repair")
    specification["active_repair"] = {
        "repair_id": repair["repair_id"],
        "path": REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": REPAIR_SHA256,
        "commit": REPAIR_COMMIT,
        "remote_reference_condition": "shared_access",
    }
    return specification


def candidate_specification() -> dict:
    """Resolve the unchanged parent recipe used by the sole candidate."""

    frozen = load_specification()
    candidate = deepcopy(load_parent_specification())
    optimization = frozen["training_freeze"]["optimization"]
    expected = {
        "batch_size": optimization["batch_size"],
        "total_steps": optimization["total_steps"],
        "total_episode_exposures": optimization["total_episode_exposures"],
        "initial_local_gain": optimization["initial_local_gain"],
        "base_backbone_learning_rate": optimization["base_backbone_learning_rate"],
        "local_learning_rate": optimization["local_gain_learning_rate"],
        "gradient_clip": optimization["gradient_clip"],
        "fast_weight_penalty": optimization["fast_weight_penalty"],
        "training_temperature": optimization["training_temperature"],
    }
    observed = {name: candidate["optimization"][name] for name in expected}
    if observed != expected:
        raise RuntimeError("candidate recipe differs from the frozen parent recipe")
    architecture = frozen["candidate_identity"]["architecture"]
    parent_architecture = candidate["architecture"]
    if (
        parent_architecture["common"]["task_input_size"]
        != architecture["task_input_size"]
        or parent_architecture[CONDITION]["backbone_parameters"]
        != architecture["parameter_counts"]["backbone"]
        or parent_architecture["common"]["persistent_state_sizes"]
        != architecture["persistent_state_sizes"]
    ):
        raise RuntimeError("candidate architecture differs from the frozen parent")
    candidate["experiment_id"] = frozen["experiment_id"]
    return candidate


def registered_seeds(specification: dict | None = None) -> tuple[int, ...]:
    specification = load_specification() if specification is None else specification
    return tuple(
        int(seed) for seed in specification["fresh_seed_contract"]["mandatory_seeds"]
    )


def registered_conditions(specification: dict | None = None) -> tuple[str, ...]:
    specification = load_specification() if specification is None else specification
    return tuple(
        str(value)
        for value in specification["evaluation_conditions"]["liu"]["conditions"]
    )
