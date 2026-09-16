"""Frozen authority for the compact global/local model study."""

from __future__ import annotations

from copy import deepcopy

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "compact_global_local_model"
    / "records"
    / "benchmarks"
    / "compact_global_local_model_v1.json"
)
PROTOCOL_SHA256 = "fd4b61bcd55b4487b7017bbfbb8fb1bebbdcefdcccbc9976a7c07167b69300e0"
PROTOCOL_COMMIT = "a15e4b34869887b4da37ce3dfacad42b0cf86dda"
REPAIR_PATH = PROTOCOL_PATH.with_name("compact_global_local_model_v1.repair1.json")
REPAIR_SHA256 = "d48196587245f40049e9297c0f94f238e638f3a66040f40623a66b808a02d49b"
REPAIR_COMMIT = "b0eb270575a194076c3848f2c3e9ad7a3395dbd3"


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen compact-model contract has changed")
    if file_sha256(REPAIR_PATH) != REPAIR_SHA256:
        raise RuntimeError("the frozen compact-model repair has changed")
    specification = deepcopy(load_json(PROTOCOL_PATH))
    repair = load_json(REPAIR_PATH)
    architecture = specification["architecture"]
    architecture["support_trial_steps"] = repair["repair"]["support_trial_steps"]
    architecture["query_steps"] = repair["repair"]["query_steps"]
    architecture["blank_initial_steps"] = repair["repair"]["blank_initial_steps"]
    specification["active_repair"] = {
        "repair_id": repair["repair_id"],
        "path": REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": REPAIR_SHA256,
        "commit": REPAIR_COMMIT,
    }
    return specification


def phase_for_step(specification: dict, step: int) -> str:
    optimization = specification["optimization"]
    if not 0 <= step < optimization["total_steps"]:
        raise ValueError("step lies outside the registered optimization budget")
    return "global" if step < optimization["global_only_steps"] else "local"


def registered_seeds(specification: dict, cohort: str) -> tuple[int, ...]:
    if cohort not in {"development", "confirmation"}:
        raise ValueError(f"unknown cohort: {cohort}")
    return tuple(int(seed) for seed in specification["seeds"][cohort])
