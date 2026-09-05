"""Frozen authority and generic configuration for the training-strategy study."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT
from fsrl.training.backbone import MetaTrainConfig

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "joint_training_strategy"
    / "records"
    / "benchmarks"
    / "joint_training_strategy_v1.json"
)
PROTOCOL_SHA256 = "af6fe7ccd0785cf5ec09437cd958f5652a36596066aec28a77793dad28133ea2"
PROTOCOL_COMMIT = "d3480514ec74c19d68b37da604472c743065d262"


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen training-strategy contract has changed")
    return load_json(PROTOCOL_PATH)


def training_config(specification: dict, seed: int) -> MetaTrainConfig:
    optimization = specification["optimization"]
    task = specification["task"]
    architecture = specification["architecture"]
    return MetaTrainConfig(
        seed=seed,
        outer_steps=optimization["total_steps"],
        batch_size=optimization["batch_size"],
        hidden_size=architecture["hidden_size"],
        cue_size=task["cue_size"],
        min_edges=task["min_edges"],
        max_edges=task["max_edges"],
        support_blocks=task["support_blocks"],
        learning_rate=optimization["backbone_learning_rate"],
        gradient_clip=optimization["gradient_clip"],
        fast_weight_penalty=optimization["fast_weight_penalty"],
        support_query_time=architecture["support_query_time"],
        save_every=optimization["total_steps"],
        subject_encoding_mode=task["subject_encoding_mode"],
    )


def phase_for_step(specification: dict, condition: str, step: int) -> str:
    optimization = specification["optimization"]
    if condition not in specification["seeds"]["conditions"]:
        raise ValueError(f"unregistered condition: {condition}")
    if not 0 <= step < optimization["total_steps"]:
        raise ValueError("step lies outside the registered optimization budget")
    if condition == "joint":
        return "joint"
    return "global" if step < optimization["global_only_steps"] else "local"
