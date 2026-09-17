"""Frozen authority and paths for the single-P anytime study."""

from __future__ import annotations

import copy

from fsrl.experiments.clean_single_p.protocol import (
    inherited_recipe as clean_inherited_recipe,
)
from fsrl.experiments.clean_single_p.protocol import (
    specification as clean_specification,
)
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_anytime"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_anytime_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
MODEL_LOCK = RECORDS / "benchmarks/model_lock.json"
RESULT = RECORDS / "results/single_p_anytime_v1.json"
REPORT = RECORDS / "reports/single_p_anytime_v1.md"
RUNS = RUNS_ROOT / "single_p_anytime_v1"
EVALUATION_RUNS = RUNS / "evaluation"
PROTOCOL_SHA256 = "3d70b51bc076b5cf75b40a5929cde4578c429ea0284e369ce2214a0606df76c0"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("single-P anytime protocol changed")
    return load_json(PROTOCOL)


def optimizer_specification() -> dict:
    """Expose the unchanged clean-single-P optimizer contract."""
    parent = clean_specification()
    if (
        parent["training"]["steps"] != specification()["training_schedule"]["updates"]
        or parent["training"]["batch_size"]
        != specification()["training_schedule"]["batch_size"]
    ):
        raise RuntimeError("single-P anytime budget differs from its parent")
    return parent


def historical_recipe(panel: int) -> dict:
    if panel not in specification()["design"]["evaluation_panels"]:
        raise ValueError("unregistered single-P anytime panel")
    result = copy.deepcopy(clean_inherited_recipe(panel))
    base = 751000 + panel * 100
    result["evaluation"]["generic"]["rng_seed"] = base
    result["evaluation"]["liu"].update(
        {
            key: base + offset
            for key, offset in {
                "cue_seed": 11,
                "support_seed": 12,
                "subject_encoding_seed": 13,
                "choice_seed": 14,
                "order_seed": 15,
                "query_shuffle_seed": 16,
                "evidence_shuffle_seed": 17,
            }.items()
        }
    )
    return result


def training_directory(seed: int, recipe: str, arm: str):
    return RUNS / "training" / str(seed) / recipe / arm


__all__ = [
    "EVALUATION_RUNS",
    "MODEL_LOCK",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "historical_recipe",
    "optimizer_specification",
    "specification",
    "training_directory",
]
