"""Frozen authority and paths for the M2 promotion study."""

from __future__ import annotations

import copy
from pathlib import Path

from fsrl.experiments.clean_single_p.protocol import inherited_recipe as parent_recipe
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "minimal_single_p_promotion"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/minimal_single_p_promotion_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
REPAIR = RECORDS / "benchmarks/reporting_repair1.json"
REPAIR_QUALIFICATION = RECORDS / "benchmarks/reporting_repair1_qualification.json"
SOURCE_REPAIR_LOCK = RECORDS / "benchmarks/source_repair1.json"
MODEL_LOCK = RECORDS / "benchmarks/model_lock.json"
GENERIC_RESULT = RECORDS / "results/minimal_single_p_promotion_v1.generic.json"
GENERIC_REPORT = RECORDS / "reports/minimal_single_p_promotion_v1.generic.md"
RESULT = RECORDS / "results/minimal_single_p_promotion_v1.json"
REPORT = RECORDS / "reports/minimal_single_p_promotion_v1.md"
RUNS = RUNS_ROOT / "minimal_single_p_promotion_v1"
INPUTS = RUNS / "inputs"
PROTOCOL_SHA256 = "4ed6486f3e1c944fd6680557c2f678834d25a687ca7ae7cf6138424d32d8c72f"
REPAIR_SHA256 = "b02260daf98fd13f5133dd13bd830a3c5315e6568ca47480913c9eb525ae9455"
ENGINEERING_REPAIR = RECORDS / "benchmarks/engineering_repair2.json"
ENGINEERING_REPAIR_SHA256 = (
    "c9ad15f09046a74fd4c7824a119bf0f0abde2902f131cbe772a2115c1b8298e5"
)
ENGINEERING_QUALIFICATION = (
    RECORDS / "benchmarks/engineering_repair2_qualification.json"
)
SOURCE_ENGINEERING_LOCK = RECORDS / "benchmarks/source_repair2.json"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("minimal single-P promotion protocol changed")
    return load_json(PROTOCOL)


def inherited_recipe(panel: int) -> dict:
    if panel not in specification()["design"]["evaluation_panels"]:
        raise ValueError("unregistered promotion panel")
    recipe = copy.deepcopy(parent_recipe(panel))
    recipe["evaluation"]["generic"]["rng_seed"] = 961000 + 100 * panel
    recipe["evaluation"]["liu"].update(
        {
            key: 971000 + 100 * panel + offset
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
    return recipe


def training_directory(seed: int) -> Path:
    return RUNS / "training" / str(seed)


def generic_directory(seed: int, panel: int) -> Path:
    return RUNS / "generic" / str(seed) / str(panel)


def liu_directory(seed: int, panel: int) -> Path:
    return RUNS / "liu" / str(seed) / str(panel)


__all__ = [
    "ENGINEERING_QUALIFICATION",
    "ENGINEERING_REPAIR",
    "ENGINEERING_REPAIR_SHA256",
    "GENERIC_REPORT",
    "GENERIC_RESULT",
    "INPUTS",
    "MODEL_LOCK",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "REPAIR",
    "REPAIR_QUALIFICATION",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_ENGINEERING_LOCK",
    "SOURCE_LOCK",
    "SOURCE_REPAIR_LOCK",
    "generic_directory",
    "inherited_recipe",
    "liu_directory",
    "specification",
    "training_directory",
]
