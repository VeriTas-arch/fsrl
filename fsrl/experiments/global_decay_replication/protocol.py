"""Prospective authority for the fresh global-decay replication."""

from __future__ import annotations

from copy import deepcopy

from fsrl.experiments.quantized_learner.protocol import (
    resolved_specification as parent_specification,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "global_decay_replication/records"
DESIGN = RECORDS / "benchmarks/global_decay_replication_v1.json"
DESIGN_HASH = "8e9c259355d88fcbee7b532b2ac7e86d29730b61d7edcd1f54fec90087a5b035"
DESIGN_COMMIT = "e4aa6df42f3fdcfe18dc33d0f4ba0388a33dcb1e"
RUN_ROOT = RUNS_ROOT / "global_decay_replication_v1"

SEEDS = (2120, 2121, 2122)
CONDITIONS = ("fixed_eta_resampled", "adaptive_eta_resampled")
EXECUTION_ORDER = {
    2120: CONDITIONS,
    2121: CONDITIONS[::-1],
    2122: CONDITIONS,
}
TRAINING_RNG_OFFSET = 1_150_000
ENCODING_RNG_OFFSET = 1_160_000
GENERIC_SEED = 1_170_001
GENERIC_ENCODING_SEED = 1_170_003
GENERIC_BOOTSTRAP_SEED_BASE = 1_171_000
QUALIFICATION_SEED = 1_190_001
COHORTS = 400
COHORT_SIZE = 77
COHORT_SHARD_SIZE = 20
COHORT_SEED_BASE = 5_200_000
COHORT_SEED_STRIDE = 1_000
COHORT_OFFSETS = {
    "cue_seed": 0,
    "support_seed": 200,
    "subject_encoding_seed": 400,
    "encoding_seed": 600,
    "choice_seed": 800,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 5_800_000
CODEBOOK = (-1.0, -1 / 3, 1 / 3, 1.0)


def specification() -> dict:
    if file_sha256(DESIGN) != DESIGN_HASH:
        raise RuntimeError("frozen global-decay replication contract changed")
    verify_reference(reference(DESIGN), commit=DESIGN_COMMIT)
    return load_json(DESIGN)


def resolved_specification() -> dict:
    specification()
    spec = deepcopy(parent_specification())
    spec["experiment_id"] = "global_decay_replication_v1"
    spec["seeds"] = {
        "mandatory": list(SEEDS),
        "conditions": list(CONDITIONS),
        "execution_order": {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS},
        "training_rng_offset": TRAINING_RNG_OFFSET,
        "encoding_rng_offset": ENCODING_RNG_OFFSET,
    }
    spec["optimization"].update(trainable_parameters=["raw_eta", "raw_global_gain"])
    spec["evaluation"]["generic"].update(episodes=256, rng_seed=GENERIC_SEED)
    return spec


def cohort_specification(index: int) -> dict:
    if not 0 <= index < COHORTS:
        raise ValueError("unregistered global-decay replication cohort")
    spec = resolved_specification()
    base = COHORT_SEED_BASE + COHORT_SEED_STRIDE * index
    spec["evaluation"]["liu"].update(
        {name: base + offset for name, offset in COHORT_OFFSETS.items()}
    )
    spec["evaluation"]["liu"]["subjects"] = COHORT_SIZE
    return spec


def run_directory(seed: int, condition: str):
    if seed not in SEEDS or condition not in CONDITIONS:
        raise ValueError("unregistered global-decay replication fit")
    return RUN_ROOT / f"seed-{seed}" / condition
