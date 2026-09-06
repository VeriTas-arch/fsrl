"""Prospective authority for the common encoding state pilot."""

from __future__ import annotations

from copy import deepcopy

from fsrl.experiments.quantized_learner.protocol import (
    resolved_specification as parent_specification,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "common_encoding_state/records"
DESIGN = RECORDS / "benchmarks/common_encoding_state_v1.json"
DESIGN_HASH = "5f55121959a9de7e396d0be7c87948b8cb071db97889dd7f5677b45e1d4ed10b"
DESIGN_COMMIT = "2734cf32a30ba2d7541a55bc22f1dacb2dc61b9d"
RUN_ROOT = RUNS_ROOT / "common_encoding_state_v1"

SEEDS = (2123, 2124, 2125)
CONDITIONS = ("independent", "relation_common", "episode_common")
EXECUTION_ORDER = {
    2123: CONDITIONS,
    2124: CONDITIONS[1:] + CONDITIONS[:1],
    2125: CONDITIONS[2:] + CONDITIONS[:2],
}
PARENT_SEEDS = (2120, 2121, 2122)
RHO_GRID = (0.125, 0.25, 0.5)
TRAINING_RNG_OFFSET = 1_210_000
ENCODING_RNG_OFFSET = 1_220_000
RECOVERY_SEED_BASE = 1_230_000
RECOVERY_DATASETS = 128
RECOVERY_EPISODES = 256
QUADRATURE_NODES = 32
QUALIFICATION_SEED = 1_240_001
GENERIC_SEED = 1_250_001
GENERIC_ENCODING_SEED = 1_250_003
GENERIC_BOOTSTRAP_SEED_BASE = 1_251_000
COHORTS = 400
COHORT_SIZE = 77
COHORT_SHARD_SIZE = 20
COHORT_SEED_BASE = 6_000_000
COHORT_SEED_STRIDE = 1_000
COHORT_OFFSETS = {
    "cue_seed": 0,
    "support_seed": 200,
    "subject_encoding_seed": 400,
    "encoding_seed": 600,
    "choice_seed": 800,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 6_600_000
CODEBOOK = (-1.0, -1 / 3, 1 / 3, 1.0)


def specification() -> dict:
    if file_sha256(DESIGN) != DESIGN_HASH:
        raise RuntimeError("frozen common-encoding contract changed")
    verify_reference(reference(DESIGN), commit=DESIGN_COMMIT)
    return load_json(DESIGN)


def resolved_specification() -> dict:
    specification()
    spec = deepcopy(parent_specification())
    spec["experiment_id"] = "common_encoding_state_v1"
    spec["seeds"] = {
        "mandatory": list(SEEDS),
        "conditions": list(CONDITIONS),
        "execution_order": {str(seed): list(EXECUTION_ORDER[seed]) for seed in SEEDS},
        "training_rng_offset": TRAINING_RNG_OFFSET,
        "encoding_rng_offset": ENCODING_RNG_OFFSET,
    }
    spec["optimization"].update(trainable_parameters=["raw_eta", "raw_global_gain"])
    spec["evaluation"]["generic"].update(
        episodes=RECOVERY_EPISODES, rng_seed=GENERIC_SEED
    )
    return spec


def cohort_specification(index: int) -> dict:
    if not 0 <= index < COHORTS:
        raise ValueError("unregistered common-encoding cohort")
    spec = resolved_specification()
    base = COHORT_SEED_BASE + COHORT_SEED_STRIDE * index
    spec["evaluation"]["liu"].update(
        {name: base + offset for name, offset in COHORT_OFFSETS.items()}
    )
    spec["evaluation"]["liu"]["subjects"] = COHORT_SIZE
    return spec


def run_directory(seed: int, condition: str):
    if seed not in SEEDS or condition not in CONDITIONS:
        raise ValueError("unregistered common-encoding fit")
    return RUN_ROOT / f"seed-{seed}" / condition
