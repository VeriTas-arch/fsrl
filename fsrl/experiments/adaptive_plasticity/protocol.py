"""Frozen candidate and execution authority."""

from __future__ import annotations

from copy import deepcopy

from fsrl.experiments.quantized_learner.protocol import (
    resolved_specification as parent_specification,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "experience_dependent_plasticity/records"
DESIGN = RECORDS / "benchmarks/experience_dependent_plasticity_v1.json"
DESIGN_HASH = "c850683382231fff2817d1e4b2dde6c44a36daa15e61b902a01025c87eba758e"
DESIGN_COMMIT = "791a6ef59aca34dba268313265447d6df88e8160"
ADMISSION = (
    RECORDS / "benchmarks/experience_dependent_plasticity_v1.execution_admission.json"
)
ADMISSION_HASH = "f95c92eb1f7d197cbf40030b292461c62bb61630477f004a34f2730ecd57e503"
ADMISSION_COMMIT = "63c40eca5fe3b35f449bab53faff6625040d4640"
RUN_ROOT = RUNS_ROOT / "experience_dependent_plasticity_v1"

SEEDS = (2117, 2118, 2119)
CONDITIONS = ("fixed_eta_resampled", "adaptive_eta_resampled")
EXECUTION_ORDER = {
    2117: CONDITIONS,
    2118: CONDITIONS[::-1],
    2119: CONDITIONS,
}
TRAINING_RNG_OFFSET = 1_050_000
ENCODING_RNG_OFFSET = 1_060_000
GENERIC_SEED = 1_070_001
CLUSTER_SEED = 1_070_002
GENERIC_ENCODING_SEED = 1_070_003
COHORTS = 400
COHORT_SIZE = 77
COHORT_SHARD_SIZE = 20
COHORT_SEED_BASE = 4_000_000
COHORT_SEED_STRIDE = 1_000
COHORT_OFFSETS = {
    "cue_seed": 0,
    "support_seed": 200,
    "subject_encoding_seed": 400,
    "encoding_seed": 600,
    "choice_seed": 800,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 4_800_000
GENERIC_BOOTSTRAP_SEED_BASE = 4_700_000
CODEBOOK = (-1.0, -1 / 3, 1 / 3, 1.0)


def specification() -> dict:
    """Load both immutable prospective records through their Git witnesses."""
    for path, digest, commit in (
        (DESIGN, DESIGN_HASH, DESIGN_COMMIT),
        (ADMISSION, ADMISSION_HASH, ADMISSION_COMMIT),
    ):
        if file_sha256(path) != digest:
            raise RuntimeError(
                f"frozen adaptive-plasticity record changed: {path.name}"
            )
        verify_reference(reference(path), commit=commit)
    return load_json(DESIGN)


def resolved_specification() -> dict:
    spec = deepcopy(parent_specification())
    spec["experiment_id"] = "experience_dependent_plasticity_v1"
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
        raise ValueError("unregistered adaptive-plasticity cohort")
    spec = resolved_specification()
    base = COHORT_SEED_BASE + COHORT_SEED_STRIDE * index
    spec["evaluation"]["liu"].update(
        {name: base + offset for name, offset in COHORT_OFFSETS.items()}
    )
    spec["evaluation"]["liu"]["subjects"] = COHORT_SIZE
    return spec


def run_directory(seed: int, condition: str):
    if seed not in SEEDS or condition not in CONDITIONS:
        raise ValueError("unregistered adaptive-plasticity fit")
    return RUN_ROOT / f"seed-{seed}" / condition
