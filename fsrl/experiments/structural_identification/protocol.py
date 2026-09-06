"""The single prospective authority for this structural comparison."""

from copy import deepcopy

from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "structural_identification/records"
DESIGN = RECORDS / "benchmarks/structural_identification_v1.json"
DESIGN_HASH = "5b3b6653f2c3ccb82368c17325c683e43c0aae1ab4f1de4112f2bb7822236040"
DESIGN_COMMIT = "f2d5af41b91e635ea2f027f70a48552712af1789"
RUN_ROOT = RUNS_ROOT / "structural_identification_v1"
CODEBOOK = (-1.0, -1 / 3, 1 / 3, 1.0)
STRUCTURES = ("M00", "M10", "M01", "M11")
SCHEDULES = ("decay", "fixed")
SEEDS = (2126, 2127, 2128)


def specification() -> dict:
    if file_sha256(DESIGN) != DESIGN_HASH:
        raise RuntimeError("structural-identification protocol changed")
    return load_json(DESIGN)


def model_specification() -> dict:
    specification()
    spec = deepcopy(resolved_specification())
    spec["experiment_id"] = "structural_identification_v1"
    return spec


def cohort_specification(index: int) -> dict:
    design = specification()["design"]
    if not 0 <= index < design["cohorts"]:
        raise ValueError("unregistered development cohort")
    spec = model_specification()
    base = design["rng"]["cohort_base"] + index * design["rng"]["cohort_stride"]
    spec["evaluation"]["liu"].update(
        {key: base + offset for key, offset in design["cohort_offsets"].items()}
    )
    spec["evaluation"]["liu"]["subjects"] = design["subjects_per_cohort"]
    return spec


def fit_order():
    for schedule in SCHEDULES:
        for index, seed in enumerate(SEEDS):
            for structure in STRUCTURES[index:] + STRUCTURES[:index]:
                yield seed, structure, schedule


def fit_path(seed: int, structure: str, schedule: str):
    if seed not in SEEDS or structure not in STRUCTURES or schedule not in SCHEDULES:
        raise ValueError("unregistered structural fit")
    return RUN_ROOT / "training" / f"{seed}-{structure}-{schedule}"
