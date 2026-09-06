"""Implementation and scientific-input inventory for the replication."""

from fsrl.experiments.training_strategy.locks import reference
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import DESIGN, RECORDS

REPAIR = RECORDS / "benchmarks/qualification_repair_v1.json"


def implementation_sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend(
        (REPO_ROOT / "tests/experiments/global_decay_replication").glob("*.py")
    )
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    paths = {
        DESIGN,
        REPAIR,
        REPO_ROOT
        / "studies/experience_dependent_plasticity/records/benchmarks/experience_dependent_plasticity_v1.json",
        REPO_ROOT
        / "studies/experience_dependent_plasticity/records/results/experience_dependent_plasticity_v1.json",
        REPO_ROOT
        / "studies/main_model_admission/records/benchmarks/main_model_admission_v1.json",
        REPO_ROOT
        / "studies/behavior_reproduction_map/records/benchmarks/model_behavior_reproduction_map_v1.json",
        protocol_path("liu_v2"),
    }
    return [reference(path) for path in sorted(paths)]
