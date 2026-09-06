"""Source and inherited scientific-input inventory."""

from fsrl.experiments.training_strategy.locks import reference
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import ADMISSION, DESIGN, RECORDS

REPAIR = RECORDS / "benchmarks/generic_selection_repair_v1.json"


def implementation_sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/adaptive_plasticity").glob("*.py"))
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    paths = {
        DESIGN,
        ADMISSION,
        REPAIR,
        REPO_ROOT
        / "studies/quantized_relational_learner/records/benchmarks/quantized_relational_learner_v1.json",
        REPO_ROOT
        / "studies/quantized_relational_learner/records/results/quantized_relational_learner_v1.json",
        REPO_ROOT
        / "studies/main_model_evaluation_v2/records/benchmarks/main_model_evaluation_v2.json",
        REPO_ROOT
        / "studies/main_model_evaluation_v2/records/results/main_model_evaluation_v2.json",
        REPO_ROOT
        / "studies/behavior_reproduction_map/records/benchmarks/model_behavior_reproduction_map_v1.json",
        protocol_path("liu_v2"),
    }
    return [reference(path) for path in sorted(paths)]
