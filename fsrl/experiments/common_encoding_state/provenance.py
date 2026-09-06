"""Implementation and scientific-input inventory for the common-state pilot."""

from fsrl.experiments.training_strategy.locks import reference
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import DESIGN


def implementation_sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/common_encoding_state").glob("*.py"))
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    paths = {
        DESIGN,
        REPO_ROOT
        / "studies/experience_dependent_plasticity/records/benchmarks/experience_dependent_plasticity_v1.json",
        REPO_ROOT
        / "studies/experience_dependent_plasticity/records/results/experience_dependent_plasticity_v1.json",
        REPO_ROOT
        / "studies/global_decay_replication/records/benchmarks/global_decay_replication_v1.json",
        REPO_ROOT
        / "studies/global_decay_replication/records/results/global_decay_replication_v1.json",
        REPO_ROOT
        / "studies/global_decay_replication/records/benchmarks/artifact_lock.json",
        protocol_path("liu_v2"),
    }
    return [reference(path) for path in sorted(paths)]
