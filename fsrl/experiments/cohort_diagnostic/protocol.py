"""Prospective cohort identity and committed parent parameter authority."""

from fsrl.experiments.quantized_learner.evidence import validate_training_record
from fsrl.experiments.quantized_learner.protocol import (
    RECORDS as PARENT_RECORDS,
)
from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "resampled_cohort_diagnostic/records"
RUN_ROOT = RUNS_ROOT / "resampled_cohort_diagnostic_v1"
PROTOCOL = RECORDS / "benchmarks/resampled_cohort_diagnostic_v1.json"
PROTOCOL_HASH = "57a1a4005a0ce4bab294f5f1068d7d566e7e64e061e54751603ab496e9fa6f17"
PROTOCOL_COMMIT = "dd7ced0b96fea66e22ad3ab39b6ab20240d077b5"
PARENT_COMMIT = "92512132be11869a96d6d926db9b70507ef13914"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_HASH:
        raise RuntimeError("cohort protocol changed")
    verify_reference(reference(PROTOCOL), commit=PROTOCOL_COMMIT)
    return load_json(PROTOCOL)


def cohort_settings(index: int) -> dict:
    spec = specification()
    settings = spec["cohorts"]
    if not 0 <= index < settings["count"]:
        raise ValueError("unregistered cohort")
    resolved = resolved_specification()
    base = settings["seed_base"] + settings["cohort_stride"] * index
    resolved["evaluation"]["liu"].update(
        {key: base + offset for key, offset in settings["offsets"].items()}
    )
    resolved["evaluation"]["liu"]["subjects"] = settings["subjects"]
    return resolved


def parameter_records() -> dict:
    path = PARENT_RECORDS / "benchmarks/artifact_lock.json"
    lock = load_json(verify_reference(reference(path), commit=PARENT_COMMIT))
    return {
        str(seed): lock["runs"][f"{seed}/resampled"] for seed in specification()["fits"]
    }


def load_parameters() -> dict:
    configs = {}
    for seed, ref in parameter_records().items():
        archive = load_json(verify_reference(ref, commit=PARENT_COMMIT))
        validate_training_record(
            archive["config"], archive["logs"], int(seed), "resampled"
        )
        configs[seed] = archive["config"]
    return configs


def scientific_inputs() -> list[dict]:
    parent_lock = load_json(PARENT_RECORDS / "benchmarks/source_lock.storage_v2.json")
    records = {row["path"]: row for row in parent_lock["inputs"]}
    for path in (
        PROTOCOL,
        PARENT_RECORDS / "benchmarks/artifact_lock.json",
        PARENT_RECORDS / "results/quantized_relational_learner_v1.json",
    ):
        row = reference(path)
        records[row["path"]] = row
    records.update({row["path"]: row for row in parameter_records().values()})
    return [records[key] for key in sorted(records)]


def implementation_sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/cohort_diagnostic").glob("*.py"))
    paths.extend((REPO_ROOT / ".envrc", REPO_ROOT / "pyproject.toml"))
    return [reference(path) for path in sorted(paths)]
