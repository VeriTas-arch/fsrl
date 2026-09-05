"""Frozen candidate authority, composed with unchanged parent task contracts."""

from copy import deepcopy

from fsrl.experiments.minimal_learner.model import MetricScoreLearner
from fsrl.experiments.minimal_learner.protocol import (
    specification as parent_specification,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "quantized_relational_learner/records"
PROTOCOL = RECORDS / "benchmarks/quantized_relational_learner_v1.json"
PROTOCOL_HASH = "0b8cfb3084af61a5dd045d08094d269f8f9e7a0e9bb7ebcec2a232aea666639b"
PROTOCOL_COMMIT = "d45183f6323eb15e3a537258e6fe4678c06e41a0"
ADMISSION_HASH = "15bfacda90d83ac2befd413f0bdbd6b14c54f830fe52a3c7b3ad7a20cef5546b"
RUN_ROOT = RUNS_ROOT / "quantized_relational_learner_v1"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_HASH:
        raise RuntimeError("quantized learner protocol changed")
    verify_reference(reference(PROTOCOL), commit=PROTOCOL_COMMIT)
    spec = load_json(PROTOCOL)
    for field, expected in (
        ("parent_protocol", spec["parent_protocol_sha256"]),
        ("admission_protocol", ADMISSION_HASH),
    ):
        path = REPO_ROOT / spec[field]
        if file_sha256(path) != expected:
            raise RuntimeError(f"frozen {field} changed")
        verify_reference(reference(path), commit=PROTOCOL_COMMIT)
    return spec


def resolved_specification() -> dict:
    """Parent estimator/config interfaces; only declared candidate overrides."""
    candidate = specification()
    resolved = deepcopy(parent_specification())
    resolved["experiment_id"] = candidate["experiment_id"]
    resolved["seeds"] = candidate["seeds"]
    resolved["optimization"].update(candidate["optimization"])
    for domain in ("generic", "liu"):
        resolved["evaluation"][domain].update(candidate["evaluation"][domain])
    return resolved


def make_model(spec: dict, device: str = "cpu") -> MetricScoreLearner:
    settings = spec["optimization"]
    return MetricScoreLearner(
        spec["task"]["cue_size"],
        with_local=False,
        initial_eta=settings["initial_eta"],
        initial_global_gain=settings["initial_global_gain"],
        initial_local_gain=0,
        epsilon=spec["model"]["epsilon"],
        device=device,
    )


def run_directory(seed: int, condition: str):
    allowed = specification()["seeds"]
    if seed not in allowed["mandatory"] or condition not in allowed["conditions"]:
        raise ValueError("unregistered quantized-learner fit")
    return RUN_ROOT / f"seed-{seed}" / condition
