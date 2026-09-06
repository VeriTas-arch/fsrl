"""Locked generic competence and global-decay confirmation."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.experiments.adaptive_plasticity.generic_selection import (
    _group_metrics,
    paired_summary,
)
from fsrl.experiments.adaptive_plasticity.model import make_model
from fsrl.experiments.adaptive_plasticity.reference import (
    occurrence_sensitivity,
    rollout,
)
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.validation_session import reuse_validation

from .encoding import encode_common
from .evidence import ARTIFACT_LOCK, SOURCE_LOCK, validate_artifacts, validate_source
from .inputs import load_group
from .protocol import (
    BOOTSTRAP_SAMPLES,
    CONDITIONS,
    GENERIC_BOOTSTRAP_SEED_BASE,
    RECORDS,
    RUN_ROOT,
    SEEDS,
    resolved_specification,
)

RESULT = RECORDS / "results/generic_confirmation.json"
ARRAYS = RECORDS / "results/generic_confirmation.npz"
LOCK = RECORDS / "benchmarks/generic_confirmation_lock.json"
METRICS = (
    "loss",
    "learned_accuracy",
    "nonlearned_accuracy",
    "h_late",
    "h_late_fixed_parameter",
    "sampled_mean_signed_margin",
    "conditional_mean_signed_margin",
    "mean_margin_variance",
    "mean_signal_to_variance",
)


def load_fit(config: dict, spec: dict, device: str = "cuda"):
    model = make_model("adaptive_eta_resampled", spec, device, scheduler="global")
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32, device=device)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("loaded common-encoding parameters differ")
    return model.requires_grad_(False).eval()


def _models(artifacts: dict, spec: dict) -> tuple[dict, dict]:
    models, runners = {}, {}
    for seed in SEEDS:
        models[seed] = {}
        runners[seed] = {}
        for condition in CONDITIONS:
            model = load_fit(
                artifacts["archives"][f"{seed}/{condition}"]["config"], spec
            )
            models[seed][condition] = model
            runners[seed][condition] = compiled(model)
    return models, runners


def _fixed_h_late(batch, model) -> np.ndarray:
    expected = rollout(
        batch,
        eta=model.eta.item(),
        gain=model.global_gain.item(),
        epsilon=model.epsilon,
        adaptive=False,
        scheduler="global",
        with_sensitivity=True,
    )
    return occurrence_sensitivity(batch, expected["sensitivity"])[1]


def _evaluate_groups(lock: dict, models: dict, runners: dict) -> dict[str, np.ndarray]:
    episodes = 256
    arrays: dict[str, np.ndarray] = {
        name: np.empty((len(SEEDS), len(CONDITIONS), episodes)) for name in METRICS
    }
    arrays["occurrence_l"] = np.empty((len(SEEDS), len(CONDITIONS), episodes, 4))
    arrays["max_reference_error"] = np.zeros((len(SEEDS), len(CONDITIONS)))
    rho = lock["selected_rho"]
    for group in lock["generic_groups"].values():
        base, auxiliary = load_group(group)
        streams = {
            key.removeprefix("latent__"): value
            for key, value in auxiliary.items()
            if key.startswith("latent__")
        }
        indices = auxiliary["episode_indices"]
        for seed_index, seed in enumerate(SEEDS):
            for condition_index, condition in enumerate(CONDITIONS):
                encoded = encode_common(base, condition, rho, streams)[0]
                result = _group_metrics(
                    encoded,
                    base,
                    models[seed][condition],
                    runners[seed][condition],
                    scheduler="global",
                )
                result["h_late_fixed_parameter"] = _fixed_h_late(
                    encoded, models[seed][condition]
                )
                for name, target in arrays.items():
                    if name == "max_reference_error":
                        target[seed_index, condition_index] = max(
                            target[seed_index, condition_index], result[name]
                        )
                    else:
                        target[seed_index, condition_index, indices] = result[name]
    return arrays


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    fits = {}
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(GENERIC_BOOTSTRAP_SEED_BASE + seed)
        counts = rng.multinomial(256, np.full(256, 1 / 256), size=BOOTSTRAP_SAMPLES)
        fits[str(seed)] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            competence = {}
            for endpoint in ("learned_accuracy", "nonlearned_accuracy"):
                value = paired_summary(
                    arrays[endpoint][seed_index, condition_index], counts
                )
                competence[endpoint] = {
                    **value,
                    "passed": value["interval"]["lower"] > 0.5,
                }
            late = paired_summary(
                arrays["h_late"][seed_index, condition_index]
                - arrays["h_late_fixed_parameter"][seed_index, condition_index],
                counts,
            )
            fits[str(seed)][condition] = {
                "competence": competence,
                "late_sensitivity": {
                    "adaptive_minus_fixed_same_parameters": late,
                    "passed": late["interval"]["upper"] < 0,
                },
                "max_reference_error": float(
                    arrays["max_reference_error"][seed_index, condition_index]
                ),
                "passed": all(row["passed"] for row in competence.values())
                and late["interval"]["upper"] < 0,
            }
    return {
        "fits": fits,
        "generic_gate_passed": all(
            condition["passed"] for seed in fits.values() for condition in seed.values()
        ),
    }


def evaluate_generic() -> dict:
    source = validate_source()
    artifacts = validate_artifacts()
    spec = resolved_specification()
    execution = runtime()
    models, runners = _models(artifacts, spec)
    directory = RUN_ROOT / "generic-confirmation"
    with ProspectiveRun.start(
        directory,
        workflow_id="common_encoding_state_v1",
        execution_id="generic-confirmation",
        producer={"module": __name__, "source_commit": source["source_commit"]},
        resolved_config={"runtime": execution},
    ):
        arrays = _evaluate_groups({**source, **artifacts}, models, runners)
        path = directory / "arrays.npz"
        write_arrays(path, arrays)
        result = {
            "source_lock": reference(SOURCE_LOCK),
            "artifact_lock": reference(ARTIFACT_LOCK),
            "selected_rho": artifacts["selected_rho"],
            "arrays": reference(path),
            "summary": summarize(arrays),
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result["summary"]


def lock_generic(directory) -> dict:
    commit = require_pushed_clean()
    source = validate_source()
    artifacts = validate_artifacts()
    validate_complete(directory)
    result = load_json(directory / "result.json")
    with np.load(verify_reference(result["arrays"]), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if result["summary"] != summarize(arrays):
        raise RuntimeError("common-encoding generic summary does not reconstruct")
    if not result["summary"]["generic_gate_passed"]:
        raise RuntimeError("common-encoding generic competence failed")
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=True)
    write_arrays(ARRAYS, arrays)
    stored = {**result, "arrays": reference(ARRAYS)}
    write_json_exclusive(RESULT, stored)
    lock = {
        "source_commit": source["source_commit"],
        "generic_commit": commit,
        "source_lock": reference(SOURCE_LOCK),
        "artifact_lock": reference(ARTIFACT_LOCK),
        "selected_rho": artifacts["selected_rho"],
        "result": reference(RESULT),
        "arrays": reference(ARRAYS),
        "generic_gate_passed": True,
        "liu_evaluated": False,
    }
    write_json_exclusive(LOCK, lock)
    return lock


@reuse_validation
def validate_generic() -> dict:
    source = validate_source()
    artifacts = validate_artifacts()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(LOCK), commit=commit))
    result = load_json(verify_reference(lock["result"], commit=commit))
    with np.load(
        verify_reference(lock["arrays"], commit=commit), allow_pickle=False
    ) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if (
        lock["source_commit"] != source["source_commit"]
        or lock["source_lock"] != reference(SOURCE_LOCK)
        or lock["artifact_lock"] != reference(ARTIFACT_LOCK)
        or lock["selected_rho"] != artifacts["selected_rho"]
        or lock["liu_evaluated"]
        or not lock["generic_gate_passed"]
        or result["summary"] != summarize(arrays)
    ):
        raise RuntimeError("common-encoding generic lock differs")
    return {**lock, "result_record": result}
