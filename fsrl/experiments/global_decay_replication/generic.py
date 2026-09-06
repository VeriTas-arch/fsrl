"""Locked generic competence and late-sensitivity confirmation."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.adaptive_plasticity.generic_selection import (
    _group_metrics,
    paired_summary,
)
from fsrl.experiments.adaptive_plasticity.model import load_model
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evidence import ARTIFACT_LOCK, SOURCE_LOCK, validate_artifacts, validate_source
from .inputs import load_group, read_arrays
from .protocol import (
    BOOTSTRAP_SAMPLES,
    CODEBOOK,
    CONDITIONS,
    GENERIC_BOOTSTRAP_SEED_BASE,
    RECORDS,
    RUN_ROOT,
    SEEDS,
    resolved_specification,
)

MODES = ("fixed_eta_resampled", "adaptive_eta_resampled")
RESULT = RECORDS / "results/generic_confirmation.json"
ARRAYS = RECORDS / "results/generic_confirmation.npz"
LOCK = RECORDS / "benchmarks/generic_confirmation_lock.json"
METRICS = (
    "loss",
    "learned_accuracy",
    "nonlearned_accuracy",
    "h_late",
    "sampled_mean_signed_margin",
    "conditional_mean_signed_margin",
    "mean_margin_variance",
    "mean_signal_to_variance",
)


def _models(artifacts: dict, spec: dict) -> tuple[dict, dict]:
    models, runners = {}, {}
    for seed in SEEDS:
        models[seed] = {}
        runners[seed] = {}
        for condition in CONDITIONS:
            scheduler = (
                "global" if condition == "adaptive_eta_resampled" else "relation"
            )
            model = load_model(
                artifacts["archives"][f"{seed}/{condition}"]["config"],
                spec,
                scheduler=scheduler,
            )
            models[seed][condition] = model
            runners[seed][condition] = compiled(model)
    return models, runners


def _evaluate_groups(lock: dict, models: dict, runners: dict) -> dict[str, np.ndarray]:
    episodes = 256
    arrays: dict[str, np.ndarray] = {
        name: np.empty((len(SEEDS), len(MODES), episodes)) for name in METRICS
    }
    arrays["occurrence_l"] = np.empty((len(SEEDS), len(MODES), episodes, 4))
    arrays["max_reference_error"] = np.zeros((len(SEEDS), len(MODES)))
    for group in lock["generic_groups"].values():
        base, auxiliary = load_group(group)
        encoded = encode_batch(
            base, "resampled", auxiliary["encoding_uniforms"], CODEBOOK
        )[0]
        indices = auxiliary["episode_indices"]
        for seed_index, seed in enumerate(SEEDS):
            for mode_index, condition in enumerate(MODES):
                scheduler = (
                    "global" if condition == "adaptive_eta_resampled" else "relation"
                )
                result = _group_metrics(
                    encoded,
                    base,
                    models[seed][condition],
                    runners[seed][condition],
                    scheduler=scheduler,
                )
                for name, target in arrays.items():
                    if name == "max_reference_error":
                        target[seed_index, mode_index] = max(
                            target[seed_index, mode_index], result[name]
                        )
                    else:
                        target[seed_index, mode_index, indices] = result[name]
    return arrays


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    mode_index = {name: index for index, name in enumerate(MODES)}
    fits = {}
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(GENERIC_BOOTSTRAP_SEED_BASE + seed)
        counts = rng.multinomial(256, np.full(256, 1 / 256), size=BOOTSTRAP_SAMPLES)
        row: dict = {"competence": {}}
        for condition in CONDITIONS:
            index = mode_index[condition]
            row["competence"][condition] = {}
            for endpoint in ("learned_accuracy", "nonlearned_accuracy"):
                value = paired_summary(arrays[endpoint][seed_index, index], counts)
                row["competence"][condition][endpoint] = {
                    **value,
                    "passed": value["interval"]["lower"] > 0.5,
                }
        fixed = mode_index["fixed_eta_resampled"]
        adaptive = mode_index["adaptive_eta_resampled"]
        h_delta = paired_summary(
            arrays["h_late"][seed_index, adaptive]
            - arrays["h_late"][seed_index, fixed],
            counts,
        )
        row["late_sensitivity"] = {
            "adaptive_minus_fixed": h_delta,
            "passed": h_delta["interval"]["upper"] < 0,
        }
        row["mean_variance_decomposition"] = {
            name: paired_summary(
                arrays[name][seed_index, adaptive] - arrays[name][seed_index, fixed],
                counts,
            )
            for name in (
                "conditional_mean_signed_margin",
                "mean_margin_variance",
                "mean_signal_to_variance",
            )
        }
        row["occurrence_l"] = {
            condition: [
                paired_summary(
                    arrays["occurrence_l"][seed_index, mode_index[condition], :, index],
                    counts,
                )
                for index in range(4)
            ]
            for condition in CONDITIONS
        }
        row["passed"] = row["late_sensitivity"]["passed"] and all(
            endpoint["passed"]
            for condition in row["competence"].values()
            for endpoint in condition.values()
        )
        fits[str(seed)] = row
    return {
        "fits": fits,
        "generic_gate_passed": all(row["passed"] for row in fits.values()),
    }


def evaluate_generic() -> dict:
    artifacts = validate_artifacts()
    source = validate_source()
    execution = runtime()
    spec = resolved_specification()
    directory = RUN_ROOT / "generic-confirmation"
    if directory.exists():
        validate_complete(directory)
        return load_json(directory / "result.json")
    models, runners = _models(artifacts, spec)
    with ProspectiveRun.start(
        directory,
        workflow_id="global_decay_replication_v1",
        execution_id="generic-confirmation",
        producer={"module": __name__, "artifact_lock": reference(ARTIFACT_LOCK)},
        resolved_config={"runtime": execution},
    ):
        arrays = _evaluate_groups(source, models, runners)
        path = directory / "arrays.npz"
        write_arrays(path, arrays)
        result = {
            "artifact_lock": reference(ARTIFACT_LOCK),
            "source_lock": reference(SOURCE_LOCK),
            "arrays": reference(path),
            **summarize(arrays),
            "liu_evaluated": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def lock_generic(directory) -> dict:
    commit = require_pushed_clean()
    artifacts = validate_artifacts()
    validate_complete(directory)
    result = load_json(directory / "result.json")
    source_arrays = read_arrays(result["arrays"])
    if json_ready(summarize(source_arrays)) != {
        key: result[key] for key in ("fits", "generic_gate_passed")
    }:
        raise RuntimeError("generic confirmation summary does not reconstruct")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(ARRAYS, source_arrays)
    published = {**result, "arrays": reference(ARRAYS)}
    write_json_exclusive(RESULT, published)
    lock = {
        "source_commit": artifacts["source_commit"],
        "artifact_commit": commit,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "result": reference(RESULT),
        "arrays": reference(ARRAYS),
        "generic_gate_passed": result["generic_gate_passed"],
        "liu_evaluated": False,
    }
    write_json_exclusive(LOCK, lock)
    return lock


def validate_generic() -> dict:
    validate_artifacts()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(LOCK), commit=commit))
    result = load_json(verify_reference(lock["result"], commit=commit))
    arrays = read_arrays(lock["arrays"])
    summary = summarize(arrays)
    if any(json_ready(summary[key]) != result[key] for key in summary):
        raise RuntimeError("locked generic confirmation differs")
    if (
        lock["artifact_lock"] != reference(ARTIFACT_LOCK)
        or lock["generic_gate_passed"] != result["generic_gate_passed"]
        or lock["liu_evaluated"]
    ):
        raise RuntimeError("generic confirmation lock differs")
    return {**lock, "summary": summary}
