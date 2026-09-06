"""Locked generic scheduler identification and stabilization diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.encoding import (
    encode_batch,
    rounding_parameters,
)
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .data import model_tensors
from .evidence import ARTIFACT_LOCK, SOURCE_LOCK, validate_artifacts
from .inputs import load_group, read_arrays
from .model import load_model
from .protocol import (
    BOOTSTRAP_SAMPLES,
    CODEBOOK,
    GENERIC_BOOTSTRAP_SEED_BASE,
    RECORDS,
    RUN_ROOT,
    SEEDS,
    resolved_specification,
)
from .reference import occurrence_sensitivity, rollout

MODES = ("fixed", "adaptive_relation", "adaptive_global")
SCHEDULES = ("balanced", "clustered")
SELECTION_RESULT = RECORDS / "results/generic_scheduler_selection.json"
SELECTION_ARRAYS = RECORDS / "results/generic_scheduler_selection.npz"
SELECTION_LOCK = RECORDS / "benchmarks/scheduler_selection_lock.json"


def paired_summary(values: np.ndarray, counts: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    means = counts @ values / values.size
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "interval": {"lower": float(low), "upper": float(high)},
    }


def _group_metrics(
    batch: ModelBatch,
    raw_batch: ModelBatch,
    model,
    runner,
    *,
    scheduler: str,
) -> dict:
    with torch.no_grad():
        observed = runner(*model_tensors(batch, "cuda"))
    margins = observed[0].cpu().numpy().astype(np.float64)
    parameters = {"eta": model.eta.item(), "gain": model.global_gain.item()}
    expected = rollout(
        batch,
        eta=parameters["eta"],
        gain=parameters["gain"],
        epsilon=model.epsilon,
        adaptive=model.adaptive,
        scheduler=scheduler,
        with_sensitivity=True,
    )
    conditional_mean = rollout(
        raw_batch,
        eta=parameters["eta"],
        gain=parameters["gain"],
        epsilon=model.epsilon,
        adaptive=model.adaptive,
        scheduler=scheduler,
    )["margins"]
    np.testing.assert_allclose(margins, expected["margins"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(observed[1].cpu(), expected["w"], atol=1e-5, rtol=1e-4)
    signs = 2 * batch.arrays["targets"] - 1
    signed_margin = signs * margins
    conditional_signed_margin = signs * conditional_mean
    learned = batch.arrays["learned"]
    _, _, code_variance = rounding_parameters(
        raw_batch.arrays["signed"], np.asarray(CODEBOOK)
    )
    code_variance *= raw_batch.arrays["retention"]
    margin_variance = np.einsum(
        "tsq,ts->sq", np.square(expected["sensitivity"]), code_variance
    )
    signal_to_variance = np.divide(
        np.square(conditional_signed_margin).mean(axis=1),
        margin_variance.mean(axis=1) + np.finfo(np.float64).eps,
    )
    occurrence_l, h_late = occurrence_sensitivity(batch, expected["sensitivity"])
    return {
        "margins": margins,
        "loss": np.logaddexp(0, -signed_margin).mean(axis=1),
        "learned_accuracy": np.asarray(
            [
                (row[mask] > 0).mean()
                for row, mask in zip(signed_margin, learned, strict=True)
            ]
        ),
        "nonlearned_accuracy": np.asarray(
            [
                (row[~mask] > 0).mean()
                for row, mask in zip(signed_margin, learned, strict=True)
            ]
        ),
        "occurrence_l": occurrence_l,
        "h_late": h_late,
        "sampled_mean_signed_margin": signed_margin.mean(axis=1),
        "conditional_mean_signed_margin": conditional_signed_margin.mean(axis=1),
        "mean_margin_variance": margin_variance.mean(axis=1),
        "mean_signal_to_variance": signal_to_variance,
        "max_reference_error": float(np.max(np.abs(margins - expected["margins"]))),
    }


def _models(artifacts: dict, spec: dict) -> tuple[dict, dict]:
    models, runners = {}, {}
    for seed in SEEDS:
        fixed = load_model(
            artifacts["archives"][f"{seed}/fixed_eta_resampled"]["config"], spec
        )
        relation = load_model(
            artifacts["archives"][f"{seed}/adaptive_eta_resampled"]["config"],
            spec,
            scheduler="relation",
        )
        global_model = load_model(
            artifacts["archives"][f"{seed}/adaptive_eta_resampled"]["config"],
            spec,
            scheduler="global",
        )
        models[seed] = {
            "fixed": fixed,
            "adaptive_relation": relation,
            "adaptive_global": global_model,
        }
        runners[seed] = {name: compiled(value) for name, value in models[seed].items()}
    return models, runners


def _evaluate_groups(lock: dict, models: dict, runners: dict) -> dict[str, np.ndarray]:
    episodes = 256
    arrays: dict[str, np.ndarray] = {
        name: np.empty((len(SEEDS), len(SCHEDULES), len(MODES), episodes))
        for name in (
            "loss",
            "learned_accuracy",
            "nonlearned_accuracy",
            "h_late",
            "sampled_mean_signed_margin",
            "conditional_mean_signed_margin",
            "mean_margin_variance",
            "mean_signal_to_variance",
        )
    }
    arrays["occurrence_l"] = np.empty(
        (len(SEEDS), len(SCHEDULES), len(MODES), episodes, 4)
    )
    arrays["max_reference_error"] = np.zeros((len(SEEDS), len(SCHEDULES), len(MODES)))
    for schedule_index, schedule in enumerate(SCHEDULES):
        for group in lock["generic_groups"][schedule].values():
            base, auxiliary = load_group(group)
            encoded = encode_batch(
                base, "resampled", auxiliary["encoding_uniforms"], CODEBOOK
            )[0]
            indices = auxiliary["episode_indices"]
            for seed_index, seed in enumerate(SEEDS):
                for mode_index, mode in enumerate(MODES):
                    scheduler = "global" if mode == "adaptive_global" else "relation"
                    result = _group_metrics(
                        encoded,
                        base,
                        models[seed][mode],
                        runners[seed][mode],
                        scheduler=scheduler,
                    )
                    for name, target in arrays.items():
                        if name == "max_reference_error":
                            target[seed_index, schedule_index, mode_index] = max(
                                target[seed_index, schedule_index, mode_index],
                                result[name],
                            )
                        else:
                            target[seed_index, schedule_index, mode_index, indices] = (
                                result[name]
                            )
    return arrays


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    schedule_index = {name: index for index, name in enumerate(SCHEDULES)}
    mode_index = {name: index for index, name in enumerate(MODES)}
    fits = {}
    specificity = []
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(GENERIC_BOOTSTRAP_SEED_BASE + seed)
        counts = rng.multinomial(256, np.full(256, 1 / 256), size=BOOTSTRAP_SAMPLES)
        clustered = schedule_index["clustered"]
        relation = mode_index["adaptive_relation"]
        global_mode = mode_index["adaptive_global"]
        loss_specificity = paired_summary(
            arrays["loss"][seed_index, clustered, relation]
            - arrays["loss"][seed_index, clustered, global_mode],
            counts,
        )
        h_specificity = paired_summary(
            arrays["h_late"][seed_index, clustered, relation]
            - arrays["h_late"][seed_index, clustered, global_mode],
            counts,
        )
        passed = (
            loss_specificity["interval"]["upper"] < 0
            and h_specificity["interval"]["upper"] < 0
        )
        specificity.append(passed)
        fits[str(seed)] = {
            "relation_specificity": {
                "loss_relation_minus_global": loss_specificity,
                "h_late_relation_minus_global": h_specificity,
                "passed": passed,
            }
        }
    selected = "relation" if all(specificity) else "global"
    selected_mode = mode_index[f"adaptive_{selected}"]
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(GENERIC_BOOTSTRAP_SEED_BASE + seed)
        counts = rng.multinomial(256, np.full(256, 1 / 256), size=BOOTSTRAP_SAMPLES)
        row = fits[str(seed)]
        row["late_sensitivity"] = {}
        row["competence"] = {}
        row["mean_variance_decomposition"] = {}
        for schedule, schedule_position in schedule_index.items():
            h_delta = paired_summary(
                arrays["h_late"][seed_index, schedule_position, selected_mode]
                - arrays["h_late"][seed_index, schedule_position, mode_index["fixed"]],
                counts,
            )
            row["late_sensitivity"][schedule] = {
                "adaptive_minus_fixed": h_delta,
                "passed": h_delta["interval"]["upper"] < 0,
            }
            row["competence"][schedule] = {}
            for endpoint in ("learned_accuracy", "nonlearned_accuracy"):
                value = paired_summary(
                    arrays[endpoint][seed_index, schedule_position, selected_mode],
                    counts,
                )
                row["competence"][schedule][endpoint] = {
                    **value,
                    "passed": value["interval"]["lower"] > 0.5,
                }
            row["mean_variance_decomposition"][schedule] = {
                name: paired_summary(
                    arrays[name][seed_index, schedule_position, selected_mode]
                    - arrays[name][seed_index, schedule_position, mode_index["fixed"]],
                    counts,
                )
                for name in (
                    "conditional_mean_signed_margin",
                    "mean_margin_variance",
                    "mean_signal_to_variance",
                )
            }
        row["passed"] = all(
            value["passed"] for value in row["late_sensitivity"].values()
        ) and all(
            endpoint["passed"]
            for schedule in row["competence"].values()
            for endpoint in schedule.values()
        )
    return {
        "selected_scheduler": selected,
        "relation_specific_supported": all(specificity),
        "fits": fits,
        "generic_gate_passed": all(row["passed"] for row in fits.values()),
    }


def evaluate_generic() -> dict:
    artifacts = validate_artifacts()
    execution = runtime()
    spec = resolved_specification()
    directory = RUN_ROOT / "generic-selection"
    if directory.exists():
        validate_complete(directory)
        return load_json(directory / "result.json")
    models, runners = _models(artifacts, spec)
    with ProspectiveRun.start(
        directory,
        workflow_id="experience_dependent_plasticity_v1",
        execution_id="generic-scheduler-selection",
        producer={"module": __name__, "artifact_lock": reference(ARTIFACT_LOCK)},
        resolved_config={"runtime": execution},
    ):
        arrays = _evaluate_groups(artifacts, models, runners)
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


def lock_selection(directory) -> dict:
    commit = require_pushed_clean()
    artifacts = validate_artifacts()
    validate_complete(directory)
    result = load_json(directory / "result.json")
    source_arrays = read_arrays(result["arrays"])
    if json_ready(summarize(source_arrays)) != {
        key: result[key]
        for key in (
            "selected_scheduler",
            "relation_specific_supported",
            "fits",
            "generic_gate_passed",
        )
    }:
        raise RuntimeError("generic scheduler summary does not reconstruct")
    SELECTION_RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(SELECTION_ARRAYS, source_arrays)
    published = {**result, "arrays": reference(SELECTION_ARRAYS)}
    write_json_exclusive(SELECTION_RESULT, published)
    lock = {
        "source_commit": artifacts["source_commit"],
        "artifact_commit": commit,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "result": reference(SELECTION_RESULT),
        "arrays": reference(SELECTION_ARRAYS),
        "selected_scheduler": result["selected_scheduler"],
        "relation_specific_supported": result["relation_specific_supported"],
        "liu_evaluated": False,
    }
    write_json_exclusive(SELECTION_LOCK, lock)
    return lock


def validate_selection() -> dict:
    validate_artifacts()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(SELECTION_LOCK), commit=commit))
    result = load_json(verify_reference(lock["result"], commit=commit))
    arrays = read_arrays(lock["arrays"])
    summary = summarize(arrays)
    if any(json_ready(summary[key]) != result[key] for key in summary):
        raise RuntimeError("locked generic scheduler result differs")
    if (
        lock["artifact_lock"] != reference(ARTIFACT_LOCK)
        or lock["selected_scheduler"] != result["selected_scheduler"]
        or lock["relation_specific_supported"] != result["relation_specific_supported"]
        or lock["liu_evaluated"]
    ):
        raise RuntimeError("scheduler selection lock differs")
    return {**lock, "summary": summary}
