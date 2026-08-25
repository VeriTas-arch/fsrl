"""Registered item-history follow-up to scalar-history reduced algorithm v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dual_state_reduced_algorithm as v1
from . import dual_state_reduced_algorithm_v2 as v2
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION_PATH = resolve_record("benchmarks/dual_state_reduced_algorithm_v3.json")
IMPLEMENTATION_LOCK_PATH = resolve_record("benchmarks/dual_state_reduced_algorithm_v3.lock.json")
RESULT_PATH = resolve_record("results/dual_state_reduced_algorithm_v3.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/dual_state_reduced_algorithm_v3.py",
    "tests": "tests/test_dual_state_reduced_algorithm_v3.py",
}
ORIGINAL_SCALAR_ROLLOUT_BATCH = v2.scalar_history_rollout_batch


@dataclass(frozen=True)
class ItemHistoryParameters:
    A: np.ndarray
    B: np.ndarray


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def validate_sources(
    specification_path: Path = SPECIFICATION_PATH,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    registrations = {
        **specification["registered_sources"],
        "specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["specification_sha256"],
        },
        **lock["implementation_sources"],
    }
    v1_contract = load_json(resolve_record(registrations["v1_contract"]["path"]))
    for seed, artifacts in v1_contract["preservation_artifacts"]["artifacts"].items():
        for name, registration in artifacts.items():
            registrations[f"preservation_{seed}_{name}"] = registration
    checks = []
    for name, registration in registrations.items():
        observed = registered_file_sha256(
            registration["path"], registration["sha256"]
        )
        checks.append(
            {
                "name": name,
                "path": registration["path"],
                "expected": registration["sha256"],
                "observed": observed,
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"item-history source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def item_history_before(evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    increments = evidence * evidence
    return np.concatenate((np.zeros((1, evidence.shape[1])), np.cumsum(increments, axis=0)[:-1]))


def item_history_step(
    states: np.ndarray,
    evidence: np.ndarray,
    history: np.ndarray,
    parameters: ItemHistoryParameters,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    delta = evidence @ parameters.A.T + (history * evidence) @ parameters.B.T
    values = states + delta
    return values - np.mean(values, axis=1, keepdims=True)


def item_history_rollout(
    evidence: np.ndarray, parameters: ItemHistoryParameters
) -> np.ndarray:
    state = np.zeros(8, dtype=np.float64)
    history = np.zeros(8, dtype=np.float64)
    values = [state.copy()]
    for step_evidence in np.asarray(evidence, dtype=np.float64):
        state = item_history_step(
            state[None], step_evidence[None], history[None], parameters
        )[0]
        history += step_evidence * step_evidence
        values.append(state.copy())
    return np.asarray(values)


def item_history_rollout_batch(
    evidence: np.ndarray, parameters: ItemHistoryParameters
) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    state = np.zeros((len(evidence), 8), dtype=np.float64)
    history = np.zeros_like(state)
    for trial_index in range(evidence.shape[1]):
        step_evidence = evidence[:, trial_index]
        state = item_history_step(state, step_evidence, history, parameters)
        history += step_evidence * step_evidence
    return state


def flatten_item_transitions(
    records: list[v1.EpisodeTrajectory],
) -> tuple[np.ndarray, ...]:
    states = []
    evidence = []
    history = []
    targets = []
    episode_index = []
    for index, record in enumerate(records):
        states.append(record.potentials[:-1])
        evidence.append(record.evidence)
        history.append(item_history_before(record.evidence))
        targets.append(record.potentials[1:])
        episode_index.append(np.full(len(record.evidence), index, dtype=np.int64))
    return tuple(
        np.concatenate(values)
        for values in (states, evidence, history, targets, episode_index)
    )


def fit_item_history(
    records: list[v1.EpisodeTrajectory], ridge: float = 1e-6
) -> ItemHistoryParameters:
    states, evidence, history, targets, _ = flatten_item_transitions(records)
    features = np.concatenate((evidence, history * evidence), axis=1)
    delta = targets - states
    coefficients = np.linalg.solve(
        features.T @ features + ridge * np.eye(16), features.T @ delta
    )
    return ItemHistoryParameters(A=coefficients[:8].T, B=coefficients[8:].T)


def evaluate_fold(
    records: list[v1.EpisodeTrajectory],
    parameters: ItemHistoryParameters,
    accumulator: v1.ReducedParameters,
    geometry: v1.Geometry,
    v2_fold: dict,
    *,
    samples: int,
    seed: int,
) -> dict:
    states, evidence, history, targets, episode_index = flatten_item_transitions(records)
    prediction = item_history_step(states, evidence, history, parameters)
    null_prediction = v1.reduced_step(states, evidence, accumulator)
    candidate_error = np.mean((prediction - targets) ** 2, axis=1)
    null_error = np.mean((null_prediction - targets) ** 2, axis=1)
    differences = np.asarray(
        [
            np.mean(candidate_error[episode_index == index] - null_error[episode_index == index])
            for index in range(len(records))
        ]
    )
    transition_energy = float(np.mean((targets - states) ** 2))
    prefix_cosines = []
    terminal_cosines = []
    terminal_agreement = []
    terminal_squared = []
    terminal_energy = []
    terminal_history_norm = []
    full_influences = []
    candidate_influences = []
    full_remote = []
    candidate_remote = []
    null_remote = []
    for record in records:
        candidate_rollout = item_history_rollout(record.evidence, parameters)
        null_rollout = v1.rollout(record.evidence, accumulator)
        norms = np.linalg.norm(record.potentials, axis=1) * np.linalg.norm(candidate_rollout, axis=1)
        cosines = np.divide(
            np.sum(record.potentials * candidate_rollout, axis=1),
            norms,
            out=np.full(len(norms), np.nan),
            where=norms > 1e-12,
        )
        prefix_cosines.extend(cosines[np.isfinite(cosines)])
        terminal_cosines.append(float(cosines[-1]) if np.isfinite(cosines[-1]) else 0.0)
        full_terminal = geometry.incidence @ record.potentials[-1]
        candidate_terminal = geometry.incidence @ candidate_rollout[-1]
        terminal_agreement.append(float(np.mean(np.sign(full_terminal) == np.sign(candidate_terminal))))
        terminal_squared.append(float(np.mean((candidate_rollout[-1] - record.potentials[-1]) ** 2)))
        terminal_energy.append(float(np.mean(record.potentials[-1] ** 2)))
        terminal_history_norm.append(float(np.linalg.norm(np.sum(record.evidence**2, axis=0))))

        loo_evidence = record.evidence.copy()
        first, second = record.loo_relation
        for index, step_evidence in enumerate(loo_evidence):
            if abs(step_evidence[first]) > 0.0 and abs(step_evidence[second]) > 0.0:
                loo_evidence[index] = 0.0
        candidate_loo = item_history_rollout(loo_evidence, parameters)[-1]
        null_loo = v1.rollout(loo_evidence, accumulator)[-1]
        full_influence = record.fields[-1] - record.loo_field
        candidate_influence = geometry.incidence @ (candidate_rollout[-1] - candidate_loo)
        null_influence = geometry.incidence @ (null_rollout[-1] - null_loo)
        full_influences.append(full_influence)
        candidate_influences.append(candidate_influence)
        endpoints = set(record.loo_relation)
        remote = np.asarray([i not in endpoints and j not in endpoints for i, j in geometry.pairs])
        full_remote.extend(full_influence[remote])
        candidate_remote.extend(candidate_influence[remote])
        null_remote.extend(null_influence[remote])
    full_influences = np.asarray(full_influences)
    candidate_influences = np.asarray(candidate_influences)
    full_remote = np.asarray(full_remote)
    candidate_remote = np.asarray(candidate_remote)
    null_remote = np.asarray(null_remote)
    candidate_mse = float(np.mean(candidate_error))
    null_mse = float(np.mean(null_error))
    v2_metrics = v2_fold["metrics"]
    one_step = {
        "candidate_mse": candidate_mse,
        "accumulator_mse": null_mse,
        "frozen_v2_mse": float(v2_metrics["one_step"]["candidate_mse"]),
        "candidate_nrmse": candidate_mse / transition_energy,
        "candidate_to_accumulator_ratio": candidate_mse / null_mse,
        "candidate_minus_accumulator_episode_bootstrap": v1._bootstrap_interval(
            differences, samples, seed
        ),
    }
    trajectory = {
        "median_all_prefix_cosine": float(np.median(prefix_cosines)),
        "mean_terminal_cosine": float(np.mean(terminal_cosines)),
        "mean_terminal_pair_order_agreement": float(np.mean(terminal_agreement)),
        "terminal_centered_rmse_ratio": float(
            np.sqrt(np.mean(terminal_squared)) / np.sqrt(np.mean(terminal_energy))
        ),
    }
    remote_denominator = float(np.mean(np.abs(full_remote)))
    remote = {
        "all_pair_influence_correlation": v1._correlation(full_influences, candidate_influences),
        "remote_magnitude_ratio": float(np.mean(np.abs(candidate_remote)) / remote_denominator),
        "candidate_remote_mse": float(np.mean((candidate_remote - full_remote) ** 2)),
        "accumulator_remote_mse": float(np.mean((null_remote - full_remote) ** 2)),
        "frozen_v2_remote_mse": float(v2_metrics["remote_reassembly"]["candidate_remote_mse"]),
    }
    history_metrics = {
        "terminal_norm_mean": float(np.mean(terminal_history_norm)),
        "terminal_norm_min": float(np.min(terminal_history_norm)),
        "terminal_norm_max": float(np.max(terminal_history_norm)),
        "B_frobenius_norm": float(np.linalg.norm(parameters.B)),
    }
    flags = {
        "one_step": bool(
            one_step["candidate_nrmse"] <= 0.50
            and one_step["candidate_to_accumulator_ratio"] <= 0.80
            and one_step["candidate_minus_accumulator_episode_bootstrap"]["upper"] < 0.0
            and one_step["candidate_mse"] < one_step["frozen_v2_mse"]
        ),
        "trajectory": bool(
            trajectory["median_all_prefix_cosine"] >= 0.95
            and trajectory["mean_terminal_cosine"] >= 0.95
            and trajectory["mean_terminal_pair_order_agreement"] >= 0.90
            and trajectory["terminal_centered_rmse_ratio"] <= 0.50
        ),
        "remote": bool(
            remote["all_pair_influence_correlation"] >= 0.70
            and 0.50 <= remote["remote_magnitude_ratio"] <= 1.50
            and remote["candidate_remote_mse"] < remote["accumulator_remote_mse"]
            and remote["candidate_remote_mse"] < remote["frozen_v2_remote_mse"]
        ),
    }
    return {
        "metrics": {
            "one_step": one_step,
            "trajectory": trajectory,
            "remote_reassembly": remote,
            "item_history": history_metrics,
        },
        "flags": flags,
        "passed": all(flags.values()),
    }


@contextmanager
def bind_item_rollout(parameters: ItemHistoryParameters):
    original = v2.scalar_history_rollout_batch
    v2.scalar_history_rollout_batch = lambda evidence, _parameters: item_history_rollout_batch(
        evidence, parameters
    )
    try:
        yield
    finally:
        v2.scalar_history_rollout_batch = original


def evaluate_preservation(
    seed: int,
    artifact: dict,
    parameters: ItemHistoryParameters,
    geometry: v1.Geometry,
) -> dict:
    with bind_item_rollout(parameters):
        return v2.evaluate_preservation(seed, artifact, parameters, geometry)


def build_result(
    specification_path: Path = SPECIFICATION_PATH,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict:
    source_validation = validate_sources(specification_path, implementation_lock_path)
    specification = load_json(specification_path)
    v1_result = load_json(
        resolve_record(specification["registered_sources"]["v1_result"]["path"])
    )
    v2_result = load_json(
        resolve_record(specification["registered_sources"]["v2_result"]["path"])
    )
    records, artifact_checks = v2.load_development_records(
        resolve_record(
            specification["registered_sources"]["v1_trajectory_artifact"]["path"]
        ),
        v1_result,
    )
    geometry = v1.complete_geometry()
    folds = {}
    for held_out in (2101, 2102, 2103):
        train = [
            record
            for seed in (2101, 2102, 2103)
            if seed != held_out
            for record in records[str(seed)]
        ]
        parameters = fit_item_history(train)
        states, evidence, targets, _ = v1.flatten_transitions(train)
        accumulator = v1.accumulator_fit(evidence, targets - states)
        folds[str(held_out)] = evaluate_fold(
            records[str(held_out)],
            parameters,
            accumulator,
            geometry,
            v2_result["development_folds"][str(held_out)],
            samples=int(specification["bootstrap"]["samples"]),
            seed=int(specification["bootstrap"]["seeds"][str(held_out)]),
        )
    all_records = [record for seed in (2101, 2102, 2103) for record in records[str(seed)]]
    final_parameters = fit_item_history(all_records)
    runtime = v1.configure_runtime()
    v1_specification = load_json(resolve_record("benchmarks/dual_state_reduced_algorithm_v1.json"))
    preservation = {
        str(seed): evaluate_preservation(
            seed,
            v1_specification["preservation_artifacts"]["artifacts"][str(seed)],
            final_parameters,
            geometry,
        )
        for seed in (2104, 2105)
    }
    integrity = {
        "source_validation": source_validation["passed"],
        "trajectory_artifact": artifact_checks["passed"],
        "all_parameters_finite": bool(
            np.all(np.isfinite(final_parameters.A)) and np.all(np.isfinite(final_parameters.B))
        ),
        "rollout_binding_restored": v2.scalar_history_rollout_batch
        is ORIGINAL_SCALAR_ROLLOUT_BATCH,
        "all_local_exact": all(
            value["local_exact_max_abs_error"] <= 1e-6 for value in preservation.values()
        ),
        "gpu_preservation": runtime["device"] == "cuda",
        "bounded_cpu_threads": runtime["torch_intraop_threads"] == 1
        and runtime["torch_interop_threads"] == 1,
    }
    integrity["all_passed"] = all(integrity.values())
    development_passed = all(fold["passed"] for fold in folds.values())
    preservation_passed = all(
        value["double_dissociation_passed"] and value["behavior_preservation_passed"]
        for value in preservation.values()
    )
    if not integrity["all_passed"]:
        outcome = "noninterpretable"
    elif development_passed and preservation_passed:
        outcome = "potential_plus_item_history_algorithm"
    else:
        outcome = "item_history_insufficient"
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "source_validation": source_validation,
        "trajectory_artifact_checks": artifact_checks,
        "runtime": runtime,
        "integrity": integrity,
        "development_folds": folds,
        "final_parameters": {"A": final_parameters.A.tolist(), "B": final_parameters.B.tolist()},
        "preservation": preservation,
        "decision": {
            "outcome": outcome,
            "development_all_passed": development_passed,
            "preservation_all_passed": preservation_passed,
            "claim_boundary": specification["claim_boundary"],
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=SPECIFICATION_PATH)
    parser.add_argument("--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("registered item-history output is write-once")
    result = build_result(arguments.specification, arguments.implementation_lock)
    write_json_exclusive(arguments.output, result)
    print(
        json.dumps(
            {
                "path": str(arguments.output),
                "sha256": file_sha256(arguments.output),
                "outcome": result["decision"]["outcome"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
