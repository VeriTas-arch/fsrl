"""Registered scalar-history follow-up to reduced-algorithm v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_frozen_retro_checkpoint,
)
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import load_ranking_protocol

from . import dual_state_v1 as v1

ROOT = REPO_ROOT
SPECIFICATION_PATH = resolve_record("benchmarks/dual_state_reduced_algorithm_v2.json")
IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/dual_state_reduced_algorithm_v2.lock.json"
)
RESULT_PATH = resolve_record("results/dual_state_reduced_algorithm_v2.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/dual_state_reduced_algorithm_v2.py",
    "tests": "tests/test_dual_state_reduced_algorithm_v2.py",
}


@dataclass(frozen=True)
class ScalarHistoryParameters:
    A: np.ndarray
    B: np.ndarray


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
        observed = registered_file_sha256(registration["path"], registration["sha256"])
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
        raise RuntimeError(f"scalar-history source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def history_before(evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    increments = 0.5 * np.sum(evidence * evidence, axis=-1)
    return np.concatenate((np.zeros(1), np.cumsum(increments)[:-1]))


def scalar_history_step(
    states: np.ndarray,
    evidence: np.ndarray,
    history: np.ndarray,
    parameters: ScalarHistoryParameters,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64).reshape(-1, 1)
    delta = evidence @ parameters.A.T + (history * evidence) @ parameters.B.T
    values = states + delta
    return values - np.mean(values, axis=1, keepdims=True)


def scalar_history_rollout(
    evidence: np.ndarray, parameters: ScalarHistoryParameters
) -> np.ndarray:
    state = np.zeros(8, dtype=np.float64)
    history = 0.0
    values = [state.copy()]
    for step_evidence in np.asarray(evidence, dtype=np.float64):
        state = scalar_history_step(
            state[None], step_evidence[None], np.asarray([history]), parameters
        )[0]
        history += 0.5 * float(np.dot(step_evidence, step_evidence))
        values.append(state.copy())
    return np.asarray(values)


def scalar_history_rollout_batch(
    evidence: np.ndarray, parameters: ScalarHistoryParameters
) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    state = np.zeros((len(evidence), 8), dtype=np.float64)
    history = np.zeros(len(evidence), dtype=np.float64)
    for trial_index in range(evidence.shape[1]):
        step_evidence = evidence[:, trial_index]
        state = scalar_history_step(state, step_evidence, history, parameters)
        history += 0.5 * np.sum(step_evidence * step_evidence, axis=1)
    return state


def flatten_history_transitions(
    records: list[v1.EpisodeTrajectory],
) -> tuple[np.ndarray, ...]:
    states = []
    evidence = []
    history = []
    targets = []
    episode_index = []
    for index, record in enumerate(records):
        length = len(record.evidence)
        states.append(record.potentials[:-1])
        evidence.append(record.evidence)
        history.append(history_before(record.evidence))
        targets.append(record.potentials[1:])
        episode_index.append(np.full(length, index, dtype=np.int64))
    return tuple(
        np.concatenate(values)
        for values in (states, evidence, history, targets, episode_index)
    )


def fit_scalar_history(
    records: list[v1.EpisodeTrajectory], ridge: float = 1e-6
) -> ScalarHistoryParameters:
    states, evidence, history, targets, _ = flatten_history_transitions(records)
    features = np.concatenate((evidence, history[:, None] * evidence), axis=1)
    delta = targets - states
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    coefficients = np.linalg.solve(gram, features.T @ delta)
    return ScalarHistoryParameters(A=coefficients[:8].T, B=coefficients[8:].T)


def load_development_records(
    trajectory_path: Path, v1_result: dict
) -> tuple[dict[str, list[v1.EpisodeTrajectory]], dict]:
    records = {}
    checks = {
        "valid_slices_finite": True,
        "padding_is_nan": True,
        "potential_gauge_max_abs": 0.0,
        "exact_seed_set": True,
    }
    with np.load(trajectory_path) as artifact:
        for seed in (2101, 2102, 2103):
            prefix = f"development_{seed}"
            lengths = artifact[f"{prefix}_lengths"]
            potentials = artifact[f"{prefix}_potentials"]
            fields = artifact[f"{prefix}_fields"]
            evidence = artifact[f"{prefix}_evidence"]
            loo_potential = artifact[f"{prefix}_loo_potential"]
            loo_field = artifact[f"{prefix}_loo_field"]
            loo_relation = artifact[f"{prefix}_loo_relation"]
            seed_records = []
            local_error = float(
                v1_result["development_folds"][str(seed)]["metrics"][
                    "local_exact_max_abs_error"
                ]
            )
            for index, length_value in enumerate(lengths):
                length = int(length_value)
                used = (
                    potentials[index, : length + 1],
                    fields[index, : length + 1],
                    evidence[index, :length],
                )
                checks["valid_slices_finite"] &= all(
                    np.all(np.isfinite(value)) for value in used
                )
                checks["potential_gauge_max_abs"] = max(
                    checks["potential_gauge_max_abs"],
                    float(np.max(np.abs(np.sum(used[0], axis=1)))),
                )
                if length + 1 < potentials.shape[1]:
                    checks["padding_is_nan"] &= bool(
                        np.all(np.isnan(potentials[index, length + 1 :]))
                        and np.all(np.isnan(fields[index, length + 1 :]))
                    )
                if length < evidence.shape[1]:
                    checks["padding_is_nan"] &= bool(
                        np.all(np.isnan(evidence[index, length:]))
                    )
                seed_records.append(
                    v1.EpisodeTrajectory(
                        potentials=used[0].copy(),
                        fields=used[1].copy(),
                        evidence=used[2].copy(),
                        loo_potential=loo_potential[index].copy(),
                        loo_field=loo_field[index].copy(),
                        loo_relation=tuple(int(value) for value in loo_relation[index]),
                        local_exact_error=local_error,
                        local_identity_raw=np.empty(0),
                        local_kernel_raw=np.empty(0),
                    )
                )
            records[str(seed)] = seed_records
    checks["exact_seed_set"] = set(records) == {"2101", "2102", "2103"}
    checks["passed"] = bool(
        checks["valid_slices_finite"]
        and checks["padding_is_nan"]
        and checks["potential_gauge_max_abs"] <= 1e-12
        and checks["exact_seed_set"]
    )
    return records, checks


def evaluate_fold(
    records: list[v1.EpisodeTrajectory],
    parameters: ScalarHistoryParameters,
    accumulator: v1.ReducedParameters,
    geometry: v1.Geometry,
    v1_fold: dict,
    *,
    samples: int,
    seed: int,
) -> dict:
    states, evidence, history, targets, episode_index = flatten_history_transitions(
        records
    )
    prediction = scalar_history_step(states, evidence, history, parameters)
    null_prediction = v1.reduced_step(states, evidence, accumulator)
    candidate_error = np.mean((prediction - targets) ** 2, axis=1)
    null_error = np.mean((null_prediction - targets) ** 2, axis=1)
    episode_difference = np.asarray(
        [
            np.mean(
                candidate_error[episode_index == index]
                - null_error[episode_index == index]
            )
            for index in range(len(records))
        ]
    )
    transition_energy = float(np.mean((targets - states) ** 2))
    prefix_cosines = []
    terminal_cosines = []
    terminal_agreement = []
    terminal_squared = []
    terminal_energy = []
    terminal_history = []
    full_influences = []
    candidate_influences = []
    null_influences = []
    full_remote = []
    candidate_remote = []
    null_remote = []
    for record in records:
        candidate_rollout = scalar_history_rollout(record.evidence, parameters)
        null_rollout = v1.rollout(record.evidence, accumulator)
        norms = np.linalg.norm(record.potentials, axis=1) * np.linalg.norm(
            candidate_rollout, axis=1
        )
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
        terminal_agreement.append(
            float(np.mean(np.sign(full_terminal) == np.sign(candidate_terminal)))
        )
        terminal_squared.append(
            float(np.mean((candidate_rollout[-1] - record.potentials[-1]) ** 2))
        )
        terminal_energy.append(float(np.mean(record.potentials[-1] ** 2)))
        terminal_history.append(float(np.sum(0.5 * np.sum(record.evidence**2, axis=1))))

        loo_evidence = record.evidence.copy()
        first, second = record.loo_relation
        for trial_index, step_evidence in enumerate(loo_evidence):
            if abs(step_evidence[first]) > 0.0 and abs(step_evidence[second]) > 0.0:
                loo_evidence[trial_index] = 0.0
        candidate_loo = scalar_history_rollout(loo_evidence, parameters)[-1]
        null_loo = v1.rollout(loo_evidence, accumulator)[-1]
        full_influence = record.fields[-1] - record.loo_field
        candidate_influence = geometry.incidence @ (
            candidate_rollout[-1] - candidate_loo
        )
        null_influence = geometry.incidence @ (null_rollout[-1] - null_loo)
        full_influences.append(full_influence)
        candidate_influences.append(candidate_influence)
        null_influences.append(null_influence)
        endpoints = set(record.loo_relation)
        remote = np.asarray(
            [i not in endpoints and j not in endpoints for i, j in geometry.pairs]
        )
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
    v1_metrics = v1_fold["metrics"]
    one_step = {
        "candidate_mse": candidate_mse,
        "accumulator_mse": null_mse,
        "frozen_v1_rank2_mse": float(v1_metrics["one_step"]["candidate_mse"]),
        "candidate_nrmse": candidate_mse / transition_energy,
        "candidate_to_accumulator_ratio": candidate_mse / null_mse,
        "candidate_minus_accumulator_episode_bootstrap": v1._bootstrap_interval(
            episode_difference, samples, seed
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
        "all_pair_influence_correlation": v1._correlation(
            full_influences, candidate_influences
        ),
        "remote_magnitude_ratio": float(
            np.mean(np.abs(candidate_remote)) / remote_denominator
        ),
        "candidate_remote_mse": float(np.mean((candidate_remote - full_remote) ** 2)),
        "accumulator_remote_mse": float(np.mean((null_remote - full_remote) ** 2)),
        "frozen_v1_rank2_remote_mse": float(
            v1_metrics["remote_reassembly"]["candidate_remote_mse"]
        ),
    }
    scalar = {
        "terminal_mean": float(np.mean(terminal_history)),
        "terminal_min": float(np.min(terminal_history)),
        "terminal_max": float(np.max(terminal_history)),
        "B_frobenius_norm": float(np.linalg.norm(parameters.B)),
    }
    flags = {
        "one_step": bool(
            one_step["candidate_nrmse"] <= 0.50
            and one_step["candidate_to_accumulator_ratio"] <= 0.80
            and one_step["candidate_minus_accumulator_episode_bootstrap"]["upper"] < 0.0
            and one_step["candidate_mse"] < one_step["frozen_v1_rank2_mse"]
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
            and remote["candidate_remote_mse"] < remote["frozen_v1_rank2_remote_mse"]
        ),
    }
    return {
        "metrics": {
            "one_step": one_step,
            "trajectory": trajectory,
            "remote_reassembly": remote,
            "scalar_history": scalar,
        },
        "flags": flags,
        "passed": all(flags.values()),
    }


def evaluate_preservation(
    seed: int,
    artifact: dict,
    parameters: ScalarHistoryParameters,
    geometry: v1.Geometry,
) -> dict:
    confirmation = load_json(
        resolve_record("benchmarks/dual_evidence_access_confirmation_v2_4.json")
    )
    evaluation = confirmation["liu_evaluation"]
    protocol = load_ranking_protocol(v1.PROTOCOL_PATH)
    net, config, _ = load_frozen_retro_checkpoint(
        resolve_record(artifact["checkpoint"]["path"]), 77
    )
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        cue_mode=evaluation["cue_mode"],
        subject_encoding_mode=evaluation["subject_encoding_mode"],
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
        test_time_value=2.0 / 3.0,
    )
    evidence = v1._liu_evidence(evaluator)
    potential = scalar_history_rollout_batch(evidence, parameters)
    global_field = potential @ geometry.incidence.T
    full_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    full_field = v1._evaluator_field(evaluator, full_weights, geometry)
    local_raw, identity_raw, local_error = v1._liu_local(evaluator, geometry)
    gain = float(load_json(resolve_record(artifact["gain"]["path"]))["lambda_L"])
    intact = global_field + gain * local_raw

    relations = tuple(protocol.support_pairs_higher_lower)
    global_loo = []
    local_loo = []
    for relation in relations:
        loo_evidence = evidence.copy()
        for subject, schedule in enumerate(evaluator.support_schedules):
            for trial_index, trial in enumerate(schedule):
                if (trial.higher_item, trial.lower_item) == relation:
                    loo_evidence[subject, trial_index] = 0.0
        global_loo.append(
            scalar_history_rollout_batch(loo_evidence, parameters)
            @ geometry.incidence.T
        )
        local_rows = []
        for subject, schedule in enumerate(evaluator.support_schedules):
            values = []
            for trial_index, trial in enumerate(schedule):
                if (trial.higher_item, trial.lower_item) == relation:
                    values.append(0.0)
                    continue
                admission = evaluator._encoding_reliability(subject, trial_index)
                probability = evaluator.subject_encoding_states[
                    subject
                ].relation_reliability(
                    trial.higher_item,
                    trial.lower_item,
                    evaluator.item_rank[trial.lower_item]
                    - evaluator.item_rank[trial.higher_item],
                )
                values.append(
                    trial.signed_magnitude
                    * float(
                        v1.access_factor(
                            np.asarray([admission]), np.asarray([probability])
                        )[0]
                    )
                )
            local_rows.append(
                v1.local_edge_compression(
                    evaluator.cue_codes[subject], schedule, np.asarray(values), geometry
                )["compressed"]
            )
        local_loo.append(np.asarray(local_rows) * gain)
    global_loo = np.asarray(global_loo)
    local_loo = np.asarray(local_loo)
    intact_loo = global_loo + local_loo

    positions = {
        item: index for index, item in enumerate(protocol.true_order_high_to_low)
    }
    true_sign = np.asarray(
        [1.0 if positions[i] < positions[j] else -1.0 for i, j in geometry.pairs]
    )
    learned = np.asarray([pair in protocol.learned_pairs for pair in geometry.pairs])
    nonlearned = ~learned
    temperature = float(evaluation["temperature"])
    exact = {
        "intact_nonlearned": v1._exact_accuracy(
            intact, true_sign, temperature, nonlearned
        ),
        "global_only_nonlearned": v1._exact_accuracy(
            global_field, true_sign, temperature, nonlearned
        ),
        "local_only_learned": v1._exact_accuracy(
            gain * local_raw, true_sign, temperature, learned
        ),
        "local_only_nonlearned": v1._exact_accuracy(
            gain * local_raw, true_sign, temperature, nonlearned
        ),
    }
    remote = {
        "intact": v1._remote_magnitude(intact, intact_loo, relations, geometry),
        "global_only": v1._remote_magnitude(
            global_field, global_loo, relations, geometry
        ),
        "local_only": v1._remote_magnitude(
            gain * local_raw, local_loo, relations, geometry
        ),
    }
    behavior = analyze_sampled_query_policy(
        protocol,
        v1._margin_logits(intact, geometry),
        seed=int(evaluation["choice_seed"]),
        temperature=temperature,
    )
    behavior_record = v1._behavior_flags(seed, behavior)
    reference = load_json(
        resolve_record("results/model_behavior_reproduction_map_v1.json")
    )["networks"][str(seed)]
    matches = {
        name: behavior_record["flags"][name] == reference["flags"][name]
        for name in reference["flags"]
    }
    double = bool(
        exact["global_only_nonlearned"] > 0.70
        and abs(exact["global_only_nonlearned"] - exact["intact_nonlearned"]) <= 0.05
        and exact["local_only_learned"] > 0.55
        and exact["local_only_nonlearned"] <= 0.55
        and remote["local_only"] <= 0.25 * remote["intact"]
        and remote["global_only"] > 0.0
    )
    return {
        "seed": seed,
        "exact_accuracy": exact,
        "remote_reassembly": remote,
        "local_exact_max_abs_error": local_error,
        "reduced_to_full_terminal_potential_correlation": v1._correlation(
            potential, v1.hodge_potential(full_field, geometry)
        ),
        "double_dissociation_passed": double,
        "behavior_flags": behavior_record["flags"],
        "reference_behavior_flags": reference["flags"],
        "behavior_flag_matches": matches,
        "behavior_preservation_passed": all(matches.values()),
        "identity_kernel_mean_abs_difference": float(
            np.mean(np.abs(local_raw - identity_raw))
        ),
    }


def _parameter_json(parameters: ScalarHistoryParameters) -> dict:
    return {"A": parameters.A.tolist(), "B": parameters.B.tolist()}


def build_result(
    specification_path: Path = SPECIFICATION_PATH,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict:
    source_validation = validate_sources(specification_path, implementation_lock_path)
    specification = load_json(specification_path)
    v1_result = load_json(
        resolve_record(specification["registered_sources"]["v1_result"]["path"])
    )
    trajectory_path = resolve_record(
        specification["registered_sources"]["v1_trajectory_artifact"]["path"]
    )
    records, artifact_checks = load_development_records(trajectory_path, v1_result)
    geometry = v1.complete_geometry()
    folds = {}
    for held_out in (2101, 2102, 2103):
        train = [
            record
            for seed in (2101, 2102, 2103)
            if seed != held_out
            for record in records[str(seed)]
        ]
        parameters = fit_scalar_history(train)
        states, evidence, targets, _ = v1.flatten_transitions(train)
        accumulator = v1.accumulator_fit(evidence, targets - states)
        folds[str(held_out)] = evaluate_fold(
            records[str(held_out)],
            parameters,
            accumulator,
            geometry,
            v1_result["development_folds"][str(held_out)],
            samples=int(specification["bootstrap"]["samples"]),
            seed=int(specification["bootstrap"]["seeds"][str(held_out)]),
        )
    all_records = [
        record for seed in (2101, 2102, 2103) for record in records[str(seed)]
    ]
    final_parameters = fit_scalar_history(all_records)
    runtime = v1.configure_runtime()
    v1_specification = load_json(
        resolve_record("benchmarks/dual_state_reduced_algorithm_v1.json")
    )
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
            np.all(np.isfinite(final_parameters.A))
            and np.all(np.isfinite(final_parameters.B))
        ),
        "all_local_exact": all(
            value["local_exact_max_abs_error"] <= 1e-6
            for value in preservation.values()
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
        outcome = "potential_plus_scalar_history_algorithm"
    else:
        outcome = "scalar_history_insufficient"
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "source_validation": source_validation,
        "trajectory_artifact_checks": artifact_checks,
        "runtime": runtime,
        "integrity": integrity,
        "development_folds": folds,
        "final_parameters": _parameter_json(final_parameters),
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
    parser.add_argument(
        "--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("registered scalar-history output is write-once")
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
