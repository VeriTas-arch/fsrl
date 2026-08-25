"""Registered functional fast-weight latent-sufficiency audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import dual_state_reduced_algorithm as v1
from .liu_eval import load_retro_checkpoint
from .meta_tasks import GenericRankingTaskGenerator
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION_PATH = (
    resolve_record("benchmarks/functional_fast_weight_latent_sufficiency_v1.json")
)
IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.lock.json")
)
FIT_ARTIFACT_PATH = (
    resolve_record("results/functional_fast_weight_latent_sufficiency_v1.fit.npz")
)
RESULT_PATH = resolve_record("results/functional_fast_weight_latent_sufficiency_v1.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/functional_fast_weight_latent_sufficiency.py",
    "tests": "tests/test_functional_fast_weight_latent_sufficiency.py",
}
RIDGE_FRACTION = 1e-4
RANK_GRID = (1, 2, 4, 7)


@dataclass(frozen=True)
class FunctionalEpisode:
    potentials: np.ndarray
    fields: np.ndarray
    evidence: np.ndarray
    functional_state: np.ndarray
    loo_potentials: np.ndarray
    loo_fields: np.ndarray
    loo_evidence: np.ndarray
    loo_functional_state: np.ndarray
    loo_relation: tuple[int, int]


@dataclass(frozen=True)
class FunctionalPredictor:
    potential_to_state: np.ndarray
    baseline: np.ndarray
    full_state: np.ndarray
    output_directions: np.ndarray
    centered_basis: np.ndarray


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
    for seed, artifacts in specification["development_artifacts"]["artifacts"].items():
        for name, registration in artifacts.items():
            registrations[f"development_{seed}_{name}"] = registration
    checks = []
    for name, registration in registrations.items():
        path = resolve_record(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
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
        raise RuntimeError(f"functional-P source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def configure_runtime() -> dict:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "registered functional-P replay and fit require a visible GPU"
        )
    return {
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def centered_basis(n_items: int = 8) -> np.ndarray:
    contrast = np.eye(n_items, dtype=np.float64)[:, :-1]
    contrast[-1] = -1.0
    basis, _ = np.linalg.qr(contrast)
    return basis


def _functional_snapshot(net, fast_weights: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        values = torch.mul(net.alpha, fast_weights).reshape(fast_weights.shape[0], -1)
    return values.detach().cpu().numpy().astype(np.float32, copy=True)


def _extract_branch(
    net,
    config,
    episodes,
    geometry: v1.Geometry,
    *,
    zero_relations: tuple[tuple[int, int], ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fast_weights = v1._initial_fast_weights(net, config)
    states = [_functional_snapshot(net, fast_weights)]
    fields = [v1._generic_field(net, config, episodes, fast_weights, geometry)]
    evidence = []
    for trial_index in range(len(episodes[0].support_trials)):
        fast_weights, step_evidence = v1._advance_generic(
            net,
            config,
            episodes,
            fast_weights,
            trial_index,
            zero_relations=zero_relations,
        )
        evidence.append(step_evidence)
        states.append(_functional_snapshot(net, fast_weights))
        fields.append(v1._generic_field(net, config, episodes, fast_weights, geometry))
    field_array = np.stack(fields, axis=1)
    return (
        v1.hodge_potential(field_array, geometry),
        field_array,
        np.stack(evidence, axis=1),
        np.stack(states, axis=1),
    )


def _maximum_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(observed, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def extract_backbone(
    checkpoint: Path,
    *,
    seed: int,
    rng_seed: int,
    frozen_artifact: np.lib.npyio.NpzFile,
    geometry: v1.Geometry,
) -> tuple[list[FunctionalEpisode], dict]:
    net, config, info = load_retro_checkpoint(checkpoint, 32)
    generator = GenericRankingTaskGenerator(
        cue_size=config.cs,
        min_edges=7,
        max_edges=10,
        support_blocks=4,
        exclude_liu_graph=True,
        subject_encoding_mode="stable_omission",
    )
    rng = np.random.default_rng(rng_seed)
    prefix = f"development_{seed}"
    frozen_lengths = frozen_artifact[f"{prefix}_lengths"]
    records = []
    identity = {
        "length_max_abs_error": 0.0,
        "natural_potential_max_abs_error": 0.0,
        "natural_field_max_abs_error": 0.0,
        "natural_evidence_max_abs_error": 0.0,
        "loo_terminal_potential_max_abs_error": 0.0,
        "loo_terminal_field_max_abs_error": 0.0,
        "loo_relation_max_abs_error": 0.0,
        "initial_functional_state_max_abs": 0.0,
    }
    offset = 0
    for _ in range(4):
        n_edges = int(rng.integers(7, 11))
        episodes = tuple(generator.sample(rng, n_edges=n_edges) for _ in range(32))
        selected = v1._selected_relations(episodes)
        natural = _extract_branch(net, config, episodes, geometry, zero_relations=None)
        loo = _extract_branch(net, config, episodes, geometry, zero_relations=selected)
        potentials, fields, evidence, functional_state = natural
        loo_potentials, loo_fields, loo_evidence, loo_functional_state = loo
        for subject in range(32):
            index = offset + subject
            length = len(episodes[subject].support_trials)
            identity["length_max_abs_error"] = max(
                identity["length_max_abs_error"],
                abs(float(length - frozen_lengths[index])),
            )
            identity["natural_potential_max_abs_error"] = max(
                identity["natural_potential_max_abs_error"],
                _maximum_error(
                    potentials[subject],
                    frozen_artifact[f"{prefix}_potentials"][index, : length + 1],
                ),
            )
            identity["natural_field_max_abs_error"] = max(
                identity["natural_field_max_abs_error"],
                _maximum_error(
                    fields[subject],
                    frozen_artifact[f"{prefix}_fields"][index, : length + 1],
                ),
            )
            identity["natural_evidence_max_abs_error"] = max(
                identity["natural_evidence_max_abs_error"],
                _maximum_error(
                    evidence[subject],
                    frozen_artifact[f"{prefix}_evidence"][index, :length],
                ),
            )
            identity["loo_terminal_potential_max_abs_error"] = max(
                identity["loo_terminal_potential_max_abs_error"],
                _maximum_error(
                    loo_potentials[subject, -1],
                    frozen_artifact[f"{prefix}_loo_potential"][index],
                ),
            )
            identity["loo_terminal_field_max_abs_error"] = max(
                identity["loo_terminal_field_max_abs_error"],
                _maximum_error(
                    loo_fields[subject, -1],
                    frozen_artifact[f"{prefix}_loo_field"][index],
                ),
            )
            identity["loo_relation_max_abs_error"] = max(
                identity["loo_relation_max_abs_error"],
                _maximum_error(
                    selected[subject],
                    frozen_artifact[f"{prefix}_loo_relation"][index],
                ),
            )
            identity["initial_functional_state_max_abs"] = max(
                identity["initial_functional_state_max_abs"],
                float(np.max(np.abs(functional_state[subject, 0]))),
                float(np.max(np.abs(loo_functional_state[subject, 0]))),
            )
            records.append(
                FunctionalEpisode(
                    potentials=potentials[subject],
                    fields=fields[subject],
                    evidence=evidence[subject],
                    functional_state=functional_state[subject],
                    loo_potentials=loo_potentials[subject],
                    loo_fields=loo_fields[subject],
                    loo_evidence=loo_evidence[subject],
                    loo_functional_state=loo_functional_state[subject],
                    loo_relation=selected[subject],
                )
            )
        offset += 32
        del natural, loo, functional_state, loo_functional_state
        gc.collect()
    identity["all_within_tolerance"] = bool(
        all(value <= 1e-6 for value in identity.values())
    )
    if not identity["all_within_tolerance"]:
        raise RuntimeError(f"seed {seed} failed frozen v1 replay identity: {identity}")
    return records, {
        "checkpoint": {
            "path": info.path,
            "sha256": info.sha256,
            "hidden_size": info.hidden_size,
            "functional_state_dimension": info.hidden_size**2,
        },
        "episodes": len(records),
        "train_episodes": 96,
        "held_out_episodes": 32,
        "identity": identity,
    }


def _flatten(
    records: list[FunctionalEpisode], *, loo: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if loo:
        potentials = [record.loo_potentials for record in records]
        evidence = [record.loo_evidence for record in records]
        states = [record.loo_functional_state for record in records]
    else:
        potentials = [record.potentials for record in records]
        evidence = [record.evidence for record in records]
        states = [record.functional_state for record in records]
    current = np.concatenate([value[:-1] for value in potentials]).astype(np.float32)
    inputs = np.concatenate(evidence).astype(np.float32)
    targets = np.concatenate([value[1:] - value[:-1] for value in potentials]).astype(
        np.float32
    )
    functional = np.concatenate([value[:-1] for value in states]).astype(np.float32)
    episode_index = np.concatenate(
        [
            np.full(len(value), index, dtype=np.int64)
            for index, value in enumerate(evidence)
        ]
    )
    return current, inputs, targets, functional, episode_index


def _ridge_penalty(values: torch.Tensor) -> torch.Tensor:
    return RIDGE_FRACTION * torch.sum(values * values) / values.shape[1]


def _primal_ridge(
    features: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, float]:
    penalty = _ridge_penalty(features)
    gram = features.T @ features
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram + penalty * identity, features.T @ targets)
    return coefficients, float(penalty.detach().cpu())


def _dual_ridge(
    features: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, float]:
    stable_features = features.to(torch.float64)
    stable_targets = targets.to(torch.float64)
    penalty = _ridge_penalty(stable_features)
    gram = stable_features @ stable_features.T
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    factor = torch.linalg.cholesky(gram + penalty * identity)
    dual = torch.cholesky_solve(stable_targets, factor)
    coefficients = stable_features.T @ dual
    return coefficients.to(features.dtype), float(penalty.detach().cpu())


def fit_predictor(
    records: list[FunctionalEpisode], *, device: str = "cuda"
) -> tuple[FunctionalPredictor, dict]:
    current, evidence, target, functional, _ = _flatten(records)
    basis = torch.as_tensor(centered_basis(), dtype=torch.float32, device=device)
    s = torch.as_tensor(current, device=device)
    x = torch.as_tensor(evidence, device=device)
    y = torch.as_tensor(target, device=device) @ basis
    p = torch.as_tensor(functional, device=device)
    features = torch.cat((s, x), dim=1)
    baseline, baseline_penalty = _primal_ridge(features, y)
    potential_to_state, residualization_penalty = _primal_ridge(s, p)
    residual_state = p - s @ potential_to_state
    target_residual = y - features @ baseline
    full_state, full_penalty = _dual_ridge(residual_state, target_residual)
    contribution = residual_state @ full_state
    _, singular_values, right = torch.linalg.svd(contribution, full_matrices=False)
    output_directions = right.T
    predictor = FunctionalPredictor(
        potential_to_state=potential_to_state.detach().cpu().numpy(),
        baseline=baseline.detach().cpu().numpy(),
        full_state=full_state.detach().cpu().numpy(),
        output_directions=output_directions.detach().cpu().numpy(),
        centered_basis=basis.detach().cpu().numpy(),
    )
    fit = {
        "transitions": len(current),
        "functional_state_dimension": int(functional.shape[1]),
        "baseline_ridge_penalty": baseline_penalty,
        "residualization_ridge_penalty": residualization_penalty,
        "full_P_ridge_penalty": full_penalty,
        "functional_variance_residual_fraction": float(
            torch.sum(residual_state * residual_state).detach().cpu()
            / torch.sum(p * p).detach().cpu()
        ),
        "supervised_contribution_singular_values": [
            float(value) for value in singular_values.detach().cpu()
        ],
    }
    del s, x, y, p, features, residual_state, target_residual, contribution
    torch.cuda.empty_cache()
    gc.collect()
    return predictor, fit


def predict_updates(
    records: list[FunctionalEpisode],
    predictor: FunctionalPredictor,
    *,
    loo: bool,
    device: str = "cuda",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    current, evidence, target, functional, episode_index = _flatten(records, loo=loo)
    basis = torch.as_tensor(predictor.centered_basis, device=device)
    s = torch.as_tensor(current, device=device)
    x = torch.as_tensor(evidence, device=device)
    p = torch.as_tensor(functional, device=device)
    potential_to_state = torch.as_tensor(predictor.potential_to_state, device=device)
    baseline_coefficients = torch.as_tensor(predictor.baseline, device=device)
    full_coefficients = torch.as_tensor(predictor.full_state, device=device)
    directions = torch.as_tensor(predictor.output_directions, device=device)
    features = torch.cat((s, x), dim=1)
    residual_state = p - s @ potential_to_state
    baseline_latent = features @ baseline_coefficients
    full_contribution = residual_state @ full_coefficients
    predictions = {"baseline": (baseline_latent @ basis.T).detach().cpu().numpy()}
    predictions["full_P"] = (
        ((baseline_latent + full_contribution) @ basis.T).detach().cpu().numpy()
    )
    for rank in RANK_GRID:
        output = directions[:, :rank]
        contribution = (full_contribution @ output) @ output.T
        predictions[f"rank_{rank}"] = (
            ((baseline_latent + contribution) @ basis.T).detach().cpu().numpy()
        )
    del s, x, p, features, residual_state, baseline_latent, full_contribution
    torch.cuda.empty_cache()
    gc.collect()
    return predictions, target.astype(np.float64), episode_index


def _bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values), np.full(len(values), 1.0 / len(values)), size=samples
    )
    draws = counts @ values / len(values)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(np.mean(values)),
        "lower": float(lower),
        "upper": float(upper),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _episode_errors(
    predictions: np.ndarray, target: np.ndarray, episode_index: np.ndarray
) -> np.ndarray:
    row_error = np.mean((predictions - target) ** 2, axis=1)
    return np.asarray(
        [
            np.mean(row_error[episode_index == index])
            for index in np.unique(episode_index)
        ]
    )


def _terminal_predictions(
    records: list[FunctionalEpisode],
    predictions: dict[str, np.ndarray],
    episode_index: np.ndarray,
    *,
    loo: bool,
) -> dict[str, np.ndarray]:
    values = {name: [] for name in predictions}
    for index, record in enumerate(records):
        initial = record.loo_potentials[0] if loo else record.potentials[0]
        selector = episode_index == index
        for name, updates in predictions.items():
            terminal = initial + np.sum(updates[selector], axis=0)
            values[name].append(terminal - np.mean(terminal))
    return {name: np.asarray(rows) for name, rows in values.items()}


def remote_metrics(
    records: list[FunctionalEpisode],
    natural: dict[str, np.ndarray],
    natural_index: np.ndarray,
    loo: dict[str, np.ndarray],
    loo_index: np.ndarray,
    geometry: v1.Geometry,
) -> dict[str, dict]:
    natural_terminal = _terminal_predictions(records, natural, natural_index, loo=False)
    loo_terminal = _terminal_predictions(records, loo, loo_index, loo=True)
    full_all = []
    full_remote = []
    candidate_all = {name: [] for name in natural}
    candidate_remote = {name: [] for name in natural}
    for index, record in enumerate(records):
        full_influence = record.fields[-1] - record.loo_fields[-1]
        endpoints = set(record.loo_relation)
        remote = np.asarray(
            [
                first not in endpoints and second not in endpoints
                for first, second in geometry.pairs
            ]
        )
        full_all.append(full_influence)
        full_remote.extend(full_influence[remote])
        for name in natural:
            influence = geometry.incidence @ (
                natural_terminal[name][index] - loo_terminal[name][index]
            )
            candidate_all[name].append(influence)
            candidate_remote[name].extend(influence[remote])
    full_all_array = np.asarray(full_all)
    full_remote_array = np.asarray(full_remote)
    denominator = float(np.mean(np.abs(full_remote_array)))
    result = {}
    for name in natural:
        all_values = np.asarray(candidate_all[name])
        remote_values = np.asarray(candidate_remote[name])
        result[name] = {
            "all_pair_influence_correlation": _correlation(full_all_array, all_values),
            "remote_magnitude_ratio": float(
                np.mean(np.abs(remote_values)) / denominator
            ),
            "remote_mse": float(np.mean((remote_values - full_remote_array) ** 2)),
        }
    result["full_neural_mean_absolute_remote"] = denominator
    return result


def evaluate_backbone(
    records: list[FunctionalEpisode],
    predictor: FunctionalPredictor,
    geometry: v1.Geometry,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    device: str,
) -> dict:
    held_out = records[96:]
    natural, target, natural_index = predict_updates(
        held_out, predictor, loo=False, device=device
    )
    loo, _, loo_index = predict_updates(held_out, predictor, loo=True, device=device)
    remote = remote_metrics(held_out, natural, natural_index, loo, loo_index, geometry)
    episode_errors = {
        name: _episode_errors(values, target, natural_index)
        for name, values in natural.items()
    }
    baseline_error = episode_errors["baseline"]
    full_error = episode_errors["full_P"]
    e0 = float(np.mean(baseline_error))
    e_full = float(np.mean(full_error))
    denominator = e0 - e_full
    one_step = {
        "baseline_mse": e0,
        "full_P_mse": e_full,
        "full_P_to_baseline_ratio": e_full / e0,
        "full_P_minus_baseline_episode_bootstrap": _bootstrap_interval(
            full_error - baseline_error, bootstrap_samples, bootstrap_seed
        ),
        "ranks": {},
    }
    full_remote = remote["full_P"]
    baseline_remote = remote["baseline"]
    full_one_step_pass = bool(
        e_full <= 0.90 * e0
        and one_step["full_P_minus_baseline_episode_bootstrap"]["upper"] < 0.0
    )
    full_remote_pass = bool(
        full_remote["all_pair_influence_correlation"] >= 0.70
        and 0.50 <= full_remote["remote_magnitude_ratio"] <= 1.50
        and full_remote["remote_mse"] < baseline_remote["remote_mse"]
    )
    sufficient = []
    for rank in RANK_GRID:
        name = f"rank_{rank}"
        error = episode_errors[name]
        value = float(np.mean(error))
        eta = float((e0 - value) / denominator) if denominator > 0.0 else None
        interval = _bootstrap_interval(
            error - baseline_error, bootstrap_samples, bootstrap_seed + rank
        )
        rank_remote = remote[name]
        passed = bool(
            eta is not None
            and eta >= 0.95
            and interval["upper"] < 0.0
            and rank_remote["all_pair_influence_correlation"] >= 0.70
            and 0.50 <= rank_remote["remote_magnitude_ratio"] <= 1.50
            and rank_remote["remote_mse"] < baseline_remote["remote_mse"]
            and rank_remote["remote_mse"] <= 1.10 * full_remote["remote_mse"]
        )
        one_step["ranks"][str(rank)] = {
            "mse": value,
            "eta_full_P_information": eta,
            "minus_baseline_episode_bootstrap": interval,
            "remote": rank_remote,
            "sufficient": passed,
        }
        if passed:
            sufficient.append(rank)
    rank7_error = float(np.max(np.abs(natural["rank_7"] - natural["full_P"])))
    return {
        "split": {
            "training_episode_indices": [0, 95],
            "held_out_episode_indices": [96, 127],
        },
        "one_step": one_step,
        "remote_reassembly": {
            "baseline": baseline_remote,
            "full_P": full_remote,
            "full_neural_mean_absolute_remote": remote[
                "full_neural_mean_absolute_remote"
            ],
        },
        "full_oracle_flags": {
            "one_step": full_one_step_pass,
            "remote": full_remote_pass,
            "all": full_one_step_pass and full_remote_pass,
        },
        "k_min": min(sufficient) if sufficient else None,
        "rank7_full_P_max_abs_prediction_error": rank7_error,
    }


def _artifact_arrays(
    seed: int, predictor: FunctionalPredictor
) -> dict[str, np.ndarray]:
    prefix = f"seed_{seed}"
    return {
        f"{prefix}_potential_to_state": predictor.potential_to_state,
        f"{prefix}_baseline": predictor.baseline,
        f"{prefix}_full_state": predictor.full_state,
        f"{prefix}_output_directions": predictor.output_directions,
        f"{prefix}_centered_basis": predictor.centered_basis,
    }


def build_result(
    specification_path: Path, implementation_lock_path: Path
) -> tuple[dict, dict[str, np.ndarray]]:
    source_validation = validate_sources(specification_path, implementation_lock_path)
    specification = load_json(specification_path)
    runtime = configure_runtime()
    geometry = v1.complete_geometry()
    v1_contract = load_json(
        resolve_record(specification["registered_sources"]["v1_contract"]["path"])
    )
    v1_trajectory = resolve_record(
        specification["registered_sources"]["v1_trajectory_artifact"]["path"]
    )
    seeds = specification["development_artifacts"]["mandatory_seeds"]
    results = {}
    arrays = {}
    identity = {}
    with np.load(v1_trajectory) as frozen_artifact:
        for seed in seeds:
            artifact = specification["development_artifacts"]["artifacts"][str(seed)]
            rng_seed = int(
                v1_contract["generic_trajectory_contract"]["episode_rng_seeds"][
                    str(seed)
                ]
            )
            records, extraction = extract_backbone(
                resolve_record(artifact["checkpoint"]["path"]),
                seed=seed,
                rng_seed=rng_seed,
                frozen_artifact=frozen_artifact,
                geometry=geometry,
            )
            identity[str(seed)] = extraction
            predictor, fit = fit_predictor(records[:96], device=runtime["device"])
            evaluation = evaluate_backbone(
                records,
                predictor,
                geometry,
                bootstrap_samples=int(specification["bootstrap"]["samples"]),
                bootstrap_seed=int(specification["bootstrap"]["seeds"][str(seed)]),
                device=runtime["device"],
            )
            arrays.update(_artifact_arrays(seed, predictor))
            results[str(seed)] = {"fit": fit, **evaluation}
            del records, predictor
            torch.cuda.empty_cache()
            gc.collect()
    integrity = {
        "source_validation": source_validation["passed"],
        "exact_seed_set": set(results) == {"2101", "2102", "2103"},
        "all_replay_identity": all(
            value["identity"]["all_within_tolerance"] for value in identity.values()
        ),
        "all_finite_parameters": all(
            np.all(np.isfinite(value)) for value in arrays.values()
        ),
        "rank7_reconstructs_full_P": all(
            value["rank7_full_P_max_abs_prediction_error"] <= 1e-5
            for value in results.values()
        ),
        "gpu_runtime": runtime["device"] == "cuda",
        "bounded_cpu_threads": runtime["torch_intraop_threads"] == 1
        and runtime["torch_interop_threads"] == 1,
        "no_human_or_Liu_evaluation": True,
    }
    integrity["all_passed"] = all(integrity.values())
    all_one_step = all(
        value["full_oracle_flags"]["one_step"] for value in results.values()
    )
    all_full = all(value["full_oracle_flags"]["all"] for value in results.values())
    k_values = [value["k_min"] for value in results.values()]
    if not integrity["all_passed"]:
        outcome = "noninterpretable"
    elif all_full and all(value is not None and value <= 4 for value in k_values):
        outcome = "replicated_low_dimensional_functional_P_latent"
    elif all_full and all(value is not None for value in k_values):
        outcome = "replicated_full_output_rank_functional_P_latent"
    elif all_one_step:
        outcome = "one_step_functional_signal_without_remote_sufficiency"
    else:
        outcome = "functional_P_linear_audit_insufficient"
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "source_validation": source_validation,
        "runtime": runtime,
        "integrity": integrity,
        "trajectory_identity": identity,
        "backbones": results,
        "decision": {
            "outcome": outcome,
            "all_full_oracle_prerequisites": all_full,
            "k_min_by_backbone": {
                seed: results[seed]["k_min"] for seed in sorted(results)
            },
            "claim_boundary": specification["claim_boundary"],
        },
    }
    json.dumps(result, allow_nan=False)
    return result, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=SPECIFICATION_PATH)
    parser.add_argument(
        "--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--fit-artifact", type=Path, default=FIT_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.fit_artifact.exists() or arguments.output.exists():
        raise FileExistsError("registered functional-P outputs are write-once")
    result, arrays = build_result(
        arguments.specification, arguments.implementation_lock
    )
    arguments.fit_artifact.parent.mkdir(parents=True, exist_ok=True)
    with arguments.fit_artifact.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result["fit_artifact"] = {
        "path": str(arguments.fit_artifact.relative_to(ROOT)),
        "sha256": file_sha256(arguments.fit_artifact),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }
    write_json_exclusive(arguments.output, result)
    print(
        json.dumps(
            {
                "path": str(arguments.output),
                "sha256": file_sha256(arguments.output),
                "fit_artifact_sha256": result["fit_artifact"]["sha256"],
                "outcome": result["decision"]["outcome"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
