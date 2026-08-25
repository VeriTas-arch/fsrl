"""Registered compression test for the confirmed dual-state mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .behavioral import analyze_sampled_query_policy
from .config import NUMRESPONSESTEP
from .conjunctive_local_trace_pilot import _behavior_summaries
from .dual_evidence_access_pilot import access_factor
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from .meta_tasks import GenericRankingTaskGenerator, RankingEpisode
from .meta_train import build_meta_input_sequence
from .model_behavior_reproduction_map import _model_record
from .ranking_protocol import load_ranking_protocol

ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION_PATH = ROOT / "benchmarks" / "dual_state_reduced_algorithm_v1.json"
IMPLEMENTATION_LOCK_PATH = (
    ROOT / "benchmarks" / "dual_state_reduced_algorithm_v1.repair3.lock.json"
)
TRAJECTORY_PATH = ROOT / "results" / "dual_state_reduced_algorithm_v1.trajectories.npz"
RESULT_PATH = ROOT / "results" / "dual_state_reduced_algorithm_v1.json"
PROTOCOL_PATH = ROOT / "benchmarks" / "liu_v2.json"
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/dual_state_reduced_algorithm.py",
    "tests": "tests/test_dual_state_reduced_algorithm.py",
}


@dataclass(frozen=True)
class Geometry:
    pairs: tuple[tuple[int, int], ...]
    incidence: np.ndarray
    score_operator: np.ndarray
    projection: np.ndarray


@dataclass(frozen=True)
class ReducedParameters:
    A: np.ndarray
    U: np.ndarray | None = None
    V: np.ndarray | None = None
    W: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeTrajectory:
    potentials: np.ndarray
    fields: np.ndarray
    evidence: np.ndarray
    loo_potential: np.ndarray
    loo_field: np.ndarray
    loo_relation: tuple[int, int]
    local_exact_error: float
    local_identity_raw: np.ndarray
    local_kernel_raw: np.ndarray


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
            "path": str(specification_path.relative_to(ROOT)),
            "sha256": lock["specification_sha256"],
        },
        **lock["implementation_sources"],
    }
    for group in ("development_artifacts", "preservation_artifacts"):
        for seed, artifacts in specification[group]["artifacts"].items():
            for name, registration in artifacts.items():
                registrations[f"{group}_{seed}_{name}"] = registration
    checks = []
    for name, registration in registrations.items():
        path = ROOT / registration["path"]
        observed = file_sha256(path)
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
        raise RuntimeError(f"reduced-algorithm source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def configure_runtime() -> dict:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("registered extraction and fitting require a visible GPU")
    return {
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def complete_geometry(n_items: int = 8) -> Geometry:
    pairs = tuple(combinations(range(n_items), 2))
    incidence = np.zeros((len(pairs), n_items), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        incidence[index, first] = 1.0
        incidence[index, second] = -1.0
    score_operator = np.linalg.pinv(incidence)
    return Geometry(
        pairs=pairs,
        incidence=incidence,
        score_operator=score_operator,
        projection=incidence @ score_operator,
    )


def hodge_potential(fields: np.ndarray, geometry: Geometry) -> np.ndarray:
    return np.asarray(fields, dtype=np.float64) @ geometry.score_operator.T


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values, axis=-1, keepdims=True)


def _numpy_key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    key = np.outer(left, right) - np.outer(right, left)
    flat = key.reshape(-1).astype(np.float64)
    norm = np.linalg.norm(flat)
    return flat / max(norm, 1e-8)


def local_edge_compression(
    item_codes: np.ndarray,
    trials,
    signed_local_values: np.ndarray,
    geometry: Geometry,
) -> dict[str, np.ndarray | float]:
    codes = np.asarray(item_codes, dtype=np.float64)
    keys = np.stack([_numpy_key(codes[i], codes[j]) for i, j in geometry.pairs])
    kernel = keys @ keys.T
    trace = np.zeros(keys.shape[1], dtype=np.float64)
    edge_state = np.zeros(len(geometry.pairs), dtype=np.float64)
    pair_index = {pair: index for index, pair in enumerate(geometry.pairs)}
    for trial, value in zip(trials, np.asarray(signed_local_values)):
        trace += float(value) * _numpy_key(
            codes[trial.left_item], codes[trial.right_item]
        )
        canonical = tuple(sorted((trial.left_item, trial.right_item)))
        orientation = 1.0 if trial.left_item < trial.right_item else -1.0
        edge_state[pair_index[canonical]] += orientation * float(value)
    direct = trace @ keys.T
    compressed = kernel @ edge_state
    return {
        "kernel": kernel,
        "edge_state": edge_state,
        "direct": direct,
        "compressed": compressed,
        "identity": edge_state.copy(),
        "max_abs_error": float(np.max(np.abs(direct - compressed))),
    }


def accumulator_fit(
    evidence: np.ndarray, delta: np.ndarray, ridge: float = 1e-6
) -> ReducedParameters:
    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(delta, dtype=np.float64)
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, x.T @ y)
    return ReducedParameters(A=coefficients.T)


def unconstrained_bilinear_fit(
    states: np.ndarray,
    evidence: np.ndarray,
    delta: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    interaction = np.einsum("bi,bj->bij", states, evidence).reshape(len(states), -1)
    features = np.concatenate((evidence, interaction), axis=1)
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ delta)


def unconstrained_bilinear_predict(
    coefficients: np.ndarray, states: np.ndarray, evidence: np.ndarray
) -> np.ndarray:
    interaction = np.einsum("bi,bj->bij", states, evidence).reshape(len(states), -1)
    features = np.concatenate((evidence, interaction), axis=1)
    return _center(features @ coefficients)


def reduced_step(
    states: np.ndarray, evidence: np.ndarray, parameters: ReducedParameters
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    evidence = np.asarray(evidence, dtype=np.float64)
    delta = evidence @ parameters.A.T
    if parameters.U is not None:
        delta = delta + ((states @ parameters.V) * (evidence @ parameters.W)) @ parameters.U.T
    return _center(states + delta)


def fit_rank2_candidate(
    states: np.ndarray,
    evidence: np.ndarray,
    targets: np.ndarray,
    accumulator: ReducedParameters,
    *,
    seed: int,
    steps: int = 2500,
    learning_rate: float = 0.01,
    device: str = "cuda",
) -> tuple[ReducedParameters, dict]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dtype = torch.float64
    s = torch.as_tensor(states, dtype=dtype, device=device)
    x = torch.as_tensor(evidence, dtype=dtype, device=device)
    y = torch.as_tensor(targets, dtype=dtype, device=device)
    A = torch.nn.Parameter(torch.as_tensor(accumulator.A, dtype=dtype, device=device))
    U = torch.nn.Parameter(torch.randn(8, 2, generator=generator, dtype=dtype, device=device) * 0.01)
    V = torch.nn.Parameter(torch.randn(8, 2, generator=generator, dtype=dtype, device=device) * 0.01)
    W = torch.nn.Parameter(torch.randn(8, 2, generator=generator, dtype=dtype, device=device) * 0.01)
    optimizer = torch.optim.Adam((A, U, V, W), lr=learning_rate)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = x @ A.T + ((s @ V) * (x @ W)) @ U.T
        prediction = s + delta
        prediction = prediction - torch.mean(prediction, dim=1, keepdim=True)
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            losses.append(float(loss.detach().cpu()))
    parameters = ReducedParameters(
        A=A.detach().cpu().numpy(),
        U=U.detach().cpu().numpy(),
        V=V.detach().cpu().numpy(),
        W=W.detach().cpu().numpy(),
    )
    return parameters, {"initial_loss": losses[0], "final_loss": losses[-1]}


def _initial_fast_weights(net, config) -> torch.Tensor:
    hidden = net.initialZeroState(config.bs)
    eligibility = net.initialZeroET(config.bs)
    fast_weights = net.initialZeroPlasticWeights(config.bs)
    blank = torch.zeros(config.bs, config.inputsize, device=fast_weights.device)
    with torch.no_grad():
        for _ in range(2):
            _, _, _, hidden, eligibility, fast_weights = net(
                blank, hidden, eligibility, fast_weights
            )
    return fast_weights.detach().clone()


def _advance_generic(
    net,
    config,
    episodes: tuple[RankingEpisode, ...],
    fast_weights: torch.Tensor,
    trial_index: int,
    *,
    zero_relations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    hidden = net.initialZeroState(config.bs)
    eligibility = net.initialZeroET(config.bs)
    trials = [episode.support_trials[trial_index] for episode in episodes]
    left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
    right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
    effective = np.asarray(
        [trial.signed_magnitude * trial.encoding_reliability for trial in trials],
        dtype=np.float32,
    )
    if zero_relations is not None:
        for subject, trial in enumerate(trials):
            if (trial.higher_item, trial.lower_item) == zero_relations[subject]:
                effective[subject] = 0.0
    evidence = np.zeros((config.bs, 8), dtype=np.float64)
    rows = np.arange(config.bs)
    evidence[rows, left] += effective
    evidence[rows, right] -= effective
    time_value = trial_index / max(1, len(episodes[0].support_trials) - 1) * (2.0 / 3.0)
    sequence = build_meta_input_sequence(
        config,
        episodes,
        left,
        right,
        effective,
        num_steps=config.triallen,
        time_value=time_value,
        support_trial=True,
    )
    with torch.no_grad():
        for inputs in sequence.unbind():
            _, _, _, hidden, eligibility, fast_weights = net(
                inputs, hidden, eligibility, fast_weights
            )
    return fast_weights.detach().clone(), evidence


def _generic_field(
    net,
    config,
    episodes: tuple[RankingEpisode, ...],
    fast_weights: torch.Tensor,
    geometry: Geometry,
) -> np.ndarray:
    fields = np.empty((config.bs, len(geometry.pairs)), dtype=np.float64)
    for pair_index, (first, second) in enumerate(geometry.pairs):
        oriented_margins = []
        for left_value, right_value in ((first, second), (second, first)):
            hidden = net.initialZeroState(config.bs)
            eligibility = net.initialZeroET(config.bs)
            left = np.full(config.bs, left_value, dtype=np.int64)
            right = np.full(config.bs, right_value, dtype=np.int64)
            sequence = build_meta_input_sequence(
                config,
                episodes,
                left,
                right,
                np.zeros(config.bs, dtype=np.float32),
                num_steps=NUMRESPONSESTEP + 1,
                time_value=2.0 / 3.0,
                support_trial=False,
            )
            response = None
            with torch.no_grad():
                for inputs in sequence.unbind():
                    logits, _, _, hidden, eligibility, _ = net(
                        inputs, hidden, eligibility, fast_weights
                    )
                    response = logits
            oriented_margins.append((response[:, 1] - response[:, 0]).cpu().numpy())
        fields[:, pair_index] = 0.5 * (oriented_margins[0] - oriented_margins[1])
    return fields


def _selected_relations(episodes: tuple[RankingEpisode, ...]) -> tuple[tuple[int, int], ...]:
    selected = []
    for episode in episodes:
        retained = sorted(
            {
                (trial.higher_item, trial.lower_item)
                for trial in episode.support_trials
                if trial.encoding_reliability > 0.0
            }
        )
        if not retained:
            raise RuntimeError("generic episode has no globally retained relation")
        selected.append(retained[0])
    return tuple(selected)


def _generic_local_values(episode: RankingEpisode) -> np.ndarray:
    rank = {item: index for index, item in enumerate(episode.true_order_high_to_low)}
    values = []
    for trial in episode.support_trials:
        distance = rank[trial.lower_item] - rank[trial.higher_item]
        probability = episode.subject_encoding.relation_reliability(
            trial.higher_item, trial.lower_item, distance
        )
        admission = float(
            access_factor(
                np.asarray([trial.encoding_reliability]), np.asarray([probability])
            )[0]
        )
        values.append(trial.signed_magnitude * admission)
    return np.asarray(values, dtype=np.float64)


def extract_development_seed(
    checkpoint: Path,
    *,
    rng_seed: int,
    batches: int,
    geometry: Geometry,
) -> list[EpisodeTrajectory]:
    net, config, _ = load_retro_checkpoint(checkpoint, 32)
    generator = GenericRankingTaskGenerator(
        cue_size=config.cs,
        min_edges=7,
        max_edges=10,
        support_blocks=4,
        exclude_liu_graph=True,
        subject_encoding_mode="stable_omission",
    )
    rng = np.random.default_rng(rng_seed)
    records: list[EpisodeTrajectory] = []
    for _ in range(batches):
        n_edges = int(rng.integers(7, 11))
        episodes = tuple(generator.sample(rng, n_edges=n_edges) for _ in range(32))
        selected = _selected_relations(episodes)
        fast_weights = _initial_fast_weights(net, config)
        natural_fields = [_generic_field(net, config, episodes, fast_weights, geometry)]
        evidence = []
        for trial_index in range(len(episodes[0].support_trials)):
            fast_weights, step_evidence = _advance_generic(
                net, config, episodes, fast_weights, trial_index
            )
            evidence.append(step_evidence)
            natural_fields.append(
                _generic_field(net, config, episodes, fast_weights, geometry)
            )
        fields = np.stack(natural_fields, axis=1)
        potentials = hodge_potential(fields, geometry)

        loo_weights = _initial_fast_weights(net, config)
        for trial_index in range(len(episodes[0].support_trials)):
            loo_weights, _ = _advance_generic(
                net,
                config,
                episodes,
                loo_weights,
                trial_index,
                zero_relations=selected,
            )
        loo_field = _generic_field(net, config, episodes, loo_weights, geometry)
        loo_potential = hodge_potential(loo_field, geometry)
        evidence_array = np.stack(evidence, axis=1)
        for subject, episode in enumerate(episodes):
            local = local_edge_compression(
                episode.item_codes,
                episode.support_trials,
                _generic_local_values(episode),
                geometry,
            )
            records.append(
                EpisodeTrajectory(
                    potentials=potentials[subject],
                    fields=fields[subject],
                    evidence=evidence_array[subject],
                    loo_potential=loo_potential[subject],
                    loo_field=loo_field[subject],
                    loo_relation=selected[subject],
                    local_exact_error=float(local["max_abs_error"]),
                    local_identity_raw=np.asarray(local["identity"]),
                    local_kernel_raw=np.asarray(local["compressed"]),
                )
            )
    return records


def flatten_transitions(records: list[EpisodeTrajectory]) -> tuple[np.ndarray, ...]:
    states = np.concatenate([record.potentials[:-1] for record in records])
    evidence = np.concatenate([record.evidence for record in records])
    targets = np.concatenate([record.potentials[1:] for record in records])
    episode_index = np.concatenate(
        [np.full(len(record.evidence), index, dtype=np.int64) for index, record in enumerate(records)]
    )
    return states, evidence, targets, episode_index


def rollout(evidence: np.ndarray, parameters: ReducedParameters, initial=None) -> np.ndarray:
    state = np.zeros(8, dtype=np.float64) if initial is None else _center(initial)
    values = [state.copy()]
    for step_evidence in evidence:
        state = reduced_step(state[None], step_evidence[None], parameters)[0]
        values.append(state.copy())
    return np.asarray(values)


def _bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(values), np.full(len(values), 1.0 / len(values)), size=samples)
    draws = counts @ values / len(values)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"point": float(np.mean(values)), "lower": float(lower), "upper": float(upper)}


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_development_fold(
    records: list[EpisodeTrajectory],
    accumulator: ReducedParameters,
    candidate: ReducedParameters,
    unconstrained: np.ndarray,
    geometry: Geometry,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    states, evidence, targets, episode_index = flatten_transitions(records)
    delta = targets - states
    null_one = reduced_step(states, evidence, accumulator)
    candidate_one = reduced_step(states, evidence, candidate)
    unconstrained_one = states + unconstrained_bilinear_predict(
        unconstrained, states, evidence
    )
    null_error = np.mean((null_one - targets) ** 2, axis=1)
    candidate_error = np.mean((candidate_one - targets) ** 2, axis=1)
    unconstrained_error = np.mean((unconstrained_one - targets) ** 2, axis=1)
    energy = float(np.mean(delta**2))
    episode_differences = np.asarray(
        [
            np.mean(candidate_error[episode_index == index] - null_error[episode_index == index])
            for index in range(len(records))
        ]
    )

    prefix_cosines = []
    terminal_cosines = []
    terminal_agreement = []
    terminal_squared = []
    terminal_energy = []
    full_influences = []
    candidate_influences = []
    null_influences = []
    full_remote = []
    candidate_remote = []
    null_remote = []
    teacher_terminal_cosines = []
    for record in records:
        candidate_rollout = rollout(record.evidence, candidate)
        null_rollout = rollout(record.evidence, accumulator)
        teacher_rollout = rollout(record.evidence, candidate, record.potentials[0])
        norms = np.linalg.norm(record.potentials, axis=1) * np.linalg.norm(
            candidate_rollout, axis=1
        )
        valid = norms > 1e-12
        cosines = np.divide(
            np.sum(record.potentials * candidate_rollout, axis=1),
            norms,
            out=np.full(len(norms), np.nan),
            where=valid,
        )
        prefix_cosines.extend(cosines[np.isfinite(cosines)])
        terminal_cosines.append(float(cosines[-1]) if np.isfinite(cosines[-1]) else 0.0)
        teacher_norm = np.linalg.norm(record.potentials[-1]) * np.linalg.norm(teacher_rollout[-1])
        teacher_terminal_cosines.append(
            0.0
            if teacher_norm <= 1e-12
            else float(np.dot(record.potentials[-1], teacher_rollout[-1]) / teacher_norm)
        )
        full_terminal_field = geometry.incidence @ record.potentials[-1]
        candidate_terminal_field = geometry.incidence @ candidate_rollout[-1]
        terminal_agreement.append(float(np.mean(np.sign(full_terminal_field) == np.sign(candidate_terminal_field))))
        terminal_squared.append(float(np.mean((candidate_rollout[-1] - record.potentials[-1]) ** 2)))
        terminal_energy.append(float(np.mean(record.potentials[-1] ** 2)))

        loo_evidence = record.evidence.copy()
        for step, step_evidence in enumerate(loo_evidence):
            first, second = record.loo_relation
            if abs(step_evidence[first]) > 0.0 and abs(step_evidence[second]) > 0.0:
                loo_evidence[step] = 0.0
        candidate_loo = rollout(loo_evidence, candidate)[-1]
        null_loo = rollout(loo_evidence, accumulator)[-1]
        full_influence = record.fields[-1] - record.loo_field
        candidate_influence = geometry.incidence @ (candidate_rollout[-1] - candidate_loo)
        null_influence = geometry.incidence @ (null_rollout[-1] - null_loo)
        full_influences.append(full_influence)
        candidate_influences.append(candidate_influence)
        null_influences.append(null_influence)
        endpoints = set(record.loo_relation)
        remote = np.asarray(
            [first not in endpoints and second not in endpoints for first, second in geometry.pairs]
        )
        full_remote.extend(full_influence[remote])
        candidate_remote.extend(candidate_influence[remote])
        null_remote.extend(null_influence[remote])

    full_influences = np.asarray(full_influences)
    candidate_influences = np.asarray(candidate_influences)
    null_influences = np.asarray(null_influences)
    full_remote = np.asarray(full_remote)
    candidate_remote = np.asarray(candidate_remote)
    null_remote = np.asarray(null_remote)
    candidate_mse = float(np.mean(candidate_error))
    null_mse = float(np.mean(null_error))
    remote_denominator = float(np.mean(np.abs(full_remote)))
    metrics = {
        "one_step": {
            "candidate_mse": candidate_mse,
            "accumulator_mse": null_mse,
            "unconstrained_bilinear_mse": float(np.mean(unconstrained_error)),
            "candidate_nrmse": candidate_mse / energy,
            "candidate_to_accumulator_ratio": candidate_mse / null_mse,
            "candidate_minus_accumulator_episode_bootstrap": _bootstrap_interval(
                episode_differences, bootstrap_samples, bootstrap_seed
            ),
        },
        "trajectory": {
            "median_all_prefix_cosine": float(np.median(prefix_cosines)),
            "mean_terminal_cosine": float(np.mean(terminal_cosines)),
            "mean_teacher_initialized_terminal_cosine": float(np.mean(teacher_terminal_cosines)),
            "mean_terminal_pair_order_agreement": float(np.mean(terminal_agreement)),
            "terminal_centered_rmse_ratio": float(
                np.sqrt(np.mean(terminal_squared)) / np.sqrt(np.mean(terminal_energy))
            ),
        },
        "remote_reassembly": {
            "all_pair_influence_correlation": _correlation(full_influences, candidate_influences),
            "remote_magnitude_ratio": float(np.mean(np.abs(candidate_remote)) / remote_denominator),
            "candidate_remote_mse": float(np.mean((candidate_remote - full_remote) ** 2)),
            "accumulator_remote_mse": float(np.mean((null_remote - full_remote) ** 2)),
            "full_mean_absolute_remote": remote_denominator,
        },
        "full_terminal_hodge_fraction_mean": float(
            np.mean(
                [
                    np.sum((record.fields[-1] @ geometry.projection.T) ** 2)
                    / np.sum(record.fields[-1] ** 2)
                    for record in records
                ]
            )
        ),
        "local_exact_max_abs_error": float(max(record.local_exact_error for record in records)),
    }
    one = metrics["one_step"]
    trajectory = metrics["trajectory"]
    remote = metrics["remote_reassembly"]
    flags = {
        "one_step_state_dependence": bool(
            one["candidate_nrmse"] <= 0.50
            and one["candidate_to_accumulator_ratio"] <= 0.80
            and one["candidate_minus_accumulator_episode_bootstrap"]["upper"] < 0.0
        ),
        "prefix_trajectory": bool(
            trajectory["median_all_prefix_cosine"] >= 0.95
            and trajectory["mean_terminal_cosine"] >= 0.95
            and trajectory["mean_terminal_pair_order_agreement"] >= 0.90
            and trajectory["terminal_centered_rmse_ratio"] <= 0.50
        ),
        "remote_reassembly": bool(
            remote["all_pair_influence_correlation"] >= 0.70
            and 0.50 <= remote["remote_magnitude_ratio"] <= 1.50
            and remote["candidate_remote_mse"] < remote["accumulator_remote_mse"]
        ),
        "local_exactness": metrics["local_exact_max_abs_error"] <= 1e-6,
    }
    return {"metrics": metrics, "flags": flags, "passed": all(flags.values())}


def _fit_models(
    records: list[EpisodeTrajectory], *, seed: int, device: str
) -> tuple[ReducedParameters, ReducedParameters, np.ndarray, dict]:
    states, evidence, targets, _ = flatten_transitions(records)
    delta = targets - states
    accumulator = accumulator_fit(evidence, delta)
    candidate, fit = fit_rank2_candidate(
        states,
        evidence,
        targets,
        accumulator,
        seed=seed,
        device=device,
    )
    unconstrained = unconstrained_bilinear_fit(states, evidence, delta)
    return accumulator, candidate, unconstrained, fit


def antisymmetric_field_from_margin_bundle(
    bundle: dict[tuple[int, int], float], geometry: Geometry
) -> np.ndarray:
    return np.asarray(
        [
            0.5 * (float(bundle[(first, second)]) - float(bundle[(second, first)]))
            for first, second in geometry.pairs
        ],
        dtype=np.float64,
    )


def _evaluator_field(evaluator: FrozenFastWeightEvaluator, fast_weights: torch.Tensor, geometry: Geometry) -> np.ndarray:
    ordered = tuple(
        oriented
        for first, second in geometry.pairs
        for oriented in ((first, second), (second, first))
    )
    logits = evaluator.readout_logits(
        fast_weights, tuple(ordered for _ in range(evaluator.config.bs))
    )
    fields = np.empty((evaluator.config.bs, len(geometry.pairs)), dtype=np.float64)
    for subject, bundle in enumerate(logits):
        fields[subject] = antisymmetric_field_from_margin_bundle(bundle, geometry)
    return fields


def _liu_evidence(evaluator: FrozenFastWeightEvaluator) -> np.ndarray:
    values = np.zeros((evaluator.config.bs, evaluator.protocol.support_trials, 8), dtype=np.float64)
    rows = np.arange(evaluator.config.bs)
    for trial_index in range(evaluator.protocol.support_trials):
        trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [
                trial.signed_magnitude * evaluator._encoding_reliability(subject, trial_index)
                for subject, trial in enumerate(trials)
            ]
        )
        values[rows, trial_index, left] += signed
        values[rows, trial_index, right] -= signed
    return values


def rollout_batch(evidence: np.ndarray, parameters: ReducedParameters) -> np.ndarray:
    state = np.zeros((len(evidence), 8), dtype=np.float64)
    for trial_index in range(evidence.shape[1]):
        state = reduced_step(state, evidence[:, trial_index], parameters)
    return state


def _liu_local(
    evaluator: FrozenFastWeightEvaluator, geometry: Geometry
) -> tuple[np.ndarray, np.ndarray, float]:
    compressed = []
    identity = []
    max_error = 0.0
    for subject, schedule in enumerate(evaluator.support_schedules):
        values = []
        for trial_index, trial in enumerate(schedule):
            admission = evaluator._encoding_reliability(subject, trial_index)
            probability = evaluator.subject_encoding_states[subject].relation_reliability(
                trial.higher_item,
                trial.lower_item,
                evaluator.item_rank[trial.lower_item] - evaluator.item_rank[trial.higher_item],
            )
            values.append(
                trial.signed_magnitude
                * float(access_factor(np.asarray([admission]), np.asarray([probability]))[0])
            )
        result = local_edge_compression(
            evaluator.cue_codes[subject], schedule, np.asarray(values), geometry
        )
        compressed.append(result["compressed"])
        identity.append(result["identity"])
        max_error = max(max_error, float(result["max_abs_error"]))
    return np.asarray(compressed), np.asarray(identity), max_error


def _margin_logits(fields: np.ndarray, geometry: Geometry) -> tuple[dict, ...]:
    outputs = []
    for row in fields:
        bundle = {}
        for index, (first, second) in enumerate(geometry.pairs):
            margin = float(row[index])
            bundle[(first, second)] = margin
            bundle[(second, first)] = -margin
        outputs.append(bundle)
    return tuple(outputs)


def _human_intervals() -> tuple[dict, dict, dict, dict]:
    map_spec = load_json(ROOT / "benchmarks" / "model_behavior_reproduction_map_v1.json")
    map_result = load_json(ROOT / "results" / "model_behavior_reproduction_map_v1.json")
    benchmark_registration = map_spec["registered_sources"]["human_benchmark"]
    benchmark = load_json(ROOT / benchmark_registration["path"])
    intervals = {
        name: {"lower": float(value["lower"]), "upper": float(value["upper"])}
        for name, value in benchmark["bootstrap"]["metrics"].items()
    }
    human = map_result["human_reference"]
    return (
        map_spec,
        intervals,
        human["serial_position_effect"]["interval"],
        human["inter_subject_ranking_diversity"]["interval"],
    )


def _behavior_flags(seed: int, behavior: dict) -> dict:
    map_spec, intervals, serial_interval, tau_interval = _human_intervals()
    evaluation = load_json(
        ROOT / "benchmarks" / "dual_evidence_access_confirmation_v2_4.json"
    )["liu_evaluation"]
    subjects = int(evaluation["subjects"])
    counts = (
        np.random.default_rng(int(evaluation["bootstrap_seeds"][str(seed)]))
        .multinomial(
            subjects,
            np.full(subjects, 1.0 / subjects),
            size=int(evaluation["bootstrap_samples"]),
        )
        .astype(np.float64)
    )
    behavior["participant_bootstrap"] = _behavior_summaries(
        behavior, counts, float(evaluation["bootstrap_interval"])
    )
    pseudo = {"seeds": {str(seed): {"behavior": {"dual_access_matched": behavior}}}}
    record = _model_record(
        pseudo,
        str(seed),
        intervals,
        serial_interval,
        tau_interval,
        map_spec,
    )
    return record


def _exact_accuracy(field: np.ndarray, true_sign: np.ndarray, temperature: float, mask: np.ndarray) -> float:
    scaled = np.clip(field * true_sign[None] / temperature, -700.0, 700.0)
    probabilities = 1.0 / (1.0 + np.exp(-scaled))
    return float(np.mean(probabilities[:, mask]))


def _remote_magnitude(
    intact: np.ndarray,
    loo: np.ndarray,
    relations: tuple[tuple[int, int], ...],
    geometry: Geometry,
) -> float:
    values = []
    for relation_index, relation in enumerate(relations):
        endpoints = set(relation)
        mask = np.asarray(
            [first not in endpoints and second not in endpoints for first, second in geometry.pairs]
        )
        values.append(np.abs(intact - loo[relation_index])[:, mask])
    return float(np.mean(values))


def evaluate_preservation_seed(
    seed: int,
    artifact: dict,
    parameters: ReducedParameters,
    specification: dict,
    geometry: Geometry,
) -> tuple[dict, dict[str, np.ndarray]]:
    evaluation = load_json(ROOT / "benchmarks" / "dual_evidence_access_confirmation_v2_4.json")["liu_evaluation"]
    protocol = load_ranking_protocol(PROTOCOL_PATH)
    net, config, _ = load_retro_checkpoint(ROOT / artifact["checkpoint"]["path"], 77)
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
    evidence = _liu_evidence(evaluator)
    potential = rollout_batch(evidence, parameters)
    global_field = potential @ geometry.incidence.T
    full_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    full_field = _evaluator_field(evaluator, full_weights, geometry)
    local_raw, identity_raw, local_error = _liu_local(evaluator, geometry)
    gain = float(load_json(ROOT / artifact["gain"]["path"])["lambda_L"])
    intact = global_field + gain * local_raw
    identity_intact = global_field + gain * identity_raw

    relations = tuple(protocol.support_pairs_higher_lower)
    global_loo = []
    local_loo = []
    for relation in relations:
        loo_evidence = evidence.copy()
        for subject, schedule in enumerate(evaluator.support_schedules):
            for trial_index, trial in enumerate(schedule):
                if (trial.higher_item, trial.lower_item) == relation:
                    loo_evidence[subject, trial_index] = 0.0
        global_loo.append(rollout_batch(loo_evidence, parameters) @ geometry.incidence.T)
        local_rows = []
        for subject, schedule in enumerate(evaluator.support_schedules):
            values = []
            for trial_index, trial in enumerate(schedule):
                if (trial.higher_item, trial.lower_item) == relation:
                    values.append(0.0)
                    continue
                admission = evaluator._encoding_reliability(subject, trial_index)
                probability = evaluator.subject_encoding_states[subject].relation_reliability(
                    trial.higher_item,
                    trial.lower_item,
                    evaluator.item_rank[trial.lower_item] - evaluator.item_rank[trial.higher_item],
                )
                values.append(
                    trial.signed_magnitude
                    * float(access_factor(np.asarray([admission]), np.asarray([probability]))[0])
                )
            local_rows.append(
                local_edge_compression(
                    evaluator.cue_codes[subject], schedule, np.asarray(values), geometry
                )["compressed"]
            )
        local_loo.append(np.asarray(local_rows) * gain)
    global_loo = np.asarray(global_loo)
    local_loo = np.asarray(local_loo)
    intact_loo = global_loo + local_loo

    true_positions = {item: index for index, item in enumerate(protocol.true_order_high_to_low)}
    true_sign = np.asarray(
        [1.0 if true_positions[i] < true_positions[j] else -1.0 for i, j in geometry.pairs]
    )
    learned = np.asarray([pair in protocol.learned_pairs for pair in geometry.pairs])
    nonlearned = ~learned
    temperature = float(evaluation["temperature"])
    exact = {
        "intact_learned": _exact_accuracy(intact, true_sign, temperature, learned),
        "intact_nonlearned": _exact_accuracy(intact, true_sign, temperature, nonlearned),
        "global_only_nonlearned": _exact_accuracy(global_field, true_sign, temperature, nonlearned),
        "local_only_learned": _exact_accuracy(gain * local_raw, true_sign, temperature, learned),
        "local_only_nonlearned": _exact_accuracy(gain * local_raw, true_sign, temperature, nonlearned),
    }
    remote = {
        "intact": _remote_magnitude(intact, intact_loo, relations, geometry),
        "global_only": _remote_magnitude(global_field, global_loo, relations, geometry),
        "local_only": _remote_magnitude(gain * local_raw, local_loo, relations, geometry),
    }
    behavior = analyze_sampled_query_policy(
        protocol,
        _margin_logits(intact, geometry),
        seed=int(evaluation["choice_seed"]),
        temperature=temperature,
    )
    identity_behavior = analyze_sampled_query_policy(
        protocol,
        _margin_logits(identity_intact, geometry),
        seed=int(evaluation["choice_seed"]),
        temperature=temperature,
    )
    behavior_record = _behavior_flags(seed, behavior)
    identity_record = _behavior_flags(seed, identity_behavior)
    reference = load_json(ROOT / "results" / "model_behavior_reproduction_map_v1.json")["networks"][str(seed)]
    behavior_matches = {
        name: behavior_record["flags"][name] == reference["flags"][name]
        for name in reference["flags"]
    }
    double_flag = bool(
        exact["global_only_nonlearned"] > 0.70
        and abs(exact["global_only_nonlearned"] - exact["intact_nonlearned"]) <= 0.05
        and exact["local_only_learned"] > 0.55
        and exact["local_only_nonlearned"] <= 0.55
        and remote["local_only"] <= 0.25 * remote["intact"]
        and remote["global_only"] > 0.0
    )
    result = {
        "seed": seed,
        "exact_accuracy": exact,
        "remote_reassembly": remote,
        "full_neural_terminal_hodge_fraction": float(
            np.mean(
                np.sum((full_field @ geometry.projection.T) ** 2, axis=1)
                / np.sum(full_field**2, axis=1)
            )
        ),
        "reduced_to_full_terminal_potential_correlation": _correlation(
            potential, hodge_potential(full_field, geometry)
        ),
        "local_exact_max_abs_error": local_error,
        "double_dissociation_passed": double_flag,
        "behavior_flags": behavior_record["flags"],
        "reference_behavior_flags": reference["flags"],
        "behavior_flag_matches": behavior_matches,
        "behavior_preservation_passed": all(behavior_matches.values()),
        "identity_kernel_behavior_flags": identity_record["flags"],
    }
    arrays = {
        f"preservation_{seed}_evidence": evidence,
        f"preservation_{seed}_potential": potential,
        f"preservation_{seed}_global_field": global_field,
        f"preservation_{seed}_local_raw": local_raw,
        f"preservation_{seed}_identity_local_raw": identity_raw,
        f"preservation_{seed}_intact_field": intact,
    }
    return result, arrays


def _pad_records(seed: int, records: list[EpisodeTrajectory]) -> dict[str, np.ndarray]:
    max_trials = max(len(record.evidence) for record in records)
    count = len(records)
    potentials = np.full((count, max_trials + 1, 8), np.nan)
    fields = np.full((count, max_trials + 1, 28), np.nan)
    evidence = np.full((count, max_trials, 8), np.nan)
    lengths = np.empty(count, dtype=np.int64)
    loo_potential = np.empty((count, 8))
    loo_field = np.empty((count, 28))
    loo_relation = np.empty((count, 2), dtype=np.int64)
    for index, record in enumerate(records):
        length = len(record.evidence)
        lengths[index] = length
        potentials[index, : length + 1] = record.potentials
        fields[index, : length + 1] = record.fields
        evidence[index, :length] = record.evidence
        loo_potential[index] = record.loo_potential
        loo_field[index] = record.loo_field
        loo_relation[index] = record.loo_relation
    prefix = f"development_{seed}"
    return {
        f"{prefix}_lengths": lengths,
        f"{prefix}_potentials": potentials,
        f"{prefix}_fields": fields,
        f"{prefix}_evidence": evidence,
        f"{prefix}_loo_potential": loo_potential,
        f"{prefix}_loo_field": loo_field,
        f"{prefix}_loo_relation": loo_relation,
    }


def _parameter_json(parameters: ReducedParameters) -> dict:
    value = {"A": parameters.A.tolist()}
    if parameters.U is not None:
        value.update(
            {"U": parameters.U.tolist(), "V": parameters.V.tolist(), "W": parameters.W.tolist()}
        )
    return value


def build_result(specification_path: Path, implementation_lock_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    source_validation = validate_sources(specification_path, implementation_lock_path)
    specification = load_json(specification_path)
    runtime = configure_runtime()
    geometry = complete_geometry()
    development = {}
    arrays: dict[str, np.ndarray] = {}
    for seed in specification["development_artifacts"]["mandatory_seeds"]:
        artifact = specification["development_artifacts"]["artifacts"][str(seed)]
        records = extract_development_seed(
            ROOT / artifact["checkpoint"]["path"],
            rng_seed=int(specification["generic_trajectory_contract"]["episode_rng_seeds"][str(seed)]),
            batches=int(specification["generic_trajectory_contract"]["batches_per_backbone"]),
            geometry=geometry,
        )
        development[str(seed)] = records
        arrays.update(_pad_records(seed, records))

    folds = {}
    mandatory = tuple(specification["development_artifacts"]["mandatory_seeds"])
    for held_out in mandatory:
        train = [record for seed in mandatory if seed != held_out for record in development[str(seed)]]
        accumulator, candidate, unconstrained, fit = _fit_models(
            train, seed=45101 + held_out, device=runtime["device"]
        )
        evaluation = evaluate_development_fold(
            development[str(held_out)],
            accumulator,
            candidate,
            unconstrained,
            geometry,
            bootstrap_samples=int(specification["bootstrap"]["samples"]),
            bootstrap_seed=int(specification["bootstrap"]["seeds"][str(held_out)]),
        )
        folds[str(held_out)] = {
            "held_out_seed": held_out,
            "training_seeds": [seed for seed in mandatory if seed != held_out],
            "fit": fit,
            **evaluation,
        }

    all_records = [record for seed in mandatory for record in development[str(seed)]]
    final_accumulator, final_candidate, final_unconstrained, final_fit = _fit_models(
        all_records, seed=45199, device=runtime["device"]
    )
    preservation = {}
    for seed in specification["preservation_artifacts"]["mandatory_seeds"]:
        result, raw = evaluate_preservation_seed(
            seed,
            specification["preservation_artifacts"]["artifacts"][str(seed)],
            final_candidate,
            specification,
            geometry,
        )
        preservation[str(seed)] = result
        arrays.update(raw)

    integrity = {
        "source_validation": source_validation["passed"],
        "exact_development_seed_set": set(development) == {"2101", "2102", "2103"},
        "episodes_per_development_seed": all(len(records) == 128 for records in development.values()),
        "all_finite_parameters": all(
            np.all(np.isfinite(value))
            for value in (final_candidate.A, final_candidate.U, final_candidate.V, final_candidate.W)
        ),
        "all_local_exact": all(
            fold["flags"]["local_exactness"] for fold in folds.values()
        ) and all(value["local_exact_max_abs_error"] <= 1e-6 for value in preservation.values()),
        "gpu_runtime": runtime["device"] == "cuda",
        "bounded_cpu_threads": runtime["torch_intraop_threads"] == 1 and runtime["torch_interop_threads"] == 1,
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
        outcome = "potential_only_state_dependent_algorithm"
    else:
        outcome = "rank_2_potential_transition_insufficient"
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "source_validation": source_validation,
        "runtime": runtime,
        "integrity": integrity,
        "development_folds": folds,
        "final_fit": {
            "fit": final_fit,
            "accumulator_parameters": _parameter_json(final_accumulator),
            "candidate_parameters": _parameter_json(final_candidate),
            "unconstrained_coefficients_shape": list(final_unconstrained.shape),
        },
        "preservation": preservation,
        "decision": {
            "outcome": outcome,
            "development_all_passed": development_passed,
            "preservation_all_passed": preservation_passed,
            "claim_boundary": specification["claim_boundary"],
        },
    }
    json.dumps(result, allow_nan=False)
    return result, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=SPECIFICATION_PATH)
    parser.add_argument("--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK_PATH)
    parser.add_argument("--trajectory-output", type=Path, default=TRAJECTORY_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.trajectory_output.exists() or arguments.output.exists():
        raise FileExistsError("registered reduced-algorithm outputs are write-once")
    result, arrays = build_result(arguments.specification, arguments.implementation_lock)
    arguments.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.trajectory_output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result["trajectory_artifact"] = {
        "path": str(arguments.trajectory_output.relative_to(ROOT)),
        "sha256": file_sha256(arguments.trajectory_output),
        "arrays": sorted(arrays),
    }
    write_json_exclusive(arguments.output, result)
    print(
        json.dumps(
            {
                "path": str(arguments.output),
                "sha256": file_sha256(arguments.output),
                "trajectory_sha256": result["trajectory_artifact"]["sha256"],
                "outcome": result["decision"]["outcome"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
