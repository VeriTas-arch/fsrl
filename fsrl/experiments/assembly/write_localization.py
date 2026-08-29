"""Registered support-write localization for frozen pilot networks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy import stats

from fsrl.analysis.hodge import (
    CompleteGraphGeometry,
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
)
from fsrl.analysis.posterior import ExactRankingPosterior
from fsrl.analysis.statistics import (
    bootstrap_counts,
    json_values,
    summarize_difference,
    summarize_subjects,
)
from fsrl.core.config import NUMRESPONSESTEP
from fsrl.evaluation.fields import ordered_query_schedule, readout_margin_fields
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.evaluation.registered import load_registered_frozen_evaluator
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.runtime import default_device
from fsrl.infra.study_registry import (
    resolve_record,
    validate_registered_file,
)
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/support_write_localization_v1.json"
)


@dataclass(frozen=True)
class SupportTrialTrace:
    final_fast_weights: torch.Tensor
    da: torch.Tensor
    eligibility_before: torch.Tensor
    intended_increment: torch.Tensor
    actual_increment: torch.Tensor
    clip_fraction: torch.Tensor
    clip_excess_mean: torch.Tensor
    forward_max_abs_error: float


@dataclass(frozen=True)
class ExactInnovations:
    delta_q: np.ndarray
    q_norm: np.ndarray
    information_gain: np.ndarray
    entropy_reduction: np.ndarray


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    names = (
        "pilot_specification",
        "protocol",
        "trajectory_specification",
        "trajectory_result",
        "model_equation_source",
        "frozen_evaluator_source",
    )
    validated: dict[str, object] = {
        name: validate_registered_file(sources[name]) for name in names
    }
    artifacts: list[dict[str, object]] = []
    for registration in sources["pilot_artifacts"]:
        row: dict[str, object] = {"seed": int(registration["seed"])}
        for prefix in ("checkpoint", "config", "behavior"):
            path = resolve_path(registration[f"{prefix}_path"])
            observed = file_sha256(path)
            if observed != registration[f"{prefix}_sha256"]:
                raise RuntimeError(f"registered SHA-256 mismatch: {path}")
            row[prefix] = {
                "path": registration[f"{prefix}_path"],
                "sha256": observed,
            }
        artifacts.append(row)
    validated["pilot_artifacts"] = artifacts
    return validated


def _trial_inputs(
    evaluator: FrozenFastWeightEvaluator,
    trial_index: int,
    evidence_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
    left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
    right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
    signed = np.asarray(
        [
            trial.signed_magnitude
            * evaluator._encoding_reliability(subject, trial_index)
            * evidence_scales[subject]
            for subject, trial in enumerate(trials)
        ],
        dtype=np.float32,
    )
    time_value = (
        trial_index
        / max(1, evaluator.protocol.support_trials - 1)
        * evaluator.test_time_value
    )
    return left, right, signed, time_value


def trace_support_trial(
    evaluator: FrozenFastWeightEvaluator,
    fast_weights: torch.Tensor,
    trial_index: int,
    *,
    evidence_scales: np.ndarray | None = None,
) -> SupportTrialTrace:
    """Advance one support slot and expose the exact pre-update write tensors."""

    if evidence_scales is None:
        scales = np.ones(evaluator.config.bs, dtype=np.float32)
    else:
        scales = np.asarray(evidence_scales, dtype=np.float32)
        if scales.shape != (evaluator.config.bs,):
            raise ValueError("evidence_scales must have one value per subject")
    left, right, signed, time_value = _trial_inputs(evaluator, trial_index, scales)
    hidden = evaluator.net.initial_hidden(evaluator.config.bs)
    eligibility = evaluator.net.initial_eligibility(evaluator.config.bs)
    current = fast_weights.detach().clone()
    da_steps = []
    eligibility_steps = []
    intended_steps = []
    actual_steps = []
    clip_fraction_steps = []
    clip_excess_steps = []
    max_error = 0.0

    with torch.no_grad():
        for step in range(evaluator.config.triallen):
            inputs = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=step,
                time_value=time_value,
                support_trial=True,
            )
            before_eligibility = eligibility.detach().clone()
            before_fast_weights = current.detach().clone()
            _, _, da, hidden, eligibility, proposed = evaluator.net(
                inputs, hidden, eligibility, current
            )
            intended = da.view(evaluator.config.bs, 1, 1) * before_eligibility
            preclip = before_fast_weights + intended
            expected = torch.clamp(preclip, min=-50.0, max=50.0)
            max_error = max(
                max_error, float(torch.max(torch.abs(expected - proposed)).item())
            )
            actual = proposed - before_fast_weights
            excess = torch.clamp(torch.abs(preclip) - 50.0, min=0.0)
            da_steps.append(da[:, 0].detach().clone())
            eligibility_steps.append(before_eligibility)
            intended_steps.append(intended.detach().clone())
            actual_steps.append(actual.detach().clone())
            clip_fraction_steps.append(
                torch.mean((excess > 0.0).to(torch.float32), dim=(1, 2))
            )
            clip_excess_steps.append(torch.mean(excess, dim=(1, 2)))
            current = proposed

    return SupportTrialTrace(
        final_fast_weights=current.detach().clone(),
        da=torch.stack(da_steps, dim=1),
        eligibility_before=torch.stack(eligibility_steps, dim=1),
        intended_increment=torch.stack(intended_steps, dim=1),
        actual_increment=torch.stack(actual_steps, dim=1),
        clip_fraction=torch.stack(clip_fraction_steps, dim=1),
        clip_excess_mean=torch.stack(clip_excess_steps, dim=1),
        forward_max_abs_error=max_error,
    )


def readout_effective_margin_fields(
    evaluator: FrozenFastWeightEvaluator,
    effective_modulation: torch.Tensor,
    geometry: CompleteGraphGeometry,
) -> np.ndarray:
    """Read frozen queries using W + explicit effective recurrent modulation."""

    expected_shape = (evaluator.config.bs, evaluator.config.hs, evaluator.config.hs)
    if effective_modulation.shape != expected_shape:
        raise ValueError("effective_modulation has the wrong shape")
    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    outputs = [{} for _ in range(evaluator.config.bs)]
    with torch.no_grad():
        for pair_index in range(len(schedules[0])):
            hidden = evaluator.net.initial_hidden(evaluator.config.bs)
            left = np.asarray(
                [schedule[pair_index][0] for schedule in schedules], dtype=np.int64
            )
            right = np.asarray(
                [schedule[pair_index][1] for schedule in schedules], dtype=np.int64
            )
            signed = np.zeros(evaluator.config.bs, dtype=np.float32)
            response = None
            for step in range(evaluator.config.triallen):
                inputs = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=step,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                hidden = evaluator.net.activ(
                    evaluator.net.i2h(inputs).view(
                        evaluator.config.bs, evaluator.config.hs, 1
                    )
                    + torch.matmul(
                        evaluator.net.w + effective_modulation,
                        hidden.view(evaluator.config.bs, evaluator.config.hs, 1),
                    )
                ).view(evaluator.config.bs, evaluator.config.hs)
                logits = evaluator.net.h2o(hidden)
                if step == NUMRESPONSESTEP:
                    response = logits[:, 1] - logits[:, 0]
            assert response is not None
            values = response.detach().cpu().numpy()
            for subject, value in enumerate(values):
                outputs[subject][(int(left[subject]), int(right[subject]))] = float(
                    value
                )
    return np.asarray(
        [
            [0.5 * (row[pair] - row[(pair[1], pair[0])]) for pair in geometry.pairs]
            for row in outputs
        ],
        dtype=np.float64,
    )


def _posterior_from_energy(
    energy: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    log_weights = -np.asarray(energy, dtype=np.float64) / temperature
    maximum = np.max(log_weights, axis=1, keepdims=True)
    shifted = log_weights - maximum
    log_normalizer = maximum + np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probabilities = log_weights - log_normalizer
    return np.exp(log_probabilities), log_probabilities


def exact_support_innovations(
    evaluator: FrozenFastWeightEvaluator,
    protocol: RankingProtocol,
    *,
    temperature: float,
) -> ExactInnovations:
    exact = ExactRankingPosterior(protocol.n_items, temperature=temperature)
    positions = exact.positions.astype(np.float64)
    energies = np.zeros((evaluator.config.bs, exact.n_hypotheses), dtype=np.float64)
    delta_q = []
    q_norm = []
    information_gain = []
    entropy_reduction = []

    before_probabilities, before_log_probabilities = _posterior_from_energy(
        energies, temperature
    )
    before_q = -(before_probabilities @ positions)
    before_q -= np.mean(before_q, axis=1, keepdims=True)
    before_entropy = -np.sum(before_probabilities * before_log_probabilities, axis=1)
    for trial_index in range(protocol.support_trials):
        for subject, schedule in enumerate(evaluator.support_schedules):
            trial = schedule[trial_index]
            predicted = (
                exact.positions[:, trial.lower_item]
                - exact.positions[:, trial.higher_item]
            ) / float(protocol.n_items - 1)
            residual = predicted - abs(trial.signed_magnitude)
            energies[subject] += (
                evaluator._encoding_reliability(subject, trial_index)
                * residual
                * residual
            )
        after_probabilities, after_log_probabilities = _posterior_from_energy(
            energies, temperature
        )
        after_q = -(after_probabilities @ positions)
        after_q -= np.mean(after_q, axis=1, keepdims=True)
        after_entropy = -np.sum(after_probabilities * after_log_probabilities, axis=1)
        update = after_q - before_q
        delta_q.append(update)
        q_norm.append(np.linalg.norm(update, axis=1))
        information_gain.append(
            np.sum(
                after_probabilities
                * (after_log_probabilities - before_log_probabilities),
                axis=1,
            )
        )
        entropy_reduction.append(before_entropy - after_entropy)
        before_probabilities = after_probabilities
        before_log_probabilities = after_log_probabilities
        before_q = after_q
        before_entropy = after_entropy

    return ExactInnovations(
        delta_q=np.asarray(delta_q),
        q_norm=np.asarray(q_norm),
        information_gain=np.asarray(information_gain),
        entropy_reduction=np.asarray(entropy_reduction),
    )


def matrix_norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values.flatten(start_dim=-2), dim=-1)


def matrix_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left = first.flatten(start_dim=-2)
    right = second.flatten(start_dim=-2)
    numerator = torch.sum(left * right, dim=-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.full_like(numerator, torch.nan),
    )


def norm_match(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate_norm = matrix_norm(candidate)
    target_norm = matrix_norm(target)
    scale = torch.where(
        candidate_norm > 0.0,
        target_norm / candidate_norm,
        torch.zeros_like(candidate_norm),
    )
    return candidate * scale[:, None, None]


def row_cosine(first: np.ndarray, second: np.ndarray, tolerance: float) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > tolerance,
    )


def _subject_trial_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask, np.asarray(values, dtype=np.float64), np.nan)
    finite = np.sum(np.isfinite(rows), axis=0)
    return np.divide(
        np.nansum(rows, axis=0),
        finite,
        out=np.full(rows.shape[1], np.nan),
        where=finite > 0,
    )


def summarize_trials(
    values: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    return summarize_subjects(
        _subject_trial_mean(values, mask), counts, interval=interval
    )


def summarize_trial_difference(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    return summarize_trials(
        np.asarray(first) - np.asarray(second),
        mask,
        counts,
        interval=interval,
    )


def within_subject_spearman(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    correlations = []
    for subject in range(first.shape[1]):
        selected = mask[:, subject]
        left = first[selected, subject]
        right = second[selected, subject]
        finite = np.isfinite(left) & np.isfinite(right)
        if (
            np.sum(finite) < 3
            or np.ptp(left[finite]) == 0.0
            or np.ptp(right[finite]) == 0.0
        ):
            correlations.append(np.nan)
        else:
            correlation = cast(Any, stats.spearmanr(left[finite], right[finite]))
            correlations.append(float(correlation.statistic))
    return np.asarray(correlations)


def exposure_adjusted_spearman(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    correlations = []
    for subject in range(first.shape[1]):
        selected = mask[:, subject]
        left = np.asarray(first[:, subject], dtype=np.float64).copy()
        right = np.asarray(second[:, subject], dtype=np.float64).copy()
        for exposure_index in range(1, 5):
            group = selected & (exposure[:, subject] == exposure_index)
            if np.any(group):
                left[group] -= np.mean(left[group])
                right[group] -= np.mean(right[group])
        finite = selected & np.isfinite(left) & np.isfinite(right)
        if (
            np.sum(finite) < 3
            or np.ptp(left[finite]) == 0.0
            or np.ptp(right[finite]) == 0.0
        ):
            correlations.append(np.nan)
        else:
            correlation = cast(Any, stats.spearmanr(left[finite], right[finite]))
            correlations.append(float(correlation.statistic))
    return np.asarray(correlations)


def _cpu(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().astype(np.float64)


def _policy_metrics(
    fields: np.ndarray,
    baseline_fields: np.ndarray,
    delta_q: np.ndarray,
    geometry: CompleteGraphGeometry,
    tolerance: float,
) -> dict[str, np.ndarray]:
    delta_field = fields - baseline_fields
    delta_potential = hodge_potentials(delta_field, geometry)
    return {
        "potential_norm": np.linalg.norm(delta_potential, axis=1),
        "exact_delta_q_cosine": row_cosine(delta_potential, delta_q, tolerance),
        "true_order_cosine": row_cosine(
            delta_potential,
            np.broadcast_to(geometry.true_potential, delta_potential.shape),
            tolerance,
        ),
        "field_gradient_fraction": gradient_energy_fraction(delta_field, geometry),
    }


def replay_without_relation_history(
    evaluator: FrozenFastWeightEvaluator,
    trial_index: int,
    focal_relations: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    fast_weights = evaluator.initialize_fast_weights()
    for previous_index in range(trial_index):
        scales = np.asarray(
            [
                float(
                    (
                        schedule[previous_index].higher_item,
                        schedule[previous_index].lower_item,
                    )
                    != focal_relations[subject]
                )
                for subject, schedule in enumerate(evaluator.support_schedules)
            ],
            dtype=np.float32,
        )
        fast_weights = trace_support_trial(
            evaluator,
            fast_weights,
            previous_index,
            evidence_scales=scales,
        ).final_fast_weights
    return fast_weights


def collect_seed_metrics(
    evaluator: FrozenFastWeightEvaluator,
    exact: ExactInnovations,
    geometry: CompleteGraphGeometry,
    *,
    alpha_shuffle_seed: int,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    steps = evaluator.config.triallen
    alpha = evaluator.net.alpha.detach()
    permutation = np.random.default_rng(alpha_shuffle_seed).permutation(alpha.numel())
    shuffled_alpha = torch.from_numpy(
        alpha.detach().cpu().numpy().reshape(-1)[permutation].reshape(alpha.shape)
    ).to(alpha.device)
    p0 = evaluator.initialize_fast_weights()
    p0_standard = readout_margin_fields(evaluator, p0, geometry)
    p0_effective = readout_effective_margin_fields(evaluator, alpha * p0, geometry)
    validation = {
        "p0_effective_readout_max_abs_error": float(
            np.max(np.abs(p0_standard - p0_effective))
        ),
        "trace_forward_max_abs_error": 0.0,
        "incremental_endpoint_max_abs_error": 0.0,
        "matched_intended_endpoint_max_abs_error": 0.0,
        "decomposition_max_abs_error": 0.0,
        "explicit_actual_readout_max_abs_error": 0.0,
    }

    step_names = (
        "da_plus",
        "da_zero",
        "absolute_da_difference",
        "eligibility_plus_norm",
        "eligibility_zero_norm",
        "matched_intended_write_norm",
        "matched_effective_write_norm",
        "da_component_effective_norm",
        "eligibility_component_effective_norm",
        "component_cosine",
        "da_component_to_total_cosine",
        "eligibility_component_to_total_cosine",
        "signed_cross_energy_fraction",
        "plus_clip_fraction",
        "zero_clip_fraction",
        "plus_clip_excess_mean",
        "zero_clip_excess_mean",
    )
    trial_names = (
        "exact_delta_q_norm",
        "exact_information_gain",
        "exact_entropy_reduction",
        "neural_potential_update_norm",
        "neural_to_exact_delta_q_cosine",
        "neural_to_true_order_cosine",
        "neural_update_gradient_fraction",
        "matched_effective_write_norm",
        "functional_gain",
        "fast_weight_norm_before",
        "da_component_effective_norm",
        "eligibility_component_effective_norm",
        "eligibility_minus_da_effective_norm",
        "component_cross_energy_fraction",
        "da_component_policy_norm",
        "eligibility_component_policy_norm",
        "da_component_to_exact_cosine",
        "eligibility_component_to_exact_cosine",
        "actual_alpha_to_exact_cosine",
        "scalar_alpha_to_exact_cosine",
        "shuffled_alpha_to_exact_cosine",
        "actual_alpha_policy_norm",
        "scalar_alpha_policy_norm",
        "shuffled_alpha_policy_norm",
        "actual_alpha_gradient_fraction",
        "scalar_alpha_gradient_fraction",
        "shuffled_alpha_gradient_fraction",
        "sum_abs_da_plus_effective_steps",
        "sum_abs_da_difference_effective_steps",
        "no_history_policy_update_norm",
        "no_history_effective_write_norm",
        "history_policy_restoration",
        "history_write_restoration",
    )
    step_rows = {name: [] for name in step_names}
    trial_rows = {name: [] for name in trial_names}
    retained_rows = []
    exposure_rows = []
    relation_rows = []
    natural_fast_weights = p0

    for trial_index in range(evaluator.protocol.support_trials):
        if trial_index % 4 == 0:
            print(
                f"[support-write] trial {trial_index + 1}/"
                f"{evaluator.protocol.support_trials}",
                file=sys.stderr,
            )
        plus = trace_support_trial(evaluator, natural_fast_weights, trial_index)
        zero = trace_support_trial(
            evaluator,
            natural_fast_weights,
            trial_index,
            evidence_scales=np.zeros(evaluator.config.bs, dtype=np.float32),
        )
        validation["trace_forward_max_abs_error"] = max(
            validation["trace_forward_max_abs_error"],
            plus.forward_max_abs_error,
            zero.forward_max_abs_error,
        )
        reference = evaluator.advance_support_trial(natural_fast_weights, trial_index)
        validation["incremental_endpoint_max_abs_error"] = max(
            validation["incremental_endpoint_max_abs_error"],
            float(torch.max(torch.abs(reference - plus.final_fast_weights)).item()),
        )

        delta_u = plus.intended_increment - zero.intended_increment
        da_plus = plus.da[:, :, None, None]
        da_zero = zero.da[:, :, None, None]
        da_component = (
            0.5
            * (da_plus - da_zero)
            * (plus.eligibility_before + zero.eligibility_before)
        )
        eligibility_component = (
            0.5
            * (da_plus + da_zero)
            * (plus.eligibility_before - zero.eligibility_before)
        )
        decomposition_error = float(
            torch.max(torch.abs(delta_u - da_component - eligibility_component)).item()
        )
        validation["decomposition_max_abs_error"] = max(
            validation["decomposition_max_abs_error"], decomposition_error
        )
        delta_m = alpha * delta_u
        da_m = alpha * da_component
        eligibility_m = alpha * eligibility_component
        total_delta_u = torch.sum(delta_u, dim=1)
        total_delta_m = torch.sum(delta_m, dim=1)
        total_da = torch.sum(da_component, dim=1)
        total_eligibility = torch.sum(eligibility_component, dim=1)
        total_da_m = alpha * total_da
        total_eligibility_m = alpha * total_eligibility
        intended_endpoint = zero.final_fast_weights + total_delta_u
        validation["matched_intended_endpoint_max_abs_error"] = max(
            validation["matched_intended_endpoint_max_abs_error"],
            float(
                torch.max(torch.abs(intended_endpoint - plus.final_fast_weights)).item()
            ),
        )

        total_energy = matrix_norm(total_delta_m) ** 2
        cross_energy = 2.0 * torch.sum(total_da_m * total_eligibility_m, dim=(1, 2))
        trial_cross_fraction = torch.where(
            total_energy > 0.0,
            cross_energy / total_energy,
            torch.full_like(total_energy, torch.nan),
        )
        step_total_energy = matrix_norm(delta_m) ** 2
        step_cross_energy = 2.0 * torch.sum(da_m * eligibility_m, dim=(2, 3))
        step_cross_fraction = torch.where(
            step_total_energy > 0.0,
            step_cross_energy / step_total_energy,
            torch.full_like(step_total_energy, torch.nan),
        )
        step_values = {
            "da_plus": plus.da,
            "da_zero": zero.da,
            "absolute_da_difference": torch.abs(plus.da - zero.da),
            "eligibility_plus_norm": matrix_norm(plus.eligibility_before),
            "eligibility_zero_norm": matrix_norm(zero.eligibility_before),
            "matched_intended_write_norm": matrix_norm(delta_u),
            "matched_effective_write_norm": matrix_norm(delta_m),
            "da_component_effective_norm": matrix_norm(da_m),
            "eligibility_component_effective_norm": matrix_norm(eligibility_m),
            "component_cosine": matrix_cosine(da_m, eligibility_m),
            "da_component_to_total_cosine": matrix_cosine(da_m, delta_m),
            "eligibility_component_to_total_cosine": matrix_cosine(
                eligibility_m, delta_m
            ),
            "signed_cross_energy_fraction": step_cross_fraction,
            "plus_clip_fraction": plus.clip_fraction,
            "zero_clip_fraction": zero.clip_fraction,
            "plus_clip_excess_mean": plus.clip_excess_mean,
            "zero_clip_excess_mean": zero.clip_excess_mean,
        }
        for name, values in step_values.items():
            step_rows[name].append(_cpu(values))

        plus_fields = readout_margin_fields(
            evaluator, plus.final_fast_weights, geometry
        )
        zero_fields = readout_margin_fields(
            evaluator, zero.final_fast_weights, geometry
        )
        delta_q = exact.delta_q[trial_index]
        actual_policy = _policy_metrics(
            plus_fields, zero_fields, delta_q, geometry, tolerance
        )

        da_fields = readout_margin_fields(
            evaluator, zero.final_fast_weights + total_da, geometry
        )
        eligibility_fields = readout_margin_fields(
            evaluator, zero.final_fast_weights + total_eligibility, geometry
        )
        da_policy = _policy_metrics(
            da_fields, zero_fields, delta_q, geometry, tolerance
        )
        eligibility_policy = _policy_metrics(
            eligibility_fields, zero_fields, delta_q, geometry, tolerance
        )

        actual_delta_p = plus.final_fast_weights - zero.final_fast_weights
        actual_delta_m = alpha * actual_delta_p
        baseline_m = alpha * zero.final_fast_weights
        actual_effective_fields = readout_effective_margin_fields(
            evaluator, alpha * plus.final_fast_weights, geometry
        )
        validation["explicit_actual_readout_max_abs_error"] = max(
            validation["explicit_actual_readout_max_abs_error"],
            float(np.max(np.abs(actual_effective_fields - plus_fields))),
        )
        scalar_delta_m = norm_match(actual_delta_p, actual_delta_m)
        shuffled_delta_m = norm_match(shuffled_alpha * actual_delta_p, actual_delta_m)
        scalar_fields = readout_effective_margin_fields(
            evaluator, baseline_m + scalar_delta_m, geometry
        )
        shuffled_fields = readout_effective_margin_fields(
            evaluator, baseline_m + shuffled_delta_m, geometry
        )
        alpha_actual_policy = _policy_metrics(
            actual_effective_fields, zero_fields, delta_q, geometry, tolerance
        )
        scalar_policy = _policy_metrics(
            scalar_fields, zero_fields, delta_q, geometry, tolerance
        )
        shuffled_policy = _policy_metrics(
            shuffled_fields, zero_fields, delta_q, geometry, tolerance
        )

        relations = tuple(
            (schedule[trial_index].higher_item, schedule[trial_index].lower_item)
            for schedule in evaluator.support_schedules
        )
        retained = np.asarray(
            [
                evaluator._encoding_reliability(subject, trial_index) > 0.0
                for subject in range(evaluator.config.bs)
            ],
            dtype=bool,
        )
        exposure = np.asarray(
            [
                schedule[trial_index].block_index + 1
                for schedule in evaluator.support_schedules
            ],
            dtype=np.int64,
        )
        if trial_index < len(evaluator.protocol.support_pairs_higher_lower):
            no_history_policy_norm = actual_policy["potential_norm"].copy()
            no_history_write_norm = _cpu(matrix_norm(total_delta_m))
        else:
            history = replay_without_relation_history(evaluator, trial_index, relations)
            history_plus = trace_support_trial(evaluator, history, trial_index)
            history_zero = trace_support_trial(
                evaluator,
                history,
                trial_index,
                evidence_scales=np.zeros(evaluator.config.bs, dtype=np.float32),
            )
            history_plus_fields = readout_margin_fields(
                evaluator, history_plus.final_fast_weights, geometry
            )
            history_zero_fields = readout_margin_fields(
                evaluator, history_zero.final_fast_weights, geometry
            )
            history_policy = _policy_metrics(
                history_plus_fields,
                history_zero_fields,
                delta_q,
                geometry,
                tolerance,
            )
            no_history_policy_norm = history_policy["potential_norm"]
            no_history_delta_u = torch.sum(
                history_plus.intended_increment - history_zero.intended_increment,
                dim=1,
            )
            no_history_write_norm = _cpu(matrix_norm(alpha * no_history_delta_u))

        write_norm = _cpu(matrix_norm(total_delta_m))
        functional_gain = np.divide(
            actual_policy["potential_norm"],
            write_norm,
            out=np.full_like(write_norm, np.nan),
            where=write_norm > tolerance,
        )
        trial_values = {
            "exact_delta_q_norm": exact.q_norm[trial_index],
            "exact_information_gain": exact.information_gain[trial_index],
            "exact_entropy_reduction": exact.entropy_reduction[trial_index],
            "neural_potential_update_norm": actual_policy["potential_norm"],
            "neural_to_exact_delta_q_cosine": actual_policy["exact_delta_q_cosine"],
            "neural_to_true_order_cosine": actual_policy["true_order_cosine"],
            "neural_update_gradient_fraction": actual_policy["field_gradient_fraction"],
            "matched_effective_write_norm": write_norm,
            "functional_gain": functional_gain,
            "fast_weight_norm_before": _cpu(matrix_norm(natural_fast_weights)),
            "da_component_effective_norm": _cpu(matrix_norm(total_da_m)),
            "eligibility_component_effective_norm": _cpu(
                matrix_norm(total_eligibility_m)
            ),
            "eligibility_minus_da_effective_norm": _cpu(
                matrix_norm(total_eligibility_m) - matrix_norm(total_da_m)
            ),
            "component_cross_energy_fraction": _cpu(trial_cross_fraction),
            "da_component_policy_norm": da_policy["potential_norm"],
            "eligibility_component_policy_norm": eligibility_policy["potential_norm"],
            "da_component_to_exact_cosine": da_policy["exact_delta_q_cosine"],
            "eligibility_component_to_exact_cosine": eligibility_policy[
                "exact_delta_q_cosine"
            ],
            "actual_alpha_to_exact_cosine": alpha_actual_policy["exact_delta_q_cosine"],
            "scalar_alpha_to_exact_cosine": scalar_policy["exact_delta_q_cosine"],
            "shuffled_alpha_to_exact_cosine": shuffled_policy["exact_delta_q_cosine"],
            "actual_alpha_policy_norm": alpha_actual_policy["potential_norm"],
            "scalar_alpha_policy_norm": scalar_policy["potential_norm"],
            "shuffled_alpha_policy_norm": shuffled_policy["potential_norm"],
            "actual_alpha_gradient_fraction": alpha_actual_policy[
                "field_gradient_fraction"
            ],
            "scalar_alpha_gradient_fraction": scalar_policy["field_gradient_fraction"],
            "shuffled_alpha_gradient_fraction": shuffled_policy[
                "field_gradient_fraction"
            ],
            "sum_abs_da_plus_effective_steps": _cpu(
                torch.sum(torch.abs(plus.da[:, 2:steps]), dim=1)
            ),
            "sum_abs_da_difference_effective_steps": _cpu(
                torch.sum(torch.abs(plus.da[:, 2:steps] - zero.da[:, 2:steps]), dim=1)
            ),
            "no_history_policy_update_norm": no_history_policy_norm,
            "no_history_effective_write_norm": no_history_write_norm,
            "history_policy_restoration": (
                no_history_policy_norm - actual_policy["potential_norm"]
            ),
            "history_write_restoration": no_history_write_norm - write_norm,
        }
        for name, values in trial_values.items():
            trial_rows[name].append(np.asarray(values, dtype=np.float64))
        retained_rows.append(retained)
        exposure_rows.append(exposure)
        relation_rows.append(relations)
        natural_fast_weights = plus.final_fast_weights

    step_metrics = {name: np.asarray(values) for name, values in step_rows.items()}
    trial_metrics = {name: np.asarray(values) for name, values in trial_rows.items()}
    trial_metrics["retained"] = np.asarray(retained_rows)
    trial_metrics["exposure"] = np.asarray(exposure_rows)
    trial_metrics["relations"] = np.asarray(relation_rows)
    final_reference = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    validation["final_endpoint_max_abs_error"] = float(
        torch.max(torch.abs(natural_fast_weights - final_reference)).item()
    )
    return step_metrics, trial_metrics, validation


def summarize_seed_metrics(
    step_metrics: dict[str, np.ndarray],
    trial_metrics: dict[str, np.ndarray],
    validation: dict,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict, dict]:
    retained = trial_metrics["retained"].astype(bool)
    omitted = ~retained
    exposure = trial_metrics["exposure"]
    step_summary = []
    for step in range(step_metrics["da_plus"].shape[2]):
        step_summary.append(
            {
                "step": step,
                **{
                    name: summarize_trials(
                        values[:, :, step], retained, counts, interval=interval
                    )
                    for name, values in step_metrics.items()
                },
            }
        )

    retained_names = tuple(
        name
        for name in trial_metrics
        if name not in ("retained", "exposure", "relations")
    )
    retained_summary = {
        name: summarize_trials(trial_metrics[name], retained, counts, interval=interval)
        for name in retained_names
    }
    exposure_summary = {}
    for value in range(1, 5):
        mask = retained & (exposure == value)
        exposure_summary[str(value)] = {
            name: summarize_trials(trial_metrics[name], mask, counts, interval=interval)
            for name in (
                "exact_information_gain",
                "exact_delta_q_norm",
                "matched_effective_write_norm",
                "neural_potential_update_norm",
                "functional_gain",
                "sum_abs_da_plus_effective_steps",
                "sum_abs_da_difference_effective_steps",
                "history_policy_restoration",
                "history_write_restoration",
                "neural_to_exact_delta_q_cosine",
                "neural_to_true_order_cosine",
                "actual_alpha_to_exact_cosine",
                "scalar_alpha_to_exact_cosine",
                "shuffled_alpha_to_exact_cosine",
            )
        }

    def exposure_four_minus_one(name: str) -> dict:
        first = _subject_trial_mean(trial_metrics[name], retained & (exposure == 1))
        fourth = _subject_trial_mean(trial_metrics[name], retained & (exposure == 4))
        return summarize_difference(fourth, first, counts, interval=interval)

    correlations = {}
    innovation_targets = {
        "exact_information_gain": trial_metrics["exact_information_gain"],
        "exact_delta_q_norm": trial_metrics["exact_delta_q_norm"],
    }
    da_signals = {
        "sum_abs_da_plus": trial_metrics["sum_abs_da_plus_effective_steps"],
        "sum_abs_da_difference": trial_metrics["sum_abs_da_difference_effective_steps"],
    }
    raw_correlations = {}
    adjusted_correlations = {}
    raw_adjusted_correlations = {}
    for target_name, target in innovation_targets.items():
        correlations[target_name] = {}
        raw_correlations[target_name] = {}
        adjusted_correlations[target_name] = {}
        raw_adjusted_correlations[target_name] = {}
        for signal_name, signal in da_signals.items():
            values = within_subject_spearman(signal, target, retained)
            correlations[target_name][signal_name] = summarize_subjects(
                values, counts, interval=interval
            )
            raw_correlations[target_name][signal_name] = json_values(values)
            adjusted = exposure_adjusted_spearman(signal, target, retained, exposure)
            adjusted_correlations[target_name][signal_name] = summarize_subjects(
                adjusted, counts, interval=interval
            )
            raw_adjusted_correlations[target_name][signal_name] = json_values(adjusted)

    comparisons = {
        "eligibility_minus_da_effective_norm": summarize_trials(
            trial_metrics["eligibility_minus_da_effective_norm"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_scalar_alpha_exact_cosine": summarize_trial_difference(
            trial_metrics["actual_alpha_to_exact_cosine"],
            trial_metrics["scalar_alpha_to_exact_cosine"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_shuffled_alpha_exact_cosine": summarize_trial_difference(
            trial_metrics["actual_alpha_to_exact_cosine"],
            trial_metrics["shuffled_alpha_to_exact_cosine"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_scalar_alpha_policy_norm": summarize_trial_difference(
            trial_metrics["actual_alpha_policy_norm"],
            trial_metrics["scalar_alpha_policy_norm"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_shuffled_alpha_policy_norm": summarize_trial_difference(
            trial_metrics["actual_alpha_policy_norm"],
            trial_metrics["shuffled_alpha_policy_norm"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_scalar_alpha_gradient_fraction": summarize_trial_difference(
            trial_metrics["actual_alpha_gradient_fraction"],
            trial_metrics["scalar_alpha_gradient_fraction"],
            retained,
            counts,
            interval=interval,
        ),
        "actual_minus_shuffled_alpha_gradient_fraction": summarize_trial_difference(
            trial_metrics["actual_alpha_gradient_fraction"],
            trial_metrics["shuffled_alpha_gradient_fraction"],
            retained,
            counts,
            interval=interval,
        ),
        "neural_exact_minus_true_order_cosine": summarize_trial_difference(
            trial_metrics["neural_to_exact_delta_q_cosine"],
            trial_metrics["neural_to_true_order_cosine"],
            retained,
            counts,
            interval=interval,
        ),
        "exposure_4_minus_1_write_norm": exposure_four_minus_one(
            "matched_effective_write_norm"
        ),
        "exposure_4_minus_1_policy_norm": exposure_four_minus_one(
            "neural_potential_update_norm"
        ),
        "exposure_4_minus_1_functional_gain": exposure_four_minus_one(
            "functional_gain"
        ),
        "history_policy_restoration_exposures_2_to_4": summarize_trials(
            trial_metrics["history_policy_restoration"],
            retained & (exposure >= 2),
            counts,
            interval=interval,
        ),
        "history_write_restoration_exposures_2_to_4": summarize_trials(
            trial_metrics["history_write_restoration"],
            retained & (exposure >= 2),
            counts,
            interval=interval,
        ),
    }

    clip_max = max(
        float(np.max(step_metrics["plus_clip_fraction"])),
        float(np.max(step_metrics["zero_clip_fraction"])),
    )
    clip_excess_max = max(
        float(np.max(step_metrics["plus_clip_excess_mean"])),
        float(np.max(step_metrics["zero_clip_excess_mean"])),
    )
    omitted_maxima = {
        name: (float(np.nanmax(np.abs(values[omitted]))) if np.any(omitted) else 0.0)
        for name, values in (
            ("exact_information_gain", trial_metrics["exact_information_gain"]),
            ("exact_delta_q_norm", trial_metrics["exact_delta_q_norm"]),
            (
                "matched_effective_write_norm",
                trial_metrics["matched_effective_write_norm"],
            ),
            (
                "neural_potential_update_norm",
                trial_metrics["neural_potential_update_norm"],
            ),
        )
    }
    reproduction_tolerance = float(max(tolerance, 32.0 * np.finfo(np.float32).eps))
    if validation["trace_forward_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError("traced forward no longer reproduces model.forward")
    if validation["incremental_endpoint_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError("traced support endpoint no longer reproduces evaluator")
    if validation["final_endpoint_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError("traced final endpoint no longer reproduces evaluator")
    if validation["decomposition_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError("DA/eligibility algebraic decomposition failed")
    if validation["p0_effective_readout_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError("explicit effective readout does not reproduce P0")
    if validation["explicit_actual_readout_max_abs_error"] > reproduction_tolerance:
        raise RuntimeError(
            "explicit effective readout does not reproduce support state"
        )
    validation["implementation_reproduction_tolerance"] = reproduction_tolerance
    early_write_max = max(
        float(
            np.max(step_metrics["matched_effective_write_norm"][:, :, step][retained])
        )
        for step in (0, 1)
    )

    registered_diagnosis = {
        "registered_write_timing": (
            early_write_max <= tolerance
            and (
                step_summary[2]["matched_effective_write_norm"]["bootstrap"]["lower"]
                > 0.0
                or step_summary[3]["matched_effective_write_norm"]["bootstrap"]["lower"]
                > 0.0
            )
        ),
        "eligibility_direction_dominance": comparisons[
            "eligibility_minus_da_effective_norm"
        ]["bootstrap"]["lower"]
        > 0.0,
        "alpha_structural_mapping": (
            comparisons["actual_minus_scalar_alpha_exact_cosine"]["bootstrap"]["lower"]
            > 0.0
            and comparisons["actual_minus_shuffled_alpha_exact_cosine"]["bootstrap"][
                "lower"
            ]
            > 0.0
        ),
        "posterior_innovation_tracking": retained_summary[
            "neural_to_exact_delta_q_cosine"
        ]["bootstrap"]["lower"]
        > 0.0,
        "da_innovation_signal": correlations["exact_information_gain"][
            "sum_abs_da_plus"
        ]["bootstrap"]["lower"]
        > 0.0,
        "practical_literal_saturation": clip_max > 1e-6,
        "relation_specific_assimilation": comparisons[
            "history_policy_restoration_exposures_2_to_4"
        ]["bootstrap"]["lower"]
        > 0.0,
        "stable_omission_controls_within_tolerance": all(
            value <= tolerance for value in omitted_maxima.values()
        ),
    }
    secondary_diagnosis = {
        "da_innovation_signal_exposure_adjusted": adjusted_correlations[
            "exact_information_gain"
        ]["sum_abs_da_plus"]["bootstrap"]["lower"]
        > 0.0,
        "alpha_functional_amplification": (
            comparisons["actual_minus_scalar_alpha_policy_norm"]["bootstrap"]["lower"]
            > 0.0
            and comparisons["actual_minus_shuffled_alpha_policy_norm"]["bootstrap"][
                "lower"
            ]
            > 0.0
        ),
    }
    summary = {
        "step_localization": step_summary,
        "retained_trial_metrics": retained_summary,
        "by_relation_exposure": exposure_summary,
        "within_subject_spearman": correlations,
        "exposure_adjusted_within_subject_spearman": adjusted_correlations,
        "paired_comparisons": comparisons,
        "clip_saturation": {
            "maximum_fraction": clip_max,
            "maximum_mean_excess": clip_excess_max,
            "practical_threshold": 1e-6,
        },
        "stable_omitted_maxima": omitted_maxima,
        "validation": validation,
        "early_step_max_abs_effective_write_norm": early_write_max,
        "registered_directional_diagnosis": registered_diagnosis,
        "secondary_exploratory_diagnosis": secondary_diagnosis,
    }
    raw = {
        "step_metrics": {
            name: json_values(values) for name, values in step_metrics.items()
        },
        "trial_metrics": {
            name: (
                values.astype(int).tolist()
                if name in ("retained", "exposure", "relations")
                else json_values(values)
            )
            for name, values in trial_metrics.items()
        },
        "within_subject_spearman": raw_correlations,
        "exposure_adjusted_within_subject_spearman": raw_adjusted_correlations,
    }
    return summary, raw, registered_diagnosis


def run_support_write_localization(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    tolerance = float(specification["execution_contract"]["floating_zero_tolerance"])
    bootstrap = specification["bootstrap"]
    interval = float(bootstrap["interval"])
    rng = np.random.default_rng(int(bootstrap["seed"]))

    reference_evidence = None
    exact = None
    counts = None
    per_seed = {}
    for registration in sources["pilot_artifacts"]:
        seed = int(registration["seed"])
        print(f"[support-write] loading frozen seed {seed}", file=sys.stderr)
        evaluator, behavior = load_registered_frozen_evaluator(
            registration, pilot_specification, protocol
        )
        evidence = evaluator.realized_support_evidence()
        if reference_evidence is None:
            reference_evidence = evidence
            print("[support-write] computing exact innovations", file=sys.stderr)
            exact = exact_support_innovations(
                evaluator,
                protocol,
                temperature=float(
                    pilot_specification["evaluation"]["posterior_temperature"]
                ),
            )
            counts = bootstrap_counts(
                rng,
                int(bootstrap["samples"]),
                evaluator.config.bs,
            )
        elif evidence != reference_evidence:
            raise RuntimeError("pilot seeds used different realized support evidence")
        assert exact is not None
        assert counts is not None

        step_metrics, trial_metrics, implementation_validation = collect_seed_metrics(
            evaluator,
            exact,
            geometry,
            alpha_shuffle_seed=int(
                specification["execution_contract"]["alpha_shuffle_seed"]
            ),
            tolerance=tolerance,
        )
        summary, raw, _diagnosis = summarize_seed_metrics(
            step_metrics,
            trial_metrics,
            implementation_validation,
            counts,
            interval=interval,
            tolerance=tolerance,
        )
        per_seed[str(seed)] = {
            "seed": seed,
            "subjects": evaluator.config.bs,
            "checkpoint": behavior["checkpoint"],
            **summary,
            "raw_subject_level": raw,
        }

    diagnosis_names = tuple(
        next(iter(per_seed.values()))["registered_directional_diagnosis"]
    )
    overall: dict[str, object] = {
        f"{name}_replicated_across_pilot_seeds": all(
            row["registered_directional_diagnosis"][name] for row in per_seed.values()
        )
        for name in diagnosis_names
    }
    overall["formal_confirmation_status"] = "deferred; formal seeds remain untouched"
    overall["next_step_rule"] = specification["reporting"]["next_step_rule"]
    secondary_names = tuple(
        next(iter(per_seed.values()))["secondary_exploratory_diagnosis"]
    )
    secondary_overall = {
        f"{name}_replicated_across_pilot_seeds": all(
            row["secondary_exploratory_diagnosis"][name] for row in per_seed.values()
        )
        for name in secondary_names
    }
    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "registration_parent_commit": specification["registration_parent_commit"],
        "claim_boundary": specification["claim_boundary"],
        "analysis_provenance": {
            "registered": "all estimands and rules in the frozen specification",
            "post_result_exploratory_robustness": [
                "DA correlations residualized within subject by relation exposure",
                "actual-alpha functional-amplification contrasts",
            ],
        },
        "device": {
            "neural_evaluation": default_device(),
            "exact_posterior": "cpu_numpy",
        },
        "artifact_validation": validation,
        "execution_contract": specification["execution_contract"],
        "step_trace_contract": specification["step_trace_contract"],
        "matched_evidence_contract": specification["matched_evidence_contract"],
        "bootstrap": bootstrap,
        "pilot_seeds": per_seed,
        "overall_diagnosis": overall,
        "overall_secondary_exploratory_diagnosis": secondary_overall,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run registered frozen support-write localization."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_support_write_localization(parsed.specification)
    write_json_exclusive(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
