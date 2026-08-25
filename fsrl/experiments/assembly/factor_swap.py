"""Registered causal factor swaps for frozen support-time plasticity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.hodge import (
    CompleteGraphGeometry,
    build_complete_graph_geometry,
    hodge_potentials,
)
from fsrl.analysis.statistics import (
    bootstrap_counts,
    json_values,
    masked_column_mean,
    summarize_subjects,
)
from fsrl.core.config import DEVICE, NUMRESPONSESTEP
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.experiments.assembly.trajectory import (
    load_frozen_evaluator,
    ordered_query_schedule,
)
from fsrl.experiments.assembly.write_localization import (
    matrix_norm,
    readout_effective_margin_fields,
    replay_without_relation_history,
    row_cosine,
    trace_support_trial,
)
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import (
    resolve_record,
    validate_registered_file,
)
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/support_factor_swap_v1.json")
DEFAULT_OUTPUT_PATH = resolve_record("results/support_factor_swap_v1.json")


@dataclass(frozen=True)
class EpisodeFactors:
    da_mean: torch.Tensor
    da_difference: torch.Tensor
    eligibility_mean: torch.Tensor
    eligibility_difference: torch.Tensor
    baseline_modulation: torch.Tensor
    baseline_fields: np.ndarray
    plus_fields: np.ndarray
    natural_potential_updates: np.ndarray
    retained: np.ndarray
    exposure: np.ndarray
    relations: np.ndarray
    validation: dict


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    names = (
        "pilot_specification",
        "protocol",
        "support_write_specification",
        "support_write_result",
        "support_write_implementation",
        "model_equation_source",
        "frozen_evaluator_source",
    )
    validated = {name: validate_registered_file(sources[name]) for name in names}
    artifacts = []
    for registration in sources["pilot_artifacts"]:
        row = {"seed": int(registration["seed"])}
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


def norm_match_trailing(
    candidate: torch.Tensor, target: torch.Tensor, *, trailing_dimensions: int
) -> torch.Tensor:
    if candidate.shape != target.shape:
        raise ValueError("candidate and target must have identical shapes")
    axes = tuple(range(candidate.ndim - trailing_dimensions, candidate.ndim))
    candidate_norm = torch.sqrt(torch.sum(candidate * candidate, dim=axes))
    target_norm = torch.sqrt(torch.sum(target * target, dim=axes))
    scale = torch.where(
        candidate_norm > 0.0,
        target_norm / candidate_norm,
        torch.zeros_like(candidate_norm),
    )
    return candidate * scale.reshape(scale.shape + (1,) * trailing_dimensions)


def compose_factors(
    da_mean: torch.Tensor,
    da_difference: torch.Tensor,
    eligibility_mean: torch.Tensor,
    eligibility_difference: torch.Tensor,
) -> torch.Tensor:
    """Compose the exact matched write from separable DA and E factors."""

    if da_mean.shape != da_difference.shape:
        raise ValueError("DA factors must have identical shapes")
    if eligibility_mean.shape != eligibility_difference.shape:
        raise ValueError("eligibility factors must have identical shapes")
    if da_mean.shape != eligibility_mean.shape[:-2]:
        raise ValueError("DA and eligibility factor shapes do not align")
    return torch.sum(
        da_difference[..., None, None] * eligibility_mean
        + da_mean[..., None, None] * eligibility_difference,
        dim=-3,
    )


def readout_effective_margin_fields_batched(
    evaluator: FrozenFastWeightEvaluator,
    effective_modulation: torch.Tensor,
    geometry: CompleteGraphGeometry,
) -> np.ndarray:
    """Read many effective-connectivity conditions in one GPU batch."""

    expected_tail = (
        evaluator.config.bs,
        evaluator.config.hs,
        evaluator.config.hs,
    )
    if (
        effective_modulation.ndim != 4
        or effective_modulation.shape[1:] != expected_tail
    ):
        raise ValueError("effective_modulation has the wrong batched shape")
    conditions = effective_modulation.shape[0]
    subjects = evaluator.config.bs
    schedules = ordered_query_schedule(geometry, subjects)
    oriented_responses = np.empty(
        (conditions, subjects, len(schedules[0])), dtype=np.float64
    )
    flattened_modulation = effective_modulation.reshape(
        conditions * subjects, evaluator.config.hs, evaluator.config.hs
    )

    with torch.no_grad():
        for pair_index in range(len(schedules[0])):
            hidden = evaluator.net.initialZeroState(conditions * subjects)
            left = np.asarray(
                [schedule[pair_index][0] for schedule in schedules], dtype=np.int64
            )
            right = np.asarray(
                [schedule[pair_index][1] for schedule in schedules], dtype=np.int64
            )
            signed = np.zeros(subjects, dtype=np.float32)
            response = None
            for step in range(evaluator.config.triallen):
                base_inputs = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=step,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                inputs = base_inputs.repeat(conditions, 1)
                hidden = evaluator.net.activ(
                    evaluator.net.i2h(inputs).view(
                        conditions * subjects, evaluator.config.hs, 1
                    )
                    + torch.matmul(
                        evaluator.net.w + flattened_modulation,
                        hidden.view(conditions * subjects, evaluator.config.hs, 1),
                    )
                ).view(conditions * subjects, evaluator.config.hs)
                logits = evaluator.net.h2o(hidden)
                if step == NUMRESPONSESTEP:
                    response = (logits[:, 1] - logits[:, 0]).reshape(
                        conditions, subjects
                    )
            assert response is not None
            oriented_responses[:, :, pair_index] = response.detach().cpu().numpy()

    return 0.5 * (oriented_responses[:, :, 0::2] - oriented_responses[:, :, 1::2])


def _potential_updates(
    fields: np.ndarray,
    baseline_fields: np.ndarray,
    geometry: CompleteGraphGeometry,
) -> np.ndarray:
    return hodge_potentials(fields - baseline_fields, geometry)


def _trial_metadata(
    evaluator: FrozenFastWeightEvaluator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = []
    exposure = []
    relations = []
    for trial_index in range(evaluator.protocol.support_trials):
        retained.append(
            [
                evaluator._encoding_reliability(subject, trial_index) > 0.0
                for subject in range(evaluator.config.bs)
            ]
        )
        exposure.append(
            [
                schedule[trial_index].block_index + 1
                for schedule in evaluator.support_schedules
            ]
        )
        relations.append(
            [
                (
                    schedule[trial_index].higher_item,
                    schedule[trial_index].lower_item,
                )
                for schedule in evaluator.support_schedules
            ]
        )
    return (
        np.asarray(retained, dtype=bool),
        np.asarray(exposure, dtype=np.int64),
        np.asarray(relations, dtype=np.int64),
    )


def trace_natural_episode(
    evaluator: FrozenFastWeightEvaluator,
    geometry: CompleteGraphGeometry,
    effective_steps: tuple[int, ...],
) -> EpisodeFactors:
    alpha = evaluator.net.alpha.detach()
    natural = evaluator.initialize_fast_weights()
    da_mean_rows = []
    da_difference_rows = []
    eligibility_mean_rows = []
    eligibility_difference_rows = []
    zero_modulation_rows = []
    plus_modulation_rows = []
    validation = {
        "trace_forward_max_abs_error": 0.0,
        "incremental_endpoint_max_abs_error": 0.0,
        "factor_identity_max_abs_error": 0.0,
        "final_endpoint_max_abs_error": 0.0,
        "batched_readout_max_abs_error": 0.0,
        "natural_common_replay_max_abs_error": 0.0,
    }

    for trial_index in range(evaluator.protocol.support_trials):
        if trial_index % 4 == 0:
            print(
                f"[factor-swap] trace trial {trial_index + 1}/"
                f"{evaluator.protocol.support_trials}",
                file=sys.stderr,
            )
        plus = trace_support_trial(evaluator, natural, trial_index)
        zero = trace_support_trial(
            evaluator,
            natural,
            trial_index,
            evidence_scales=np.zeros(evaluator.config.bs, dtype=np.float32),
        )
        reference = evaluator.advance_support_trial(natural, trial_index)
        validation["trace_forward_max_abs_error"] = max(
            validation["trace_forward_max_abs_error"],
            plus.forward_max_abs_error,
            zero.forward_max_abs_error,
        )
        validation["incremental_endpoint_max_abs_error"] = max(
            validation["incremental_endpoint_max_abs_error"],
            float(torch.max(torch.abs(reference - plus.final_fast_weights)).item()),
        )
        step_index = torch.as_tensor(effective_steps, device=plus.da.device)
        da_plus = torch.index_select(plus.da, 1, step_index)
        da_zero = torch.index_select(zero.da, 1, step_index)
        eligibility_plus = torch.index_select(plus.eligibility_before, 1, step_index)
        eligibility_zero = torch.index_select(zero.eligibility_before, 1, step_index)
        da_mean = 0.5 * (da_plus + da_zero)
        da_difference = da_plus - da_zero
        eligibility_mean = 0.5 * (eligibility_plus + eligibility_zero)
        eligibility_difference = eligibility_plus - eligibility_zero
        composed = compose_factors(
            da_mean, da_difference, eligibility_mean, eligibility_difference
        )
        direct = torch.sum(
            torch.index_select(
                plus.intended_increment - zero.intended_increment, 1, step_index
            ),
            dim=1,
        )
        validation["factor_identity_max_abs_error"] = max(
            validation["factor_identity_max_abs_error"],
            float(torch.max(torch.abs(composed - direct)).item()),
        )
        da_mean_rows.append(da_mean)
        da_difference_rows.append(da_difference)
        eligibility_mean_rows.append(eligibility_mean)
        eligibility_difference_rows.append(eligibility_difference)
        zero_modulation_rows.append(alpha * zero.final_fast_weights)
        plus_modulation_rows.append(alpha * plus.final_fast_weights)
        natural = plus.final_fast_weights

    da_mean_tensor = torch.stack(da_mean_rows)
    da_difference_tensor = torch.stack(da_difference_rows)
    eligibility_mean_tensor = torch.stack(eligibility_mean_rows)
    eligibility_difference_tensor = torch.stack(eligibility_difference_rows)
    baseline_modulation = torch.stack(zero_modulation_rows)
    plus_modulation = torch.stack(plus_modulation_rows)
    all_fields = readout_effective_margin_fields_batched(
        evaluator,
        torch.cat((baseline_modulation, plus_modulation), dim=0),
        geometry,
    )
    trials = evaluator.protocol.support_trials
    baseline_fields = all_fields[:trials]
    plus_fields = all_fields[trials:]
    natural_potential_updates = _potential_updates(
        plus_fields, baseline_fields, geometry
    )
    unbatched = readout_effective_margin_fields(
        evaluator, baseline_modulation[0], geometry
    )
    validation["batched_readout_max_abs_error"] = float(
        np.max(np.abs(unbatched - baseline_fields[0]))
    )
    natural_composed = compose_factors(
        da_mean_tensor,
        da_difference_tensor,
        eligibility_mean_tensor,
        eligibility_difference_tensor,
    )
    common_replay_fields = readout_effective_margin_fields_batched(
        evaluator,
        baseline_modulation + alpha * natural_composed,
        geometry,
    )
    validation["natural_common_replay_max_abs_error"] = float(
        np.max(np.abs(common_replay_fields - plus_fields))
    )
    final_reference = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    validation["final_endpoint_max_abs_error"] = float(
        torch.max(torch.abs(natural - final_reference)).item()
    )
    retained, exposure, relations = _trial_metadata(evaluator)
    return EpisodeFactors(
        da_mean=da_mean_tensor,
        da_difference=da_difference_tensor,
        eligibility_mean=eligibility_mean_tensor,
        eligibility_difference=eligibility_difference_tensor,
        baseline_modulation=baseline_modulation,
        baseline_fields=baseline_fields,
        plus_fields=plus_fields,
        natural_potential_updates=natural_potential_updates,
        retained=retained,
        exposure=exposure,
        relations=relations,
        validation=validation,
    )


def donor_indices(
    retained: np.ndarray,
    exposure: np.ndarray,
    relations: np.ndarray,
    canonical_relations: tuple[tuple[int, int], ...],
) -> np.ndarray:
    canonical = {tuple(pair): index for index, pair in enumerate(canonical_relations)}
    donors = np.full(retained.shape, -1, dtype=np.int64)
    for subject in range(retained.shape[1]):
        for exposure_value in range(1, 5):
            selected = [
                trial
                for trial in range(retained.shape[0])
                if retained[trial, subject]
                and exposure[trial, subject] == exposure_value
            ]
            ordered = sorted(
                selected,
                key=lambda trial: canonical[tuple(relations[trial, subject])],
            )
            if len(ordered) < 2:
                raise RuntimeError("eligibility donor set has fewer than two trials")
            for position, trial in enumerate(ordered):
                donors[trial, subject] = ordered[(position + 1) % len(ordered)]
    return donors


def da_extreme_indices(
    factors: EpisodeFactors,
    canonical_relations: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    canonical = {tuple(pair): index for index, pair in enumerate(canonical_relations)}
    strength = torch.linalg.vector_norm(factors.da_mean, dim=-1).detach().cpu().numpy()
    low = np.full(factors.retained.shape, -1, dtype=np.int64)
    high = np.full(factors.retained.shape, -1, dtype=np.int64)
    for subject in range(factors.retained.shape[1]):
        for exposure_value in range(1, 5):
            selected = [
                trial
                for trial in range(factors.retained.shape[0])
                if factors.retained[trial, subject]
                and factors.exposure[trial, subject] == exposure_value
            ]
            low_trial = min(
                selected,
                key=lambda trial: (
                    strength[trial, subject],
                    canonical[tuple(factors.relations[trial, subject])],
                ),
            )
            high_trial = min(
                selected,
                key=lambda trial: (
                    -strength[trial, subject],
                    canonical[tuple(factors.relations[trial, subject])],
                ),
            )
            for trial in selected:
                low[trial, subject] = low_trial
                high[trial, subject] = high_trial
    return low, high


def _gather_trial_subject(values, indices: np.ndarray):
    safe = np.where(indices >= 0, indices, 0)
    subjects = np.broadcast_to(np.arange(indices.shape[1]), indices.shape).copy()
    if isinstance(values, torch.Tensor):
        return values[
            torch.as_tensor(safe, device=values.device),
            torch.as_tensor(subjects, device=values.device),
        ]
    return values[safe, subjects]


def eligibility_transfer_metrics(
    evaluator: FrozenFastWeightEvaluator,
    factors: EpisodeFactors,
    geometry: CompleteGraphGeometry,
    donors: np.ndarray,
    tolerance: float,
) -> dict[str, np.ndarray]:
    donor_delta_e = _gather_trial_subject(factors.eligibility_difference, donors)
    matched_delta_e = norm_match_trailing(
        donor_delta_e,
        factors.eligibility_difference,
        trailing_dimensions=3,
    )
    synthetic_u = torch.sum(factors.da_mean[..., None, None] * matched_delta_e, dim=2)
    recipient_u = torch.sum(
        factors.da_mean[..., None, None] * factors.eligibility_difference,
        dim=2,
    )
    alpha = evaluator.net.alpha.detach()
    synthetic_m = norm_match_trailing(
        alpha * synthetic_u,
        alpha * recipient_u,
        trailing_dimensions=2,
    )
    fields = readout_effective_margin_fields_batched(
        evaluator, factors.baseline_modulation + synthetic_m, geometry
    )
    synthetic_potential = _potential_updates(fields, factors.baseline_fields, geometry)
    donor_potential = _gather_trial_subject(factors.natural_potential_updates, donors)
    recipient_potential = factors.natural_potential_updates
    donor_cosine = row_cosine(synthetic_potential, donor_potential, tolerance)
    recipient_cosine = row_cosine(synthetic_potential, recipient_potential, tolerance)
    target_cosine = row_cosine(donor_potential, recipient_potential, tolerance)
    target_norm = matrix_norm(alpha * recipient_u).detach().cpu().numpy()
    synthetic_norm = matrix_norm(synthetic_m).detach().cpu().numpy()
    return {
        "donor_cosine": donor_cosine,
        "recipient_cosine": recipient_cosine,
        "donor_identity_advantage": donor_cosine - recipient_cosine,
        "target_separation": 1.0 - target_cosine,
        "effective_norm_match_error": synthetic_norm - target_norm,
        "synthetic_policy_norm": np.linalg.norm(synthetic_potential, axis=-1),
    }


def da_transfer_metrics(
    evaluator: FrozenFastWeightEvaluator,
    factors: EpisodeFactors,
    geometry: CompleteGraphGeometry,
    low_indices: np.ndarray,
    high_indices: np.ndarray,
    tolerance: float,
) -> dict[str, np.ndarray]:
    low_u = compose_factors(
        _gather_trial_subject(factors.da_mean, low_indices),
        _gather_trial_subject(factors.da_difference, low_indices),
        factors.eligibility_mean,
        factors.eligibility_difference,
    )
    high_u = compose_factors(
        _gather_trial_subject(factors.da_mean, high_indices),
        _gather_trial_subject(factors.da_difference, high_indices),
        factors.eligibility_mean,
        factors.eligibility_difference,
    )
    alpha = evaluator.net.alpha.detach()
    low_m = alpha * low_u
    high_m = alpha * high_u
    trials = factors.retained.shape[0]
    fields = readout_effective_margin_fields_batched(
        evaluator,
        torch.cat(
            (
                factors.baseline_modulation + high_m,
                factors.baseline_modulation + low_m,
            ),
            dim=0,
        ),
        geometry,
    )
    high_potential = _potential_updates(
        fields[:trials], factors.baseline_fields, geometry
    )
    low_potential = _potential_updates(
        fields[trials:], factors.baseline_fields, geometry
    )
    high_policy_norm = np.linalg.norm(high_potential, axis=-1)
    low_policy_norm = np.linalg.norm(low_potential, axis=-1)
    high_write_norm = matrix_norm(high_m).detach().cpu().numpy()
    low_write_norm = matrix_norm(low_m).detach().cpu().numpy()
    return {
        "high_policy_norm": high_policy_norm,
        "low_policy_norm": low_policy_norm,
        "policy_norm_difference": high_policy_norm - low_policy_norm,
        "high_write_norm": high_write_norm,
        "low_write_norm": low_write_norm,
        "write_norm_difference": high_write_norm - low_write_norm,
        "direction_cosine": row_cosine(high_potential, low_potential, tolerance),
    }


def history_factorial_metrics(
    evaluator: FrozenFastWeightEvaluator,
    factors: EpisodeFactors,
    geometry: CompleteGraphGeometry,
    effective_steps: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict]:
    trial_indices = np.flatnonzero(np.all(factors.exposure == 4, axis=1))
    if len(trial_indices) != len(evaluator.protocol.support_pairs_higher_lower):
        raise RuntimeError("fourth-exposure trial count changed")
    alpha = evaluator.net.alpha.detach()
    all_modulations = [[] for _ in range(4)]
    all_write_norms = [[] for _ in range(4)]
    validation = {"history_nn_replay_max_abs_error": 0.0}

    for trial_index in trial_indices:
        relations = tuple(
            tuple(values) for values in factors.relations[trial_index].tolist()
        )
        history = replay_without_relation_history(
            evaluator, int(trial_index), relations
        )
        history_plus = trace_support_trial(evaluator, history, int(trial_index))
        history_zero = trace_support_trial(
            evaluator,
            history,
            int(trial_index),
            evidence_scales=np.zeros(evaluator.config.bs, dtype=np.float32),
        )
        step_index = torch.as_tensor(effective_steps, device=history.device)
        history_da_plus = torch.index_select(history_plus.da, 1, step_index)
        history_da_zero = torch.index_select(history_zero.da, 1, step_index)
        history_e_plus = torch.index_select(
            history_plus.eligibility_before, 1, step_index
        )
        history_e_zero = torch.index_select(
            history_zero.eligibility_before, 1, step_index
        )
        history_da_mean = 0.5 * (history_da_plus + history_da_zero)
        history_da_difference = history_da_plus - history_da_zero
        history_e_mean = 0.5 * (history_e_plus + history_e_zero)
        history_e_difference = history_e_plus - history_e_zero
        natural_da_mean = factors.da_mean[trial_index]
        natural_da_difference = factors.da_difference[trial_index]
        natural_e_mean = factors.eligibility_mean[trial_index]
        natural_e_difference = factors.eligibility_difference[trial_index]
        combinations = (
            (
                natural_da_mean,
                natural_da_difference,
                natural_e_mean,
                natural_e_difference,
            ),
            (
                history_da_mean,
                history_da_difference,
                natural_e_mean,
                natural_e_difference,
            ),
            (
                natural_da_mean,
                natural_da_difference,
                history_e_mean,
                history_e_difference,
            ),
            (
                history_da_mean,
                history_da_difference,
                history_e_mean,
                history_e_difference,
            ),
        )
        for condition, combination in enumerate(combinations):
            write = compose_factors(*combination)
            effective = alpha * write
            all_modulations[condition].append(
                factors.baseline_modulation[trial_index] + effective
            )
            all_write_norms[condition].append(
                matrix_norm(effective).detach().cpu().numpy()
            )

    condition_modulations = [torch.stack(rows) for rows in all_modulations]
    fields = readout_effective_margin_fields_batched(
        evaluator, torch.cat(condition_modulations, dim=0), geometry
    )
    count = len(trial_indices)
    baseline = factors.baseline_fields[trial_indices]
    policy_norms = []
    for condition in range(4):
        condition_fields = fields[condition * count : (condition + 1) * count]
        potential = _potential_updates(condition_fields, baseline, geometry)
        policy_norms.append(np.linalg.norm(potential, axis=-1))
    validation["history_nn_replay_max_abs_error"] = float(
        np.max(np.abs(fields[:count] - factors.plus_fields[trial_indices]))
    )
    write_norms = [np.asarray(rows) for rows in all_write_norms]

    def contrasts(values: list[np.ndarray]) -> dict[str, np.ndarray]:
        nn, hn, nh, hh = values
        return {
            "natural_natural": nn,
            "history_da_natural_e": hn,
            "natural_da_history_e": nh,
            "history_history": hh,
            "total_restoration": hh - nn,
            "da_main_effect": 0.5 * ((hn - nn) + (hh - nh)),
            "eligibility_main_effect": 0.5 * ((nh - nn) + (hh - hn)),
            "interaction": hh - hn - nh + nn,
        }

    return {
        "trial_indices": trial_indices,
        "retained": factors.retained[trial_indices],
        "policy": contrasts(policy_norms),
        "write": contrasts(write_norms),
    }, validation


def alpha_gain_metrics(
    evaluator: FrozenFastWeightEvaluator,
    factors: EpisodeFactors,
    geometry: CompleteGraphGeometry,
    *,
    permutation_count: int,
    permutation_seed: int,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], dict]:
    alpha = evaluator.net.alpha.detach()
    flattened = alpha.detach().cpu().numpy().reshape(-1)
    rng = np.random.default_rng(permutation_seed)
    permuted = np.stack(
        [flattened[rng.permutation(alpha.numel())] for _ in range(permutation_count)]
    ).reshape(permutation_count, *alpha.shape)
    null_alpha = torch.from_numpy(permuted).to(alpha.device)
    natural_u = compose_factors(
        factors.da_mean,
        factors.da_difference,
        factors.eligibility_mean,
        factors.eligibility_difference,
    )
    trials, subjects = factors.retained.shape
    actual_gain = np.full((trials, subjects), np.nan, dtype=np.float64)
    null_mean = np.full_like(actual_gain, np.nan)
    null_q05 = np.full_like(actual_gain, np.nan)
    null_median = np.full_like(actual_gain, np.nan)
    null_q95 = np.full_like(actual_gain, np.nan)
    fraction_beaten = np.full_like(actual_gain, np.nan)
    percentile = np.full_like(actual_gain, np.nan)
    validation = {"alpha_actual_replay_max_abs_error": 0.0}

    for trial_index in range(trials):
        if trial_index % 4 == 0:
            print(
                f"[factor-swap] alpha null trial {trial_index + 1}/{trials}",
                file=sys.stderr,
            )
        trial_u = natural_u[trial_index]
        actual_m = alpha * trial_u
        null_m = null_alpha[:, None, :, :] * trial_u[None, :, :, :]
        target = actual_m[None, :, :, :].expand_as(null_m)
        null_m = norm_match_trailing(null_m, target, trailing_dimensions=2)
        increments = torch.cat((actual_m[None, :, :, :], null_m), dim=0)
        fields = readout_effective_margin_fields_batched(
            evaluator,
            factors.baseline_modulation[trial_index][None, :, :, :] + increments,
            geometry,
        )
        potential = _potential_updates(
            fields, factors.baseline_fields[trial_index][None, :, :], geometry
        )
        policy_norm = np.linalg.norm(potential, axis=-1)
        increment_norm = matrix_norm(increments).detach().cpu().numpy()
        gains = np.divide(
            policy_norm,
            increment_norm,
            out=np.full_like(policy_norm, np.nan),
            where=increment_norm > tolerance,
        )
        actual_gain[trial_index] = gains[0]
        null_values = gains[1:]
        valid = np.isfinite(gains[0])
        if np.any(valid):
            valid_null = null_values[:, valid]
            null_mean[trial_index, valid] = np.mean(valid_null, axis=0)
            null_q05[trial_index, valid] = np.quantile(valid_null, 0.05, axis=0)
            null_median[trial_index, valid] = np.median(valid_null, axis=0)
            null_q95[trial_index, valid] = np.quantile(valid_null, 0.95, axis=0)
            beaten = valid_null < gains[0, valid][None, :]
            fraction_beaten[trial_index, valid] = np.mean(beaten, axis=0)
            percentile[trial_index, valid] = (1.0 + np.sum(beaten, axis=0)) / (
                permutation_count + 1.0
            )
        validation["alpha_actual_replay_max_abs_error"] = max(
            validation["alpha_actual_replay_max_abs_error"],
            float(np.max(np.abs(fields[0] - factors.plus_fields[trial_index]))),
        )
    return {
        "actual_gain": actual_gain,
        "null_mean_gain": null_mean,
        "actual_minus_null_mean_gain": actual_gain - null_mean,
        "null_q05_gain": null_q05,
        "null_median_gain": null_median,
        "null_q95_gain": null_q95,
        "fraction_null_beaten": fraction_beaten,
        "empirical_percentile": percentile,
    }, validation


def _summarize_metrics(
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> tuple[dict, dict]:
    subject_values = {
        name: masked_column_mean(values, mask) for name, values in metrics.items()
    }
    return (
        {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in subject_values.items()
        },
        {name: json_values(values) for name, values in subject_values.items()},
    )


def summarize_seed(
    factors: EpisodeFactors,
    eligibility: dict[str, np.ndarray],
    da: dict[str, np.ndarray],
    history: dict,
    alpha: dict[str, np.ndarray],
    validation: dict,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict]:
    retained = factors.retained
    eligibility_summary, eligibility_raw = _summarize_metrics(
        eligibility, retained, counts, interval
    )
    da_summary, da_raw = _summarize_metrics(da, retained, counts, interval)
    alpha_summary, alpha_raw = _summarize_metrics(alpha, retained, counts, interval)
    history_summary = {}
    history_raw = {}
    for outcome in ("policy", "write"):
        history_summary[outcome], history_raw[outcome] = _summarize_metrics(
            history[outcome], history["retained"], counts, interval
        )

    by_exposure = {}
    for exposure_value in range(1, 5):
        mask = retained & (factors.exposure == exposure_value)
        by_exposure[str(exposure_value)] = {
            "eligibility": _summarize_metrics(eligibility, mask, counts, interval)[0],
            "da": _summarize_metrics(da, mask, counts, interval)[0],
            "alpha": _summarize_metrics(alpha, mask, counts, interval)[0],
        }

    reproduction_tolerance = float(max(tolerance, 32.0 * np.finfo(np.float32).eps))
    validation["implementation_reproduction_tolerance"] = reproduction_tolerance
    for name in (
        "trace_forward_max_abs_error",
        "incremental_endpoint_max_abs_error",
        "factor_identity_max_abs_error",
        "final_endpoint_max_abs_error",
        "batched_readout_max_abs_error",
        "natural_common_replay_max_abs_error",
        "history_nn_replay_max_abs_error",
        "alpha_actual_replay_max_abs_error",
    ):
        if validation[name] > reproduction_tolerance:
            raise RuntimeError(f"implementation validation failed: {name}")
    if (
        abs(eligibility_summary["effective_norm_match_error"]["mean"] or 0.0)
        > reproduction_tolerance
    ):
        raise RuntimeError("eligibility direction swap changed registered norm")

    decisions = {
        "eligibility_identity_transfer": (
            eligibility_summary["target_separation"]["bootstrap"]["lower"] > 0.0
            and eligibility_summary["donor_identity_advantage"]["bootstrap"]["lower"]
            > 0.0
        ),
        "da_magnitude_transfer": (
            da_summary["write_norm_difference"]["bootstrap"]["lower"] > 0.0
            and da_summary["policy_norm_difference"]["bootstrap"]["lower"] > 0.0
        ),
        "history_policy_attribution_competent": history_summary["policy"][
            "total_restoration"
        ]["bootstrap"]["lower"]
        > 0.0,
        "history_write_restoration": history_summary["write"]["total_restoration"][
            "bootstrap"
        ]["lower"]
        > 0.0,
        "history_policy_da_main_positive": history_summary["policy"]["da_main_effect"][
            "bootstrap"
        ]["lower"]
        > 0.0,
        "history_policy_eligibility_main_positive": history_summary["policy"][
            "eligibility_main_effect"
        ]["bootstrap"]["lower"]
        > 0.0,
        "history_policy_interaction_positive": history_summary["policy"]["interaction"][
            "bootstrap"
        ]["lower"]
        > 0.0,
        "history_write_da_main_positive": history_summary["write"]["da_main_effect"][
            "bootstrap"
        ]["lower"]
        > 0.0,
        "history_write_eligibility_main_positive": history_summary["write"][
            "eligibility_main_effect"
        ]["bootstrap"]["lower"]
        > 0.0,
        "history_write_interaction_positive": history_summary["write"]["interaction"][
            "bootstrap"
        ]["lower"]
        > 0.0,
        "alpha_systematic_high_gain": alpha_summary["actual_minus_null_mean_gain"][
            "bootstrap"
        ]["lower"]
        > 0.0,
    }
    return {
        "eligibility_identity_transfer": eligibility_summary,
        "da_magnitude_transfer": da_summary,
        "exposure_four_history_factorial": history_summary,
        "alpha_systematic_gain": alpha_summary,
        "by_exposure": by_exposure,
        "registered_directional_diagnosis": decisions,
        "validation": validation,
    }, {
        "eligibility_identity_transfer": eligibility_raw,
        "da_magnitude_transfer": da_raw,
        "exposure_four_history_factorial": history_raw,
        "alpha_systematic_gain": alpha_raw,
    }


def run_support_factor_swap(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    artifact_validation = validate_registered_sources(specification)
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    contract = specification["execution_contract"]
    effective_steps = tuple(int(value) for value in contract["effective_support_steps"])
    tolerance = float(contract["floating_zero_tolerance"])
    bootstrap = specification["bootstrap"]
    interval = float(bootstrap["interval"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    per_seed = {}

    for registration in sources["pilot_artifacts"]:
        seed = int(registration["seed"])
        print(f"[factor-swap] loading frozen seed {seed}", file=sys.stderr)
        evaluator, behavior = load_frozen_evaluator(
            registration, pilot_specification, protocol
        )
        counts = bootstrap_counts(rng, int(bootstrap["samples"]), evaluator.config.bs)
        factors = trace_natural_episode(evaluator, geometry, effective_steps)
        donors = donor_indices(
            factors.retained,
            factors.exposure,
            factors.relations,
            protocol.support_pairs_higher_lower,
        )
        eligibility = eligibility_transfer_metrics(
            evaluator, factors, geometry, donors, tolerance
        )
        low, high = da_extreme_indices(factors, protocol.support_pairs_higher_lower)
        da = da_transfer_metrics(evaluator, factors, geometry, low, high, tolerance)
        history, history_validation = history_factorial_metrics(
            evaluator, factors, geometry, effective_steps
        )
        alpha, alpha_validation = alpha_gain_metrics(
            evaluator,
            factors,
            geometry,
            permutation_count=int(contract["alpha_permutation_count"]),
            permutation_seed=int(contract["alpha_permutation_seed"]),
            tolerance=tolerance,
        )
        validation = {
            **factors.validation,
            **history_validation,
            **alpha_validation,
        }
        summary, raw = summarize_seed(
            factors,
            eligibility,
            da,
            history,
            alpha,
            validation,
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
    overall = {
        f"{name}_replicated_across_pilot_seeds": all(
            row["registered_directional_diagnosis"][name] for row in per_seed.values()
        )
        for name in diagnosis_names
    }
    overall["formal_confirmation_status"] = "deferred; formal seeds remain untouched"
    overall["pilot_stop_rule"] = specification["decision_logic"]["pilot_stop_rule"]
    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "registration_parent_commit": specification["registration_parent_commit"],
        "claim_boundary": specification["claim_boundary"],
        "working_theory": specification["working_theory"],
        "device": {"neural_replay": DEVICE, "summaries": "cpu_numpy"},
        "artifact_validation": artifact_validation,
        "execution_contract": contract,
        "factor_contract": specification["factor_contract"],
        "bootstrap": bootstrap,
        "pilot_seeds": per_seed,
        "overall_diagnosis": overall,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run registered frozen support-factor swaps."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_support_factor_swap(parsed.specification)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
