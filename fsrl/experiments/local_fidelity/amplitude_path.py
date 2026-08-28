"""Trace frozen relation-operator semantics over a fixed amplitude path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.hodge import CompleteGraphGeometry, build_complete_graph_geometry
from fsrl.analysis.statistics import bootstrap_counts, json_values, summarize_subjects
from fsrl.core.config import NUMRESPONSESTEP
from fsrl.evaluation.fields import ordered_query_schedule
from fsrl.experiments.assembly.trajectory import load_frozen_evaluator
from fsrl.experiments.local_fidelity.hidden_residual import validate_registered_sources
from fsrl.experiments.local_fidelity.operator_binding import replay_terminal_states
from fsrl.experiments.local_fidelity.output_semantics import (
    STAGES,
    hodge_components,
    masked_relation_mean,
    normalized_direct_correctness,
    relation_geometry,
    stack_step,
    stage_relation_metrics,
)
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import legacy_identifier, resolve_record
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/operator_amplitude_path_v1.json"
)
DEFAULT_OUTPUT_PATH = resolve_record("results/operator_amplitude_path_v1.json")


def quadratic_coefficient(baseline: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return the coefficient of lambda squared in tanh(b + lambda A)-tanh(b)."""

    baseline_hidden = torch.tanh(baseline)
    sensitivity = 1.0 - baseline_hidden.square()
    return -baseline_hidden * sensitivity * action.square()


def curve_status(summary: dict, amplitude: float) -> str:
    if amplitude == 0.0:
        return "structural_zero"
    lower = summary["bootstrap"]["lower"]
    upper = summary["bootstrap"]["upper"]
    if lower is not None and lower > 0.0:
        return "robust_positive"
    if upper is not None and upper < 0.0:
        return "robust_negative"
    return "unresolved"


def mean_sign_change_bracket(amplitudes: np.ndarray, means: np.ndarray) -> dict | None:
    grid = np.asarray(amplitudes, dtype=np.float64)
    values = np.asarray(means, dtype=np.float64)
    for index in range(1, len(grid) - 1):
        if values[index] > 0.0 and values[index + 1] <= 0.0:
            return {
                "lambda_minus": float(grid[index]),
                "lambda_plus": float(grid[index + 1]),
                "value_minus": float(values[index]),
                "value_plus": float(values[index + 1]),
            }
    return None


def robust_transition_bracket(
    amplitudes: np.ndarray, statuses: list[str]
) -> dict | None:
    grid = np.asarray(amplitudes, dtype=np.float64)
    for negative_index in range(1, len(grid)):
        if statuses[negative_index] != "robust_negative":
            continue
        positive = [
            index
            for index in range(1, negative_index)
            if statuses[index] == "robust_positive"
        ]
        if positive:
            return {
                "last_robust_positive_lambda": float(grid[max(positive)]),
                "first_later_robust_negative_lambda": float(grid[negative_index]),
            }
    return None


def crossing_regime(bracket: dict | None) -> str:
    if bracket is None:
        return "no_crossing"
    if bracket["lambda_plus"] <= 0.20:
        return "early"
    if bracket["lambda_minus"] >= 0.60:
        return "late"
    return "intermediate"


def subject_crossing_summary(
    values: np.ndarray,
    retained: np.ndarray,
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    """Summarize first fixed-grid positive-to-nonpositive subject crossings."""

    curve = np.asarray(values, dtype=np.float64)
    mask = np.asarray(retained, dtype=bool)
    grid = np.asarray(amplitudes, dtype=np.float64)
    if curve.shape != (len(grid), len(mask)):
        raise ValueError("subject curve does not match amplitude and subject axes")
    lower = np.full(len(mask), np.nan)
    upper = np.full(len(mask), np.nan)
    ever_positive = np.any(curve[1:] > 0.0, axis=0)
    for subject in np.flatnonzero(mask):
        for index in range(1, len(grid) - 1):
            if curve[index, subject] > 0.0 and curve[index + 1, subject] <= 0.0:
                lower[subject] = grid[index]
                upper[subject] = grid[index + 1]
                break
    crossed = np.isfinite(lower) & mask
    nonpositive_first = (curve[1] <= 0.0) & mask
    negative_at_one = (curve[-1] < 0.0) & mask
    positive_without_crossing = ever_positive & ~crossed & mask
    never_positive = ~ever_positive & mask
    reentered = np.zeros(len(mask), dtype=bool)
    for subject in np.flatnonzero(crossed):
        upper_index = int(np.flatnonzero(grid == upper[subject])[0])
        reentered[subject] = np.any(curve[upper_index + 1 :, subject] > 0.0)

    def indicator_summary(selector: np.ndarray) -> dict:
        return summarize_subjects(
            np.where(mask, selector.astype(np.float64), np.nan),
            counts,
            interval=interval,
        )

    bracket_labels = [
        f"{lower[subject]:.2f}-{upper[subject]:.2f}"
        for subject in np.flatnonzero(crossed)
    ]
    return {
        "retained_subjects": int(np.sum(mask)),
        "crossing_proportion": indicator_summary(crossed),
        "nonpositive_at_first_nonzero_proportion": indicator_summary(nonpositive_first),
        "negative_at_lambda_one_proportion": indicator_summary(negative_at_one),
        "positive_without_crossing_proportion": indicator_summary(
            positive_without_crossing
        ),
        "never_positive_proportion": indicator_summary(never_positive),
        "reentry_after_first_crossing_proportion": indicator_summary(reentered),
        "crossing_lambda_minus": summarize_subjects(lower, counts, interval=interval),
        "crossing_lambda_plus": summarize_subjects(upper, counts, interval=interval),
        "crossing_bracket_counts": dict(sorted(Counter(bracket_labels).items())),
        "raw_subject_level": {
            "lambda_minus": json_values(lower),
            "lambda_plus": json_values(upper),
            "crossed": [
                bool(value) if mask[index] else None
                for index, value in enumerate(crossed)
            ],
        },
    }


def select_v2_outcome(primary_regime: str, crossing_relation_count: int) -> str:
    if primary_regime == "no_crossing":
        return "integrity_contradiction_no_H_greater_A_crossing"
    widespread = crossing_relation_count >= 4
    if primary_regime == "late":
        return (
            "widespread_late_crossing_global_magnitude_v2"
            if widespread
            else "isolated_or_sparse_late_crossing_relation_conditioned_amplitude_v2"
        )
    return (
        "widespread_early_or_intermediate_crossing_near_linear_residual_v2"
        if widespread
        else "isolated_or_sparse_early_or_intermediate_crossing_relation_conditioned_routing_v2"
    )


def collect_amplitude_fields(
    evaluator,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    intact: torch.Tensor,
    loo: torch.Tensor,
    effective: torch.Tensor,
    retained: np.ndarray,
    amplitudes: np.ndarray,
    *,
    tolerance: float,
) -> tuple[dict, dict]:
    relation_count = len(protocol.support_pairs_higher_lower)
    subjects = evaluator.config.bs
    device = evaluator.device
    edge_count = len(geometry.pairs)
    oriented = tuple((pair, (pair[1], pair[0])) for pair in geometry.pairs)
    schedules = ordered_query_schedule(geometry, subjects)
    intact_hidden, intact_logits = evaluator.readout_hidden_and_logit_trajectories(
        intact, schedules
    )
    loo_trajectories = [
        evaluator.readout_hidden_and_logit_trajectories(loo[index], schedules)
        for index in range(relation_count)
    ]

    curve_scalars = np.empty(
        (len(amplitudes), relation_count, subjects, edge_count, 2),
        dtype=np.float64,
    )
    jacobian_scalars = np.empty(
        (relation_count, subjects, edge_count, 2), dtype=np.float64
    )
    curvature_scalars = np.empty_like(jacobian_scalars)
    output = (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0]).detach()
    retained_device = torch.from_numpy(retained).to(device)
    validation = {
        "manual_h0_max_abs_error": 0.0,
        "loo_h0_invariance_max_abs_error": 0.0,
        "lambda_one_preactivation_reconstruction_max_abs_error": 0.0,
        "lambda_one_hidden_reconstruction_max_abs_error": 0.0,
        "lambda_one_logit_influence_max_abs_error": 0.0,
        "stable_omitted_oriented_scalar_max_abs": 0.0,
    }

    with torch.no_grad():
        for edge, pair_orientations in enumerate(oriented):
            for orientation, pair in enumerate(pair_orientations):
                left = np.full(subjects, pair[0], dtype=np.int64)
                right = np.full(subjects, pair[1], dtype=np.int64)
                signed = np.zeros(subjects, dtype=np.float32)
                x0 = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                x1 = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=NUMRESPONSESTEP,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                h0 = evaluator.net.activ(evaluator.net.i2h(x0))
                h0_column = h0.view(1, subjects, evaluator.config.hs, 1)
                input_drive = evaluator.net.i2h(x1).view(
                    1, subjects, evaluator.config.hs, 1
                )
                baseline = input_drive + torch.matmul(
                    evaluator.net.w + evaluator.net.alpha.detach() * loo,
                    h0_column,
                )
                action = torch.matmul(effective, h0_column)
                baseline_hidden = torch.tanh(baseline)
                jacobian = (1.0 - baseline_hidden.square()) * action
                curvature = quadratic_coefficient(baseline, action)
                jacobian_scalars[:, :, edge, orientation] = (
                    torch.einsum("qsh,h->qs", jacobian[..., 0], output)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                curvature_scalars[:, :, edge, orientation] = (
                    torch.einsum("qsh,h->qs", curvature[..., 0], output)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                for amplitude_index, amplitude in enumerate(amplitudes):
                    hidden_effect = (
                        torch.tanh(baseline + float(amplitude) * action)
                        - baseline_hidden
                    )
                    scalars = torch.einsum("qsh,h->qs", hidden_effect[..., 0], output)
                    curve_scalars[amplitude_index, :, :, edge, orientation] = (
                        scalars.cpu().numpy().astype(np.float64)
                    )
                    validation["stable_omitted_oriented_scalar_max_abs"] = max(
                        validation["stable_omitted_oriented_scalar_max_abs"],
                        float(torch.max(torch.abs(scalars[~retained_device])).item()),
                    )

                intact_preactivation = input_drive[0] + torch.matmul(
                    evaluator.net.w + evaluator.net.alpha.detach() * intact,
                    h0.view(subjects, evaluator.config.hs, 1),
                )
                validation["lambda_one_preactivation_reconstruction_max_abs_error"] = (
                    max(
                        validation[
                            "lambda_one_preactivation_reconstruction_max_abs_error"
                        ],
                        float(
                            torch.max(
                                torch.abs(
                                    baseline + action - intact_preactivation[None]
                                )
                            ).item()
                        ),
                    )
                )
                actual_h0 = torch.from_numpy(stack_step(intact_hidden, pair, 0)).to(
                    device
                )
                validation["manual_h0_max_abs_error"] = max(
                    validation["manual_h0_max_abs_error"],
                    float(torch.max(torch.abs(actual_h0 - h0)).item()),
                )
                exact_one = torch.tanh(baseline + action) - baseline_hidden
                actual_intact_h1 = torch.from_numpy(
                    stack_step(intact_hidden, pair, NUMRESPONSESTEP)
                ).to(device)
                actual_intact_logit = torch.from_numpy(
                    stack_step(intact_logits, pair, NUMRESPONSESTEP)
                ).to(device)
                for relation_index, (loo_hidden, loo_logits) in enumerate(
                    loo_trajectories
                ):
                    actual_loo_h0 = torch.from_numpy(
                        stack_step(loo_hidden, pair, 0)
                    ).to(device)
                    actual_loo_h1 = torch.from_numpy(
                        stack_step(loo_hidden, pair, NUMRESPONSESTEP)
                    ).to(device)
                    actual_loo_logit = torch.from_numpy(
                        stack_step(loo_logits, pair, NUMRESPONSESTEP)
                    ).to(device)
                    validation["loo_h0_invariance_max_abs_error"] = max(
                        validation["loo_h0_invariance_max_abs_error"],
                        float(torch.max(torch.abs(actual_h0 - actual_loo_h0)).item()),
                    )
                    validation["lambda_one_hidden_reconstruction_max_abs_error"] = max(
                        validation["lambda_one_hidden_reconstruction_max_abs_error"],
                        float(
                            torch.max(
                                torch.abs(
                                    exact_one[relation_index, :, :, 0]
                                    - (actual_intact_h1 - actual_loo_h1)
                                )
                            ).item()
                        ),
                    )
                    projected = torch.einsum(
                        "sh,h->s", exact_one[relation_index, :, :, 0], output
                    )
                    validation["lambda_one_logit_influence_max_abs_error"] = max(
                        validation["lambda_one_logit_influence_max_abs_error"],
                        float(
                            torch.max(
                                torch.abs(
                                    projected - (actual_intact_logit - actual_loo_logit)
                                )
                            ).item()
                        ),
                    )

    curve_fields = 0.5 * (curve_scalars[..., 0] - curve_scalars[..., 1])
    jacobian_field = 0.5 * (jacobian_scalars[..., 0] - jacobian_scalars[..., 1])
    curvature_field = 0.5 * (curvature_scalars[..., 0] - curvature_scalars[..., 1])
    curve_residuals = np.empty_like(curve_fields)
    hodge_error = 0.0
    for amplitude_index in range(len(amplitudes)):
        gradient, residual = hodge_components(curve_fields[amplitude_index], geometry)
        curve_residuals[amplitude_index] = residual
        hodge_error = max(
            hodge_error,
            float(np.max(np.abs(curve_fields[amplitude_index] - gradient - residual))),
        )
    jacobian_gradient, jacobian_residual = hodge_components(jacobian_field, geometry)
    curvature_gradient, curvature_residual = hodge_components(curvature_field, geometry)
    hodge_error = max(
        hodge_error,
        float(np.max(np.abs(jacobian_field - jacobian_gradient - jacobian_residual))),
        float(
            np.max(np.abs(curvature_field - curvature_gradient - curvature_residual))
        ),
    )
    omitted_curve = np.broadcast_to((~retained)[None, ..., None], curve_fields.shape)
    omitted_stage = np.broadcast_to((~retained)[..., None], jacobian_field.shape)
    validation.update(
        {
            "lambda_zero_field_max_abs": float(np.max(np.abs(curve_fields[0]))),
            "lambda_zero_residual_max_abs": float(np.max(np.abs(curve_residuals[0]))),
            "stable_omitted_field_max_abs": float(
                np.max(np.abs(curve_fields[omitted_curve]))
            ),
            "stable_omitted_residual_max_abs": float(
                np.max(np.abs(curve_residuals[omitted_curve]))
            ),
            "stable_omitted_jacobian_max_abs": float(
                np.max(np.abs(jacobian_field[omitted_stage]))
            ),
            "stable_omitted_curvature_max_abs": float(
                np.max(np.abs(curvature_field[omitted_stage]))
            ),
            "hodge_reconstruction_max_abs_error": hodge_error,
            "floating_reproduction_tolerance": tolerance,
        }
    )
    reproduction_names = (
        "manual_h0_max_abs_error",
        "loo_h0_invariance_max_abs_error",
        "lambda_one_preactivation_reconstruction_max_abs_error",
        "lambda_one_hidden_reconstruction_max_abs_error",
        "lambda_one_logit_influence_max_abs_error",
        "hodge_reconstruction_max_abs_error",
    )
    if any(validation[name] > tolerance for name in reproduction_names):
        raise RuntimeError(f"amplitude-path reconstruction failed: {validation}")
    return {
        "curve_fields": curve_fields,
        "curve_residuals": curve_residuals,
        "jacobian_field": jacobian_field,
        "jacobian_residual": jacobian_residual,
        "curvature_field": curvature_field,
        "curvature_residual": curvature_residual,
    }, validation


def _aggregate_curve(
    metrics_by_amplitude: list[dict[str, np.ndarray]],
    correctness_signs: np.ndarray,
    retained: np.ndarray,
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    subject_values = {
        "direct_correctness": [],
        "normalized_direct_correctness_rho": [],
        "direct_minus_remote_correctness": [],
    }
    rows = []
    for amplitude, metrics in zip(amplitudes, metrics_by_amplitude, strict=True):
        current = {
            "direct_correctness": masked_relation_mean(
                metrics["direct_correctness"], retained
            ),
            "normalized_direct_correctness_rho": normalized_direct_correctness(
                metrics["direct_residual"],
                correctness_signs,
                retained,
                tolerance=tolerance,
            ),
            "direct_minus_remote_correctness": masked_relation_mean(
                metrics["direct_minus_remote_correctness"], retained
            ),
        }
        for name, values in current.items():
            subject_values[name].append(values)
        rows.append(
            {
                "lambda": float(amplitude),
                "summary": {
                    name: summarize_subjects(values, counts, interval=interval)
                    for name, values in current.items()
                },
                "raw_subject_level": {
                    name: json_values(values) for name, values in current.items()
                },
            }
        )
    return rows, {name: np.asarray(values) for name, values in subject_values.items()}


def _relation_curves(
    metrics_by_amplitude: list[dict[str, np.ndarray]],
    protocol: RankingProtocol,
    retained: np.ndarray,
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> list[dict]:
    rows = []
    for relation_index, relation in enumerate(protocol.support_pairs_higher_lower):
        mask = retained[relation_index]
        values = np.asarray(
            [
                metrics["direct_correctness"][relation_index]
                for metrics in metrics_by_amplitude
            ]
        )
        trajectory = []
        for amplitude, subject_values in zip(amplitudes, values, strict=True):
            summary = summarize_subjects(
                np.where(mask, subject_values, np.nan),
                counts,
                interval=interval,
            )
            trajectory.append(
                {
                    "lambda": float(amplitude),
                    "direct_correctness": summary,
                    "status": curve_status(summary, float(amplitude)),
                    "raw_subject_level": json_values(
                        np.where(mask, subject_values, np.nan)
                    ),
                }
            )
        means = np.asarray(
            [row["direct_correctness"]["mean"] for row in trajectory],
            dtype=np.float64,
        )
        statuses = [row["status"] for row in trajectory]
        mean_bracket = mean_sign_change_bracket(amplitudes, means)
        rows.append(
            {
                "relation_index": relation_index,
                "relation": [int(value) for value in relation],
                "relation_label": (
                    f"{protocol.item_labels[relation[0]]}>"
                    f"{protocol.item_labels[relation[1]]}"
                ),
                "retained_subjects": int(np.sum(mask)),
                "trajectory": trajectory,
                "mean_sign_change_bracket": mean_bracket,
                "robust_transition_bracket": robust_transition_bracket(
                    amplitudes, statuses
                ),
                "crossing_regime": crossing_regime(mean_bracket),
                "subject_crossing": subject_crossing_summary(
                    values,
                    mask,
                    amplitudes,
                    counts,
                    interval=interval,
                ),
            }
        )
    return rows


def _curvature_summary(
    curvature_metrics: dict[str, np.ndarray],
    protocol: RankingProtocol,
    retained: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    direct = curvature_metrics["direct_correctness"]
    aggregate = masked_relation_mean(direct, retained)
    per_relation = []
    for relation_index, relation in enumerate(protocol.support_pairs_higher_lower):
        mask = retained[relation_index]
        per_relation.append(
            {
                "relation_index": relation_index,
                "relation_label": (
                    f"{protocol.item_labels[relation[0]]}>"
                    f"{protocol.item_labels[relation[1]]}"
                ),
                "direct_correctness": summarize_subjects(
                    np.where(mask, direct[relation_index], np.nan),
                    counts,
                    interval=interval,
                ),
            }
        )
    primary_index = protocol.support_pairs_higher_lower.index((7, 0))
    primary_mask = retained[primary_index]
    other_sum = np.sum(np.where(retained, direct, 0.0), axis=0) - np.where(
        primary_mask, direct[primary_index], 0.0
    )
    other_count = np.sum(retained, axis=0) - primary_mask.astype(np.int64)
    other_mean = np.divide(
        other_sum,
        other_count,
        out=np.full_like(other_sum, np.nan),
        where=other_count > 0,
    )
    primary_minus_other = np.where(
        primary_mask, direct[primary_index] - other_mean, np.nan
    )
    return {
        "aggregate_direct_correctness": summarize_subjects(
            aggregate, counts, interval=interval
        ),
        "per_relation": per_relation,
        "H_greater_A_minus_other_relations": summarize_subjects(
            primary_minus_other, counts, interval=interval
        ),
        "raw_subject_level": {
            "aggregate_direct_correctness": json_values(aggregate),
            "H_greater_A_minus_other_relations": json_values(primary_minus_other),
        },
    }


def _approximation_summary(
    curve_residuals: np.ndarray,
    jacobian_residual: np.ndarray,
    curvature_residual: np.ndarray,
    direct_edges: np.ndarray,
    correctness_signs: np.ndarray,
    retained: np.ndarray,
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> list[dict]:
    relation_count = len(direct_edges)
    rows = []
    for amplitude_index, amplitude in enumerate(amplitudes):
        approximation = (
            float(amplitude) * jacobian_residual
            + float(amplitude) ** 2 * curvature_residual
        )
        error = curve_residuals[amplitude_index] - approximation
        direct_error = np.stack(
            [
                error[index, :, direct_edges[index]] * correctness_signs[index]
                for index in range(relation_count)
            ]
        )
        subject_absolute = masked_relation_mean(np.abs(direct_error), retained)
        rows.append(
            {
                "lambda": float(amplitude),
                "aggregate_absolute_direct_error": summarize_subjects(
                    subject_absolute, counts, interval=interval
                ),
                "raw_subject_level": json_values(subject_absolute),
            }
        )
    return rows


def _prior_consistency_error(
    prior_seed: dict,
    aggregate_curve: list[dict],
    jacobian_metrics: dict[str, np.ndarray],
    correctness_signs: np.ndarray,
    retained: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> float:
    jacobian_subject = {
        "direct_correctness": masked_relation_mean(
            jacobian_metrics["direct_correctness"], retained
        ),
        "normalized_direct_correctness_rho": normalized_direct_correctness(
            jacobian_metrics["direct_residual"],
            correctness_signs,
            retained,
            tolerance=tolerance,
        ),
        "direct_minus_remote_correctness": masked_relation_mean(
            jacobian_metrics["direct_minus_remote_correctness"], retained
        ),
    }
    jacobian_summary = {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in jacobian_subject.items()
    }
    exact_summary = aggregate_curve[-1]["summary"]
    comparisons = (
        (jacobian_summary, prior_seed["stages"][STAGES[1]]["summary"]),
        (exact_summary, prior_seed["stages"][STAGES[2]]["summary"]),
    )
    maximum = 0.0
    for observed, expected in comparisons:
        for name in observed:
            for key in ("mean",):
                maximum = max(maximum, abs(observed[name][key] - expected[name][key]))
            for key in ("mean", "lower", "upper"):
                maximum = max(
                    maximum,
                    abs(
                        observed[name]["bootstrap"][key]
                        - expected[name]["bootstrap"][key]
                    ),
                )
    return float(maximum)


def _run_seed(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    specification: dict,
    prior_seed: dict,
    counts: np.ndarray,
) -> dict:
    evaluator, _behavior = load_frozen_evaluator(
        registration, pilot_specification, protocol
    )
    intact, loo, effective, retained = replay_terminal_states(evaluator, protocol)
    execution = specification["execution_contract"]
    interval = float(execution["bootstrap_interval"])
    scientific_zero = float(execution["scientific_zero_tolerance"])
    amplitudes = np.asarray(
        specification["amplitude_path_contract"]["fixed_lambda_grid"],
        dtype=np.float64,
    )
    levels, validation = collect_amplitude_fields(
        evaluator,
        protocol,
        geometry,
        intact,
        loo,
        effective,
        retained,
        amplitudes,
        tolerance=float(execution["floating_reproduction_tolerance"]),
    )
    direct_edges, correctness_signs, remote_masks = relation_geometry(
        protocol, geometry
    )
    metrics_by_amplitude = [
        stage_relation_metrics(
            levels["curve_fields"][index],
            levels["curve_residuals"][index],
            geometry,
            direct_edges,
            correctness_signs,
            remote_masks,
        )
        for index in range(len(amplitudes))
    ]
    jacobian_metrics = stage_relation_metrics(
        levels["jacobian_field"],
        levels["jacobian_residual"],
        geometry,
        direct_edges,
        correctness_signs,
        remote_masks,
    )
    curvature_metrics = stage_relation_metrics(
        levels["curvature_field"],
        levels["curvature_residual"],
        geometry,
        direct_edges,
        correctness_signs,
        remote_masks,
    )
    aggregate_curve, _aggregate_subject = _aggregate_curve(
        metrics_by_amplitude,
        correctness_signs,
        retained,
        amplitudes,
        counts,
        interval=interval,
        tolerance=scientific_zero,
    )
    per_relation = _relation_curves(
        metrics_by_amplitude,
        protocol,
        retained,
        amplitudes,
        counts,
        interval=interval,
    )
    primary = next(row for row in per_relation if row["relation"] == [7, 0])
    crossing_relations = [
        row["relation_label"]
        for row in per_relation
        if row["mean_sign_change_bracket"] is not None
    ]
    prior_error = _prior_consistency_error(
        prior_seed,
        aggregate_curve,
        jacobian_metrics,
        correctness_signs,
        retained,
        counts,
        interval=interval,
        tolerance=scientific_zero,
    )
    validation["prior_J_and_H_summary_consistency_max_abs_error"] = prior_error
    zero_names = (
        "lambda_zero_field_max_abs",
        "lambda_zero_residual_max_abs",
        "stable_omitted_oriented_scalar_max_abs",
        "stable_omitted_field_max_abs",
        "stable_omitted_residual_max_abs",
        "stable_omitted_jacobian_max_abs",
        "stable_omitted_curvature_max_abs",
    )
    validation["scientific_zero_tolerance"] = scientific_zero
    validation["zero_controls_pass"] = all(
        validation[name] <= scientific_zero for name in zero_names
    )
    if not validation["zero_controls_pass"] or prior_error > scientific_zero:
        raise RuntimeError(f"amplitude-path integrity failed: {validation}")
    outcome = select_v2_outcome(primary["crossing_regime"], len(crossing_relations))
    return {
        "seed": int(registration["seed"]),
        "subjects": evaluator.config.bs,
        "lambda_grid": [float(value) for value in amplitudes],
        "aggregate_curve": aggregate_curve,
        "per_relation": per_relation,
        "prospective_H_greater_A": primary,
        "curvature": _curvature_summary(
            curvature_metrics,
            protocol,
            retained,
            counts,
            interval=interval,
        ),
        "second_order_approximation": _approximation_summary(
            levels["curve_residuals"],
            levels["jacobian_residual"],
            levels["curvature_residual"],
            direct_edges,
            correctness_signs,
            retained,
            amplitudes,
            counts,
            interval=interval,
        ),
        "diagnosis": {
            "H_greater_A_crossing_regime": primary["crossing_regime"],
            "relations_with_mean_crossing": crossing_relations,
            "mean_crossing_relation_count": len(crossing_relations),
            "v2_selection": outcome,
            "read_only_localization_complete": True,
        },
        "validation": validation,
        "checkpoint": {
            "path": registration["checkpoint_path"],
            "sha256": registration["checkpoint_sha256"],
        },
    }


def _overall_diagnosis(seed_results: dict[str, dict]) -> dict:
    selections = {row["diagnosis"]["v2_selection"] for row in seed_results.values()}
    crossing_sets = [
        set(row["diagnosis"]["relations_with_mean_crossing"])
        for row in seed_results.values()
    ]
    return {
        "v2_selection": (
            next(iter(selections))
            if len(selections) == 1
            else "heterogeneous_or_nonreplicated_crossings_register_online_relation_conditioned_v2"
        ),
        "seed_selections": {
            seed: row["diagnosis"]["v2_selection"] for seed, row in seed_results.items()
        },
        "replicated_mean_crossing_relations": sorted(set.intersection(*crossing_sets)),
        "union_mean_crossing_relations": sorted(set.union(*crossing_sets)),
        "read_only_localization_complete": True,
        "next_step": "freeze an online-computable v2 on one to three new development seeds",
        "formal_interpretation": "hypothesis_generating_only",
    }


def run_operator_amplitude_path(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = resolve_path(str(specification_path))
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    runtime = configure_formal_runtime()
    if not runtime["cuda_available"] or runtime["device"] != "cuda":
        raise RuntimeError("operator amplitude path requires a visible CUDA device")
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    prior_result = load_json(resolve_path(sources["operator_semantics_result"]["path"]))
    execution = specification["execution_contract"]
    counts = bootstrap_counts(
        np.random.default_rng(int(execution["bootstrap_seed"])),
        int(execution["bootstrap_samples"]),
        int(execution["subjects"]),
    )
    seed_results = {
        str(registration["seed"]): _run_seed(
            registration,
            pilot_specification,
            protocol,
            geometry,
            specification,
            prior_result["seed_results"][str(registration["seed"])],
            counts,
        )
        for registration in sources["pilot_artifacts"]
    }
    declared = {str(seed) for seed in execution["seeds"]}
    if set(seed_results) != declared:
        raise RuntimeError("not every declared development seed was reported")
    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "specification": {
            "path": legacy_identifier(specification_path),
            "sha256": file_sha256(specification_path),
        },
        "implementation": {
            "path": "fsrl/operator_amplitude_path.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "execution_runtime": runtime,
        "artifact_validation": validation,
        "contracts": {
            "amplitude_path": specification["amplitude_path_contract"],
            "aggregate_crossing": specification["aggregate_crossing_contract"],
            "subject_crossing": specification["subject_crossing_contract"],
            "curvature": specification["curvature_contract"],
            "v2_selection": specification["outcome_contingent_v2_selection"],
            "v2_entry": specification["v2_entry_contract"],
        },
        "seed_results": seed_results,
        "overall_diagnosis": _overall_diagnosis(seed_results),
        "formal_seed_access": False,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Trace frozen operator semantics over a fixed amplitude grid."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_operator_amplitude_path(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
