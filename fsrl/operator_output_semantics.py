"""Localize correctness semantics in frozen operator output maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .assembly_diagnostics import file_sha256, load_json, resolve_path
from .assembly_trajectory import (
    CompleteGraphGeometry,
    bootstrap_counts,
    build_complete_graph_geometry,
    load_frozen_evaluator,
    ordered_query_schedule,
    summarize_subjects,
)
from .config import DEVICE, NUMRESPONSESTEP
from .formal_runtime import configure_formal_runtime
from .hidden_residual_audit import validate_registered_sources
from .ranking_protocol import RankingProtocol, load_ranking_protocol
from .state_query_operator_binding import _replay_terminal_states
from .study_registry import legacy_identifier, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/operator_output_semantics_v1.json")
DEFAULT_OUTPUT_PATH = resolve_record("results/operator_output_semantics_v1.json")
STAGES = ("A_operator_value", "J_linearized_expression", "H_exact_expression")


def _json_values(values: np.ndarray) -> list:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _masked_relation_mean(values: np.ndarray, retained: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.shape != retained.shape:
        raise ValueError("relation values and retained mask do not match")
    numerator = np.sum(np.where(retained, rows, 0.0), axis=0)
    denominator = np.sum(retained, axis=0)
    if np.any(denominator == 0):
        raise RuntimeError("every subject must retain at least one relation")
    return numerator / denominator


def normalized_direct_correctness(
    direct_residual: np.ndarray,
    correctness_signs: np.ndarray,
    retained: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Cosine of each subject's retained direct field with correct signs."""

    values = np.asarray(direct_residual, dtype=np.float64)
    signs = np.asarray(correctness_signs, dtype=np.float64)
    if values.shape != retained.shape or signs.shape != (values.shape[0],):
        raise ValueError("normalized correctness inputs do not match")
    selected = np.where(retained, values, 0.0)
    numerator = np.sum(selected * signs[:, None], axis=0)
    value_norm = np.linalg.norm(selected, axis=0)
    sign_norm = np.sqrt(np.sum(retained, axis=0).astype(np.float64))
    denominator = value_norm * sign_norm
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=denominator > tolerance,
    )


def hodge_components(
    fields: np.ndarray, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[-1] != len(geometry.pairs):
        raise ValueError("scalar field does not match complete-graph edge order")
    gradient = np.einsum("ef,...f->...e", geometry.projection, values)
    return gradient, values - gradient


def _relation_geometry(
    protocol: RankingProtocol, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    edge_lookup = {pair: index for index, pair in enumerate(geometry.pairs)}
    direct_edges = []
    remote_masks = []
    for relation in protocol.support_pairs_higher_lower:
        direct_edges.append(edge_lookup[tuple(sorted(relation))])
        endpoints = set(relation)
        remote_masks.append(
            np.asarray(
                [not endpoints.intersection(pair) for pair in geometry.pairs],
                dtype=bool,
            )
        )
    direct = np.asarray(direct_edges, dtype=np.int64)
    if any(int(np.sum(mask)) != 15 for mask in remote_masks):
        raise RuntimeError("each frozen relation must have 15 disjoint remote edges")
    return direct, geometry.true_sign[direct], tuple(remote_masks)


def _stack_step(rows, pair: tuple[int, int], step: int) -> np.ndarray:
    return np.stack([subject[pair][step] for subject in rows])


def collect_nested_fields(
    evaluator,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    intact: torch.Tensor,
    loo: torch.Tensor,
    effective: torch.Tensor,
    retained: np.ndarray,
    *,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Collect output-projected A, J_b A, and exact H over all query edges."""

    relation_count = len(protocol.support_pairs_higher_lower)
    subjects = evaluator.config.bs
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

    scalar_maps = {
        stage: np.empty((relation_count, subjects, edge_count, 2), dtype=np.float64)
        for stage in STAGES
    }
    validation = {
        "manual_h0_max_abs_error": 0.0,
        "loo_h0_invariance_max_abs_error": 0.0,
        "operator_preactivation_reconstruction_max_abs_error": 0.0,
        "nonlinear_hidden_reconstruction_max_abs_error": 0.0,
        "exact_hidden_to_actual_logit_influence_max_abs_error": 0.0,
        "bilinear_operator_identity_max_abs_error": 0.0,
    }
    output = evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0]
    output = output.detach()
    retained_device = torch.from_numpy(retained).to(device=DEVICE)
    omitted_scalar_max = {stage: 0.0 for stage in STAGES}

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
                baseline_hidden = evaluator.net.activ(baseline)
                jacobian_action = (1.0 - baseline_hidden.square()) * action
                hidden_effect = evaluator.net.activ(baseline + action) - baseline_hidden
                maps = {
                    "A_operator_value": action[..., 0],
                    "J_linearized_expression": jacobian_action[..., 0],
                    "H_exact_expression": hidden_effect[..., 0],
                }
                for stage, values in maps.items():
                    scalars = torch.einsum("qsh,h->qs", values, output)
                    scalar_maps[stage][:, :, edge, orientation] = (
                        scalars.detach().cpu().numpy().astype(np.float64)
                    )
                    if torch.any(~retained_device):
                        omitted_scalar_max[stage] = max(
                            omitted_scalar_max[stage],
                            float(
                                torch.max(torch.abs(scalars[~retained_device])).item()
                            ),
                        )

                intact_preactivation = input_drive[0] + torch.matmul(
                    evaluator.net.w + evaluator.net.alpha.detach() * intact,
                    h0.view(subjects, evaluator.config.hs, 1),
                )
                validation["operator_preactivation_reconstruction_max_abs_error"] = max(
                    validation["operator_preactivation_reconstruction_max_abs_error"],
                    float(
                        torch.max(
                            torch.abs(baseline + action - intact_preactivation[None])
                        ).item()
                    ),
                )
                actual_h0 = torch.from_numpy(_stack_step(intact_hidden, pair, 0)).to(
                    DEVICE
                )
                validation["manual_h0_max_abs_error"] = max(
                    validation["manual_h0_max_abs_error"],
                    float(torch.max(torch.abs(actual_h0 - h0)).item()),
                )
                direct_scalar = torch.einsum("qsh,h->qs", action[..., 0], output)
                covector = torch.einsum("qshk,h->qsk", effective, output)
                bilinear_scalar = torch.einsum("qsh,sh->qs", covector, h0)
                validation["bilinear_operator_identity_max_abs_error"] = max(
                    validation["bilinear_operator_identity_max_abs_error"],
                    float(torch.max(torch.abs(direct_scalar - bilinear_scalar)).item()),
                )

                actual_intact_h1 = torch.from_numpy(
                    _stack_step(intact_hidden, pair, NUMRESPONSESTEP)
                ).to(DEVICE)
                actual_intact_logit = torch.from_numpy(
                    _stack_step(intact_logits, pair, NUMRESPONSESTEP)
                ).to(DEVICE)
                for relation_index, (loo_hidden, loo_logits) in enumerate(
                    loo_trajectories
                ):
                    actual_loo_h0 = torch.from_numpy(
                        _stack_step(loo_hidden, pair, 0)
                    ).to(DEVICE)
                    actual_loo_h1 = torch.from_numpy(
                        _stack_step(loo_hidden, pair, NUMRESPONSESTEP)
                    ).to(DEVICE)
                    actual_loo_logit = torch.from_numpy(
                        _stack_step(loo_logits, pair, NUMRESPONSESTEP)
                    ).to(DEVICE)
                    validation["loo_h0_invariance_max_abs_error"] = max(
                        validation["loo_h0_invariance_max_abs_error"],
                        float(torch.max(torch.abs(actual_h0 - actual_loo_h0)).item()),
                    )
                    validation["nonlinear_hidden_reconstruction_max_abs_error"] = max(
                        validation["nonlinear_hidden_reconstruction_max_abs_error"],
                        float(
                            torch.max(
                                torch.abs(
                                    hidden_effect[relation_index, :, :, 0]
                                    - (actual_intact_h1 - actual_loo_h1)
                                )
                            ).item()
                        ),
                    )
                    projected = torch.einsum(
                        "sh,h->s", hidden_effect[relation_index, :, :, 0], output
                    )
                    validation[
                        "exact_hidden_to_actual_logit_influence_max_abs_error"
                    ] = max(
                        validation[
                            "exact_hidden_to_actual_logit_influence_max_abs_error"
                        ],
                        float(
                            torch.max(
                                torch.abs(
                                    projected - (actual_intact_logit - actual_loo_logit)
                                )
                            ).item()
                        ),
                    )

    fields = {
        stage: 0.5 * (values[..., 0] - values[..., 1])
        for stage, values in scalar_maps.items()
    }
    residuals = {}
    hodge_error = 0.0
    stable_omitted_residual_max = {}
    omitted_field_max = {}
    omitted_selector = np.broadcast_to(
        (~retained)[..., None], (relation_count, subjects, edge_count)
    )
    for stage, field in fields.items():
        gradient, residual = hodge_components(field, geometry)
        residuals[stage] = residual
        hodge_error = max(
            hodge_error, float(np.max(np.abs(field - gradient - residual)))
        )
        omitted_field_max[stage] = float(np.max(np.abs(field[omitted_selector])))
        stable_omitted_residual_max[stage] = float(
            np.max(np.abs(residual[omitted_selector]))
        )
    validation.update(
        {
            "hodge_reconstruction_max_abs_error": hodge_error,
            "stable_omitted_oriented_scalar_max_abs": omitted_scalar_max,
            "stable_omitted_field_max_abs": omitted_field_max,
            "stable_omitted_residual_max_abs": stable_omitted_residual_max,
            "floating_reproduction_tolerance": tolerance,
        }
    )
    error_names = (
        "manual_h0_max_abs_error",
        "loo_h0_invariance_max_abs_error",
        "operator_preactivation_reconstruction_max_abs_error",
        "nonlinear_hidden_reconstruction_max_abs_error",
        "exact_hidden_to_actual_logit_influence_max_abs_error",
        "bilinear_operator_identity_max_abs_error",
        "hodge_reconstruction_max_abs_error",
    )
    if any(validation[name] > tolerance for name in error_names):
        raise RuntimeError(f"nested-map reconstruction failed: {validation}")
    return {
        "fields": fields,
        "residuals": residuals,
    }, validation


def stage_relation_metrics(
    field: np.ndarray,
    residual: np.ndarray,
    geometry: CompleteGraphGeometry,
    direct_edges: np.ndarray,
    correctness_signs: np.ndarray,
    remote_masks: tuple[np.ndarray, ...],
) -> dict[str, np.ndarray]:
    relation_count = field.shape[0]
    direct = np.stack(
        [residual[index, :, direct_edges[index]] for index in range(relation_count)]
    )
    remote_correctness = np.stack(
        [
            np.mean(residual[index][:, mask] * geometry.true_sign[mask], axis=1)
            for index, mask in enumerate(remote_masks)
        ]
    )
    remote_absolute = np.stack(
        [
            np.mean(np.abs(residual[index][:, mask]), axis=1)
            for index, mask in enumerate(remote_masks)
        ]
    )
    gradient, _residual = hodge_components(field, geometry)
    gradient_energy = np.sum(gradient * gradient, axis=2)
    residual_energy = np.sum(residual * residual, axis=2)
    total_energy = gradient_energy + residual_energy
    gradient_fraction = np.divide(
        gradient_energy,
        total_energy,
        out=np.full_like(total_energy, np.nan),
        where=total_energy > 0.0,
    )
    residual_fraction = np.divide(
        residual_energy,
        total_energy,
        out=np.full_like(total_energy, np.nan),
        where=total_energy > 0.0,
    )
    direct_correctness = direct * correctness_signs[:, None]
    direct_absolute = np.abs(direct)
    return {
        "direct_residual": direct,
        "direct_correctness": direct_correctness,
        "remote_correctness": remote_correctness,
        "direct_minus_remote_correctness": direct_correctness - remote_correctness,
        "direct_absolute_residual": direct_absolute,
        "remote_absolute_residual": remote_absolute,
        "direct_minus_remote_absolute_residual": direct_absolute - remote_absolute,
        "gradient_energy_fraction": gradient_fraction,
        "residual_energy_fraction": residual_fraction,
    }


def _stage_summary(
    metrics: dict[str, np.ndarray],
    correctness_signs: np.ndarray,
    retained: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    subject = {
        name: _masked_relation_mean(values, retained)
        for name, values in metrics.items()
        if name != "direct_residual"
    }
    subject["normalized_direct_correctness_rho"] = normalized_direct_correctness(
        metrics["direct_residual"],
        correctness_signs,
        retained,
        tolerance=tolerance,
    )
    return {
        "summary": {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in subject.items()
        },
        "raw_subject_level": {
            name: _json_values(values) for name, values in subject.items()
        },
    }, subject


def _relation_summary(
    metrics_by_stage: dict[str, dict[str, np.ndarray]],
    protocol: RankingProtocol,
    retained: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> list[dict]:
    rows = []
    for relation_index, relation in enumerate(protocol.support_pairs_higher_lower):
        mask = retained[relation_index]
        stage_rows = {}
        for stage, metrics in metrics_by_stage.items():
            stage_rows[stage] = {
                name: summarize_subjects(
                    np.where(mask, values[relation_index], np.nan),
                    counts,
                    interval=interval,
                )
                for name, values in metrics.items()
                if name != "direct_residual"
            }
        stage_rows["transitions"] = {
            "J_minus_A_direct_correctness": summarize_subjects(
                np.where(
                    mask,
                    metrics_by_stage[STAGES[1]]["direct_correctness"][relation_index]
                    - metrics_by_stage[STAGES[0]]["direct_correctness"][relation_index],
                    np.nan,
                ),
                counts,
                interval=interval,
            ),
            "H_minus_J_direct_correctness": summarize_subjects(
                np.where(
                    mask,
                    metrics_by_stage[STAGES[2]]["direct_correctness"][relation_index]
                    - metrics_by_stage[STAGES[1]]["direct_correctness"][relation_index],
                    np.nan,
                ),
                counts,
                interval=interval,
            ),
        }
        rows.append(
            {
                "relation_index": relation_index,
                "relation": [int(value) for value in relation],
                "relation_label": (
                    f"{protocol.item_labels[relation[0]]}>"
                    f"{protocol.item_labels[relation[1]]}"
                ),
                "retained_subjects": int(np.sum(mask)),
                "stable_omitted_subjects": int(np.sum(~mask)),
                "stages": stage_rows,
            }
        )
    return rows


def _bounds(summary: dict) -> tuple[float, float]:
    bootstrap = summary["bootstrap"]
    return float(bootstrap["lower"]), float(bootstrap["upper"])


def classify_stage(stage: dict) -> str:
    direct_lower, direct_upper = _bounds(stage["direct_correctness"])
    rho_lower, rho_upper = _bounds(stage["normalized_direct_correctness_rho"])
    if direct_lower > 0.0 and rho_lower > 0.0:
        return "correctness_aligned"
    if direct_upper < 0.0 and rho_upper < 0.0:
        return "correctness_opposed"
    return "unresolved"


def classify_transition(summary: dict) -> str:
    lower, upper = _bounds(summary)
    if upper < 0.0:
        return "degradation"
    if lower > 0.0:
        return "improvement"
    return "unresolved"


def classify_relation_direct(summary: dict) -> str:
    lower, upper = _bounds(summary)
    if lower > 0.0:
        return "correctness_aligned"
    if upper < 0.0:
        return "correctness_opposed"
    return "unresolved"


def decide_outcome(
    stage_status: dict[str, str],
    rho_transitions: dict[str, str],
    h_greater_a_status: dict[str, str],
) -> str:
    if any(
        value == "correctness_opposed" for value in h_greater_a_status.values()
    ) and any(value == "correctness_aligned" for value in stage_status.values()):
        return "aggregate_aligned_but_H_greater_A_opposed"
    if stage_status[STAGES[0]] == "correctness_aligned":
        if stage_status[STAGES[1]] == "correctness_opposed" or (
            stage_status[STAGES[1]] != "correctness_aligned"
            and rho_transitions["J_minus_A"] == "degradation"
        ):
            return "operator_value_correct_operating_point_corruption"
        if stage_status[STAGES[1]] == "correctness_aligned" and (
            stage_status[STAGES[2]] == "correctness_opposed"
            or (
                stage_status[STAGES[2]] != "correctness_aligned"
                and rho_transitions["H_minus_J"] == "degradation"
            )
        ):
            return "linearized_value_correct_finite_amplitude_corruption"
        if all(value == "correctness_aligned" for value in stage_status.values()):
            return "correct_semantics_reach_policy_amplitude_or_combination_unresolved"
    if all(value != "correctness_aligned" for value in stage_status.values()):
        return "operator_value_semantics_not_correctness_aligned"
    return "mixed_stage_semantics_requires_narrower_discriminator"


def _run_seed(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    specification: dict,
    counts: np.ndarray,
) -> dict:
    evaluator, _behavior = load_frozen_evaluator(
        registration, pilot_specification, protocol
    )
    intact, loo, effective, retained = _replay_terminal_states(evaluator, protocol)
    execution = specification["execution_contract"]
    interval = float(execution["bootstrap_interval"])
    tolerance = float(execution["scientific_zero_tolerance"])
    levels, validation = collect_nested_fields(
        evaluator,
        protocol,
        geometry,
        intact,
        loo,
        effective,
        retained,
        tolerance=float(execution["floating_reproduction_tolerance"]),
    )
    direct_edges, correctness_signs, remote_masks = _relation_geometry(
        protocol, geometry
    )
    metrics_by_stage = {
        stage: stage_relation_metrics(
            levels["fields"][stage],
            levels["residuals"][stage],
            geometry,
            direct_edges,
            correctness_signs,
            remote_masks,
        )
        for stage in STAGES
    }
    stage_results = {}
    subject_by_stage = {}
    for stage, metrics in metrics_by_stage.items():
        stage_results[stage], subject_by_stage[stage] = _stage_summary(
            metrics,
            correctness_signs,
            retained,
            counts,
            interval=interval,
            tolerance=tolerance,
        )

    transitions_subject = {
        "J_minus_A_direct_correctness": (
            subject_by_stage[STAGES[1]]["direct_correctness"]
            - subject_by_stage[STAGES[0]]["direct_correctness"]
        ),
        "H_minus_J_direct_correctness": (
            subject_by_stage[STAGES[2]]["direct_correctness"]
            - subject_by_stage[STAGES[1]]["direct_correctness"]
        ),
        "J_minus_A_rho": (
            subject_by_stage[STAGES[1]]["normalized_direct_correctness_rho"]
            - subject_by_stage[STAGES[0]]["normalized_direct_correctness_rho"]
        ),
        "H_minus_J_rho": (
            subject_by_stage[STAGES[2]]["normalized_direct_correctness_rho"]
            - subject_by_stage[STAGES[1]]["normalized_direct_correctness_rho"]
        ),
    }
    transitions = {
        "summary": {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in transitions_subject.items()
        },
        "raw_subject_level": {
            name: _json_values(values) for name, values in transitions_subject.items()
        },
    }
    per_relation = _relation_summary(
        metrics_by_stage,
        protocol,
        retained,
        counts,
        interval=interval,
    )
    h_contract = specification["prospective_H_greater_A_contract"]
    h_relation = tuple(int(value) for value in h_contract["relation_higher_lower"])
    h_index = protocol.support_pairs_higher_lower.index(h_relation)
    h_row = per_relation[h_index]
    stage_status = {
        stage: classify_stage(result["summary"])
        for stage, result in stage_results.items()
    }
    rho_transitions = {
        "J_minus_A": classify_transition(transitions["summary"]["J_minus_A_rho"]),
        "H_minus_J": classify_transition(transitions["summary"]["H_minus_J_rho"]),
    }
    raw_transitions = {
        "J_minus_A": classify_transition(
            transitions["summary"]["J_minus_A_direct_correctness"]
        ),
        "H_minus_J": classify_transition(
            transitions["summary"]["H_minus_J_direct_correctness"]
        ),
    }
    attenuation_only = {
        "J_minus_A": bool(
            raw_transitions["J_minus_A"] == "degradation"
            and rho_transitions["J_minus_A"] == "unresolved"
            and stage_status[STAGES[1]] == "correctness_aligned"
        ),
        "H_minus_J": bool(
            raw_transitions["H_minus_J"] == "degradation"
            and rho_transitions["H_minus_J"] == "unresolved"
            and stage_status[STAGES[2]] == "correctness_aligned"
        ),
    }
    h_status = {
        stage: classify_relation_direct(h_row["stages"][stage]["direct_correctness"])
        for stage in STAGES
    }
    locality = {
        stage: bool(
            result["summary"]["direct_minus_remote_correctness"]["bootstrap"]["lower"]
            > 0.0
            and result["summary"]["direct_minus_remote_absolute_residual"]["bootstrap"][
                "lower"
            ]
            > 0.0
        )
        for stage, result in stage_results.items()
    }
    scientific_zero = float(execution["scientific_zero_tolerance"])
    stable_omission_pass = all(
        value <= scientific_zero
        for name in (
            "stable_omitted_oriented_scalar_max_abs",
            "stable_omitted_field_max_abs",
            "stable_omitted_residual_max_abs",
        )
        for value in validation[name].values()
    )
    if not stable_omission_pass:
        raise RuntimeError("stable-omission semantic control failed")
    return {
        "seed": int(registration["seed"]),
        "subjects": evaluator.config.bs,
        "relations": [
            [int(first), int(second)]
            for first, second in protocol.support_pairs_higher_lower
        ],
        "query_edges": [[int(first), int(second)] for first, second in geometry.pairs],
        "stages": stage_results,
        "transitions": transitions,
        "per_relation": per_relation,
        "prospective_H_greater_A": h_row,
        "diagnosis": {
            "stage_status": stage_status,
            "raw_transition_status": raw_transitions,
            "rho_transition_status": rho_transitions,
            "raw_attenuation_without_resolved_rho_degradation": attenuation_only,
            "local_specificity": locality,
            "H_greater_A_stage_status": h_status,
            "stable_omission_pass": stable_omission_pass,
            "outcome": decide_outcome(stage_status, rho_transitions, h_status),
        },
        "validation": {
            **validation,
            "scientific_zero_tolerance": scientific_zero,
            "relations": len(protocol.support_pairs_higher_lower),
            "query_edges": len(geometry.pairs),
            "orientations": 2,
            "remote_edges_per_relation": 15,
        },
        "checkpoint": {
            "path": registration["checkpoint_path"],
            "sha256": registration["checkpoint_sha256"],
        },
    }


def _overall_diagnosis(seed_results: dict[str, dict]) -> dict:
    outcomes = {row["diagnosis"]["outcome"] for row in seed_results.values()}
    stage_status = {
        stage: sorted(
            {row["diagnosis"]["stage_status"][stage] for row in seed_results.values()}
        )
        for stage in STAGES
    }
    h_status = {
        stage: sorted(
            {
                row["diagnosis"]["H_greater_A_stage_status"][stage]
                for row in seed_results.values()
            }
        )
        for stage in STAGES
    }
    return {
        "outcome": (
            next(iter(outcomes))
            if len(outcomes) == 1
            else "mixed_across_development_seeds"
        ),
        "seed_outcomes": {
            seed: row["diagnosis"]["outcome"] for seed, row in seed_results.items()
        },
        "stage_status_across_seeds": stage_status,
        "H_greater_A_stage_status_across_seeds": h_status,
        "formal_interpretation": "hypothesis_generating_only",
    }


def run_operator_output_semantics(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = resolve_path(str(specification_path))
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    runtime = configure_formal_runtime()
    if not runtime["cuda_available"] or DEVICE != "cuda":
        raise RuntimeError("operator-output semantics requires a visible CUDA device")
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    execution = specification["execution_contract"]
    counts = bootstrap_counts(
        np.random.default_rng(int(execution["bootstrap_seed"])),
        int(execution["bootstrap_samples"]),
        int(pilot_specification["evaluation"]["batch_size"]),
    )
    seed_results = {
        str(registration["seed"]): _run_seed(
            registration,
            pilot_specification,
            protocol,
            geometry,
            specification,
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
            "path": "fsrl/operator_output_semantics.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "execution_runtime": runtime,
        "artifact_validation": validation,
        "contracts": {
            "nested_maps": specification["nested_map_contract"],
            "hodge_semantics": specification["hodge_semantics_contract"],
            "normalized_semantics": specification["normalized_semantics_contract"],
            "H_greater_A": specification["prospective_H_greater_A_contract"],
            "directional_rules": specification["directional_rules_within_seed"],
            "decision_tree": specification["outcome_contingent_decision_tree"],
        },
        "seed_results": seed_results,
        "overall_diagnosis": _overall_diagnosis(seed_results),
        "formal_seed_access": False,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Localize frozen operator-output correctness semantics."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_operator_output_semantics(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
