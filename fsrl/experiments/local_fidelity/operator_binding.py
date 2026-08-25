"""Test relation identity in the frozen state-by-query operator factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.statistics import bootstrap_counts, json_values, summarize_subjects
from fsrl.core.config import DEVICE
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    retained_relation_mask,
)
from fsrl.experiments.assembly.trajectory import load_frozen_evaluator
from fsrl.experiments.local_fidelity.hidden_residual import validate_registered_sources
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import legacy_identifier, resolve_record
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/state_query_operator_binding_v1.json"
)
DEFAULT_OUTPUT_PATH = resolve_record("results/state_query_operator_binding_v1.json")


def _unit_rows(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    return (
        np.divide(
            rows,
            norms[:, None],
            out=np.zeros_like(rows),
            where=norms[:, None] > tolerance,
        ),
        norms,
    )


def _masked_mean(
    values: np.ndarray, valid: np.ndarray, axis: int | tuple[int, ...]
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    selector = np.asarray(valid, dtype=bool) & np.isfinite(rows)
    numerator = np.sum(np.where(selector, rows, 0.0), axis=axis)
    denominator = np.sum(selector, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def contextual_identity_metrics(
    traces: np.ndarray,
    valid: np.ndarray,
    *,
    subject_folds: int,
    tolerance: float,
    cross_context: bool,
) -> dict[str, np.ndarray]:
    """Match identity prototypes at fixed or held-out contexts."""

    values = np.asarray(traces)
    identities, subjects, contexts = values.shape[:3]
    flat = values.reshape(identities, subjects, contexts, -1).astype(
        np.float64, copy=False
    )
    if valid.shape != (identities, subjects, contexts):
        raise ValueError("valid mask does not match contextual traces")
    if cross_context and contexts < 2:
        raise ValueError("cross-context identity requires at least two contexts")

    subject_fold = np.arange(subjects) % subject_folds
    similarities = np.full((identities, subjects, contexts, identities), np.nan)
    for fold in range(subject_folds):
        test_subjects = np.flatnonzero(subject_fold == fold)
        train_subjects = subject_fold != fold
        for test_context in range(contexts):
            prototypes = np.empty((identities, flat.shape[-1]), dtype=np.float64)
            for identity in range(identities):
                if cross_context:
                    context_mask = np.arange(contexts) != test_context
                    selected = flat[identity, train_subjects][:, context_mask]
                    selected_valid = valid[identity, train_subjects][:, context_mask]
                    prototype = np.mean(selected[selected_valid], axis=0)
                else:
                    selected_valid = valid[identity, train_subjects, test_context]
                    prototype = np.mean(
                        flat[identity, train_subjects, test_context][selected_valid],
                        axis=0,
                    )
                norm = np.linalg.norm(prototype)
                if not np.isfinite(norm) or norm <= tolerance:
                    raise RuntimeError("contextual identity prototype is undefined")
                prototypes[identity] = prototype / norm

            for identity in range(identities):
                rows, norms = _unit_rows(
                    flat[identity, test_subjects, test_context], tolerance
                )
                row_valid = valid[identity, test_subjects, test_context] & (
                    norms > tolerance
                )
                scores = rows @ prototypes.T
                scores[~row_valid] = np.nan
                similarities[identity, test_subjects, test_context] = scores

    own = np.empty((identities, subjects, contexts), dtype=np.float64)
    for identity in range(identities):
        own[identity] = similarities[identity, :, :, identity]
    other = (np.nansum(similarities, axis=3) - own) / float(identities - 1)
    identification = np.full_like(own, np.nan)
    for identity in range(identities):
        rows = similarities[identity]
        row_valid = np.isfinite(own[identity])
        identification[identity, row_valid] = (
            np.argmax(rows[row_valid], axis=1) == identity
        ).astype(np.float64)
    return {
        "own_prototype_cosine": own,
        "other_prototype_mean_cosine": other,
        "own_minus_other_selectivity": own - other,
        "eight_way_identification_accuracy": identification,
        "trace_l2_norm": np.linalg.norm(flat, axis=3),
    }


def summarize_contextual_identity(
    traces: np.ndarray,
    valid: np.ndarray,
    counts: np.ndarray,
    *,
    subject_folds: int,
    interval: float,
    chance: float,
    tolerance: float,
    cross_context: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    metrics = contextual_identity_metrics(
        traces,
        valid,
        subject_folds=subject_folds,
        tolerance=tolerance,
        cross_context=cross_context,
    )
    subject_metrics = {
        name: _masked_mean(values, valid, axis=(0, 2))
        for name, values in metrics.items()
    }
    summaries = {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in subject_metrics.items()
    }
    omitted_selector = np.broadcast_to(
        (~valid).reshape(*valid.shape, *([1] * (traces.ndim - 3))), traces.shape
    )
    omitted_max = (
        float(np.max(np.abs(traces[omitted_selector]))) if np.any(~valid) else 0.0
    )
    present = (
        summaries["own_minus_other_selectivity"]["bootstrap"]["lower"] > 0.0
        and summaries["eight_way_identification_accuracy"]["bootstrap"]["lower"]
        > chance
        and omitted_max <= tolerance
    )
    per_identity = []
    for identity in range(traces.shape[0]):
        identity_metrics = {
            name: _masked_mean(values[identity], valid[identity], axis=1)
            for name, values in metrics.items()
        }
        per_identity.append(
            {
                "identity_index": identity,
                "summary": {
                    name: summarize_subjects(values, counts, interval=interval)
                    for name, values in identity_metrics.items()
                },
            }
        )
    per_context = []
    for context in range(traces.shape[2]):
        context_metrics = {
            name: _masked_mean(values[:, :, context], valid[:, :, context], axis=0)
            for name, values in metrics.items()
        }
        per_context.append(
            {
                "context_index": context,
                "summary": {
                    name: summarize_subjects(values, counts, interval=interval)
                    for name, values in context_metrics.items()
                },
            }
        )
    return {
        "summary": summaries,
        "per_identity": per_identity,
        "per_context": per_context,
        "omitted_max_abs": omitted_max,
        "presence_rule_passed": bool(present),
        "raw_subject_level": {
            name: json_values(values) for name, values in subject_metrics.items()
        },
    }, metrics


def overlap_classes(relations: tuple[tuple[int, int], ...]) -> np.ndarray:
    overlap = np.asarray(
        [
            [
                len(set(state_relation) & set(query_relation))
                for query_relation in relations
            ]
            for state_relation in relations
        ],
        dtype=np.int64,
    )
    expected = np.asarray([1, 2, 5])
    observed = np.asarray(
        [
            [np.sum(row == overlap_count) for overlap_count in (2, 1, 0)]
            for row in overlap
        ]
    )
    if not np.all(observed == expected):
        raise RuntimeError("support relations do not have the frozen overlap structure")
    return overlap


def structured_contrasts(
    values: np.ndarray,
    retained: np.ndarray,
    overlap: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    rows = np.asarray(values, dtype=np.float64)
    relations, subjects, contexts = rows.shape
    if retained.shape != (relations, subjects):
        raise ValueError("retained mask does not match structured values")
    if overlap.shape != (relations, contexts):
        raise ValueError("overlap matrix does not match structured values")
    matched = np.stack([rows[index, :, index] for index in range(relations)])
    shared = np.stack(
        [
            _masked_mean(
                rows[index][:, overlap[index] == 1],
                np.isfinite(rows[index][:, overlap[index] == 1]),
                axis=1,
            )
            for index in range(relations)
        ]
    )
    disjoint = np.stack(
        [
            _masked_mean(
                rows[index][:, overlap[index] == 0],
                np.isfinite(rows[index][:, overlap[index] == 0]),
                axis=1,
            )
            for index in range(relations)
        ]
    )
    relation_values = {
        "matched": matched,
        "shared_endpoint_mismatch": shared,
        "disjoint_mismatch": disjoint,
        "matched_minus_shared_endpoint": matched - shared,
        "matched_minus_disjoint": matched - disjoint,
    }
    subject_values = {
        name: _masked_mean(value, retained, axis=0)
        for name, value in relation_values.items()
    }
    return {
        "summary": {
            name: summarize_subjects(value, counts, interval=interval)
            for name, value in subject_values.items()
        },
        "per_relation": [
            {
                "relation_index": relation,
                "retained_subjects": int(np.sum(retained[relation])),
                "summary": {
                    name: summarize_subjects(
                        np.where(retained[relation], value[relation], np.nan),
                        counts,
                        interval=interval,
                    )
                    for name, value in relation_values.items()
                },
            }
            for relation in range(relations)
        ],
        "raw_subject_level": {
            name: json_values(value) for name, value in subject_values.items()
        },
    }


def replay_terminal_states(
    evaluator, protocol: RankingProtocol
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    intact = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    loo_states = []
    for relation in protocol.support_pairs_higher_lower:
        state = evaluator.initialize_fast_weights()
        for trial_index in range(protocol.support_trials):
            state = evaluator.advance_support_trial(
                state,
                trial_index,
                zero_relations=frozenset((relation,)),
            )
        loo_states.append(state)
    loo = torch.stack(loo_states)
    effective = evaluator.net.alpha.detach() * (intact[None] - loo)
    return (
        intact,
        loo,
        effective,
        retained_relation_mask(evaluator, protocol.support_pairs_higher_lower),
    )


def _oriented_query_pairs(
    relations: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return tuple(((higher, lower), (lower, higher)) for higher, lower in relations)


def _stack_hidden_step(rows, pair: tuple[int, int], step: int) -> np.ndarray:
    return np.stack([subject[pair][step] for subject in rows])


def collect_factorial(
    evaluator,
    protocol: RankingProtocol,
    intact: torch.Tensor,
    loo: torch.Tensor,
    effective: torch.Tensor,
    retained: np.ndarray,
    *,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], dict]:
    relations = protocol.support_pairs_higher_lower
    oriented = _oriented_query_pairs(relations)
    relation_count = len(relations)
    subjects = evaluator.config.bs
    hidden_size = evaluator.config.hs
    schedules = tuple(
        tuple(pair for relation_pairs in oriented for pair in relation_pairs)
        for _ in range(subjects)
    )
    intact_rows, _ = evaluator.readout_hidden_and_logit_trajectories(intact, schedules)
    loo_rows = [
        evaluator.readout_hidden_and_logit_trajectories(loo[index], schedules)[0]
        for index in range(relation_count)
    ]

    action = torch.empty(
        relation_count,
        subjects,
        relation_count,
        2,
        hidden_size,
        device=DEVICE,
    )
    hidden_effect = torch.empty_like(action)
    h0_values = torch.empty(subjects, relation_count, 2, hidden_size, device=DEVICE)
    validation = {
        "manual_h0_max_abs_error": 0.0,
        "loo_h0_invariance_max_abs_error": 0.0,
        "operator_preactivation_reconstruction_max_abs_error": 0.0,
        "nonlinear_hidden_reconstruction_max_abs_error": 0.0,
    }

    with torch.no_grad():
        for query_index, relation_pairs in enumerate(oriented):
            for orientation, pair in enumerate(relation_pairs):
                left = np.full(subjects, pair[0], dtype=np.int64)
                right = np.full(subjects, pair[1], dtype=np.int64)
                signed = np.zeros(subjects, dtype=np.float32)
                step0_inputs = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                step1_inputs = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=1,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                h0 = evaluator.net.activ(evaluator.net.i2h(step0_inputs))
                h0_values[:, query_index, orientation] = h0
                actual_h0 = torch.from_numpy(
                    _stack_hidden_step(intact_rows, pair, 0)
                ).to(DEVICE)
                validation["manual_h0_max_abs_error"] = max(
                    validation["manual_h0_max_abs_error"],
                    float(torch.max(torch.abs(h0 - actual_h0)).item()),
                )
                input_drive = evaluator.net.i2h(step1_inputs).view(
                    subjects, hidden_size, 1
                )
                intact_preactivation = input_drive + torch.matmul(
                    evaluator.net.w + evaluator.net.alpha.detach() * intact,
                    h0.view(subjects, hidden_size, 1),
                )
                actual_intact_h1 = torch.from_numpy(
                    _stack_hidden_step(intact_rows, pair, 1)
                ).to(DEVICE)
                for state_index in range(relation_count):
                    baseline = input_drive + torch.matmul(
                        evaluator.net.w
                        + evaluator.net.alpha.detach() * loo[state_index],
                        h0.view(subjects, hidden_size, 1),
                    )
                    current_action = torch.matmul(
                        effective[state_index],
                        h0.view(subjects, hidden_size, 1),
                    )
                    current_hidden = evaluator.net.activ(
                        baseline + current_action
                    ) - evaluator.net.activ(baseline)
                    action[state_index, :, query_index, orientation] = current_action[
                        :, :, 0
                    ]
                    hidden_effect[state_index, :, query_index, orientation] = (
                        current_hidden[:, :, 0]
                    )
                    validation[
                        "operator_preactivation_reconstruction_max_abs_error"
                    ] = max(
                        validation[
                            "operator_preactivation_reconstruction_max_abs_error"
                        ],
                        float(
                            torch.max(
                                torch.abs(
                                    baseline + current_action - intact_preactivation
                                )
                            ).item()
                        ),
                    )
                    actual_loo_h0 = torch.from_numpy(
                        _stack_hidden_step(loo_rows[state_index], pair, 0)
                    ).to(DEVICE)
                    actual_loo_h1 = torch.from_numpy(
                        _stack_hidden_step(loo_rows[state_index], pair, 1)
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
                                    current_hidden[:, :, 0]
                                    - (actual_intact_h1 - actual_loo_h1)
                                )
                            ).item()
                        ),
                    )

    action_np = action.detach().cpu().numpy().astype(np.float64)
    hidden_np = hidden_effect.detach().cpu().numpy().astype(np.float64)
    h0_np = h0_values.detach().cpu().numpy().astype(np.float64)
    effective_np = effective.detach().cpu().numpy().astype(np.float64)
    action_antisymmetric = 0.5 * (action_np[:, :, :, 0] - action_np[:, :, :, 1])
    hidden_antisymmetric = 0.5 * (hidden_np[:, :, :, 0] - hidden_np[:, :, :, 1])
    h0_antisymmetric = 0.5 * (h0_np[:, :, 0] - h0_np[:, :, 1])

    operator_norm = np.linalg.norm(
        effective_np.reshape(relation_count, subjects, -1), axis=2
    )
    h0_norm = np.linalg.norm(h0_np, axis=3)
    action_norm_oriented = np.linalg.norm(action_np, axis=4)
    hidden_norm_oriented = np.linalg.norm(hidden_np, axis=4)
    denominator = operator_norm[:, :, None, None] * h0_norm[None]
    normalized_gain_oriented = np.divide(
        action_norm_oriented,
        denominator,
        out=np.full_like(action_norm_oriented, np.nan),
        where=denominator > tolerance,
    )
    nonlinear_gain_oriented = np.divide(
        hidden_norm_oriented,
        action_norm_oriented,
        out=np.full_like(hidden_norm_oriented, np.nan),
        where=action_norm_oriented > tolerance,
    )
    omitted_matrix = np.broadcast_to(
        (~retained)[:, :, None, None, None], action_np.shape
    )
    validation.update(
        {
            "stable_omitted_effective_operator_max_abs": float(
                np.max(np.abs(effective_np[(~retained)]))
            ),
            "stable_omitted_operator_action_max_abs": float(
                np.max(np.abs(action_np[omitted_matrix]))
            ),
            "stable_omitted_hidden_effect_max_abs": float(
                np.max(np.abs(hidden_np[omitted_matrix]))
            ),
            "floating_reproduction_tolerance": tolerance,
        }
    )
    reproduction_names = (
        "manual_h0_max_abs_error",
        "loo_h0_invariance_max_abs_error",
        "operator_preactivation_reconstruction_max_abs_error",
        "nonlinear_hidden_reconstruction_max_abs_error",
    )
    if any(validation[name] > tolerance for name in reproduction_names):
        raise RuntimeError(f"operator factorial reproduction failed: {validation}")
    return {
        "operator_action": action_antisymmetric,
        "nonlinear_hidden_effect": hidden_antisymmetric,
        "query_key_h0": h0_antisymmetric.transpose(1, 0, 2),
        "normalized_operator_gain": np.mean(normalized_gain_oriented, axis=3),
        "operator_action_norm": np.mean(action_norm_oriented, axis=3),
        "nonlinear_hidden_effect_norm": np.mean(hidden_norm_oriented, axis=3),
        "nonlinear_gain": np.mean(nonlinear_gain_oriented, axis=3),
    }, validation


def _add_relation_labels(
    summary: dict,
    identity_relations: tuple[tuple[int, int], ...],
    context_relations: tuple[tuple[int, int], ...] | None,
):
    for index, row in enumerate(summary.get("per_identity", ())):
        row["identity_relation"] = [int(value) for value in identity_relations[index]]
    if context_relations is not None:
        for index, row in enumerate(summary.get("per_context", ())):
            row["context_relation"] = [int(value) for value in context_relations[index]]


def _identity_change(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    valid: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    names = (
        "own_minus_other_selectivity",
        "eight_way_identification_accuracy",
    )
    subject = {
        name: _masked_mean(second[name] - first[name], valid, axis=(0, 2))
        for name in names
    }
    return {
        "summary": {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in subject.items()
        },
        "raw_subject_level": {
            name: json_values(values) for name, values in subject.items()
        },
    }


def operator_binding_decision(flags: dict[str, bool]) -> str:
    operator_state = flags["operator_state_identity"]
    binding = flags["operator_binding"]
    hidden_state = flags["hidden_state_identity"]
    hidden_query = flags["hidden_query_identity_control"]
    if operator_state and hidden_state and binding:
        return "query_keyed_operator_missing_fidelity_transformation"
    if operator_state and hidden_state and not binding:
        return "state_operator_identity_without_matching_key_access"
    if operator_state and not hidden_state:
        return "operator_identity_lost_in_recurrent_expression"
    if not operator_state and not binding and hidden_state:
        return "identity_generated_by_operating_point_nonlinearity"
    if not operator_state and not binding and not hidden_state and hidden_query:
        return "persistent_local_state_identity_not_supported"
    return "mixed_pattern_requires_new_registered_hierarchy"


def _run_seed(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
    specification: dict,
    counts: np.ndarray,
) -> dict:
    evaluator, _behavior = load_frozen_evaluator(
        registration, pilot_specification, protocol
    )
    execution = specification["execution_contract"]
    tolerance = float(execution["scientific_zero_tolerance"])
    interval = float(execution["bootstrap_interval"])
    folds = int(execution["subject_folds"])
    chance = 1.0 / len(protocol.support_pairs_higher_lower)
    intact, loo, effective, retained = replay_terminal_states(evaluator, protocol)
    levels, validation = collect_factorial(
        evaluator,
        protocol,
        intact,
        loo,
        effective,
        retained,
        tolerance=float(execution["floating_reproduction_tolerance"]),
    )
    relations = protocol.support_pairs_higher_lower
    relation_count = len(relations)
    valid_state = np.broadcast_to(
        retained[:, :, None], (relation_count, evaluator.config.bs, relation_count)
    )
    valid_query = np.broadcast_to(
        retained.T[None], (relation_count, evaluator.config.bs, relation_count)
    )
    all_valid_h0 = np.ones((relation_count, evaluator.config.bs, 1), dtype=bool)

    identity_results = {}
    identity_metrics = {}
    for level_name in ("operator_action", "nonlinear_hidden_effect"):
        identity_results[level_name] = {}
        identity_metrics[level_name] = {}
        for mode, cross_context in (("fixed_query", False), ("cross_query", True)):
            result, metrics = summarize_contextual_identity(
                levels[level_name],
                valid_state,
                counts,
                subject_folds=folds,
                interval=interval,
                chance=chance,
                tolerance=tolerance,
                cross_context=cross_context,
            )
            _add_relation_labels(result, relations, relations)
            identity_results[level_name][mode] = result
            identity_metrics[level_name][mode] = metrics

    query_identity = {}
    for level_name in ("operator_action", "nonlinear_hidden_effect"):
        transposed = levels[level_name].transpose(2, 1, 0, 3)
        query_identity[level_name] = {}
        for mode, cross_context in (("fixed_state", False), ("cross_state", True)):
            result, _metrics = summarize_contextual_identity(
                transposed,
                valid_query,
                counts,
                subject_folds=folds,
                interval=interval,
                chance=chance,
                tolerance=tolerance,
                cross_context=cross_context,
            )
            _add_relation_labels(result, relations, relations)
            query_identity[level_name][mode] = result
    h0_result, _h0_metrics = summarize_contextual_identity(
        levels["query_key_h0"][:, :, None],
        all_valid_h0,
        counts,
        subject_folds=folds,
        interval=interval,
        chance=chance,
        tolerance=tolerance,
        cross_context=False,
    )
    _add_relation_labels(h0_result, relations, None)
    query_identity["query_key_h0"] = h0_result

    overlap = overlap_classes(relations)
    binding = {
        name: structured_contrasts(
            levels[name], retained, overlap, counts, interval=interval
        )
        for name in (
            "normalized_operator_gain",
            "operator_action_norm",
            "nonlinear_hidden_effect_norm",
            "nonlinear_gain",
        )
    }
    for row in binding.values():
        for index, relation_row in enumerate(row["per_relation"]):
            relation_row["relation"] = [int(value) for value in relations[index]]
    binding["fixed_query_state_selectivity"] = structured_contrasts(
        identity_metrics["operator_action"]["fixed_query"][
            "own_minus_other_selectivity"
        ],
        retained,
        overlap,
        counts,
        interval=interval,
    )
    for index, relation_row in enumerate(
        binding["fixed_query_state_selectivity"]["per_relation"]
    ):
        relation_row["relation"] = [int(value) for value in relations[index]]

    normalized = binding["normalized_operator_gain"]["summary"]
    operator_binding = (
        normalized["matched_minus_shared_endpoint"]["bootstrap"]["lower"] > 0.0
        and normalized["matched_minus_disjoint"]["bootstrap"]["lower"] > 0.0
        and validation["stable_omitted_operator_action_max_abs"] <= tolerance
    )
    flags = {
        "operator_state_identity": identity_results["operator_action"]["fixed_query"][
            "presence_rule_passed"
        ],
        "cross_query_operator_state_identity": identity_results["operator_action"][
            "cross_query"
        ]["presence_rule_passed"],
        "operator_binding": bool(operator_binding),
        "hidden_state_identity": identity_results["nonlinear_hidden_effect"][
            "fixed_query"
        ]["presence_rule_passed"],
        "cross_query_hidden_state_identity": identity_results[
            "nonlinear_hidden_effect"
        ]["cross_query"]["presence_rule_passed"],
        "hidden_query_identity_control": query_identity["nonlinear_hidden_effect"][
            "fixed_state"
        ]["presence_rule_passed"],
    }
    return {
        "seed": int(registration["seed"]),
        "subjects": evaluator.config.bs,
        "relations": [[int(first), int(second)] for first, second in relations],
        "state_identity": identity_results,
        "query_identity_controls": query_identity,
        "binding": binding,
        "nonlinear_H_minus_A_identity": {
            mode: _identity_change(
                identity_metrics["operator_action"][mode],
                identity_metrics["nonlinear_hidden_effect"][mode],
                valid_state,
                counts,
                interval=interval,
            )
            for mode in ("fixed_query", "cross_query")
        },
        "primary_presence": flags,
        "outcome": operator_binding_decision(flags),
        "validation": {
            **validation,
            "scientific_zero_tolerance": tolerance,
            "overlap_count_per_relation": {
                "matched": 1,
                "shared_endpoint_mismatch": 2,
                "disjoint_mismatch": 5,
            },
        },
        "checkpoint": {
            "path": registration["checkpoint_path"],
            "sha256": registration["checkpoint_sha256"],
        },
    }


def _overall_diagnosis(seed_results: dict[str, dict]) -> dict:
    names = tuple(next(iter(seed_results.values()))["primary_presence"])
    replicated = {
        name: all(row["primary_presence"][name] for row in seed_results.values())
        for name in names
    }
    seed_outcomes = {row["outcome"] for row in seed_results.values()}
    return {
        "replicated_primary_presence": replicated,
        "outcome": (
            operator_binding_decision(replicated)
            if len(seed_outcomes) == 1
            else "mixed_across_development_seeds"
        ),
        "seed_outcomes": {seed: row["outcome"] for seed, row in seed_results.items()},
        "operator_state_geometry": (
            "query_invariant"
            if replicated["cross_query_operator_state_identity"]
            else "basis_dependent_or_absent"
        ),
        "hidden_state_geometry": (
            "query_invariant"
            if replicated["cross_query_hidden_state_identity"]
            else "basis_dependent_or_absent"
        ),
        "formal_interpretation": "hypothesis_generating_only",
    }


def run_state_query_operator_binding(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = resolve_path(str(specification_path))
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    runtime = configure_formal_runtime()
    if not runtime["cuda_available"] or DEVICE != "cuda":
        raise RuntimeError("state-query operator binding requires visible CUDA")
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
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
            specification,
            counts,
        )
        for registration in sources["pilot_artifacts"]
    }
    if set(seed_results) != {str(seed) for seed in execution["seeds"]}:
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
            "path": "fsrl/state_query_operator_binding.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "execution_runtime": runtime,
        "artifact_validation": validation,
        "contracts": {
            "terminal_state": specification["terminal_state_contract"],
            "factorial": specification["factorial_contract"],
            "state_identity": specification["state_identity_contract"],
            "query_identity_control": specification["query_identity_control"],
            "binding": specification["binding_contract"],
            "nonlinear_attribution": specification["nonlinear_attribution_contract"],
            "decision_tree": specification["outcome_contingent_decision_tree"],
        },
        "seed_results": seed_results,
        "overall_diagnosis": _overall_diagnosis(seed_results),
        "formal_seed_access": False,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Test frozen state-query operator binding."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_state_query_operator_binding(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
