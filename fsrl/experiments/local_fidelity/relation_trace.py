"""Localize relation identity across frozen support and query states."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy import stats

from fsrl.analysis.hodge import CompleteGraphGeometry, build_complete_graph_geometry
from fsrl.analysis.statistics import bootstrap_counts, json_values, summarize_subjects
from fsrl.evaluation.fields import ordered_query_schedule
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    retained_relation_mask,
)
from fsrl.evaluation.registered import load_registered_frozen_evaluator
from fsrl.experiments.assembly.write_localization import trace_support_trial
from fsrl.experiments.local_fidelity.hidden_residual import (
    validate_registered_sources,
    vector_hodge_components,
)
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import legacy_identifier, resolve_record
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/relation_trace_localization_v1_1.json"
)


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


def prototype_identity_metrics(
    traces: np.ndarray,
    retained: np.ndarray,
    *,
    subject_folds: int,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values = np.asarray(traces)
    relations, subjects = values.shape[:2]
    flat = values.reshape(relations, subjects, -1).astype(np.float64, copy=False)
    if retained.shape != (relations, subjects):
        raise ValueError("retained mask does not match trace population")

    subject_fold = np.arange(subjects) % subject_folds
    similarities = np.full((relations, subjects, relations), np.nan)
    prototypes = np.empty((subject_folds, relations, flat.shape[-1]), dtype=np.float64)
    for fold in range(subject_folds):
        for relation in range(relations):
            train = (subject_fold != fold) & retained[relation]
            prototype = np.mean(flat[relation, train], axis=0)
            norm = np.linalg.norm(prototype)
            if norm <= tolerance:
                raise RuntimeError("same-relation training prototype is zero")
            prototypes[fold, relation] = prototype / norm
        test_subjects = np.flatnonzero(subject_fold == fold)
        for relation in range(relations):
            rows, norms = _unit_rows(flat[relation, test_subjects], tolerance)
            valid = norms > tolerance
            relation_similarities = rows @ prototypes[fold].T
            relation_similarities[~valid] = np.nan
            similarities[relation, test_subjects] = relation_similarities

    own = np.stack(
        [similarities[relation, :, relation] for relation in range(relations)]
    )
    other = (np.nansum(similarities, axis=2) - own) / float(relations - 1)
    selectivity = own - other
    identification = np.full((relations, subjects), np.nan)
    for relation in range(relations):
        valid = np.isfinite(own[relation])
        identification[relation, valid] = (
            np.argmax(similarities[relation, valid], axis=1) == relation
        ).astype(np.float64)
    trace_norm = np.linalg.norm(flat, axis=2)
    return {
        "own_prototype_cosine": own,
        "other_prototype_mean_cosine": other,
        "own_minus_other_selectivity": selectivity,
        "eight_way_identification_accuracy": identification,
        "trace_l2_norm": trace_norm,
    }, prototypes


def _retained_subject_mean(values: np.ndarray, retained: np.ndarray) -> np.ndarray:
    rows = np.where(retained, np.asarray(values, dtype=np.float64), np.nan)
    return np.nanmean(rows, axis=0)


def summarize_identity_level(
    traces: np.ndarray,
    retained: np.ndarray,
    counts: np.ndarray,
    *,
    subject_folds: int,
    interval: float,
    chance: float,
    tolerance: float,
) -> tuple[dict, np.ndarray]:
    relation_metrics, prototypes = prototype_identity_metrics(
        traces,
        retained,
        subject_folds=subject_folds,
        tolerance=tolerance,
    )
    subject_metrics = {
        name: _retained_subject_mean(values, retained)
        for name, values in relation_metrics.items()
    }
    summaries = {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in subject_metrics.items()
    }
    omitted_selector = np.broadcast_to(
        (~retained).reshape(*retained.shape, *([1] * (traces.ndim - 2))),
        traces.shape,
    )
    omitted_max = (
        float(np.max(np.abs(traces[omitted_selector]))) if np.any(~retained) else 0.0
    )
    present = (
        summaries["own_minus_other_selectivity"]["bootstrap"]["lower"] > 0.0
        and summaries["eight_way_identification_accuracy"]["bootstrap"]["lower"]
        > chance
        and omitted_max <= tolerance
    )
    per_relation = []
    for relation in range(traces.shape[0]):
        mask = retained[relation]
        per_relation.append(
            {
                "relation_index": relation,
                "retained_subjects": int(np.sum(mask)),
                "stable_omitted_subjects": int(np.sum(~mask)),
                "retained": {
                    name: summarize_subjects(
                        np.where(mask, values[relation], np.nan),
                        counts,
                        interval=interval,
                    )
                    for name, values in relation_metrics.items()
                },
            }
        )
    return {
        "subjects": traces.shape[1],
        "summary": summaries,
        "per_relation": per_relation,
        "stable_omitted_max_abs": omitted_max,
        "presence_rule_passed": bool(present),
        "raw_subject_level": {
            name: json_values(values) for name, values in subject_metrics.items()
        },
    }, prototypes


def _prototype_rdm(prototypes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(prototypes, axis=2, keepdims=True)
    unit = prototypes / norms
    return 1.0 - np.einsum("fri,fqi->frq", unit, unit)


def prototype_rdm_similarity(
    reference: np.ndarray, comparison: np.ndarray
) -> dict[str, object]:
    first = _prototype_rdm(reference)
    second = _prototype_rdm(comparison)
    upper = np.triu_indices(first.shape[1], k=1)
    values = np.asarray(
        [
            cast(
                Any, stats.spearmanr(first[fold][upper], second[fold][upper])
            ).statistic
            for fold in range(first.shape[0])
        ],
        dtype=np.float64,
    )
    return {
        "folds": len(values),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "values": json_values(values),
    }


def _relation_indices(
    evaluator, protocol: RankingProtocol, trial_index: int
) -> np.ndarray:
    lookup = {
        relation: index
        for index, relation in enumerate(protocol.support_pairs_higher_lower)
    }
    return np.asarray(
        [
            lookup[(trial.higher_item, trial.lower_item)]
            for trial in (
                schedule[trial_index] for schedule in evaluator.support_schedules
            )
        ],
        dtype=np.int64,
    )


def trace_generated_writes(
    evaluator, protocol: RankingProtocol, *, tolerance: float
) -> tuple[dict[str, np.ndarray], torch.Tensor, np.ndarray, dict]:
    relations = len(protocol.support_pairs_higher_lower)
    shape = (relations, evaluator.config.bs, evaluator.config.hs, evaluator.config.hs)
    cumulative_raw = torch.zeros(shape, device=evaluator.device)
    cumulative_intended = torch.zeros(shape, device=evaluator.device)
    exposure_counts = np.zeros((relations, evaluator.config.bs), dtype=np.int64)
    natural = evaluator.initialize_fast_weights()
    validation = {
        "trace_forward_max_abs_error": 0.0,
        "incremental_endpoint_max_abs_error": 0.0,
        "realized_minus_intended_max_abs_error": 0.0,
    }

    for trial_index in range(protocol.support_trials):
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
        realized = plus.final_fast_weights - zero.final_fast_weights
        delta_intended = plus.intended_increment - zero.intended_increment
        intended = torch.sum(delta_intended, dim=1)
        validation["realized_minus_intended_max_abs_error"] = max(
            validation["realized_minus_intended_max_abs_error"],
            float(
                torch.max(
                    torch.abs(
                        realized.to(torch.float64)
                        - torch.sum(delta_intended.to(torch.float64), dim=1)
                    )
                ).item()
            ),
        )
        relation_indices = _relation_indices(evaluator, protocol, trial_index)
        for relation in range(relations):
            subjects = np.flatnonzero(relation_indices == relation)
            cumulative_raw[relation, subjects] += realized[subjects]
            cumulative_intended[relation, subjects] += intended[subjects]
            exposure_counts[relation, subjects] += 1
        natural = plus.final_fast_weights

    reference = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    validation["final_endpoint_max_abs_error"] = float(
        torch.max(torch.abs(reference - natural)).item()
    )
    if not np.all(exposure_counts == protocol.support_blocks):
        raise RuntimeError("every subject-relation must have four support slots")
    if max(validation.values()) > tolerance:
        raise RuntimeError(
            "generated-write replay exceeded frozen tolerance: "
            f"tolerance={tolerance}, validation={validation}"
        )
    alpha = evaluator.net.alpha.detach()
    retained = retained_relation_mask(evaluator, protocol.support_pairs_higher_lower)
    return (
        {
            "generated_effective_write": (alpha * cumulative_raw).cpu().numpy(),
            "generated_raw_write": cumulative_raw.cpu().numpy(),
            "generated_intended_raw_write": cumulative_intended.cpu().numpy(),
        },
        natural,
        retained,
        validation,
    )


def _hidden_trajectory_fields(
    evaluator, fast_weights, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, float]:
    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    hidden_rows, logit_rows = evaluator.readout_hidden_and_logit_trajectories(
        fast_weights, schedules
    )
    fields = np.empty(
        (
            evaluator.config.bs,
            evaluator.config.triallen,
            len(geometry.pairs),
            evaluator.config.hs,
        ),
        dtype=np.float64,
    )
    margins = np.empty(fields.shape[:-1], dtype=np.float64)
    for subject in range(evaluator.config.bs):
        for edge, pair in enumerate(geometry.pairs):
            reverse = (pair[1], pair[0])
            fields[subject, :, edge] = 0.5 * (
                hidden_rows[subject][pair] - hidden_rows[subject][reverse]
            )
            margins[subject, :, edge] = 0.5 * (
                logit_rows[subject][pair] - logit_rows[subject][reverse]
            )
    output_direction = (
        (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0])
        .detach()
        .cpu()
        .numpy()
    )
    analytic = np.einsum("bkeh,h->bke", fields, output_direction)
    return fields, float(np.max(np.abs(analytic - margins)))


def trace_terminal_and_query(
    evaluator,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    intact_weights: torch.Tensor,
    *,
    response_step: int,
) -> tuple[dict[str, np.ndarray], dict]:
    relations = tuple(protocol.support_pairs_higher_lower)
    direct_edges = {
        relation: geometry.pairs.index(tuple(sorted(relation)))
        for relation in relations
    }
    terminal_raw = []
    query_direct = []
    response_residual = []
    intact_hidden, reproduction_error = _hidden_trajectory_fields(
        evaluator, intact_weights, geometry
    )

    for relation in relations:
        weights = evaluator.initialize_fast_weights()
        for trial_index in range(protocol.support_trials):
            weights = evaluator.advance_support_trial(
                weights,
                trial_index,
                zero_relations=frozenset((relation,)),
            )
        terminal_raw.append((intact_weights - weights).detach().cpu().numpy())
        loo_hidden, error = _hidden_trajectory_fields(evaluator, weights, geometry)
        reproduction_error = max(reproduction_error, error)
        influence = intact_hidden - loo_hidden
        edge = direct_edges[relation]
        query_direct.append(influence[:, :, edge])
        _gradient, residual = vector_hodge_components(
            influence[:, response_step], geometry
        )
        response_residual.append(residual[:, edge])

    terminal_raw_array = np.asarray(terminal_raw)
    alpha = evaluator.net.alpha.detach().cpu().numpy()
    query_array = np.asarray(query_direct)
    result = {
        "terminal_effective_fast_weight": alpha * terminal_raw_array,
        "terminal_raw_fast_weight": terminal_raw_array,
        "response_full_hidden": query_array[:, :, response_step],
        "response_hodge_residual": np.asarray(response_residual),
    }
    for step in range(evaluator.config.triallen):
        result[f"query_hidden_step_{step}"] = query_array[:, :, step]
    return result, {
        "hidden_to_logit_projection_max_abs_error": reproduction_error,
        "query_step_0_max_abs_influence": float(np.max(np.abs(query_array[:, :, 0]))),
    }


def relation_trace_decision(flags: dict[str, bool]) -> str:
    write = flags["generated_effective_write"]
    terminal = flags["terminal_effective_fast_weight"]
    response = flags["response_full_hidden"]
    residual = flags["response_hodge_residual"]
    if write and terminal and response and residual:
        return "storage_access_present_missing_fidelity_transformation"
    if write and terminal and response and not residual:
        return "response_identity_outside_direct_hodge_residual"
    if write and terminal and not response and not residual:
        return "persistent_storage_present_query_access_missing"
    if write and not terminal and not response and not residual:
        return "identity_generated_then_compressed_during_integration"
    if not write and not terminal and not response and not residual:
        return "persistent_relation_identity_not_detected"
    return "mixed_pattern_requires_new_registered_hierarchy"


def _run_seed(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    specification: dict,
    counts: np.ndarray,
) -> dict:
    evaluator, _behavior = load_registered_frozen_evaluator(
        registration, pilot_specification, protocol
    )
    execution = specification["execution_contract"]
    tolerance = float(execution["floating_reproduction_tolerance"])
    write_levels, intact_weights, retained, write_validation = trace_generated_writes(
        evaluator, protocol, tolerance=tolerance
    )
    state_levels, state_validation = trace_terminal_and_query(
        evaluator,
        protocol,
        geometry,
        intact_weights,
        response_step=int(
            specification["query_state_contract"]["primary_response_step"]
        ),
    )
    levels = {**write_levels, **state_levels}
    primary_names = (
        "generated_effective_write",
        "terminal_effective_fast_weight",
        "response_full_hidden",
        "response_hodge_residual",
    )
    secondary_names = (
        "generated_raw_write",
        "generated_intended_raw_write",
        "terminal_raw_fast_weight",
        "query_hidden_step_2",
        "query_hidden_step_3",
    )
    summaries = {}
    prototypes = {}
    for name in (*primary_names, *secondary_names):
        summaries[name], prototypes[name] = summarize_identity_level(
            levels[name],
            retained,
            counts,
            subject_folds=int(execution["subject_folds"]),
            interval=float(execution["bootstrap_interval"]),
            chance=1.0 / len(protocol.support_pairs_higher_lower),
            tolerance=float(execution["scientific_zero_tolerance"]),
        )
        for index, relation in enumerate(protocol.support_pairs_higher_lower):
            summaries[name]["per_relation"][index]["relation"] = [
                int(relation[0]),
                int(relation[1]),
            ]

    step_zero = levels["query_hidden_step_0"]
    step_zero_max = float(np.max(np.abs(step_zero)))
    summaries["query_hidden_step_0"] = {
        "max_abs_influence": step_zero_max,
        "within_scientific_zero_tolerance": step_zero_max
        <= float(execution["scientific_zero_tolerance"]),
    }
    reference = prototypes["terminal_effective_fast_weight"]
    rdm = {
        name: prototype_rdm_similarity(reference, value)
        for name, value in prototypes.items()
    }
    primary_flags = {
        name: summaries[name]["presence_rule_passed"] for name in primary_names
    }
    return {
        "seed": int(registration["seed"]),
        "subjects": evaluator.config.bs,
        "relations": [
            [int(first), int(second)]
            for first, second in protocol.support_pairs_higher_lower
        ],
        "primary_levels": {name: summaries[name] for name in primary_names},
        "secondary_levels": {
            name: summaries[name] for name in (*secondary_names, "query_hidden_step_0")
        },
        "prototype_rdm_similarity_to_terminal_effective": rdm,
        "primary_presence": primary_flags,
        "outcome": relation_trace_decision(primary_flags),
        "validation": {
            **write_validation,
            **state_validation,
            "scientific_zero_tolerance": float(execution["scientific_zero_tolerance"]),
            "floating_reproduction_tolerance": tolerance,
        },
        "checkpoint": {
            "path": registration["checkpoint_path"],
            "sha256": registration["checkpoint_sha256"],
        },
    }


def _overall_diagnosis(seed_results: dict[str, dict]) -> dict:
    names = (
        "generated_effective_write",
        "terminal_effective_fast_weight",
        "response_full_hidden",
        "response_hodge_residual",
    )
    replicated = {
        name: all(row["primary_presence"][name] for row in seed_results.values())
        for name in names
    }
    seed_outcomes = {row["outcome"] for row in seed_results.values()}
    return {
        "replicated_primary_presence": replicated,
        "outcome": (
            relation_trace_decision(replicated)
            if len(seed_outcomes) == 1
            else "mixed_across_development_seeds"
        ),
        "seed_outcomes": {seed: row["outcome"] for seed, row in seed_results.items()},
        "formal_interpretation": "hypothesis_generating_only",
    }


def run_relation_trace_localization(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = resolve_path(str(specification_path))
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    runtime = configure_formal_runtime()
    if not runtime["cuda_available"] or runtime["device"] != "cuda":
        raise RuntimeError("relation trace localization requires a visible CUDA device")

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
            "path": "fsrl/relation_trace_localization.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "execution_runtime": runtime,
        "artifact_validation": validation,
        "contracts": {
            "generated_write": specification["generated_write_contract"],
            "terminal_state": specification["terminal_state_contract"],
            "query_state": specification["query_state_contract"],
            "relation_identity": specification["relation_identity_contract"],
            "decision_tree": specification["outcome_contingent_decision_tree"],
        },
        "seed_results": seed_results,
        "overall_diagnosis": _overall_diagnosis(seed_results),
        "formal_seed_access": False,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Localize frozen relation-conditioned traces."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_relation_trace_localization(parsed.specification)
    write_json_exclusive(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
