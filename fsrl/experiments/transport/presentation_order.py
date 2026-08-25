"""Registered Liu presentation-order transport evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    hodge_potentials,
    kendall_tau_scores,
)
from fsrl.analysis.policy import bundle_logits, margin_fields
from fsrl.analysis.statistics import (
    finite_column_mean,
    json_values,
    summarize_difference,
    summarize_subjects,
)
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
    retained_relation_mask,
)
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    build_access_trace,
    build_fast_weight_loo,
    measure_presentation_invariance,
)
from fsrl.experiments.local_fidelity.trace_pilot import query_pass
from fsrl.experiments.transport.topology import (
    bootstrap_counts,
    condition_metrics,
    constructive_metrics,
    edge_key,
    finite_primary,
    individualized_metrics,
    reconstruct_local_ledger,
    relation_loo_metrics,
    serial_position_endpoint,
)
from fsrl.experiments.transport.topology import (
    within_cell_decision as topology_within_cell_decision,
)
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.study_registry import (
    canonical_file_registration,
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.registered_protocol import (
    RankingProtocol,
    SupportTrial,
    load_ranking_protocol,
)

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/liu_presentation_order_transport_v1.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/liu_presentation_order_transport_v1.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/liu_presentation_order_transport_v1.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/presentation_order_transport.py",
    "runtime_entrypoint": "fsrl/presentation_order_runtime.py",
    "tests": "tests/test_presentation_order_transport.py",
}


def write_implementation_lock(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    lock = {
        "schema_version": 1,
        "experiment_id": "liu-presentation-order-transport-v1",
        "implementation_status": "frozen_before_any_nonbaseline_schedule_model_evaluation",
        "registration_commit": "ff48f0d1aed863a79986b39be53116f6cd727d83",
        "specification_sha256": file_sha256(specification_path),
        "implementation_sources": {
            name: canonical_file_registration(path)
            for name, path in IMPLEMENTATION_SOURCES.items()
        },
    }
    write_json_exclusive(lock_path, lock)
    return lock


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(lock_path)
    registrations = {
        **specification["registered_sources"],
        "specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["specification_sha256"],
        },
        **lock["implementation_sources"],
    }
    for seed, artifacts in specification["development_backbones"]["artifacts"].items():
        for name, registration in artifacts.items():
            registrations[f"seed_{seed}_{name}"] = registration
    checks = []
    for name, registration in registrations.items():
        path = resolve_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "expected": registration["sha256"],
                "observed": observed,
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(
            f"presentation-order source or artifact lock failed: {checks}"
        )
    return {"passed": True, "checks": checks, "lock": lock}


def transform_schedule(
    schedule: tuple[SupportTrial, ...],
    protocol: RankingProtocol,
    condition: str,
) -> tuple[SupportTrial, ...]:
    if condition == "blockwise_random":
        return schedule
    if condition == "relation_clustered":
        relation_index = {
            relation: index
            for index, relation in enumerate(protocol.support_pairs_higher_lower)
        }
        ordered = sorted(
            schedule,
            key=lambda trial: relation_index[(trial.higher_item, trial.lower_item)],
        )
    elif condition == "reverse":
        ordered = list(reversed(schedule))
    else:
        raise ValueError(f"unknown presentation-order condition: {condition}")
    return tuple(
        replace(trial, block_index=index // len(protocol.support_pairs_higher_lower))
        for index, trial in enumerate(ordered)
    )


def _trial_identity(trial: SupportTrial) -> tuple:
    return (
        trial.left_item,
        trial.right_item,
        trial.higher_item,
        trial.lower_item,
        trial.signed_magnitude,
        trial.encoding_reliability,
    )


def schedule_integrity(
    baseline: tuple[tuple[SupportTrial, ...], ...],
    transformed: tuple[tuple[SupportTrial, ...], ...],
    protocol: RankingProtocol,
) -> dict:
    multiset_equal = all(
        Counter(map(_trial_identity, original))
        == Counter(map(_trial_identity, candidate))
        for original, candidate in zip(baseline, transformed, strict=True)
    )
    relation_counts = all(
        Counter((trial.higher_item, trial.lower_item) for trial in schedule)
        == Counter(
            {
                relation: protocol.support_blocks
                for relation in protocol.support_pairs_higher_lower
            }
        )
        for schedule in transformed
    )
    return {
        "same_subject_trial_multiset_excluding_block_index": multiset_equal,
        "every_relation_occurs_four_times": relation_counts,
        "passed": bool(multiset_equal and relation_counts),
    }


def schedule_hash(schedules) -> str:
    payload = json.dumps(
        [[asdict(trial) for trial in schedule] for schedule in schedules],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def configure_schedule(
    evaluator: FrozenFastWeightEvaluator,
    baseline_schedules: tuple[tuple[SupportTrial, ...], ...],
    condition: str,
) -> dict:
    transformed = tuple(
        transform_schedule(schedule, evaluator.protocol, condition)
        for schedule in baseline_schedules
    )
    integrity = schedule_integrity(baseline_schedules, transformed, evaluator.protocol)
    evaluator.support_schedules = transformed
    gain_error = 0.0
    if evaluator.subject_relation_gains is not None:
        trial_gains = []
        for subject, schedule in enumerate(transformed):
            values = tuple(
                evaluator.subject_relation_gains[subject][
                    (trial.higher_item, trial.lower_item)
                ]
                for trial in schedule
            )
            trial_gains.append(values)
            for trial_index, trial in enumerate(schedule):
                expected = evaluator.subject_relation_gains[subject][
                    (trial.higher_item, trial.lower_item)
                ]
                gain_error = max(gain_error, abs(values[trial_index] - expected))
        evaluator.subject_trial_gains = tuple(trial_gains)
    integrity["relation_gain_reassembly_max_abs_error"] = gain_error
    integrity["passed"] = bool(integrity["passed"] and gain_error <= 1e-12)
    return integrity


def local_ledger_arrays(evaluator, natural_scalars: np.ndarray) -> dict:
    pairs = tuple(combinations(range(evaluator.protocol.n_items), 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    ledgers = []
    reads = []
    states = []
    for subject, schedule in enumerate(evaluator.support_schedules):
        codes = np.asarray(evaluator.cue_codes[subject], dtype=np.float64)
        keys = np.stack(
            [edge_key(codes[first], codes[second]) for first, second in pairs]
        )
        ledger = np.zeros(len(pairs), dtype=np.float64)
        for trial_index, trial in enumerate(schedule):
            canonical = tuple(sorted((trial.left_item, trial.right_item)))
            orientation = 1.0 if trial.left_item < trial.right_item else -1.0
            ledger[pair_index[canonical]] += orientation * float(
                natural_scalars[subject, trial_index]
            )
        ledgers.append(ledger)
        states.append(ledger @ keys)
        reads.append((keys @ keys.T) @ ledger)
    return {
        "ledger": np.asarray(ledgers),
        "state": np.asarray(states),
        "reads": np.asarray(reads),
    }


def evaluate_schedule(
    specification: dict,
    seed: int,
    condition: str,
    backbone,
    model_config,
    local: ConjunctiveLocalTrace,
    runtime: dict,
    source_validation: dict,
) -> tuple[dict, dict]:
    evaluation = specification["evaluation"]
    protocol = load_ranking_protocol(
        resolve_path(specification["fixed_task_contract"]["protocol"])
    )
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocol,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
    )
    baseline_schedules = evaluator.support_schedules
    schedule_check = configure_schedule(evaluator, baseline_schedules, condition)
    schedule_index = 1 + specification["schedule_contract"][
        "conditions_in_execution_order"
    ].index(condition)
    bootstrap_seed = 760000 + 100 * seed + schedule_index
    rng = np.random.default_rng(bootstrap_seed)
    counts = bootstrap_counts(
        rng,
        int(evaluation["bootstrap_samples"]),
        int(evaluation["subjects_per_schedule_and_backbone"]),
    )
    interval = float(evaluation["bootstrap_interval"])
    geometry = build_complete_graph_geometry(protocol)
    relations = tuple(protocol.support_pairs_higher_lower)
    learned_mask = np.asarray(
        [pair in protocol.learned_pairs for pair in geometry.pairs]
    )
    schedules = tuple(ordered_pairs(protocol.n_items) for _ in range(model_config.bs))
    before = tensor_hashes(backbone)
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    loo_fast_weights = build_fast_weight_loo(evaluator, relations)
    intact_trace = build_access_trace(evaluator, local, dual_access=True)
    loo_traces = [
        build_access_trace(
            evaluator, local, dual_access=True, zero_relations=frozenset((relation,))
        )
        for relation in relations
    ]
    intact_bundle = query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        schedules,
        local_off=False,
        global_off=False,
        shuffled_indices=None,
    )
    a_off_bundle = query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        schedules,
        local_off=True,
        global_off=False,
        shuffled_indices=None,
    )
    p_off_bundle = query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        schedules,
        local_off=False,
        global_off=True,
        shuffled_indices=None,
    )
    loo_global_bundles = [
        query_pass(
            evaluator,
            local,
            loo_fast_weights[index],
            loo_traces[index].state,
            schedules,
            local_off=True,
            global_off=False,
            shuffled_indices=None,
        )
        for index in range(len(relations))
    ]
    loo_local_bundles = [
        query_pass(
            evaluator,
            local,
            intact_fast_weights,
            loo_traces[index].state,
            schedules,
            local_off=False,
            global_off=True,
            shuffled_indices=None,
        )
        for index in range(len(relations))
    ]
    fields = {
        "intact": margin_fields(intact_bundle, protocol.n_items),
        "a_off": margin_fields(a_off_bundle, protocol.n_items),
        "P_off_a_on": margin_fields(p_off_bundle, protocol.n_items),
    }
    loo_global_fields = np.asarray(
        [margin_fields(bundle, protocol.n_items) for bundle in loo_global_bundles]
    )
    loo_local_fields = np.asarray(
        [margin_fields(bundle, protocol.n_items) for bundle in loo_local_bundles]
    )
    conditions = {
        name: condition_metrics(
            field,
            geometry,
            learned_mask,
            counts,
            interval,
            float(evaluation["temperature"]),
        )
        for name, field in fields.items()
    }
    global_loo = relation_loo_metrics(
        fields["a_off"], loo_global_fields, relations, geometry, counts, interval
    )
    local_loo = relation_loo_metrics(
        fields["P_off_a_on"], loo_local_fields, relations, geometry, counts, interval
    )

    def raw(condition_name: str, group: str) -> np.ndarray:
        return np.asarray(
            conditions[condition_name]["raw_subject"]["correct_probability"][group]
        )

    global_remote = np.asarray(global_loo["raw_subject"]["remote_absolute"])
    local_remote = np.asarray(local_loo["raw_subject"]["remote_absolute"])
    contrasts = {
        "intact_minus_a_off_learned_probability": summarize_difference(
            raw("intact", "learned"),
            raw("a_off", "learned"),
            counts,
            interval=interval,
        ),
        "P_off_learned_minus_nonlearned_probability": summarize_difference(
            raw("P_off_a_on", "learned"),
            raw("P_off_a_on", "nonlearned"),
            counts,
            interval=interval,
        ),
        "P_off_local_remote_minus_quarter_global": summarize_subjects(
            local_remote - 0.25 * global_remote, counts, interval=interval
        ),
    }
    behavior = analyze_sampled_query_policy(
        protocol,
        bundle_logits(intact_bundle, schedules),
        seed=int(evaluation["choice_seed"]),
        temperature=float(evaluation["temperature"]),
    )
    sampled_accuracy = {
        name: summarize_subjects(
            np.asarray([row[name] for row in behavior["subjects"]]),
            counts,
            interval=interval,
        )
        for name in ("overall_accuracy", "learned_accuracy", "nonlearned_accuracy")
    }
    exact = reconstruct_local_ledger(
        evaluator.cue_codes,
        evaluator.support_schedules,
        intact_trace.natural_scalars,
        intact_trace.state.detach().cpu().numpy().astype(np.float64),
        intact_bundle["raw_local_margins"][:, 0::2],
    )
    retained = retained_relation_mask(evaluator, relations)
    correct_probability = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                fields["intact"]
                * geometry.true_sign[None]
                / float(evaluation["temperature"]),
                -60.0,
                60.0,
            )
        )
    )
    relation_indices = [
        geometry.pairs.index(tuple(sorted(relation))) for relation in relations
    ]
    learned_probability = correct_probability[:, relation_indices].T
    metrics = {
        "conditions": conditions,
        "constructive": constructive_metrics(
            fields["intact"], fields["a_off"], geometry, counts, interval
        ),
        "individualized": individualized_metrics(
            behavior, rng, int(evaluation["bootstrap_samples"])
        ),
        "global_relation_LOO": global_loo,
        "P_off_local_relation_LOO": local_loo,
        "contrasts": contrasts,
        "local_exactness": exact,
        "retained_omitted": {
            "retained_counts_per_subject": json_values(np.sum(retained, axis=0)),
            "omitted_counts_per_subject": json_values(np.sum(~retained, axis=0)),
            "retained_correct_probability": summarize_subjects(
                finite_column_mean(np.where(retained, learned_probability, np.nan)),
                counts,
                interval=interval,
            ),
            "omitted_correct_probability": summarize_subjects(
                finite_column_mean(np.where(~retained, learned_probability, np.nan)),
                counts,
                interval=interval,
            ),
        },
        "sampled_behavior": behavior,
        "sampled_accuracy_bootstrap": sampled_accuracy,
        "serial_position_endpoint": serial_position_endpoint(behavior, protocol),
    }
    presentation = measure_presentation_invariance(
        evaluator, local, intact_trace.natural_scalars
    )
    after = tensor_hashes(backbone)
    integrity = {
        "source_validation_passed": bool(source_validation["passed"]),
        "schedule_integrity": schedule_check,
        "bounded_gpu_runtime": bool(
            runtime["active"]
            and runtime["cuda_available"]
            and runtime["torch_intraop_threads"] == 1
            and runtime["torch_interop_threads"] == 1
        ),
        "backbone_tensor_hashes_unchanged": before == after,
        "local_off_global_logit_max_abs_error": float(
            np.max(np.abs(a_off_bundle["logits"] - a_off_bundle["global_logits"]))
        ),
        **presentation,
        "primary_values_finite": finite_primary(metrics),
    }
    integrity["all_passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["schedule_integrity"]["passed"]
        and integrity["bounded_gpu_runtime"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and integrity["local_off_global_logit_max_abs_error"] <= 1e-6
        and integrity["support_write_reversal_max_abs_error"] <= 1e-7
        and integrity["query_key_reversal_max_abs_error"] <= 1e-7
        and integrity["primary_values_finite"]
    )
    public = {
        "condition": condition,
        "support_schedule_sha256": schedule_hash(evaluator.support_schedules),
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "integrity": integrity,
    }
    internal = {
        "counts": counts,
        "ledger": local_ledger_arrays(evaluator, intact_trace.natural_scalars),
        "global_field": fields["a_off"],
        "global_remote": global_remote,
        "geometry": geometry,
    }
    return public, internal


def cross_schedule_metrics(
    baseline: dict,
    candidate: dict,
    counts: np.ndarray,
    interval: float,
) -> dict:
    state_error = float(
        np.max(np.abs(candidate["ledger"]["state"] - baseline["ledger"]["state"]))
    )
    read_error = float(
        np.max(np.abs(candidate["ledger"]["reads"] - baseline["ledger"]["reads"]))
    )
    ledger_error = float(
        np.max(np.abs(candidate["ledger"]["ledger"] - baseline["ledger"]["ledger"]))
    )
    first = baseline["global_field"]
    second = candidate["global_field"]
    correlations = []
    for left, right in zip(first, second, strict=True):
        correlations.append(
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > 0.0 and np.std(right) > 0.0
            else np.nan
        )
    rmse = np.sqrt(np.mean((first - second) ** 2, axis=1))
    agreement = np.mean((first > 0.0) == (second > 0.0), axis=1)
    potential_tau = kendall_tau_scores(
        hodge_potentials(first, baseline["geometry"]),
        hodge_potentials(second, baseline["geometry"]),
    )
    remote_difference = candidate["global_remote"] - baseline["global_remote"]
    return {
        "local_identity": {
            "ledger_max_abs_error": ledger_error,
            "tensor_state_max_abs_error": state_error,
            "all_query_read_max_abs_error": read_error,
        },
        "global_field_diagnostics": {
            "pearson": summarize_subjects(
                np.asarray(correlations), counts, interval=interval
            ),
            "centered_rmse": summarize_subjects(rmse, counts, interval=interval),
            "exact_decision_agreement": summarize_subjects(
                agreement, counts, interval=interval
            ),
            "hodge_potential_kendall_tau": summarize_subjects(
                potential_tau, counts, interval=interval
            ),
            "remote_LOO_difference": summarize_subjects(
                remote_difference, counts, interval=interval
            ),
        },
    }


def within_cell_decision(metrics: dict, integrity: dict) -> dict:
    decision = topology_within_cell_decision(metrics, integrity)
    topology_exact = decision["flags"].pop("exact_local_compression")
    cross = metrics["cross_schedule_local_identity"]
    exact = bool(
        topology_exact
        and cross["ledger_max_abs_error"] <= 1e-12
        and cross["tensor_state_max_abs_error"] <= 1e-12
        and cross["all_query_read_max_abs_error"] <= 1e-12
    )
    decision["flags"]["exact_order_invariant_local_algorithm"] = exact
    decision["all_eight_primary_links_pass"] = all(decision["flags"].values())
    return decision


def cross_cell_decision(
    seeds: dict, conditions: list[str], mandatory_seeds: list[int]
) -> dict:
    cells = [
        seeds[str(seed)]["conditions"][condition]
        for condition in conditions
        for seed in mandatory_seeds
    ]
    if not all(cell["decision"]["interpretable"] for cell in cells):
        outcome = "NONINTERPRETABLE_EXECUTION"
        condition_passes = None
        heterogeneous = None
    elif not all(cell["decision"]["competence_passed"] for cell in cells):
        outcome = "PRESENTATION_ORDER_COMPETENCE_NOT_ESTABLISHED"
        condition_passes = None
        heterogeneous = None
    else:
        condition_passes = {
            condition: all(
                seeds[str(seed)]["conditions"][condition]["decision"][
                    "all_eight_primary_links_pass"
                ]
                for seed in mandatory_seeds
            )
            for condition in conditions
        }
        links = next(iter(cells))["decision"]["flags"]
        heterogeneous = any(
            0
            < sum(
                seeds[str(seed)]["conditions"][condition]["decision"]["flags"][link]
                for seed in mandatory_seeds
            )
            < len(mandatory_seeds)
            for condition in conditions
            for link in links
        )
        passed = sum(condition_passes.values())
        if passed == len(conditions):
            outcome = "LIU_PRESENTATION_ORDER_MECHANISM_TRANSPORTED"
        elif passed > 0 or heterogeneous:
            outcome = "ORDER_DEPENDENT_OR_UNRESOLVED"
        else:
            outcome = "FUNCTIONAL_ASYMMETRY_NOT_ORDER_TRANSPORTED"
    return {
        "outcome": outcome,
        "condition_passes": condition_passes,
        "heterogeneous_across_backbones": heterogeneous,
    }


def evaluate(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    runtime = require_formal_runtime()
    specification = load_json(specification_path)
    source_validation = validate_sources(specification_path, lock_path)
    evaluation = specification["evaluation"]
    condition_order = list(
        specification["schedule_contract"]["conditions_in_execution_order"]
    )
    seeds = {}
    for seed in specification["development_backbones"]["mandatory_seeds"]:
        artifacts = specification["development_backbones"]["artifacts"][str(seed)]
        backbone, model_config, checkpoint = load_retro_checkpoint(
            resolve_path(artifacts["checkpoint"]["path"]),
            int(evaluation["subjects_per_schedule_and_backbone"]),
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        gain = load_json(resolve_path(artifacts["gain"]["path"]))
        local = ConjunctiveLocalTrace(model_config.cs)
        with torch.no_grad():
            local.raw_gain.fill_(float(gain["raw_lambda_L"]))
        conditions = {}
        internals = {}
        for condition in condition_order:
            conditions[condition], internals[condition] = evaluate_schedule(
                specification,
                int(seed),
                condition,
                backbone,
                model_config,
                local,
                runtime,
                source_validation,
            )
        baseline = internals["blockwise_random"]
        for condition in condition_order:
            if condition == "blockwise_random":
                cross = {
                    "local_identity": {
                        "ledger_max_abs_error": 0.0,
                        "tensor_state_max_abs_error": 0.0,
                        "all_query_read_max_abs_error": 0.0,
                    },
                    "global_field_diagnostics": {
                        "reference_condition": "self",
                    },
                }
            else:
                cross = cross_schedule_metrics(
                    baseline,
                    internals[condition],
                    internals[condition]["counts"],
                    float(evaluation["bootstrap_interval"]),
                )
            conditions[condition]["metrics"]["cross_schedule_local_identity"] = cross[
                "local_identity"
            ]
            conditions[condition]["metrics"]["global_field_vs_blockwise_random"] = (
                cross["global_field_diagnostics"]
            )
            conditions[condition]["decision"] = within_cell_decision(
                conditions[condition]["metrics"], conditions[condition]["integrity"]
            )
        seeds[str(seed)] = {
            "seed": int(seed),
            "checkpoint": asdict(checkpoint),
            "gain_path": artifacts["gain"]["path"],
            "lambda_L": float(local.gain.detach().cpu()),
            "conditions": conditions,
        }
    mandatory_seeds = [
        int(seed) for seed in specification["development_backbones"]["mandatory_seeds"]
    ]
    return {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "execution_runtime": runtime,
        "source_validation": source_validation,
        "seeds": seeds,
        "decision": cross_cell_decision(seeds, condition_order, mandatory_seeds),
        "registered_primary_links": specification["primary_links"],
        "registered_outcome_tree": specification["outcome_tree"],
        "known_limitations_carried_forward": [
            "excessive symbolic-distance slope",
            "weak serial-position endpoint contrast",
            "original-graph seed-2104 self-inconsistency mismatch",
        ],
    }


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--write-lock", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.write_lock:
        write_implementation_lock(parsed.specification, parsed.implementation_lock)
        return 0
    result = evaluate(parsed.specification, parsed.implementation_lock)
    write_json_exclusive(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
