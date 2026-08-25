"""Registered Liu evidence-sparsity transport evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import torch

from .assembly_trajectory import summarize_difference, summarize_subjects
from .behavioral import analyze_sampled_query_policy
from .confirmation import file_sha256
from .conjunctive_local_trace import ConjunctiveLocalTrace
from .conjunctive_local_trace_pilot import _query_pass
from .curvature_gate_pilot import (
    _ordered_pairs,
    _retained_mask,
    _tensor_hashes,
    bundle_logits,
    margin_fields,
)
from .dual_evidence_access_pilot import (
    _build_fast_weight_loo,
    _presentation_invariance,
    build_access_trace,
)
from .formal_runtime import require_formal_runtime
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from .ranking_protocol import RankingProtocol, SupportTrial, load_ranking_protocol
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record
from .support_topology_transport import (
    ROOT,
    _finite_primary,
    _json_values,
    _schedule_hash,
    _sigmoid,
    bootstrap_counts,
    condition_metrics,
    constructive_metrics,
    graph_descriptor,
    individualized_metrics,
    load_json,
    protocol_for_graph,
    reconstruct_local_ledger,
    relation_loo_metrics,
    resolve_path,
    serial_position_endpoint,
    write_json_exclusive,
)
from .support_topology_transport import (
    within_cell_decision as topology_within_cell_decision,
)

DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/liu_evidence_sparsity_transport_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/liu_evidence_sparsity_transport_v1.lock.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/liu_evidence_sparsity_transport_v1.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/evidence_sparsity_transport.py",
    "runtime_entrypoint": "fsrl/evidence_sparsity_runtime.py",
    "tests": "tests/test_evidence_sparsity_transport.py",
}
REGISTRATION_COMMIT = "6f2d7cdfa7c2c45549f76de1a0b08caca6696864"
CORE_METRIC_KEYS = (
    "conditions",
    "constructive",
    "individualized",
    "global_relation_LOO",
    "P_off_local_relation_LOO",
    "contrasts",
    "local_exactness",
    "retained_omitted",
    "sampled_behavior",
    "sampled_accuracy_bootstrap",
    "serial_position_endpoint",
)


def _registration(path: str) -> dict:
    return {"path": path, "sha256": file_sha256(resolve_record(path))}


def write_implementation_lock(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    lock = {
        "schema_version": 1,
        "experiment_id": "liu-evidence-sparsity-transport-v1",
        "implementation_status": "frozen_before_any_nonbaseline_density_model_evaluation",
        "registration_commit": REGISTRATION_COMMIT,
        "specification_sha256": file_sha256(specification_path),
        "implementation_sources": {
            name: _registration(path) for name, path in IMPLEMENTATION_SOURCES.items()
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
            f"evidence-sparsity source or artifact lock failed: {checks}"
        )
    return {"passed": True, "checks": checks, "lock": lock}


def _edge_tuple(graph: dict) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, graph["rank_edges"]))


def validate_sparsity_contract(specification: dict) -> dict:
    contract = specification["matched_nested_graph_contract"]
    all_rank_edges = tuple(combinations(range(8), 2))
    checks = []
    distance_by_count: dict[str, list[list[int]]] = {}
    for family in contract["families"]:
        graphs = family["graphs"]
        base = _edge_tuple(graphs["8"])
        deletion = next(
            edge
            for edge in sorted(base)
            if edge[1] - edge[0] == 3
            and graph_descriptor(tuple(item for item in base if item != edge))[
                "connected"
            ]
        )
        additions = tuple(
            edge
            for edge in all_rank_edges
            if edge[1] - edge[0] == 3 and edge not in base
        )[:2]
        expected = {
            "7": tuple(edge for edge in base if edge != deletion),
            "8": base,
            "9": tuple(sorted((*base, additions[0]))),
            "10": tuple(sorted((*base, *additions))),
        }
        family_passed = bool(
            tuple(family["deleted_edge_for_E7"]) == deletion
            and tuple(map(tuple, family["added_edges_for_E9_E10"])) == additions
        )
        rows = []
        for edge_count in map(str, contract["edge_counts_in_execution_order"]):
            graph = graphs[edge_count]
            edges = _edge_tuple(graph)
            descriptor = graph_descriptor(edges)
            row_passed = bool(
                edges == expected[edge_count]
                and descriptor["connected"]
                and descriptor["edge_count"] == int(edge_count)
                and descriptor["distance_multiset"] == graph["distance_multiset"]
                and descriptor["sorted_degree_sequence"]
                == graph["sorted_degree_sequence"]
                and descriptor["triangle_count"] == graph["triangle_count"]
                and descriptor["diameter"] == graph["diameter"]
            )
            family_passed = family_passed and row_passed
            distance_by_count.setdefault(edge_count, []).append(
                descriptor["distance_multiset"]
            )
            rows.append(
                {
                    "edge_count": int(edge_count),
                    "descriptor": descriptor,
                    "expected_edges": [list(edge) for edge in expected[edge_count]],
                    "passed": row_passed,
                }
            )
        checks.append(
            {
                "family_id": family["family_id"],
                "selected_deletion": list(deletion),
                "selected_additions": [list(edge) for edge in additions],
                "graphs": rows,
                "passed": family_passed,
            }
        )
    distance_matched = all(rows[0] == rows[1] for rows in distance_by_count.values())
    passed = all(check["passed"] for check in checks) and distance_matched
    if not passed:
        raise RuntimeError("registered evidence-sparsity graph contract failed")
    return {
        "passed": True,
        "distance_multisets_matched_between_families": distance_matched,
        "checks": checks,
    }


def _rank_edge_to_relation(
    protocol: RankingProtocol, edge: tuple[int, int]
) -> tuple[int, int]:
    return (
        protocol.true_order_high_to_low[edge[0]],
        protocol.true_order_high_to_low[edge[1]],
    )


def family_protocols(base: RankingProtocol, family: dict) -> dict[int, RankingProtocol]:
    if family["family_id"] == "liu_cycle_centered":
        base_protocol = base
    else:
        base_graph = {
            "graph_id": "balanced_branched_no_triangle",
            "rank_edges": family["graphs"]["8"]["rank_edges"],
        }
        base_protocol = protocol_for_graph(base, base_graph)
    deleted = _rank_edge_to_relation(base, tuple(family["deleted_edge_for_E7"]))
    additions = tuple(
        _rank_edge_to_relation(base, tuple(edge))
        for edge in family["added_edges_for_E9_E10"]
    )
    pairs = {
        7: tuple(
            pair for pair in base_protocol.support_pairs_higher_lower if pair != deleted
        ),
        8: base_protocol.support_pairs_higher_lower,
        9: (*base_protocol.support_pairs_higher_lower, additions[0]),
        10: (*base_protocol.support_pairs_higher_lower, *additions),
    }
    protocols = {}
    for edge_count, support_pairs in pairs.items():
        protocols[edge_count] = RankingProtocol(
            protocol_id=(
                base_protocol.protocol_id
                if edge_count == 8
                else f"liu-evidence-sparsity-v1-{family['family_id']}-E{edge_count}"
            ),
            item_labels=base.item_labels,
            true_order_high_to_low=base.true_order_high_to_low,
            support_pairs_higher_lower=tuple(support_pairs),
            support_blocks=base.support_blocks,
            query_blocks=base.query_blocks,
            human_targets={} if edge_count != 8 else base_protocol.human_targets,
        )
    return protocols


def build_nested_schedules(
    base_schedules: tuple[tuple[SupportTrial, ...], ...],
    base_protocol: RankingProtocol,
    protocols: dict[int, RankingProtocol],
    family: dict,
    family_index: int,
) -> dict[int, tuple[tuple[SupportTrial, ...], ...]]:
    additions = tuple(
        _rank_edge_to_relation(base_protocol, tuple(edge))
        for edge in family["added_edges_for_E9_E10"]
    )
    rank = {
        item: position
        for position, item in enumerate(base_protocol.true_order_high_to_low)
    }
    master_schedules = []
    for subject, schedule in enumerate(base_schedules):
        rng = np.random.default_rng(840000 + 1000 * family_index + subject)
        master = []
        for block_index in range(base_protocol.support_blocks):
            block = [trial for trial in schedule if trial.block_index == block_index]
            entries = [
                (index + 0.5, 0, index, trial) for index, trial in enumerate(block)
            ]
            for addition_index, (higher, lower) in enumerate(additions):
                magnitude = (rank[lower] - rank[higher]) / float(
                    base_protocol.n_items - 1
                )
                if rng.random() < 0.5:
                    left, right, signed = higher, lower, magnitude
                else:
                    left, right, signed = lower, higher, -magnitude
                key = float(rng.uniform(0.0, 8.0))
                trial = SupportTrial(
                    left_item=left,
                    right_item=right,
                    higher_item=higher,
                    lower_item=lower,
                    signed_magnitude=float(signed),
                    block_index=block_index,
                )
                entries.append((key, 1, addition_index, trial))
            master.extend(row[-1] for row in sorted(entries, key=lambda row: row[:3]))
        master_schedules.append(tuple(master))
    schedules = {}
    for edge_count, protocol in protocols.items():
        allowed = set(protocol.support_pairs_higher_lower)
        schedules[edge_count] = tuple(
            tuple(
                trial
                for trial in schedule
                if (trial.higher_item, trial.lower_item) in allowed
            )
            for schedule in master_schedules
        )
    return schedules


def build_nested_relation_gains(
    base_evaluator: FrozenFastWeightEvaluator,
    protocols: dict[int, RankingProtocol],
    family: dict,
    family_index: int,
) -> dict[int, tuple[dict[tuple[int, int], float], ...]]:
    if (
        base_evaluator.subject_relation_gains is None
        or base_evaluator.subject_encoding_states is None
    ):
        raise RuntimeError("nested stable admission requires subject encoding state")
    additions = tuple(
        _rank_edge_to_relation(base_evaluator.protocol, tuple(edge))
        for edge in family["added_edges_for_E9_E10"]
    )
    rank = base_evaluator.item_rank
    rng = np.random.default_rng(850000 + family_index)
    masters = []
    for subject, base_gains in enumerate(base_evaluator.subject_relation_gains):
        values = dict(base_gains)
        state = base_evaluator.subject_encoding_states[subject]
        for higher, lower in additions:
            probability = state.relation_reliability(
                higher, lower, rank[lower] - rank[higher]
            )
            values[(higher, lower)] = float(rng.random() < probability)
        masters.append(values)
    return {
        edge_count: tuple(
            {
                relation: master[relation]
                for relation in protocol.support_pairs_higher_lower
            }
            for master in masters
        )
        for edge_count, protocol in protocols.items()
    }


def _admission_hash(relation_gains: tuple[dict[tuple[int, int], float], ...]) -> str:
    payload = [
        [[list(relation), value] for relation, value in sorted(subject.items())]
        for subject in relation_gains
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def _relation_trials(
    schedule: tuple[SupportTrial, ...], relation: tuple[int, int]
) -> tuple[SupportTrial, ...]:
    return tuple(
        trial for trial in schedule if (trial.higher_item, trial.lower_item) == relation
    )


def nested_integrity(
    base_schedules: tuple[tuple[SupportTrial, ...], ...],
    schedules: dict[int, tuple[tuple[SupportTrial, ...], ...]],
    relation_gains: dict[int, tuple[dict[tuple[int, int], float], ...]],
    protocols: dict[int, RankingProtocol],
) -> dict:
    counts_pass = True
    common_trials_pass = schedules[8] == base_schedules
    common_admission_pass = True
    for edge_count, protocol in protocols.items():
        expected_relations = set(protocol.support_pairs_higher_lower)
        for subject, schedule in enumerate(schedules[edge_count]):
            counts_pass = counts_pass and len(schedule) == 4 * edge_count
            counts_pass = counts_pass and all(
                len(_relation_trials(schedule, relation)) == 4
                for relation in expected_relations
            )
            counts_pass = (
                counts_pass
                and {(trial.higher_item, trial.lower_item) for trial in schedule}
                == expected_relations
            )
            counts_pass = (
                counts_pass
                and set(relation_gains[edge_count][subject]) == expected_relations
            )
    edge_counts = sorted(protocols)
    for first, second in pairwise(edge_counts):
        common = set(protocols[first].support_pairs_higher_lower) & set(
            protocols[second].support_pairs_higher_lower
        )
        for subject in range(len(base_schedules)):
            for relation in common:
                common_trials_pass = common_trials_pass and _relation_trials(
                    schedules[first][subject], relation
                ) == _relation_trials(schedules[second][subject], relation)
                common_admission_pass = common_admission_pass and (
                    relation_gains[first][subject][relation]
                    == relation_gains[second][subject][relation]
                )
    passed = bool(counts_pass and common_trials_pass and common_admission_pass)
    return {
        "passed": passed,
        "support_counts_and_relation_multiplicity": bool(counts_pass),
        "nested_common_physical_trials": bool(common_trials_pass),
        "nested_common_admission": bool(common_admission_pass),
    }


def prepare_family_evaluators(
    specification: dict,
    family: dict,
    family_index: int,
    backbone,
    model_config,
) -> tuple[dict[int, FrozenFastWeightEvaluator], dict]:
    evaluation = specification["evaluation"]
    base = load_ranking_protocol(
        resolve_path(specification["fixed_task_contract"]["base_protocol"])
    )
    protocols = family_protocols(base, family)
    base_evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocols[8],
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode="stable_omission",
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
    )
    schedules = build_nested_schedules(
        base_evaluator.support_schedules,
        base,
        protocols,
        family,
        family_index,
    )
    relation_gains = build_nested_relation_gains(
        base_evaluator, protocols, family, family_index
    )
    integrity = nested_integrity(
        base_evaluator.support_schedules, schedules, relation_gains, protocols
    )
    evaluators = {}
    cue_identity = True
    state_identity = True
    for edge_count, protocol in protocols.items():
        if edge_count == 8:
            evaluator = base_evaluator
        else:
            evaluator = FrozenFastWeightEvaluator(
                backbone,
                model_config,
                protocol,
                cue_seed=int(evaluation["cue_seed"]),
                support_seed=int(evaluation["support_seed"]),
                cue_mode=str(evaluation["cue_mode"]),
                subject_encoding_mode="stable_omission",
                subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
            )
            cue_identity = cue_identity and np.array_equal(
                evaluator.cue_codes, base_evaluator.cue_codes
            )
            state_identity = state_identity and (
                evaluator.subject_encoding_states
                == base_evaluator.subject_encoding_states
            )
            evaluator.support_schedules = schedules[edge_count]
            evaluator.subject_encoding_states = base_evaluator.subject_encoding_states
            evaluator.subject_relation_gains = relation_gains[edge_count]
            evaluator.subject_trial_gains = tuple(
                tuple(
                    relation_gains[edge_count][subject][
                        (trial.higher_item, trial.lower_item)
                    ]
                    for trial in schedule
                )
                for subject, schedule in enumerate(schedules[edge_count])
            )
        evaluators[edge_count] = evaluator
    integrity.update(
        {
            "cue_codes_identical_across_densities": bool(cue_identity),
            "subject_encoding_states_identical_across_densities": bool(state_identity),
            "schedule_hashes": {
                str(edge_count): _schedule_hash(evaluator)
                for edge_count, evaluator in evaluators.items()
            },
            "admission_hashes": {
                str(edge_count): _admission_hash(relation_gains[edge_count])
                for edge_count in sorted(relation_gains)
            },
        }
    )
    integrity["passed"] = bool(integrity["passed"] and cue_identity and state_identity)
    return evaluators, integrity


def _bootstrap_seed(seed: int, family_index: int, edge_count: int) -> int:
    if edge_count == 8:
        return 760000 + 100 * seed + 1 if family_index == 1 else 530000 + 100 * seed + 1
    return 880000 + 1000 * family_index + 100 * seed + edge_count


def _masked_subject_mean(values: np.ndarray) -> np.ndarray:
    finite = np.sum(np.isfinite(values), axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        finite,
        out=np.full(values.shape[1], np.nan),
        where=finite > 0,
    )


def _metric_projection(metrics: dict) -> dict:
    return {key: metrics[key] for key in CORE_METRIC_KEYS}


def evaluate_prepared_cell(
    specification: dict,
    seed: int,
    family: dict,
    family_index: int,
    edge_count: int,
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    runtime: dict,
    source_validation: dict,
    graph_validation: dict,
    family_integrity: dict,
    source_metrics: dict | None,
    source_schedule_hash: str | None,
) -> dict:
    evaluation = specification["evaluation"]
    protocol = evaluator.protocol
    bootstrap_seed = _bootstrap_seed(seed, family_index, edge_count)
    rng = np.random.default_rng(bootstrap_seed)
    counts = bootstrap_counts(
        rng,
        int(evaluation["bootstrap_samples"]),
        int(evaluation["subjects_per_cell"]),
    )
    interval = float(evaluation["bootstrap_interval"])
    from .assembly_trajectory import build_complete_graph_geometry

    geometry = build_complete_graph_geometry(protocol)
    relations = tuple(protocol.support_pairs_higher_lower)
    learned_mask = np.asarray(
        [pair in protocol.learned_pairs for pair in geometry.pairs]
    )
    query_schedules = tuple(
        _ordered_pairs(protocol.n_items) for _ in range(evaluator.config.bs)
    )
    before = _tensor_hashes(evaluator.net)
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    loo_fast_weights = _build_fast_weight_loo(evaluator, relations)
    intact_trace = build_access_trace(evaluator, local, dual_access=True)
    loo_traces = [
        build_access_trace(
            evaluator, local, dual_access=True, zero_relations=frozenset((relation,))
        )
        for relation in relations
    ]
    intact_bundle = _query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        query_schedules,
        local_off=False,
        global_off=False,
        shuffled_indices=None,
    )
    a_off_bundle = _query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        query_schedules,
        local_off=True,
        global_off=False,
        shuffled_indices=None,
    )
    p_off_bundle = _query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        query_schedules,
        local_off=False,
        global_off=True,
        shuffled_indices=None,
    )
    loo_global_bundles = [
        _query_pass(
            evaluator,
            local,
            loo_fast_weights[index],
            loo_traces[index].state,
            query_schedules,
            local_off=True,
            global_off=False,
            shuffled_indices=None,
        )
        for index in range(len(relations))
    ]
    loo_local_bundles = [
        _query_pass(
            evaluator,
            local,
            intact_fast_weights,
            loo_traces[index].state,
            query_schedules,
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

    def raw(condition: str, group: str) -> np.ndarray:
        return np.asarray(
            conditions[condition]["raw_subject"]["correct_probability"][group]
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
        bundle_logits(intact_bundle, query_schedules),
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
    retained = _retained_mask(evaluator, relations)
    exact_probability = {
        name: _sigmoid(
            field * geometry.true_sign[None] / float(evaluation["temperature"])
        )
        for name, field in fields.items()
    }
    relation_indices = [
        geometry.pairs.index(tuple(sorted(relation))) for relation in relations
    ]
    learned_probability = exact_probability["intact"][:, relation_indices].T
    dependencies = {
        "global_dependence_all_pairs": np.mean(
            exact_probability["intact"] - exact_probability["P_off_a_on"], axis=1
        ),
        "local_dependence_all_pairs": np.mean(
            exact_probability["intact"] - exact_probability["a_off"], axis=1
        ),
    }
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
            "retained_counts_per_subject": _json_values(np.sum(retained, axis=0)),
            "omitted_counts_per_subject": _json_values(np.sum(~retained, axis=0)),
            "retained_correct_probability": summarize_subjects(
                _masked_subject_mean(np.where(retained, learned_probability, np.nan)),
                counts,
                interval=interval,
            ),
            "omitted_correct_probability": summarize_subjects(
                _masked_subject_mean(np.where(~retained, learned_probability, np.nan)),
                counts,
                interval=interval,
            ),
        },
        "sampled_behavior": behavior,
        "sampled_accuracy_bootstrap": sampled_accuracy,
        "serial_position_endpoint": serial_position_endpoint(behavior, protocol),
        "density_dependencies": {
            name: {
                "raw_subject": _json_values(values),
                "summary": summarize_subjects(values, counts, interval=interval),
            }
            for name, values in dependencies.items()
        },
    }
    presentation = _presentation_invariance(
        evaluator, local, intact_trace.natural_scalars
    )
    schedule_hash = _schedule_hash(evaluator)
    e8_metrics_exact = bool(
        edge_count != 8
        or _metric_projection(metrics) == _metric_projection(source_metrics or {})
    )
    e8_schedule_exact = bool(edge_count != 8 or schedule_hash == source_schedule_hash)
    after = _tensor_hashes(evaluator.net)
    integrity = {
        "source_validation_passed": bool(source_validation["passed"]),
        "graph_validation_passed": bool(graph_validation["passed"]),
        "nested_schedule_and_admission": family_integrity,
        "E8_metric_projection_exact": e8_metrics_exact,
        "E8_schedule_hash_exact": e8_schedule_exact,
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
        "primary_values_finite": _finite_primary(metrics),
    }
    integrity["all_passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["graph_validation_passed"]
        and family_integrity["passed"]
        and integrity["E8_metric_projection_exact"]
        and integrity["E8_schedule_hash_exact"]
        and integrity["bounded_gpu_runtime"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and integrity["local_off_global_logit_max_abs_error"] <= 1e-6
        and integrity["support_write_reversal_max_abs_error"] <= 1e-7
        and integrity["query_key_reversal_max_abs_error"] <= 1e-7
        and integrity["primary_values_finite"]
    )
    decision = topology_within_cell_decision(metrics, integrity)
    exact_pass = bool(
        decision["flags"]["exact_local_compression"]
        and exact["tensor_state_max_abs_error"] <= 1e-12
        and exact["ledger_tensor_state_max_abs_error"] <= 1e-12
        and exact["all_query_raw_read_max_abs_error"] <= 1e-12
    )
    decision["flags"]["exact_local_compression"] = exact_pass
    decision["all_eight_primary_links_pass"] = all(decision["flags"].values())
    return {
        "family_id": family["family_id"],
        "edge_count": edge_count,
        "protocol_id": protocol.protocol_id,
        "rank_edges": family["graphs"][str(edge_count)]["rank_edges"],
        "item_edges_higher_lower": [list(relation) for relation in relations],
        "support_schedule_sha256": schedule_hash,
        "admission_sha256": family_integrity["admission_hashes"][str(edge_count)],
        "bootstrap_seed": bootstrap_seed,
        "learned_query_fraction": edge_count / 28.0,
        "metrics": metrics,
        "integrity": integrity,
        "decision": decision,
    }


def density_trend_metrics(
    cells: dict[str, dict],
    seed: int,
    family_index: int,
    specification: dict,
) -> dict:
    edge_counts = np.asarray(
        specification["matched_nested_graph_contract"][
            "edge_counts_in_execution_order"
        ],
        dtype=np.float64,
    )
    centered = edge_counts - np.mean(edge_counts)
    denominator = float(centered @ centered)
    counts = bootstrap_counts(
        np.random.default_rng(860000 + 1000 * family_index + 100 * seed),
        int(specification["evaluation"]["bootstrap_samples"]),
        int(specification["evaluation"]["subjects_per_cell"]),
    )
    interval = float(specification["evaluation"]["bootstrap_interval"])
    summaries = {}
    raw_slopes = {}
    for name in (
        "global_dependence_all_pairs",
        "local_dependence_all_pairs",
    ):
        values = np.asarray(
            [
                cells[str(edge_count)]["metrics"]["density_dependencies"][name][
                    "raw_subject"
                ]
                for edge_count in edge_counts.astype(int)
            ],
            dtype=np.float64,
        )
        slopes = centered @ values / denominator
        raw_slopes[name] = _json_values(slopes)
        summaries[name] = summarize_subjects(slopes, counts, interval=interval)
    flags = {
        "global_dependence_decreases": bool(
            summaries["global_dependence_all_pairs"]["bootstrap"]["upper"] < 0.0
        ),
        "local_dependence_increases": bool(
            summaries["local_dependence_all_pairs"]["bootstrap"]["lower"] > 0.0
        ),
    }
    return {
        "bootstrap_seed": 860000 + 1000 * family_index + 100 * seed,
        "raw_subject_slopes": raw_slopes,
        "summary": summaries,
        "flags": flags,
        "bidirectional_prediction_passed": all(flags.values()),
    }


def cross_cell_decision(
    seeds: dict,
    family_ids: list[str],
    edge_counts: list[int],
    mandatory_seeds: list[int],
) -> dict:
    cells = [
        seeds[str(seed)]["families"][family_id]["densities"][str(edge_count)]
        for family_id in family_ids
        for edge_count in edge_counts
        for seed in mandatory_seeds
    ]
    if not all(cell["decision"]["interpretable"] for cell in cells):
        outcome = "NONINTERPRETABLE_EXECUTION"
        density_passes = None
        heterogeneous = None
    elif not all(cell["decision"]["competence_passed"] for cell in cells):
        outcome = "SPARSITY_COMPETENCE_NOT_ESTABLISHED"
        density_passes = None
        heterogeneous = None
    else:
        density_passes = {
            str(edge_count): all(
                seeds[str(seed)]["families"][family_id]["densities"][str(edge_count)][
                    "decision"
                ]["all_eight_primary_links_pass"]
                for family_id in family_ids
                for seed in mandatory_seeds
            )
            for edge_count in edge_counts
        }
        links = next(iter(cells))["decision"]["flags"]
        heterogeneous = any(
            0
            < sum(
                seeds[str(seed)]["families"][family_id]["densities"][str(edge_count)][
                    "decision"
                ]["flags"][link]
                for family_id in family_ids
                for seed in mandatory_seeds
            )
            < len(family_ids) * len(mandatory_seeds)
            for edge_count in edge_counts
            for link in links
        )
        if all(density_passes.values()):
            outcome = "LIU_SPARSITY_MECHANISM_TRANSPORTED"
        elif density_passes.get("8") and not any(
            density_passes[str(edge_count)]
            for edge_count in edge_counts
            if edge_count != 8
        ):
            outcome = "FUNCTIONAL_ASYMMETRY_NOT_SPARSITY_TRANSPORTED"
        else:
            outcome = "SPARSITY_DEPENDENT_OR_UNRESOLVED"
    trend_rows = [
        seeds[str(seed)]["families"][family_id]["density_trend"]
        for family_id in family_ids
        for seed in mandatory_seeds
    ]
    trend_passes = [row["bidirectional_prediction_passed"] for row in trend_rows]
    if all(trend_passes):
        trend_outcome = "BIDIRECTIONAL_ALLOCATION_TREND_REPLICATED"
    elif any(any(row["flags"].values()) for row in trend_rows):
        trend_outcome = "PARTIAL_OR_HETEROGENEOUS_ALLOCATION_TREND"
    else:
        trend_outcome = "NO_REGISTERED_ALLOCATION_TREND"
    return {
        "outcome": outcome,
        "density_passes": density_passes,
        "heterogeneous_across_families_or_backbones": heterogeneous,
        "secondary_density_allocation_outcome": trend_outcome,
    }


def _source_cell(source_results: dict[str, dict], family_id: str, seed: int) -> dict:
    if family_id == "liu_cycle_centered":
        return source_results["presentation"]["seeds"][str(seed)]["conditions"][
            "blockwise_random"
        ]
    return source_results["topology"]["seeds"][str(seed)]["graphs"][
        "balanced_branched_no_triangle"
    ]


def evaluate(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    runtime = require_formal_runtime()
    specification = load_json(specification_path)
    source_validation = validate_sources(specification_path, lock_path)
    graph_validation = validate_sparsity_contract(specification)
    source_results = {
        "presentation": load_json(
            resolve_path(
                specification["registered_sources"]["presentation_order_result"]["path"]
            )
        ),
        "topology": load_json(
            resolve_path(specification["registered_sources"]["topology_result"]["path"])
        ),
    }
    evaluation = specification["evaluation"]
    families = specification["matched_nested_graph_contract"]["families"]
    edge_counts = specification["matched_nested_graph_contract"][
        "edge_counts_in_execution_order"
    ]
    seeds = {}
    for seed in specification["development_backbones"]["mandatory_seeds"]:
        artifacts = specification["development_backbones"]["artifacts"][str(seed)]
        backbone, model_config, checkpoint = load_retro_checkpoint(
            resolve_path(artifacts["checkpoint"]["path"]),
            int(evaluation["subjects_per_cell"]),
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        gain = load_json(resolve_path(artifacts["gain"]["path"]))
        local = ConjunctiveLocalTrace(model_config.cs)
        with torch.no_grad():
            local.raw_gain.fill_(float(gain["raw_lambda_L"]))
        family_results = {}
        for family_index, family in enumerate(families, start=1):
            evaluators, family_integrity = prepare_family_evaluators(
                specification,
                family,
                family_index,
                backbone,
                model_config,
            )
            source = _source_cell(source_results, family["family_id"], int(seed))
            family_integrity["E8_source_schedule_sha256"] = source[
                "support_schedule_sha256"
            ]
            densities = {}
            for edge_count in edge_counts:
                densities[str(edge_count)] = evaluate_prepared_cell(
                    specification,
                    int(seed),
                    family,
                    family_index,
                    int(edge_count),
                    evaluators[int(edge_count)],
                    local,
                    runtime,
                    source_validation,
                    graph_validation,
                    family_integrity,
                    source["metrics"] if int(edge_count) == 8 else None,
                    source["support_schedule_sha256"] if int(edge_count) == 8 else None,
                )
            family_results[family["family_id"]] = {
                "family_integrity": family_integrity,
                "densities": densities,
                "density_trend": density_trend_metrics(
                    densities, int(seed), family_index, specification
                ),
            }
        seeds[str(seed)] = {
            "seed": int(seed),
            "checkpoint": asdict(checkpoint),
            "gain_path": artifacts["gain"]["path"],
            "lambda_L": float(local.gain.detach().cpu()),
            "families": family_results,
        }
    family_ids = [family["family_id"] for family in families]
    decision = cross_cell_decision(
        seeds,
        family_ids,
        [int(edge_count) for edge_count in edge_counts],
        [
            int(seed)
            for seed in specification["development_backbones"]["mandatory_seeds"]
        ],
    )
    return {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "execution_runtime": runtime,
        "source_validation": source_validation,
        "graph_validation": graph_validation,
        "seeds": seeds,
        "decision": decision,
        "registered_primary_links": specification["primary_links"],
        "registered_density_allocation_prediction": specification[
            "registered_density_allocation_prediction"
        ],
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
