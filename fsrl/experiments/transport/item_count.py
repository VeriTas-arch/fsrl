"""Registered Liu item-count transport evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, pairwise, permutations
from pathlib import Path
from typing import cast

import numpy as np
import torch

from fsrl.analysis.behavioral import (
    analyze_sampled_query_policy,
    kendall_tau_positions,
)
from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
    kendall_tau_scores,
)
from fsrl.analysis.policy import bundle_logits, margin_fields
from fsrl.analysis.relational_transport import condition_metrics, finite_primary
from fsrl.analysis.statistics import (
    bootstrap_counts,
    finite_column_mean,
    json_values,
    stable_sigmoid,
    summarize_difference,
    summarize_subjects,
)
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.evaluation.frozen_fast_weight import (
    FrozenFastWeightEvaluator,
    load_frozen_retro_checkpoint,
    retained_relation_mask,
)
from fsrl.evaluation.local_access import (
    measure_presentation_invariance,
    readout_dual_access_query_conditions,
)
from fsrl.evaluation.local_ledger import (
    reconstruct_local_ledger,
    support_schedule_hash,
)
from fsrl.evaluation.metrics import count_circular_triads
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.semantic_contract import (
    evaluate_semantic_contract,
    load_semantic_contract,
)
from fsrl.infra.study_registry import (
    canonical_file_registration,
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol, ordered_pairs
from fsrl.tasks.transport_graph import graph_descriptor

DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/liu_item_count_transport_v1.json"
)
ROOT = REPO_ROOT
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/liu_item_count_transport_v1.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/liu_item_count_transport_v1.json")
DECISION_CONTRACT_PATH = (
    Path(__file__).with_name("contracts") / "item_count_within_cell_v1.json"
)
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/item_count_transport.py",
    "runtime_entrypoint": "fsrl/item_count_runtime.py",
    "tests": "tests/test_item_count_transport.py",
}
REGISTRATION_COMMIT = "9577d381c205516a9eac2a82a935d4ddcaafca19"


def write_implementation_lock(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    lock = {
        "schema_version": 1,
        "experiment_id": "liu-item-count-transport-v1",
        "implementation_status": "frozen_before_any_non_eight_item_model_evaluation",
        "registration_commit": REGISTRATION_COMMIT,
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
        raise RuntimeError(f"item-count source or artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _exact_wasserstein(first: list[Fraction], second: list[Fraction]) -> Fraction:
    support = sorted(set(first + second))
    value = Fraction(0)
    for lower, upper in pairwise(support):
        first_cdf = Fraction(sum(item <= lower for item in first), len(first))
        second_cdf = Fraction(sum(item <= lower for item in second), len(second))
        value += abs(first_cdf - second_cdf) * (upper - lower)
    return value


def enumerate_matched_cycle(
    n_items: int, target: tuple[Fraction, ...]
) -> tuple[tuple[tuple[int, int], ...], Fraction, int]:
    best_distance = None
    best_edges = None
    minimizers = 0
    distance_cache: dict[tuple[Fraction, ...], Fraction] = {}
    for remainder in permutations(range(1, n_items)):
        if remainder[0] >= remainder[-1]:
            continue
        cycle = (0, *remainder)
        edges = tuple(
            sorted(
                tuple(sorted((cycle[index], cycle[(index + 1) % n_items])))
                for index in range(n_items)
            )
        )
        distances = tuple(
            sorted(Fraction(second - first, n_items - 1) for first, second in edges)
        )
        distance = distance_cache.get(distances)
        if distance is None:
            distance = _exact_wasserstein(list(distances), list(target))
            distance_cache[distances] = distance
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_edges = edges
            minimizers = 1
        elif distance == best_distance:
            minimizers += 1
            if best_edges is None:
                raise RuntimeError("cycle minimizer state is incomplete")
            best_edges = min(best_edges, edges)
    if best_distance is None or best_edges is None:
        raise RuntimeError("could not enumerate a Hamiltonian cycle")
    return (
        cast(tuple[tuple[int, int], ...], best_edges),
        best_distance,
        minimizers,
    )


def _rank_edges(protocol: RankingProtocol) -> tuple[tuple[int, int], ...]:
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    return cast(
        tuple[tuple[int, int], ...],
        tuple(
            sorted(
                tuple(sorted((rank[higher], rank[lower])))
                for higher, lower in protocol.support_pairs_higher_lower
            )
        ),
    )


def protocol_for_size(base: RankingProtocol, graph: dict) -> RankingProtocol:
    n_items = int(graph["n_items"])
    if n_items == 8:
        return base
    labels = tuple(graph["item_labels"])
    label_to_item = {label: item for item, label in enumerate(labels)}
    true_order = tuple(
        label_to_item[label] for label in graph["true_order_high_to_low"]
    )
    relations = tuple(
        (true_order[higher], true_order[lower])
        for higher, lower in map(tuple, graph["rank_edges"])
    )
    return RankingProtocol(
        protocol_id=f"liu-item-count-transport-v1-N{n_items}",
        item_labels=labels,
        true_order_high_to_low=true_order,
        support_pairs_higher_lower=relations,
        support_blocks=int(base.support_blocks),
        query_blocks=int(base.query_blocks),
        human_targets={},
    )


def validate_graph_contract(specification: dict) -> dict:
    contract = specification["size_matched_graph_contract"]
    base = load_ranking_protocol(
        resolve_path(specification["registered_sources"]["base_liu_protocol"]["path"])
    )
    target = tuple(
        Fraction(value) for value in contract["graphs"][1]["normalized_rank_distances"]
    )
    checks = []
    for graph in contract["graphs"]:
        n_items = int(graph["n_items"])
        protocol = protocol_for_size(base, graph)
        edges = tuple(map(tuple, graph["rank_edges"]))
        descriptor = graph_descriptor(edges, n_items)
        if n_items == 8:
            expected_edges = _rank_edges(base)
            expected_distance = Fraction(0)
            minimizers = 1
        else:
            expected_edges, expected_distance, minimizers = enumerate_matched_cycle(
                n_items, target
            )
        expected_normalized = [
            str(Fraction(distance, n_items - 1))
            for distance in descriptor["distance_multiset"]
        ]
        row_passed = bool(
            edges == expected_edges
            and descriptor["connected"]
            and descriptor["edge_count"] == n_items
            and descriptor["sorted_degree_sequence"] == [2] * n_items
            and descriptor["distance_multiset"] == graph["rank_distance_multiset"]
            and expected_normalized == graph["normalized_rank_distances"]
            and expected_distance == Fraction(graph["wasserstein_to_N8_target"])
            and minimizers == int(graph["number_of_exact_minimizers"])
            and protocol.support_trials == int(graph["support_trials"])
            and protocol.query_trials == int(graph["query_trials"])
            and protocol.n_items * (protocol.n_items - 1) // 2
            == int(graph["query_pairs"])
        )
        checks.append(
            {
                "size_id": graph["size_id"],
                "descriptor": descriptor,
                "enumerated_edges": [list(edge) for edge in expected_edges],
                "wasserstein_to_N8_target": str(expected_distance),
                "number_of_exact_minimizers": minimizers,
                "passed": row_passed,
            }
        )
    passed = all(check["passed"] for check in checks)
    if not passed:
        raise RuntimeError(f"registered item-count graph contract failed: {checks}")
    return {"passed": True, "checks": checks}


def _admission_hash(evaluator: FrozenFastWeightEvaluator) -> str:
    payload = [
        [[list(relation), value] for relation, value in sorted(subject.items())]
        for subject in evaluator.subject_relation_gains or ()
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def _cue_hash(evaluator: FrozenFastWeightEvaluator) -> str:
    return hashlib.sha256(evaluator.cue_codes.tobytes()).hexdigest()


def validate_size_interface(evaluator: FrozenFastWeightEvaluator) -> dict:
    protocol = evaluator.protocol
    expected_relations = set(protocol.support_pairs_higher_lower)
    counts = True
    magnitude = True
    orientations = True
    admission = True
    for subject, schedule in enumerate(evaluator.support_schedules):
        counts = counts and len(schedule) == protocol.support_trials
        for relation in expected_relations:
            counts = (
                counts
                and sum(
                    (trial.higher_item, trial.lower_item) == relation
                    for trial in schedule
                )
                == protocol.support_blocks
            )
        for trial in schedule:
            relation = (trial.higher_item, trial.lower_item)
            counts = counts and relation in expected_relations
            rank_gap = (
                evaluator.item_rank[trial.lower_item]
                - evaluator.item_rank[trial.higher_item]
            )
            magnitude = magnitude and np.isclose(
                abs(trial.signed_magnitude), rank_gap / float(protocol.n_items - 1)
            )
            orientations = orientations and {trial.left_item, trial.right_item} == set(
                relation
            )
            if evaluator.subject_relation_gains is not None:
                admission = (
                    admission and relation in evaluator.subject_relation_gains[subject]
                )
    cue_shape = evaluator.cue_codes.shape == (
        evaluator.config.bs,
        protocol.n_items,
        evaluator.config.cs,
    )
    state_shape = bool(
        evaluator.subject_encoding_states is not None
        and all(
            len(state.item_salience) == protocol.n_items
            for state in evaluator.subject_encoding_states
        )
    )
    passed = bool(
        counts
        and magnitude
        and orientations
        and admission
        and cue_shape
        and state_shape
    )
    return {
        "passed": passed,
        "support_counts_and_multiplicity": bool(counts),
        "normalized_signed_magnitudes": bool(magnitude),
        "left_right_orientation_valid": bool(orientations),
        "admission_relations_complete": bool(admission),
        "cue_shape_valid": bool(cue_shape),
        "subject_state_shape_valid": bool(state_shape),
        "support_schedule_sha256": support_schedule_hash(evaluator),
        "cue_codes_sha256": _cue_hash(evaluator),
        "admission_sha256": _admission_hash(evaluator),
    }


def validate_n8_evaluator_interface(
    variable: FrozenFastWeightEvaluator,
    frozen: FrozenFastWeightEvaluator,
) -> dict:
    checks = {
        "cue_codes_exact": np.array_equal(variable.cue_codes, frozen.cue_codes),
        "support_schedules_exact": variable.support_schedules
        == frozen.support_schedules,
        "subject_encoding_states_exact": (
            variable.subject_encoding_states == frozen.subject_encoding_states
        ),
        "subject_relation_gains_exact": (
            variable.subject_relation_gains == frozen.subject_relation_gains
        ),
        "subject_trial_gains_exact": (
            variable.subject_trial_gains == frozen.subject_trial_gains
        ),
        "item_rank_exact": variable.item_rank == frozen.item_rank,
        "test_time_value_exact": variable.test_time_value == frozen.test_time_value,
    }
    return {"passed": all(checks.values()), **checks}


def constructive_metrics_generic(
    intact_field: np.ndarray,
    global_field: np.ndarray,
    geometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    intact_gradient = gradient_energy_fraction(intact_field, geometry)
    global_gradient = gradient_energy_fraction(global_field, geometry)
    potentials = hodge_potentials(intact_field, geometry)
    true = np.broadcast_to(geometry.true_potential, potentials.shape)
    hodge_tau = kendall_tau_scores(potentials, true)
    n_items = len(geometry.true_potential)
    n_triads = len(tuple(combinations(range(n_items), 3)))
    transitivity = []
    for row in intact_field:
        winners = {
            pair: pair[0] if row[index] > 0.0 else pair[1]
            for index, pair in enumerate(geometry.pairs)
        }
        circular = count_circular_triads(winners, n_items)
        transitivity.append(1.0 - circular / n_triads)
    raw = {
        "intact_gradient_energy_fraction": intact_gradient,
        "a_off_gradient_energy_fraction": global_gradient,
        "intact_transitive_triplet_fraction": np.asarray(transitivity),
        "intact_hodge_order_kendall_tau_to_true": hodge_tau,
    }
    return {
        "summary": {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in raw.items()
        },
        "raw_subject": {name: json_values(values) for name, values in raw.items()},
    }


def relation_loo_metrics_generic(
    intact: np.ndarray,
    loo: np.ndarray,
    relations: tuple[tuple[int, int], ...],
    geometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    influence = intact[None] - loo
    remote = np.empty((len(relations), intact.shape[0]), dtype=np.float64)
    third_party = np.full_like(remote, np.nan)
    intact_potential = hodge_potentials(intact, geometry)
    n_items = len(geometry.true_potential)
    for relation_index, relation in enumerate(relations):
        endpoints = set(relation)
        remote_mask = np.asarray(
            [not endpoints.intersection(pair) for pair in geometry.pairs], dtype=bool
        )
        remote[relation_index] = np.mean(
            np.abs(influence[relation_index][:, remote_mask]), axis=1
        )
        delta = intact_potential - hodge_potentials(loo[relation_index], geometry)
        denominator = np.sum(delta * delta, axis=1)
        third_items = np.asarray(
            [item for item in range(n_items) if item not in endpoints], dtype=np.int64
        )
        relational = delta[:, third_items] - np.mean(
            delta[:, third_items], axis=1, keepdims=True
        )
        numerator = np.sum(relational * relational, axis=1)
        third_party[relation_index] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-14,
        )
    subject_remote = np.mean(remote, axis=0)
    finite = np.sum(np.isfinite(third_party), axis=0)
    subject_third = np.divide(
        np.nansum(third_party, axis=0),
        finite,
        out=np.full(intact.shape[0], np.nan),
        where=finite > 0,
    )
    return {
        "summary": {
            "remote_absolute": summarize_subjects(
                subject_remote, counts, interval=interval
            ),
            "third_party_relational": summarize_subjects(
                subject_third, counts, interval=interval
            ),
        },
        "raw_subject": {
            "remote_absolute": json_values(subject_remote),
            "third_party_relational": json_values(subject_third),
        },
        "raw_relation_subject": {
            "remote_absolute": json_values(remote),
            "third_party_relational": json_values(third_party),
        },
    }


def individualized_metrics_generic(
    behavior: dict,
    rng: np.random.Generator,
    samples: int,
    n_items: int,
) -> dict:
    eligible = [row for row in behavior["subjects"] if row["overall_accuracy"] >= 0.5]
    analysis = [row for row in eligible if row["ranking_class"] != "correct"]
    stable_pair_counts = np.asarray(
        [row["stable_error_pair_counts"]["80"] for row in analysis],
        dtype=np.float64,
    )
    stable = (stable_pair_counts > 0).astype(np.float64)
    pair_density = stable_pair_counts / float(n_items * (n_items - 1) // 2)
    if len(stable):
        stable_counts = bootstrap_counts(rng, samples, len(stable))
        stable_summary = summarize_subjects(stable, stable_counts, interval=0.95)
        density_summary = summarize_subjects(pair_density, stable_counts, interval=0.95)
    else:
        empty_counts = np.zeros((samples, 0))
        stable_summary = summarize_subjects(
            np.asarray([], dtype=np.float64), empty_counts, interval=0.95
        )
        density_summary = summarize_subjects(
            np.asarray([], dtype=np.float64), empty_counts, interval=0.95
        )
    orders = [row["subjective_order_high_to_low"] for row in analysis]
    if len(orders) >= 2:
        positions = []
        for order in orders:
            row = np.empty(n_items, dtype=np.int64)
            row[np.asarray(order, dtype=np.int64)] = np.arange(n_items)
            positions.append(row)
        positions = np.asarray(positions)
        matrix = np.eye(len(positions), dtype=np.float64)
        for first, second in combinations(range(len(positions)), 2):
            value = kendall_tau_positions(positions[first], positions[second])
            matrix[first, second] = value
            matrix[second, first] = value
        point = float(np.mean(matrix[np.triu_indices(len(positions), 1)]))
        tau_counts = bootstrap_counts(rng, samples, len(positions))
        quadratic = np.einsum(
            "bi,ij,bj->b", tau_counts, matrix, tau_counts, optimize=True
        )
        diagonal = np.sum(tau_counts, axis=1)
        draws = (quadratic - diagonal) / (len(positions) * (len(positions) - 1))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        tau = {
            "subjects": len(positions),
            "mean": point,
            "bootstrap": {
                "mean": float(np.mean(draws)),
                "lower": float(lower),
                "upper": float(upper),
            },
        }
    else:
        tau = {
            "subjects": len(orders),
            "mean": None,
            "bootstrap": {"mean": None, "lower": None, "upper": None},
        }
    return {
        "eligible_subjects": len(eligible),
        "eligible_noncorrect_subjects": len(analysis),
        "stable_error_80_prevalence": stable_summary,
        "stable_error_80_pair_density": density_summary,
        "mean_pairwise_kendall_tau": tau,
    }


def serial_position_endpoint_generic(behavior: dict, protocol: RankingProtocol) -> dict:
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    totals = np.zeros(protocol.n_items, dtype=np.float64)
    counts = np.zeros(protocol.n_items, dtype=np.float64)
    for row in behavior["pairs"]:
        value = float(row["mean_accuracy_all"])
        for item in row["pair"]:
            totals[rank[item]] += value
            counts[rank[item]] += 1.0
    profile = totals / counts
    interior = float(np.mean(profile[1:-1]))
    return {
        "profile_high_to_low": json_values(profile),
        "interior_mean": interior,
        "mean_endpoint_contrast": float(np.mean(profile[[0, -1]]) - interior),
        "minimum_endpoint_advantage": float(min(profile[0], profile[-1]) - interior),
    }


def _distance_summaries(
    behavior: dict, counts: np.ndarray, interval: float, n_items: int
) -> dict:
    distances = np.arange(1, n_items, dtype=np.float64)
    symbolic = np.asarray(
        [
            np.polyfit(
                distances,
                [row["distance_accuracy"][str(int(value))] for value in distances],
                1,
            )[0]
            for row in behavior["subjects"]
        ],
        dtype=np.float64,
    )
    normalized = symbolic * float(n_items - 1)
    return {
        "symbolic_distance_slope": summarize_subjects(
            symbolic, counts, interval=interval
        ),
        "normalized_distance_slope": summarize_subjects(
            normalized, counts, interval=interval
        ),
        "raw_subject_symbolic_distance_slope": json_values(symbolic),
        "raw_subject_normalized_distance_slope": json_values(normalized),
    }


def _legacy_metric_projection(metrics: dict) -> dict:
    projected = copy.deepcopy(metrics)
    projected["individualized"].pop("stable_error_80_pair_density")
    return projected


@lru_cache(maxsize=1)
def _decision_contract() -> dict:
    return load_semantic_contract(DECISION_CONTRACT_PATH)


def within_cell_decision(metrics: dict, integrity: dict) -> dict:
    evaluation = evaluate_semantic_contract(
        {"metrics": metrics, "integrity": integrity}, _decision_contract()
    )
    return {
        "interpretable": evaluation["interpretable"],
        **evaluation["decision_outputs"],
        "flags": evaluation["flags"],
    }


def _bootstrap_seed(seed: int, size_index: int, n_items: int) -> int:
    if n_items == 8:
        return 760000 + 100 * seed + 1
    return 980000 + 100 * seed + size_index


def evaluate_cell(
    specification: dict,
    seed: int,
    graph: dict,
    size_index: int,
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    runtime: dict,
    source_validation: dict,
    graph_validation: dict,
    size_interface: dict,
    n8_interface: dict,
    source_cell: dict | None,
) -> dict:
    evaluation = specification["evaluation"]
    protocol = evaluator.protocol
    n_items = protocol.n_items
    bootstrap_seed = _bootstrap_seed(seed, size_index, n_items)
    rng = np.random.default_rng(bootstrap_seed)
    counts = bootstrap_counts(
        rng,
        int(evaluation["bootstrap_samples"]),
        int(evaluation["subjects_per_size_and_backbone"]),
    )
    interval = float(evaluation["bootstrap_interval"])
    geometry = build_complete_graph_geometry(protocol)
    relations = tuple(protocol.support_pairs_higher_lower)
    learned_mask = np.asarray(
        [pair in protocol.learned_pairs for pair in geometry.pairs]
    )
    query_schedules = tuple(ordered_pairs(n_items) for _ in range(evaluator.config.bs))
    before = tensor_hashes(evaluator.net)
    readout = readout_dual_access_query_conditions(evaluator, local, query_schedules)
    intact_trace = readout["intact_trace"]
    condition_bundles = readout["condition_bundles"]
    intact_bundle = condition_bundles["intact"]
    a_off_bundle = condition_bundles["a_off"]
    fields = {
        name: margin_fields(bundle, n_items)
        for name, bundle in condition_bundles.items()
    }
    loo_global_fields = np.asarray(
        [margin_fields(bundle, n_items) for bundle in readout["global_loo_bundles"]]
    )
    loo_local_fields = np.asarray(
        [margin_fields(bundle, n_items) for bundle in readout["local_loo_bundles"]]
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
    global_loo = relation_loo_metrics_generic(
        fields["a_off"], loo_global_fields, relations, geometry, counts, interval
    )
    local_loo = relation_loo_metrics_generic(
        fields["P_off_a_on"],
        loo_local_fields,
        relations,
        geometry,
        counts,
        interval,
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
    retained = retained_relation_mask(evaluator, relations)
    exact_probability = {
        name: stable_sigmoid(
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
        "constructive": constructive_metrics_generic(
            fields["intact"], fields["a_off"], geometry, counts, interval
        ),
        "individualized": individualized_metrics_generic(
            behavior,
            rng,
            int(evaluation["bootstrap_samples"]),
            n_items,
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
        "serial_position_endpoint": serial_position_endpoint_generic(
            behavior, protocol
        ),
        "density_dependencies": {
            name: {
                "raw_subject": json_values(values),
                "summary": summarize_subjects(values, counts, interval=interval),
            }
            for name, values in dependencies.items()
        },
    }
    size_metrics = {
        **_distance_summaries(behavior, counts, interval, n_items),
        "item_count": n_items,
        "support_relations": len(relations),
        "support_trials": protocol.support_trials,
        "query_pairs": len(geometry.pairs),
        "query_trials": protocol.query_trials,
        "direct_query_fraction": len(relations) / float(len(geometry.pairs)),
    }
    presentation = measure_presentation_invariance(
        evaluator, local, intact_trace.natural_scalars
    )
    schedule_hash = support_schedule_hash(evaluator)
    n8_metrics_exact = bool(
        n_items != 8
        or _legacy_metric_projection(metrics) == (source_cell or {})["metrics"]
    )
    n8_schedule_exact = bool(
        n_items != 8 or schedule_hash == (source_cell or {})["support_schedule_sha256"]
    )
    after = tensor_hashes(evaluator.net)
    individualized = metrics["individualized"]
    individualized_finite = bool(
        individualized["eligible_noncorrect_subjects"] < 2
        or (
            np.isfinite(
                individualized["mean_pairwise_kendall_tau"]["bootstrap"]["upper"]
            )
            and np.isfinite(
                individualized["stable_error_80_pair_density"]["bootstrap"]["lower"]
            )
        )
    )
    integrity = {
        "source_validation_passed": bool(source_validation["passed"]),
        "graph_validation_passed": bool(graph_validation["passed"]),
        "size_interface": size_interface,
        "N8_evaluator_interface": n8_interface,
        "N8_legacy_metrics_exact": n8_metrics_exact,
        "N8_schedule_hash_exact": n8_schedule_exact,
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
        "primary_values_finite": bool(
            finite_primary(metrics) and individualized_finite
        ),
    }
    integrity["all_passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["graph_validation_passed"]
        and size_interface["passed"]
        and n8_interface["passed"]
        and integrity["N8_legacy_metrics_exact"]
        and integrity["N8_schedule_hash_exact"]
        and integrity["bounded_gpu_runtime"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and integrity["local_off_global_logit_max_abs_error"] <= 1e-6
        and integrity["support_write_reversal_max_abs_error"] <= 1e-7
        and integrity["query_key_reversal_max_abs_error"] <= 1e-7
        and integrity["primary_values_finite"]
    )
    decision = within_cell_decision(metrics, integrity)
    return {
        "size_id": graph["size_id"],
        "n_items": n_items,
        "protocol_id": protocol.protocol_id,
        "rank_edges": graph["rank_edges"],
        "item_edges_higher_lower": [list(relation) for relation in relations],
        "support_schedule_sha256": schedule_hash,
        "cue_codes_sha256": size_interface["cue_codes_sha256"],
        "admission_sha256": size_interface["admission_sha256"],
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "size_specific_metrics": size_metrics,
        "integrity": integrity,
        "decision": decision,
    }


def cross_cell_decision(
    seeds: dict, sizes: list[int], mandatory_seeds: list[int]
) -> dict:
    cells = [
        seeds[str(seed)]["sizes"][str(n_items)]
        for n_items in sizes
        for seed in mandatory_seeds
    ]
    if not all(cell["decision"]["interpretable"] for cell in cells):
        return {
            "outcome": "NONINTERPRETABLE_EXECUTION",
            "size_passes": None,
            "heterogeneous_across_backbones": None,
        }
    if not all(cell["decision"]["competence_passed"] for cell in cells):
        return {
            "outcome": "ITEM_COUNT_COMPETENCE_NOT_ESTABLISHED",
            "size_passes": None,
            "heterogeneous_across_backbones": None,
        }
    size_passes = {
        str(n_items): all(
            seeds[str(seed)]["sizes"][str(n_items)]["decision"][
                "all_primary_links_pass"
            ]
            for seed in mandatory_seeds
        )
        for n_items in sizes
    }
    links = next(iter(cells))["decision"]["flags"]
    heterogeneous = any(
        0
        < sum(
            seeds[str(seed)]["sizes"][str(n_items)]["decision"]["flags"][link]
            for seed in mandatory_seeds
        )
        < len(mandatory_seeds)
        for n_items in sizes
        for link in links
    )
    if all(size_passes.values()):
        outcome = "LIU_ITEM_COUNT_MECHANISM_TRANSPORTED"
    elif (
        size_passes.get("8") and not size_passes.get("6") and not size_passes.get("10")
    ):
        outcome = "FUNCTIONAL_ASYMMETRY_NOT_ITEM_COUNT_TRANSPORTED"
    else:
        outcome = "ITEM_COUNT_DEPENDENT_OR_UNRESOLVED"
    return {
        "outcome": outcome,
        "size_passes": size_passes,
        "heterogeneous_across_backbones": heterogeneous,
    }


def evaluate(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    runtime = require_formal_runtime()
    specification = load_json(specification_path)
    source_validation = validate_sources(specification_path, lock_path)
    graph_validation = validate_graph_contract(specification)
    source = load_json(
        resolve_path(specification["registered_sources"]["N8_source_result"]["path"])
    )
    base = load_ranking_protocol(
        resolve_path(specification["registered_sources"]["base_liu_protocol"]["path"])
    )
    evaluation = specification["evaluation"]
    graphs = specification["size_matched_graph_contract"]["graphs"]
    seeds = {}
    for seed in specification["development_backbones"]["mandatory_seeds"]:
        artifacts = specification["development_backbones"]["artifacts"][str(seed)]
        backbone, model_config, checkpoint = load_frozen_retro_checkpoint(
            resolve_path(artifacts["checkpoint"]["path"]),
            int(evaluation["subjects_per_size_and_backbone"]),
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        gain = load_json(resolve_path(artifacts["gain"]["path"]))
        local = ConjunctiveLocalTrace(model_config.cs)
        with torch.no_grad():
            local.raw_gain.fill_(float(gain["raw_lambda_L"]))
        size_results = {}
        for size_index, graph in enumerate(graphs, start=1):
            protocol = protocol_for_size(base, graph)
            evaluator = FrozenFastWeightEvaluator(
                backbone,
                model_config,
                protocol,
                cue_seed=int(evaluation["cue_seed"]),
                support_seed=int(evaluation["support_seed"]),
                cue_mode=str(evaluation["cue_mode"]),
                subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
                subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
                required_item_count=None,
            )
            size_interface = validate_size_interface(evaluator)
            if protocol.n_items == 8:
                frozen = FrozenFastWeightEvaluator(
                    backbone,
                    model_config,
                    protocol,
                    cue_seed=int(evaluation["cue_seed"]),
                    support_seed=int(evaluation["support_seed"]),
                    cue_mode=str(evaluation["cue_mode"]),
                    subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
                    subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
                )
                n8_interface = validate_n8_evaluator_interface(evaluator, frozen)
                source_cell = source["seeds"][str(seed)]["families"][
                    "liu_cycle_centered"
                ]["densities"]["8"]
            else:
                n8_interface = {"passed": True, "not_applicable": True}
                source_cell = None
            size_results[str(protocol.n_items)] = evaluate_cell(
                specification,
                int(seed),
                graph,
                size_index,
                evaluator,
                local,
                runtime,
                source_validation,
                graph_validation,
                size_interface,
                n8_interface,
                source_cell,
            )
        seeds[str(seed)] = {
            "seed": int(seed),
            "checkpoint": asdict(checkpoint),
            "gain_path": artifacts["gain"]["path"],
            "lambda_L": float(local.gain.detach().cpu()),
            "sizes": size_results,
        }
    sizes = [int(graph["n_items"]) for graph in graphs]
    mandatory_seeds = [
        int(seed) for seed in specification["development_backbones"]["mandatory_seeds"]
    ]
    decision = cross_cell_decision(seeds, sizes, mandatory_seeds)
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
        "registered_outcome_tree": specification["outcome_tree"],
        "claim_boundary": specification["claim_boundary"],
        "known_limitations_carried_forward": [
            "N=6 and N=10 are strict cardinality OOD tests",
            "item count co-varies with support count, support duration, query count, and direct-query fraction",
            "density progressively regularizes individualized global order in the prior N=8 audit",
            "excessive symbolic-distance slope remains a separate global-policy problem",
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
