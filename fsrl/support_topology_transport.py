"""Registered Liu support-topology transport evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from .assembly_trajectory import (
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
    kendall_tau_scores,
    summarize_difference,
    summarize_subjects,
)
from .behavioral import (
    analyze_sampled_query_policy,
    count_circular_triads,
    kendall_tau_positions,
)
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
from .ranking_protocol import RankingProtocol, load_ranking_protocol
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/liu_support_topology_transport_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/liu_support_topology_transport_v1.lock.json")
)
DEFAULT_REPAIR_PATH = (
    resolve_record("benchmarks/liu_support_topology_transport_v1.repair1.json")
)
DEFAULT_REPAIR_LOCK_PATH = (
    resolve_record("benchmarks/liu_support_topology_transport_v1.repair1.lock.json")
)
DEFAULT_ATTEMPT1_PATH = (
    resolve_record("results/liu_support_topology_transport_v1.attempt1.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/liu_support_topology_transport_v1.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/support_topology_transport.py",
    "tests": "tests/test_support_topology_transport.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}


def load_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_exclusive(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def _registration(path: str) -> dict:
    return {"path": path, "sha256": file_sha256(resolve_record(path))}


def write_implementation_lock(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    repairing = lock_path.resolve() == DEFAULT_REPAIR_LOCK_PATH.resolve()
    lock = {
        "schema_version": 1,
        "experiment_id": "liu-support-topology-transport-v1",
        "implementation_status": (
            "frozen_after_attempt1_and_before_estimator_only_repair_rerun"
            if repairing
            else "frozen_before_any_alternative_graph_model_evaluation"
        ),
        "registration_commit": (
            "a8ecef196ba42b894bf2be314727ad797c2609f2"
            if repairing
            else "b378bec2f5a3ec842b13dd6e6b1340ba6996db00"
        ),
        "specification_sha256": file_sha256(specification_path),
        "implementation_sources": {
            name: _registration(path) for name, path in IMPLEMENTATION_SOURCES.items()
        },
    }
    if repairing:
        lock["repair_sources"] = {
            "repair_registration": {
                "path": legacy_identifier(DEFAULT_REPAIR_PATH),
                "sha256": file_sha256(DEFAULT_REPAIR_PATH),
            },
            "attempt1": {
                "path": str(DEFAULT_ATTEMPT1_PATH.relative_to(ROOT)),
                "sha256": file_sha256(DEFAULT_ATTEMPT1_PATH),
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
        **lock.get("repair_sources", {}),
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
        raise RuntimeError(f"support-topology source or artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _graph_connected(edges: tuple[tuple[int, int], ...], n_items: int = 8) -> bool:
    adjacency = [set() for _ in range(n_items)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = {0}
    frontier = [0]
    while frontier:
        item = frontier.pop()
        for neighbor in adjacency[item]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == n_items


def graph_descriptor(edges: tuple[tuple[int, int], ...], n_items: int = 8) -> dict:
    edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    adjacency = [set() for _ in range(n_items)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    distances = []
    for source in range(n_items):
        shortest = {source: 0}
        frontier = [source]
        while frontier:
            item = frontier.pop(0)
            for neighbor in adjacency[item]:
                if neighbor not in shortest:
                    shortest[neighbor] = shortest[item] + 1
                    frontier.append(neighbor)
        if len(shortest) != n_items:
            diameter = None
            break
        distances.extend(shortest.values())
    else:
        diameter = max(distances)
    triangles = sum(
        int(
            tuple(sorted((first, second))) in edges
            and tuple(sorted((first, third))) in edges
            and tuple(sorted((second, third))) in edges
        )
        for first, second, third in combinations(range(n_items), 3)
    )
    return {
        "edge_count": len(edges),
        "connected": _graph_connected(edges, n_items),
        "distance_multiset": sorted(abs(first - second) for first, second in edges),
        "sorted_degree_sequence": sorted(len(neighbors) for neighbors in adjacency),
        "triangle_count": triangles,
        "diameter": diameter,
    }


def enumerate_registered_graphs(
    specification: dict,
) -> dict[str, tuple[tuple[int, int], ...]]:
    contract = specification["matched_graph_contract"]
    required = tuple(contract["required_rank_distance_multiset"])
    multiplicities = {
        distance: required.count(distance) for distance in sorted(set(required))
    }
    distance_edges = {
        distance: tuple((first, first + distance) for first in range(8 - distance))
        for distance in multiplicities
    }
    choice_groups = [
        tuple(combinations(distance_edges[distance], count))
        for distance, count in multiplicities.items()
    ]
    targets = {
        graph["graph_id"]: (
            graph["target_sorted_degree_sequence"],
            graph["target_triangle_count"],
            graph["target_diameter"],
        )
        for graph in contract["graphs"]
    }
    selected: dict[str, tuple[tuple[int, int], ...]] = {}
    for grouped_edges in product(*choice_groups):
        edges = tuple(sorted(edge for group in grouped_edges for edge in group))
        descriptor = graph_descriptor(edges)
        if not descriptor["connected"]:
            continue
        key = (
            descriptor["sorted_degree_sequence"],
            descriptor["triangle_count"],
            descriptor["diameter"],
        )
        for graph_id, target in targets.items():
            if graph_id not in selected and key == target:
                selected[graph_id] = edges
    if set(selected) != set(targets):
        raise RuntimeError("could not reconstruct every registered graph stratum")
    return selected


def validate_graph_contract(specification: dict) -> dict:
    contract = specification["matched_graph_contract"]
    enumerated = enumerate_registered_graphs(specification)
    original_degree = graph_descriptor(
        tuple(map(tuple, contract["original_source_correct_rank_edges"]))
    )["sorted_degree_sequence"]
    checks = []
    observed_degrees = []
    for graph in contract["graphs"]:
        edges = tuple(map(tuple, graph["rank_edges"]))
        descriptor = graph_descriptor(edges)
        expected = enumerated[graph["graph_id"]]
        passed = bool(
            edges == expected
            and descriptor["edge_count"] == contract["required_edge_count"]
            and descriptor["connected"] == contract["required_connected"]
            and descriptor["distance_multiset"]
            == contract["required_rank_distance_multiset"]
            and descriptor["sorted_degree_sequence"]
            == graph["target_sorted_degree_sequence"]
            and descriptor["triangle_count"] == graph["target_triangle_count"]
            and descriptor["diameter"] == graph["target_diameter"]
        )
        observed_degrees.append(tuple(descriptor["sorted_degree_sequence"]))
        checks.append(
            {
                "graph_id": graph["graph_id"],
                "expected_lexicographic_edges": [list(edge) for edge in expected],
                "descriptor": descriptor,
                "passed": passed,
            }
        )
    nonisomorphic = bool(
        len(set(observed_degrees)) == len(observed_degrees)
        and tuple(original_degree) not in observed_degrees
    )
    passed = all(check["passed"] for check in checks) and nonisomorphic
    if not passed:
        raise RuntimeError("registered support graph validation failed")
    return {
        "passed": True,
        "pairwise_nonisomorphic_degree_certificate": nonisomorphic,
        "original_sorted_degree_sequence": original_degree,
        "checks": checks,
    }


def protocol_for_graph(base: RankingProtocol, graph: dict) -> RankingProtocol:
    rank_edges = tuple(map(tuple, graph["rank_edges"]))
    support_pairs = tuple(
        (base.true_order_high_to_low[higher], base.true_order_high_to_low[lower])
        for higher, lower in rank_edges
    )
    return RankingProtocol(
        protocol_id=f"liu-support-topology-v1-{graph['graph_id']}",
        item_labels=base.item_labels,
        true_order_high_to_low=base.true_order_high_to_low,
        support_pairs_higher_lower=support_pairs,
        support_blocks=base.support_blocks,
        query_blocks=base.query_blocks,
        human_targets={},
    )


def bootstrap_counts(
    rng: np.random.Generator, samples: int, subjects: int
) -> np.ndarray:
    return rng.multinomial(
        subjects, np.full(subjects, 1.0 / subjects), size=samples
    ).astype(np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _json_values(values: np.ndarray) -> list:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return None if not np.isfinite(array) else float(array)
    return [_json_values(row) for row in array]


def _subject_group_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(values, dtype=np.float64)[:, mask], axis=1)


def condition_metrics(
    field: np.ndarray,
    geometry,
    learned_mask: np.ndarray,
    counts: np.ndarray,
    interval: float,
    temperature: float,
) -> dict:
    correct = np.asarray(field, dtype=np.float64) * geometry.true_sign[None]
    decision = (correct > 0.0).astype(np.float64)
    probability = _sigmoid(correct / temperature)
    groups = {
        "overall": np.ones(len(geometry.pairs), dtype=bool),
        "learned": learned_mask,
        "nonlearned": ~learned_mask,
    }
    raw = {
        "exact_decision_accuracy": {
            name: _subject_group_mean(decision, mask) for name, mask in groups.items()
        },
        "correct_probability": {
            name: _subject_group_mean(probability, mask)
            for name, mask in groups.items()
        },
    }
    return {
        "summary": {
            metric: {
                name: summarize_subjects(values, counts, interval=interval)
                for name, values in rows.items()
            }
            for metric, rows in raw.items()
        },
        "raw_subject": {
            metric: {name: _json_values(values) for name, values in rows.items()}
            for metric, rows in raw.items()
        },
        "correct_signed_field": _json_values(correct),
    }


def constructive_metrics(
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
    transitivity = []
    for row in intact_field:
        winners = {
            pair: pair[0] if row[index] > 0.0 else pair[1]
            for index, pair in enumerate(geometry.pairs)
        }
        circular = count_circular_triads(winners, len(geometry.true_potential))
        transitivity.append(1.0 - circular / len(tuple(combinations(range(8), 3))))
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
        "raw_subject": {name: _json_values(values) for name, values in raw.items()},
    }


def relation_loo_metrics(
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
            [item for item in range(8) if item not in endpoints], dtype=np.int64
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
            "remote_absolute": _json_values(subject_remote),
            "third_party_relational": _json_values(subject_third),
        },
        "raw_relation_subject": {
            "remote_absolute": _json_values(remote),
            "third_party_relational": _json_values(third_party),
        },
    }


def _key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    flat = (
        (np.outer(left, right) - np.outer(right, left)).reshape(-1).astype(np.float64)
    )
    return flat / max(float(np.linalg.norm(flat)), 1e-8)


def reconstruct_local_ledger(
    item_codes: np.ndarray,
    schedules,
    natural_scalars: np.ndarray,
    actual_state: np.ndarray,
    actual_canonical_reads: np.ndarray,
) -> dict:
    pairs = tuple(combinations(range(item_codes.shape[1]), 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    state_errors = []
    read_errors = []
    gpu_state_errors = []
    gpu_read_errors = []
    for subject, schedule in enumerate(schedules):
        codes = np.asarray(item_codes[subject], dtype=np.float64)
        keys = np.stack([_key(codes[first], codes[second]) for first, second in pairs])
        reconstructed = np.zeros(keys.shape[1], dtype=np.float64)
        ledger = np.zeros(len(pairs), dtype=np.float64)
        for trial_index, trial in enumerate(schedule):
            scalar = float(natural_scalars[subject, trial_index])
            reconstructed += scalar * _key(
                codes[trial.left_item], codes[trial.right_item]
            )
            canonical = tuple(sorted((trial.left_item, trial.right_item)))
            orientation = 1.0 if trial.left_item < trial.right_item else -1.0
            ledger[pair_index[canonical]] += orientation * scalar
        ledger_state = ledger @ keys
        direct_reads = reconstructed @ keys.T
        compressed_reads = (keys @ keys.T) @ ledger
        state_errors.append(float(np.max(np.abs(reconstructed - ledger_state))))
        read_errors.append(float(np.max(np.abs(direct_reads - compressed_reads))))
        gpu_state_errors.append(
            float(np.max(np.abs(reconstructed - actual_state[subject])))
        )
        gpu_read_errors.append(
            float(np.max(np.abs(compressed_reads - actual_canonical_reads[subject])))
        )
    return {
        "tensor_state_max_abs_error": max(state_errors, default=0.0),
        "ledger_tensor_state_max_abs_error": max(state_errors, default=0.0),
        "all_query_raw_read_max_abs_error": max(read_errors, default=0.0),
        "raw_subject_tensor_state_max_abs_error": state_errors,
        "raw_subject_ledger_tensor_state_max_abs_error": state_errors,
        "raw_subject_query_read_max_abs_error": read_errors,
        "gpu_tensor_state_max_abs_error_diagnostic": max(gpu_state_errors, default=0.0),
        "gpu_query_read_max_abs_error_diagnostic": max(gpu_read_errors, default=0.0),
        "raw_subject_gpu_tensor_state_max_abs_error_diagnostic": gpu_state_errors,
        "raw_subject_gpu_query_read_max_abs_error_diagnostic": gpu_read_errors,
    }


def individualized_metrics(
    behavior: dict, rng: np.random.Generator, samples: int
) -> dict:
    eligible = [row for row in behavior["subjects"] if row["overall_accuracy"] >= 0.5]
    analysis = [row for row in eligible if row["ranking_class"] != "correct"]
    stable = np.asarray(
        [row["stable_error_pair_counts"]["80"] > 0 for row in analysis],
        dtype=np.float64,
    )
    if len(stable):
        stable_counts = bootstrap_counts(rng, samples, len(stable))
        stable_summary = summarize_subjects(stable, stable_counts, interval=0.95)
    else:
        stable_summary = summarize_subjects(
            np.asarray([], dtype=np.float64), np.zeros((samples, 0)), interval=0.95
        )
    orders = [row["subjective_order_high_to_low"] for row in analysis]
    if len(orders) >= 2:
        positions = []
        for order in orders:
            row = np.empty(8, dtype=np.int64)
            row[np.asarray(order, dtype=np.int64)] = np.arange(8)
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
        "mean_pairwise_kendall_tau": tau,
    }


def serial_position_endpoint(behavior: dict, protocol: RankingProtocol) -> dict:
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    totals = np.zeros(8, dtype=np.float64)
    counts = np.zeros(8, dtype=np.float64)
    for row in behavior["pairs"]:
        value = float(row["mean_accuracy_all"])
        for item in row["pair"]:
            totals[rank[item]] += value
            counts[rank[item]] += 1.0
    profile = totals / counts
    interior = float(np.mean(profile[1:7]))
    return {
        "profile_high_to_low": _json_values(profile),
        "interior_mean": interior,
        "mean_endpoint_contrast": float(np.mean(profile[[0, 7]]) - interior),
        "minimum_endpoint_advantage": float(min(profile[0], profile[7]) - interior),
    }


def _schedule_hash(evaluator: FrozenFastWeightEvaluator) -> str:
    payload = json.dumps(
        [
            [asdict(trial) for trial in schedule]
            for schedule in evaluator.support_schedules
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _summary_value(summary: dict, boundary: str) -> float | None:
    return summary.get("bootstrap", {}).get(boundary)


def within_cell_decision(metrics: dict, integrity: dict) -> dict:
    intact = metrics["conditions"]["intact"]["summary"]
    p_off = metrics["conditions"]["P_off_a_on"]["summary"]
    constructive = metrics["constructive"]["summary"]
    individualized = metrics["individualized"]
    global_loo = metrics["global_relation_LOO"]["summary"]
    contrasts = metrics["contrasts"]
    local = metrics["local_exactness"]
    criteria = {
        "intact_competence": bool(
            _summary_value(intact["exact_decision_accuracy"]["learned"], "lower") > 0.50
            and _summary_value(intact["exact_decision_accuracy"]["nonlearned"], "lower")
            > 0.50
        ),
        "constructive_global_structure": bool(
            _summary_value(constructive["intact_gradient_energy_fraction"], "lower")
            >= 0.95
            and _summary_value(constructive["a_off_gradient_energy_fraction"], "lower")
            >= 0.95
            and _summary_value(
                constructive["intact_transitive_triplet_fraction"], "lower"
            )
            >= 0.95
            and _summary_value(
                constructive["intact_hodge_order_kendall_tau_to_true"], "lower"
            )
            > 0.0
        ),
        "individualized_stable_structure": bool(
            individualized["eligible_noncorrect_subjects"] >= 2
            and individualized["mean_pairwise_kendall_tau"]["bootstrap"]["upper"] < 0.80
            and individualized["stable_error_80_prevalence"]["bootstrap"]["lower"]
            >= 0.80
        ),
        "P_off_global_collapse": bool(
            _summary_value(p_off["correct_probability"]["nonlearned"], "upper") <= 0.55
            and _summary_value(
                contrasts["P_off_local_remote_minus_quarter_global"], "upper"
            )
            < 0.0
        ),
        "P_remote_reassembly": bool(
            _summary_value(global_loo["remote_absolute"], "lower") > 0.0
            and _summary_value(global_loo["third_party_relational"], "lower") > 0.0
        ),
        "a_off_direct_loss": bool(
            _summary_value(contrasts["intact_minus_a_off_learned_probability"], "lower")
            > 0.0
        ),
        "P_off_a_on_direct_nontransitive": bool(
            _summary_value(p_off["correct_probability"]["learned"], "lower") > 0.50
            and _summary_value(
                contrasts["P_off_learned_minus_nonlearned_probability"], "lower"
            )
            > 0.0
        ),
        "exact_local_compression": bool(
            local["tensor_state_max_abs_error"] <= 1e-6
            and local["ledger_tensor_state_max_abs_error"] <= 1e-6
            and local["all_query_raw_read_max_abs_error"] <= 1e-6
        ),
    }
    flags = {
        name: bool(integrity["all_passed"] and value)
        for name, value in criteria.items()
    }
    return {
        "interpretable": bool(integrity["all_passed"]),
        "competence_passed": flags["intact_competence"],
        "all_eight_primary_links_pass": all(flags.values()),
        "flags": flags,
    }


def cross_cell_decision(
    seeds: dict, graph_ids: list[str], mandatory_seeds: list[int]
) -> dict:
    cells = [
        seeds[str(seed)]["graphs"][graph_id]
        for graph_id in graph_ids
        for seed in mandatory_seeds
    ]
    if not all(cell["decision"]["interpretable"] for cell in cells):
        outcome = "NONINTERPRETABLE_EXECUTION"
    elif not all(cell["decision"]["competence_passed"] for cell in cells):
        outcome = "STRUCTURAL_COMPETENCE_NOT_ESTABLISHED"
    else:
        graph_passes = {
            graph_id: all(
                seeds[str(seed)]["graphs"][graph_id]["decision"][
                    "all_eight_primary_links_pass"
                ]
                for seed in mandatory_seeds
            )
            for graph_id in graph_ids
        }
        links = next(iter(cells))["decision"]["flags"]
        heterogeneous = any(
            0
            < sum(
                seeds[str(seed)]["graphs"][graph_id]["decision"]["flags"][link]
                for seed in mandatory_seeds
            )
            < len(mandatory_seeds)
            for graph_id in graph_ids
            for link in links
        )
        passed_graphs = sum(graph_passes.values())
        if passed_graphs == len(graph_ids):
            outcome = "LIU_STRUCTURAL_MECHANISM_TRANSPORTED"
        elif passed_graphs > 0 or heterogeneous:
            outcome = "TOPOLOGY_DEPENDENT_OR_UNRESOLVED"
        else:
            outcome = "FUNCTIONAL_ASYMMETRY_NOT_TRANSPORTED"
        return {
            "outcome": outcome,
            "graph_passes": graph_passes,
            "heterogeneous_across_backbones": heterogeneous,
        }
    return {
        "outcome": outcome,
        "graph_passes": None,
        "heterogeneous_across_backbones": None,
    }


def _finite_primary(metrics: dict) -> bool:
    values = []
    for condition in metrics["conditions"].values():
        for metric in condition["summary"].values():
            for group in metric.values():
                values.extend(group["bootstrap"].values())
    for row in metrics["constructive"]["summary"].values():
        values.extend(row["bootstrap"].values())
    for row in metrics["global_relation_LOO"]["summary"].values():
        values.extend(row["bootstrap"].values())
    for row in metrics["contrasts"].values():
        values.extend(row["bootstrap"].values())
    return all(value is not None and np.isfinite(value) for value in values)


def nonrepair_projection(result: dict) -> dict:
    """Remove only metadata and estimands authorized to change in repair 1."""

    projected = copy.deepcopy(result)
    projected.pop("source_validation", None)
    projected.pop("decision", None)
    projected.pop("repair_validation", None)
    for seed in projected["seeds"].values():
        for cell in seed["graphs"].values():
            cell.pop("decision", None)
            cell["metrics"].pop("local_exactness", None)
    return projected


def evaluate_cell(
    specification: dict,
    seed: int,
    graph: dict,
    backbone,
    model_config,
    local: ConjunctiveLocalTrace,
    runtime: dict,
    source_validation: dict,
    graph_validation: dict,
) -> dict:
    base = load_ranking_protocol(
        resolve_path(specification["task_fidelity"]["base_protocol"])
    )
    protocol = protocol_for_graph(base, graph)
    evaluation = specification["evaluation"]
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
    graph_index = 1 + [
        row["graph_id"] for row in specification["matched_graph_contract"]["graphs"]
    ].index(graph["graph_id"])
    bootstrap_seed = 530000 + 100 * seed + graph_index
    rng = np.random.default_rng(bootstrap_seed)
    counts = bootstrap_counts(
        rng,
        int(evaluation["bootstrap_samples"]),
        int(evaluation["subjects_per_graph_and_backbone"]),
    )
    interval = float(evaluation["bootstrap_interval"])
    geometry = build_complete_graph_geometry(protocol)
    relations = tuple(protocol.support_pairs_higher_lower)
    learned_mask = np.asarray(
        [pair in protocol.learned_pairs for pair in geometry.pairs]
    )
    schedules = tuple(_ordered_pairs(protocol.n_items) for _ in range(model_config.bs))
    before = _tensor_hashes(backbone)
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
        schedules,
        local_off=False,
        global_off=False,
        shuffled_indices=None,
    )
    a_off_bundle = _query_pass(
        evaluator,
        local,
        intact_fast_weights,
        intact_trace.state,
        schedules,
        local_off=True,
        global_off=False,
        shuffled_indices=None,
    )
    p_off_bundle = _query_pass(
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
        _query_pass(
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
    loo_p_off_bundles = [
        _query_pass(
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
    loo_p_off_fields = np.asarray(
        [margin_fields(bundle, protocol.n_items) for bundle in loo_p_off_bundles]
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
        fields["P_off_a_on"], loo_p_off_fields, relations, geometry, counts, interval
    )
    intact_learned = np.asarray(
        conditions["intact"]["raw_subject"]["correct_probability"]["learned"]
    )
    a_off_learned = np.asarray(
        conditions["a_off"]["raw_subject"]["correct_probability"]["learned"]
    )
    p_off_learned = np.asarray(
        conditions["P_off_a_on"]["raw_subject"]["correct_probability"]["learned"]
    )
    p_off_nonlearned = np.asarray(
        conditions["P_off_a_on"]["raw_subject"]["correct_probability"]["nonlearned"]
    )
    global_remote = np.asarray(global_loo["raw_subject"]["remote_absolute"])
    local_remote = np.asarray(local_loo["raw_subject"]["remote_absolute"])
    contrasts = {
        "intact_minus_a_off_learned_probability": summarize_difference(
            intact_learned, a_off_learned, counts, interval=interval
        ),
        "P_off_learned_minus_nonlearned_probability": summarize_difference(
            p_off_learned, p_off_nonlearned, counts, interval=interval
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
    individual = individualized_metrics(
        behavior, rng, int(evaluation["bootstrap_samples"])
    )
    exact = reconstruct_local_ledger(
        evaluator.cue_codes,
        evaluator.support_schedules,
        intact_trace.natural_scalars,
        intact_trace.state.detach().cpu().numpy().astype(np.float64),
        intact_bundle["raw_local_margins"][:, 0::2],
    )
    retained = _retained_mask(evaluator, relations)
    correct_probability = _sigmoid(
        fields["intact"] * geometry.true_sign[None] / float(evaluation["temperature"])
    )
    relation_indices = [
        geometry.pairs.index(tuple(sorted(relation))) for relation in relations
    ]
    learned_probability = correct_probability[:, relation_indices].T
    retained_values = np.where(retained, learned_probability, np.nan)
    omitted_values = np.where(~retained, learned_probability, np.nan)

    def masked_subject_mean(values: np.ndarray) -> np.ndarray:
        finite = np.sum(np.isfinite(values), axis=0)
        return np.divide(
            np.nansum(values, axis=0),
            finite,
            out=np.full(model_config.bs, np.nan),
            where=finite > 0,
        )

    local_direct = ((fields["intact"] - fields["a_off"]) * geometry.true_sign[None])[
        :, relation_indices
    ].T
    metrics = {
        "conditions": conditions,
        "constructive": constructive_metrics(
            fields["intact"], fields["a_off"], geometry, counts, interval
        ),
        "individualized": individual,
        "global_relation_LOO": global_loo,
        "P_off_local_relation_LOO": local_loo,
        "contrasts": contrasts,
        "local_exactness": exact,
        "retained_omitted": {
            "retained_counts_per_subject": _json_values(np.sum(retained, axis=0)),
            "omitted_counts_per_subject": _json_values(np.sum(~retained, axis=0)),
            "retained_correct_probability": summarize_subjects(
                masked_subject_mean(retained_values), counts, interval=interval
            ),
            "omitted_correct_probability": summarize_subjects(
                masked_subject_mean(omitted_values), counts, interval=interval
            ),
        },
        "raw_relation_subject_local_direct_correctness": _json_values(local_direct),
        "sampled_behavior": behavior,
        "sampled_accuracy_bootstrap": sampled_accuracy,
        "serial_position_endpoint": serial_position_endpoint(behavior, protocol),
    }
    presentation = _presentation_invariance(
        evaluator, local, intact_trace.natural_scalars
    )
    local_off_error = float(
        np.max(np.abs(a_off_bundle["logits"] - a_off_bundle["global_logits"]))
    )
    after = _tensor_hashes(backbone)
    integrity = {
        "source_validation_passed": bool(source_validation["passed"]),
        "graph_validation_passed": bool(graph_validation["passed"]),
        "bounded_gpu_runtime": bool(
            runtime["active"]
            and runtime["cuda_available"]
            and runtime["torch_intraop_threads"] == 1
            and runtime["torch_interop_threads"] == 1
        ),
        "backbone_tensor_hashes_unchanged": before == after,
        "local_off_global_logit_max_abs_error": local_off_error,
        **presentation,
        "primary_values_finite": _finite_primary(metrics),
    }
    integrity["all_passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["graph_validation_passed"]
        and integrity["bounded_gpu_runtime"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and integrity["local_off_global_logit_max_abs_error"] <= 1e-6
        and integrity["support_write_reversal_max_abs_error"] <= 1e-7
        and integrity["query_key_reversal_max_abs_error"] <= 1e-7
        and integrity["primary_values_finite"]
    )
    decision = within_cell_decision(metrics, integrity)
    return {
        "graph_id": graph["graph_id"],
        "protocol_id": protocol.protocol_id,
        "rank_edges": graph["rank_edges"],
        "item_edges_higher_lower": [list(relation) for relation in relations],
        "support_schedule_sha256": _schedule_hash(evaluator),
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "integrity": integrity,
        "decision": decision,
    }


def evaluate(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    runtime = require_formal_runtime()
    specification = load_json(specification_path)
    source_validation = validate_sources(specification_path, lock_path)
    graph_validation = validate_graph_contract(specification)
    evaluation = specification["evaluation"]
    seeds = {}
    for seed in specification["development_backbones"]["mandatory_seeds"]:
        artifacts = specification["development_backbones"]["artifacts"][str(seed)]
        backbone, model_config, checkpoint = load_retro_checkpoint(
            resolve_path(artifacts["checkpoint"]["path"]),
            int(evaluation["subjects_per_graph_and_backbone"]),
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        gain = load_json(resolve_path(artifacts["gain"]["path"]))
        local = ConjunctiveLocalTrace(model_config.cs)
        with torch.no_grad():
            local.raw_gain.fill_(float(gain["raw_lambda_L"]))
        graphs = {}
        for graph in specification["matched_graph_contract"]["graphs"]:
            graphs[graph["graph_id"]] = evaluate_cell(
                specification,
                int(seed),
                graph,
                backbone,
                model_config,
                local,
                runtime,
                source_validation,
                graph_validation,
            )
        seeds[str(seed)] = {
            "seed": int(seed),
            "checkpoint": asdict(checkpoint),
            "gain_path": artifacts["gain"]["path"],
            "lambda_L": float(local.gain.detach().cpu()),
            "graphs": graphs,
        }
    graph_ids = [
        graph["graph_id"] for graph in specification["matched_graph_contract"]["graphs"]
    ]
    decision = cross_cell_decision(
        seeds,
        graph_ids,
        [
            int(seed)
            for seed in specification["development_backbones"]["mandatory_seeds"]
        ],
    )
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "execution_runtime": runtime,
        "source_validation": source_validation,
        "graph_validation": graph_validation,
        "seeds": seeds,
        "decision": decision,
        "registered_primary_links": specification["primary_links"],
        "registered_outcome_tree": specification["cell_and_cross_cell_decision"],
        "known_limitations_carried_forward": [
            "excessive symbolic-distance slope",
            "weak serial-position endpoint contrast",
            "original-graph seed-2104 self-inconsistency mismatch",
        ],
    }
    lock = source_validation["lock"]
    if "repair_sources" in lock:
        attempt_path = resolve_path(lock["repair_sources"]["attempt1"]["path"])
        attempt = load_json(attempt_path)
        identity = nonrepair_projection(result) == nonrepair_projection(attempt)
        result["repair_validation"] = {
            "repair_id": "liu-support-topology-transport-v1-repair1",
            "attempt1_path": str(attempt_path.relative_to(ROOT)),
            "attempt1_sha256": file_sha256(attempt_path),
            "nonrepair_values_exactly_equal": identity,
            "changed_scope": (
                "local_exactness estimand, dependent within-cell flags, cross-cell "
                "decision, and source-validation metadata only"
            ),
        }
        if not identity:
            raise RuntimeError("repair 1 changed a nonrepair model output")
    return result


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
