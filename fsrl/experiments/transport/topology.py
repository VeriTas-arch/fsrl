"""Registered Liu support-topology transport evaluation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.policy import bundle_logits, margin_fields
from fsrl.analysis.relational_transport import (
    condition_metrics,
    constructive_metrics,
    finite_primary,
    individualized_metrics,
    relation_loo_metrics,
    serial_position_endpoint,
)
from fsrl.analysis.statistics import (
    bootstrap_counts,
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
from fsrl.tasks.protocol import load_ranking_protocol, ordered_pairs
from fsrl.tasks.transport_graph import graph_descriptor, protocol_for_graph

from .topology_decision import within_cell_decision

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/liu_support_topology_transport_v1.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/liu_support_topology_transport_v1.lock.json"
)
DEFAULT_REPAIR_PATH = resolve_record(
    "benchmarks/liu_support_topology_transport_v1.repair1.json"
)
DEFAULT_REPAIR_LOCK_PATH = resolve_record(
    "benchmarks/liu_support_topology_transport_v1.repair1.lock.json"
)
DEFAULT_ATTEMPT1_PATH = resolve_record(
    "results/liu_support_topology_transport_v1.attempt1.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/liu_support_topology_transport_v1.json")
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/support_topology_transport.py",
    "tests": "tests/test_support_topology_transport.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}


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
            name: canonical_file_registration(path)
            for name, path in IMPLEMENTATION_SOURCES.items()
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
    schedules = tuple(ordered_pairs(protocol.n_items) for _ in range(model_config.bs))
    before = tensor_hashes(backbone)
    readout = readout_dual_access_query_conditions(evaluator, local, schedules)
    intact_trace = readout["intact_trace"]
    condition_bundles = readout["condition_bundles"]
    intact_bundle = condition_bundles["intact"]
    a_off_bundle = condition_bundles["a_off"]
    fields = {
        name: margin_fields(bundle, protocol.n_items)
        for name, bundle in condition_bundles.items()
    }
    loo_global_fields = np.asarray(
        [
            margin_fields(bundle, protocol.n_items)
            for bundle in readout["global_loo_bundles"]
        ]
    )
    loo_p_off_fields = np.asarray(
        [
            margin_fields(bundle, protocol.n_items)
            for bundle in readout["local_loo_bundles"]
        ]
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
    retained = retained_relation_mask(evaluator, relations)
    correct_probability = stable_sigmoid(
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
            "retained_counts_per_subject": json_values(np.sum(retained, axis=0)),
            "omitted_counts_per_subject": json_values(np.sum(~retained, axis=0)),
            "retained_correct_probability": summarize_subjects(
                masked_subject_mean(retained_values), counts, interval=interval
            ),
            "omitted_correct_probability": summarize_subjects(
                masked_subject_mean(omitted_values), counts, interval=interval
            ),
        },
        "raw_relation_subject_local_direct_correctness": json_values(local_direct),
        "sampled_behavior": behavior,
        "sampled_accuracy_bootstrap": sampled_accuracy,
        "serial_position_endpoint": serial_position_endpoint(behavior, protocol),
    }
    presentation = measure_presentation_invariance(
        evaluator, local, intact_trace.natural_scalars
    )
    local_off_error = float(
        np.max(np.abs(a_off_bundle["logits"] - a_off_bundle["global_logits"]))
    )
    after = tensor_hashes(backbone)
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
        "primary_values_finite": finite_primary(metrics),
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
        "support_schedule_sha256": support_schedule_hash(evaluator),
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
        backbone, model_config, checkpoint = load_frozen_retro_checkpoint(
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
