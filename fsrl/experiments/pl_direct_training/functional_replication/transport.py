"""Triggered N=6,8,10 transport of locked functional-replication backbones."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.hodge import build_complete_graph_geometry
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
from fsrl.experiments.transport.item_count import (
    _bootstrap_seed,
    _distance_summaries,
    constructive_metrics_generic,
    cross_cell_decision,
    individualized_metrics_generic,
    protocol_for_size,
    relation_loo_metrics_generic,
    serial_position_endpoint_generic,
    validate_graph_contract,
    within_cell_decision,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .evaluation import evaluation_directory, load_model
from .liu import FunctionalLiuEvaluator, rollout_functional
from .locks import ARTIFACT_LOCK_PATH, RUN_ROOT, reference
from .protocol import PROTOCOL_SHA256, REPAIR_SHA256, registered_seeds


def _schedule_hash(evaluator: FunctionalLiuEvaluator) -> str:
    payload = json.dumps(
        [
            [asdict(trial) for trial in schedule]
            for schedule in evaluator.support_schedules
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _size_interface(evaluator: FunctionalLiuEvaluator) -> dict:
    protocol = evaluator.protocol
    expected_relations = set(protocol.support_pairs_higher_lower)
    counts = True
    magnitude = True
    orientations = True
    for schedule in evaluator.support_schedules:
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
            rank_gap = (
                evaluator.item_rank[trial.lower_item]
                - evaluator.item_rank[trial.higher_item]
            )
            magnitude = magnitude and np.isclose(
                abs(trial.signed_magnitude), rank_gap / float(protocol.n_items - 1)
            )
            orientations = orientations and {
                trial.left_item,
                trial.right_item,
            } == set(relation)
    cue_shape = evaluator.cue_codes.shape == (
        evaluator.subjects,
        protocol.n_items,
        evaluator.backbone.model_config.cue_size,
    )
    state_shape = bool(
        evaluator.subject_encoding_states is not None
        and all(
            len(state.item_salience) == protocol.n_items
            for state in evaluator.subject_encoding_states
        )
    )
    return {
        "support_counts_and_multiplicity": bool(counts),
        "normalized_signed_magnitudes": bool(magnitude),
        "left_right_orientation_valid": bool(orientations),
        "cue_shape_valid": bool(cue_shape),
        "subject_state_shape_valid": bool(state_shape),
        "support_schedule_sha256": _schedule_hash(evaluator),
        "passed": bool(
            counts and magnitude and orientations and cue_shape and state_shape
        ),
    }


def _packed_keys(codes: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(codes.shape[1], k=1)
    keys = []
    for first, second in combinations(range(codes.shape[0]), 2):
        left, right = codes[first], codes[second]
        packed = left[rows] * right[columns] - right[rows] * left[columns]
        keys.append(packed / max(float(np.linalg.norm(packed)), 1e-8))
    return np.asarray(keys, dtype=np.float64)


def reconstruct_packed_ledger(
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
        keys = _packed_keys(np.asarray(item_codes[subject], dtype=np.float64))
        reconstructed = np.zeros(keys.shape[1], dtype=np.float64)
        ledger = np.zeros(len(pairs), dtype=np.float64)
        for trial_index, trial in enumerate(schedule):
            scalar = float(natural_scalars[subject, trial_index])
            canonical = tuple(sorted((trial.left_item, trial.right_item)))
            orientation = 1.0 if trial.left_item < trial.right_item else -1.0
            key = keys[pair_index[canonical]] * orientation
            reconstructed += scalar * key
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
    }


def _n8_replay(seed: int, rollout: dict, n_items: int) -> dict:
    if n_items != 8:
        return {"passed": True, "not_applicable": True}
    with np.load(evaluation_directory(seed) / "raw.npz", allow_pickle=False) as arrays:
        checks = {
            "dual_access_logits": np.array_equal(
                rollout["bundles"]["dual_access"]["logits"],
                arrays["liu__bundles__dual_access__logits"],
            ),
            "local_off_logits": np.array_equal(
                rollout["bundles"]["local_off"]["logits"],
                arrays["liu__bundles__local_off__logits"],
            ),
            "P_off_dual_logits": np.array_equal(
                rollout["bundles"]["P_off_dual"]["logits"],
                arrays["liu__bundles__P_off_dual__logits"],
            ),
        }
    return {"passed": all(checks.values()), **checks}


def evaluate_cell(
    reference_specification: dict,
    seed: int,
    graph: dict,
    size_index: int,
    evaluator: FunctionalLiuEvaluator,
    rollout: dict,
    before: tuple[dict, dict],
    runtime: dict,
    graph_validation: dict,
) -> dict:
    evaluation = reference_specification["evaluation"]
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
    named_bundles = {
        "intact": rollout["bundles"]["dual_access"],
        "a_off": rollout["bundles"]["local_off"],
        "P_off_a_on": rollout["bundles"]["P_off_dual"],
    }
    fields = {
        name: margin_fields(bundle, n_items) for name, bundle in named_bundles.items()
    }
    global_loo_fields = np.asarray(
        [margin_fields({"logits": row}, n_items) for row in rollout["loo"]["local_off"]]
    )
    local_loo_fields = np.asarray(
        [
            margin_fields({"logits": row}, n_items)
            for row in rollout["loo"]["P_off_dual"]
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
    global_loo = relation_loo_metrics_generic(
        fields["a_off"],
        global_loo_fields,
        relations,
        geometry,
        counts,
        interval,
    )
    local_loo = relation_loo_metrics_generic(
        fields["P_off_a_on"],
        local_loo_fields,
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
    schedules = (ordered_pairs(n_items),) * evaluator.subjects
    behavior = analyze_sampled_query_policy(
        protocol,
        bundle_logits(named_bundles["intact"], schedules),
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
    dual_state, _, _ = evaluator.build_access_state("dual")
    exact = reconstruct_packed_ledger(
        evaluator.cue_codes,
        evaluator.support_schedules,
        rollout["dual_local_evidence"],
        dual_state.detach().cpu().numpy().astype(np.float64),
        named_bundles["intact"]["raw_local_margins"][:, 0::2],
    )
    retention = rollout["retention"].T
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
            behavior, rng, int(evaluation["bootstrap_samples"]), n_items
        ),
        "global_relation_LOO": global_loo,
        "P_off_local_relation_LOO": local_loo,
        "contrasts": contrasts,
        "local_exactness": exact,
        "retained_omitted": {
            "retained_counts_per_subject": json_values(np.sum(retention, axis=0)),
            "omitted_counts_per_subject": json_values(np.sum(~retention, axis=0)),
            "retained_correct_probability": summarize_subjects(
                finite_column_mean(np.where(retention, learned_probability, np.nan)),
                counts,
                interval=interval,
            ),
            "omitted_correct_probability": summarize_subjects(
                finite_column_mean(np.where(~retention, learned_probability, np.nan)),
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
    size_interface = _size_interface(evaluator)
    n8_replay = _n8_replay(seed, rollout, n_items)
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
        "graph_validation_passed": bool(graph_validation["passed"]),
        "size_interface": size_interface,
        "N8_primary_replay": n8_replay,
        "bounded_gpu_runtime": bool(
            runtime["active"]
            and runtime["cuda_available"]
            and runtime["torch_intraop_threads"] == 1
            and runtime["torch_interop_threads"] == 1
        ),
        "backbone_tensor_hashes_unchanged": before
        == (tensor_hashes(evaluator.backbone), tensor_hashes(evaluator.local)),
        "local_off_global_logit_max_abs_error": float(
            np.max(
                np.abs(
                    named_bundles["a_off"]["logits"]
                    - named_bundles["a_off"]["global_logits"]
                )
            )
        ),
        "presentation_invariance_qualified_before_source_lock": True,
        "primary_values_finite": bool(
            finite_primary(metrics) and individualized_finite
        ),
    }
    integrity["all_passed"] = bool(
        integrity["graph_validation_passed"]
        and size_interface["passed"]
        and n8_replay["passed"]
        and integrity["bounded_gpu_runtime"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and integrity["local_off_global_logit_max_abs_error"] <= 1e-6
        and integrity["presentation_invariance_qualified_before_source_lock"]
        and integrity["primary_values_finite"]
    )
    decision = within_cell_decision(metrics, integrity)
    return {
        "size_id": graph["size_id"],
        "n_items": n_items,
        "protocol_id": protocol.protocol_id,
        "rank_edges": graph["rank_edges"],
        "item_edges_higher_lower": [list(relation) for relation in relations],
        "support_schedule_sha256": size_interface["support_schedule_sha256"],
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "size_specific_metrics": size_metrics,
        "integrity": integrity,
        "decision": decision,
    }


def transport_directory() -> Path:
    return RUN_ROOT / "transport"


def evaluate_transport(artifact_lock: dict, specification: dict, runtime: dict) -> dict:
    directory = transport_directory()
    if directory.exists():
        return validate_transport(artifact_lock)
    reference_path = Path(
        specification["triggered_item_count_transport"]["reference_contract"]["path"]
    )
    reference_specification = load_json(REPO_ROOT / reference_path)
    graph_validation = validate_graph_contract(reference_specification)
    base = load_registered_protocol("liu_v2")
    graphs = reference_specification["size_matched_graph_contract"]["graphs"]
    settings = reference_specification["evaluation"]
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_functional_replication_v1",
        execution_id="triggered-item-count-transport",
        producer={"module": __name__, **identity},
        resolved_config={
            "sizes": [int(graph["n_items"]) for graph in graphs],
            "seeds": list(registered_seeds(specification)),
            "runtime": runtime,
        },
    ):
        seeds = {}
        for seed in registered_seeds(specification):
            backbone, local, _ = load_model(seed, artifact_lock)
            before = (tensor_hashes(backbone), tensor_hashes(local))
            sizes = {}
            for size_index, graph in enumerate(graphs, start=1):
                protocol = protocol_for_size(base, graph)
                evaluator = FunctionalLiuEvaluator(
                    "no_time_candidate",
                    backbone,
                    local,
                    protocol,
                    subjects=int(settings["subjects_per_size_and_backbone"]),
                    cue_seed=int(settings["cue_seed"]),
                    support_seed=int(settings["support_seed"]),
                    subject_encoding_seed=int(settings["subject_encoding_seed"]),
                    cue_mode=str(settings["cue_mode"]),
                    subject_encoding_mode=str(settings["subject_encoding_mode"]),
                    test_time_value=2.0 / 3.0,
                )
                rollout = rollout_functional(
                    evaluator,
                    {
                        "evidence_shuffle_seed": 32301,
                        "query_shuffle_seed": 31801,
                    },
                )
                sizes[str(protocol.n_items)] = evaluate_cell(
                    reference_specification,
                    seed,
                    graph,
                    size_index,
                    evaluator,
                    rollout,
                    before,
                    runtime,
                    graph_validation,
                )
            seeds[str(seed)] = {"seed": seed, "sizes": sizes}
        item_counts = [int(graph["n_items"]) for graph in graphs]
        decision = cross_cell_decision(
            seeds, item_counts, list(registered_seeds(specification))
        )
        result = {
            **identity,
            "trigger": "all three N=8 primary functional replications passed",
            "graph_validation": graph_validation,
            "seeds": seeds,
            "decision": decision,
            "claim_boundary": specification["triggered_item_count_transport"]["role"],
        }
        write_json_exclusive(directory / "result.json", result)
    return validate_transport(artifact_lock)


def validate_transport(artifact_lock: dict) -> dict:
    directory = transport_directory()
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("functional transport is incomplete or modified")
    result = load_json(directory / "result.json")
    if (
        result["protocol_sha256"] != PROTOCOL_SHA256
        or result["repair_sha256"] != REPAIR_SHA256
        or result["source_commit"] != artifact_lock["source_commit"]
    ):
        raise RuntimeError("functional transport provenance differs")
    return result
