"""Registered one-seed curvature-gate sufficiency pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    hodge_potentials,
    normalize_potentials,
    potential_alignment,
)
from fsrl.analysis.statistics import (
    json_values,
    masked_column_mean,
    summarize_difference,
    summarize_subjects,
)
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
    load_training_provenance,
    retained_relation_mask,
    run_causal_suite,
)
from fsrl.evaluation.qualification import evaluate_qualification
from fsrl.experiments.assembly.trajectory import exact_prefix_trajectory
from fsrl.experiments.local_fidelity.amplitude_path import collect_amplitude_fields
from fsrl.experiments.local_fidelity.curvature_gate import (
    CurvatureGateTransition,
    make_gate_tasks,
    run_gate_batch,
)
from fsrl.infra.formal_runtime import formal_runtime_snapshot
from fsrl.infra.provenance import load_json, tensor_hashes, write_json
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.registered_protocol import load_ranking_protocol
from fsrl.training.backbone import MetaTrainConfig, train_meta_model

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/curvature_gate_pilot_v2.json")
DEFAULT_LOCK_PATH = resolve_record("benchmarks/curvature_gate_pilot_v2.lock.json")
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "curvature-gate-pilot-v2"
DEFAULT_RESULT_PATH = resolve_record("results/curvature_gate_pilot_v2.json")
CONDITIONS = (
    "original_v1",
    "conditioned_gate",
    "matched_global_scalar",
    "shuffled_gate",
)


def configure_runtime() -> dict:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    snapshot = formal_runtime_snapshot()
    if not snapshot["cuda_available"]:
        raise RuntimeError("curvature-gate pilot requires a visible CUDA GPU")
    if snapshot["torch_intraop_threads"] != 1 or snapshot["torch_interop_threads"] != 1:
        raise RuntimeError("curvature-gate pilot requires one PyTorch CPU thread")
    return snapshot


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict:
    specification_path = specification_path.resolve()
    specification = load_json(specification_path)
    checks = []
    for name, registration in specification["registered_sources"].items():
        path = resolve_registered_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": registration["sha256"],
                "passed": observed == registration["sha256"],
            }
        )
    lock = load_json(lock_path)
    protocol_hash = file_sha256(specification_path)
    checks.append(
        {
            "name": "pilot_specification",
            "path": legacy_identifier(specification_path),
            "observed": protocol_hash,
            "expected": lock["pilot_specification_sha256"],
            "passed": protocol_hash == lock["pilot_specification_sha256"],
        }
    )
    for name, registration in lock["implementation_sources"].items():
        path = resolve_registered_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": registration["sha256"],
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"curvature-gate source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _backbone_training_config(specification: dict) -> MetaTrainConfig:
    registered = dict(specification["v1_backbone_training"])
    for name in ("held_out_graph", "architecture"):
        registered.pop(name)
    return MetaTrainConfig(**registered)


def train_backbone(specification: dict, output_root: Path, runtime: dict) -> Path:
    training = _backbone_training_config(specification)
    seed_dir = output_root / f"seed-{training.seed}" / "backbone"
    checkpoint = seed_dir / "net.dat"
    if not checkpoint.exists():
        train_meta_model(training, seed_dir, compile_model=True)
    metadata = load_json(seed_dir / "config.json")
    if metadata["training"] != asdict(training):
        raise RuntimeError("backbone training configuration mismatch")
    if metadata["completed_outer_steps"] != training.outer_steps:
        raise RuntimeError("backbone is not the registered final step")
    if metadata["checkpoint"]["sha256"] != checkpoint_sha256(checkpoint):
        raise RuntimeError("backbone checkpoint hash mismatch")
    manifest = {
        "runtime": runtime,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha256(checkpoint),
        },
        "training": asdict(training),
    }
    write_json(seed_dir / "pilot_manifest.json", manifest)
    return checkpoint


def adaptation_config(specification: dict) -> MetaTrainConfig:
    backbone = specification["v1_backbone_training"]
    adaptation = specification["gate_only_adaptation"]
    return MetaTrainConfig(
        seed=int(adaptation["adaptation_rng_seed"]),
        outer_steps=int(adaptation["outer_steps"]),
        batch_size=int(adaptation["batch_size"]),
        hidden_size=int(backbone["hidden_size"]),
        cue_size=int(backbone["cue_size"]),
        min_edges=int(backbone["min_edges"]),
        max_edges=int(backbone["max_edges"]),
        support_blocks=int(backbone["support_blocks"]),
        learning_rate=float(adaptation["learning_rate"]),
        gradient_clip=float(adaptation["gradient_clip"]),
        fast_weight_penalty=0.0,
        support_query_time=float(backbone["support_query_time"]),
        save_every=int(adaptation["outer_steps"]),
        subject_encoding_mode=str(backbone["subject_encoding_mode"]),
    )


def calibrate_global_gamma(
    specification: dict,
    backbone,
    model_config,
    gate,
    *,
    compile_models: bool,
) -> dict:
    adaptation = adaptation_config(specification)
    calibration = specification["matched_global_calibration"]
    calibration_config = MetaTrainConfig(
        **{
            **asdict(adaptation),
            "seed": int(calibration["calibration_rng_seed"]),
            "outer_steps": int(calibration["batches"]),
        }
    )
    rng = np.random.default_rng(calibration_config.seed)
    task_generator = make_gate_tasks(calibration_config)
    replay_backbone = (
        torch.compile(backbone, fullgraph=True) if compile_models else backbone
    )
    replay_gate = torch.compile(gate, fullgraph=True) if compile_models else gate
    gamma_sum = 0.0
    risk_sum = 0.0
    gamma_count = 0
    with torch.no_grad():
        for _ in range(int(calibration["batches"])):
            stats_row = run_gate_batch(
                calibration_config,
                model_config,
                replay_backbone,
                replay_gate,
                task_generator,
                rng,
            )
            gamma_sum += stats_row.gamma_sum
            risk_sum += stats_row.risk_sum
            gamma_count += stats_row.gamma_count
    return {
        "gamma_global": gamma_sum / gamma_count,
        "gamma_sum": gamma_sum,
        "gamma_count": gamma_count,
        "mean_risk": risk_sum / gamma_count,
        "rng_seed": calibration_config.seed,
        "batches": int(calibration["batches"]),
    }


def adapt_gate(
    specification: dict,
    checkpoint: Path,
    output_root: Path,
    source_validation: dict,
    runtime: dict,
) -> Path:
    adaptation = adaptation_config(specification)
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, adaptation.batch_size
    )
    before = tensor_hashes(backbone)
    gate_specification = specification["gate_equation"]
    gate = CurvatureGateTransition(
        backbone,
        epsilon=float(gate_specification["epsilon"]),
        initial_beta=float(gate_specification["initial_beta"]),
    )
    task_generator = make_gate_tasks(adaptation)
    rng = np.random.default_rng(adaptation.seed)
    np.random.seed(adaptation.seed)
    torch.manual_seed(adaptation.seed)
    training_backbone = torch.compile(backbone, fullgraph=True)
    training_gate = torch.compile(gate, fullgraph=True)
    optimizer = torch.optim.Adam(
        [gate.raw_beta],
        lr=float(specification["gate_only_adaptation"]["learning_rate"]),
    )
    seed_dir = (
        output_root
        / f"seed-{specification['development_seed_contract']['mandatory_seed']}"
    )
    gate_dir = seed_dir / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    log_path = gate_dir / "train_log.jsonl"
    if log_path.exists():
        raise RuntimeError("gate adaptation log already exists; refusing to append")
    for step in range(adaptation.outer_steps):
        optimizer.zero_grad()
        stats_row = run_gate_batch(
            adaptation,
            model_config,
            training_backbone,
            training_gate,
            task_generator,
            rng,
        )
        stats_row.loss.backward()
        torch.nn.utils.clip_grad_norm_([gate.raw_beta], adaptation.gradient_clip)
        optimizer.step()
        record = {
            "outer_step": step,
            "query_cross_entropy": stats_row.query_cross_entropy,
            "query_accuracy": stats_row.query_accuracy,
            "beta": float(gate.beta.detach()),
            "mean_gamma": stats_row.gamma_sum / stats_row.gamma_count,
            "mean_risk": stats_row.risk_sum / stats_row.gamma_count,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    after = tensor_hashes(backbone)
    if before != after:
        raise RuntimeError("frozen backbone tensors changed during gate adaptation")
    calibration = calibrate_global_gamma(
        specification,
        backbone,
        model_config,
        gate,
        compile_models=True,
    )
    artifact = {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "seed": int(specification["development_seed_contract"]["mandatory_seed"]),
        "backbone": asdict(checkpoint_info),
        "adaptation": asdict(adaptation),
        "raw_beta": float(gate.raw_beta.detach()),
        "beta": float(gate.beta.detach()),
        "calibration": calibration,
        "backbone_tensor_hashes_before": before,
        "backbone_tensor_hashes_after": after,
        "source_validation": source_validation,
        "runtime": runtime,
    }
    artifact_path = gate_dir / "gate.json"
    write_json(artifact_path, artifact)
    return artifact_path


def query_pass(
    evaluator,
    gate,
    fast_weights,
    pair_schedules,
    *,
    gamma_overrides: np.ndarray | None,
    alpha_zero: bool,
) -> dict:
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    logits = np.empty((subjects, pair_count), dtype=np.float64)
    risks = np.empty_like(logits)
    conditioned_gammas = np.empty_like(logits)
    applied_gammas = np.empty_like(logits)
    with torch.no_grad(), evaluator._alpha_zeroed(alpha_zero):
        for pair_index in range(pair_count):
            hidden = evaluator.net.initialZeroState(subjects)
            eligibility = evaluator.net.initialZeroET(subjects)
            left = np.asarray(
                [schedule[pair_index][0] for schedule in pair_schedules],
                dtype=np.int64,
            )
            right = np.asarray(
                [schedule[pair_index][1] for schedule in pair_schedules],
                dtype=np.int64,
            )
            signed = np.zeros(subjects, dtype=np.float32)
            step0 = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=0,
                time_value=evaluator.test_time_value,
                support_trial=False,
            )
            response = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=1,
                time_value=evaluator.test_time_value,
                support_trial=False,
            )
            _, _, _, hidden, eligibility, _ = evaluator.net(
                step0, hidden, eligibility, fast_weights
            )
            override = None
            if gamma_overrides is not None:
                override = torch.from_numpy(
                    gamma_overrides[:, pair_index].astype(np.float32)
                ).to(fast_weights.device)[:, None]
            output = gate(response, hidden, eligibility, fast_weights, override)
            logits[:, pair_index] = (
                (output[0][:, 1] - output[0][:, 0]).detach().cpu().numpy()
            )
            risks[:, pair_index] = output[6][:, 0].detach().cpu().numpy()
            conditioned_gammas[:, pair_index] = output[7][:, 0].detach().cpu().numpy()
            applied_gammas[:, pair_index] = output[8][:, 0].detach().cpu().numpy()
    return {
        "logits": logits,
        "risks": risks,
        "conditioned_gammas": conditioned_gammas,
        "applied_gammas": applied_gammas,
    }


def query_bundle(
    evaluator,
    gate,
    fast_weights,
    pair_schedules,
    *,
    condition: str,
    gamma_global: float,
    shuffle_seed: int,
    alpha_zero: bool = False,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown gate condition: {condition}")
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    if condition == "conditioned_gate":
        return query_pass(
            evaluator,
            gate,
            fast_weights,
            pair_schedules,
            gamma_overrides=None,
            alpha_zero=alpha_zero,
        )
    if condition == "original_v1":
        overrides = np.ones((subjects, pair_count), dtype=np.float64)
    elif condition == "matched_global_scalar":
        overrides = np.full((subjects, pair_count), gamma_global, dtype=np.float64)
    else:
        natural = query_pass(
            evaluator,
            gate,
            fast_weights,
            pair_schedules,
            gamma_overrides=None,
            alpha_zero=alpha_zero,
        )
        permutations = np.stack(
            [
                np.random.default_rng(shuffle_seed + subject).permutation(pair_count)
                for subject in range(subjects)
            ]
        )
        overrides = np.take_along_axis(
            natural["conditioned_gammas"], permutations, axis=1
        )
    return query_pass(
        evaluator,
        gate,
        fast_weights,
        pair_schedules,
        gamma_overrides=overrides,
        alpha_zero=alpha_zero,
    )


def bundle_logits(bundle: dict, pair_schedules) -> tuple[dict, ...]:
    return tuple(
        {
            pair: float(bundle["logits"][subject, index])
            for index, pair in enumerate(pair_schedules[subject])
        }
        for subject in range(len(pair_schedules))
    )


def margin_fields(bundle: dict, n_items: int) -> np.ndarray:
    return 0.5 * (bundle["logits"][:, 0::2] - bundle["logits"][:, 1::2])


def field_metrics(
    fields: dict[str, np.ndarray],
    relations,
    retained: np.ndarray,
    geometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    intact = fields["intact"]
    loo = fields["loo"]
    influence = intact[None] - loo
    gradient = np.einsum("ef,rsf->rse", geometry.projection, influence)
    residual = influence - gradient
    direct_correctness = np.empty(retained.shape, dtype=np.float64)
    remote_absolute = np.empty_like(direct_correctness)
    third_party = np.empty_like(direct_correctness)
    intact_potential = hodge_potentials(intact, geometry)
    for relation_index, relation in enumerate(relations):
        direct_pair = tuple(sorted(relation))
        direct_edge = geometry.pairs.index(direct_pair)
        direct_correctness[relation_index] = (
            residual[relation_index, :, direct_edge] * geometry.true_sign[direct_edge]
        )
        endpoints = set(relation)
        remote_mask = np.asarray(
            [not endpoints.intersection(pair) for pair in geometry.pairs], dtype=bool
        )
        remote_absolute[relation_index] = np.mean(
            np.abs(influence[relation_index][:, remote_mask]), axis=1
        )
        loo_potential = hodge_potentials(loo[relation_index], geometry)
        delta = intact_potential - loo_potential
        denominator = np.sum(delta * delta, axis=1)
        third_items = np.asarray(
            [
                item
                for item in range(len(geometry.true_potential))
                if item not in relation
            ]
        )
        third_delta = delta[:, third_items]
        relational = third_delta - np.mean(third_delta, axis=1, keepdims=True)
        numerator = np.sum(relational * relational, axis=1)
        third_party[relation_index] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-14,
        )

    aggregate = masked_column_mean(direct_correctness, retained)
    remote = masked_column_mean(remote_absolute, retained)
    third = masked_column_mean(third_party, retained)
    primary_index = relations.index((7, 0))
    primary = np.where(
        retained[primary_index], direct_correctness[primary_index], np.nan
    )
    other_mask = retained.copy()
    other_mask[primary_index] = False
    other = masked_column_mean(direct_correctness, other_mask)
    omitted_selector = np.broadcast_to((~retained)[..., None], influence.shape)
    omitted_max = (
        float(np.max(np.abs(influence[omitted_selector]))) if np.any(~retained) else 0.0
    )
    return {
        "summary": {
            "retained_relation_mean_direct_correctness": summarize_subjects(
                aggregate, counts, interval=interval
            ),
            "H_greater_A_direct_correctness": summarize_subjects(
                primary, counts, interval=interval
            ),
            "other_relation_mean_direct_correctness": summarize_subjects(
                other, counts, interval=interval
            ),
            "remote_absolute": summarize_subjects(remote, counts, interval=interval),
            "gauge_invariant_R_third_rel": summarize_subjects(
                third, counts, interval=interval
            ),
            "stable_omitted_max_abs_pair_influence": omitted_max,
        },
        "subject_level": {
            "retained_relation_mean_direct_correctness": aggregate,
            "H_greater_A_direct_correctness": primary,
            "other_relation_mean_direct_correctness": other,
            "remote_absolute": remote,
            "gauge_invariant_R_third_rel": third,
        },
        "raw_relation_subject": {
            "retained": retained.astype(int).tolist(),
            "direct_correctness": json_values(direct_correctness),
            "remote_absolute": json_values(remote_absolute),
            "gauge_invariant_R_third_rel": json_values(third_party),
        },
    }


def _condition_metrics(protocol, bundle: dict, fast_weights, intervention: str):
    logits = bundle_logits(
        bundle,
        tuple(
            ordered_pairs(protocol.n_items) for _ in range(bundle["logits"].shape[0])
        ),
    )
    canonical = tuple(combinations(range(protocol.n_items), 2))
    learned = protocol.learned_pairs
    positions = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    overall_rows = []
    learned_rows = []
    nonlearned_rows = []
    probability_rows = []
    cycle_rows = []
    winner_rows = []
    for subject_logits in logits:
        correct = []
        correct_learned = []
        correct_nonlearned = []
        probabilities = []
        winners = {}
        for pair in canonical:
            forward = subject_logits[pair]
            reverse = subject_logits[(pair[1], pair[0])]
            first_higher = positions[pair[0]] < positions[pair[1]]
            pair_correct = (
                (float(forward > 0.0), float(reverse < 0.0))
                if first_higher
                else (float(forward < 0.0), float(reverse > 0.0))
            )
            correct.extend(pair_correct)
            (correct_learned if pair in learned else correct_nonlearned).extend(
                pair_correct
            )
            correct_forward = forward if first_higher else -forward
            correct_reverse = -reverse if first_higher else reverse
            probabilities.extend(
                [
                    float(1.0 / (1.0 + np.exp(-correct_forward))),
                    float(1.0 / (1.0 + np.exp(-correct_reverse))),
                ]
            )
            winners[pair] = pair[0] if 0.5 * (forward - reverse) > 0.0 else pair[1]
        cycles = 0
        for a, b, c in combinations(range(protocol.n_items), 3):
            ab, ac, bc = winners[(a, b)], winners[(a, c)], winners[(b, c)]
            cycles += int(
                (ab == a and bc == b and ac == c) or (ab == b and bc == c and ac == a)
            )
        overall_rows.append(np.mean(correct))
        learned_rows.append(np.mean(correct_learned))
        nonlearned_rows.append(np.mean(correct_nonlearned))
        probability_rows.append(np.mean(probabilities))
        cycle_rows.append(cycles)
        winner_rows.append(winners)
    n_triads = len(tuple(combinations(range(protocol.n_items), 3)))
    return {
        "intervention": intervention,
        "overall_accuracy": float(np.mean(overall_rows)),
        "learned_accuracy": float(np.mean(learned_rows)),
        "nonlearned_accuracy": float(np.mean(nonlearned_rows)),
        "mean_probability_correct": float(np.mean(probability_rows)),
        "mean_abs_fast_weight": float(torch.mean(torch.abs(fast_weights)).cpu()),
        "mean_circular_triads": float(np.mean(cycle_rows)),
        "mean_transitive_triplet_fraction": float(1.0 - np.mean(cycle_rows) / n_triads),
    }, tuple(winner_rows)


def conditioned_causal_suite(
    checkpoint: Path,
    evaluator,
    gate,
    gamma_global: float,
    specification: dict,
) -> dict:
    evaluation = specification["liu_evaluation"]
    protocol = evaluator.protocol
    pairs = ordered_pairs(protocol.n_items)
    schedules = tuple(pairs for _ in range(evaluator.config.bs))
    conditions = {}
    winners = {}
    for intervention in FastWeightIntervention:
        fast_weights = evaluator.learn_fast_weights(intervention)
        bundle = query_bundle(
            evaluator,
            gate,
            fast_weights,
            schedules,
            condition="conditioned_gate",
            gamma_global=gamma_global,
            shuffle_seed=int(evaluation["shuffle_seed"]),
            alpha_zero=intervention == FastWeightIntervention.ALPHA_ZERO,
        )
        conditions[intervention.value], winners[intervention.value] = (
            _condition_metrics(protocol, bundle, fast_weights, intervention.value)
        )
    intact_winners = winners[FastWeightIntervention.INTACT.value]
    for name, subject_rows in winners.items():
        agreements = []
        for subject, row in enumerate(subject_rows):
            agreements.extend(
                int(winner == intact_winners[subject][pair])
                for pair, winner in row.items()
            )
        conditions[name]["mean_pair_decision_agreement_to_intact"] = float(
            np.mean(agreements)
        )

    intact = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    rng = np.random.default_rng(int(evaluation["order_seed"]))
    order_runs = []
    for _ in range(8):
        order = rng.permutation(len(pairs))
        schedule = tuple(pairs[int(index)] for index in order)
        current_schedules = tuple(schedule for _ in range(evaluator.config.bs))
        current = query_bundle(
            evaluator,
            gate,
            intact,
            current_schedules,
            condition="conditioned_gate",
            gamma_global=gamma_global,
            shuffle_seed=int(evaluation["shuffle_seed"]),
        )
        order_runs.append(bundle_logits(current, current_schedules))
    deltas = []
    for current in order_runs[1:]:
        for subject in range(evaluator.config.bs):
            for pair in pairs:
                deltas.append(
                    abs(current[subject][pair] - order_runs[0][subject][pair])
                )
    provenance = load_training_provenance(checkpoint, checkpoint_sha256(checkpoint))
    return {
        "cue_mode": evaluation["cue_mode"],
        "subject_encoding": {
            "mode": evaluation["subject_encoding_mode"],
            "seed": evaluation["subject_encoding_seed"],
        },
        "training_provenance": provenance,
        "conditions": conditions,
        "order_invariance": {
            "max_abs_logit_delta": float(max(deltas, default=0.0)),
            "mean_abs_logit_delta": float(np.mean(deltas)) if deltas else 0.0,
        },
    }


def query_binding_summary(
    evaluator,
    intact,
    loo,
    retained: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    effective = evaluator.net.alpha.detach() * (intact[None] - loo)
    relation_count = len(relations)
    subjects = evaluator.config.bs
    normalized = np.empty((relation_count, subjects, relation_count), dtype=np.float64)
    operator_norm = torch.linalg.vector_norm(
        effective.reshape(relation_count, subjects, -1), dim=2
    )
    with torch.no_grad():
        for query_index, relation in enumerate(relations):
            orientation_rows = []
            for pair in (relation, (relation[1], relation[0])):
                left = np.full(subjects, pair[0], dtype=np.int64)
                right = np.full(subjects, pair[1], dtype=np.int64)
                signed = np.zeros(subjects, dtype=np.float32)
                inputs = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                h0 = torch.tanh(evaluator.net.i2h(inputs))
                action = torch.matmul(
                    effective, h0.view(1, subjects, evaluator.config.hs, 1)
                )[..., 0]
                denominator = operator_norm * torch.linalg.vector_norm(h0, dim=1)[None]
                orientation_rows.append(
                    torch.divide(
                        torch.linalg.vector_norm(action, dim=2),
                        denominator,
                    )
                    .cpu()
                    .numpy()
                )
            normalized[:, :, query_index] = np.mean(orientation_rows, axis=0)

    shared_rows = np.empty(retained.shape, dtype=np.float64)
    disjoint_rows = np.empty(retained.shape, dtype=np.float64)
    for state_index, state_relation in enumerate(relations):
        overlaps = np.asarray(
            [len(set(state_relation).intersection(query)) for query in relations]
        )
        matched = normalized[state_index, :, state_index]
        shared_rows[state_index] = matched - np.nanmean(
            normalized[state_index][:, overlaps == 1], axis=1
        )
        disjoint_rows[state_index] = matched - np.nanmean(
            normalized[state_index][:, overlaps == 0], axis=1
        )
    shared = masked_column_mean(shared_rows, retained)
    disjoint = masked_column_mean(disjoint_rows, retained)
    summaries = {
        "matched_minus_shared_endpoint": summarize_subjects(
            shared, counts, interval=interval
        ),
        "matched_minus_disjoint": summarize_subjects(
            disjoint, counts, interval=interval
        ),
    }
    return {
        "original_v1": summaries,
        "conditioned_gate": summaries,
        "conditioned_minus_original_max_abs": 0.0,
        "raw_subject_level": {
            "matched_minus_shared_endpoint": json_values(shared),
            "matched_minus_disjoint": json_values(disjoint),
        },
    }


def crossing_alignment(
    evaluator,
    intact,
    loo,
    retained: np.ndarray,
    conditioned_bundle: dict,
    geometry,
) -> dict:
    amplitudes = np.linspace(0.0, 1.0, 21)
    effective = evaluator.net.alpha.detach() * (intact[None] - loo)
    fields, validation = collect_amplitude_fields(
        evaluator,
        evaluator.protocol,
        geometry,
        intact,
        loo,
        effective,
        retained,
        amplitudes,
        tolerance=2.0**-17,
    )
    residuals = fields["curve_residuals"]
    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    direct = np.empty(
        (len(amplitudes), len(relations), evaluator.config.bs), dtype=np.float64
    )
    gammas = np.empty((len(relations), evaluator.config.bs), dtype=np.float64)
    ordered = ordered_pairs(evaluator.protocol.n_items)
    pair_index = {pair: index for index, pair in enumerate(ordered)}
    for relation_index, relation in enumerate(relations):
        edge = geometry.pairs.index(tuple(sorted(relation)))
        direct[:, relation_index] = (
            residuals[:, relation_index, :, edge] * geometry.true_sign[edge]
        )
        forward = pair_index[tuple(relation)]
        reverse = pair_index[(relation[1], relation[0])]
        gammas[relation_index] = 0.5 * (
            conditioned_bundle["conditioned_gammas"][:, forward]
            + conditioned_bundle["conditioned_gammas"][:, reverse]
        )
    midpoints = np.full(retained.shape, np.nan, dtype=np.float64)
    for relation in range(len(relations)):
        for subject in range(evaluator.config.bs):
            if not retained[relation, subject]:
                continue
            curve = direct[:, relation, subject]
            for index in range(1, len(amplitudes) - 1):
                if curve[index] > 0.0 and curve[index + 1] <= 0.0:
                    midpoints[relation, subject] = 0.5 * (
                        amplitudes[index] + amplitudes[index + 1]
                    )
                    break
    selector = np.isfinite(midpoints)
    if np.sum(selector) >= 3:
        correlation = stats.spearmanr(gammas[selector], midpoints[selector])
        rho = float(correlation.statistic)
        pvalue = float(correlation.pvalue)
    else:
        rho = None
        pvalue = None
    return {
        "crossing_cases": int(np.sum(selector)),
        "retained_cases": int(np.sum(retained)),
        "spearman_rho": rho,
        "pvalue": pvalue,
        "orientation_reduction": "mean conditioned gamma over the two direct orientations",
        "validation": validation,
        "raw": {
            "crossing_midpoint": json_values(midpoints),
            "online_gamma": json_values(gammas),
        },
    }


def terminal_projection_summary(
    evaluator,
    geometry,
    condition_fields: dict[str, np.ndarray],
    counts: np.ndarray,
    interval: float,
    posterior_temperature: float,
) -> dict:
    exact = exact_prefix_trajectory(
        evaluator,
        evaluator.protocol,
        geometry,
        temperature=posterior_temperature,
    )
    output = {}
    raw = {}
    for condition in ("original_v1", "conditioned_gate"):
        potentials = normalize_potentials(
            hodge_potentials(condition_fields[condition], geometry)
        )
        expected = potential_alignment(potentials, exact.expected_rank_potentials[-1])[
            "cosine"
        ]
        map_alignment = potential_alignment(potentials, exact.map_potentials[-1])[
            "cosine"
        ]
        difference = expected - map_alignment
        output[condition] = summarize_subjects(difference, counts, interval=interval)
        raw[condition] = json_values(difference)
    return {"summary": output, "raw_subject_level": raw}


def decision_summary(
    specification: dict,
    original_qualification: dict,
    conditioned_qualification: dict,
    causal: dict,
    local: dict[str, dict],
    binding: dict,
    terminal: dict,
    crossing: dict,
) -> dict:
    conditioned = local["conditioned_gate"]
    original = local["original_v1"]
    global_control = local["matched_global_scalar"]
    shuffled = local["shuffled_gate"]
    key = "retained_relation_mean_direct_correctness"
    other_key = "other_relation_mean_direct_correctness"
    local_difference = summarize_difference(
        conditioned["subject_level"][key],
        original["subject_level"][key],
        local["counts"],
        interval=local["interval"],
    )
    other_difference = summarize_difference(
        conditioned["subject_level"][other_key],
        original["subject_level"][other_key],
        local["counts"],
        interval=local["interval"],
    )
    global_difference = summarize_difference(
        conditioned["subject_level"][key],
        global_control["subject_level"][key],
        local["counts"],
        interval=local["interval"],
    )
    shuffled_difference = summarize_difference(
        conditioned["subject_level"][key],
        shuffled["subject_level"][key],
        local["counts"],
        interval=local["interval"],
    )
    h_summary = conditioned["summary"]["H_greater_A_direct_correctness"]
    local_rescue = (
        local_difference["bootstrap"]["lower"] > 0.0
        and h_summary["bootstrap"]["upper"] >= 0.0
    )
    preserve_other = other_difference["bootstrap"]["lower"] >= -0.01
    conditioned_mean = conditioned["summary"][key]["mean"]
    global_mean = global_control["summary"][key]["mean"]
    shuffled_mean = shuffled["summary"][key]["mean"]
    specificity = (
        conditioned_mean > global_mean
        and conditioned_mean > shuffled_mean
        and (
            global_difference["bootstrap"]["lower"] > 0.0
            or shuffled_difference["bootstrap"]["lower"] > 0.0
        )
    )
    conditioned_intact = causal["conditions"]["intact"]
    original_nonlearned = original_qualification["causal_result"]["conditions"][
        "intact"
    ]["nonlearned_accuracy"]
    nonlearned = (
        conditioned_intact["nonlearned_accuracy"] >= 0.70
        and conditioned_intact["nonlearned_accuracy"] >= original_nonlearned - 0.02
    )
    global_preservation = True
    for metric in ("remote_absolute", "gauge_invariant_R_third_rel"):
        current = conditioned["summary"][metric]
        reference = original["summary"][metric]
        global_preservation &= (
            current["bootstrap"]["lower"] > 0.0
            and current["mean"] >= 0.8 * reference["mean"]
        )
    binding_pass = (
        binding["conditioned_minus_original_max_abs"] <= 1e-7
        and binding["conditioned_gate"]["matched_minus_shared_endpoint"]["mean"] > 0.0
        and binding["conditioned_gate"]["matched_minus_disjoint"]["mean"] > 0.0
    )
    terminal_pass = terminal["summary"]["conditioned_gate"]["bootstrap"]["lower"] > 0.0
    crossing_pass = (
        crossing["spearman_rho"] is not None and crossing["spearman_rho"] > 0.0
    )
    flags = {
        "original_backbone_competence": bool(original_qualification["passed"]),
        "local_rescue": bool(local_rescue),
        "preserve_other_relations": bool(preserve_other),
        "state_conditioning_specificity": bool(specificity),
        "nonlearned_preservation": bool(nonlearned),
        "fast_weight_necessity": bool(conditioned_qualification["passed"]),
        "global_reassembly_preservation": bool(global_preservation),
        "query_binding_preservation": bool(binding_pass),
        "terminal_projection_preservation": bool(terminal_pass),
        "online_mechanism_alignment": bool(crossing_pass),
    }
    all_pass = all(flags.values())
    if not flags["original_backbone_competence"]:
        outcome = "competence_or_integrity_failure"
    elif flags["local_rescue"] and not flags["global_reassembly_preservation"]:
        outcome = "local_rescue_with_global_failure"
    elif flags["local_rescue"] and not flags["state_conditioning_specificity"]:
        outcome = "local_rescue_without_specificity"
    elif all_pass:
        outcome = "all_primary_rules_pass"
    else:
        outcome = "no_local_rescue_with_nonzero_beta"
    return {
        "all_primary_rules_pass": all_pass,
        "outcome": outcome,
        "flags": flags,
        "registered_rules": specification["primary_decision_rules"],
        "paired_contrasts": {
            "conditioned_minus_original_local": local_difference,
            "conditioned_minus_original_other_relations": other_difference,
            "conditioned_minus_matched_global_local": global_difference,
            "conditioned_minus_shuffled_local": shuffled_difference,
        },
    }


def evaluate_pilot(
    specification: dict,
    checkpoint: Path,
    gate_artifact_path: Path,
    source_validation: dict,
    runtime: dict,
) -> dict:
    evaluation = specification["liu_evaluation"]
    artifact = load_json(gate_artifact_path)
    if artifact["backbone"]["sha256"] != checkpoint_sha256(checkpoint):
        raise RuntimeError("gate artifact and backbone checkpoint do not match")
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, int(evaluation["subjects"])
    )
    gate = CurvatureGateTransition(
        backbone,
        epsilon=float(specification["gate_equation"]["epsilon"]),
        initial_beta=float(specification["gate_equation"]["initial_beta"]),
    )
    with torch.no_grad():
        gate.raw_beta.fill_(float(artifact["raw_beta"]))
    gamma_global = float(artifact["calibration"]["gamma_global"])
    recalibration_backbone, recalibration_config, _ = load_retro_checkpoint(
        checkpoint, int(specification["matched_global_calibration"]["batch_size"])
    )
    recalibration_gate = CurvatureGateTransition(
        recalibration_backbone,
        epsilon=float(specification["gate_equation"]["epsilon"]),
        initial_beta=float(specification["gate_equation"]["initial_beta"]),
    )
    with torch.no_grad():
        recalibration_gate.raw_beta.fill_(float(artifact["raw_beta"]))
    recalibration = calibrate_global_gamma(
        specification,
        recalibration_backbone,
        recalibration_config,
        recalibration_gate,
        compile_models=True,
    )
    calibration_error = abs(recalibration["gamma_global"] - gamma_global)
    if calibration_error > 1e-7:
        raise RuntimeError("matched global scalar did not reproduce")

    protocol = load_ranking_protocol(
        resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        )
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
    geometry = build_complete_graph_geometry(protocol)
    pairs = ordered_pairs(protocol.n_items)
    schedules = tuple(pairs for _ in range(model_config.bs))
    intact = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    relations = tuple(protocol.support_pairs_higher_lower)
    loo_rows = []
    for relation in relations:
        state = evaluator.initialize_fast_weights()
        for trial_index in range(protocol.support_trials):
            state = evaluator.advance_support_trial(
                state,
                trial_index,
                zero_relations=frozenset((relation,)),
            )
        loo_rows.append(state)
    loo = torch.stack(loo_rows)
    retained = retained_relation_mask(evaluator, relations)
    counts = (
        np.random.default_rng(int(evaluation["bootstrap_seed"]))
        .multinomial(
            model_config.bs,
            np.full(model_config.bs, 1.0 / model_config.bs),
            size=int(evaluation["bootstrap_samples"]),
        )
        .astype(np.float64)
    )
    interval = float(evaluation["bootstrap_interval"])

    condition_bundles = {}
    condition_fields = {}
    local = {"counts": counts, "interval": interval}
    behavior = {}
    shuffle_errors = []
    for condition in CONDITIONS:
        intact_bundle = query_bundle(
            evaluator,
            gate,
            intact,
            schedules,
            condition=condition,
            gamma_global=gamma_global,
            shuffle_seed=int(evaluation["shuffle_seed"]),
        )
        loo_bundles = [
            query_bundle(
                evaluator,
                gate,
                loo[index],
                schedules,
                condition=condition,
                gamma_global=gamma_global,
                shuffle_seed=int(evaluation["shuffle_seed"]),
            )
            for index in range(len(relations))
        ]
        condition_bundles[condition] = intact_bundle
        fields = {
            "intact": margin_fields(intact_bundle, protocol.n_items),
            "loo": np.asarray(
                [margin_fields(bundle, protocol.n_items) for bundle in loo_bundles]
            ),
        }
        condition_fields[condition] = fields["intact"]
        local[condition] = field_metrics(
            fields, relations, retained, geometry, counts, interval
        )
        behavior[condition] = analyze_sampled_query_policy(
            protocol,
            bundle_logits(intact_bundle, schedules),
            seed=int(evaluation["choice_seed"]),
            temperature=float(evaluation["temperature"]),
        )
        if condition == "shuffled_gate":
            natural = condition_bundles["conditioned_gate"]["conditioned_gammas"]
            shuffled_values = intact_bundle["applied_gammas"]
            for subject in range(model_config.bs):
                shuffle_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.sort(natural[subject])
                                - np.sort(shuffled_values[subject])
                            )
                        )
                    )
                )

    original_readout = evaluator.readout_logits(intact, schedules)
    original_bundle_logits = bundle_logits(condition_bundles["original_v1"], schedules)
    gate_equivalence_error = max(
        abs(original_readout[subject][pair] - original_bundle_logits[subject][pair])
        for subject in range(model_config.bs)
        for pair in pairs
    )
    if gate_equivalence_error > 1e-6:
        raise RuntimeError("gamma=1 failed to reproduce v1 logits")

    original_causal_result = run_causal_suite(
        checkpoint,
        batch_size=model_config.bs,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        order_seed=int(evaluation["order_seed"]),
        order_schedules=8,
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
        protocol_path=resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        ),
    )
    qualification_specification = load_json(
        resolve_registered_path(
            specification["registered_sources"]["qualification"]["path"]
        )
    )
    original_qualification = evaluate_qualification(
        original_causal_result, qualification_specification
    )
    original_qualification["causal_result"] = original_causal_result
    conditioned_causal = conditioned_causal_suite(
        checkpoint, evaluator, gate, gamma_global, specification
    )
    conditioned_qualification = evaluate_qualification(
        conditioned_causal, qualification_specification
    )
    binding = query_binding_summary(evaluator, intact, loo, retained, counts, interval)
    terminal = terminal_projection_summary(
        evaluator,
        geometry,
        condition_fields,
        counts,
        interval,
        float(evaluation["posterior_temperature"]),
    )
    crossing = crossing_alignment(
        evaluator,
        intact,
        loo,
        retained,
        condition_bundles["conditioned_gate"],
        geometry,
    )
    decision = decision_summary(
        specification,
        original_qualification,
        conditioned_qualification,
        conditioned_causal,
        local,
        binding,
        terminal,
        crossing,
    )
    local_output = {
        name: {
            "summary": row["summary"],
            "raw_subject_level": {
                key: json_values(value) for key, value in row["subject_level"].items()
            },
            "raw_relation_subject": row["raw_relation_subject"],
        }
        for name, row in local.items()
        if name in CONDITIONS
    }
    return {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "seed": int(specification["development_seed_contract"]["mandatory_seed"]),
        "checkpoint": asdict(checkpoint_info),
        "gate_artifact": {
            "path": str(gate_artifact_path.resolve()),
            "sha256": file_sha256(gate_artifact_path),
            "beta": artifact["beta"],
            "gamma_global": gamma_global,
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "integrity": {
            "calibration_reproduction_error": calibration_error,
            "gate_off_v1_logit_max_abs_error": gate_equivalence_error,
            "shuffle_multiset_max_abs_error": max(shuffle_errors, default=0.0),
            "stable_omitted_max_abs_pair_influence": max(
                local[condition]["summary"]["stable_omitted_max_abs_pair_influence"]
                for condition in CONDITIONS
            ),
        },
        "local_fidelity": local_output,
        "behavior": behavior,
        "original_v1_qualification": original_qualification,
        "conditioned_qualification": {
            **conditioned_qualification,
            "causal_result": conditioned_causal,
        },
        "query_binding": binding,
        "terminal_projection": terminal,
        "crossing_alignment": crossing,
        "decision": decision,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered curvature-gate v2 development pilot."
    )
    parser.add_argument(
        "stage", choices=("train-backbone", "adapt-gate", "evaluate", "all")
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    runtime = configure_runtime()
    source_validation = validate_sources(parsed.specification, parsed.lock)
    specification = load_json(parsed.specification)
    seed = int(specification["development_seed_contract"]["mandatory_seed"])
    checkpoint = parsed.output_root / f"seed-{seed}" / "backbone" / "net.dat"
    artifact = parsed.output_root / f"seed-{seed}" / "gate" / "gate.json"
    if parsed.stage in ("train-backbone", "all"):
        checkpoint = train_backbone(specification, parsed.output_root, runtime)
    if parsed.stage in ("adapt-gate", "all"):
        if not checkpoint.is_file():
            raise FileNotFoundError("registered backbone checkpoint is missing")
        artifact = adapt_gate(
            specification,
            checkpoint,
            parsed.output_root,
            source_validation,
            runtime,
        )
    if parsed.stage in ("evaluate", "all"):
        if not checkpoint.is_file() or not artifact.is_file():
            raise FileNotFoundError("backbone or gate artifact is missing")
        result = evaluate_pilot(
            specification,
            checkpoint,
            artifact,
            source_validation,
            runtime,
        )
        write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
