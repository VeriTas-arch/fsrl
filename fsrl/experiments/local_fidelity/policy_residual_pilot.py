"""Registered frozen-backbone first-order policy-residual v2.2 pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
    run_causal_suite,
)
from fsrl.evaluation.qualification import evaluate_qualification
from fsrl.experiments.assembly.trajectory import (
    build_complete_graph_geometry,
    summarize_difference,
)
from fsrl.experiments.confirmation.behavioral import file_sha256
from fsrl.experiments.local_fidelity.curvature_gate import (
    make_gate_tasks,
    run_gate_batch,
)
from fsrl.experiments.local_fidelity.curvature_gate_pilot import (
    _adaptation_config,
    _field_metrics,
    _json_values,
    _ordered_pairs,
    _resolve_registered,
    _retained_mask,
    _tensor_hashes,
    bundle_logits,
    conditioned_causal_suite,
    configure_runtime,
    load_json,
    margin_fields,
    query_binding_summary,
    terminal_projection_summary,
    write_json,
)
from fsrl.experiments.local_fidelity.policy_residual import PolicyResidualTransition
from fsrl.infrastructure.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/policy_residual_pilot_v2_2.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/policy_residual_pilot_v2_2.lock.json"
)
DEFAULT_ARTIFACT_LOCK_PATH = resolve_record(
    "benchmarks/policy_residual_pilot_v2_2.artifact_lock.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "policy-residual-pilot-v2-2"
DEFAULT_RESULT_PATH = resolve_record("results/policy_residual_pilot_v2_2.json")
UNSIGNED_SPECIFICATION_PATH = resolve_record("benchmarks/curvature_gate_pilot_v2.json")
CONDITIONS = (
    "original_v1",
    "policy_residual",
    "matched_magnitude_null",
    "shuffled_residual",
)


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification_path = specification_path.resolve()
    implementation_lock_path = implementation_lock_path.resolve()
    specification = load_json(specification_path)
    checks = []
    for name, registration in specification["registered_sources"].items():
        path = _resolve_registered(registration["path"])
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
    backbone = specification["frozen_backbone_contract"]
    backbone_path = _resolve_registered(backbone["path"])
    observed_backbone = checkpoint_sha256(backbone_path)
    checks.append(
        {
            "name": "frozen_seed_2101_backbone",
            "path": str(backbone_path.relative_to(ROOT)),
            "observed": observed_backbone,
            "expected": backbone["sha256"],
            "passed": observed_backbone == backbone["sha256"],
        }
    )
    lock = load_json(implementation_lock_path)
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
        path = _resolve_registered(registration["path"])
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
        raise RuntimeError(f"policy-residual source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def validate_artifact(
    specification_path: Path,
    implementation_lock_path: Path,
    artifact_lock_path: Path,
    artifact_path: Path,
) -> dict:
    artifact_lock = load_json(artifact_lock_path)
    checks = []
    registrations = {
        "pilot_specification": (
            specification_path.resolve(),
            artifact_lock["pilot_specification_sha256"],
        ),
        "implementation_lock": (
            implementation_lock_path.resolve(),
            artifact_lock["implementation_lock_sha256"],
        ),
        "frozen_backbone": (
            _resolve_registered(artifact_lock["frozen_backbone"]["path"]),
            artifact_lock["frozen_backbone"]["sha256"],
        ),
        "eta_artifact": (
            artifact_path.resolve(),
            artifact_lock["eta_artifact"]["sha256"],
        ),
    }
    for name, (path, expected) in registrations.items():
        observed = file_sha256(path)
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )
    declared_artifact = _resolve_registered(artifact_lock["eta_artifact"]["path"])
    if declared_artifact.resolve() != artifact_path.resolve():
        raise RuntimeError("artifact lock points to a different eta artifact")
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"policy-residual artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": artifact_lock}


def _runtime_specification(specification: dict) -> dict:
    """Supply unchanged v2 generic-task metadata to the v2.2 runner."""

    unsigned = load_json(UNSIGNED_SPECIFICATION_PATH)
    return {
        **unsigned,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "registered_sources": specification["registered_sources"],
        "development_seed_contract": {"mandatory_seed": 2101},
        "gate_only_adaptation": specification["eta_only_adaptation"],
        "liu_evaluation": specification["liu_evaluation"],
        "primary_decision_rules": specification["primary_decision_rules"],
    }


def _new_residual(backbone, specification: dict) -> PolicyResidualTransition:
    return PolicyResidualTransition(
        backbone,
        initial_eta=float(specification["policy_residual_equation"]["initial_eta"]),
    )


def adapt_eta(
    specification: dict,
    checkpoint: Path,
    output_root: Path,
    source_validation: dict,
    runtime: dict,
) -> Path:
    runtime_specification = _runtime_specification(specification)
    adaptation = _adaptation_config(runtime_specification)
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, adaptation.batch_size
    )
    before = _tensor_hashes(backbone)
    residual = _new_residual(backbone, specification)
    task_generator = make_gate_tasks(adaptation)
    rng = np.random.default_rng(adaptation.seed)
    np.random.seed(adaptation.seed)
    torch.manual_seed(adaptation.seed)
    training_backbone = torch.compile(backbone, fullgraph=True, mode="default")
    training_residual = torch.compile(residual, fullgraph=True, mode="default")
    optimizer = torch.optim.Adam(
        [residual.raw_eta],
        lr=float(specification["eta_only_adaptation"]["learning_rate"]),
    )
    artifact_dir = output_root / "seed-2101" / "residual"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "train_log.jsonl"
    if log_path.exists():
        raise RuntimeError("v2.2 eta adaptation log exists; refusing to append")
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(adaptation.outer_steps):
            optimizer.zero_grad()
            stats_row = run_gate_batch(
                adaptation,
                model_config,
                training_backbone,
                training_residual,
                task_generator,
                rng,
            )
            stats_row.loss.backward()
            torch.nn.utils.clip_grad_norm_([residual.raw_eta], adaptation.gradient_clip)
            optimizer.step()
            record = {
                "outer_step": step,
                "query_cross_entropy": stats_row.query_cross_entropy,
                "query_accuracy": stats_row.query_accuracy,
                "eta": float(residual.eta.detach()),
                "mean_eta": stats_row.gamma_sum / stats_row.gamma_count,
                "mean_signed_policy_residual": (
                    stats_row.risk_sum / stats_row.gamma_count
                ),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    after = _tensor_hashes(backbone)
    if before != after:
        raise RuntimeError("frozen backbone changed during v2.2 eta adaptation")
    artifact = {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "seed": 2101,
        "backbone": asdict(checkpoint_info),
        "adaptation": asdict(adaptation),
        "raw_eta": float(residual.raw_eta.detach()),
        "eta": float(residual.eta.detach()),
        "backbone_tensor_hashes_before": before,
        "backbone_tensor_hashes_after": after,
        "source_validation": source_validation,
        "runtime": runtime,
    }
    artifact_path = artifact_dir / "eta.json"
    write_json(artifact_path, artifact)
    return artifact_path


def balanced_magnitude_signs(subjects: int, pair_count: int, seed: int) -> np.ndarray:
    if pair_count % 2:
        raise ValueError("balanced magnitude null requires an even query count")
    base = np.concatenate(
        (
            np.ones(pair_count // 2, dtype=np.float64),
            -np.ones(pair_count // 2, dtype=np.float64),
        )
    )
    return np.stack(
        [
            np.random.default_rng(seed + subject).permutation(base)
            for subject in range(subjects)
        ]
    )


def _query_pass(
    evaluator,
    residual,
    fast_weights,
    pair_schedules,
    *,
    eta_overrides: np.ndarray | None,
    residual_overrides: np.ndarray | None,
    magnitude_signs: np.ndarray | None,
    alpha_zero: bool,
) -> dict:
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    names = (
        "logits",
        "policy_residuals",
        "exact_policy_increments",
        "linear_policy_increments",
        "hidden_residual_norms",
        "etas",
        "applied_residual_bases",
        "applied_corrections",
    )
    arrays = {
        name: np.empty((subjects, pair_count), dtype=np.float64) for name in names
    }
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
            statistics = residual.statistics(response, hidden, fast_weights)
            natural_basis = statistics[0]
            eta_override = None
            if eta_overrides is not None:
                eta_override = torch.from_numpy(
                    eta_overrides[:, pair_index].astype(np.float32)
                ).to(fast_weights.device)[:, None]
            basis_override = None
            if residual_overrides is not None:
                basis_override = torch.from_numpy(
                    residual_overrides[:, pair_index].astype(np.float32)
                ).to(fast_weights.device)[:, None]
            elif magnitude_signs is not None:
                signs = torch.from_numpy(
                    magnitude_signs[:, pair_index].astype(np.float32)
                ).to(fast_weights.device)[:, None]
                basis_override = signs * torch.abs(natural_basis)
            output = residual(
                response,
                hidden,
                eligibility,
                fast_weights,
                eta_override,
                basis_override,
            )
            arrays["logits"][:, pair_index] = (
                (output[0][:, 1] - output[0][:, 0]).cpu().numpy()
            )
            arrays["policy_residuals"][:, pair_index] = (
                natural_basis[:, 0].cpu().numpy()
            )
            arrays["exact_policy_increments"][:, pair_index] = (
                statistics[1][:, 0].cpu().numpy()
            )
            arrays["linear_policy_increments"][:, pair_index] = (
                statistics[2][:, 0].cpu().numpy()
            )
            arrays["hidden_residual_norms"][:, pair_index] = (
                statistics[3][:, 0].cpu().numpy()
            )
            arrays["etas"][:, pair_index] = output[7][:, 0].cpu().numpy()
            applied_basis = natural_basis if basis_override is None else basis_override
            arrays["applied_residual_bases"][:, pair_index] = (
                applied_basis[:, 0].cpu().numpy()
            )
            arrays["applied_corrections"][:, pair_index] = output[8][:, 0].cpu().numpy()
    return arrays


def query_bundle(
    evaluator,
    residual,
    fast_weights,
    pair_schedules,
    *,
    condition: str,
    shuffle_seed: int,
    magnitude_null_seed: int,
    alpha_zero: bool = False,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown policy-residual condition: {condition}")
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    if condition == "policy_residual":
        return _query_pass(
            evaluator,
            residual,
            fast_weights,
            pair_schedules,
            eta_overrides=None,
            residual_overrides=None,
            magnitude_signs=None,
            alpha_zero=alpha_zero,
        )
    if condition == "original_v1":
        eta_overrides = np.zeros((subjects, pair_count), dtype=np.float64)
        return _query_pass(
            evaluator,
            residual,
            fast_weights,
            pair_schedules,
            eta_overrides=eta_overrides,
            residual_overrides=None,
            magnitude_signs=None,
            alpha_zero=alpha_zero,
        )
    if condition == "matched_magnitude_null":
        signs = balanced_magnitude_signs(subjects, pair_count, magnitude_null_seed)
        return _query_pass(
            evaluator,
            residual,
            fast_weights,
            pair_schedules,
            eta_overrides=None,
            residual_overrides=None,
            magnitude_signs=signs,
            alpha_zero=alpha_zero,
        )
    natural = _query_pass(
        evaluator,
        residual,
        fast_weights,
        pair_schedules,
        eta_overrides=None,
        residual_overrides=None,
        magnitude_signs=None,
        alpha_zero=alpha_zero,
    )
    permutations = np.stack(
        [
            np.random.default_rng(shuffle_seed + subject).permutation(pair_count)
            for subject in range(subjects)
        ]
    )
    shuffled = np.take_along_axis(natural["policy_residuals"], permutations, axis=1)
    return _query_pass(
        evaluator,
        residual,
        fast_weights,
        pair_schedules,
        eta_overrides=None,
        residual_overrides=shuffled,
        magnitude_signs=None,
        alpha_zero=alpha_zero,
    )


def _paired(
    first: np.ndarray,
    second: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    return summarize_difference(first, second, counts, interval=interval)


def decision_summary(
    specification: dict,
    original_qualification: dict,
    residual_qualification: dict,
    causal: dict,
    local: dict,
    binding: dict,
    terminal: dict,
) -> dict:
    candidate = local["policy_residual"]
    original = local["original_v1"]
    key = "retained_relation_mean_direct_correctness"
    other_key = "other_relation_mean_direct_correctness"
    h_key = "H_greater_A_direct_correctness"
    contrasts = {
        "policy_residual_minus_original_local": _paired(
            candidate["subject_level"][key],
            original["subject_level"][key],
            local["counts"],
            local["interval"],
        ),
        "policy_residual_minus_original_H_greater_A": _paired(
            candidate["subject_level"][h_key],
            original["subject_level"][h_key],
            local["counts"],
            local["interval"],
        ),
        "policy_residual_minus_original_other_relations": _paired(
            candidate["subject_level"][other_key],
            original["subject_level"][other_key],
            local["counts"],
            local["interval"],
        ),
    }
    for control in ("matched_magnitude_null", "shuffled_residual"):
        contrasts[f"policy_residual_minus_{control}_local"] = _paired(
            candidate["subject_level"][key],
            local[control]["subject_level"][key],
            local["counts"],
            local["interval"],
        )
    local_rescue = (
        contrasts["policy_residual_minus_original_local"]["bootstrap"]["lower"] > 0.0
    )
    h_rescue = (
        contrasts["policy_residual_minus_original_H_greater_A"]["bootstrap"]["lower"]
        > 0.0
        and candidate["summary"][h_key]["bootstrap"]["upper"] >= 0.0
    )
    preserve_other = (
        contrasts["policy_residual_minus_original_other_relations"]["bootstrap"][
            "lower"
        ]
        >= -0.01
    )
    control_specificity = all(
        contrasts[f"policy_residual_minus_{control}_local"]["bootstrap"]["lower"] > 0.0
        for control in ("matched_magnitude_null", "shuffled_residual")
    )
    candidate_intact = causal["conditions"]["intact"]
    original_nonlearned = original_qualification["causal_result"]["conditions"][
        "intact"
    ]["nonlearned_accuracy"]
    nonlearned = (
        candidate_intact["nonlearned_accuracy"] >= 0.70
        and candidate_intact["nonlearned_accuracy"] >= original_nonlearned - 0.02
    )
    global_preservation = True
    for metric in ("remote_absolute", "gauge_invariant_R_third_rel"):
        current = candidate["summary"][metric]
        reference = original["summary"][metric]
        global_preservation &= (
            current["bootstrap"]["lower"] > 0.0
            and current["mean"] >= 0.8 * reference["mean"]
        )
    binding_pass = (
        binding["policy_residual_minus_original_max_abs"] <= 1e-7
        and binding["policy_residual"]["matched_minus_shared_endpoint"]["mean"] > 0.0
        and binding["policy_residual"]["matched_minus_disjoint"]["mean"] > 0.0
    )
    terminal_pass = terminal["summary"]["policy_residual"]["bootstrap"]["lower"] > 0.0
    flags = {
        "original_backbone_competence": bool(original_qualification["passed"]),
        "local_rescue": bool(local_rescue),
        "H_greater_A_rescue": bool(h_rescue),
        "preserve_other_relations": bool(preserve_other),
        "control_specificity": bool(control_specificity),
        "nonlearned_preservation": bool(nonlearned),
        "fast_weight_necessity": bool(residual_qualification["passed"]),
        "global_reassembly_preservation": bool(global_preservation),
        "query_binding_preservation": bool(binding_pass),
        "terminal_projection_preservation": bool(terminal_pass),
    }
    all_pass = all(flags.values())
    if not flags["original_backbone_competence"]:
        outcome = "competence_or_integrity_failure"
    elif flags["local_rescue"] and not flags["global_reassembly_preservation"]:
        outcome = "local_rescue_with_global_failure"
    elif all_pass:
        outcome = "all_primary_rules_pass"
    else:
        outcome = "valid_local_or_specificity_failure"
    return {
        "all_primary_rules_pass": all_pass,
        "outcome": outcome,
        "flags": flags,
        "registered_rules": specification["primary_decision_rules"],
        "paired_contrasts": contrasts,
    }


def _distribution(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(np.abs(array))),
        "lower_quartile": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "upper_quartile": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _allocation_summary(evaluator, bundles: dict, retained: np.ndarray) -> dict:
    candidate = bundles["policy_residual"]
    pairs = _ordered_pairs(evaluator.protocol.n_items)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    labels = evaluator.protocol.item_labels
    relation_rows = []
    for relation_index, relation in enumerate(
        evaluator.protocol.support_pairs_higher_lower
    ):
        indices = (pair_index[relation], pair_index[(relation[1], relation[0])])
        mask = retained[relation_index]
        residual_values = candidate["policy_residuals"][:, indices][mask, :]
        relation_rows.append(
            {
                "relation": f"{labels[relation[0]]}>{labels[relation[1]]}",
                "retained_subjects": int(np.sum(mask)),
                "policy_residual": _distribution(residual_values),
                "applied_correction_by_condition": {
                    condition: _distribution(
                        bundles[condition]["applied_corrections"][:, indices][mask, :]
                    )
                    for condition in CONDITIONS
                },
            }
        )
    return {
        "eta": float(candidate["etas"][0, 0]),
        "all_query_policy_residual": _distribution(candidate["policy_residuals"]),
        "all_query_applied_correction_by_condition": {
            condition: _distribution(bundles[condition]["applied_corrections"])
            for condition in CONDITIONS
        },
        "by_retained_relation": relation_rows,
        "raw_intact_query": {
            name: _json_values(candidate[name])
            for name in (
                "policy_residuals",
                "exact_policy_increments",
                "linear_policy_increments",
                "hidden_residual_norms",
            )
        },
        "raw_applied_correction_by_condition": {
            condition: _json_values(bundles[condition]["applied_corrections"])
            for condition in CONDITIONS
        },
    }


def evaluate_pilot(
    specification: dict,
    checkpoint: Path,
    artifact_path: Path,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    runtime_specification = _runtime_specification(specification)
    evaluation = specification["liu_evaluation"]
    artifact = load_json(artifact_path)
    if artifact["pilot_id"] != specification["pilot_id"]:
        raise RuntimeError("eta artifact belongs to a different pilot")
    if artifact["backbone"]["sha256"] != checkpoint_sha256(checkpoint):
        raise RuntimeError("eta artifact and frozen backbone do not match")
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, int(evaluation["subjects"])
    )
    residual = _new_residual(backbone, specification)
    with torch.no_grad():
        residual.raw_eta.fill_(float(artifact["raw_eta"]))

    protocol = load_ranking_protocol(
        _resolve_registered(specification["registered_sources"]["liu_protocol"]["path"])
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
    pairs = _ordered_pairs(protocol.n_items)
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
    retained = _retained_mask(evaluator, relations)
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
    magnitude_errors = []
    shuffle_errors = []
    unchanged_shuffle_rows = 0
    for condition in CONDITIONS:
        intact_bundle = query_bundle(
            evaluator,
            residual,
            intact,
            schedules,
            condition=condition,
            shuffle_seed=int(evaluation["shuffle_seed"]),
            magnitude_null_seed=int(evaluation["magnitude_null_seed"]),
        )
        loo_bundles = [
            query_bundle(
                evaluator,
                residual,
                loo[index],
                schedules,
                condition=condition,
                shuffle_seed=int(evaluation["shuffle_seed"]),
                magnitude_null_seed=int(evaluation["magnitude_null_seed"]),
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
        local[condition] = _field_metrics(
            fields, relations, retained, geometry, counts, interval
        )
        behavior[condition] = analyze_sampled_query_policy(
            protocol,
            bundle_logits(intact_bundle, schedules),
            seed=int(evaluation["choice_seed"]),
            temperature=float(evaluation["temperature"]),
        )
        if condition == "matched_magnitude_null":
            for bundle in (intact_bundle, *loo_bundles):
                magnitude_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.abs(bundle["applied_residual_bases"])
                                - np.abs(bundle["policy_residuals"])
                            )
                        )
                    )
                )
        if condition == "shuffled_residual":
            for bundle in (intact_bundle, *loo_bundles):
                for subject in range(model_config.bs):
                    natural = bundle["policy_residuals"][subject]
                    shuffled = bundle["applied_residual_bases"][subject]
                    shuffle_errors.append(
                        float(np.max(np.abs(np.sort(natural) - np.sort(shuffled))))
                    )
                    if np.ptp(natural) > 0.0 and np.array_equal(natural, shuffled):
                        unchanged_shuffle_rows += 1

    original_readout = evaluator.readout_logits(intact, schedules)
    original_logits = bundle_logits(condition_bundles["original_v1"], schedules)
    equivalence_error = max(
        abs(original_readout[subject][pair] - original_logits[subject][pair])
        for subject in range(model_config.bs)
        for pair in pairs
    )
    if equivalence_error > 1e-6:
        raise RuntimeError("eta=0 failed to reproduce frozen v1 logits")
    candidate = condition_bundles["policy_residual"]
    margin_delta = candidate["logits"] - condition_bundles["original_v1"]["logits"]
    residual_identity_error = float(
        np.max(np.abs(margin_delta - candidate["applied_corrections"]))
    )
    if residual_identity_error > 1e-6:
        raise RuntimeError("policy residual margin identity failed")
    magnitude_error = max(magnitude_errors, default=0.0)
    if magnitude_error > 1e-7:
        raise RuntimeError("matched-magnitude null failed")
    signs = balanced_magnitude_signs(
        model_config.bs, len(pairs), int(evaluation["magnitude_null_seed"])
    )
    sign_balance_error = int(
        max(
            abs(np.sum(row > 0.0) - len(pairs) // 2)
            + abs(np.sum(row < 0.0) - len(pairs) // 2)
            for row in signs
        )
    )
    if sign_balance_error or unchanged_shuffle_rows:
        raise RuntimeError("control assignment integrity failed")
    shuffle_error = max(shuffle_errors, default=0.0)
    if shuffle_error > 1e-7:
        raise RuntimeError("shuffled residual multiset failed")

    original_causal = run_causal_suite(
        checkpoint,
        batch_size=model_config.bs,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        order_seed=int(evaluation["order_seed"]),
        order_schedules=8,
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
        protocol_path=_resolve_registered(
            specification["registered_sources"]["liu_protocol"]["path"]
        ),
    )
    qualification_specification = load_json(
        _resolve_registered(
            specification["registered_sources"]["qualification"]["path"]
        )
    )
    original_qualification = evaluate_qualification(
        original_causal, qualification_specification
    )
    original_qualification["causal_result"] = original_causal
    residual_causal = conditioned_causal_suite(
        checkpoint, evaluator, residual, 0.0, runtime_specification
    )
    residual_qualification = evaluate_qualification(
        residual_causal, qualification_specification
    )
    binding = query_binding_summary(evaluator, intact, loo, retained, counts, interval)
    binding["policy_residual"] = binding.pop("conditioned_gate")
    binding["policy_residual_minus_original_max_abs"] = binding.pop(
        "conditioned_minus_original_max_abs"
    )
    terminal = terminal_projection_summary(
        evaluator,
        geometry,
        {
            "original_v1": condition_fields["original_v1"],
            "conditioned_gate": condition_fields["policy_residual"],
        },
        counts,
        interval,
        float(evaluation["posterior_temperature"]),
    )
    terminal["summary"]["policy_residual"] = terminal["summary"].pop("conditioned_gate")
    terminal["raw_subject_level"]["policy_residual"] = terminal[
        "raw_subject_level"
    ].pop("conditioned_gate")
    decision = decision_summary(
        specification,
        original_qualification,
        residual_qualification,
        residual_causal,
        local,
        binding,
        terminal,
    )
    local_output = {
        name: {
            "summary": row["summary"],
            "raw_subject_level": {
                key: _json_values(value) for key, value in row["subject_level"].items()
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
        "seed": 2101,
        "checkpoint": asdict(checkpoint_info),
        "eta_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": file_sha256(artifact_path),
            "eta": artifact["eta"],
            "same_eta_for_both_controls": True,
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "integrity": {
            "eta_zero_v1_logit_max_abs_error": equivalence_error,
            "policy_residual_margin_identity_max_abs_error": residual_identity_error,
            "matched_magnitude_max_abs_error": magnitude_error,
            "matched_magnitude_sign_balance_error": sign_balance_error,
            "shuffle_multiset_max_abs_error": shuffle_error,
            "unchanged_nonconstant_shuffle_rows": unchanged_shuffle_rows,
            "stable_omitted_max_abs_pair_influence": max(
                local[condition]["summary"]["stable_omitted_max_abs_pair_influence"]
                for condition in CONDITIONS
            ),
        },
        "residual_allocation": _allocation_summary(
            evaluator, condition_bundles, retained
        ),
        "local_fidelity": local_output,
        "behavior": behavior,
        "original_v1_qualification": original_qualification,
        "policy_residual_qualification": {
            **residual_qualification,
            "causal_result": residual_causal,
        },
        "query_binding": binding,
        "terminal_projection": terminal,
        "decision": decision,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen-backbone first-order policy-residual v2.2 pilot."
    )
    parser.add_argument("stage", choices=("adapt-eta", "evaluate"))
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument(
        "--artifact-lock", type=Path, default=DEFAULT_ARTIFACT_LOCK_PATH
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    runtime = configure_runtime()
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    specification = load_json(parsed.specification)
    checkpoint = _resolve_registered(specification["frozen_backbone_contract"]["path"])
    artifact = parsed.output_root / "seed-2101" / "residual" / "eta.json"
    if parsed.stage == "adapt-eta":
        adapt_eta(
            specification,
            checkpoint,
            parsed.output_root,
            source_validation,
            runtime,
        )
        return 0
    if not artifact.is_file():
        raise FileNotFoundError("frozen v2.2 eta artifact is missing")
    artifact_validation = validate_artifact(
        parsed.specification,
        parsed.implementation_lock,
        parsed.artifact_lock,
        artifact,
    )
    result = evaluate_pilot(
        specification,
        checkpoint,
        artifact,
        source_validation,
        artifact_validation,
        runtime,
    )
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
