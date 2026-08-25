"""Registered frozen-backbone policy-opposition gate v2.1 pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .assembly_trajectory import build_complete_graph_geometry, summarize_difference
from .behavioral import analyze_sampled_query_policy
from .confirmation import file_sha256
from .curvature_gate import make_gate_tasks, run_gate_batch
from .curvature_gate_pilot import (
    _adaptation_config,
    _field_metrics,
    _json_values,
    _ordered_pairs,
    _resolve_registered,
    _retained_mask,
    _tensor_hashes,
    bundle_logits,
    calibrate_global_gamma,
    conditioned_causal_suite,
    configure_runtime,
    crossing_alignment,
    load_json,
    margin_fields,
    query_binding_summary,
    terminal_projection_summary,
    write_json,
)
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
    run_causal_suite,
)
from .policy_opposition_gate import PolicyOppositionGateTransition
from .qualification import evaluate_qualification
from .ranking_protocol import load_ranking_protocol
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/policy_opposition_gate_pilot_v2_1.json")
)
DEFAULT_LOCK_PATH = (
    resolve_record("benchmarks/policy_opposition_gate_pilot_v2_1.lock_v3.json")
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "runs" / "policy-opposition-gate-pilot-v2-1"
)
DEFAULT_RESULT_PATH = resolve_record("results/policy_opposition_gate_pilot_v2_1.json")
UNSIGNED_SPECIFICATION_PATH = resolve_record("benchmarks/curvature_gate_pilot_v2.json")
CONDITIONS = (
    "original_v1",
    "opposition_gate",
    "matched_global_scalar",
    "shuffled_opposition_gate",
    "sign_reversed_support_gate",
)


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict:
    specification_path = specification_path.resolve()
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
    if "frozen_gate_artifact" in lock:
        registration = lock["frozen_gate_artifact"]
        path = _resolve_registered(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": "frozen_gate_artifact",
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": registration["sha256"],
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"policy-opposition source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _runtime_specification(specification: dict) -> dict:
    """Supply unchanged v2 generic-task metadata to the v2.1 runner."""

    unsigned = load_json(UNSIGNED_SPECIFICATION_PATH)
    return {
        **unsigned,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "registered_sources": specification["registered_sources"],
        "development_seed_contract": {"mandatory_seed": 2101},
        "gate_equation": specification["gate_equation"],
        "gate_only_adaptation": specification["gate_only_adaptation"],
        "matched_global_calibration": specification["matched_global_calibration"],
        "liu_evaluation": specification["liu_evaluation"],
        "primary_decision_rules": specification["primary_decision_rules"],
    }


def _new_gate(backbone, specification: dict) -> PolicyOppositionGateTransition:
    equation = specification["gate_equation"]
    return PolicyOppositionGateTransition(
        backbone,
        tau=float(equation["tau"]),
        epsilon=float(equation["epsilon"]),
        initial_beta=float(equation["initial_beta"]),
    )


def adapt_gate(
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
    gate = _new_gate(backbone, specification)
    task_generator = make_gate_tasks(adaptation)
    rng = np.random.default_rng(adaptation.seed)
    np.random.seed(adaptation.seed)
    torch.manual_seed(adaptation.seed)
    training_backbone = torch.compile(backbone, fullgraph=True, mode="default")
    training_gate = torch.compile(gate, fullgraph=True, mode="default")
    optimizer = torch.optim.Adam(
        [gate.raw_beta],
        lr=float(specification["gate_only_adaptation"]["learning_rate"]),
    )
    gate_dir = output_root / "seed-2101" / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    log_path = gate_dir / "train_log.jsonl"
    if log_path.exists():
        raise RuntimeError("v2.1 gate adaptation log exists; refusing to append")
    with log_path.open("w", encoding="utf-8") as handle:
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
                "mean_opposition_risk": stats_row.risk_sum / stats_row.gamma_count,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    after = _tensor_hashes(backbone)
    if before != after:
        raise RuntimeError("frozen backbone changed during v2.1 gate adaptation")
    calibration = calibrate_global_gamma(
        runtime_specification,
        backbone,
        model_config,
        gate,
        compile_models=True,
    )
    artifact = {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "seed": 2101,
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


def _query_pass(
    evaluator,
    gate,
    fast_weights,
    pair_schedules,
    *,
    gamma_overrides: np.ndarray | None,
    alpha_zero: bool,
    use_support_risk: bool,
) -> dict:
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    names = (
        "logits",
        "risks",
        "conditioned_gammas",
        "applied_gammas",
        "opposition_risks",
        "support_risks",
        "first_order_values",
        "quadratic_values",
        "scale_squared",
        "denominators",
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
            override = None
            if gamma_overrides is not None:
                override = torch.from_numpy(
                    gamma_overrides[:, pair_index].astype(np.float32)
                ).to(fast_weights.device)[:, None]
            statistics = gate.statistics(response, hidden, fast_weights)
            output = gate(
                response,
                hidden,
                eligibility,
                fast_weights,
                override,
                use_support_risk,
            )
            arrays["logits"][:, pair_index] = (
                (output[0][:, 1] - output[0][:, 0]).detach().cpu().numpy()
            )
            arrays["risks"][:, pair_index] = output[6][:, 0].cpu().numpy()
            arrays["conditioned_gammas"][:, pair_index] = output[7][:, 0].cpu().numpy()
            arrays["applied_gammas"][:, pair_index] = output[8][:, 0].cpu().numpy()
            for name, value in zip(
                (
                    "opposition_risks",
                    "support_risks",
                    "first_order_values",
                    "quadratic_values",
                    "scale_squared",
                    "denominators",
                ),
                statistics,
            ):
                arrays[name][:, pair_index] = value[:, 0].cpu().numpy()
    return arrays


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
        raise ValueError(f"unknown policy-opposition condition: {condition}")
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    if condition in ("opposition_gate", "sign_reversed_support_gate"):
        return _query_pass(
            evaluator,
            gate,
            fast_weights,
            pair_schedules,
            gamma_overrides=None,
            alpha_zero=alpha_zero,
            use_support_risk=condition == "sign_reversed_support_gate",
        )
    if condition == "original_v1":
        overrides = np.ones((subjects, pair_count), dtype=np.float64)
    elif condition == "matched_global_scalar":
        overrides = np.full((subjects, pair_count), gamma_global, dtype=np.float64)
    else:
        natural = _query_pass(
            evaluator,
            gate,
            fast_weights,
            pair_schedules,
            gamma_overrides=None,
            alpha_zero=alpha_zero,
            use_support_risk=False,
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
    return _query_pass(
        evaluator,
        gate,
        fast_weights,
        pair_schedules,
        gamma_overrides=overrides,
        alpha_zero=alpha_zero,
        use_support_risk=False,
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
    opposition_qualification: dict,
    causal: dict,
    local: dict,
    binding: dict,
    terminal: dict,
) -> dict:
    opposition = local["opposition_gate"]
    original = local["original_v1"]
    key = "retained_relation_mean_direct_correctness"
    other_key = "other_relation_mean_direct_correctness"
    h_key = "H_greater_A_direct_correctness"
    contrasts = {
        "opposition_minus_original_local": _paired(
            opposition["subject_level"][key],
            original["subject_level"][key],
            local["counts"],
            local["interval"],
        ),
        "opposition_minus_original_H_greater_A": _paired(
            opposition["subject_level"][h_key],
            original["subject_level"][h_key],
            local["counts"],
            local["interval"],
        ),
        "opposition_minus_original_other_relations": _paired(
            opposition["subject_level"][other_key],
            original["subject_level"][other_key],
            local["counts"],
            local["interval"],
        ),
    }
    for control in (
        "matched_global_scalar",
        "shuffled_opposition_gate",
        "sign_reversed_support_gate",
    ):
        contrasts[f"opposition_minus_{control}_local"] = _paired(
            opposition["subject_level"][key],
            local[control]["subject_level"][key],
            local["counts"],
            local["interval"],
        )
    local_rescue = (
        contrasts["opposition_minus_original_local"]["bootstrap"]["lower"] > 0.0
    )
    control_specificity = all(
        contrasts[f"opposition_minus_{control}_local"]["bootstrap"]["lower"] > 0.0
        for control in (
            "matched_global_scalar",
            "shuffled_opposition_gate",
            "sign_reversed_support_gate",
        )
    )
    h_rescue = (
        contrasts["opposition_minus_original_H_greater_A"]["bootstrap"]["lower"] > 0.0
        and opposition["summary"][h_key]["bootstrap"]["upper"] >= 0.0
    )
    preserve_other = (
        contrasts["opposition_minus_original_other_relations"]["bootstrap"]["lower"]
        >= -0.01
    )
    opposition_intact = causal["conditions"]["intact"]
    original_nonlearned = original_qualification["causal_result"]["conditions"][
        "intact"
    ]["nonlearned_accuracy"]
    nonlearned = (
        opposition_intact["nonlearned_accuracy"] >= 0.70
        and opposition_intact["nonlearned_accuracy"] >= original_nonlearned - 0.02
    )
    global_preservation = True
    for metric in ("remote_absolute", "gauge_invariant_R_third_rel"):
        current = opposition["summary"][metric]
        reference = original["summary"][metric]
        global_preservation &= (
            current["bootstrap"]["lower"] > 0.0
            and current["mean"] >= 0.8 * reference["mean"]
        )
    binding_pass = (
        binding["opposition_minus_original_max_abs"] <= 1e-7
        and binding["opposition_gate"]["matched_minus_shared_endpoint"]["mean"] > 0.0
        and binding["opposition_gate"]["matched_minus_disjoint"]["mean"] > 0.0
    )
    terminal_pass = terminal["summary"]["opposition_gate"]["bootstrap"]["lower"] > 0.0
    flags = {
        "original_backbone_competence": bool(original_qualification["passed"]),
        "local_rescue": bool(local_rescue),
        "control_specificity": bool(control_specificity),
        "H_greater_A_rescue": bool(h_rescue),
        "preserve_other_relations": bool(preserve_other),
        "nonlearned_preservation": bool(nonlearned),
        "fast_weight_necessity": bool(opposition_qualification["passed"]),
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


def _allocation_summary(evaluator, bundle: dict, retained: np.ndarray) -> dict:
    pairs = _ordered_pairs(evaluator.protocol.n_items)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    labels = evaluator.protocol.item_labels
    relation_rows = []
    for relation_index, relation in enumerate(
        evaluator.protocol.support_pairs_higher_lower
    ):
        indices = (pair_index[relation], pair_index[(relation[1], relation[0])])
        mask = retained[relation_index]
        row = {
            "relation": f"{labels[relation[0]]}>{labels[relation[1]]}",
            "retained_subjects": int(np.sum(mask)),
        }
        for name in (
            "first_order_values",
            "quadratic_values",
            "opposition_risks",
            "support_risks",
            "conditioned_gammas",
        ):
            values = np.mean(bundle[name][:, indices], axis=1)
            row[f"mean_{name}"] = float(np.mean(values[mask]))
        product = (
            bundle["first_order_values"][:, indices]
            * bundle["quadratic_values"][:, indices]
        )
        row["opposed_fraction"] = float(np.mean(product[mask, :] < 0.0))
        relation_rows.append(row)
    product = bundle["first_order_values"] * bundle["quadratic_values"]
    return {
        "all_query_state_sign_fractions": {
            "opposed_jk_below_zero": float(np.mean(product < 0.0)),
            "supportive_jk_above_zero": float(np.mean(product > 0.0)),
            "exact_zero": float(np.mean(product == 0.0)),
        },
        "by_retained_relation": relation_rows,
        "raw_intact_query": {
            name: _json_values(bundle[name])
            for name in (
                "first_order_values",
                "quadratic_values",
                "opposition_risks",
                "support_risks",
                "conditioned_gammas",
            )
        },
    }


def evaluate_pilot(
    specification: dict,
    checkpoint: Path,
    gate_artifact_path: Path,
    source_validation: dict,
    runtime: dict,
) -> dict:
    runtime_specification = _runtime_specification(specification)
    evaluation = specification["liu_evaluation"]
    artifact = load_json(gate_artifact_path)
    if artifact["pilot_id"] != specification["pilot_id"]:
        raise RuntimeError("gate artifact belongs to a different pilot")
    if artifact["backbone"]["sha256"] != checkpoint_sha256(checkpoint):
        raise RuntimeError("gate artifact and frozen backbone do not match")
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, int(evaluation["subjects"])
    )
    gate = _new_gate(backbone, specification)
    with torch.no_grad():
        gate.raw_beta.fill_(float(artifact["raw_beta"]))
    gamma_global = float(artifact["calibration"]["gamma_global"])

    calibration_size = int(specification["matched_global_calibration"]["batch_size"])
    calibration_backbone, calibration_config, _ = load_retro_checkpoint(
        checkpoint, calibration_size
    )
    calibration_gate = _new_gate(calibration_backbone, specification)
    with torch.no_grad():
        calibration_gate.raw_beta.fill_(float(artifact["raw_beta"]))
    recalibration = calibrate_global_gamma(
        runtime_specification,
        calibration_backbone,
        calibration_config,
        calibration_gate,
        compile_models=True,
    )
    calibration_error = abs(recalibration["gamma_global"] - gamma_global)
    if calibration_error > 1e-7:
        raise RuntimeError("matched global scalar did not reproduce")

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
    all_bundles = []
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
        all_bundles.extend((intact_bundle, *loo_bundles))
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
        if condition == "shuffled_opposition_gate":
            natural = condition_bundles["opposition_gate"]["conditioned_gammas"]
            shuffled = intact_bundle["applied_gammas"]
            for subject in range(model_config.bs):
                shuffle_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.sort(natural[subject]) - np.sort(shuffled[subject])
                            )
                        )
                    )
                )

    original_readout = evaluator.readout_logits(intact, schedules)
    original_logits = bundle_logits(condition_bundles["original_v1"], schedules)
    equivalence_error = max(
        abs(original_readout[subject][pair] - original_logits[subject][pair])
        for subject in range(model_config.bs)
        for pair in pairs
    )
    if equivalence_error > 1e-6:
        raise RuntimeError("gamma=1 failed to reproduce frozen v1 logits")
    opposition_zero_errors = []
    support_zero_errors = []
    minimum_risk = float("inf")
    for bundle in all_bundles:
        product = bundle["first_order_values"] * bundle["quadratic_values"]
        if np.any(product > 0.0):
            opposition_zero_errors.append(
                float(np.max(np.abs(bundle["opposition_risks"][product > 0.0])))
            )
        if np.any(product < 0.0):
            support_zero_errors.append(
                float(np.max(np.abs(bundle["support_risks"][product < 0.0])))
            )
        minimum_risk = min(
            minimum_risk,
            float(np.min(bundle["opposition_risks"])),
            float(np.min(bundle["support_risks"])),
        )
    risk_identity_error = max(
        max(opposition_zero_errors, default=0.0),
        max(support_zero_errors, default=0.0),
        max(0.0, -minimum_risk),
    )
    if risk_identity_error > 1e-7:
        raise RuntimeError("signed risk identity failed")

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
    opposition_causal = conditioned_causal_suite(
        checkpoint, evaluator, gate, gamma_global, runtime_specification
    )
    opposition_qualification = evaluate_qualification(
        opposition_causal, qualification_specification
    )
    binding = query_binding_summary(evaluator, intact, loo, retained, counts, interval)
    binding["opposition_gate"] = binding.pop("conditioned_gate")
    binding["opposition_minus_original_max_abs"] = binding.pop(
        "conditioned_minus_original_max_abs"
    )
    terminal = terminal_projection_summary(
        evaluator,
        geometry,
        {
            "original_v1": condition_fields["original_v1"],
            "conditioned_gate": condition_fields["opposition_gate"],
        },
        counts,
        interval,
        float(evaluation["posterior_temperature"]),
    )
    terminal["summary"]["opposition_gate"] = terminal["summary"].pop("conditioned_gate")
    terminal["raw_subject_level"]["opposition_gate"] = terminal[
        "raw_subject_level"
    ].pop("conditioned_gate")
    crossing = crossing_alignment(
        evaluator,
        intact,
        loo,
        retained,
        condition_bundles["opposition_gate"],
        geometry,
    )
    decision = decision_summary(
        specification,
        original_qualification,
        opposition_qualification,
        opposition_causal,
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
        "gate_artifact": {
            "path": str(gate_artifact_path.resolve()),
            "sha256": file_sha256(gate_artifact_path),
            "beta": artifact["beta"],
            "gamma_global": gamma_global,
            "same_beta_for_sign_reversed": True,
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "integrity": {
            "calibration_reproduction_error": calibration_error,
            "gate_off_v1_logit_max_abs_error": equivalence_error,
            "signed_risk_identity_max_abs_error": risk_identity_error,
            "shuffle_multiset_max_abs_error": max(shuffle_errors, default=0.0),
            "stable_omitted_max_abs_pair_influence": max(
                local[condition]["summary"]["stable_omitted_max_abs_pair_influence"]
                for condition in CONDITIONS
            ),
        },
        "gate_allocation": _allocation_summary(
            evaluator, condition_bundles["opposition_gate"], retained
        ),
        "local_fidelity": local_output,
        "behavior": behavior,
        "original_v1_qualification": original_qualification,
        "opposition_qualification": {
            **opposition_qualification,
            "causal_result": opposition_causal,
        },
        "query_binding": binding,
        "terminal_projection": terminal,
        "crossing_alignment_diagnostic_only": crossing,
        "decision": decision,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen-backbone policy-opposition v2.1 pilot."
    )
    parser.add_argument("stage", choices=("adapt-gate", "evaluate", "all"))
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
    checkpoint = _resolve_registered(specification["frozen_backbone_contract"]["path"])
    artifact = parsed.output_root / "seed-2101" / "gate" / "gate.json"
    if parsed.stage in ("adapt-gate", "all"):
        artifact = adapt_gate(
            specification,
            checkpoint,
            parsed.output_root,
            source_validation,
            runtime,
        )
    if parsed.stage in ("evaluate", "all"):
        if not artifact.is_file():
            raise FileNotFoundError("frozen v2.1 gate artifact is missing")
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
