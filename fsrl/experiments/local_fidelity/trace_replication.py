"""Two-backbone replication of the frozen conjunctive local trace v2.3."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
)
from fsrl.experiments.assembly.trajectory import (
    summarize_difference,
    summarize_subjects,
)
from fsrl.experiments.confirmation.behavioral import file_sha256
from fsrl.experiments.local_fidelity.behavior_attribution import (
    _json_values,
    _pair_correct_probabilities,
    _self_local_margins,
    _self_traces,
    boundary_and_probability_attribution,
    error_mass_attribution,
    learned_cells,
    local_only_attribution,
    self_cross_attribution,
    slope_decomposition,
)
from fsrl.experiments.local_fidelity.curvature_gate import make_gate_tasks
from fsrl.experiments.local_fidelity.curvature_gate_pilot import (
    configure_runtime,
    load_json,
    write_json,
)
from fsrl.experiments.local_fidelity.trace_pilot import (
    CONDITIONS,
    _behavior_subject_values,
    _new_local_trace,
    _ordered_pairs,
    _retained_mask,
    _tensor_hashes,
    build_local_trace,
    evaluate_pilot,
    query_bundle,
    run_local_batch,
)
from fsrl.infra.study_registry import registered_file_sha256, resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol
from fsrl.training.backbone import MetaTrainConfig, train_meta_model

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_replication_v2_3.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_replication_v2_3.lock.json"
)
DEFAULT_ARTIFACT_LOCK_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_replication_v2_3.artifact_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "runs" / "conjunctive-local-trace-replication-v2-3"
)
DEFAULT_RESULT_PATH = resolve_record(
    "results/conjunctive_local_trace_replication_v2_3.json"
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification_path = specification_path.resolve()
    implementation_lock_path = implementation_lock_path.resolve()
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    checks = []
    registrations = {
        **specification["registered_sources"],
        "replication_specification": {
            "path": str(specification_path),
            "sha256": lock["replication_specification_sha256"],
        },
        **lock["implementation_sources"],
        **lock["reused_frozen_sources"],
    }
    for name, registration in registrations.items():
        path = _resolve(registration["path"])
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
        raise RuntimeError(
            f"conjunctive-local-trace replication source lock failed: {checks}"
        )
    return {"passed": True, "checks": checks, "lock": lock}


def _network_seeds(specification: dict) -> tuple[int, ...]:
    seeds = tuple(
        int(seed) for seed in specification["network_seed_contract"]["mandatory_seeds"]
    )
    if seeds != (2102, 2103):
        raise RuntimeError("the frozen replication requires seeds 2102 and 2103")
    return seeds


def _seed_paths(output_root: Path, seed: int) -> dict[str, Path]:
    seed_dir = output_root / f"seed-{seed}"
    return {
        "backbone_dir": seed_dir / "backbone",
        "checkpoint": seed_dir / "backbone" / "net.dat",
        "backbone_config": seed_dir / "backbone" / "config.json",
        "backbone_log": seed_dir / "backbone" / "train_log.jsonl",
        "backbone_manifest": seed_dir / "backbone" / "replication_manifest.json",
        "local_dir": seed_dir / "local",
        "gain": seed_dir / "local" / "gain.json",
        "local_log": seed_dir / "local" / "train_log.jsonl",
    }


def backbone_training_config(specification: dict, seed: int) -> MetaTrainConfig:
    registered = dict(specification["v1_backbone_training"])
    declared = tuple(int(value) for value in registered.pop("seeds"))
    if seed not in declared:
        raise ValueError(f"seed {seed} is not registered")
    for name in ("held_out_graph", "architecture"):
        registered.pop(name)
    return MetaTrainConfig(seed=seed, **registered)


def local_adaptation_config(specification: dict, seed: int) -> MetaTrainConfig:
    backbone = specification["v1_backbone_training"]
    adaptation = specification["local_only_adaptation"]
    rng_seed = int(adaptation["adaptation_rng_seeds"][str(seed)])
    return MetaTrainConfig(
        seed=rng_seed,
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


def seed_specification(specification: dict, seed: int) -> dict:
    pilot_registration = specification["registered_sources"]["v2_3_specification"]
    seed_specification = load_json(_resolve(pilot_registration["path"]))
    seed_specification["pilot_id"] = specification["replication_id"]
    seed_specification["registration_status"] = specification["registration_status"]
    seed_specification["claim_boundary"] = specification["claim_boundary"]
    seed_specification["local_only_adaptation"] = {
        **seed_specification["local_only_adaptation"],
        "adaptation_rng_seed": int(
            specification["local_only_adaptation"]["adaptation_rng_seeds"][str(seed)]
        ),
    }
    seed_specification["liu_evaluation"] = {
        **seed_specification["liu_evaluation"],
        "bootstrap_seed": int(
            specification["liu_evaluation"]["bootstrap_seeds"][str(seed)]
        ),
    }
    return seed_specification


def _validate_complete_backbone(
    specification: dict, output_root: Path, seed: int
) -> Path:
    paths = _seed_paths(output_root, seed)
    required = tuple(
        paths[name]
        for name in (
            "checkpoint",
            "backbone_config",
            "backbone_log",
            "backbone_manifest",
        )
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"seed {seed} backbone artifact set is incomplete")
    training = backbone_training_config(specification, seed)
    metadata = load_json(paths["backbone_config"])
    if metadata["training"] != asdict(training):
        raise RuntimeError(f"seed {seed} backbone training configuration mismatch")
    if int(metadata["completed_outer_steps"]) != training.outer_steps:
        raise RuntimeError(f"seed {seed} backbone is not the final registered step")
    observed = checkpoint_sha256(paths["checkpoint"])
    if metadata["checkpoint"]["sha256"] != observed:
        raise RuntimeError(f"seed {seed} backbone checkpoint hash mismatch")
    if (
        sum(1 for _ in paths["backbone_log"].open(encoding="utf-8"))
        != training.outer_steps
    ):
        raise RuntimeError(f"seed {seed} backbone log length mismatch")
    return paths["checkpoint"]


def train_backbone(
    specification: dict, output_root: Path, seed: int, runtime: dict
) -> Path:
    paths = _seed_paths(output_root, seed)
    if paths["backbone_dir"].exists():
        return _validate_complete_backbone(specification, output_root, seed)
    training = backbone_training_config(specification, seed)
    train_meta_model(training, paths["backbone_dir"], compile_model=True)
    manifest = {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "seed": seed,
        "runtime": runtime,
        "training": asdict(training),
        "checkpoint": {
            "path": str(paths["checkpoint"].resolve()),
            "sha256": checkpoint_sha256(paths["checkpoint"]),
        },
    }
    write_json(paths["backbone_manifest"], manifest)
    return _validate_complete_backbone(specification, output_root, seed)


def _validate_complete_gain(specification: dict, output_root: Path, seed: int) -> Path:
    paths = _seed_paths(output_root, seed)
    if not paths["gain"].is_file() or not paths["local_log"].is_file():
        raise RuntimeError(f"seed {seed} local-gain artifact set is incomplete")
    artifact = load_json(paths["gain"])
    adaptation = local_adaptation_config(specification, seed)
    if artifact["replication_id"] != specification["replication_id"]:
        raise RuntimeError(f"seed {seed} gain belongs to a different replication")
    if int(artifact["seed"]) != seed or artifact["adaptation"] != asdict(adaptation):
        raise RuntimeError(f"seed {seed} gain adaptation configuration mismatch")
    if artifact["backbone"]["sha256"] != checkpoint_sha256(paths["checkpoint"]):
        raise RuntimeError(f"seed {seed} gain and backbone do not match")
    if (
        artifact["backbone_tensor_hashes_before"]
        != artifact["backbone_tensor_hashes_after"]
    ):
        raise RuntimeError(f"seed {seed} backbone changed during gain adaptation")
    if (
        sum(1 for _ in paths["local_log"].open(encoding="utf-8"))
        != adaptation.outer_steps
    ):
        raise RuntimeError(f"seed {seed} local-gain log length mismatch")
    return paths["gain"]


def adapt_gain(
    specification: dict,
    checkpoint: Path,
    output_root: Path,
    seed: int,
    source_validation: dict,
    runtime: dict,
) -> Path:
    paths = _seed_paths(output_root, seed)
    if paths["local_dir"].exists():
        return _validate_complete_gain(specification, output_root, seed)
    adaptation = local_adaptation_config(specification, seed)
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, adaptation.batch_size
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    before = _tensor_hashes(backbone)
    local = _new_local_trace(seed_specification(specification, seed), model_config.cs)
    tasks = make_gate_tasks(adaptation)
    rng = np.random.default_rng(adaptation.seed)
    np.random.seed(adaptation.seed)
    torch.manual_seed(adaptation.seed)
    training_backbone = torch.compile(backbone, fullgraph=True, mode="default")
    training_local = torch.compile(local, fullgraph=True, mode="default")
    training_write = torch.compile(local.write, fullgraph=True, mode="default")
    optimizer = torch.optim.Adam(
        [local.raw_gain],
        lr=float(specification["local_only_adaptation"]["learning_rate"]),
    )
    paths["local_dir"].mkdir(parents=True, exist_ok=False)
    with paths["local_log"].open("w", encoding="utf-8") as handle:
        for step in range(adaptation.outer_steps):
            optimizer.zero_grad()
            stats = run_local_batch(
                adaptation,
                model_config,
                training_backbone,
                training_local,
                training_write,
                tasks,
                rng,
            )
            stats.loss.backward()
            torch.nn.utils.clip_grad_norm_([local.raw_gain], adaptation.gradient_clip)
            optimizer.step()
            handle.write(
                json.dumps(
                    {
                        "outer_step": step,
                        "query_cross_entropy": stats.query_cross_entropy,
                        "query_accuracy": stats.query_accuracy,
                        "lambda_L": float(local.gain.detach()),
                        "mean_raw_local_margin": stats.mean_raw_local_margin,
                        "mean_absolute_raw_local_margin": stats.mean_absolute_raw_local_margin,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    after = _tensor_hashes(backbone)
    if before != after:
        raise RuntimeError(f"seed {seed} backbone changed during gain adaptation")
    artifact = {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "pilot_id": specification["replication_id"],
        "seed": seed,
        "backbone": asdict(checkpoint_info),
        "adaptation": asdict(adaptation),
        "raw_lambda_L": float(local.raw_gain.detach()),
        "lambda_L": float(local.gain.detach()),
        "backbone_tensor_hashes_before": before,
        "backbone_tensor_hashes_after": after,
        "source_validation": source_validation,
        "runtime": runtime,
    }
    write_json(paths["gain"], artifact)
    return _validate_complete_gain(specification, output_root, seed)


def train_artifacts(
    specification: dict,
    output_root: Path,
    source_validation: dict,
    runtime: dict,
) -> None:
    for seed in _network_seeds(specification):
        checkpoint = train_backbone(specification, output_root, seed, runtime)
        adapt_gain(
            specification,
            checkpoint,
            output_root,
            seed,
            source_validation,
            runtime,
        )


def artifact_lock_document(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    output_root: Path,
) -> dict:
    artifacts = {}
    for seed in _network_seeds(specification):
        _validate_complete_backbone(specification, output_root, seed)
        gain_path = _validate_complete_gain(specification, output_root, seed)
        paths = _seed_paths(output_root, seed)
        gain = load_json(gain_path)
        artifacts[str(seed)] = {
            name: {
                "path": str(paths[name].resolve().relative_to(ROOT)),
                "sha256": file_sha256(paths[name]),
            }
            for name in (
                "checkpoint",
                "backbone_config",
                "backbone_log",
                "backbone_manifest",
                "gain",
                "local_log",
            )
        }
        artifacts[str(seed)]["gain"]["lambda_L"] = gain["lambda_L"]
        artifacts[str(seed)]["backbone_log"]["records"] = specification[
            "v1_backbone_training"
        ]["outer_steps"]
        artifacts[str(seed)]["local_log"]["records"] = specification[
            "local_only_adaptation"
        ]["outer_steps"]
    return {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "freeze_status": "both_network_backbones_and_local_gains_frozen_before_either_liu_evaluation",
        "replication_specification_sha256": file_sha256(specification_path),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "artifacts": artifacts,
        "mandatory_joint_freeze": "Both seed artifact sets were complete before this lock was written; neither seed had been evaluated on Liu.",
        "next_step": "After this lock is committed and pushed, evaluate both mandatory seeds under the unchanged registered contract.",
    }


def validate_artifacts(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    artifact_lock_path: Path,
    output_root: Path,
) -> dict:
    lock = load_json(artifact_lock_path)
    checks = []
    top_level = {
        "replication_specification": (
            specification_path,
            lock["replication_specification_sha256"],
        ),
        "implementation_lock": (
            implementation_lock_path,
            lock["implementation_lock_sha256"],
        ),
    }
    for name, (path, expected) in top_level.items():
        observed = file_sha256(path)
        checks.append(
            {
                "name": name,
                "path": str(path.resolve().relative_to(ROOT)),
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )
    for seed in _network_seeds(specification):
        _validate_complete_backbone(specification, output_root, seed)
        _validate_complete_gain(specification, output_root, seed)
        for name, registration in lock["artifacts"][str(seed)].items():
            path = _resolve(registration["path"])
            observed = file_sha256(path)
            checks.append(
                {
                    "name": f"seed_{seed}_{name}",
                    "path": str(path.relative_to(ROOT)),
                    "observed": observed,
                    "expected": registration["sha256"],
                    "passed": observed == registration["sha256"],
                }
            )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(
            f"conjunctive-local-trace replication artifact lock failed: {checks}"
        )
    return {"passed": True, "checks": checks, "lock": lock}


def _bootstrap_counts(evaluation: dict, seed: int) -> np.ndarray:
    subjects = int(evaluation["subjects"])
    return (
        np.random.default_rng(int(evaluation["bootstrap_seeds"][str(seed)]))
        .multinomial(
            subjects,
            np.full(subjects, 1.0 / subjects),
            size=int(evaluation["bootstrap_samples"]),
        )
        .astype(np.float64)
    )


def _attribution_for_seed(
    specification: dict,
    seed: int,
    checkpoint: Path,
    gain_path: Path,
) -> dict:
    evaluation = specification["liu_evaluation"]
    gain = load_json(gain_path)
    backbone, model_config, _ = load_retro_checkpoint(
        checkpoint, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    local = _new_local_trace(seed_specification(specification, seed), model_config.cs)
    with torch.no_grad():
        local.raw_gain.fill_(float(gain["raw_lambda_L"]))
    protocol = load_ranking_protocol(
        _resolve(specification["registered_sources"]["liu_protocol"]["path"])
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
    pairs = _ordered_pairs(protocol.n_items)
    schedules = tuple(pairs for _ in range(model_config.bs))
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    full_trace = build_local_trace(evaluator, local)
    bundles = {
        condition: query_bundle(
            evaluator,
            local,
            fast_weights,
            full_trace,
            schedules,
            condition=condition,
            shuffle_seed=int(evaluation["local_shuffle_seed"]),
        )
        for condition in CONDITIONS
    }
    relations = tuple(protocol.support_pairs_higher_lower)
    retained = _retained_mask(evaluator, relations)
    self_local = _self_local_margins(
        evaluator, local, _self_traces(evaluator, local, relations), relations
    )
    cells = learned_cells(
        evaluator,
        bundles["original_v1_local_off"],
        bundles["dual_intact"],
        bundles["global_P_off_local_intact"],
        self_local,
        retained,
        float(evaluation["temperature"]),
    )
    counts = _bootstrap_counts(evaluation, seed)
    interval = float(evaluation["bootstrap_interval"])
    probabilities = boundary_and_probability_attribution(cells, counts, interval)
    self_cross = self_cross_attribution(cells, counts, interval)
    local_only = local_only_attribution(cells, counts, interval)
    pair_probabilities = {
        condition: _pair_correct_probabilities(
            evaluator, bundle, float(evaluation["temperature"])
        )
        for condition, bundle in bundles.items()
    }
    slope = slope_decomposition(
        evaluator, pair_probabilities, retained, counts, interval
    )
    return {
        "error_mass": error_mass_attribution(cells, counts, interval),
        "boundary_and_probability": probabilities,
        "self_cross": self_cross,
        "local_only": local_only,
        "exact_probability_slope_decomposition": slope,
        "raw_learned_cells": {
            name: _json_values(value) for name, value in cells.items()
        },
        "integrity": {
            "dual_margin_identity_max_abs_error": float(
                np.max(
                    np.abs(
                        cells["dual_margin"]
                        - cells["global_margin"]
                        - cells["full_local_margin"]
                    )
                )
            ),
            "self_plus_cross_identity_max_abs_error": self_cross[
                "self_plus_cross_identity_max_abs_error"
            ],
            "stable_omitted_self_max_abs": self_cross["stable_omitted_self_max_abs"],
            "slope_additive_identity_max_abs_error": max(
                row["additive_identity_max_abs_error"]
                for row in slope["conditions"].values()
            ),
        },
    }


def within_seed_decision(
    specification: dict,
    pilot: dict,
    attribution: dict,
    seed: int,
) -> dict:
    evaluation = specification["liu_evaluation"]
    counts = _bootstrap_counts(evaluation, seed)
    interval = float(evaluation["bootstrap_interval"])
    local = pilot["local_fidelity"]
    key = "retained_relation_mean_direct_correctness"
    direct = summarize_difference(
        np.asarray(local["dual_intact"]["raw_subject_level"][key]),
        np.asarray(local["original_v1_local_off"]["raw_subject_level"][key]),
        counts,
        interval=interval,
    )
    address = summarize_difference(
        np.asarray(local["dual_intact"]["raw_subject_level"][key]),
        np.asarray(local["local_query_key_shuffle"]["raw_subject_level"][key]),
        counts,
        interval=interval,
    )
    p_off_behavior = pilot["behavior"]["global_P_off_local_intact"]
    learned_minus_nonlearned = summarize_difference(
        _behavior_subject_values(p_off_behavior, "learned_accuracy"),
        _behavior_subject_values(p_off_behavior, "nonlearned_accuracy"),
        counts,
        interval=interval,
    )
    p_off_remote = np.asarray(
        local["global_P_off_local_intact"]["raw_subject_level"]["remote_absolute"]
    )
    v1_remote = np.asarray(
        local["original_v1_local_off"]["raw_subject_level"]["remote_absolute"]
    )
    remote_collapse = summarize_subjects(
        p_off_remote - 0.25 * v1_remote, counts, interval=interval
    )
    exact = attribution["boundary_and_probability"]["delta_probability"]["retained"][
        "summary"
    ]
    self_contribution = attribution["self_cross"]["retained_signed_self"]
    p_off_exact = attribution["local_only"]["P_off_local_intact"]
    source_integrity = all(
        (
            attribution["integrity"]["dual_margin_identity_max_abs_error"] <= 1e-6,
            attribution["integrity"]["self_plus_cross_identity_max_abs_error"] <= 1e-6,
            attribution["integrity"]["stable_omitted_self_max_abs"] <= 1e-7,
            attribution["integrity"]["slope_additive_identity_max_abs_error"] <= 1e-12,
            pilot["integrity"]["local_off_v1_logit_max_abs_error"] <= 1e-6,
            pilot["integrity"]["local_margin_identity_max_abs_error"] <= 1e-6,
            pilot["integrity"]["stable_omitted_max_abs_pair_influence"] <= 1e-7,
        )
    )
    global_branch = bool(pilot["decision"]["flags"]["global_branch_preservation"])
    competence = bool(pilot["original_v1_qualification"]["passed"])
    interpretable = source_integrity and competence
    criteria = {
        "retained_exact_probability_rescue": exact["bootstrap"]["lower"] > 0.0,
        "causal_direct_rescue": direct["bootstrap"]["lower"] > 0.0,
        "address_and_self_specificity": (
            address["bootstrap"]["lower"] > 0.0
            and self_contribution["bootstrap"]["lower"] > 0.0
        ),
        "P_L_double_dissociation": (
            p_off_exact["retained_minus_omitted_exact_probability"]["bootstrap"][
                "lower"
            ]
            > 0.0
            and learned_minus_nonlearned["bootstrap"]["lower"] > 0.0
            and remote_collapse["bootstrap"]["upper"] < 0.0
            and global_branch
        ),
    }
    flags = {name: bool(interpretable and value) for name, value in criteria.items()}
    return {
        "interpretable": interpretable,
        "competence_passed": competence,
        "integrity_passed": source_integrity,
        "all_four_primary_links_pass": all(flags.values()),
        "flags": flags,
        "primary_effects": {
            "retained_dual_minus_v1_exact_probability": exact,
            "dual_minus_v1_direct_correctness": direct,
            "dual_minus_query_shuffle_direct_correctness": address,
            "retained_correct_signed_self_contribution": self_contribution,
            "P_off_retained_minus_omitted_exact_probability": p_off_exact[
                "retained_minus_omitted_exact_probability"
            ],
            "P_off_learned_minus_nonlearned_sampled_accuracy": learned_minus_nonlearned,
            "P_off_remote_minus_quarter_v1_remote": remote_collapse,
        },
        "descriptive_controls": {
            "retained_absolute_cross_to_self_ratio": attribution["self_cross"][
                "retained_absolute_cross_to_self_ratio"
            ],
            "historical_P_off_retained_exact_probability": p_off_exact[
                "retained_exact_probability"
            ],
            "historical_0_65_lower_bound_passed": p_off_exact[
                "retained_exact_probability"
            ]["bootstrap"]["lower"]
            > 0.65,
            "L_off_global_branch_preserved": global_branch,
        },
        "registered_rules": specification["primary_links"],
    }


def cross_seed_decision(specification: dict, seed_results: dict[str, dict]) -> dict:
    seeds = _network_seeds(specification)
    link_names = tuple(specification["primary_links"])
    links = {}
    for name in link_names:
        values = [seed_results[str(seed)]["decision"]["flags"][name] for seed in seeds]
        if all(values):
            status = "replicated"
        elif any(values):
            status = "heterogeneous_or_unresolved"
        else:
            status = "not_replicated"
        links[name] = {
            "status": status,
            "seed_passes": {
                str(seed): bool(value) for seed, value in zip(seeds, values)
            },
        }
    effect_keys = tuple(seed_results[str(seeds[0])]["decision"]["primary_effects"])
    descriptive_means = {
        name: float(
            np.mean(
                [
                    seed_results[str(seed)]["decision"]["primary_effects"][name]["mean"]
                    for seed in seeds
                ]
            )
        )
        for name in effect_keys
    }
    interpretable = all(
        seed_results[str(seed)]["decision"]["interpretable"] for seed in seeds
    )
    all_replicated = interpretable and all(
        row["status"] == "replicated" for row in links.values()
    )
    if not interpretable:
        outcome = "noninterpretable_seed"
    elif all_replicated:
        outcome = "replicated_mechanism"
    elif any(row["status"] == "heterogeneous_or_unresolved" for row in links.values()):
        outcome = "heterogeneous_or_unresolved"
    else:
        outcome = "valid_link_failure"
    return {
        "outcome": outcome,
        "all_four_links_replicate": all_replicated,
        "all_seeds_interpretable": interpretable,
        "links": links,
        "descriptive_two_seed_mean_effects": descriptive_means,
        "network_population_inference": "not_performed",
        "registered_rules": specification["cross_seed_decision"],
    }


def evaluate_replication(
    specification: dict,
    output_root: Path,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    seed_results = {}
    for seed in _network_seeds(specification):
        paths = _seed_paths(output_root, seed)
        pilot = evaluate_pilot(
            seed_specification(specification, seed),
            paths["checkpoint"],
            paths["gain"],
            source_validation,
            artifact_validation,
            runtime,
        )
        pilot["seed"] = seed
        attribution = _attribution_for_seed(
            specification, seed, paths["checkpoint"], paths["gain"]
        )
        decision = within_seed_decision(specification, pilot, attribution, seed)
        seed_results[str(seed)] = {
            "seed": seed,
            "checkpoint": pilot["checkpoint"],
            "gain_artifact": pilot["gain_artifact"],
            "integrity": pilot["integrity"],
            "original_v1_qualification": pilot["original_v1_qualification"],
            "local_fidelity": pilot["local_fidelity"],
            "local_branch_specificity": pilot["local_branch_specificity"],
            "behavior": pilot["behavior"],
            "query_binding": pilot["query_binding"],
            "terminal_projection": pilot["terminal_projection"],
            "residual_fingerprint": pilot["residual_fingerprint"],
            "legacy_v2_3_decision": pilot["decision"],
            "attribution": attribution,
            "decision": decision,
        }
    return {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "seeds": seed_results,
        "decision": cross_seed_decision(specification, seed_results),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered conjunctive-local-trace v2.3 replication."
    )
    parser.add_argument(
        "stage", choices=("train-artifacts", "write-artifact-lock", "evaluate")
    )
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
    if parsed.stage == "train-artifacts":
        train_artifacts(specification, parsed.output_root, source_validation, runtime)
        return 0
    if parsed.stage == "write-artifact-lock":
        if parsed.artifact_lock.exists():
            raise RuntimeError("replication artifact lock already exists")
        write_json(
            parsed.artifact_lock,
            artifact_lock_document(
                specification,
                parsed.specification,
                parsed.implementation_lock,
                parsed.output_root,
            ),
        )
        return 0
    artifact_validation = validate_artifacts(
        specification,
        parsed.specification,
        parsed.implementation_lock,
        parsed.artifact_lock,
        parsed.output_root,
    )
    result = evaluate_replication(
        specification,
        parsed.output_root,
        source_validation,
        artifact_validation,
        runtime,
    )
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
