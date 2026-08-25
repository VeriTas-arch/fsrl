"""Registered frozen-backbone conjunctive local-trace v2.3 pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.evaluation.frozen_fast_weight import (
    DISTANCE_INPUT_OFFSET,
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
    summarize_subjects,
)
from fsrl.experiments.confirmation.behavioral import file_sha256
from fsrl.experiments.local_fidelity.curvature_gate import make_gate_tasks
from fsrl.experiments.local_fidelity.curvature_gate_pilot import (
    _adaptation_config,
    _field_metrics,
    _json_values,
    _ordered_pairs,
    _resolve_registered,
    _retained_mask,
    _tensor_hashes,
    bundle_logits,
    configure_runtime,
    load_json,
    margin_fields,
    query_binding_summary,
    terminal_projection_summary,
    write_json,
)
from fsrl.experiments.local_fidelity.policy_residual import policy_residual_statistics
from fsrl.infrastructure.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.meta_tasks import GenericRankingTaskGenerator, RankingEpisode
from fsrl.tasks.registered_protocol import load_ranking_protocol
from fsrl.training.backbone import MetaTrainConfig, build_meta_input_sequence

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_pilot_v2_3.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_pilot_v2_3.lock.json"
)
DEFAULT_ARTIFACT_LOCK_PATH = resolve_record(
    "benchmarks/conjunctive_local_trace_pilot_v2_3.artifact_lock.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "conjunctive-local-trace-pilot-v2-3"
DEFAULT_RESULT_PATH = resolve_record("results/conjunctive_local_trace_pilot_v2_3.json")
UNSIGNED_SPECIFICATION_PATH = resolve_record("benchmarks/curvature_gate_pilot_v2.json")
CONDITIONS = (
    "original_v1_local_off",
    "dual_intact",
    "local_query_key_shuffle",
    "global_P_off_local_intact",
)
KEY_EPSILON = 1e-8


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
        raise RuntimeError(f"conjunctive-local-trace source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def validate_artifact(
    specification_path: Path,
    implementation_lock_path: Path,
    artifact_lock_path: Path,
    artifact_path: Path,
) -> dict:
    artifact_lock = load_json(artifact_lock_path)
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
        "gain_artifact": (
            artifact_path.resolve(),
            artifact_lock["gain_artifact"]["sha256"],
        ),
    }
    checks = []
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
    declared = _resolve_registered(artifact_lock["gain_artifact"]["path"])
    if declared.resolve() != artifact_path.resolve():
        raise RuntimeError("artifact lock points to a different gain artifact")
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"conjunctive-local-trace artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": artifact_lock}


def _runtime_specification(specification: dict) -> dict:
    unsigned = load_json(UNSIGNED_SPECIFICATION_PATH)
    return {
        **unsigned,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "registered_sources": specification["registered_sources"],
        "development_seed_contract": {"mandatory_seed": 2101},
        "gate_only_adaptation": specification["local_only_adaptation"],
        "liu_evaluation": specification["liu_evaluation"],
        "primary_decision_rules": specification["primary_decision_rules"],
    }


def _new_local_trace(specification: dict, cue_size: int) -> ConjunctiveLocalTrace:
    adaptation = specification["local_only_adaptation"]
    return ConjunctiveLocalTrace(
        cue_size,
        initial_gain=float(adaptation["initial_lambda_L"]),
        epsilon=KEY_EPSILON,
    )


@dataclass(frozen=True)
class LocalBatchStats:
    loss: torch.Tensor
    query_cross_entropy: float
    query_accuracy: float
    mean_raw_local_margin: float
    mean_absolute_raw_local_margin: float


def run_local_batch(
    training_config: MetaTrainConfig,
    model_config,
    backbone,
    local,
    local_write,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
) -> LocalBatchStats:
    """Run one generic batch with exact-v1 global state and the local trace."""

    n_edges = int(
        rng.integers(training_config.min_edges, training_config.max_edges + 1)
    )
    episodes: tuple[RankingEpisode, ...] = tuple(
        task_generator.sample(rng, n_edges=n_edges)
        for _ in range(training_config.batch_size)
    )
    hidden = backbone.initialZeroState(model_config.bs)
    eligibility = backbone.initialZeroET(model_config.bs)
    fast_weights = backbone.initialZeroPlasticWeights(model_config.bs)
    local_state = local.initial_state(model_config.bs)
    blank = torch.zeros(
        model_config.bs, model_config.inputsize, device=fast_weights.device
    )
    for _ in range(2):
        _, _, _, hidden, eligibility, fast_weights = backbone(
            blank, hidden, eligibility, fast_weights
        )

    n_support = len(episodes[0].support_trials)
    zero_hidden = backbone.initialZeroState(model_config.bs)
    zero_eligibility = backbone.initialZeroET(model_config.bs)
    evidence_index = model_config.nbstimbits + DISTANCE_INPUT_OFFSET
    for trial_index in range(n_support):
        hidden = zero_hidden
        eligibility = zero_eligibility
        trials = [episode.support_trials[trial_index] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [trial.signed_magnitude * trial.encoding_reliability for trial in trials],
            dtype=np.float32,
        )
        time_value = (
            trial_index / max(1, n_support - 1) * training_config.support_query_time
        )
        input_sequence = build_meta_input_sequence(
            model_config,
            episodes,
            left,
            right,
            signed,
            num_steps=model_config.triallen,
            time_value=time_value,
            support_trial=True,
        )
        step0 = input_sequence[0]
        local_state = local_write(
            local_state,
            step0[:, : 2 * model_config.cs],
            step0[:, evidence_index],
        )
        for inputs in input_sequence.unbind():
            _, _, _, hidden, eligibility, fast_weights = backbone(
                inputs, hidden, eligibility, fast_weights
            )

    query_loss = torch.zeros((), device=fast_weights.device)
    correct = 0
    total = 0
    raw_sum = torch.zeros((), device=fast_weights.device)
    raw_abs_sum = torch.zeros((), device=fast_weights.device)
    n_queries = len(episodes[0].query_trials)
    for query_index in range(n_queries):
        hidden = zero_hidden
        eligibility = zero_eligibility
        trials = [episode.query_trials[query_index] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        targets = torch.tensor(
            [trial.correct_action for trial in trials],
            dtype=torch.long,
            device=fast_weights.device,
        )
        input_sequence = build_meta_input_sequence(
            model_config,
            episodes,
            left,
            right,
            np.zeros(model_config.bs, dtype=np.float32),
            num_steps=2,
            time_value=training_config.support_query_time,
            support_trial=False,
        )
        step0, response = input_sequence.unbind()
        _, _, _, hidden, eligibility, _ = backbone(
            step0, hidden, eligibility, fast_weights
        )
        logits, _, _, _, _, _ = backbone(response, hidden, eligibility, fast_weights)
        corrected, raw_margin, _gain, _correction = local(
            logits, local_state, step0[:, : 2 * model_config.cs]
        )
        query_loss = query_loss + F.cross_entropy(corrected, targets)
        correct += int(torch.sum(torch.argmax(corrected, dim=1) == targets))
        total += model_config.bs
        raw_sum = raw_sum + torch.sum(raw_margin)
        raw_abs_sum = raw_abs_sum + torch.sum(torch.abs(raw_margin))
    query_loss = query_loss / n_queries
    return LocalBatchStats(
        loss=query_loss,
        query_cross_entropy=float(query_loss.detach()),
        query_accuracy=correct / total,
        mean_raw_local_margin=float(raw_sum.detach()) / total,
        mean_absolute_raw_local_margin=float(raw_abs_sum.detach()) / total,
    )


def adapt_gain(
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
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    before = _tensor_hashes(backbone)
    local = _new_local_trace(specification, model_config.cs)
    task_generator = make_gate_tasks(adaptation)
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
    artifact_dir = output_root / "seed-2101" / "local"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "train_log.jsonl"
    if log_path.exists():
        raise RuntimeError("v2.3 local-gain adaptation log exists; refusing to append")
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(adaptation.outer_steps):
            optimizer.zero_grad()
            stats = run_local_batch(
                adaptation,
                model_config,
                training_backbone,
                training_local,
                training_write,
                task_generator,
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
                        "mean_absolute_raw_local_margin": (
                            stats.mean_absolute_raw_local_margin
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    after = _tensor_hashes(backbone)
    if before != after:
        raise RuntimeError("frozen backbone changed during v2.3 gain adaptation")
    artifact = {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "seed": 2101,
        "backbone": asdict(checkpoint_info),
        "adaptation": asdict(adaptation),
        "raw_lambda_L": float(local.raw_gain.detach()),
        "lambda_L": float(local.gain.detach()),
        "backbone_tensor_hashes_before": before,
        "backbone_tensor_hashes_after": after,
        "source_validation": source_validation,
        "runtime": runtime,
    }
    artifact_path = artifact_dir / "gain.json"
    write_json(artifact_path, artifact)
    return artifact_path


def canonical_derangements(subjects: int, n_items: int, seed: int) -> np.ndarray:
    canonical_count = len(tuple(combinations(range(n_items), 2)))
    identity = np.arange(canonical_count)
    rows = []
    for subject in range(subjects):
        rng = np.random.default_rng(seed + subject)
        for _ in range(1000):
            candidate = rng.permutation(canonical_count)
            if np.all(candidate != identity):
                rows.append(candidate)
                break
        else:
            raise RuntimeError("could not construct local-query derangement")
    return np.asarray(rows, dtype=np.int64)


def shuffled_pair_indices(subjects: int, n_items: int, seed: int) -> np.ndarray:
    canonical = tuple(combinations(range(n_items), 2))
    ordered = _ordered_pairs(n_items)
    ordered_index = {pair: index for index, pair in enumerate(ordered)}
    derangements = canonical_derangements(subjects, n_items, seed)
    rows = np.empty((subjects, len(ordered)), dtype=np.int64)
    canonical_index = {pair: index for index, pair in enumerate(canonical)}
    for subject in range(subjects):
        for index, pair in enumerate(ordered):
            base = tuple(sorted(pair))
            mapped = canonical[int(derangements[subject, canonical_index[base]])]
            oriented = mapped if pair == base else (mapped[1], mapped[0])
            rows[subject, index] = ordered_index[oriented]
    return rows


def build_local_trace(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    *,
    zero_relations: frozenset[tuple[int, int]] = frozenset(),
) -> torch.Tensor:
    state = local.initial_state(evaluator.config.bs)
    evidence_index = evaluator.config.nbstimbits + DISTANCE_INPUT_OFFSET
    with torch.no_grad():
        for trial_index in range(evaluator.protocol.support_trials):
            trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
            left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
            right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
            signed = np.asarray(
                [
                    0.0
                    if (trial.higher_item, trial.lower_item) in zero_relations
                    else trial.signed_magnitude
                    * evaluator._encoding_reliability(subject, trial_index)
                    for subject, trial in enumerate(trials)
                ],
                dtype=np.float32,
            )
            step0 = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=0,
                time_value=(
                    trial_index
                    / max(1, evaluator.protocol.support_trials - 1)
                    * evaluator.test_time_value
                ),
                support_trial=True,
            )
            state = local.write(
                state,
                step0[:, : 2 * evaluator.config.cs],
                step0[:, evidence_index],
            )
    return state.detach().clone()


def _query_pass(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    fast_weights: torch.Tensor,
    local_state: torch.Tensor,
    pair_schedules,
    *,
    local_off: bool,
    global_off: bool,
    shuffled_indices: np.ndarray | None,
) -> dict:
    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    arrays = {
        name: np.empty((subjects, pair_count), dtype=np.float64)
        for name in (
            "logits",
            "global_logits",
            "raw_local_margins",
            "applied_local_margins",
            "local_gains",
            "policy_residuals",
        )
    }
    query_weights = torch.zeros_like(fast_weights) if global_off else fast_weights
    all_pairs = _ordered_pairs(evaluator.protocol.n_items)
    with torch.no_grad():
        for pair_index in range(pair_count):
            hidden = evaluator.net.initialZeroState(subjects)
            eligibility = evaluator.net.initialZeroET(subjects)
            left = np.asarray(
                [schedule[pair_index][0] for schedule in pair_schedules], dtype=np.int64
            )
            right = np.asarray(
                [schedule[pair_index][1] for schedule in pair_schedules], dtype=np.int64
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
                step0, hidden, eligibility, query_weights
            )
            global_output, _, _, _, _, _ = evaluator.net(
                response, hidden, eligibility, query_weights
            )
            local_step0 = step0
            if shuffled_indices is not None:
                mapped = [
                    all_pairs[int(shuffled_indices[subject, pair_index])]
                    for subject in range(subjects)
                ]
                local_step0 = evaluator._step_inputs(
                    np.asarray([pair[0] for pair in mapped], dtype=np.int64),
                    np.asarray([pair[1] for pair in mapped], dtype=np.int64),
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
            gain_override = (
                torch.zeros(subjects, 1, device=fast_weights.device)
                if local_off
                else None
            )
            corrected, raw_margin, gain, correction = local(
                global_output,
                local_state,
                local_step0[:, : 2 * evaluator.config.cs],
                gain_override,
            )
            hidden_column = hidden.view(subjects, evaluator.config.hs, 1)
            baseline = (
                evaluator.net.i2h(response).view(subjects, evaluator.config.hs, 1)
                + torch.matmul(evaluator.net.w, hidden_column)
            ).view(subjects, evaluator.config.hs)
            drive = torch.matmul(
                evaluator.net.alpha * query_weights, hidden_column
            ).view(subjects, evaluator.config.hs)
            margin_direction = evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0]
            residual = policy_residual_statistics(baseline, drive, margin_direction)[0]
            arrays["logits"][:, pair_index] = (
                (corrected[:, 1] - corrected[:, 0]).cpu().numpy()
            )
            arrays["global_logits"][:, pair_index] = (
                (global_output[:, 1] - global_output[:, 0]).cpu().numpy()
            )
            arrays["raw_local_margins"][:, pair_index] = raw_margin[:, 0].cpu().numpy()
            arrays["applied_local_margins"][:, pair_index] = (
                correction[:, 0].cpu().numpy()
            )
            arrays["local_gains"][:, pair_index] = gain[:, 0].cpu().numpy()
            arrays["policy_residuals"][:, pair_index] = residual[:, 0].cpu().numpy()
    return arrays


def query_bundle(
    evaluator,
    local,
    fast_weights,
    local_state,
    pair_schedules,
    *,
    condition: str,
    shuffle_seed: int,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown local-trace condition: {condition}")
    shuffled = None
    if condition == "local_query_key_shuffle":
        shuffled = shuffled_pair_indices(
            evaluator.config.bs, evaluator.protocol.n_items, shuffle_seed
        )
    return _query_pass(
        evaluator,
        local,
        fast_weights,
        local_state,
        pair_schedules,
        local_off=condition == "original_v1_local_off",
        global_off=condition == "global_P_off_local_intact",
        shuffled_indices=shuffled,
    )


def _local_specificity(
    intact_field: np.ndarray,
    loo_fields: np.ndarray,
    relations,
    retained: np.ndarray,
    geometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    influence = intact_field[None] - loo_fields
    direct = np.empty(retained.shape, dtype=np.float64)
    remote = np.empty(retained.shape, dtype=np.float64)
    for relation_index, relation in enumerate(relations):
        edge = geometry.pairs.index(tuple(sorted(relation)))
        direct[relation_index] = np.abs(influence[relation_index, :, edge])
        endpoints = set(relation)
        remote_mask = np.asarray(
            [not endpoints.intersection(pair) for pair in geometry.pairs], dtype=bool
        )
        remote[relation_index] = np.mean(
            np.abs(influence[relation_index][:, remote_mask]), axis=1
        )
    masked = retained.astype(np.float64)
    denominator = np.sum(masked, axis=0)
    subject_direct = np.sum(np.where(retained, direct, 0.0), axis=0) / denominator
    subject_remote = np.sum(np.where(retained, remote, 0.0), axis=0) / denominator
    difference = subject_direct - 3.0 * subject_remote
    return {
        "summary": {
            "direct_absolute": summarize_subjects(
                subject_direct, counts, interval=interval
            ),
            "remote_absolute": summarize_subjects(
                subject_remote, counts, interval=interval
            ),
            "direct_minus_three_remote": summarize_subjects(
                difference, counts, interval=interval
            ),
        },
        "raw_subject_level": {
            "direct_absolute": _json_values(subject_direct),
            "remote_absolute": _json_values(subject_remote),
            "direct_minus_three_remote": _json_values(difference),
        },
        "raw_relation_subject": {
            "direct_absolute": _json_values(direct),
            "remote_absolute": _json_values(remote),
        },
    }


def _behavior_subject_values(result: dict, name: str) -> np.ndarray:
    return np.asarray([row[name] for row in result["subjects"]], dtype=np.float64)


def _behavior_summaries(result: dict, counts: np.ndarray, interval: float) -> dict:
    return {
        name: summarize_subjects(
            _behavior_subject_values(result, name), counts, interval=interval
        )
        for name in ("overall_accuracy", "learned_accuracy", "nonlearned_accuracy")
    }


def decision_summary(
    specification: dict,
    original_qualification: dict,
    local: dict,
    local_specificity: dict,
    behavior: dict,
    binding: dict,
    terminal: dict,
) -> dict:
    counts = local["counts"]
    interval = local["interval"]
    key = "retained_relation_mean_direct_correctness"
    h_key = "H_greater_A_direct_correctness"
    dual = local["dual_intact"]
    original = local["original_v1_local_off"]
    shuffled = local["local_query_key_shuffle"]
    p_off = local["global_P_off_local_intact"]
    contrasts = {
        "dual_minus_original_local": summarize_difference(
            dual["subject_level"][key],
            original["subject_level"][key],
            counts,
            interval=interval,
        ),
        "dual_minus_original_H_greater_A": summarize_difference(
            dual["subject_level"][h_key],
            original["subject_level"][h_key],
            counts,
            interval=interval,
        ),
        "dual_minus_shuffled_local": summarize_difference(
            dual["subject_level"][key],
            shuffled["subject_level"][key],
            counts,
            interval=interval,
        ),
    }
    behavior_contrasts = {
        "dual_minus_original_learned_accuracy": summarize_difference(
            _behavior_subject_values(behavior["dual_intact"], "learned_accuracy"),
            _behavior_subject_values(
                behavior["original_v1_local_off"], "learned_accuracy"
            ),
            counts,
            interval=interval,
        ),
        "dual_minus_original_nonlearned_accuracy": summarize_difference(
            _behavior_subject_values(behavior["dual_intact"], "nonlearned_accuracy"),
            _behavior_subject_values(
                behavior["original_v1_local_off"], "nonlearned_accuracy"
            ),
            counts,
            interval=interval,
        ),
        "P_off_learned_minus_nonlearned": summarize_difference(
            _behavior_subject_values(
                behavior["global_P_off_local_intact"], "learned_accuracy"
            ),
            _behavior_subject_values(
                behavior["global_P_off_local_intact"], "nonlearned_accuracy"
            ),
            counts,
            interval=interval,
        ),
    }
    aggregate_rescue = (
        contrasts["dual_minus_original_local"]["bootstrap"]["lower"] > 0.0
    )
    h_rescue = (
        contrasts["dual_minus_original_H_greater_A"]["bootstrap"]["lower"] > 0.0
        and dual["summary"][h_key]["bootstrap"]["upper"] >= 0.0
    )
    learned_rescue = (
        behavior_contrasts["dual_minus_original_learned_accuracy"]["bootstrap"]["lower"]
        > 0.0
    )
    address_specificity = (
        contrasts["dual_minus_shuffled_local"]["bootstrap"]["lower"] > 0.0
    )
    direct_specificity = (
        local_specificity["summary"]["direct_minus_three_remote"]["bootstrap"]["lower"]
        > 0.0
    )
    p_behavior = {
        name: _behavior_summaries(
            behavior["global_P_off_local_intact"], counts, interval
        )[name]
        for name in ("learned_accuracy", "nonlearned_accuracy")
    }
    local_only = (
        p_behavior["learned_accuracy"]["bootstrap"]["lower"] > 0.55
        and p_behavior["nonlearned_accuracy"]["bootstrap"]["upper"] <= 0.60
        and behavior_contrasts["P_off_learned_minus_nonlearned"]["bootstrap"]["lower"]
        > 0.10
    )
    remote_collapse_contrast = summarize_subjects(
        p_off["subject_level"]["remote_absolute"]
        - 0.25 * original["subject_level"]["remote_absolute"],
        counts,
        interval=interval,
    )
    remote_collapse = remote_collapse_contrast["bootstrap"]["upper"] < 0.0
    dual_nonlearned = _behavior_summaries(behavior["dual_intact"], counts, interval)[
        "nonlearned_accuracy"
    ]
    nonlearned = (
        dual_nonlearned["mean"] >= 0.70
        and behavior_contrasts["dual_minus_original_nonlearned_accuracy"]["bootstrap"][
            "lower"
        ]
        >= -0.02
    )
    global_reassembly = all(
        dual["summary"][metric]["bootstrap"]["lower"] > 0.0
        and dual["summary"][metric]["mean"] >= 0.8 * original["summary"][metric]["mean"]
        for metric in ("remote_absolute", "gauge_invariant_R_third_rel")
    )
    global_branch = (
        original_qualification["passed"]
        and binding["conditioned_minus_original_max_abs"] <= 1e-7
        and binding["original_v1"]["matched_minus_shared_endpoint"]["mean"] > 0.0
        and binding["original_v1"]["matched_minus_disjoint"]["mean"] > 0.0
        and terminal["summary"]["original_v1"]["bootstrap"]["lower"] > 0.0
    )
    flags = {
        "aggregate_local_rescue": bool(aggregate_rescue),
        "H_greater_A_rescue": bool(h_rescue),
        "learned_accuracy_rescue": bool(learned_rescue),
        "query_address_specificity": bool(address_specificity),
        "local_direct_specificity": bool(direct_specificity),
        "local_only_direct_memory": bool(local_only),
        "global_off_remote_collapse": bool(remote_collapse),
        "nonlearned_preservation": bool(nonlearned),
        "global_reassembly_preservation": bool(global_reassembly),
        "global_branch_preservation": bool(global_branch),
    }
    all_pass = all(flags.values())
    double_dissociation = all(
        flags[name]
        for name in (
            "query_address_specificity",
            "local_direct_specificity",
            "local_only_direct_memory",
            "global_off_remote_collapse",
            "global_branch_preservation",
        )
    )
    if not original_qualification["passed"]:
        outcome = "competence_or_integrity_failure"
    elif all_pass:
        outcome = "all_primary_rules_pass"
    elif aggregate_rescue and not double_dissociation:
        outcome = "local_rescue_without_double_dissociation"
    elif double_dissociation and not aggregate_rescue:
        outcome = "double_dissociation_without_local_rescue"
    else:
        outcome = "valid_local_or_specificity_failure"
    return {
        "all_primary_rules_pass": all_pass,
        "outcome": outcome,
        "flags": flags,
        "registered_rules": specification["primary_decision_rules"],
        "paired_contrasts": contrasts,
        "behavior_contrasts": behavior_contrasts,
        "global_off_remote_collapse_contrast": remote_collapse_contrast,
        "global_P_off_behavior": p_behavior,
    }


def _residual_fingerprint(
    evaluator,
    dual_bundle: dict,
    retained: np.ndarray,
    local_metrics: dict,
) -> dict:
    applied = np.abs(dual_bundle["applied_local_margins"])
    residual = np.abs(dual_bundle["policy_residuals"])
    correlation = float(np.corrcoef(applied.ravel(), residual.ravel())[0, 1])
    threshold = float(np.quantile(residual, 0.75))
    high = residual >= threshold
    relation_rows = []
    labels = evaluator.protocol.item_labels
    direct = np.asarray(
        local_metrics["raw_relation_subject"]["direct_correctness"], dtype=np.float64
    )
    for index, relation in enumerate(evaluator.protocol.support_pairs_higher_lower):
        mask = retained[index]
        relation_rows.append(
            {
                "relation": f"{labels[relation[0]]}>{labels[relation[1]]}",
                "retained_subjects": int(np.sum(mask)),
                "mean_dual_direct_correctness": float(np.mean(direct[index, mask])),
            }
        )
    return {
        "absolute_local_margin_vs_absolute_v1_policy_residual_correlation": correlation,
        "v1_residual_top_quartile_threshold": threshold,
        "mean_absolute_local_margin_top_residual_quartile": float(
            np.mean(applied[high])
        ),
        "mean_absolute_local_margin_other_cells": float(np.mean(applied[~high])),
        "by_relation": relation_rows,
        "raw_intact_absolute_local_margin": _json_values(applied),
        "raw_intact_absolute_policy_residual": _json_values(residual),
    }


def evaluate_pilot(
    specification: dict,
    checkpoint: Path,
    artifact_path: Path,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    evaluation = specification["liu_evaluation"]
    artifact = load_json(artifact_path)
    if artifact["pilot_id"] != specification["pilot_id"]:
        raise RuntimeError("gain artifact belongs to a different pilot")
    if artifact["backbone"]["sha256"] != checkpoint_sha256(checkpoint):
        raise RuntimeError("gain artifact and frozen backbone do not match")
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        checkpoint, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    local_module = _new_local_trace(specification, model_config.cs)
    with torch.no_grad():
        local_module.raw_gain.fill_(float(artifact["raw_lambda_L"]))
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
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    intact_local = build_local_trace(evaluator, local_module)
    relations = tuple(protocol.support_pairs_higher_lower)
    loo_fast_weights = []
    loo_local = []
    for relation in relations:
        state = evaluator.initialize_fast_weights()
        for trial_index in range(protocol.support_trials):
            state = evaluator.advance_support_trial(
                state, trial_index, zero_relations=frozenset((relation,))
            )
        loo_fast_weights.append(state)
        loo_local.append(
            build_local_trace(
                evaluator, local_module, zero_relations=frozenset((relation,))
            )
        )
    loo_fast_weights_tensor = torch.stack(loo_fast_weights)
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

    local = {"counts": counts, "interval": interval}
    behavior = {}
    condition_bundles = {}
    condition_fields = {}
    local_branch_fields = None
    common_mode_errors = []
    for condition in CONDITIONS:
        intact_bundle = query_bundle(
            evaluator,
            local_module,
            intact_fast_weights,
            intact_local,
            schedules,
            condition=condition,
            shuffle_seed=int(evaluation["local_shuffle_seed"]),
        )
        loo_bundles = [
            query_bundle(
                evaluator,
                local_module,
                loo_fast_weights[index],
                loo_local[index],
                schedules,
                condition=condition,
                shuffle_seed=int(evaluation["local_shuffle_seed"]),
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
        behavior[condition]["participant_bootstrap"] = _behavior_summaries(
            behavior[condition], counts, interval
        )
        common_mode_errors.append(
            float(
                np.max(
                    np.abs(
                        (intact_bundle["logits"] - intact_bundle["global_logits"])
                        - intact_bundle["applied_local_margins"]
                    )
                )
            )
        )
        if condition == "dual_intact":
            local_intact_field = margin_fields(
                {"logits": intact_bundle["applied_local_margins"]}, protocol.n_items
            )
            local_loo_fields = np.asarray(
                [
                    margin_fields(
                        {"logits": bundle["applied_local_margins"]},
                        protocol.n_items,
                    )
                    for bundle in loo_bundles
                ]
            )
            local_branch_fields = (local_intact_field, local_loo_fields)

    assert local_branch_fields is not None
    specificity = _local_specificity(
        local_branch_fields[0],
        local_branch_fields[1],
        relations,
        retained,
        geometry,
        counts,
        interval,
    )
    original_readout = evaluator.readout_logits(intact_fast_weights, schedules)
    original_logits = bundle_logits(
        condition_bundles["original_v1_local_off"], schedules
    )
    equivalence_error = max(
        abs(original_readout[subject][pair] - original_logits[subject][pair])
        for subject in range(model_config.bs)
        for pair in pairs
    )
    if equivalence_error > 1e-6:
        raise RuntimeError("local-off failed to reproduce frozen v1 logits")
    common_mode_error = max(common_mode_errors, default=0.0)
    if common_mode_error > 1e-6:
        raise RuntimeError("local correction margin identity failed")
    derangements = canonical_derangements(
        model_config.bs, protocol.n_items, int(evaluation["local_shuffle_seed"])
    )
    if np.any(derangements == np.arange(derangements.shape[1])[None]):
        raise RuntimeError("local-query shuffle is not a derangement")
    natural_local_stable_omitted = float(
        local["dual_intact"]["summary"]["stable_omitted_max_abs_pair_influence"]
    )
    if natural_local_stable_omitted > 1e-7:
        raise RuntimeError("stable-omitted local influence is nonzero")

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
    binding = query_binding_summary(
        evaluator,
        intact_fast_weights,
        loo_fast_weights_tensor,
        retained,
        counts,
        interval,
    )
    terminal = terminal_projection_summary(
        evaluator,
        geometry,
        {
            "original_v1": condition_fields["original_v1_local_off"],
            "conditioned_gate": condition_fields["dual_intact"],
        },
        counts,
        interval,
        float(evaluation["posterior_temperature"]),
    )
    terminal["summary"]["dual_intact"] = terminal["summary"].pop("conditioned_gate")
    terminal["raw_subject_level"]["dual_intact"] = terminal["raw_subject_level"].pop(
        "conditioned_gate"
    )
    decision = decision_summary(
        specification,
        original_qualification,
        local,
        specificity,
        behavior,
        binding,
        terminal,
    )
    local_output = {
        condition: {
            "summary": local[condition]["summary"],
            "raw_subject_level": {
                key: _json_values(value)
                for key, value in local[condition]["subject_level"].items()
            },
            "raw_relation_subject": local[condition]["raw_relation_subject"],
        }
        for condition in CONDITIONS
    }
    return {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "seed": 2101,
        "checkpoint": asdict(checkpoint_info),
        "gain_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": file_sha256(artifact_path),
            "lambda_L": artifact["lambda_L"],
            "same_gain_for_all_conditions": True,
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "integrity": {
            "local_off_v1_logit_max_abs_error": equivalence_error,
            "local_margin_identity_max_abs_error": common_mode_error,
            "stable_omitted_max_abs_pair_influence": natural_local_stable_omitted,
            "all_shuffle_maps_are_derangements": True,
            "local_gain_is_only_trainable_parameter": True,
        },
        "local_fidelity": local_output,
        "local_branch_specificity": specificity,
        "behavior": behavior,
        "original_v1_qualification": original_qualification,
        "query_binding": binding,
        "terminal_projection": terminal,
        "residual_fingerprint": _residual_fingerprint(
            evaluator,
            condition_bundles["dual_intact"],
            retained,
            local_output["dual_intact"],
        ),
        "decision": decision,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen-backbone conjunctive local-trace v2.3 pilot."
    )
    parser.add_argument("stage", choices=("adapt-gain", "evaluate"))
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
    artifact = parsed.output_root / "seed-2101" / "local" / "gain.json"
    if parsed.stage == "adapt-gain":
        adapt_gain(
            specification,
            checkpoint,
            parsed.output_root,
            source_validation,
            runtime,
        )
        return 0
    if not artifact.is_file():
        raise FileNotFoundError("frozen v2.3 gain artifact is missing")
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
