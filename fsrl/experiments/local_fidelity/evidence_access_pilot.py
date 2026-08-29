"""Registered zero-parameter dual-evidence-access v2.4 pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.analysis.policy import bundle_logits, exact_probability, margin_fields
from fsrl.analysis.statistics import (
    json_values,
    masked_column_mean,
    summarize_difference,
    summarize_subjects,
)
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_frozen_retro_checkpoint,
    retained_relation_mask,
)
from fsrl.evaluation.relational_query import readout_relational_query_bundle
from fsrl.experiments.local_fidelity.behavior_attribution import (
    pair_correct_probabilities,
)
from fsrl.experiments.local_fidelity.trace_pilot import (
    behavior_summaries,
    build_local_trace,
    create_local_trace,
    shuffled_pair_indices,
)
from fsrl.experiments.local_fidelity.trace_replication import (
    DEFAULT_ARTIFACT_LOCK_PATH as V2_3_ARTIFACT_LOCK_PATH,
)
from fsrl.experiments.local_fidelity.trace_replication import (
    DEFAULT_IMPLEMENTATION_LOCK_PATH as V2_3_IMPLEMENTATION_LOCK_PATH,
)
from fsrl.experiments.local_fidelity.trace_replication import (
    DEFAULT_OUTPUT_ROOT as V2_3_OUTPUT_ROOT,
)
from fsrl.experiments.local_fidelity.trace_replication import (
    DEFAULT_SPECIFICATION_PATH as V2_3_SPECIFICATION_PATH,
)
from fsrl.experiments.local_fidelity.trace_replication import (
    seed_paths,
    seed_specification,
    validate_artifacts,
)
from fsrl.infra.formal_runtime import configure_formal_cuda_runtime
from fsrl.infra.provenance import (
    load_json,
    tensor_hashes,
    write_json_exclusive,
)
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import (
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.evidence import broader_local_admission
from fsrl.tasks.protocol import load_ranking_protocol, ordered_pairs

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/dual_evidence_access_pilot_v2_4.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/dual_evidence_access_pilot_v2_4.repair1.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/dual_evidence_access_pilot_v2_4.json")
V2_3_RESULT_PATH = resolve_record(
    "results/conjunctive_local_trace_replication_v2_3.json"
)
CONDITIONS = (
    "local_off_v1",
    "shared_access_v2_3",
    "dual_access_matched",
    "dual_access_evidence_shuffle",
    "dual_access_query_shuffle",
    "global_P_off_shared_access",
    "global_P_off_dual_access",
)


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    registrations = {
        **specification["registered_sources"],
        "v2_4_specification": {
            "path": str(specification_path.resolve()),
            "sha256": lock["pilot_specification_sha256"],
        },
        **lock["implementation_sources"],
        **lock["reused_frozen_sources"],
    }
    checks = []
    for name, registration in registrations.items():
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
        raise RuntimeError(f"dual-evidence-access source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def access_factor(global_admission: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Broaden local admission while preserving every retained write exactly."""

    admission = np.asarray(global_admission, dtype=np.float64)
    probability = np.asarray(reliability, dtype=np.float64)
    if admission.shape != probability.shape:
        raise ValueError("global admission and reliability must have the same shape")
    if not np.all((admission == 0.0) | (admission == 1.0)):
        raise ValueError("global admission must be binary")
    if not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("reliability must lie in [0, 1]")
    return broader_local_admission(admission, probability)


def blockwise_derangements(
    subjects: int, blocks: int, block_size: int, seed: int
) -> np.ndarray:
    if subjects < 1 or blocks < 1 or block_size < 2:
        raise ValueError("invalid blockwise-derangement dimensions")
    identity = np.arange(block_size)
    maps = np.empty((subjects, blocks, block_size), dtype=np.int64)
    for subject in range(subjects):
        for block in range(blocks):
            rng = np.random.default_rng(seed + subject * blocks + block)
            for _ in range(1000):
                candidate = rng.permutation(block_size)
                if np.all(candidate != identity):
                    maps[subject, block] = candidate
                    break
            else:
                raise RuntimeError("could not construct support-evidence derangement")
    return maps


def apply_blockwise_route(values: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Route donor support scalars to recipient slots under fixed block maps."""

    scalars = np.asarray(values)
    if scalars.ndim != 2 or maps.ndim != 3:
        raise ValueError("values or maps have the wrong rank")
    subjects, trials = scalars.shape
    if maps.shape[0] != subjects or maps.shape[1] * maps.shape[2] != trials:
        raise ValueError("values do not match block maps")
    block_size = maps.shape[2]
    routed = np.empty_like(scalars)
    for subject in range(subjects):
        for block in range(maps.shape[1]):
            start = block * block_size
            routed[subject, start : start + block_size] = scalars[
                subject, start + maps[subject, block]
            ]
    return routed


def _relation_probability(
    evaluator: FrozenFastWeightEvaluator, subject: int, higher: int, lower: int
) -> float:
    if evaluator.subject_encoding_states is None:
        raise RuntimeError("dual access requires the frozen subject encoding state")
    distance = evaluator.item_rank[lower] - evaluator.item_rank[higher]
    return evaluator.subject_encoding_states[subject].relation_reliability(
        higher, lower, distance
    )


@dataclass(frozen=True)
class AccessTrace:
    state: torch.Tensor
    natural_scalars: np.ndarray
    applied_scalars: np.ndarray
    route_maps: np.ndarray | None


def _natural_local_scalars(
    evaluator: FrozenFastWeightEvaluator,
    *,
    dual_access: bool,
    zero_relations: frozenset[tuple[int, int]],
) -> np.ndarray:
    values = np.empty(
        (evaluator.config.bs, evaluator.protocol.support_trials), dtype=np.float32
    )
    for subject, schedule in enumerate(evaluator.support_schedules):
        for trial_index, trial in enumerate(schedule):
            relation = (trial.higher_item, trial.lower_item)
            if relation in zero_relations:
                values[subject, trial_index] = 0.0
                continue
            admission = evaluator._encoding_reliability(subject, trial_index)
            if dual_access:
                probability = _relation_probability(
                    evaluator, subject, trial.higher_item, trial.lower_item
                )
                admission = float(
                    access_factor(np.asarray([admission]), np.asarray([probability]))[0]
                )
            values[subject, trial_index] = trial.signed_magnitude * admission
    return values


def build_access_trace(
    evaluator: FrozenFastWeightEvaluator,
    local,
    *,
    dual_access: bool,
    route_maps: np.ndarray | None = None,
    zero_relations: frozenset[tuple[int, int]] = frozenset(),
) -> AccessTrace:
    natural = _natural_local_scalars(
        evaluator, dual_access=dual_access, zero_relations=zero_relations
    )
    applied = (
        natural if route_maps is None else apply_blockwise_route(natural, route_maps)
    )
    state = local.initial_state(evaluator.config.bs)
    with torch.no_grad():
        for trial_index in range(evaluator.protocol.support_trials):
            trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
            left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
            right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
            step0 = evaluator._step_inputs(
                left,
                right,
                applied[:, trial_index],
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
                torch.from_numpy(applied[:, trial_index]).to(state.device),
            )
    return AccessTrace(state.detach().clone(), natural, applied, route_maps)


def build_fast_weight_loo(evaluator, relations) -> torch.Tensor:
    rows = []
    for relation in relations:
        state = evaluator.initialize_fast_weights()
        for trial_index in range(evaluator.protocol.support_trials):
            state = evaluator.advance_support_trial(
                state, trial_index, zero_relations=frozenset((relation,))
            )
        rows.append(state)
    return torch.stack(rows)


def field_metrics(
    intact: np.ndarray,
    loo: np.ndarray,
    relations,
    retained: np.ndarray,
    geometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    influence = intact[None] - loo
    gradient = np.einsum("ef,rsf->rse", geometry.projection, influence)
    residual = influence - gradient
    direct = np.empty(retained.shape, dtype=np.float64)
    remote = np.empty_like(direct)
    third_party = np.empty_like(direct)
    intact_potential = hodge_potentials(intact, geometry)
    for relation_index, relation in enumerate(relations):
        edge = geometry.pairs.index(tuple(sorted(relation)))
        direct[relation_index] = (
            residual[relation_index, :, edge] * geometry.true_sign[edge]
        )
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

    groups = {
        "retained": retained,
        "omitted": ~retained,
        "all": np.ones_like(retained, dtype=bool),
    }
    subject_level = {}
    summary = {}
    for group, mask in groups.items():
        subject_level[group] = {
            "direct_correctness": masked_column_mean(direct, mask),
            "remote_absolute": masked_column_mean(remote, mask),
            "third_party_relational": masked_column_mean(third_party, mask),
        }
        summary[group] = {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in subject_level[group].items()
        }
    return {
        "summary": summary,
        "raw_subject_level": {
            group: {name: json_values(values) for name, values in row.items()}
            for group, row in subject_level.items()
        },
        "raw_relation_subject": {
            "retained": retained.astype(int).tolist(),
            "direct_correctness": json_values(direct),
            "remote_absolute": json_values(remote),
            "third_party_relational": json_values(third_party),
        },
    }


def learned_probabilities(evaluator, bundle: dict, temperature: float) -> np.ndarray:
    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    pair_index = {
        pair: index
        for index, pair in enumerate(ordered_pairs(evaluator.protocol.n_items))
    }
    values = np.empty((evaluator.config.bs, len(relations), 2), dtype=np.float64)
    for relation_index, relation in enumerate(relations):
        for orientation, pair in enumerate((relation, relation[::-1])):
            sign = 1.0 if orientation == 0 else -1.0
            margin = sign * (
                bundle["global_logits"][:, pair_index[pair]]
                + bundle["applied_local_margins"][:, pair_index[pair]]
            )
            values[:, relation_index, orientation] = exact_probability(
                margin, temperature
            )
    return values


def _probability_metrics(
    probabilities: np.ndarray,
    retained_relation_subject: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    retained = np.broadcast_to(
        retained_relation_subject.T[:, :, None], probabilities.shape
    )

    def subject_mean(mask):
        rows = np.where(mask, probabilities, np.nan).reshape(probabilities.shape[0], -1)
        finite = np.sum(np.isfinite(rows), axis=1)
        return np.divide(
            np.nansum(rows, axis=1),
            finite,
            out=np.full(rows.shape[0], np.nan, dtype=np.float64),
            where=finite > 0,
        )

    raw = {
        "retained": subject_mean(retained),
        "omitted": subject_mean(~retained),
        "all": np.mean(probabilities, axis=(1, 2)),
    }
    return {
        "summary": {
            group: summarize_subjects(values, counts, interval=interval)
            for group, values in raw.items()
        },
        "raw_subject_level": {
            group: json_values(values) for group, values in raw.items()
        },
        "raw_relation_orientation": json_values(probabilities),
    }


def _bootstrap_counts(specification: dict, seed: int) -> np.ndarray:
    evaluation = specification["liu_evaluation"]
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


def _exact_slope_decomposition(
    evaluator,
    condition_probabilities: dict[str, np.ndarray],
    retained_relation_subject: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    protocol = evaluator.protocol
    canonical = tuple(combinations(range(protocol.n_items), 2))
    pair_index = {pair: index for index, pair in enumerate(canonical)}
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    distance = np.asarray(
        [abs(rank[first] - rank[second]) for first, second in canonical],
        dtype=np.float64,
    )
    centered_distance = distance - np.mean(distance)
    denominator = float(np.sum(centered_distance**2))
    learned_retained = np.zeros((evaluator.config.bs, len(canonical)), dtype=bool)
    learned_omitted = np.zeros_like(learned_retained)
    for relation_index, relation in enumerate(protocol.support_pairs_higher_lower):
        edge = pair_index[tuple(sorted(relation))]
        learned_retained[:, edge] = retained_relation_subject[relation_index]
        learned_omitted[:, edge] = ~retained_relation_subject[relation_index]
    masks = {
        "learned_retained": learned_retained,
        "learned_omitted": learned_omitted,
        "nonlearned": ~(learned_retained | learned_omitted),
    }
    output = {"denominator": denominator, "conditions": {}}
    for condition, probabilities in condition_probabilities.items():
        centered = probabilities - np.mean(probabilities, axis=1, keepdims=True)
        numerator = centered_distance[None] * centered
        contributions = {
            name: np.sum(np.where(mask, numerator, 0.0), axis=1) / denominator
            for name, mask in masks.items()
        }
        total = np.sum(numerator, axis=1) / denominator
        identity = float(
            np.max(abs(total - sum(contributions.values(), np.zeros_like(total))))
        )
        output["conditions"][condition] = {
            "total": summarize_subjects(total, counts, interval=interval),
            "group_contributions": {
                name: summarize_subjects(values, counts, interval=interval)
                for name, values in contributions.items()
            },
            "additive_identity_max_abs_error": identity,
            "raw_subject_total": json_values(total),
            "raw_subject_group_contributions": {
                name: json_values(values) for name, values in contributions.items()
            },
        }
    return output


def _condition_configuration(condition: str) -> tuple[str, bool, bool, bool]:
    configurations = {
        "local_off_v1": ("shared", True, False, False),
        "shared_access_v2_3": ("shared", False, False, False),
        "dual_access_matched": ("dual", False, False, False),
        "dual_access_evidence_shuffle": ("evidence_shuffle", False, False, False),
        "dual_access_query_shuffle": ("dual", False, False, True),
        "global_P_off_shared_access": ("shared", False, True, False),
        "global_P_off_dual_access": ("dual", False, True, False),
    }
    return configurations[condition]


def measure_presentation_invariance(
    evaluator, local, natural_scalars: np.ndarray
) -> dict:
    support_error = 0.0
    with torch.no_grad():
        for trial_index in range(evaluator.protocol.support_trials):
            trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
            left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
            right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
            zero = np.zeros(evaluator.config.bs, dtype=np.float32)
            forward = evaluator._step_inputs(
                left,
                right,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=True,
            )
            reverse = evaluator._step_inputs(
                right,
                left,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=True,
            )
            scalar = torch.from_numpy(natural_scalars[:, trial_index]).to(
                forward.device
            )[:, None]
            support_error = max(
                support_error,
                float(
                    torch.max(
                        torch.abs(
                            scalar * local.key(forward[:, : 2 * evaluator.config.cs])
                            - (-scalar)
                            * local.key(reverse[:, : 2 * evaluator.config.cs])
                        )
                    ).cpu()
                ),
            )

        query_error = 0.0
        zero = np.zeros(evaluator.config.bs, dtype=np.float32)
        for left_item, right_item in combinations(range(evaluator.protocol.n_items), 2):
            left = np.full(evaluator.config.bs, left_item, dtype=np.int64)
            right = np.full(evaluator.config.bs, right_item, dtype=np.int64)
            forward = evaluator._step_inputs(
                left,
                right,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=False,
            )
            reverse = evaluator._step_inputs(
                right,
                left,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=False,
            )
            query_error = max(
                query_error,
                float(
                    torch.max(
                        torch.abs(
                            local.key(forward[:, : 2 * evaluator.config.cs])
                            + local.key(reverse[:, : 2 * evaluator.config.cs])
                        )
                    ).cpu()
                ),
            )
    return {
        "support_write_reversal_max_abs_error": support_error,
        "query_key_reversal_max_abs_error": query_error,
    }


def _paired(
    first: list | np.ndarray,
    second: list | np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    return summarize_difference(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        counts,
        interval=interval,
    )


def _shared_identity(
    seed: int,
    shared_trace: torch.Tensor,
    legacy_trace: torch.Tensor,
    probabilities: dict,
    fields: dict,
    behavior: dict,
    frozen_seed: dict,
) -> dict:
    frozen_probability = np.asarray(
        frozen_seed["attribution"]["raw_learned_cells"]["p_dual"],
        dtype=np.float64,
    )
    frozen_direct = np.asarray(
        frozen_seed["local_fidelity"]["dual_intact"]["raw_relation_subject"][
            "direct_correctness"
        ],
        dtype=np.float64,
    )
    current_direct = np.asarray(
        fields["shared_access_v2_3"]["raw_relation_subject"]["direct_correctness"],
        dtype=np.float64,
    )
    frozen_behavior = frozen_seed["behavior"]["dual_intact"]["summary"]
    current_behavior = behavior["shared_access_v2_3"]["summary"]
    behavior_names = ("learned_accuracy", "nonlearned_accuracy", "overall_accuracy")
    behavior_errors = [
        abs(current_behavior[name] - frozen_behavior[name]) for name in behavior_names
    ]
    behavior_errors.append(
        abs(
            current_behavior["symbolic_distance_slope"]["mean"]
            - frozen_behavior["symbolic_distance_slope"]["mean"]
        )
    )
    return {
        "seed": seed,
        "local_state_max_abs_error": float(
            torch.max(torch.abs(shared_trace - legacy_trace)).cpu()
        ),
        "exact_probability_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(
                        probabilities["shared_access_v2_3"]["raw_relation_orientation"]
                    )
                    - frozen_probability
                )
            )
        ),
        "direct_causal_max_abs_error": float(
            np.max(np.abs(current_direct - frozen_direct))
        ),
        "sampled_behavior_max_abs_error": float(max(behavior_errors, default=0.0)),
    }


def within_seed_decision(
    specification: dict,
    seed: int,
    contrasts: dict,
    probability: dict,
    field: dict,
    behavior: dict,
    integrity: dict,
    global_branch: bool,
) -> dict:
    retained_margin = -0.005
    omitted = contrasts["dual_minus_shared_omitted_exact_probability"]
    omitted_direct = contrasts["dual_minus_shared_omitted_direct_correctness"]
    retained = contrasts["dual_minus_shared_retained_exact_probability"]
    evidence_probability = contrasts[
        "dual_minus_evidence_shuffle_omitted_exact_probability"
    ]
    evidence_direct = contrasts[
        "dual_minus_evidence_shuffle_omitted_direct_correctness"
    ]
    query_direct = contrasts["dual_minus_query_shuffle_omitted_direct_correctness"]
    p_off_probability = probability["global_P_off_dual_access"]["summary"]["omitted"]
    p_off_rescue = contrasts["P_off_dual_minus_shared_omitted_exact_probability"]
    p_off_nonlearned = behavior["global_P_off_dual_access"]["participant_bootstrap"][
        "nonlearned_accuracy"
    ]
    p_off_remote = contrasts["P_off_all_remote_minus_quarter_shared_all_remote"]
    interpretable = bool(integrity["all_passed"] and global_branch)
    criteria = {
        "omitted_direct_fidelity_rescue": (
            omitted["bootstrap"]["lower"] > 0.0
            and omitted_direct["bootstrap"]["lower"] > 0.0
        ),
        "retained_fidelity_preservation": (
            retained["bootstrap"]["lower"] >= retained_margin
            and integrity["retained_own_write_max_abs_error"] <= 1e-7
        ),
        "evidence_and_query_specificity": (
            evidence_probability["bootstrap"]["lower"] > 0.0
            and evidence_direct["bootstrap"]["lower"] > 0.0
            and query_direct["bootstrap"]["lower"] > 0.0
        ),
        "local_only_nontransitive_access": (
            p_off_probability["bootstrap"]["lower"] > 0.50
            and p_off_rescue["bootstrap"]["lower"] > 0.0
            and p_off_nonlearned["bootstrap"]["upper"] <= 0.55
            and p_off_remote["bootstrap"]["upper"] < 0.0
            and global_branch
        ),
    }
    flags = {name: bool(interpretable and value) for name, value in criteria.items()}
    return {
        "seed": seed,
        "interpretable": interpretable,
        "all_four_primary_links_pass": all(flags.values()),
        "flags": flags,
        "primary_effects": contrasts,
        "P_off_dual_omitted_exact_probability": p_off_probability,
        "P_off_dual_nonlearned_accuracy": p_off_nonlearned,
        "registered_rules": specification["primary_links"],
    }


def cross_seed_decision(specification: dict, seeds: dict[str, dict]) -> dict:
    mandatory = tuple(
        int(seed)
        for seed in specification["development_seed_contract"]["mandatory_frozen_seeds"]
    )
    links = {}
    for name in specification["primary_links"]:
        values = [seeds[str(seed)]["decision"]["flags"][name] for seed in mandatory]
        status = (
            "replicated"
            if all(values)
            else "heterogeneous_or_unresolved"
            if any(values)
            else "not_replicated"
        )
        links[name] = {
            "status": status,
            "seed_passes": {
                str(seed): bool(value)
                for seed, value in zip(mandatory, values, strict=True)
            },
        }
    interpretable = all(
        seeds[str(seed)]["decision"]["interpretable"] for seed in mandatory
    )
    all_pass = interpretable and all(
        row["status"] == "replicated" for row in links.values()
    )
    if not interpretable:
        outcome = "competence_or_integrity_failure"
    elif all_pass:
        outcome = "all_links_pass"
    elif any(row["status"] == "heterogeneous_or_unresolved" for row in links.values()):
        outcome = "heterogeneous_or_unresolved"
    elif any(
        seeds[str(seed)]["decision"]["flags"]["omitted_direct_fidelity_rescue"]
        and not seeds[str(seed)]["decision"]["flags"]["evidence_and_query_specificity"]
        for seed in mandatory
    ):
        outcome = "valid_omitted_rescue_without_specificity"
    elif any(
        not seeds[str(seed)]["decision"]["flags"]["retained_fidelity_preservation"]
        for seed in mandatory
    ):
        outcome = "valid_rescue_with_retained_harm"
    elif any(
        not seeds[str(seed)]["decision"]["flags"]["local_only_nontransitive_access"]
        for seed in mandatory
    ):
        outcome = "valid_local_only_globalization"
    else:
        outcome = "valid_failure"
    return {
        "outcome": outcome,
        "all_four_links_pass": all_pass,
        "all_seeds_interpretable": interpretable,
        "links": links,
        "network_population_inference": "not_performed",
        "registered_rules": specification["cross_seed_decision"],
    }


def evaluate_seed(
    specification: dict,
    seed: int,
    frozen_replication: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    evaluation = specification["liu_evaluation"]
    paths = seed_paths(V2_3_OUTPUT_ROOT, seed)
    gain_artifact = load_json(paths["gain"])
    backbone, model_config, checkpoint = load_frozen_retro_checkpoint(
        paths["checkpoint"], int(evaluation["subjects"])
    )
    before = tensor_hashes(backbone)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    local = create_local_trace(
        seed_specification(load_json(V2_3_SPECIFICATION_PATH), seed), model_config.cs
    )
    with torch.no_grad():
        local.raw_gain.fill_(float(gain_artifact["raw_lambda_L"]))
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
    relations = tuple(protocol.support_pairs_higher_lower)
    retained = retained_relation_mask(evaluator, relations)
    counts = _bootstrap_counts(specification, seed)
    interval = float(evaluation["bootstrap_interval"])
    geometry = build_complete_graph_geometry(protocol)
    schedules = tuple(ordered_pairs(protocol.n_items) for _ in range(model_config.bs))
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    loo_fast_weights = build_fast_weight_loo(evaluator, relations)
    route_maps = blockwise_derangements(
        model_config.bs,
        protocol.support_blocks,
        len(relations),
        int(specification["control_contract"]["evidence_shuffle_seed"]),
    )
    trace_builders = {
        "shared": lambda zero=frozenset(): build_access_trace(
            evaluator, local, dual_access=False, zero_relations=zero
        ),
        "dual": lambda zero=frozenset(): build_access_trace(
            evaluator, local, dual_access=True, zero_relations=zero
        ),
        "evidence_shuffle": lambda zero=frozenset(): build_access_trace(
            evaluator,
            local,
            dual_access=True,
            route_maps=route_maps,
            zero_relations=zero,
        ),
    }
    intact_traces = {name: builder() for name, builder in trace_builders.items()}
    loo_traces = {
        name: [builder(frozenset((relation,))) for relation in relations]
        for name, builder in trace_builders.items()
    }
    query_shuffle = shuffled_pair_indices(
        model_config.bs,
        protocol.n_items,
        int(specification["control_contract"]["query_shuffle_seed"]),
    )
    condition_bundles = {}
    condition_loo_bundles = {}
    fields = {}
    probability = {}
    behavior = {}
    pair_probabilities = {}
    for condition in CONDITIONS:
        trace_name, local_off, global_off, query_shuffled = _condition_configuration(
            condition
        )
        shuffled = query_shuffle if query_shuffled else None
        condition_bundles[condition] = readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            intact_traces[trace_name].state,
            schedules,
            local_off=local_off,
            global_off=global_off,
            shuffled_indices=shuffled,
        )
        condition_loo_bundles[condition] = [
            readout_relational_query_bundle(
                evaluator,
                local,
                loo_fast_weights[index],
                loo_traces[trace_name][index].state,
                schedules,
                local_off=local_off,
                global_off=global_off,
                shuffled_indices=shuffled,
            )
            for index in range(len(relations))
        ]
        intact_field = margin_fields(condition_bundles[condition], protocol.n_items)
        loo_field = np.asarray(
            [
                margin_fields(bundle, protocol.n_items)
                for bundle in condition_loo_bundles[condition]
            ]
        )
        fields[condition] = field_metrics(
            intact_field,
            loo_field,
            relations,
            retained,
            geometry,
            counts,
            interval,
        )
        learned = learned_probabilities(
            evaluator, condition_bundles[condition], float(evaluation["temperature"])
        )
        probability[condition] = _probability_metrics(
            learned, retained, counts, interval
        )
        behavior[condition] = analyze_sampled_query_policy(
            protocol,
            bundle_logits(condition_bundles[condition], schedules),
            seed=int(evaluation["choice_seed"]),
            temperature=float(evaluation["temperature"]),
        )
        behavior[condition]["participant_bootstrap"] = behavior_summaries(
            behavior[condition], counts, interval
        )
        pair_probabilities[condition] = pair_correct_probabilities(
            evaluator,
            condition_bundles[condition],
            float(evaluation["temperature"]),
        )

    def prob_raw(condition, group):
        return probability[condition]["raw_subject_level"][group]

    def field_raw(condition, group, metric):
        return fields[condition]["raw_subject_level"][group][metric]

    contrasts = {
        "dual_minus_shared_omitted_exact_probability": _paired(
            prob_raw("dual_access_matched", "omitted"),
            prob_raw("shared_access_v2_3", "omitted"),
            counts,
            interval,
        ),
        "dual_minus_shared_retained_exact_probability": _paired(
            prob_raw("dual_access_matched", "retained"),
            prob_raw("shared_access_v2_3", "retained"),
            counts,
            interval,
        ),
        "dual_minus_shared_omitted_direct_correctness": _paired(
            field_raw("dual_access_matched", "omitted", "direct_correctness"),
            field_raw("shared_access_v2_3", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_shared_retained_direct_correctness": _paired(
            field_raw("dual_access_matched", "retained", "direct_correctness"),
            field_raw("shared_access_v2_3", "retained", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_evidence_shuffle_omitted_exact_probability": _paired(
            prob_raw("dual_access_matched", "omitted"),
            prob_raw("dual_access_evidence_shuffle", "omitted"),
            counts,
            interval,
        ),
        "dual_minus_evidence_shuffle_omitted_direct_correctness": _paired(
            field_raw("dual_access_matched", "omitted", "direct_correctness"),
            field_raw("dual_access_evidence_shuffle", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_query_shuffle_omitted_direct_correctness": _paired(
            field_raw("dual_access_matched", "omitted", "direct_correctness"),
            field_raw("dual_access_query_shuffle", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "P_off_dual_minus_shared_omitted_exact_probability": _paired(
            prob_raw("global_P_off_dual_access", "omitted"),
            prob_raw("global_P_off_shared_access", "omitted"),
            counts,
            interval,
        ),
        "P_off_all_remote_minus_quarter_shared_all_remote": summarize_subjects(
            np.asarray(field_raw("global_P_off_dual_access", "all", "remote_absolute"))
            - 0.25
            * np.asarray(field_raw("shared_access_v2_3", "all", "remote_absolute")),
            counts,
            interval=interval,
        ),
    }
    legacy_trace = build_local_trace(evaluator, local)
    shared_identity = _shared_identity(
        seed,
        intact_traces["shared"].state,
        legacy_trace,
        probability,
        fields,
        behavior,
        frozen_replication["seeds"][str(seed)],
    )
    natural = intact_traces["dual"].natural_scalars
    shared_natural = intact_traces["shared"].natural_scalars
    retained_scalar_error = []
    omitted_scalar_error = []
    for subject, schedule in enumerate(evaluator.support_schedules):
        for trial_index, trial in enumerate(schedule):
            z_value = evaluator._encoding_reliability(subject, trial_index)
            if z_value > 0.0:
                retained_scalar_error.append(
                    abs(
                        natural[subject, trial_index]
                        - shared_natural[subject, trial_index]
                    )
                )
            else:
                probability_value = _relation_probability(
                    evaluator, subject, trial.higher_item, trial.lower_item
                )
                expected = trial.signed_magnitude * probability_value
                omitted_scalar_error.append(
                    abs(natural[subject, trial_index] - expected)
                )
    routed = intact_traces["evidence_shuffle"].applied_scalars
    shuffle_multiset_error = 0.0
    for subject in range(model_config.bs):
        for block in range(protocol.support_blocks):
            start = block * len(relations)
            stop = start + len(relations)
            shuffle_multiset_error = max(
                shuffle_multiset_error,
                float(
                    np.max(
                        abs(
                            np.sort(natural[subject, start:stop])
                            - np.sort(routed[subject, start:stop])
                        )
                    )
                ),
            )
    global_conditions = (
        "local_off_v1",
        "shared_access_v2_3",
        "dual_access_matched",
        "dual_access_evidence_shuffle",
        "dual_access_query_shuffle",
    )
    reference_global = condition_bundles["shared_access_v2_3"]["global_logits"]
    global_logit_error = max(
        float(
            np.max(
                abs(condition_bundles[condition]["global_logits"] - reference_global)
            )
        )
        for condition in global_conditions
    )
    p_off_global_error = float(
        np.max(
            abs(
                condition_bundles["global_P_off_dual_access"]["global_logits"]
                - condition_bundles["global_P_off_shared_access"]["global_logits"]
            )
        )
    )
    local_off_readout = evaluator.readout_logits(intact_fast_weights, schedules)
    local_off_bundle = bundle_logits(condition_bundles["local_off_v1"], schedules)
    local_off_error = max(
        abs(local_off_readout[subject][pair] - local_off_bundle[subject][pair])
        for subject in range(model_config.bs)
        for pair in ordered_pairs(protocol.n_items)
    )
    common_mode_error = max(
        float(
            np.max(
                abs(
                    condition_bundles[condition]["logits"]
                    - condition_bundles[condition]["global_logits"]
                    - condition_bundles[condition]["applied_local_margins"]
                )
            )
        )
        for condition in CONDITIONS
    )
    presentation_invariance = measure_presentation_invariance(evaluator, local, natural)
    slope_decomposition = _exact_slope_decomposition(
        evaluator, pair_probabilities, retained, counts, interval
    )
    slope_identity_error = max(
        condition["additive_identity_max_abs_error"]
        for condition in slope_decomposition["conditions"].values()
    )
    after = tensor_hashes(backbone)
    frozen_seed = frozen_replication["seeds"][str(seed)]
    global_branch = bool(
        frozen_seed["original_v1_qualification"]["passed"]
        and frozen_seed["legacy_v2_3_decision"]["flags"]["global_branch_preservation"]
    )
    integrity_values = {
        "shared_local_state_max_abs_error": shared_identity[
            "local_state_max_abs_error"
        ],
        "shared_exact_probability_max_abs_error": shared_identity[
            "exact_probability_max_abs_error"
        ],
        "shared_direct_causal_max_abs_error": shared_identity[
            "direct_causal_max_abs_error"
        ],
        "shared_sampled_behavior_max_abs_error": shared_identity[
            "sampled_behavior_max_abs_error"
        ],
        "global_condition_logit_max_abs_error": global_logit_error,
        "P_off_global_logit_max_abs_error": p_off_global_error,
        "local_off_v1_logit_max_abs_error": local_off_error,
        "local_margin_identity_max_abs_error": common_mode_error,
        "retained_own_write_max_abs_error": float(
            max(retained_scalar_error, default=0.0)
        ),
        "omitted_weak_scalar_max_abs_error": float(
            max(omitted_scalar_error, default=0.0)
        ),
        "evidence_shuffle_multiset_max_abs_error": shuffle_multiset_error,
        "all_evidence_maps_are_derangements": bool(
            np.all(route_maps != np.arange(len(relations))[None, None])
        ),
        "all_query_maps_are_derangements": bool(
            np.all(
                query_shuffle[:, 0::2] // 2
                != np.arange(query_shuffle.shape[1] // 2)[None]
            )
        ),
        **presentation_invariance,
        "slope_additive_identity_max_abs_error": slope_identity_error,
        "backbone_tensor_hashes_unchanged": before == after,
        "same_frozen_gain_all_conditions": True,
        "artifact_validation_passed": bool(artifact_validation["passed"]),
    }
    integrity_values["all_passed"] = bool(
        all(
            (
                integrity_values["shared_local_state_max_abs_error"] <= 1e-7,
                integrity_values["shared_exact_probability_max_abs_error"] <= 1e-12,
                integrity_values["shared_direct_causal_max_abs_error"] <= 1e-12,
                integrity_values["shared_sampled_behavior_max_abs_error"] <= 1e-12,
                integrity_values["global_condition_logit_max_abs_error"] <= 1e-7,
                integrity_values["P_off_global_logit_max_abs_error"] <= 1e-7,
                integrity_values["local_off_v1_logit_max_abs_error"] <= 1e-6,
                integrity_values["local_margin_identity_max_abs_error"] <= 1e-6,
                integrity_values["retained_own_write_max_abs_error"] <= 1e-7,
                integrity_values["omitted_weak_scalar_max_abs_error"] <= 1e-7,
                integrity_values["evidence_shuffle_multiset_max_abs_error"] <= 1e-7,
                integrity_values["all_evidence_maps_are_derangements"],
                integrity_values["all_query_maps_are_derangements"],
                integrity_values["support_write_reversal_max_abs_error"] <= 1e-7,
                integrity_values["query_key_reversal_max_abs_error"] <= 1e-7,
                integrity_values["slope_additive_identity_max_abs_error"] <= 1e-12,
                integrity_values["backbone_tensor_hashes_unchanged"],
                integrity_values["artifact_validation_passed"],
            )
        )
    )
    decision = within_seed_decision(
        specification,
        seed,
        contrasts,
        probability,
        fields,
        behavior,
        integrity_values,
        global_branch,
    )
    return {
        "seed": seed,
        "checkpoint": {"path": str(paths["checkpoint"]), "sha256": checkpoint.sha256},
        "gain": {
            "path": str(paths["gain"]),
            "sha256": file_sha256(paths["gain"]),
            "lambda_L": gain_artifact["lambda_L"],
        },
        "runtime": runtime,
        "integrity": integrity_values,
        "global_branch": {
            "passed": global_branch,
            "qualification": frozen_seed["original_v1_qualification"],
            "query_binding": frozen_seed["query_binding"],
            "terminal_projection": frozen_seed["terminal_projection"],
            "legacy_global_flags": frozen_seed["legacy_v2_3_decision"]["flags"],
        },
        "probability": probability,
        "causal_fields": fields,
        "behavior": behavior,
        "exact_probability_slope_decomposition": slope_decomposition,
        "contrasts": contrasts,
        "decision": decision,
    }


def evaluate_pilot(
    specification: dict,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    frozen = load_json(V2_3_RESULT_PATH)
    seeds = {}
    for seed in specification["development_seed_contract"]["mandatory_frozen_seeds"]:
        seeds[str(seed)] = evaluate_seed(
            specification,
            int(seed),
            frozen,
            artifact_validation,
            runtime,
        )
    return {
        "schema_version": 1,
        "pilot_id": specification["pilot_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "seeds": seeds,
        "decision": cross_seed_decision(specification, seeds),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered dual-evidence-access v2.4 pilot."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    runtime = configure_formal_cuda_runtime()
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    specification = load_json(parsed.specification)
    v2_3_specification = load_json(V2_3_SPECIFICATION_PATH)
    artifact_validation = validate_artifacts(
        v2_3_specification,
        V2_3_SPECIFICATION_PATH,
        V2_3_IMPLEMENTATION_LOCK_PATH,
        V2_3_ARTIFACT_LOCK_PATH,
        V2_3_OUTPUT_ROOT,
    )
    result = evaluate_pilot(
        specification, source_validation, artifact_validation, runtime
    )
    write_json_exclusive(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
