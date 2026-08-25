"""Read-only retained/omitted attribution of the frozen v2.3 local trace."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .assembly_trajectory import summarize_difference, summarize_subjects
from .behavioral import analyze_sampled_query_policy
from .confirmation import file_sha256
from .conjunctive_local_trace_pilot import (
    _new_local_trace,
    _ordered_pairs,
    _resolve_registered,
    _retained_mask,
    build_local_trace,
    bundle_logits,
    configure_runtime,
    load_json,
    query_bundle,
    write_json,
)
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
)
from .ranking_protocol import load_ranking_protocol
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/local_behavior_attribution_v2_3.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/local_behavior_attribution_v2_3.lock.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/local_behavior_attribution_v2_3.json")


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification_path = specification_path.resolve()
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    registrations = {
        **specification["registered_sources"],
        "frozen_backbone": specification["frozen_artifacts"]["backbone"],
        "frozen_gain": specification["frozen_artifacts"]["gain"],
        "analysis_specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["analysis_specification_sha256"],
        },
        **lock["implementation_sources"],
    }
    checks = []
    for name, registration in registrations.items():
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
        raise RuntimeError(f"local-behavior attribution source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def exact_probability(
    correct_signed_margin: np.ndarray, temperature: float
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(correct_signed_margin, dtype=np.float64) / temperature
    scaled = np.clip(scaled, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _masked_subject_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if rows.shape != selected.shape:
        raise ValueError("values and mask must have the same shape")
    counts = np.sum(selected, axis=(1, 2))
    totals = np.sum(np.where(selected, rows, 0.0), axis=(1, 2))
    return np.divide(
        totals,
        counts,
        out=np.full(rows.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    point_denominator = float(np.sum(denominator))
    point = (
        None
        if point_denominator <= 0.0
        else float(np.sum(numerator) / point_denominator)
    )
    boot_numerator = counts @ numerator
    boot_denominator = counts @ denominator
    samples = np.divide(
        boot_numerator,
        boot_denominator,
        out=np.full_like(boot_numerator, np.nan),
        where=boot_denominator > 0.0,
    )
    samples = samples[np.isfinite(samples)]
    tail = (1.0 - interval) / 2.0
    return {
        "point": point,
        "bootstrap": {
            "mean": float(np.mean(samples)),
            "lower": float(np.quantile(samples, tail)),
            "upper": float(np.quantile(samples, 1.0 - tail)),
        },
    }


def _group_summary(
    values: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    subject_values = _masked_subject_mean(values, mask)
    return {
        "summary": summarize_subjects(subject_values, counts, interval=interval),
        "raw_subject_level": _json_values(subject_values),
        "cells": int(np.sum(mask)),
        "cell_mean": float(np.mean(values[mask])) if np.any(mask) else None,
    }


def _json_values(values: np.ndarray) -> list:
    array = np.asarray(values)

    def convert(value):
        value = float(value)
        return None if not np.isfinite(value) else value

    if array.ndim == 0:
        return convert(array)
    if array.ndim == 1:
        return [convert(value) for value in array]
    return [_json_values(row) for row in array]


def _self_traces(evaluator, local_module, relations) -> torch.Tensor:
    relation_set = frozenset(relations)
    return torch.stack(
        [
            build_local_trace(
                evaluator,
                local_module,
                zero_relations=relation_set.difference((relation,)),
            )
            for relation in relations
        ]
    )


def _self_local_margins(evaluator, local_module, self_traces, relations) -> np.ndarray:
    subjects = evaluator.config.bs
    values = np.empty((subjects, len(relations), 2), dtype=np.float64)
    with torch.no_grad():
        for relation_index, relation in enumerate(relations):
            for orientation, pair in enumerate((relation, relation[::-1])):
                left = np.full(subjects, pair[0], dtype=np.int64)
                right = np.full(subjects, pair[1], dtype=np.int64)
                step0 = evaluator._step_inputs(
                    left,
                    right,
                    np.zeros(subjects, dtype=np.float32),
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                _raw, correction = local_module.read(
                    self_traces[relation_index],
                    step0[:, : 2 * evaluator.config.cs],
                )
                sign = 1.0 if orientation == 0 else -1.0
                values[:, relation_index, orientation] = (
                    sign * correction[:, 0].cpu().numpy()
                )
    return values


def learned_cells(
    evaluator,
    global_bundle: dict,
    dual_bundle: dict,
    p_off_bundle: dict,
    self_local: np.ndarray,
    retained_relation_subject: np.ndarray,
    temperature: float,
) -> dict[str, np.ndarray]:
    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    pairs = _ordered_pairs(evaluator.protocol.n_items)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    subjects = evaluator.config.bs
    shape = (subjects, len(relations), 2)
    global_margin = np.empty(shape, dtype=np.float64)
    full_local = np.empty(shape, dtype=np.float64)
    p_off_margin = np.empty(shape, dtype=np.float64)
    for relation_index, relation in enumerate(relations):
        for orientation, pair in enumerate((relation, relation[::-1])):
            index = pair_index[pair]
            sign = 1.0 if orientation == 0 else -1.0
            global_margin[:, relation_index, orientation] = (
                sign * global_bundle["global_logits"][:, index]
            )
            full_local[:, relation_index, orientation] = (
                sign * dual_bundle["applied_local_margins"][:, index]
            )
            p_off_margin[:, relation_index, orientation] = (
                sign * p_off_bundle["logits"][:, index]
            )
    dual_margin = global_margin + full_local
    cross_local = full_local - self_local
    retained = np.broadcast_to(retained_relation_subject.T[:, :, None], shape).copy()
    return {
        "retained": retained,
        "global_margin": global_margin,
        "self_local_margin": self_local,
        "cross_local_margin": cross_local,
        "full_local_margin": full_local,
        "dual_margin": dual_margin,
        "p_off_local_intact_margin": p_off_margin,
        "p_global": exact_probability(global_margin, temperature),
        "p_dual": exact_probability(dual_margin, temperature),
        "p_p_off_local_intact": exact_probability(p_off_margin, temperature),
        "p_pure_local": exact_probability(full_local, temperature),
    }


def error_mass_attribution(
    cells: dict[str, np.ndarray], counts: np.ndarray, interval: float
) -> dict:
    retained = cells["retained"]
    error = 1.0 - cells["p_global"]
    retained_error = np.sum(np.where(retained, error, 0.0), axis=(1, 2))
    omitted_error = np.sum(np.where(~retained, error, 0.0), axis=(1, 2))
    omitted_fraction = _ratio_summary(
        omitted_error, retained_error + omitted_error, counts, interval
    )
    return {
        "retained_error_mass": float(np.sum(retained_error)),
        "omitted_error_mass": float(np.sum(omitted_error)),
        "omitted_error_mass_fraction": omitted_fraction,
        "retained_cells": int(np.sum(retained)),
        "omitted_cells": int(np.sum(~retained)),
        "mean_error_per_retained_cell": float(np.mean(error[retained])),
        "mean_error_per_omitted_cell": float(np.mean(error[~retained])),
        "raw_subject_retained_error_mass": _json_values(retained_error),
        "raw_subject_omitted_error_mass": _json_values(omitted_error),
    }


def boundary_and_probability_attribution(
    cells: dict[str, np.ndarray], counts: np.ndarray, interval: float
) -> dict:
    retained = cells["retained"]
    global_margin = cells["global_margin"]
    dual_margin = cells["dual_margin"]
    self_margin = global_margin + cells["self_local_margin"]
    baseline_wrong = retained & (global_margin <= 0.0)
    baseline_correct = retained & (global_margin > 0.0)
    rescued = baseline_wrong & (dual_margin > 0.0)
    self_rescued = baseline_wrong & (self_margin > 0.0)
    harmed = baseline_correct & (dual_margin <= 0.0)

    def subject_rate(event, denominator):
        event_count = np.sum(event, axis=(1, 2)).astype(np.float64)
        denominator_count = np.sum(denominator, axis=(1, 2)).astype(np.float64)
        return np.divide(
            event_count,
            denominator_count,
            out=np.full_like(event_count, np.nan),
            where=denominator_count > 0.0,
        )

    rescue_rate = subject_rate(rescued, baseline_wrong)
    self_rescue_rate = subject_rate(self_rescued, baseline_wrong)
    harm_rate = subject_rate(harmed, baseline_correct)
    delta_p = cells["p_dual"] - cells["p_global"]
    output = {
        "retained_baseline_wrong_cells": int(np.sum(baseline_wrong)),
        "retained_rescued_cells": int(np.sum(rescued)),
        "retained_self_only_rescued_cells": int(np.sum(self_rescued)),
        "retained_harmed_cells": int(np.sum(harmed)),
        "rescue_rate": summarize_subjects(rescue_rate, counts, interval=interval),
        "self_only_rescue_rate": summarize_subjects(
            self_rescue_rate, counts, interval=interval
        ),
        "harm_rate": summarize_subjects(harm_rate, counts, interval=interval),
        "delta_probability": {
            "retained": _group_summary(delta_p, retained, counts, interval),
            "omitted": _group_summary(delta_p, ~retained, counts, interval),
        },
    }
    for name, group in (("retained", retained), ("omitted", ~retained)):
        baseline_error = np.sum(
            np.where(group, 1.0 - cells["p_global"], 0.0), axis=(1, 2)
        )
        removed = np.sum(np.where(group, delta_p, 0.0), axis=(1, 2))
        output["delta_probability"][name]["fraction_error_mass_removed"] = (
            _ratio_summary(removed, baseline_error, counts, interval)
        )
    bins = (
        ("below_0_5", 0.0, 0.5),
        ("0_5_to_0_9", 0.5, 0.9),
        ("0_9_to_0_99", 0.9, 0.99),
        ("0_99_to_1", 0.99, 1.0 + 1e-12),
    )
    output["retained_saturation_bins"] = {}
    for name, lower, upper in bins:
        selected = retained & (cells["p_global"] >= lower) & (cells["p_global"] < upper)
        output["retained_saturation_bins"][name] = {
            "cells": int(np.sum(selected)),
            "mean_p_global": (
                float(np.mean(cells["p_global"][selected]))
                if np.any(selected)
                else None
            ),
            "mean_delta_p": (
                float(np.mean(delta_p[selected])) if np.any(selected) else None
            ),
        }
    return output


def self_cross_attribution(
    cells: dict[str, np.ndarray], counts: np.ndarray, interval: float
) -> dict:
    retained = cells["retained"]
    self_margin = cells["self_local_margin"]
    cross_margin = cells["cross_local_margin"]
    full_margin = cells["full_local_margin"]
    retained_abs_self = _masked_subject_mean(np.abs(self_margin), retained)
    retained_abs_cross = _masked_subject_mean(np.abs(cross_margin), retained)
    retained_signed_self = _masked_subject_mean(self_margin, retained)
    retained_signed_cross = _masked_subject_mean(cross_margin, retained)
    ratio = float(np.mean(retained_abs_cross) / np.mean(retained_abs_self))
    self_positive = _masked_subject_mean((self_margin > 0.0).astype(float), retained)
    return {
        "retained_signed_self": summarize_subjects(
            retained_signed_self, counts, interval=interval
        ),
        "retained_signed_cross": summarize_subjects(
            retained_signed_cross, counts, interval=interval
        ),
        "retained_absolute_self": summarize_subjects(
            retained_abs_self, counts, interval=interval
        ),
        "retained_absolute_cross": summarize_subjects(
            retained_abs_cross, counts, interval=interval
        ),
        "retained_absolute_cross_to_self_ratio": ratio,
        "retained_self_positive_fraction": summarize_subjects(
            self_positive, counts, interval=interval
        ),
        "self_plus_cross_identity_max_abs_error": float(
            np.max(np.abs(full_margin - self_margin - cross_margin))
        ),
        "stable_omitted_self_max_abs": float(np.max(np.abs(self_margin[~retained]))),
        "raw_subject_retained_signed_self": _json_values(retained_signed_self),
        "raw_subject_retained_signed_cross": _json_values(retained_signed_cross),
        "raw_subject_retained_absolute_self": _json_values(retained_abs_self),
        "raw_subject_retained_absolute_cross": _json_values(retained_abs_cross),
    }


def local_only_attribution(
    cells: dict[str, np.ndarray], counts: np.ndarray, interval: float
) -> dict:
    retained = cells["retained"]
    output = {}
    for name, probability, margin in (
        (
            "P_off_local_intact",
            cells["p_p_off_local_intact"],
            cells["p_off_local_intact_margin"],
        ),
        ("pure_local", cells["p_pure_local"], cells["full_local_margin"]),
    ):
        retained_probability = _masked_subject_mean(probability, retained)
        omitted_probability = _masked_subject_mean(probability, ~retained)
        retained_hard = _masked_subject_mean((margin > 0.0).astype(float), retained)
        omitted_hard = _masked_subject_mean((margin > 0.0).astype(float), ~retained)
        output[name] = {
            "retained_exact_probability": summarize_subjects(
                retained_probability, counts, interval=interval
            ),
            "omitted_exact_probability": summarize_subjects(
                omitted_probability, counts, interval=interval
            ),
            "retained_minus_omitted_exact_probability": summarize_difference(
                retained_probability, omitted_probability, counts, interval=interval
            ),
            "retained_hard_accuracy": summarize_subjects(
                retained_hard, counts, interval=interval
            ),
            "omitted_hard_accuracy": summarize_subjects(
                omitted_hard, counts, interval=interval
            ),
            "raw_subject_retained_exact_probability": _json_values(
                retained_probability
            ),
            "raw_subject_omitted_exact_probability": _json_values(omitted_probability),
        }
    return output


def _pair_correct_probabilities(
    evaluator, bundle: dict, temperature: float
) -> np.ndarray:
    canonical = tuple(combinations(range(evaluator.protocol.n_items), 2))
    ordered = _ordered_pairs(evaluator.protocol.n_items)
    pair_index = {pair: index for index, pair in enumerate(ordered)}
    rank = {
        item: position
        for position, item in enumerate(evaluator.protocol.true_order_high_to_low)
    }
    values = np.empty((evaluator.config.bs, len(canonical)), dtype=np.float64)
    for edge, pair in enumerate(canonical):
        first_higher = rank[pair[0]] < rank[pair[1]]
        forward_sign = 1.0 if first_higher else -1.0
        reverse_sign = -forward_sign
        forward = forward_sign * bundle["logits"][:, pair_index[pair]]
        reverse = reverse_sign * bundle["logits"][:, pair_index[pair[::-1]]]
        values[:, edge] = 0.5 * (
            exact_probability(forward, temperature)
            + exact_probability(reverse, temperature)
        )
    return values


def slope_decomposition(
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
    nonlearned = ~(learned_retained | learned_omitted)
    masks = {
        "learned_retained": learned_retained,
        "learned_omitted": learned_omitted,
        "nonlearned": nonlearned,
    }
    output = {"denominator": denominator, "conditions": {}}
    raw_contributions = {}
    for condition, probabilities in condition_probabilities.items():
        centered_probability = probabilities - np.mean(
            probabilities, axis=1, keepdims=True
        )
        numerator_cells = centered_distance[None] * centered_probability
        contributions = {
            name: np.sum(np.where(mask, numerator_cells, 0.0), axis=1) / denominator
            for name, mask in masks.items()
        }
        total = np.sum(numerator_cells, axis=1) / denominator
        identity_error = float(
            np.max(np.abs(total - sum(contributions.values(), np.zeros_like(total))))
        )
        raw_contributions[condition] = contributions
        output["conditions"][condition] = {
            "total_exact_probability_slope": summarize_subjects(
                total, counts, interval=interval
            ),
            "group_contributions": {
                name: summarize_subjects(values, counts, interval=interval)
                for name, values in contributions.items()
            },
            "additive_identity_max_abs_error": identity_error,
            "raw_subject_total": _json_values(total),
            "raw_subject_group_contributions": {
                name: _json_values(values) for name, values in contributions.items()
            },
        }
    output["dual_minus_original_group_contributions"] = {
        name: summarize_difference(
            raw_contributions["dual_intact"][name],
            raw_contributions["original_v1_local_off"][name],
            counts,
            interval=interval,
        )
        for name in masks
    }
    original_means = {
        name: float(np.mean(values))
        for name, values in raw_contributions["original_v1_local_off"].items()
    }
    output["largest_positive_original_contributor"] = max(
        original_means, key=original_means.get
    )
    return output


def sampled_endpoint_reproduction(
    specification: dict,
    evaluator,
    bundles: dict,
    schedules,
) -> dict:
    execution = specification["execution_contract"]
    frozen = load_json(
        _resolve_registered(specification["registered_sources"]["v2_3_result"]["path"])
    )
    output = {}
    errors = []
    for condition in ("original_v1_local_off", "dual_intact"):
        current = analyze_sampled_query_policy(
            evaluator.protocol,
            bundle_logits(bundles[condition], schedules),
            seed=int(execution["choice_seed"]),
            temperature=float(execution["choice_temperature"]),
        )
        metrics = {
            "learned_accuracy": current["summary"]["learned_accuracy"],
            "nonlearned_accuracy": current["summary"]["nonlearned_accuracy"],
            "overall_accuracy": current["summary"]["overall_accuracy"],
            "symbolic_distance_slope": current["summary"]["symbolic_distance_slope"][
                "mean"
            ],
        }
        expected = {
            "learned_accuracy": frozen["behavior"][condition]["summary"][
                "learned_accuracy"
            ],
            "nonlearned_accuracy": frozen["behavior"][condition]["summary"][
                "nonlearned_accuracy"
            ],
            "overall_accuracy": frozen["behavior"][condition]["summary"][
                "overall_accuracy"
            ],
            "symbolic_distance_slope": frozen["behavior"][condition]["summary"][
                "symbolic_distance_slope"
            ]["mean"],
        }
        errors.extend(abs(metrics[name] - expected[name]) for name in metrics)
        output[condition] = {"observed": metrics, "expected": expected}
    output["max_abs_error"] = float(max(errors, default=0.0))
    output["frozen_sampled_learned_accuracy_contrast"] = frozen["decision"][
        "behavior_contrasts"
    ]["dual_minus_original_learned_accuracy"]
    return output


def decision_summary(
    specification: dict,
    error_mass: dict,
    probability: dict,
    self_cross: dict,
    local_only: dict,
    slope: dict,
    sampled: dict,
) -> dict:
    omission_dominant = (
        error_mass["omitted_error_mass_fraction"]["bootstrap"]["lower"] > 0.50
    )
    p_off = local_only["P_off_local_intact"]
    retained_local_sufficient = (
        p_off["retained_exact_probability"]["bootstrap"]["lower"] > 0.65
        and p_off["omitted_exact_probability"]["bootstrap"]["upper"] <= 0.55
        and p_off["retained_minus_omitted_exact_probability"]["bootstrap"]["lower"]
        > 0.10
    )
    retained_error_fraction = 1.0 - float(
        error_mass["omitted_error_mass_fraction"]["point"]
    )
    retained_value_limited = (
        not retained_local_sufficient
        and self_cross["retained_signed_self"]["bootstrap"]["lower"] > 0.0
        and retained_error_fraction >= 0.50
    )
    exact_rescue = (
        probability["delta_probability"]["retained"]["summary"]["bootstrap"]["lower"]
        > 0.0
    )
    sampled_interval = sampled["frozen_sampled_learned_accuracy_contrast"]["bootstrap"]
    sampled_insensitive = sampled_interval["lower"] <= 0.0 <= sampled_interval["upper"]
    exact_with_sampled_insensitivity = exact_rescue and sampled_insensitive
    address_interference = (
        self_cross["retained_absolute_cross_to_self_ratio"] >= 1.0 / 3.0
    )
    flags = {
        "omission_dominant": bool(omission_dominant),
        "retained_local_sufficient": bool(retained_local_sufficient),
        "retained_value_conversion_limited": bool(retained_value_limited),
        "exact_rescue_with_sampled_insensitivity": bool(
            exact_with_sampled_insensitivity
        ),
        "address_interference_material": bool(address_interference),
    }
    if omission_dominant and retained_local_sufficient:
        outcome = "dual_evidence_access"
    elif retained_value_limited:
        outcome = "shared_value_transform"
    elif exact_with_sampled_insensitivity:
        outcome = "confirmation_estimand_sensitivity"
    else:
        outcome = "mixed_or_unresolved"
    return {
        "outcome": outcome,
        "flags": flags,
        "registered_rules": specification["decision_rules"],
        "retained_error_mass_fraction": retained_error_fraction,
        "slope_source": slope["largest_positive_original_contributor"],
        "outcome_interpretation": specification["outcome_tree"][outcome],
    }


def run_attribution(
    specification: dict, source_validation: dict, runtime: dict
) -> dict:
    execution = specification["execution_contract"]
    backbone_path = _resolve_registered(
        specification["frozen_artifacts"]["backbone"]["path"]
    )
    gain_path = _resolve_registered(specification["frozen_artifacts"]["gain"]["path"])
    gain_artifact = load_json(gain_path)
    backbone, model_config, checkpoint_info = load_retro_checkpoint(
        backbone_path, int(execution["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    v2_specification = load_json(
        _resolve_registered(
            specification["registered_sources"]["v2_3_specification"]["path"]
        )
    )
    local_module = _new_local_trace(v2_specification, model_config.cs)
    with torch.no_grad():
        local_module.raw_gain.fill_(float(gain_artifact["raw_lambda_L"]))
    protocol = load_ranking_protocol(
        _resolve_registered(specification["registered_sources"]["liu_protocol"]["path"])
    )
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocol,
        cue_seed=int(execution["cue_seed"]),
        support_seed=int(execution["support_seed"]),
        cue_mode=str(execution["cue_mode"]),
        subject_encoding_mode=str(execution["subject_encoding_mode"]),
        subject_encoding_seed=int(execution["subject_encoding_seed"]),
    )
    pairs = _ordered_pairs(protocol.n_items)
    schedules = tuple(pairs for _ in range(model_config.bs))
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    full_trace = build_local_trace(evaluator, local_module)
    bundles = {
        condition: query_bundle(
            evaluator,
            local_module,
            fast_weights,
            full_trace,
            schedules,
            condition=condition,
            shuffle_seed=0,
        )
        for condition in (
            "original_v1_local_off",
            "dual_intact",
            "global_P_off_local_intact",
        )
    }
    relations = tuple(protocol.support_pairs_higher_lower)
    retained = _retained_mask(evaluator, relations)
    self_traces = _self_traces(evaluator, local_module, relations)
    self_local = _self_local_margins(evaluator, local_module, self_traces, relations)
    cells = learned_cells(
        evaluator,
        bundles["original_v1_local_off"],
        bundles["dual_intact"],
        bundles["global_P_off_local_intact"],
        self_local,
        retained,
        float(execution["choice_temperature"]),
    )
    counts = (
        np.random.default_rng(int(execution["bootstrap_seed"]))
        .multinomial(
            model_config.bs,
            np.full(model_config.bs, 1.0 / model_config.bs),
            size=int(execution["bootstrap_samples"]),
        )
        .astype(np.float64)
    )
    interval = float(execution["bootstrap_interval"])
    error_mass = error_mass_attribution(cells, counts, interval)
    probability = boundary_and_probability_attribution(cells, counts, interval)
    self_cross = self_cross_attribution(cells, counts, interval)
    local_only = local_only_attribution(cells, counts, interval)
    pair_probabilities = {
        condition: _pair_correct_probabilities(
            evaluator, bundle, float(execution["choice_temperature"])
        )
        for condition, bundle in bundles.items()
    }
    slope = slope_decomposition(
        evaluator, pair_probabilities, retained, counts, interval
    )
    sampled = sampled_endpoint_reproduction(
        specification, evaluator, bundles, schedules
    )
    integrity = {
        "backbone_sha256": checkpoint_sha256(backbone_path),
        "gain_sha256": file_sha256(gain_path),
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
        "sampled_endpoint_reproduction_max_abs_error": sampled["max_abs_error"],
        "slope_additive_identity_max_abs_error": max(
            row["additive_identity_max_abs_error"]
            for row in slope["conditions"].values()
        ),
    }
    if integrity["dual_margin_identity_max_abs_error"] > 1e-6:
        raise RuntimeError("dual margin identity failed")
    if integrity["self_plus_cross_identity_max_abs_error"] > 1e-6:
        raise RuntimeError("self/cross decomposition identity failed")
    if integrity["stable_omitted_self_max_abs"] > 1e-7:
        raise RuntimeError("stable-omitted self trace is nonzero")
    if integrity["sampled_endpoint_reproduction_max_abs_error"] > 1e-12:
        raise RuntimeError("frozen sampled endpoint reproduction failed")
    if integrity["slope_additive_identity_max_abs_error"] > 1e-12:
        raise RuntimeError("slope contribution identity failed")
    decision = decision_summary(
        specification,
        error_mass,
        probability,
        self_cross,
        local_only,
        slope,
        sampled,
    )
    return {
        "schema_version": 1,
        "analysis_id": specification["analysis_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "checkpoint": {
            "path": str(backbone_path),
            "sha256": checkpoint_info.sha256,
        },
        "gain_artifact": {
            "path": str(gain_path),
            "sha256": file_sha256(gain_path),
            "lambda_L": gain_artifact["lambda_L"],
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "integrity": integrity,
        "error_mass": error_mass,
        "boundary_and_probability": probability,
        "self_cross": self_cross,
        "local_only": local_only,
        "slope_decomposition": slope,
        "sampled_endpoint_reproduction": sampled,
        "decision": decision,
        "raw_learned_cells": {
            name: _json_values(value) for name, value in cells.items()
        },
        "raw_pair_exact_probabilities": {
            name: _json_values(value) for name, value in pair_probabilities.items()
        },
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered v2.3 local-behavior attribution."
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
    runtime = configure_runtime()
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    specification = load_json(parsed.specification)
    result = run_attribution(specification, source_validation, runtime)
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
