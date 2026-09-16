"""Execute the frozen exact decomposition on the locked parent arrays."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

from fsrl.analysis.statistics import (
    bootstrap_counts,
    stable_sigmoid,
    summarize_subjects,
)
from fsrl.evaluation.sampling import deterministic_cue_codes
from fsrl.infra.formal_runtime import formal_runtime_snapshot
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .estimands import (
    keyed_numeric_max_error,
    packed_keys,
    probability_components,
    relation_source_contributions,
    retained_subject_mean,
    source_concentration,
)
from .locks import (
    ACTIVE_SOURCE_LOCK_PATH,
    ARRAY_PATH,
    RUNTIME_ARRAY_PATH,
    RUNTIME_RESULT_PATH,
    reference,
    validate_source_lock,
)
from .protocol import PROTOCOL_SHA256, load_specification, registered_seeds
from .storage import write_npz_exclusive

EXPECTED_SUBJECTS = 77
EXPECTED_RELATIONS = 8
EXPECTED_SUPPORT_TRIALS = 32
EXPECTED_QUERIES = 56

BUNDLE_FIELDS = (
    "logits",
    "global_logits",
    "raw_local_margins",
    "applied_local_margins",
    "local_gains",
)


def _analysis_runtime() -> dict:
    runtime = formal_runtime_snapshot()
    if not (
        runtime["active"]
        and runtime["torch_intraop_threads"] == 1
        and runtime["torch_interop_threads"] == 1
        and runtime["blas_thread_limit"] == 1
    ):
        raise RuntimeError("use the registered bounded formal runtime")
    return runtime


def _bundle_name(condition: str, field: str) -> str:
    return f"liu__bundles__{condition}__{field}"


def _required_array_names() -> tuple[str, ...]:
    names = [
        "liu__dual_local_evidence",
        "liu__shared_local_evidence",
        "liu__retention",
    ]
    for condition in ("dual_access", "shared_access"):
        names.extend(_bundle_name(condition, field) for field in BUNDLE_FIELDS)
    return tuple(names)


def _load_parent_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(_required_array_names()) - set(payload.files))
        if missing:
            raise RuntimeError(f"parent arrays are missing fields: {missing}")
        arrays = {name: payload[name].copy() for name in _required_array_names()}
    expected_shapes = {
        "liu__dual_local_evidence": (EXPECTED_SUBJECTS, EXPECTED_SUPPORT_TRIALS),
        "liu__shared_local_evidence": (
            EXPECTED_SUBJECTS,
            EXPECTED_SUPPORT_TRIALS,
        ),
        "liu__retention": (EXPECTED_SUBJECTS, EXPECTED_RELATIONS),
    }
    expected_shapes.update(
        {
            _bundle_name(condition, field): (EXPECTED_SUBJECTS, EXPECTED_QUERIES)
            for condition in ("dual_access", "shared_access")
            for field in BUNDLE_FIELDS
        }
    )
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"unexpected shape for {name}: {arrays[name].shape}")
        if arrays[name].dtype == np.dtype("O"):
            raise RuntimeError(f"object dtype is forbidden for {name}")
        if arrays[name].dtype != np.bool_ and not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"non-finite values in {name}")
    if arrays["liu__retention"].dtype != np.bool_:
        raise RuntimeError("retention must be boolean")
    return arrays


def _query_geometry(parent_contract: dict) -> dict:
    settings = parent_contract["evaluation_conditions"]["liu"]
    protocol = load_registered_protocol(str(settings["protocol_id"]))
    relations = tuple(
        (int(pair[0]), int(pair[1])) for pair in protocol.support_pairs_higher_lower
    )
    if (
        protocol.n_items != 8
        or len(relations) != EXPECTED_RELATIONS
        or protocol.support_trials != EXPECTED_SUPPORT_TRIALS
    ):
        raise RuntimeError("the frozen Liu-v2 geometry changed")
    cue_codes = deterministic_cue_codes(
        EXPECTED_SUBJECTS,
        protocol.n_items,
        15,
        int(settings["cue_seed"]),
        mode=str(settings["cue_mode"]),
    )
    schedules = tuple(
        protocol.support_schedule(
            np.random.default_rng(int(settings["support_seed"]) + subject)
        )
        for subject in range(EXPECTED_SUBJECTS)
    )
    relation_index = {relation: index for index, relation in enumerate(relations)}
    support_left = np.empty(
        (EXPECTED_SUBJECTS, EXPECTED_SUPPORT_TRIALS, 15), dtype=np.float32
    )
    support_right = np.empty_like(support_left)
    source_indices = np.empty(
        (EXPECTED_SUBJECTS, EXPECTED_SUPPORT_TRIALS), dtype=np.int64
    )
    for subject, schedule in enumerate(schedules):
        for trial_index, trial in enumerate(schedule):
            support_left[subject, trial_index] = cue_codes[subject, trial.left_item]
            support_right[subject, trial_index] = cue_codes[subject, trial.right_item]
            source_indices[subject, trial_index] = relation_index[
                (trial.higher_item, trial.lower_item)
            ]

    learned_pairs = tuple(
        pair
        for higher, lower in relations
        for pair in ((higher, lower), (lower, higher))
    )
    query_left = np.stack([cue_codes[:, pair[0]] for pair in learned_pairs], axis=1)
    query_right = np.stack([cue_codes[:, pair[1]] for pair in learned_pairs], axis=1)
    all_pair_index = {
        pair: index for index, pair in enumerate(ordered_pairs(protocol.n_items))
    }
    query_indices = np.asarray(
        [all_pair_index[pair] for pair in learned_pairs], dtype=np.int64
    )
    return {
        "protocol": protocol,
        "relations": relations,
        "relation_pairs": np.asarray(relations, dtype=np.int64),
        "query_indices": query_indices,
        "query_signs": np.tile(
            np.asarray([1.0, -1.0], dtype=np.float64),
            (EXPECTED_RELATIONS, 1),
        ),
        "support_keys": packed_keys(support_left, support_right),
        "query_keys": packed_keys(query_left, query_right),
        "source_indices": source_indices,
        "temperature": float(settings["temperature"]),
    }


def _source_contributions(arrays: dict[str, np.ndarray], geometry: dict) -> np.ndarray:
    delta_evidence = (
        arrays["liu__dual_local_evidence"] - arrays["liu__shared_local_evidence"]
    )
    flat = relation_source_contributions(
        delta_evidence,
        geometry["support_keys"],
        geometry["source_indices"],
        geometry["query_keys"],
        relation_count=EXPECTED_RELATIONS,
    )
    return np.transpose(
        flat.reshape(
            EXPECTED_SUBJECTS,
            EXPECTED_RELATIONS,
            EXPECTED_RELATIONS,
            2,
        ),
        (0, 2, 3, 1),
    )


def _learned(array: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
    return array[:, query_indices].reshape(EXPECTED_SUBJECTS, EXPECTED_RELATIONS, 2)


def _constant_gain(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(
        [
            arrays[_bundle_name(condition, "local_gains")].reshape(-1)
            for condition in ("shared_access", "dual_access")
        ]
    )
    spread = float(np.max(values) - np.min(values))
    return float(values[0]), spread


def _distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "lower_quartile": None,
            "upper_quartile": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "lower_quartile": float(np.quantile(finite, 0.25)),
        "upper_quartile": float(np.quantile(finite, 0.75)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _summary(
    values: np.ndarray,
    *,
    seed: int,
    interval: float,
    samples: int,
) -> dict:
    counts = bootstrap_counts(
        np.random.default_rng(89000 + seed), samples, EXPECTED_SUBJECTS
    )
    return summarize_subjects(values, counts, interval=interval)


def _retained_summary(
    values: np.ndarray,
    *,
    retention: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    return summarize_subjects(
        retained_subject_mean(values, retention), counts, interval=interval
    )


def _source_pattern_summary(
    source_values: np.ndarray,
    retention: np.ndarray,
    query_signs: np.ndarray,
    relations: tuple[tuple[int, int], ...],
) -> tuple[dict, dict[str, np.ndarray]]:
    cell_metrics = source_concentration(source_values, query_signs)
    subject_metrics = {
        name: retained_subject_mean(values, retention)
        for name, values in cell_metrics.items()
    }
    aggregate_rows = []
    for target in range(EXPECTED_RELATIONS):
        for source in range(EXPECTED_RELATIONS):
            eligible = retention[:, target] & ~retention[:, source]
            values = np.abs(source_values[eligible, target, :, source]).reshape(-1)
            if not len(values):
                continue
            aggregate_rows.append(
                {
                    "source_index": source,
                    "source_relation": list(relations[source]),
                    "target_index": target,
                    "target_relation": list(relations[target]),
                    "eligible_oriented_cells": len(values),
                    "mean_absolute_gain_free_contribution": float(np.mean(values)),
                    "maximum_absolute_gain_free_contribution": float(np.max(values)),
                }
            )
    aggregate_rows.sort(
        key=lambda row: row["mean_absolute_gain_free_contribution"], reverse=True
    )
    summary = {
        "participant_weighted": {
            name: _distribution(values) for name, values in subject_metrics.items()
        },
        "source_count_80pct_cell_distribution": _distribution(
            np.where(
                np.broadcast_to(
                    retention[:, :, None], cell_metrics["source_count_80pct"].shape
                ),
                cell_metrics["source_count_80pct"],
                np.nan,
            )
        ),
        "largest_relation_to_relation_absolute_contributions": aggregate_rows[:10],
        "classification": "continuous_report_only_no_concentrated_diffuse_cutoff",
    }
    return summary, cell_metrics


def _shapley_rows(factorial: dict, passing_seed: int) -> dict:
    failure_seed = 3006

    def mean(operating: int, gain: int) -> float:
        return float(factorial[str(operating)][str(gain)]["summary"]["mean"])

    total = mean(failure_seed, failure_seed) - mean(passing_seed, passing_seed)
    gain_component = 0.5 * (
        mean(passing_seed, failure_seed)
        - mean(passing_seed, passing_seed)
        + mean(failure_seed, failure_seed)
        - mean(failure_seed, passing_seed)
    )
    operating_component = 0.5 * (
        mean(failure_seed, passing_seed)
        - mean(passing_seed, passing_seed)
        + mean(failure_seed, failure_seed)
        - mean(passing_seed, failure_seed)
    )
    return {
        "passing_reference_seed": passing_seed,
        "total_mean_difference": total,
        "gain_component": gain_component,
        "operating_point_component": operating_component,
        "additivity_error": abs(total - gain_component - operating_component),
        "gain_fraction_of_total": None if total == 0.0 else gain_component / total,
        "operating_point_fraction_of_total": (
            None if total == 0.0 else operating_component / total
        ),
    }


def run_analysis() -> dict:
    runtime = _analysis_runtime()
    source_lock = validate_source_lock()
    specification = load_specification()
    seeds = registered_seeds(specification)
    frozen = specification["design"]["frozen_inputs"]
    parent_contract = load_json(REPO_ROOT / frozen["parent_contract"]["path"])
    parent_result = load_json(REPO_ROOT / frozen["parent_result"]["path"])
    geometry = _query_geometry(parent_contract)
    query_indices = geometry["query_indices"]
    query_signs = geometry["query_signs"]
    temperature = geometry["temperature"]
    interval = float(specification["statistics"]["interval"])
    samples = int(specification["statistics"]["bootstrap_samples"])

    arrays_by_seed = {
        seed: _load_parent_arrays(REPO_ROOT / frozen["raw_arrays"][str(seed)]["path"])
        for seed in seeds
    }
    first = arrays_by_seed[seeds[0]]
    retention = first["liu__retention"]
    shared_evidence = first["liu__shared_local_evidence"]
    dual_evidence = first["liu__dual_local_evidence"]
    cohort_identity_error = 0.0
    for seed in seeds[1:]:
        arrays = arrays_by_seed[seed]
        if not np.array_equal(retention, arrays["liu__retention"]):
            cohort_identity_error = float("inf")
        cohort_identity_error = max(
            cohort_identity_error,
            float(
                np.max(np.abs(shared_evidence - arrays["liu__shared_local_evidence"]))
            ),
            float(np.max(np.abs(dual_evidence - arrays["liu__dual_local_evidence"]))),
        )

    source_by_seed = {
        seed: _source_contributions(arrays_by_seed[seed], geometry) for seed in seeds
    }
    common_source = source_by_seed[seeds[0]]
    geometry_seed_error = max(
        float(np.max(np.abs(source_by_seed[seed] - common_source)))
        for seed in seeds[1:]
    )
    raw_crosstalk = np.sum(common_source, axis=-1)

    source_indices = geometry["source_indices"]
    retained_trial = np.take_along_axis(retention, source_indices, axis=1)
    delta_evidence = dual_evidence - shared_evidence
    retained_write_error = float(
        np.max(np.abs(delta_evidence[retained_trial]), initial=0.0)
    )

    per_seed = {}
    baseline_by_seed: dict[int, np.ndarray] = {}
    global_by_seed: dict[int, np.ndarray] = {}
    gains: dict[int, float] = {}
    components_by_seed: dict[int, dict[str, np.ndarray]] = {}
    subject_effect_by_seed: dict[int, np.ndarray] = {}
    integrity_by_seed = {}

    for seed in seeds:
        arrays = arrays_by_seed[seed]
        shared_logits = _learned(
            arrays[_bundle_name("shared_access", "logits")], query_indices
        )
        dual_logits = _learned(
            arrays[_bundle_name("dual_access", "logits")], query_indices
        )
        shared_global = _learned(
            arrays[_bundle_name("shared_access", "global_logits")], query_indices
        )
        dual_global = _learned(
            arrays[_bundle_name("dual_access", "global_logits")], query_indices
        )
        shared_raw_local = _learned(
            arrays[_bundle_name("shared_access", "raw_local_margins")],
            query_indices,
        )
        dual_raw_local = _learned(
            arrays[_bundle_name("dual_access", "raw_local_margins")],
            query_indices,
        )
        shared_applied_local = _learned(
            arrays[_bundle_name("shared_access", "applied_local_margins")],
            query_indices,
        )
        gain, gain_spread = _constant_gain(arrays)
        gains[seed] = gain
        signs = query_signs[None, :, :]
        baseline = signs * shared_logits
        global_margin = signs * shared_global
        baseline_by_seed[seed] = baseline
        global_by_seed[seed] = global_margin
        components = probability_components(
            baseline,
            source_by_seed[seed].sum(axis=-1),
            query_signs,
            gain=gain,
            temperature=temperature,
        )
        components_by_seed[seed] = components
        computed_subject = retained_subject_mean(components["exact_effect"], retention)
        subject_effect_by_seed[seed] = computed_subject

        raw_effect = stable_sigmoid(signs * dual_logits / temperature) - stable_sigmoid(
            signs * shared_logits / temperature
        )
        raw_subject = retained_subject_mean(raw_effect, retention)
        raw_summary = _summary(
            raw_subject,
            seed=seed,
            interval=interval,
            samples=samples,
        )
        canonical_summary = parent_result["per_seed"][str(seed)]["contrasts"][
            "dual_minus_shared_retained_exact_probability"
        ]
        raw_local_difference = dual_raw_local - shared_raw_local
        total_logit_difference = dual_logits - shared_logits
        computed_delta = gain * source_by_seed[seed].sum(axis=-1)
        reconstructed_dual_probability = stable_sigmoid(
            (signs * shared_logits + signs * total_logit_difference) / temperature
        )
        direct_dual_probability = stable_sigmoid(signs * dual_logits / temperature)
        p_probability = stable_sigmoid(global_margin / temperature)
        p_sensitivity = p_probability * (1.0 - p_probability) / temperature

        errors = {
            "global_condition_identity_max_abs": float(
                np.max(np.abs(dual_global - shared_global))
            ),
            "gain_constancy_max_abs": gain_spread,
            "source_sum_to_raw_margin_difference_max_abs": float(
                np.max(np.abs(source_by_seed[seed].sum(axis=-1) - raw_local_difference))
            ),
            "gain_scaled_sum_to_logit_difference_max_abs": float(
                np.max(np.abs(computed_delta - total_logit_difference))
            ),
            "computed_crosstalk_probability_to_raw_probability_max_abs": float(
                np.max(np.abs(components["exact_effect"] - raw_effect))
            ),
            "raw_probability_reconstruction_max_abs": float(
                np.max(np.abs(reconstructed_dual_probability - direct_dual_probability))
            ),
            "canonical_retained_subject_and_summary_max_abs": max(
                float(
                    np.max(
                        np.abs(
                            raw_subject - retained_subject_mean(raw_effect, retention)
                        )
                    )
                ),
                keyed_numeric_max_error(raw_summary, canonical_summary),
            ),
        }
        integrity_by_seed[str(seed)] = errors

        counts = bootstrap_counts(
            np.random.default_rng(89000 + seed), samples, EXPECTED_SUBJECTS
        )

        retained_summary = partial(
            _retained_summary,
            retention=retention,
            counts=counts,
            interval=interval,
        )

        correct_crosstalk = source_by_seed[seed].sum(axis=-1) * signs
        per_seed[str(seed)] = {
            "local_gain": gain,
            "parent_retained_contrast": canonical_summary,
            "reconstructed_retained_contrast": retained_summary(
                components["exact_effect"]
            ),
            "raw_retained_contrast_reproduction": raw_summary,
            "retained_cell_summaries": {
                "shared_total_correct_margin": retained_summary(baseline),
                "global_P_correct_margin": retained_summary(global_margin),
                "shared_local_correct_margin": retained_summary(
                    signs * shared_applied_local
                ),
                "gain_free_correct_crosstalk": retained_summary(correct_crosstalk),
                "gain_scaled_correct_perturbation": retained_summary(
                    components["correct_perturbation"]
                ),
                "shared_total_probability_sensitivity": retained_summary(
                    components["sensitivity"]
                ),
                "P_only_probability_sensitivity": retained_summary(p_sensitivity),
                "first_order_probability_effect": retained_summary(
                    components["first_order_effect"]
                ),
                "exact_probability_effect": retained_summary(
                    components["exact_effect"]
                ),
                "nonlinear_remainder": retained_summary(
                    components["nonlinear_remainder"]
                ),
            },
            "retained_oriented_cells": int(2 * np.sum(retention)),
            "negative_exact_effect_fraction": float(
                np.mean(
                    components["exact_effect"][
                        np.broadcast_to(
                            retention[:, :, None], components["exact_effect"].shape
                        )
                    ]
                    < 0.0
                )
            ),
            "integrity_errors": errors,
        }

    factorial: dict[str, dict[str, dict]] = {}
    factorial_subjects = np.empty(
        (len(seeds), len(seeds), EXPECTED_SUBJECTS), dtype=np.float64
    )
    for operating_index, operating_seed in enumerate(seeds):
        factorial[str(operating_seed)] = {}
        counts = bootstrap_counts(
            np.random.default_rng(89000 + operating_seed),
            samples,
            EXPECTED_SUBJECTS,
        )
        for gain_index, gain_seed in enumerate(seeds):
            components = probability_components(
                baseline_by_seed[operating_seed],
                raw_crosstalk,
                query_signs,
                gain=gains[gain_seed],
                temperature=temperature,
            )
            subject_values = retained_subject_mean(
                components["exact_effect"], retention
            )
            factorial_subjects[operating_index, gain_index] = subject_values
            summary = summarize_subjects(subject_values, counts, interval=interval)
            factorial[str(operating_seed)][str(gain_seed)] = {
                "operating_point_seed": operating_seed,
                "gain_seed": gain_seed,
                "gain": gains[gain_seed],
                "summary": summary,
                "frozen_reference_gate_passed": bool(
                    summary["bootstrap"]["lower"] >= -0.005
                ),
            }

    shapley = [_shapley_rows(factorial, seed) for seed in (3004, 3005)]
    shapley_error = max(row["additivity_error"] for row in shapley)
    lower_gain_rescues_3006 = {
        str(seed): factorial["3006"][str(seed)]["frozen_reference_gate_passed"]
        for seed in (3004, 3005)
    }
    gain_3006_fails_passing_operating_points = {
        str(seed): not factorial[str(seed)]["3006"]["frozen_reference_gate_passed"]
        for seed in (3004, 3005)
    }
    joint_boundary = bool(
        any(lower_gain_rescues_3006.values())
        and not any(gain_3006_fails_passing_operating_points.values())
    )

    source_summary, concentration_arrays = _source_pattern_summary(
        common_source,
        retention,
        query_signs,
        geometry["relations"],
    )

    tolerances = specification["integrity_and_decision"]["algebra_checks"]
    algebra_checks = {
        "retained_source_write": {
            "value": retained_write_error,
            "maximum": float(tolerances["retained_source_write_max_abs"]),
        },
        "source_sum_to_raw_margin_difference": {
            "value": max(
                row["source_sum_to_raw_margin_difference_max_abs"]
                for row in integrity_by_seed.values()
            ),
            "maximum": float(tolerances["source_sum_to_raw_margin_difference_max_abs"]),
        },
        "gain_scaled_sum_to_logit_difference": {
            "value": max(
                row["gain_scaled_sum_to_logit_difference_max_abs"]
                for row in integrity_by_seed.values()
            ),
            "maximum": float(tolerances["gain_scaled_sum_to_logit_difference_max_abs"]),
        },
        "computed_crosstalk_probability_to_raw_probability": {
            "value": max(
                row["computed_crosstalk_probability_to_raw_probability_max_abs"]
                for row in integrity_by_seed.values()
            ),
            "maximum": float(
                tolerances["computed_crosstalk_probability_to_raw_probability_max_abs"]
            ),
        },
        "raw_probability_reconstruction": {
            "value": max(
                row["raw_probability_reconstruction_max_abs"]
                for row in integrity_by_seed.values()
            ),
            "maximum": float(tolerances["raw_probability_reconstruction_max_abs"]),
        },
        "canonical_retained_subject_and_summary": {
            "value": max(
                row["canonical_retained_subject_and_summary_max_abs"]
                for row in integrity_by_seed.values()
            ),
            "maximum": float(
                tolerances["canonical_retained_subject_and_summary_max_abs"]
            ),
        },
        "shapley_additivity": {
            "value": shapley_error,
            "maximum": float(tolerances["shapley_additivity_max_abs"]),
        },
    }
    for row in algebra_checks.values():
        row["passed"] = bool(row["value"] <= row["maximum"])
    input_checks = {
        "locked_inputs_validated": True,
        "cohort_evidence_and_retention_identity": bool(cohort_identity_error == 0.0),
        "gain_is_constant_within_each_seed": bool(
            all(
                row["gain_constancy_max_abs"] == 0.0
                for row in integrity_by_seed.values()
            )
        ),
        "global_logits_identical_between_access_conditions": bool(
            all(
                row["global_condition_identity_max_abs"] == 0.0
                for row in integrity_by_seed.values()
            )
        ),
        "gain_free_geometry_identical_across_seeds": bool(geometry_seed_error == 0.0),
    }
    passed = all(input_checks.values()) and all(
        row["passed"] for row in algebra_checks.values()
    )
    outcome = (
        "exact_cross_talk_decomposed"
        if passed
        else "noninterpretable_input_or_algebra_failure"
    )

    seed_axis = np.asarray(seeds, dtype=np.int64)
    baseline_stack = np.stack([baseline_by_seed[seed] for seed in seeds])
    global_stack = np.stack([global_by_seed[seed] for seed in seeds])
    components_stack = {
        name: np.stack([components_by_seed[seed][name] for seed in seeds])
        for name in (
            "correct_perturbation",
            "sensitivity",
            "exact_effect",
            "first_order_effect",
            "nonlinear_remainder",
        )
    }
    p_probability = stable_sigmoid(global_stack / temperature)
    p_only_sensitivity = p_probability * (1.0 - p_probability) / temperature
    support_arrays = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "network_seeds": seed_axis,
        "relation_pairs": geometry["relation_pairs"],
        "query_signs": query_signs.astype(np.int8),
        "retention": retention,
        "source_contributions_gain_free": common_source,
        "raw_crosstalk_oriented": raw_crosstalk,
        "baseline_shared_correct_margin": baseline_stack,
        "global_P_correct_margin": global_stack,
        "shared_local_correct_margin": baseline_stack - global_stack,
        "local_gains": np.asarray([gains[seed] for seed in seeds]),
        "gain_scaled_correct_perturbation": components_stack["correct_perturbation"],
        "shared_total_sensitivity": components_stack["sensitivity"],
        "P_only_sensitivity": p_only_sensitivity,
        "exact_probability_effect": components_stack["exact_effect"],
        "first_order_probability_effect": components_stack["first_order_effect"],
        "nonlinear_remainder": components_stack["nonlinear_remainder"],
        "subject_retained_effect": np.stack(
            [subject_effect_by_seed[seed] for seed in seeds]
        ),
        "factorial_subject_retained_effect": factorial_subjects,
        **{
            f"concentration_{name}": values
            for name, values in concentration_arrays.items()
        },
    }
    RUNTIME_ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_npz_exclusive(RUNTIME_ARRAY_PATH, support_arrays)
    array_reference = reference(RUNTIME_ARRAY_PATH)
    tracked_array_reference = {
        **array_reference,
        "path": ARRAY_PATH.relative_to(REPO_ROOT).as_posix(),
    }
    result = {
        "schema_version": 1,
        "study_id": specification["study_id"],
        "experiment_id": specification["experiment_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(ACTIVE_SOURCE_LOCK_PATH),
        "source_commit": source_lock["source_commit"],
        "implementation_repair": source_lock["repair"],
        "runtime": runtime,
        "parent_outcome_unchanged": {
            "outcome": parent_result["outcome"],
            "seed_3006_retained_fidelity_passed": parent_result["per_seed"]["3006"][
                "four_v2_4_links"
            ]["flags"]["retained_fidelity_preservation"],
            "interpretation": "The parent remains a formal 2/3 failure and is not relabeled by this diagnostic.",
        },
        "outcome": outcome,
        "integrity": {
            "passed": passed,
            "input_checks": input_checks,
            "cohort_identity_max_abs_error": cohort_identity_error,
            "gain_free_geometry_across_seed_max_abs_error": geometry_seed_error,
            "algebra_checks": algebra_checks,
            "per_seed_errors": integrity_by_seed,
        },
        "per_seed": per_seed,
        "gain_by_operating_point_factorial": factorial,
        "factorial_diagnostics": {
            "lower_gain_rescues_seed_3006_gate": lower_gain_rescues_3006,
            "seed_3006_gain_fails_passing_operating_points": (
                gain_3006_fails_passing_operating_points
            ),
            "classification": (
                "joint_gain_by_operating_point_boundary"
                if joint_boundary
                else "other_registered_factorial_pattern"
            ),
            "mean_shapley_splits": shapley,
        },
        "source_concentration": source_summary,
        "supporting_arrays": {
            **tracked_array_reference,
            "runtime_source": array_reference,
            "load_policy": "np.load(path, allow_pickle=False)",
            "array_schema": {
                name: {"shape": list(values.shape), "dtype": str(values.dtype)}
                for name, values in support_arrays.items()
            },
        },
        "claim_boundary": specification["claim_boundary"],
        "next_step_boundary": "No training or transport is authorized. Interpret the decomposition first; any successor requires a new prospective question.",
    }
    if not passed:
        result["factorial_diagnostics"]["classification"] = "not_interpretable"
    write_json_exclusive(RUNTIME_RESULT_PATH, result)
    return result
