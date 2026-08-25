"""External human-field adequacy audit for the frozen exact posterior."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assembly_trajectory import build_complete_graph_geometry
from .config import TrainConfig
from .confirmation import file_sha256
from .curvature_gate_pilot import load_json
from .formal_runtime import require_formal_runtime
from .global_policy_amplitude_provenance import (
    NonInterpretableEstimate,
    _interval_summary,
    _posterior_descriptors,
)
from .global_policy_field_fingerprint_replication import require_pushed_freeze
from .human_benchmark import REQUIRED_COLUMNS
from .liu_eval import FrozenFastWeightEvaluator
from .ranking_protocol import load_ranking_protocol
from .study_registry import legacy_identifier, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/global_policy_comparator_adequacy_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/global_policy_comparator_adequacy_v1.lock.json")
)
DEFAULT_PROTOCOL_REPAIR_PATH = (
    resolve_record("benchmarks/global_policy_comparator_adequacy_v1.repair1.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/global_policy_comparator_adequacy_v1.json")

REQUIRED_IMPLEMENTATION_SOURCE_PATHS = {
    "audit_runner": "fsrl/global_policy_comparator_adequacy.py",
    "audit_tests": "tests/test_global_policy_comparator_adequacy.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}
REQUIRED_REUSED_SOURCE_NAMES = {
    "human_benchmark_source",
    "posterior_descriptor_source",
    "frozen_evaluator_source",
    "subject_encoding_source",
    "ranking_protocol_source",
    "exact_posterior_source",
    "configuration_source",
    "assembly_source",
    "hash_validation_source",
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def _canonical_paths(parsed: argparse.Namespace) -> None:
    expected = {
        "specification": DEFAULT_SPECIFICATION_PATH,
        "implementation_lock": DEFAULT_IMPLEMENTATION_LOCK_PATH,
        "protocol_repair": DEFAULT_PROTOCOL_REPAIR_PATH,
    }
    for name, canonical in expected.items():
        if getattr(parsed, name).resolve() != canonical.resolve():
            raise RuntimeError(f"formal workflow requires canonical {name}")
    result = parsed.result.resolve()
    if result != DEFAULT_RESULT_PATH.resolve():
        try:
            relative = result.relative_to(Path("/tmp").resolve())
        except ValueError as error:
            raise RuntimeError(
                "formal replay result must be a file below /tmp"
            ) from error
        if not relative.parts:
            raise RuntimeError("formal replay result must be a file below /tmp")
    if parsed.result.exists() or parsed.result.is_symlink():
        raise RuntimeError("comparator-adequacy result already exists or is a symlink")
    if parsed.result.parent.is_symlink():
        raise RuntimeError("comparator-adequacy result parent may not be a symlink")


def write_json_exclusive(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def apply_protocol_repair(
    specification: dict,
    repair_path: Path = DEFAULT_PROTOCOL_REPAIR_PATH,
) -> tuple[dict, dict]:
    repair = load_json(repair_path)
    original = repair.get("original_specification", {})
    upstream = repair.get("upstream_identity", {})
    correction = repair.get("repair", {})
    if not (
        repair.get("schema_version") == 1
        and repair.get("audit_id") == specification.get("audit_id")
        and repair.get("repair_status")
        == "prospectively_frozen_after_static_edge_contract_failure_and_before_any_human_or_posterior_adequacy_evaluation"
        and original.get("path") == legacy_identifier(DEFAULT_SPECIFICATION_PATH)
        and original.get("sha256") == file_sha256(DEFAULT_SPECIFICATION_PATH)
        and correction.get("json_pointer")
        == "/field_contract/distance_level_pair_counts"
        and correction.get("registered_value")
        == specification["field_contract"]["distance_level_pair_counts"]
        == [4, 5, 4, 3, 2, 2]
        and correction.get("corrected_value") == [6, 5, 2, 3, 2, 2]
    ):
        raise RuntimeError("comparator-adequacy Repair 1 identity mismatch")
    for name in ("allocation_specification", "allocation_result"):
        registration = upstream.get(name, {})
        if not (
            registration.get("distance_level_pair_counts") == [6, 5, 2, 3, 2, 2]
            and file_sha256(_resolve(registration["path"]))
            == registration.get("sha256")
        ):
            raise RuntimeError(f"comparator-adequacy repair anchor mismatch: {name}")
    effective = json.loads(json.dumps(specification))
    effective["field_contract"]["distance_level_pair_counts"] = correction[
        "corrected_value"
    ]
    return effective, {
        "passed": True,
        "path": str(repair_path.relative_to(ROOT)),
        "sha256": file_sha256(repair_path),
        "json_pointer": correction["json_pointer"],
        "registered_value": correction["registered_value"],
        "corrected_value": correction["corrected_value"],
        "scientific_estimands_changed": False,
    }


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    protocol_repair_path: Path = DEFAULT_PROTOCOL_REPAIR_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    if not (
        lock.get("schema_version") == 1
        and lock.get("audit_id") == specification.get("audit_id")
        and lock.get("freeze_status")
        == "implementation_frozen_after_protocol_commit_and_before_formal_evaluation"
        and lock.get("audit_specification_sha256") == file_sha256(specification_path)
        and lock.get("protocol_repair")
        == {
            "path": str(protocol_repair_path.relative_to(ROOT)),
            "sha256": file_sha256(protocol_repair_path),
        }
    ):
        raise RuntimeError("comparator-adequacy implementation lock mismatch")
    implementation = lock.get("implementation_sources", {})
    reused = lock.get("reused_frozen_sources", {})
    if set(implementation) != set(REQUIRED_IMPLEMENTATION_SOURCE_PATHS):
        raise RuntimeError("comparator-adequacy implementation source set mismatch")
    if set(reused) != REQUIRED_REUSED_SOURCE_NAMES:
        raise RuntimeError("comparator-adequacy reused source set mismatch")
    for name, path in REQUIRED_IMPLEMENTATION_SOURCE_PATHS.items():
        if Path(implementation[name].get("path", "")) != Path(path):
            raise RuntimeError(f"implementation source path mismatch: {name}")
    for name in REQUIRED_REUSED_SOURCE_NAMES:
        if reused[name] != specification["registered_sources"][name]:
            raise RuntimeError(f"reused source registration mismatch: {name}")

    registrations = {
        **specification["registered_sources"],
        "audit_specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["audit_specification_sha256"],
        },
        "protocol_repair": lock["protocol_repair"],
        **implementation,
    }
    checks = []
    for name, registration in registrations.items():
        path = _resolve(registration["path"])
        observed = file_sha256(path)
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
        raise RuntimeError(f"comparator-adequacy source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def validate_prerequisite(specification: dict) -> dict:
    allocation = load_json(
        _resolve(specification["registered_sources"]["allocation_result"]["path"])
    )
    decision = allocation.get("decision", {})
    axes = decision.get("axes", {})
    checks = {
        "source_validation": allocation.get("source_validation", {}).get("passed")
        is True,
        "artifact_validation": allocation.get("artifact_validation", {}).get("passed")
        is True,
        "fingerprint_prerequisite": allocation.get("fingerprint_prerequisite", {}).get(
            "passed"
        )
        is True,
        "seed_integrity": set(allocation.get("seeds", {})) == {"2106", "2107"}
        and all(
            row.get("integrity", {}).get("passed") is True
            for row in allocation.get("seeds", {}).values()
        ),
        "cross_network_integrity": allocation.get("cross_network_integrity", {}).get(
            "passed"
        )
        is True,
        "outcome": decision.get("outcome") == "policy_effective_allocation_localized",
        "scope": decision.get("localization_scope") == "structural_only",
        "pair_policy_effective": axes.get("pair_identity", {}).get("policy_effective")
        is True,
        "distance_policy_effective": axes.get("symbolic_distance", {}).get(
            "policy_effective"
        )
        is True,
        "uncertainty_not_policy_effective": axes.get("posterior_uncertainty", {}).get(
            "policy_effective"
        )
        is False,
        "coverage_not_policy_effective": axes.get(
            "effective_evidence_coverage", {}
        ).get("policy_effective")
        is False,
        "registered_next_step": decision.get("conditional_next_step")
        == "prospective_comparator_adequacy",
    }
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"allocation prerequisite failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "allocation_outcome": decision["outcome"],
        "allocation_scope": decision["localization_scope"],
        "conditional_next_step": decision["conditional_next_step"],
    }


def edge_metadata(specification: dict, protocol) -> dict:
    geometry = build_complete_graph_geometry(protocol)
    labels = tuple(protocol.item_labels)
    pair_labels = tuple(
        f"{labels[first]}-{labels[second]}" for first, second in geometry.pairs
    )
    positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        positions[item] = position
    distances = np.asarray(
        [abs(positions[first] - positions[second]) for first, second in geometry.pairs],
        dtype=np.float64,
    )
    nonlearned = np.asarray(
        [pair not in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    selected_labels = tuple(
        label
        for label, selected in zip(pair_labels, nonlearned, strict=True)
        if selected
    )
    contract = specification["field_contract"]
    selected_distances = distances[nonlearned]
    levels = np.asarray(contract["distance_levels"], dtype=np.int64)
    counts = np.asarray(
        [np.sum(selected_distances == level) for level in levels], dtype=np.int64
    )
    centered = selected_distances - np.mean(selected_distances)
    denominator = float(centered @ centered)
    if not (
        len(geometry.pairs) == 28
        and int(np.sum(nonlearned)) == 20
        and selected_labels == tuple(contract["nonlearned_pair_labels"])
        and counts.tolist() == contract["distance_level_pair_counts"]
        and abs(float(np.mean(selected_distances)) - 2.8) <= 1e-10
        and abs(denominator - 57.2) <= 1e-10
    ):
        raise RuntimeError("frozen comparator edge design mismatch")
    design = np.column_stack((np.ones(len(selected_distances)), selected_distances))
    residualizer = np.eye(len(selected_distances)) - design @ np.linalg.pinv(design)
    return {
        "geometry": geometry,
        "n_items": protocol.n_items,
        "pair_labels": pair_labels,
        "selected_labels": selected_labels,
        "nonlearned": nonlearned,
        "correct_sign": np.asarray(geometry.true_sign, dtype=np.float64),
        "distances": distances,
        "selected_distances": selected_distances,
        "distance_levels": levels,
        "distance_counts": counts,
        "distance_weights": centered / denominator,
        "distance_mean": float(np.mean(selected_distances)),
        "distance_denominator": denominator,
        "design": design,
        "residualizer": residualizer,
    }


def _load_trial_cohort(
    path: Path,
    cohort: str,
    pair_to_index: dict[tuple[int, int], int],
    n_items: int,
) -> tuple[np.ndarray, list[str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise RuntimeError(f"{cohort} human source columns changed")
        for row in reader:
            grouped[int(row["id"])].append(row)
    expected_subjects = {"preregistered": 40, "replication": 37}[cohort]
    if len(grouped) != expected_subjects:
        raise RuntimeError(f"{cohort} human participant count changed")

    arrays = []
    labels = []
    for source_id, rows in sorted(grouped.items()):
        matrix = np.full((10, len(pair_to_index)), np.nan, dtype=np.float64)
        identifiers = set()
        for row in rows:
            trial = int(row["trial"])
            block = int(row["block"])
            first_source = int(row["film_index_1"])
            second_source = int(row["film_index_2"])
            chosen_source = int(row["film_choose_index"])
            correct = int(row["r_or_w"])
            if not (
                1 <= block <= 10
                and 1 <= first_source <= n_items
                and 1 <= second_source <= n_items
                and first_source != second_source
                and chosen_source in {first_source, second_source}
                and correct in {0, 1}
                and correct == int(chosen_source == max(first_source, second_source))
            ):
                raise RuntimeError(f"invalid human trial for {cohort}:{source_id}")
            pair = tuple(sorted((first_source - 1, second_source - 1)))
            pair_index = pair_to_index[pair]
            if np.isfinite(matrix[block - 1, pair_index]):
                raise RuntimeError(
                    f"duplicate human pair-block for {cohort}:{source_id}"
                )
            matrix[block - 1, pair_index] = float(correct)
            identifiers.add((block, trial))
        if (
            len(rows) != 280
            or len(identifiers) != 280
            or not np.all(np.isfinite(matrix))
        ):
            raise RuntimeError(f"incomplete human trials for {cohort}:{source_id}")
        arrays.append(matrix)
        labels.append(f"{cohort}:{source_id}")
    return np.stack(arrays), labels


def load_human_fields(specification: dict, metadata: dict) -> tuple[dict, dict]:
    sources = specification["registered_sources"]
    pair_to_index = {
        pair: index for index, pair in enumerate(metadata["geometry"].pairs)
    }
    preregistered, preregistered_labels = _load_trial_cohort(
        _resolve(sources["human_preregistered_trials"]["path"]),
        "preregistered",
        pair_to_index,
        metadata["n_items"],
    )
    replication, replication_labels = _load_trial_cohort(
        _resolve(sources["human_replication_trials"]["path"]),
        "replication",
        pair_to_index,
        metadata["n_items"],
    )
    trials = np.concatenate((preregistered, replication), axis=0)
    labels = preregistered_labels + replication_labels
    full_28 = np.mean(trials, axis=1)
    odd_28 = np.mean(trials[:, 0::2, :], axis=1)
    even_28 = np.mean(trials[:, 1::2, :], axis=1)
    eligible = np.mean(full_28, axis=1) >= 0.5
    selected = metadata["nonlearned"]
    full = full_28[:, selected]
    odd = odd_28[:, selected]
    even = even_28[:, selected]

    benchmark = load_json(_resolve(sources["human_benchmark"]["path"]))
    benchmark_rows = benchmark.get("combined", {}).get("pairs", [])
    benchmark_labels = tuple("-".join(row["pair"]) for row in benchmark_rows)
    benchmark_means = np.asarray(
        [row["mean_accuracy"] for row in benchmark_rows], dtype=np.float64
    )
    benchmark_error = (
        float(np.max(np.abs(np.mean(full_28, axis=0) - benchmark_means)))
        if len(benchmark_means) == 28
        else float("inf")
    )
    integrity = {
        "preregistered_subjects": len(preregistered),
        "replication_subjects": len(replication),
        "combined_subjects": len(trials),
        "eligible_subjects": int(np.sum(eligible)),
        "blocks": int(trials.shape[1]),
        "edges": int(trials.shape[2]),
        "observations_per_subject_pair": 10,
        "observations_per_half_subject_pair": 5,
        "all_human_values_binary": bool(np.all((trials == 0.0) | (trials == 1.0))),
        "all_human_arrays_finite": bool(
            all(np.all(np.isfinite(value)) for value in (trials, full, odd, even))
        ),
        "full_equals_half_average_max_abs_error": float(
            np.max(np.abs(full - 0.5 * (odd + even)))
        ),
        "human_benchmark_pair_order_matches": benchmark_labels
        == metadata["pair_labels"],
        "human_benchmark_pair_mean_max_abs_error": benchmark_error,
        "human_benchmark_status_matches": benchmark.get("status")
        == "source_recomputed_and_paper_checks_reproduced",
        "human_benchmark_eligible_subjects": benchmark.get("combined", {}).get(
            "eligible_subjects"
        ),
    }
    return {
        "participant_labels": labels,
        "full": full,
        "odd": odd,
        "even": even,
        "full_28": full_28,
        "cohort_slices": {
            "preregistered": slice(0, len(preregistered)),
            "replication": slice(len(preregistered), len(trials)),
        },
        "historical_all_pair_human_slope": benchmark["combined"][
            "symbolic_distance_slope"
        ]["mean"],
    }, integrity


def posterior_fields(specification: dict, metadata: dict) -> tuple[dict, dict]:
    contract = specification["posterior_comparator"]
    allocation_specification = load_json(
        _resolve(
            specification["registered_sources"]["allocation_specification"]["path"]
        )
    )
    evaluation = allocation_specification["evaluation"]
    expected = {
        "subjects": int(evaluation["subjects"]),
        "cue_seed": int(evaluation["cue_seed"]),
        "support_seed": int(evaluation["support_seed"]),
        "subject_encoding_seed": int(evaluation["subject_encoding_seed"]),
        "cue_mode": str(evaluation["cue_mode"]),
        "subject_encoding_mode": str(evaluation["subject_encoding_mode"]),
        "posterior_temperature": float(evaluation["posterior_temperature"]),
        "choice_temperature": float(evaluation["choice_temperature"]),
    }
    observed = {name: contract[name] for name in expected}
    if observed != expected:
        raise RuntimeError("frozen comparator differs from allocation audit")

    config = TrainConfig(bs=int(contract["subjects"]))
    evaluator = FrozenFastWeightEvaluator(
        None,
        config,
        load_ranking_protocol(
            _resolve(specification["registered_sources"]["liu_protocol"]["path"])
        ),
        cue_seed=int(contract["cue_seed"]),
        support_seed=int(contract["support_seed"]),
        cue_mode=str(contract["cue_mode"]),
        subject_encoding_mode=str(contract["subject_encoding_mode"]),
        subject_encoding_seed=int(contract["subject_encoding_seed"]),
    )
    posterior, posterior_integrity = _posterior_descriptors(
        evaluator,
        metadata["geometry"],
        {"posterior_comparator": contract},
    )
    correct_probability = 0.5 * (
        1.0
        + metadata["correct_sign"][None, :]
        * np.asarray(posterior["fields"]["pair_probability_field"], dtype=np.float64)
    )
    selected = correct_probability[:, metadata["nonlearned"]]
    subject_slopes = selected @ metadata["distance_weights"]

    fingerprint = load_json(
        _resolve(specification["registered_sources"]["fingerprint_result"]["path"])
    )
    anchors = {
        seed: np.asarray(
            fingerprint["seeds"][seed]["statistics"]["raw_subject_level"]["S_PP"],
            dtype=np.float64,
        )
        for seed in ("2106", "2107")
    }
    integrity = {
        **posterior_integrity,
        "subjects": int(selected.shape[0]),
        "edges": int(correct_probability.shape[1]),
        "nonlearned_pairs": int(selected.shape[1]),
        "all_posterior_probabilities_finite": bool(np.all(np.isfinite(selected))),
        "posterior_probability_min": float(np.min(selected)),
        "posterior_probability_max": float(np.max(selected)),
        "fingerprint_anchor_cross_seed_max_abs_error": float(
            np.max(np.abs(anchors["2106"] - anchors["2107"]))
        ),
        "posterior_slope_anchor_2106_max_abs_error": float(
            np.max(np.abs(subject_slopes - anchors["2106"]))
        ),
        "posterior_slope_anchor_2107_max_abs_error": float(
            np.max(np.abs(subject_slopes - anchors["2107"]))
        ),
    }
    return {
        "subject_probability": selected,
        "cohort_probability": np.mean(selected, axis=0),
        "subject_slopes": subject_slopes,
        "historical_track_b_posterior_slope": float(np.mean(anchors["2106"])),
    }, integrity


def bootstrap_counts(specification: dict, subjects: int) -> np.ndarray:
    contract = specification["statistical_contract"]
    rng = np.random.default_rng(int(contract["bootstrap_seed"]))
    return rng.multinomial(
        subjects,
        np.full(subjects, 1.0 / subjects),
        size=int(contract["bootstrap_samples"]),
    )


def residualize(values: np.ndarray, residualizer: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.shape[-1] != residualizer.shape[0]:
        raise ValueError("residualizer and pair vector dimensions differ")
    return rows @ residualizer.T


def vector_correlation(first: np.ndarray, second: np.ndarray, minimum: float) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= minimum:
        raise NonInterpretableEstimate("pair correlation has a degenerate vector")
    value = float((left @ right) / denominator)
    if not np.isfinite(value):
        raise NonInterpretableEstimate("pair correlation is nonfinite")
    return value


def row_correlations(
    first: np.ndarray, second: np.ndarray, minimum: float
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim == 1:
        left = np.broadcast_to(left, right.shape)
    if right.ndim == 1:
        right = np.broadcast_to(right, left.shape)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row correlations require matching row-by-pair arrays")
    left = left - np.mean(left, axis=1, keepdims=True)
    right = right - np.mean(right, axis=1, keepdims=True)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    values = np.divide(
        np.sum(left * right, axis=1),
        norms,
        out=np.full(len(left), np.nan, dtype=np.float64),
        where=norms > minimum,
    )
    return values, norms


def distance_profiles(values: np.ndarray, metadata: dict) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    return np.stack(
        [
            np.mean(rows[:, metadata["selected_distances"] == level], axis=1)
            for level in metadata["distance_levels"]
        ],
        axis=1,
    )


def adequacy_statistics(
    specification: dict,
    human: dict,
    posterior: dict,
    metadata: dict,
) -> tuple[dict, dict, dict]:
    subjects = len(human["full"])
    counts = bootstrap_counts(specification, subjects)
    full_boot = counts @ human["full"] / subjects
    odd_boot = counts @ human["odd"] / subjects
    even_boot = counts @ human["even"] / subjects
    human_point = np.mean(human["full"], axis=0)
    odd_point = np.mean(human["odd"], axis=0)
    even_point = np.mean(human["even"], axis=0)
    posterior_point = posterior["cohort_probability"]
    weights = metadata["distance_weights"]
    human_subject_slopes = human["full"] @ weights
    human_slope = float(human_point @ weights)
    posterior_slope = float(posterior_point @ weights)
    human_slope_boot = full_boot @ weights

    residualizer = metadata["residualizer"]
    residual_human = residualize(human_point, residualizer)
    residual_odd = residualize(odd_point, residualizer)
    residual_even = residualize(even_point, residualizer)
    residual_posterior = residualize(posterior_point, residualizer)
    residual_full_boot = residualize(full_boot, residualizer)
    residual_odd_boot = residualize(odd_boot, residualizer)
    residual_even_boot = residualize(even_boot, residualizer)
    minimum = float(specification["statistical_contract"]["minimum_vector_norm"])

    r_ph = vector_correlation(residual_posterior, residual_human, minimum)
    r_hh = vector_correlation(residual_odd, residual_even, minimum)
    if r_hh <= 0.0:
        raise NonInterpretableEstimate(
            "point human split-half reliability is not positive"
        )
    rho_h = 2.0 * r_hh / (1.0 + r_hh)
    eta_pair = r_ph / np.sqrt(rho_h)

    r_ph_boot, norm_ph = row_correlations(
        residual_posterior, residual_full_boot, minimum
    )
    r_hh_boot, norm_hh = row_correlations(
        residual_odd_boot, residual_even_boot, minimum
    )
    if not (
        np.all(np.isfinite(r_ph_boot))
        and np.all(np.isfinite(r_hh_boot))
        and np.all(r_hh_boot > 0.0)
    ):
        raise NonInterpretableEstimate(
            "registered human split-half correlation bootstrap failed"
        )
    rho_h_boot = 2.0 * r_hh_boot / (1.0 + r_hh_boot)
    if not (np.all(np.isfinite(rho_h_boot)) and np.all(rho_h_boot > 0.0)):
        raise NonInterpretableEstimate(
            "registered Spearman-Brown reliability bootstrap failed"
        )
    eta_boot = r_ph_boot / np.sqrt(rho_h_boot)
    if not np.all(np.isfinite(eta_boot)):
        raise NonInterpretableEstimate("corrected ceiling ratio is nonfinite")

    human_profiles_by_subject = distance_profiles(human["full"], metadata)
    posterior_profiles_by_subject = distance_profiles(
        posterior["subject_probability"], metadata
    )
    human_profile = np.mean(human_profiles_by_subject, axis=0)
    posterior_profile = np.mean(posterior_profiles_by_subject, axis=0)
    human_profile_boot = counts @ human_profiles_by_subject / subjects

    distance_interval = _interval_summary(human_slope, human_slope_boot)
    lower = distance_interval["bootstrap"]["lower95"]
    upper = distance_interval["bootstrap"]["upper95"]
    if posterior_slope < lower:
        distance_status = "inadequate_below"
    elif posterior_slope > upper:
        distance_status = "inadequate_above"
    else:
        distance_status = "adequate"
    eta_summary = _interval_summary(float(eta_pair), eta_boot)
    pair_status = (
        "adequate" if eta_summary["bootstrap"]["lower90"] >= 0.80 else "inadequate"
    )

    source_cohorts = {}
    for name, indices in human["cohort_slices"].items():
        field = np.mean(human["full"][indices], axis=0)
        source_cohorts[name] = {
            "subjects": len(human["full"][indices]),
            "field": field.tolist(),
            "slope": float(field @ weights),
        }
    distance_rows = []
    for index, level in enumerate(metadata["distance_levels"]):
        human_summary = _interval_summary(
            float(human_profile[index]), human_profile_boot[:, index]
        )
        difference = human_profile_boot[:, index] - posterior_profile[index]
        distance_rows.append(
            {
                "distance": int(level),
                "pairs": int(metadata["distance_counts"][index]),
                "human": human_summary,
                "posterior_point": float(posterior_profile[index]),
                "human_minus_posterior": _interval_summary(
                    float(human_profile[index] - posterior_profile[index]), difference
                ),
            }
        )

    statistics = {
        "primary": {
            "distance": {
                "human_S_H": distance_interval,
                "posterior_S_P": posterior_slope,
                "status": distance_status,
                "adequate": distance_status == "adequate",
            },
            "pair": {
                "r_PH": _interval_summary(r_ph, r_ph_boot),
                "r_HH": _interval_summary(r_hh, r_hh_boot),
                "rho_H_spearman_brown": _interval_summary(rho_h, rho_h_boot),
                "eta_pair": eta_summary,
                "threshold": 0.80,
                "interval_rule": "lower90_at_least_threshold",
                "status": pair_status,
                "adequate": pair_status == "adequate",
            },
        },
        "secondary": {
            "distance_profile": distance_rows,
            "source_cohorts": source_cohorts,
            "historical_context": {
                "all_pair_human_slope": human["historical_all_pair_human_slope"],
                "track_b_posterior_slope": posterior[
                    "historical_track_b_posterior_slope"
                ],
            },
        },
        "pair_vectors": {
            "labels": list(metadata["selected_labels"]),
            "human_h": human_point.tolist(),
            "posterior_p_P": posterior_point.tolist(),
            "human_residual": residual_human.tolist(),
            "posterior_residual": residual_posterior.tolist(),
        },
    }

    integrity = {
        "bootstrap_samples": len(counts),
        "bootstrap_subjects": int(counts.shape[1]),
        "bootstrap_count_row_sum_max_abs_error": int(
            np.max(np.abs(np.sum(counts, axis=1) - subjects))
        ),
        "all_bootstrap_arrays_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in (
                    full_boot,
                    odd_boot,
                    even_boot,
                    human_slope_boot,
                    r_ph_boot,
                    r_hh_boot,
                    rho_h_boot,
                    eta_boot,
                    human_profile_boot,
                )
            )
        ),
        "human_slope_linear_identity_max_abs_error": float(
            abs(human_slope - float(np.mean(human_subject_slopes)))
        ),
        "human_residual_design_orthogonality_max_abs_error": float(
            np.max(np.abs(metadata["design"].T @ residual_human))
        ),
        "posterior_residual_design_orthogonality_max_abs_error": float(
            np.max(np.abs(metadata["design"].T @ residual_posterior))
        ),
        "minimum_pair_correlation_norm_product": float(
            min(
                np.min(norm_ph),
                np.min(norm_hh),
                np.linalg.norm(residual_posterior) * np.linalg.norm(residual_human),
                np.linalg.norm(residual_odd) * np.linalg.norm(residual_even),
            )
        ),
        "minimum_bootstrap_r_HH": float(np.min(r_hh_boot)),
        "minimum_bootstrap_rho_H": float(np.min(rho_h_boot)),
        "point_r_HH_positive": bool(r_hh > 0.0),
        "point_rho_H_positive": bool(rho_h > 0.0),
    }
    raw = {
        "human": {
            "participant_labels": human["participant_labels"],
            "full_nonlearned": human["full"].tolist(),
            "odd_nonlearned": human["odd"].tolist(),
            "even_nonlearned": human["even"].tolist(),
        },
        "posterior": {
            "subject_probability_nonlearned": posterior["subject_probability"].tolist(),
            "subject_slopes": posterior["subject_slopes"].tolist(),
        },
        "bootstrap": {
            "counts": counts.tolist(),
            "human_S_H": human_slope_boot.tolist(),
            "r_PH": r_ph_boot.tolist(),
            "r_HH": r_hh_boot.tolist(),
            "rho_H": rho_h_boot.tolist(),
            "eta_pair": eta_boot.tolist(),
            "human_distance_profile": human_profile_boot.tolist(),
        },
    }
    return statistics, integrity, raw


def decide(statistics: dict, all_gates_pass: bool) -> dict:
    if not all_gates_pass:
        return {
            "outcome": "noninterpretable",
            "distance_adequate": None,
            "pair_adequate": None,
            "neural_intervention_authorized": False,
        }
    distance = bool(statistics["primary"]["distance"]["adequate"])
    pair = bool(statistics["primary"]["pair"]["adequate"])
    outcomes = {
        (True, True): "comparator_externally_adequate",
        (False, False): "comparator_externally_inadequate",
        (True, False): "distance_adequate_pair_inadequate",
        (False, True): "pair_adequate_distance_inadequate",
    }
    outcome = outcomes[(distance, pair)]
    return {
        "outcome": outcome,
        "distance_adequate": distance,
        "pair_adequate": pair,
        "neural_intervention_authorized": False,
        "conditional_next_step": (
            "separately_register_P_T_to_g_N_generation_question"
            if outcome == "comparator_externally_adequate"
            else "stop_neural_intervention_and_reassess_comparator_theory"
        ),
    }


def evaluate(
    specification: dict,
    runtime: dict,
    source_validation: dict,
    prerequisite: dict,
) -> dict:
    protocol = load_ranking_protocol(
        _resolve(specification["registered_sources"]["liu_protocol"]["path"])
    )
    metadata = edge_metadata(specification, protocol)
    human, human_integrity = load_human_fields(specification, metadata)
    posterior, posterior_integrity = posterior_fields(specification, metadata)
    statistics, statistical_integrity, raw = adequacy_statistics(
        specification, human, posterior, metadata
    )
    tolerance = float(specification["statistical_contract"]["floating_tolerance"])
    minimum = float(specification["statistical_contract"]["minimum_vector_norm"])
    error_names = (
        "full_equals_half_average_max_abs_error",
        "human_benchmark_pair_mean_max_abs_error",
        "posterior_inverse_link_max_abs_error",
        "posterior_orientation_reversal_max_abs_error",
        "posterior_expected_rank_Hodge_max_abs_error",
        "fingerprint_anchor_cross_seed_max_abs_error",
        "posterior_slope_anchor_2106_max_abs_error",
        "posterior_slope_anchor_2107_max_abs_error",
        "human_slope_linear_identity_max_abs_error",
        "human_residual_design_orthogonality_max_abs_error",
        "posterior_residual_design_orthogonality_max_abs_error",
    )
    combined = {**human_integrity, **posterior_integrity, **statistical_integrity}
    all_gates_pass = bool(
        source_validation["passed"]
        and prerequisite["passed"]
        and runtime["active"]
        and runtime["cuda_available"]
        and runtime["torch_intraop_threads"] == 1
        and runtime["torch_interop_threads"] == 1
        and all(float(combined[name]) <= tolerance for name in error_names)
        and combined["preregistered_subjects"] == 40
        and combined["replication_subjects"] == 37
        and combined["combined_subjects"] == 77
        and combined["eligible_subjects"] == 77
        and combined["blocks"] == 10
        and combined["edges"] == 28
        and combined["observations_per_subject_pair"] == 10
        and combined["observations_per_half_subject_pair"] == 5
        and combined["all_human_values_binary"]
        and combined["all_human_arrays_finite"]
        and combined["human_benchmark_pair_order_matches"]
        and combined["human_benchmark_status_matches"]
        and combined["human_benchmark_eligible_subjects"] == 77
        and combined["subjects"] == 77
        and combined["nonlearned_pairs"] == 20
        and combined["all_posterior_probabilities_finite"]
        and 0.0 <= combined["posterior_probability_min"]
        and combined["posterior_probability_max"] <= 1.0
        and combined["bootstrap_samples"]
        == int(specification["statistical_contract"]["bootstrap_samples"])
        and combined["bootstrap_subjects"] == 77
        and combined["bootstrap_count_row_sum_max_abs_error"] == 0
        and combined["all_bootstrap_arrays_finite"]
        and combined["minimum_pair_correlation_norm_product"] > minimum
        and combined["minimum_bootstrap_r_HH"] > 0.0
        and combined["minimum_bootstrap_rho_H"] > 0.0
        and combined["point_r_HH_positive"]
        and combined["point_rho_H_positive"]
    )
    combined["passed"] = all_gates_pass
    decision = decide(statistics, all_gates_pass)
    return {
        "schema_version": 1,
        "audit_id": specification["audit_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "allocation_prerequisite": prerequisite,
        "edge_contract": {
            "pair_labels": list(metadata["selected_labels"]),
            "distances": metadata["selected_distances"].astype(int).tolist(),
            "distance_levels": metadata["distance_levels"].astype(int).tolist(),
            "distance_level_pair_counts": metadata["distance_counts"].tolist(),
            "distance_mean": metadata["distance_mean"],
            "distance_denominator": metadata["distance_denominator"],
        },
        "statistics": statistics,
        "integrity": combined,
        "decision": decision,
        "raw_arrays": raw,
        "claim_boundaries": {
            "outcome_contingent_interpretation": specification[
                "outcome_contingent_interpretation"
            ],
            "required": specification["reporting"]["required_claim_boundaries"],
        },
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen global-policy comparator-adequacy audit."
    )
    parser.add_argument("stage", choices=("evaluate",))
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument(
        "--protocol-repair", type=Path, default=DEFAULT_PROTOCOL_REPAIR_PATH
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    _canonical_paths(parsed)
    runtime = require_formal_runtime()
    registered_specification = load_json(parsed.specification)
    specification, repair_validation = apply_protocol_repair(
        registered_specification, parsed.protocol_repair
    )
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock, parsed.protocol_repair
    )
    git_freeze = require_pushed_freeze(
        (parsed.specification, parsed.protocol_repair, parsed.implementation_lock)
    )
    prerequisite = validate_prerequisite(specification)
    result = evaluate(specification, runtime, source_validation, prerequisite)
    result["protocol_repair_validation"] = repair_validation
    result["git_freeze_validation"] = git_freeze
    write_json_exclusive(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
