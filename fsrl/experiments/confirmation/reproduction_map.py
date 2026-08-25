"""Read-only Liu phenomenon map for the frozen v2.4 computational model."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from fsrl.analysis.behavioral import kendall_tau_positions
from fsrl.experiments.human.benchmark import load_human_cohort
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT
SPECIFICATION_PATH = resolve_record(
    "benchmarks/model_behavior_reproduction_map_v1.json"
)
IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/model_behavior_reproduction_map_v1.lock.json"
)
OUTPUT_PATH = resolve_record("results/model_behavior_reproduction_map_v1.json")
PROTOCOL_PATH = resolve_record("benchmarks/liu_v2.json")

IMPLEMENTATION_SOURCES = {
    "map_runner": "fsrl/model_behavior_reproduction_map.py",
    "map_tests": "tests/test_model_behavior_reproduction_map.py",
}


def validate_sources(
    specification_path: Path = SPECIFICATION_PATH,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    if not (
        lock.get("schema_version") == 1
        and lock.get("map_id") == specification.get("map_id")
        and lock.get("freeze_status")
        == "implementation_frozen_after_protocol_commit_and_before_map_execution"
        and lock.get("specification_sha256") == file_sha256(specification_path)
        and set(lock.get("implementation_sources", {})) == set(IMPLEMENTATION_SOURCES)
        and all(
            lock["implementation_sources"][name].get("path") == path
            for name, path in IMPLEMENTATION_SOURCES.items()
        )
    ):
        raise RuntimeError("behavior reproduction map implementation lock mismatch")
    registrations = {
        "map_specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["specification_sha256"],
        },
        **specification["registered_sources"],
        **lock["implementation_sources"],
    }
    checks = []
    for name, registration in registrations.items():
        path = resolve_record(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": registration["path"],
                "expected": registration["sha256"],
                "observed": observed,
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"behavior reproduction map source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _pair_indices(pair: list[int] | list[str]) -> tuple[int, int]:
    if isinstance(pair[0], str):
        return ord(pair[0]) - ord("A"), ord(pair[1]) - ord("A")
    return int(pair[0]), int(pair[1])


def position_profile(
    pair_rows: list[dict],
    value_key: str,
    *,
    learned: bool | None = None,
) -> np.ndarray:
    values: list[list[float]] = [[] for _ in range(8)]
    for row in pair_rows:
        if learned is not None and bool(row["learned"]) != learned:
            continue
        first, second = _pair_indices(row["pair"])
        value = float(row[value_key])
        values[first].append(value)
        values[second].append(value)
    if any(not item_values for item_values in values):
        raise RuntimeError("serial-position profile has an empty item position")
    return np.asarray([np.mean(item_values) for item_values in values])


def endpoint_statistics(profile: np.ndarray) -> dict:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.shape != (8,):
        raise ValueError("serial-position profile must have eight entries")
    interior = float(np.mean(profile[1:7]))
    return {
        "profile_low_to_high": [float(value) for value in profile],
        "interior_mean": interior,
        "mean_endpoint_contrast": float(np.mean(profile[[0, 7]]) - interior),
        "minimum_endpoint_advantage": float(min(profile[0], profile[7]) - interior),
        "both_endpoints_above_interior": bool(
            profile[0] > interior and profile[7] > interior
        ),
    }


def _interval(values: np.ndarray) -> dict:
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def bootstrap_mean_interval(values: np.ndarray, *, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values), np.full(len(values), 1.0 / len(values)), size=samples
    )
    draws = counts @ values / len(values)
    return {"point": float(np.mean(values)), "interval": _interval(draws)}


def _rank_positions(order_high_to_low: list[int]) -> np.ndarray:
    if sorted(order_high_to_low) != list(range(8)):
        raise RuntimeError("stored Hodge order is not a permutation")
    positions = np.empty(8, dtype=np.int64)
    for position, item in enumerate(order_high_to_low):
        positions[item] = position
    return positions


def mean_pairwise_tau_interval(
    orders: list[list[int]], *, samples: int, seed: int
) -> dict:
    positions = np.asarray([_rank_positions(order) for order in orders])
    count = len(positions)
    if count < 2:
        raise RuntimeError("at least two rankings are required")
    matrix = np.eye(count, dtype=np.float64)
    for first, second in combinations(range(count), 2):
        value = kendall_tau_positions(positions[first], positions[second])
        matrix[first, second] = value
        matrix[second, first] = value
    point = float(np.mean(matrix[np.triu_indices(count, 1)]))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(count, np.full(count, 1.0 / count), size=samples)
    quadratic = np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True)
    diagonal = np.sum(counts, axis=1)
    draws = 0.5 * (quadratic - diagonal) / (count * (count - 1) / 2.0)
    return {"point": point, "interval": _interval(draws)}


def classify_network_flags(flags: dict[str, dict[str, bool]]) -> str:
    if not all(flag["qualitative"] for flag in flags.values()):
        return "not_reproduced"
    if all(flag["calibration"] for flag in flags.values()):
        return "reproduced"
    return "qualitatively_reproduced_quantitatively_mismatched"


def _inside(value: float, interval: dict) -> bool:
    return bool(interval["lower"] <= value <= interval["upper"])


def _human_subject_serial(subject: dict, learned_mask: np.ndarray) -> dict:
    pairs = tuple(combinations(range(8), 2))
    pair_accuracy = np.asarray(subject["pair_accuracy"], dtype=np.float64)
    rows = [
        {
            "pair": list(pair),
            "learned": bool(learned_mask[index]),
            "value": float(pair_accuracy[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    return {
        name: endpoint_statistics(position_profile(rows, "value", learned=selector))
        for name, selector in (
            ("all", None),
            ("learned", True),
            ("nonlearned", False),
        )
    }


def model_record(
    result: dict,
    seed: str,
    human_intervals: dict,
    human_serial_interval: dict,
    human_tau_interval: dict,
    specification: dict,
) -> dict:
    behavior = result["seeds"][seed]["behavior"]["dual_access_matched"]
    summary = behavior["summary"]
    subjects = behavior["subjects"]
    bootstrap_samples = int(specification["bootstrap"]["samples"])
    bootstrap_seed = int(specification["bootstrap"]["human_new_estimands_seed"]) + int(
        seed
    )
    profiles = {
        name: endpoint_statistics(
            position_profile(behavior["pairs"], "mean_accuracy_all", learned=selector)
        )
        for name, selector in (
            ("all", None),
            ("learned", True),
            ("nonlearned", False),
        )
    }
    eligible = [subject for subject in subjects if subject["overall_accuracy"] >= 0.5]
    analysis = [
        subject for subject in eligible if subject["ranking_class"] != "correct"
    ]
    stable_values = np.asarray(
        [subject["stable_error_pair_counts"]["80"] > 0 for subject in analysis],
        dtype=np.float64,
    )
    class_values = {
        name: np.asarray(
            [subject["ranking_class"] == name for subject in eligible],
            dtype=np.float64,
        )
        for name in (
            "correct",
            "self_consistent_incorrect",
            "self_inconsistent",
        )
    }
    tau = mean_pairwise_tau_interval(
        [subject["subjective_order_high_to_low"] for subject in analysis],
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    if not np.isclose(tau["point"], summary["mean_inter_subject_kendall_tau"]):
        raise RuntimeError("stored inter-subject Kendall tau does not reconstruct")
    stable = bootstrap_mean_interval(
        stable_values, samples=bootstrap_samples, seed=bootstrap_seed + 2
    )
    classes = {
        name: bootstrap_mean_interval(
            values, samples=bootstrap_samples, seed=bootstrap_seed + offset
        )
        for offset, (name, values) in enumerate(class_values.items(), 3)
    }
    beta_counts = summary["beta_pair_class_counts_analysis"]
    metrics = {
        "learned_accuracy": {
            "point": float(summary["learned_accuracy"]),
            "interval": behavior["participant_bootstrap"]["learned_accuracy"][
                "bootstrap"
            ],
        },
        "nonlearned_accuracy": {
            "point": float(summary["nonlearned_accuracy"]),
            "interval": behavior["participant_bootstrap"]["nonlearned_accuracy"][
                "bootstrap"
            ],
        },
        "symbolic_distance_effect": summary["symbolic_distance_slope"],
        "serial_position_effect": {
            "profiles": profiles,
            "participant_interval": "unavailable_from_frozen_model_artifact",
        },
        "difficult_pair_bimodality": beta_counts,
        "stable_within_subject_errors": stable,
        "self_consistent_vs_inconsistent_errors": {
            "self_consistent_incorrect": classes["self_consistent_incorrect"],
            "self_inconsistent": classes["self_inconsistent"],
        },
        "hodge_reconstructed_subjective_ranking": {
            "correct_ranker": classes["correct"],
            "all_orders_are_permutations": all(
                sorted(subject["subjective_order_high_to_low"]) == list(range(8))
                for subject in eligible
            ),
        },
        "inter_subject_ranking_diversity": tau,
    }
    flags = {
        "learned_accuracy": {
            "qualitative": metrics["learned_accuracy"]["point"] > 0.5,
            "calibration": _inside(
                metrics["learned_accuracy"]["point"],
                human_intervals["learned_accuracy"],
            ),
        },
        "nonlearned_accuracy": {
            "qualitative": metrics["nonlearned_accuracy"]["point"] > 0.5,
            "calibration": _inside(
                metrics["nonlearned_accuracy"]["point"],
                human_intervals["nonlearned_accuracy"],
            ),
        },
        "symbolic_distance_effect": {
            "qualitative": metrics["symbolic_distance_effect"]["mean"] > 0.0
            and metrics["symbolic_distance_effect"]["p_vs_zero"] < 0.05,
            "calibration": _inside(
                metrics["symbolic_distance_effect"]["mean"],
                human_intervals["symbolic_distance_slope"],
            ),
        },
        "serial_position_effect": {
            "qualitative": profiles["all"]["both_endpoints_above_interior"],
            "calibration": _inside(
                profiles["all"]["mean_endpoint_contrast"], human_serial_interval
            ),
        },
        "difficult_pair_bimodality": {
            "qualitative": beta_counts["bimodal"] >= 15
            and beta_counts["ordinary_unimodal"] == 0
            and beta_counts["low_accuracy"] == 0,
            "calibration": beta_counts["bimodal"] >= 15
            and beta_counts["ordinary_unimodal"] == 0
            and beta_counts["low_accuracy"] == 0,
        },
        "stable_within_subject_errors": {
            "qualitative": stable["point"] >= 0.80,
            "calibration": _inside(
                stable["point"], human_intervals["stable_error_80_analysis_proportion"]
            ),
        },
        "self_consistent_vs_inconsistent_errors": {
            "qualitative": classes["self_consistent_incorrect"]["point"]
            > classes["self_inconsistent"]["point"],
            "calibration": _inside(
                classes["self_consistent_incorrect"]["point"],
                human_intervals["self_consistent_incorrect_proportion"],
            )
            and _inside(
                classes["self_inconsistent"]["point"],
                human_intervals["self_inconsistent_proportion"],
            ),
        },
        "hodge_reconstructed_subjective_ranking": {
            "qualitative": metrics["hodge_reconstructed_subjective_ranking"][
                "all_orders_are_permutations"
            ]
            and classes["correct"]["point"] < 0.5,
            "calibration": _inside(
                classes["correct"]["point"],
                human_intervals["correct_ranker_proportion"],
            ),
        },
        "inter_subject_ranking_diversity": {
            "qualitative": tau["point"] < 0.80,
            "calibration": _inside(tau["point"], human_tau_interval),
        },
    }
    return {
        "seed": int(seed),
        "eligible_subjects": len(eligible),
        "analysis_subjects_excluding_correct_rankers": len(analysis),
        "metrics": metrics,
        "flags": flags,
    }


def build_map(
    specification_path: Path = SPECIFICATION_PATH,
    implementation_lock_path: Path = IMPLEMENTATION_LOCK_PATH,
) -> dict:
    source_validation = validate_sources(specification_path, implementation_lock_path)
    specification = load_json(specification_path)
    sources = specification["registered_sources"]
    human_benchmark = load_json(resolve_record(sources["human_benchmark"]["path"]))
    model_result = load_json(resolve_record(sources["model_result"]["path"]))
    protocol = load_ranking_protocol(PROTOCOL_PATH)
    human_subjects = []
    for name, cohort in (
        ("human_preregistered_trials", "preregistered"),
        ("human_replication_trials", "replication"),
    ):
        registration = sources[name]
        human_subjects.extend(
            load_human_cohort(
                resolve_record(registration["path"]),
                cohort,
                protocol,
                expected_sha256=registration["sha256"],
            )
        )
    model_seeds = set(model_result.get("seeds", {}))
    identity_gates = {
        "human_benchmark_identity": human_benchmark.get("status")
        == "source_recomputed_and_paper_checks_reproduced"
        and human_benchmark["combined"]["eligible_subjects"] == 77
        and human_benchmark["combined"]["excluded_below_chance"] == 0,
        "human_raw_identity": len(human_subjects) == 77
        and all(subject["overall_accuracy"] >= 0.5 for subject in human_subjects),
        "model_confirmation_identity": model_result["decision"]["outcome"]
        == "fresh_backbone_confirmation_pass"
        and model_result["source_validation"]["passed"]
        and model_result["artifact_validation"]["passed"],
        "exact_mandatory_networks": model_seeds == {"2104", "2105"},
        "dual_access_matched_complete": all(
            model_result["seeds"][seed]["behavior"]["dual_access_matched"]["summary"][
                "eligible_subjects"
            ]
            == 77
            and model_result["seeds"][seed]["behavior"]["dual_access_matched"][
                "summary"
            ]["excluded_below_chance"]
            == 0
            and len(
                model_result["seeds"][seed]["behavior"]["dual_access_matched"][
                    "subjects"
                ]
            )
            == 77
            for seed in model_seeds
        ),
    }
    if not all(identity_gates.values()):
        raise RuntimeError(
            f"behavior reproduction map identity gate failed: {identity_gates}"
        )

    bootstrap_samples = int(specification["bootstrap"]["samples"])
    human_seed = int(specification["bootstrap"]["human_new_estimands_seed"])
    learned_mask = np.asarray(
        [pair in protocol.learned_pairs for pair in combinations(range(8), 2)]
    )
    human_serial_by_subject = [
        _human_subject_serial(subject, learned_mask) for subject in human_subjects
    ]
    human_serial_contrasts = np.asarray(
        [record["all"]["mean_endpoint_contrast"] for record in human_serial_by_subject]
    )
    human_serial = bootstrap_mean_interval(
        human_serial_contrasts, samples=bootstrap_samples, seed=human_seed
    )
    human_serial["profiles"] = {
        name: endpoint_statistics(
            np.mean(
                np.asarray(
                    [
                        record[name]["profile_low_to_high"]
                        for record in human_serial_by_subject
                    ]
                ),
                axis=0,
            )
        )
        for name in ("all", "learned", "nonlearned")
    }
    human_analysis = [
        subject for subject in human_subjects if subject["ranking_class"] != "correct"
    ]
    human_tau = mean_pairwise_tau_interval(
        [subject["subjective_order_high_to_low"] for subject in human_analysis],
        samples=bootstrap_samples,
        seed=human_seed + 1,
    )
    if not np.isclose(
        human_tau["point"],
        human_benchmark["combined"]["mean_inter_subject_kendall_tau"],
    ):
        raise RuntimeError("human inter-subject Kendall tau does not reconstruct")
    human_intervals = {
        name: {
            "lower": float(values["lower"]),
            "upper": float(values["upper"]),
        }
        for name, values in human_benchmark["bootstrap"]["metrics"].items()
    }
    human = {
        "subjects": 77,
        "analysis_subjects_excluding_correct_rankers": len(human_analysis),
        "learned_accuracy": human_benchmark["combined"]["learned_accuracy"],
        "nonlearned_accuracy": human_benchmark["combined"]["nonlearned_accuracy"],
        "symbolic_distance_effect": human_benchmark["combined"][
            "symbolic_distance_slope"
        ],
        "serial_position_effect": human_serial,
        "difficult_pair_bimodality": human_benchmark["combined"][
            "published_figure_checks"
        ]["beta_pair_class_counts"],
        "stable_within_subject_errors": {
            "point": human_benchmark["combined"]["stable_error_subject_prevalence"][
                "80"
            ]["analysis_proportion"],
            "interval": human_intervals["stable_error_80_analysis_proportion"],
        },
        "self_consistent_vs_inconsistent_errors": {
            "counts": human_benchmark["combined"]["ranking_class_counts"],
            "self_consistent_incorrect_interval": human_intervals[
                "self_consistent_incorrect_proportion"
            ],
            "self_inconsistent_interval": human_intervals[
                "self_inconsistent_proportion"
            ],
        },
        "hodge_reconstructed_subjective_ranking": {
            "correct_ranker_proportion": human_benchmark["combined"][
                "ranking_class_counts"
            ]["correct"]
            / 77.0,
            "interval": human_intervals["correct_ranker_proportion"],
        },
        "inter_subject_ranking_diversity": human_tau,
    }
    networks = {
        seed: model_record(
            model_result,
            seed,
            human_intervals,
            human_serial["interval"],
            human_tau["interval"],
            specification,
        )
        for seed in ("2104", "2105")
    }
    rows = []
    for registered in specification["rows"]:
        row_id = registered["row_id"]
        flags = {seed: network["flags"][row_id] for seed, network in networks.items()}
        rows.append(
            {
                "row_id": row_id,
                "status": classify_network_flags(flags),
                "network_flags": flags,
            }
        )
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "reproduced",
            "qualitatively_reproduced_quantitatively_mismatched",
            "not_reproduced",
        )
    }
    result = {
        "schema_version": 1,
        "map_id": specification["map_id"],
        "execution_mode": "read_only_existing_artifacts_no_checkpoint_load",
        "source_validation": source_validation,
        "identity_gates": identity_gates,
        "human_reference": human,
        "networks": networks,
        "rows": rows,
        "summary": {"status_counts": status_counts, "rows": len(rows)},
        "claim_boundary": specification["outcome_contract"]["interpretation"],
        "next_step": specification["outcome_contract"]["next_step"],
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=SPECIFICATION_PATH)
    parser.add_argument(
        "--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args(argv)
    result = build_map(arguments.specification, arguments.implementation_lock)
    write_json_exclusive(arguments.output, result)
    print(
        json.dumps(
            {
                "path": str(arguments.output),
                "sha256": file_sha256(arguments.output),
                "status_counts": result["summary"]["status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
