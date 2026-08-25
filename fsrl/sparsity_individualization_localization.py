"""Read-only localization of Liu sparsity individualization."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from .behavioral import kendall_tau_positions
from .confirmation import file_sha256
from .study_registry import legacy_identifier, registered_file_sha256, resolve_record
from .support_topology_transport import (
    ROOT,
    _json_values,
    bootstrap_counts,
    load_json,
    resolve_path,
    summarize_subjects,
    write_json_exclusive,
)

DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/liu_sparsity_individualization_localization_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/liu_sparsity_individualization_localization_v1.lock.json")
)
DEFAULT_RESULT_PATH = (
    resolve_record("results/liu_sparsity_individualization_localization_v1.json")
)
IMPLEMENTATION_SOURCES = {
    "runner": "fsrl/sparsity_individualization_localization.py",
    "tests": "tests/test_sparsity_individualization_localization.py",
}
REGISTRATION_COMMIT = "619500eaef3aa5a9bedb3ab74397d14cc7e81969"


def _registration(path: str) -> dict:
    return {"path": path, "sha256": file_sha256(resolve_record(path))}


def write_implementation_lock(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    lock = {
        "schema_version": 1,
        "experiment_id": "liu-sparsity-individualization-localization-v1",
        "implementation_status": "frozen_before_participant_level_decomposition",
        "registration_commit": REGISTRATION_COMMIT,
        "specification_sha256": file_sha256(specification_path),
        "implementation_sources": {
            name: _registration(path) for name, path in IMPLEMENTATION_SOURCES.items()
        },
    }
    write_json_exclusive(lock_path, lock)
    return lock


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(lock_path)
    registrations = {
        **specification["frozen_source"],
        "specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["specification_sha256"],
        },
        **lock["implementation_sources"],
    }
    checks = []
    for name, registration in registrations.items():
        path = resolve_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "expected": registration["sha256"],
                "observed": observed,
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"sparsity-individualization source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def participant_rows(cell: dict) -> list[dict]:
    rows = sorted(
        cell["metrics"]["sampled_behavior"]["subjects"],
        key=lambda row: int(row["subject"]),
    )
    return rows


def order_positions(rows: list[dict]) -> np.ndarray:
    positions = []
    for row in rows:
        order = np.asarray(row["subjective_order_high_to_low"], dtype=np.int64)
        position = np.empty(8, dtype=np.int64)
        position[order] = np.arange(8)
        positions.append(position)
    return np.asarray(positions)


def pairwise_tau_matrix(positions: np.ndarray) -> np.ndarray:
    subjects = len(positions)
    matrix = np.eye(subjects, dtype=np.float64)
    for first, second in combinations(range(subjects), 2):
        value = kendall_tau_positions(positions[first], positions[second])
        matrix[first, second] = value
        matrix[second, first] = value
    return matrix


def weighted_pairwise_tau(matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    subjects = matrix.shape[0]
    if matrix.shape != (subjects, subjects) or counts.shape[-1] != subjects:
        raise ValueError("pairwise matrix and bootstrap counts do not align")
    quadratic = np.einsum("...i,ij,...j->...", counts, matrix, counts, optimize=True)
    denominator = subjects * (subjects - 1)
    return (quadratic - subjects) / denominator


def ols_slope(values: np.ndarray, edge_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edge_counts = np.asarray(edge_counts, dtype=np.float64)
    if values.shape[0] != len(edge_counts):
        raise ValueError("density values and edge counts do not align")
    centered = edge_counts - np.mean(edge_counts)
    return centered @ values / float(centered @ centered)


def _bootstrap_draw_summary(point: float, draws: np.ndarray, interval: float) -> dict:
    alpha = 1.0 - interval
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(point),
        "bootstrap": {
            "mean": float(np.mean(draws)),
            "lower": float(lower),
            "upper": float(upper),
        },
    }


def extract_density_arrays(
    family: dict, edge_counts: list[int], subjects: int
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    arrays = {
        "stable_error_incidence": [],
        "stable_error_count": [],
        "truth_alignment": [],
        "exact_overall_error": [],
        "noncorrect_ranker_incidence": [],
    }
    positions = []
    for edge_count in edge_counts:
        cell = family["densities"][str(edge_count)]
        rows = participant_rows(cell)
        if [int(row["subject"]) for row in rows] != list(range(subjects)):
            raise RuntimeError("participant indices are not aligned")
        stable_counts = np.asarray(
            [row["stable_error_pair_counts"]["80"] for row in rows],
            dtype=np.float64,
        )
        arrays["stable_error_incidence"].append((stable_counts > 0).astype(float))
        arrays["stable_error_count"].append(stable_counts)
        arrays["truth_alignment"].append(
            np.asarray(
                [row["kendall_tau_subjective_to_true"] for row in rows],
                dtype=np.float64,
            )
        )
        arrays["noncorrect_ranker_incidence"].append(
            np.asarray(
                [row["ranking_class"] != "correct" for row in rows],
                dtype=np.float64,
            )
        )
        exact_accuracy = np.asarray(
            cell["metrics"]["conditions"]["intact"]["raw_subject"][
                "exact_decision_accuracy"
            ]["overall"],
            dtype=np.float64,
        )
        arrays["exact_overall_error"].append(1.0 - exact_accuracy)
        positions.append(order_positions(rows))
    return {name: np.asarray(values) for name, values in arrays.items()}, positions


def validate_source_structure(specification: dict, source: dict) -> dict:
    scope = specification["fixed_scope"]
    expected_families = set(scope["families"])
    expected_densities = {str(value) for value in scope["edge_counts"]}
    expected_seeds = {str(value) for value in scope["backbones"]}
    subjects = int(scope["participants_per_cell"])
    checks = []
    checks.append(
        {
            "name": "registered_source_outcome",
            "passed": source["decision"]["outcome"]
            == "SPARSITY_DEPENDENT_OR_UNRESOLVED",
        }
    )
    checks.append(
        {"name": "seed_set", "passed": set(source["seeds"]) == expected_seeds}
    )
    finite = True
    permutations = True
    aligned = True
    eligible = True
    decomposition_error = 0.0
    for seed in expected_seeds:
        families = source["seeds"][seed]["families"]
        aligned = aligned and set(families) == expected_families
        for family_id in expected_families:
            densities = families[family_id]["densities"]
            aligned = aligned and set(densities) == expected_densities
            for edge_count in expected_densities:
                cell = densities[edge_count]
                rows = participant_rows(cell)
                aligned = aligned and [int(row["subject"]) for row in rows] == list(
                    range(subjects)
                )
                permutations = permutations and all(
                    sorted(row["subjective_order_high_to_low"]) == list(range(8))
                    for row in rows
                )
                eligible = eligible and all(
                    float(row["overall_accuracy"]) >= 0.5 for row in rows
                )
                values = [
                    float(row["kendall_tau_subjective_to_true"]) for row in rows
                ] + [float(row["stable_error_pair_counts"]["80"]) for row in rows]
                finite = finite and bool(np.all(np.isfinite(values)))
                noncorrect = [row for row in rows if row["ranking_class"] != "correct"]
                stable = sum(row["stable_error_pair_counts"]["80"] > 0 for row in rows)
                conditional = (
                    sum(row["stable_error_pair_counts"]["80"] > 0 for row in noncorrect)
                    / len(noncorrect)
                    if noncorrect
                    else 0.0
                )
                identity = len(noncorrect) / subjects * conditional
                decomposition_error = max(
                    decomposition_error, abs(stable / subjects - identity)
                )
    checks.extend(
        [
            {"name": "participant_alignment", "passed": bool(aligned)},
            {
                "name": "subjective_orders_are_permutations",
                "passed": bool(permutations),
            },
            {"name": "all_participants_eligible", "passed": bool(eligible)},
            {"name": "registered_values_finite", "passed": bool(finite)},
            {
                "name": "stable_error_decomposition_identity",
                "max_abs_error": decomposition_error,
                "passed": decomposition_error <= 1e-15,
            },
        ]
    )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"sparsity source structure failed: {checks}")
    return {"passed": True, "checks": checks}


def analyze_family_seed(
    family: dict,
    edge_counts: list[int],
    subjects: int,
    counts: np.ndarray,
    interval: float,
) -> dict:
    arrays, positions = extract_density_arrays(family, edge_counts, subjects)
    x = np.asarray(edge_counts, dtype=np.float64)
    slopes = {name: ols_slope(values, x) for name, values in arrays.items()}
    summaries = {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in slopes.items()
    }
    tau_matrices = [pairwise_tau_matrix(value) for value in positions]
    tau_points = np.asarray(
        [np.mean(matrix[np.triu_indices(subjects, 1)]) for matrix in tau_matrices]
    )
    tau_draws = np.asarray(
        [weighted_pairwise_tau(matrix, counts) for matrix in tau_matrices]
    )
    tau_slope_point = float(ols_slope(tau_points, x))
    tau_slope_draws = ols_slope(tau_draws, x)
    tau_summary = _bootstrap_draw_summary(tau_slope_point, tau_slope_draws, interval)
    flags = {
        "stable_error_incidence_decreases": bool(
            summaries["stable_error_incidence"]["bootstrap"]["upper"] < 0.0
        ),
        "all_participant_pairwise_tau_increases": bool(
            tau_summary["bootstrap"]["lower"] > 0.0
        ),
    }
    conditional = {
        str(edge_count): family["densities"][str(edge_count)]["metrics"][
            "individualized"
        ]
        for edge_count in edge_counts
    }
    return {
        "raw_subject_slopes": {
            name: _json_values(values) for name, values in slopes.items()
        },
        "participant_slope_summaries": summaries,
        "all_participant_pairwise_tau_by_density": {
            str(edge_count): float(value)
            for edge_count, value in zip(edge_counts, tau_points, strict=True)
        },
        "all_participant_pairwise_tau_slope": tau_summary,
        "historical_conditional_metrics": conditional,
        "flags": flags,
    }


def analyze_e10_family_contrast(
    first: dict,
    second: dict,
    subjects: int,
    counts: np.ndarray,
    interval: float,
) -> dict:
    edge_counts = [10]
    first_arrays, first_positions = extract_density_arrays(first, edge_counts, subjects)
    second_arrays, second_positions = extract_density_arrays(
        second, edge_counts, subjects
    )
    summaries = {
        name: summarize_subjects(
            second_arrays[name][0] - first_arrays[name][0],
            counts,
            interval=interval,
        )
        for name in first_arrays
    }
    first_matrix = pairwise_tau_matrix(first_positions[0])
    second_matrix = pairwise_tau_matrix(second_positions[0])
    upper = np.triu_indices(subjects, 1)
    point = float(np.mean(second_matrix[upper]) - np.mean(first_matrix[upper]))
    draws = weighted_pairwise_tau(second_matrix, counts) - weighted_pairwise_tau(
        first_matrix, counts
    )
    return {
        "balanced_branched_minus_liu_cycle": summaries,
        "all_participant_pairwise_tau_difference": _bootstrap_draw_summary(
            point, draws, interval
        ),
    }


def outcome(analyses: list[dict]) -> str:
    stable = [row["flags"]["stable_error_incidence_decreases"] for row in analyses]
    tau = [row["flags"]["all_participant_pairwise_tau_increases"] for row in analyses]
    if all(stable) and all(tau):
        return "DENSITY_LINKED_INDIVIDUALIZATION_CONVERGENCE"
    if all(tau):
        return "ORDER_CONVERGENCE_WITHOUT_REPLICATED_STABLE_ERROR_LOSS"
    if all(stable):
        return "STABLE_ERROR_LOSS_WITHOUT_REPLICATED_ORDER_CONVERGENCE"
    if any(stable) or any(tau):
        return "FAMILY_OR_BACKBONE_SPECIFIC_INDIVIDUALIZATION_CHANGE"
    return "NO_REPLICATED_DENSITY_LOCALIZATION"


def evaluate(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    source_validation = validate_sources(specification_path, lock_path)
    source = load_json(resolve_path(specification["frozen_source"]["result"]["path"]))
    structure_validation = validate_source_structure(specification, source)
    scope = specification["fixed_scope"]
    bootstrap = specification["bootstrap"]
    edge_counts = [int(value) for value in scope["edge_counts"]]
    subjects = int(scope["participants_per_cell"])
    interval = float(bootstrap["interval"])
    seeds = {}
    all_analyses = []
    for seed in scope["backbones"]:
        families = {}
        for family_index, family_id in enumerate(scope["families"], start=1):
            seed_value = 920000 + 1000 * family_index + 100 * int(seed)
            counts = bootstrap_counts(
                np.random.default_rng(seed_value),
                int(bootstrap["samples"]),
                subjects,
            )
            analysis = analyze_family_seed(
                source["seeds"][str(seed)]["families"][family_id],
                edge_counts,
                subjects,
                counts,
                interval,
            )
            analysis["bootstrap_seed"] = seed_value
            families[family_id] = analysis
            all_analyses.append(analysis)
        contrast_seed = 930000 + 100 * int(seed)
        contrast_counts = bootstrap_counts(
            np.random.default_rng(contrast_seed),
            int(bootstrap["samples"]),
            subjects,
        )
        contrast = analyze_e10_family_contrast(
            source["seeds"][str(seed)]["families"][scope["families"][0]],
            source["seeds"][str(seed)]["families"][scope["families"][1]],
            subjects,
            contrast_counts,
            interval,
        )
        contrast["bootstrap_seed"] = contrast_seed
        seeds[str(seed)] = {"families": families, "E10_family_contrast": contrast}
    return {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "registration_status": specification["registration_status"],
        "source_validation": source_validation,
        "structure_validation": structure_validation,
        "seeds": seeds,
        "decision": {"outcome": outcome(all_analyses)},
        "registered_primary_estimands": specification["primary_estimands"],
        "registered_outcome_tree": specification["outcome_tree"],
    }


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--write-lock", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.write_lock:
        write_implementation_lock(parsed.specification, parsed.implementation_lock)
        return 0
    result = evaluate(parsed.specification, parsed.implementation_lock)
    write_json_exclusive(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
