"""Pure synthetic validation for the magnitude-placement behavior contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import integrate, stats

from fsrl.infrastructure.study_registry import legacy_identifier, resolve_record
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/magnitude_placement_behavior_v1.json"
)
DEFAULT_RESULT_PATH = resolve_record(
    "results/magnitude_placement_behavior_v1_validation.json"
)
VALIDATOR_TEST_PATH = (
    ROOT / "tests" / "experiments" / "human" / "test_magnitude_behavior.py"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_exclusive(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def canonical_pair(first: str, second: str, item_roles: list[str]) -> str:
    order = {item: index for index, item in enumerate(item_roles)}
    left, right = sorted((first, second), key=order.__getitem__)
    return f"{left}-{right}"


def parse_relation(relation: str) -> tuple[str, str]:
    parts = relation.split(">")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid support relation: {relation}")
    return parts[0], parts[1]


def planned_power(n: int, standard_deviation: float, sesoi: float) -> dict:
    """Exact registered normal-theory t/TOST planning power."""
    if n <= 4 or standard_deviation <= 0.0 or sesoi <= 0.0:
        raise ValueError("invalid power-planning input")
    degrees_freedom = n - 4
    critical = float(stats.t.ppf(0.95, degrees_freedom))
    noncentrality = sesoi * math.sqrt(n) / standard_deviation
    directional = 1.0 - float(stats.nct.cdf(critical, degrees_freedom, noncentrality))

    upper = float(stats.chi2.ppf(1.0 - 1e-12, degrees_freedom))

    def integrand(variance_draw: float) -> float:
        standardized_bound = noncentrality - critical * math.sqrt(
            variance_draw / degrees_freedom
        )
        conditional = max(0.0, 2.0 * float(stats.norm.cdf(standardized_bound)) - 1.0)
        return conditional * float(stats.chi2.pdf(variance_draw, degrees_freedom))

    equivalence = float(
        integrate.quad(integrand, 0.0, upper, epsabs=1e-10, limit=300)[0]
    )
    return {
        "directional_at_positive_sesoi": directional,
        "equivalence_at_zero": equivalence,
    }


def _assignment_geometry(specification: dict) -> dict:
    frozen = specification["frozen_assignments"]
    items = list(frozen["item_roles"])
    support_relations = list(frozen["support_relation_order"])
    assignments = {}
    for name in ("assignment_A", "assignment_B"):
        assignment = frozen[name]
        levels = {item: int(value) for item, value in assignment["rank_levels"].items()}
        order = list(assignment["low_to_high_order"])
        derived_gaps = {}
        signs = {}
        for relation in support_relations:
            higher, lower = parse_relation(relation)
            derived_gaps[relation] = levels[higher] - levels[lower]
            signs[relation] = int(np.sign(derived_gaps[relation]))
        assignments[name] = {
            "low_to_high_order": order,
            "rank_levels": levels,
            "derived_support_gaps": derived_gaps,
            "registered_support_gaps": {
                relation: int(value)
                for relation, value in assignment["support_gaps"].items()
            },
            "support_signs": signs,
        }

    all_pairs = [
        canonical_pair(first, second, items) for first, second in combinations(items, 2)
    ]
    learned = sorted(
        canonical_pair(*parse_relation(relation), items)
        for relation in support_relations
    )
    flips = []
    same_nonlearned = []
    levels_a = assignments["assignment_A"]["rank_levels"]
    levels_b = assignments["assignment_B"]["rank_levels"]
    for pair in all_pairs:
        if pair in learned:
            continue
        first, second = pair.split("-")
        sign_a = int(np.sign(levels_a[first] - levels_a[second]))
        sign_b = int(np.sign(levels_b[first] - levels_b[second]))
        target = flips if sign_a != sign_b else same_nonlearned
        target.append(pair)

    distance_change = {}
    for pair in same_nonlearned:
        first, second = pair.split("-")
        distance_change[pair] = abs(levels_b[first] - levels_b[second]) - abs(
            levels_a[first] - levels_a[second]
        )
    magnitude_change = {}
    gaps_a = assignments["assignment_A"]["derived_support_gaps"]
    gaps_b = assignments["assignment_B"]["derived_support_gaps"]
    for relation in support_relations:
        pair = canonical_pair(*parse_relation(relation), items)
        magnitude_change[pair] = gaps_b[relation] - gaps_a[relation]

    return {
        "item_roles": items,
        "support_relations": support_relations,
        "assignments": assignments,
        "all_pairs": all_pairs,
        "learned_pairs": learned,
        "nonlearned_order_flip_pairs": sorted(flips),
        "nonlearned_same_direction_pairs": sorted(same_nonlearned),
        "same_direction_distance_change_B_minus_A": distance_change,
        "learned_magnitude_change_B_minus_A": magnitude_change,
    }


def validate_specification(specification: dict) -> dict:
    """Validate only frozen synthetic geometry, task counts, and power equations."""
    frozen = specification["frozen_assignments"]
    design = specification["experimental_design"]
    planning = specification["sample_size_plan"]
    estimands = specification["registered_estimands"]
    geometry = _assignment_geometry(specification)
    items = geometry["item_roles"]
    support_relations = geometry["support_relations"]
    expected_levels = list(range(len(items)))
    assignments = geometry["assignments"]

    assignment_gates = {}
    for name, assignment in assignments.items():
        assignment_gates[f"{name}_rank_level_permutation"] = (
            sorted(assignment["rank_levels"].values()) == expected_levels
        )
        assignment_gates[f"{name}_order_matches_levels"] = assignment[
            "low_to_high_order"
        ] == sorted(items, key=assignment["rank_levels"].__getitem__)
        assignment_gates[f"{name}_registered_gaps_match_levels"] = (
            assignment["registered_support_gaps"] == assignment["derived_support_gaps"]
        )
        assignment_gates[f"{name}_all_support_signs_positive"] = all(
            value == 1 for value in assignment["support_signs"].values()
        )
        assignment_gates[f"{name}_magnitude_multiset"] = sorted(
            assignment["derived_support_gaps"].values()
        ) == list(frozen["magnitude_multiset"])
    assignment_gates["support_signs_identical"] = (
        assignments["assignment_A"]["support_signs"]
        == assignments["assignment_B"]["support_signs"]
    )
    assignment_gates["all_support_magnitudes_change"] = all(
        assignments["assignment_A"]["derived_support_gaps"][relation]
        != assignments["assignment_B"]["derived_support_gaps"][relation]
        for relation in support_relations
    )

    registered_partition = frozen["pair_partition"]
    registered_learned = sorted(registered_partition["learned"])
    registered_flips = sorted(registered_partition["nonlearned_order_flip"])
    registered_same = sorted(registered_partition["nonlearned_same_direction"])
    assignment_gates["learned_pair_set"] = (
        registered_learned == geometry["learned_pairs"]
    )
    assignment_gates["order_flip_pair_set"] = (
        registered_flips == geometry["nonlearned_order_flip_pairs"]
    )
    assignment_gates["same_direction_pair_set"] = (
        registered_same == geometry["nonlearned_same_direction_pairs"]
    )
    assignment_gates["pair_partition_complete"] = (
        len(registered_learned) == 8
        and len(registered_flips) == 7
        and len(registered_same) == 13
        and sorted(registered_learned + registered_flips + registered_same)
        == sorted(geometry["all_pairs"])
    )
    registered_distance_change = {
        pair: int(value)
        for pair, value in frozen["same_direction_distance_change_B_minus_A"].items()
    }
    registered_magnitude_change = {
        pair: int(value)
        for pair, value in frozen["learned_magnitude_change_B_minus_A"].items()
    }
    assignment_gates["distance_change_vector"] = (
        registered_distance_change
        == geometry["same_direction_distance_change_B_minus_A"]
    )
    assignment_gates["magnitude_change_vector"] = (
        registered_magnitude_change == geometry["learned_magnitude_change_B_minus_A"]
    )
    assignment_gates["distance_change_centered"] = (
        sum(registered_distance_change.values()) == 0
    )
    assignment_gates["magnitude_change_centered"] = (
        sum(registered_magnitude_change.values()) == 0
    )

    support = design["support_phase"]
    query = design["query_phase"]
    cells = design["counterbalance"]["cells"]
    task_gates = {
        "two_list_within_participant_design": design["design"].startswith(
            "Within-participant two-list crossover"
        ),
        "disjoint_image_sets_registered": "disjoint image sets" in design["design"],
        "four_counterbalance_cells": len(cells) == 4 and len(set(cells)) == 4,
        "support_schedule_complete": support["relations_per_block"] == 8
        and support["blocks"] == 4
        and support["trials"] == 32
        and support["trials"] == support["relations_per_block"] * support["blocks"],
        "support_is_passive_without_feedback": support["choice_required"] is False
        and support["feedback"] is False,
        "query_schedule_complete": query["pairs_per_block"] == 28
        and query["blocks"] == 10
        and query["trials"] == 280
        and query["trials"] == query["pairs_per_block"] * query["blocks"],
        "query_has_no_feedback": query["feedback"] is False,
        "all_pairs_are_queried": query["pairs_per_block"] == len(geometry["all_pairs"]),
    }

    sesoi = estimands["sesoi"]
    axis_inputs = {
        "delta_flip": (
            float(planning["null_standard_deviation"]["delta_flip"]),
            float(sesoi["delta_flip_probability"]),
        ),
        "beta_conf": (
            float(planning["null_standard_deviation"]["beta_conf"]),
            float(sesoi["beta_conf_probability_per_rank_gap_unit"]),
        ),
        "beta_learned": (
            float(planning["null_standard_deviation"]["beta_learned"]),
            float(sesoi["beta_learned_probability_per_displayed_gap_unit"]),
        ),
    }
    n = int(planning["minimum_analyzable_n"])
    power_n = {
        axis: planned_power(n, standard_deviation, margin)
        for axis, (standard_deviation, margin) in axis_inputs.items()
    }
    power_previous = {
        axis: planned_power(n - 4, standard_deviation, margin)
        for axis, (standard_deviation, margin) in axis_inputs.items()
    }
    registered_power = planning["power_at_n_100"]
    power_match = all(
        math.isclose(
            power_n[axis][metric],
            float(registered_power[axis][metric]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        for axis in axis_inputs
        for metric in ("directional_at_positive_sesoi", "equivalence_at_zero")
    )
    analysis_gates = {
        "positive_sesoi": all(margin > 0.0 for _, margin in axis_inputs.values()),
        "n_is_100": n == 100,
        "balanced_four_cell_n": n % 4 == 0
        and planning["analyzable_per_counterbalance_cell"] == n // 4,
        "n_100_power_matches_registration": power_match,
        "n_100_all_axis_power_at_least_0_90": min(
            value for axis in power_n.values() for value in axis.values()
        )
        >= 0.90,
        "n_96_fails_all_axis_power_rule": min(
            value for axis in power_previous.values() for value in axis.values()
        )
        < 0.90,
        "equivalence_priority_registered": "Equivalence has practical-interpretation priority"
        in estimands["axis_flags"]["precedence"],
        "fixed_hierarchical_outcome_tree": "step_1_global_order"
        in specification["hierarchical_outcome_tree"]
        and "step_2_nonlearned_confidence" in specification["hierarchical_outcome_tree"]
        and "step_3_learned_relations" in specification["hierarchical_outcome_tree"],
    }

    gates = {
        "assignment": assignment_gates,
        "task": task_gates,
        "analysis": analysis_gates,
    }
    passed = all(value for group in gates.values() for value in group.values())
    return {
        "schema_version": 1,
        "study_id": specification["study_id"],
        "status": "synthetic_validation_passed" if passed else "noninterpretable",
        "passed": passed,
        "data_files_opened": [],
        "derived": {
            "assignment_geometry": geometry,
            "power_at_n_100": power_n,
            "power_at_n_96": power_previous,
        },
        "gates": gates,
    }


def run_validation(specification_path: Path, output_path: Path) -> dict:
    specification = load_json(specification_path)
    result = validate_specification(specification)
    result["provenance"] = {
        "specification": {
            "path": str(specification_path),
            "sha256": file_sha256(specification_path),
        },
        "validator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "validator_tests": {
            "path": legacy_identifier(VALIDATOR_TEST_PATH),
            "sha256": file_sha256(VALIDATOR_TEST_PATH),
        },
        "registered_source_declarations": specification["registered_sources"],
        "registered_source_files_opened": [],
    }
    write_json_exclusive(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    arguments = parser.parse_args(argv)
    result = run_validation(arguments.specification, arguments.output)
    print(json.dumps({"status": result["status"], "passed": result["passed"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
