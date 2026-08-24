"""Locked analysis for magnitude-placement behavior v1.1."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

from .magnitude_placement_collection import PROTOCOL_PATH, load_json

ROLES = tuple("ABCDEFGH")
PAIRS = tuple(f"{first}-{second}" for first, second in combinations(ROLES, 2))


def _positions(low_to_high: list[str]) -> dict[str, int]:
    return {role: index for index, role in enumerate(low_to_high)}


def _higher(pair: str, positions: dict[str, int]) -> str:
    first, second = pair.split("-")
    return first if positions[first] > positions[second] else second


def _choice_probability(pair_probability: dict[str, float], pair: str, role: str) -> float:
    first, second = pair.split("-")
    if role == first:
        return float(pair_probability[pair])
    if role == second:
        return 1.0 - float(pair_probability[pair])
    raise ValueError("role is not in pair")


def _set_choice_probability(
    pair_probability: dict[str, float], pair: str, role: str, probability: float
) -> None:
    first, second = pair.split("-")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("choice probability is outside [0,1]")
    if role == first:
        pair_probability[pair] = float(probability)
    elif role == second:
        pair_probability[pair] = float(1.0 - probability)
    else:
        raise ValueError("role is not in pair")


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1])


def participant_estimands(profile: dict, protocol: dict) -> dict:
    inherited = protocol["inherited_frozen_contract"]
    partition = inherited["pair_partition"]
    positions_a = _positions(inherited["assignment_A_low_to_high"])
    positions_b = _positions(inherited["assignment_B_low_to_high"])
    field_a = profile["pair_probability"]["A"]
    field_b = profile["pair_probability"]["B"]

    flip_values = []
    for pair in partition["nonlearned_order_flip"]:
        higher_a = _higher(pair, positions_a)
        flip_values.append(
            _choice_probability(field_a, pair, higher_a)
            - _choice_probability(field_b, pair, higher_a)
        )

    confidence_x = []
    confidence_y = []
    for pair in partition["nonlearned_same_direction"]:
        higher_a = _higher(pair, positions_a)
        higher_b = _higher(pair, positions_b)
        if higher_a != higher_b:
            raise RuntimeError("registered same-direction pair changed direction")
        first, second = pair.split("-")
        confidence_x.append(
            abs(positions_b[first] - positions_b[second])
            - abs(positions_a[first] - positions_a[second])
        )
        confidence_y.append(
            _choice_probability(field_b, pair, higher_a)
            - _choice_probability(field_a, pair, higher_a)
        )

    relation_gaps = {
        condition: dict(
            zip(
                inherited["support_relation_order"],
                inherited[f"assignment_{condition}_gaps"],
                strict=True,
            )
        )
        for condition in ("A", "B")
    }
    learned_x = []
    learned_y = []
    for relation in inherited["support_relation_order"]:
        higher, lower = relation.split(">")
        pair = "-".join(sorted((higher, lower), key=ROLES.index))
        learned_x.append(relation_gaps["B"][relation] - relation_gaps["A"][relation])
        learned_y.append(
            _choice_probability(field_b, pair, higher)
            - _choice_probability(field_a, pair, higher)
        )
    return {
        "delta_flip": float(np.mean(flip_values)),
        "beta_conf": _slope(
            np.asarray(confidence_x, dtype=np.float64),
            np.asarray(confidence_y, dtype=np.float64),
        ),
        "beta_learned": _slope(
            np.asarray(learned_x, dtype=np.float64),
            np.asarray(learned_y, dtype=np.float64),
        ),
    }


def hodge_rank_positions(pair_probability: dict[str, float]) -> np.ndarray:
    rows = []
    values = []
    for first, second in combinations(range(len(ROLES)), 2):
        row = np.zeros(len(ROLES), dtype=np.float64)
        row[first] = 1.0
        row[second] = -1.0
        rows.append(row)
        pair = f"{ROLES[first]}-{ROLES[second]}"
        values.append(2.0 * float(pair_probability[pair]) - 1.0)
    rows.append(np.ones(len(ROLES), dtype=np.float64))
    values.append(0.0)
    scores, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(values), rcond=None)
    order = np.argsort(-scores)
    positions = np.empty_like(order)
    positions[order] = np.arange(len(ROLES))
    return positions


def kendall_tau_positions(first: np.ndarray, second: np.ndarray) -> float:
    products = [
        (first[left] - first[right]) * (second[left] - second[right])
        for left, right in combinations(range(len(first)), 2)
    ]
    return float(np.mean(np.sign(products)))


def profile_metrics(profile: dict, protocol: dict) -> dict:
    inherited = protocol["inherited_frozen_contract"]
    partition = inherited["pair_partition"]
    learned = set(partition["learned"])
    flip = set(partition["nonlearned_order_flip"])
    same = set(partition["nonlearned_same_direction"])
    positions = {
        condition: _positions(inherited[f"assignment_{condition}_low_to_high"])
        for condition in ("A", "B")
    }
    condition_metrics = {}
    ranks = {}
    for condition in ("A", "B"):
        field = profile["pair_probability"][condition]
        correct = {
            pair: _choice_probability(field, pair, _higher(pair, positions[condition]))
            for pair in PAIRS
        }
        condition_metrics[condition] = {
            "overall_accuracy": float(np.mean(list(correct.values()))),
            "learned_accuracy": float(np.mean([correct[pair] for pair in learned])),
            "nonlearned_accuracy": float(
                np.mean([correct[pair] for pair in set(PAIRS) - learned])
            ),
            "flip_accuracy": float(np.mean([correct[pair] for pair in flip])),
            "same_direction_accuracy": float(
                np.mean([correct[pair] for pair in same])
            ),
        }
        ranks[condition] = hodge_rank_positions(field)
    order_positions = {
        condition: np.asarray(
            [
                list(
                    reversed(inherited[f"assignment_{condition}_low_to_high"])
                ).index(role)
                for role in ROLES
            ],
            dtype=np.int64,
        )
        for condition in ("A", "B")
    }
    assignment_following = (
        kendall_tau_positions(ranks["A"], order_positions["A"])
        - kendall_tau_positions(ranks["A"], order_positions["B"])
        + kendall_tau_positions(ranks["B"], order_positions["B"])
        - kendall_tau_positions(ranks["B"], order_positions["A"])
    )
    combined_accuracy = float(
        np.mean(
            [
                condition_metrics["A"]["overall_accuracy"],
                condition_metrics["B"]["overall_accuracy"],
            ]
        )
    )
    return {
        "condition": condition_metrics,
        "combined_accuracy": combined_accuracy,
        "hodge_positions": {
            condition: [int(value) for value in ranks[condition]]
            for condition in ("A", "B")
        },
        "hodge_order_high_to_low": {
            condition: [ROLES[index] for index in np.argsort(ranks[condition])]
            for condition in ("A", "B")
        },
        "assignment_following_score": assignment_following,
    }


def profile_from_session_bundles(bundles: list[dict]) -> dict:
    if len(bundles) != 2:
        raise ValueError("exactly two session bundles are required")
    participant_ids = {bundle["participant_id"] for bundle in bundles}
    slots = {bundle["enrollment_slot"] for bundle in bundles}
    cells = {bundle["counterbalance_cell"] for bundle in bundles}
    conditions = {bundle["condition"] for bundle in bundles}
    sessions = {bundle["session_index"] for bundle in bundles}
    if not (
        len(participant_ids) == len(slots) == len(cells) == 1
        and conditions == {"A", "B"}
        and sessions == {1, 2}
    ):
        raise ValueError("session bundles do not form one complete participant")
    pair_probability = {}
    technical_events = Counter()
    for bundle in bundles:
        query = [trial for trial in bundle["trials"] if trial["phase"] == "query"]
        support = [trial for trial in bundle["trials"] if trial["phase"] == "support"]
        if len(query) != 280 or len(support) != 32:
            raise ValueError("incomplete session bundle")
        choices: dict[str, list[float]] = defaultdict(list)
        for trial in query:
            first, second = sorted((trial["role_left"], trial["role_right"]), key=ROLES.index)
            chosen = trial["role_left"] if trial["choice_side"] == "left" else trial["role_right"]
            choices[f"{first}-{second}"].append(float(chosen == first))
        if set(choices) != set(PAIRS) or any(len(values) != 10 for values in choices.values()):
            raise ValueError("query pair repetition contract failed")
        pair_probability[bundle["condition"]] = {
            pair: float(np.mean(values)) for pair, values in sorted(choices.items())
        }
        technical_events.update(
            event["event_kind"]
            for event in bundle["acquisition_events"]
            if event["event_kind"] not in {"session_start", "session_end"}
        )
    return {
        "participant_id": next(iter(participant_ids)),
        "enrollment_slot": next(iter(slots)),
        "counterbalance_cell": next(iter(cells)),
        "complete": True,
        "pair_probability": pair_probability,
        "technical_event_counts": dict(sorted(technical_events.items())),
    }


def _cell_codes(cell: str) -> tuple[float, float]:
    order = 1.0 if cell.startswith("A_first") else -1.0
    mapping = 1.0 if cell.endswith("C1_to_A") else -1.0
    return order, mapping


def adjusted_ols(values: np.ndarray, cells: list[str], margin: float) -> dict:
    rows = []
    for cell in cells:
        order, mapping = _cell_codes(cell)
        rows.append((1.0, order, mapping, order * mapping))
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) != 4:
        raise RuntimeError("counterbalance design is rank deficient")
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    degrees_freedom = len(values) - design.shape[1]
    if degrees_freedom <= 0:
        raise RuntimeError("insufficient residual degrees of freedom")
    variance = float(residual @ residual / degrees_freedom)
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_error = math.sqrt(max(0.0, float(covariance[0, 0])))
    intervals = {}
    for level in (0.90, 0.95):
        critical = float(stats.t.ppf(0.5 + level / 2.0, degrees_freedom))
        intervals[str(level)] = {
            "lower": float(coefficients[0] - critical * standard_error),
            "upper": float(coefficients[0] + critical * standard_error),
        }
    primary = intervals["0.9"]
    flags = {
        "directional_positive": primary["lower"] > 0.0,
        "directional_negative": primary["upper"] < 0.0,
        "equivalent_to_zero": primary["lower"] >= -margin
        and primary["upper"] <= margin,
    }
    return {
        "point": float(coefficients[0]),
        "coefficients": {
            "intercept": float(coefficients[0]),
            "condition_order": float(coefficients[1]),
            "codebook_condition_mapping": float(coefficients[2]),
            "interaction": float(coefficients[3]),
        },
        "standard_error": standard_error,
        "degrees_freedom": degrees_freedom,
        "intervals": intervals,
        "sesoi": margin,
        "flags": flags,
    }


def classify_outcome(results: dict) -> str:
    flip = results["delta_flip"]["flags"]
    if flip["equivalent_to_zero"]:
        confidence = results["beta_conf"]["flags"]
        if confidence["equivalent_to_zero"]:
            learned = results["beta_learned"]["flags"]
            if learned["equivalent_to_zero"]:
                return "stronger_ordinalization_or_metric_loss"
            if learned["directional_positive"]:
                return "metric_retained_locally_not_globally"
            return "mixed_or_unresolved"
        if confidence["directional_positive"]:
            return "ordinal_global_order_with_metric_confidence_modulation"
        return "mixed_or_unresolved"
    if flip["directional_positive"]:
        return "metric_enters_global_order_construction"
    return "mixed_or_unresolved"


def analyze_profiles(profiles: list[dict], protocol: dict) -> dict:
    derived = []
    excluded = Counter()
    for profile in profiles:
        if not profile.get("complete", False):
            excluded["incomplete"] += 1
            continue
        metrics = profile_metrics(profile, protocol)
        if metrics["combined_accuracy"] < 0.50:
            excluded["combined_below_chance"] += 1
            continue
        derived.append(
            {
                **profile,
                "estimands": participant_estimands(profile, protocol),
                "metrics": metrics,
            }
        )
    cells = [profile["counterbalance_cell"] for profile in derived]
    cell_counts = Counter(cells)
    expected_cells = {
        "A_first/C1_to_A",
        "A_first/C1_to_B",
        "B_first/C1_to_A",
        "B_first/C1_to_B",
    }
    gates = {
        "100_analyzable_participants": len(derived) == 100,
        "25_per_counterbalance_cell": set(cell_counts) == expected_cells
        and set(cell_counts.values()) == {25},
        "all_pair_fields_complete": all(
            set(profile["pair_probability"][condition]) == set(PAIRS)
            for profile in derived
            for condition in ("A", "B")
        ),
        "all_probabilities_finite_and_bounded": all(
            np.isfinite(probability) and 0.0 <= probability <= 1.0
            for profile in derived
            for condition in ("A", "B")
            for probability in profile["pair_probability"][condition].values()
        ),
    }
    if not all(gates.values()):
        return {
            "outcome": "noninterpretable",
            "gates": gates,
            "analyzable_participants": len(derived),
            "cell_counts": dict(sorted(cell_counts.items())),
            "exclusions": dict(sorted(excluded.items())),
        }
    margins = protocol["inherited_frozen_contract"]["sesoi"]
    results = {
        name: adjusted_ols(
            np.asarray([profile["estimands"][name] for profile in derived]),
            cells,
            float(margins[name]),
        )
        for name in protocol["inherited_frozen_contract"]["estimand_order"]
    }
    results["assignment_following_score"] = adjusted_ols(
        np.asarray(
            [profile["metrics"]["assignment_following_score"] for profile in derived]
        ),
        cells,
        1.0,
    )
    condition_descriptives = {}
    for condition in ("A", "B"):
        condition_descriptives[condition] = {
            metric: float(
                np.mean(
                    [
                        profile["metrics"]["condition"][condition][metric]
                        for profile in derived
                    ]
                )
            )
            for metric in (
                "overall_accuracy",
                "learned_accuracy",
                "nonlearned_accuracy",
                "flip_accuracy",
                "same_direction_accuracy",
            )
        }
    pair_fields = {
        condition: {
            pair: float(
                np.mean(
                    [
                        profile["pair_probability"][condition][pair]
                        for profile in derived
                    ]
                )
            )
            for pair in PAIRS
        }
        for condition in ("A", "B")
    }
    return {
        "outcome": classify_outcome(results),
        "gates": gates,
        "analyzable_participants": len(derived),
        "cell_counts": dict(sorted(cell_counts.items())),
        "exclusions": dict(sorted(excluded.items())),
        "primary": {name: results[name] for name in margins},
        "secondary": {"assignment_following_score": results["assignment_following_score"]},
        "descriptive": {
            "condition_metrics": condition_descriptives,
            "pair_choice_probability": pair_fields,
            "pair_difference_B_minus_A": {
                pair: pair_fields["B"][pair] - pair_fields["A"][pair]
                for pair in PAIRS
            },
            "hodge_orders": {
                profile["participant_id"]: profile["metrics"]["hodge_order_high_to_low"]
                for profile in derived
            },
        },
    }


def synthetic_profiles(outcome: str, protocol: dict) -> list[dict]:
    targets = {
        "metric_enters_global_order_construction": (0.20, 0.0, 0.0),
        "ordinal_global_order_with_metric_confidence_modulation": (0.0, 0.05, 0.0),
        "metric_retained_locally_not_globally": (0.0, 0.0, 0.05),
        "stronger_ordinalization_or_metric_loss": (0.0, 0.0, 0.0),
        "mixed_or_unresolved": (-0.20, 0.0, 0.0),
    }
    if outcome not in targets:
        raise ValueError("unknown synthetic outcome")
    delta_flip, beta_conf, beta_learned = targets[outcome]
    inherited = protocol["inherited_frozen_contract"]
    positions_a = _positions(inherited["assignment_A_low_to_high"])
    positions_b = _positions(inherited["assignment_B_low_to_high"])
    partition = inherited["pair_partition"]
    fields = {condition: {pair: 0.5 for pair in PAIRS} for condition in ("A", "B")}
    for condition, positions in (("A", positions_a), ("B", positions_b)):
        for pair in PAIRS:
            _set_choice_probability(fields[condition], pair, _higher(pair, positions), 0.75)
    for pair in partition["nonlearned_order_flip"]:
        higher_a = _higher(pair, positions_a)
        probability_a = _choice_probability(fields["A"], pair, higher_a)
        _set_choice_probability(
            fields["B"], pair, higher_a, probability_a - delta_flip
        )
    for pair in partition["nonlearned_same_direction"]:
        first, second = pair.split("-")
        higher = _higher(pair, positions_a)
        delta_distance = abs(positions_b[first] - positions_b[second]) - abs(
            positions_a[first] - positions_a[second]
        )
        _set_choice_probability(
            fields["B"],
            pair,
            higher,
            _choice_probability(fields["A"], pair, higher)
            + beta_conf * delta_distance,
        )
    gaps = {
        condition: dict(
            zip(
                inherited["support_relation_order"],
                inherited[f"assignment_{condition}_gaps"],
                strict=True,
            )
        )
        for condition in ("A", "B")
    }
    for relation in inherited["support_relation_order"]:
        higher, lower = relation.split(">")
        pair = "-".join(sorted((higher, lower), key=ROLES.index))
        delta_magnitude = gaps["B"][relation] - gaps["A"][relation]
        _set_choice_probability(
            fields["B"],
            pair,
            higher,
            _choice_probability(fields["A"], pair, higher)
            + beta_learned * delta_magnitude,
        )
    cells = [
        "A_first/C1_to_A",
        "A_first/C1_to_B",
        "B_first/C1_to_A",
        "B_first/C1_to_B",
    ]
    return [
        {
            "participant_id": f"SYNTH-FIXTURE-{index + 1:03d}",
            "enrollment_slot": f"MPB-{index + 1:03d}",
            "counterbalance_cell": cells[index // 25],
            "complete": True,
            "pair_probability": json.loads(json.dumps(fields)),
            "technical_event_counts": {},
        }
        for index in range(100)
    ]


def validate_all_synthetic_branches(protocol: dict | None = None) -> dict:
    protocol = load_json(PROTOCOL_PATH) if protocol is None else protocol
    outcomes = (
        "metric_enters_global_order_construction",
        "ordinal_global_order_with_metric_confidence_modulation",
        "metric_retained_locally_not_globally",
        "stronger_ordinalization_or_metric_loss",
        "mixed_or_unresolved",
    )
    observed = {
        expected: analyze_profiles(synthetic_profiles(expected, protocol), protocol)
        for expected in outcomes
    }
    gates = {
        expected: result["outcome"] == expected
        and all(result["gates"].values())
        for expected, result in observed.items()
    }
    return {"passed": all(gates.values()), "gates": gates, "results": observed}


if __name__ == "__main__":
    validation = validate_all_synthetic_branches()
    print(json.dumps({"passed": validation["passed"], "gates": validation["gates"]}))
    raise SystemExit(0 if validation["passed"] else 2)
