"""Pure registered decisions for the M2 solution distribution."""

from __future__ import annotations

import math

ROWS = (
    "learned_accuracy",
    "nonlearned_accuracy",
    "symbolic_distance_effect",
    "serial_position_effect",
    "difficult_pair_bimodality",
    "stable_within_subject_errors",
    "self_consistent_vs_inconsistent_errors",
    "hodge_reconstructed_subjective_ranking",
    "inter_subject_ranking_diversity",
)


def panel_passed(panel: dict) -> bool:
    return (
        all(
            panel[family][group]["bootstrap"]["lower"] > threshold
            for family, groups, threshold in (
                ("competence", ("learned", "nonlearned"), 0.5),
                ("P_dependence", ("learned", "nonlearned"), 0.0),
            )
            for group in groups
        )
        and panel["coherence"]["bootstrap"]["lower"] > 0.95
    )


def generic_category(panels: dict[str, dict]) -> str:
    count = sum(panel_passed(panel) for panel in panels.values())
    if count == 3:
        return "stable_constructive"
    if count:
        return "panel_variable"
    return "nonconstructive"


def wilson(successes: int, total: int) -> dict:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid Wilson count")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return {
        "count": successes,
        "total": total,
        "proportion": p,
        "wilson95": {"lower": center - radius, "upper": center + radius},
    }


def row_flags(liu: dict) -> dict:
    flags = liu["routes"]["full"]["behavior"]["historical_nine_rows"]["flags"]
    if set(flags) != set(ROWS) or len(flags) != len(ROWS):
        raise RuntimeError("historical behavior row identity differs")
    return flags


def liu_panel_passed(liu: dict) -> bool:
    flags = row_flags(liu)
    binding = liu["effects"]["intact_minus_evidence_shuffle_learned"]["bootstrap"]
    return (
        all(row["qualitative"] for row in flags.values())
        and all(row["calibration"] for row in flags.values())
        and binding["lower"] is not None
        and binding["lower"] > 0
    )


def pilot_compatible(generic_panels: dict, liu_panels: dict) -> bool:
    return all(panel_passed(panel) for panel in generic_panels.values()) and all(
        liu_panel_passed(panel) for panel in liu_panels.values()
    )


def study_outcome(categories: dict[str, str], compatible: dict[str, bool]) -> str:
    if not any(value == "stable_constructive" for value in categories.values()):
        return "generic_recipe_failure"
    if compatible and all(compatible.values()):
        return "uniform_complete_pilot_compatibility"
    return "heterogeneous_solution_distribution"


__all__ = [
    "ROWS",
    "generic_category",
    "liu_panel_passed",
    "panel_passed",
    "pilot_compatible",
    "row_flags",
    "study_outcome",
    "wilson",
]
