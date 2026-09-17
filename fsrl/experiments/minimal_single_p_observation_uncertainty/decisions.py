"""Pure registered summaries and decisions for acute M2 observation uncertainty."""

from __future__ import annotations

import math

import numpy as np

from fsrl.experiments.minimal_single_p_promotion.decisions import ROWS, wilson

CONDITIONS = ("clean", "folded", "noisy")
PRIMARY = (
    "difficult_pair_bimodality",
    "stable_within_subject_errors",
    "inter_subject_ranking_diversity",
)
PRESERVATION = tuple(name for name in ROWS if name not in PRIMARY)


def _behavior(liu: dict) -> dict:
    return liu["routes"]["full"]["behavior"]["historical_nine_rows"]


def compact_panel(liu: dict) -> dict:
    behavior = _behavior(liu)
    flags = behavior["flags"]
    if set(flags) != set(ROWS):
        raise RuntimeError("behavior row identity differs")
    metrics = behavior["metrics"]
    primary = {
        "difficult_pair_bimodality": float(
            metrics["difficult_pair_bimodality"]["bimodal"]
        ),
        "stable_within_subject_errors": metrics["stable_within_subject_errors"][
            "point"
        ],
        "inter_subject_ranking_diversity": metrics["inter_subject_ranking_diversity"][
            "point"
        ],
    }
    binding = liu["effects"]["intact_minus_evidence_shuffle_learned"]["bootstrap"]
    binding_passed = binding["lower"] is not None and binding["lower"] > 0
    qualitative = {name: bool(flags[name]["qualitative"]) for name in ROWS}
    calibration = {name: bool(flags[name]["calibration"]) for name in ROWS}
    structured = all(qualitative.values()) and binding_passed
    return {
        "primary": primary,
        "qualitative": qualitative,
        "calibration": calibration,
        "evidence_binding": {
            "mean": liu["effects"]["intact_minus_evidence_shuffle_learned"]["mean"],
            "lower": binding["lower"],
            "upper": binding["upper"],
            "passed": binding_passed,
        },
        "eligible_subjects": behavior["eligible_subjects"],
        "analysis_subjects": behavior["analysis_subjects_excluding_correct_rankers"],
        "structured_complete": structured,
    }


def stable_complete(panels: dict[str, dict]) -> bool:
    return len(panels) == 3 and all(
        panel["structured_complete"] for panel in panels.values()
    )


def paired_contrast(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("paired network arrays differ")
    differences = first - second
    if not np.isfinite(differences).all():
        return {
            "mean": None,
            "bootstrap": {"lower": None, "upper": None},
            "networks": len(differences),
            "defined": False,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    tail = (1.0 - 0.95) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return {
        "mean": float(differences.mean()),
        "bootstrap": {"lower": float(lower), "upper": float(upper)},
        "networks": len(differences),
        "defined": True,
    }


def expected_direction_supported(name: str, contrast: dict) -> bool:
    if not contrast["defined"]:
        return False
    interval = contrast["bootstrap"]
    if name == "inter_subject_ranking_diversity":
        return interval["upper"] < 0
    if name in PRIMARY:
        return interval["lower"] > 0
    raise ValueError("unknown primary endpoint")


def study_outcome(noisy_complete: int, clean_support: bool, direction: bool) -> str:
    if noisy_complete == 0:
        return "no_acute_completion"
    if clean_support and direction:
        return "direction_specific_acute_completion"
    return "acute_completion_without_direction_specificity"


def condition_distribution(networks: dict[str, dict], condition: str) -> dict:
    count = sum(
        row["conditions"][condition]["stable_complete"] for row in networks.values()
    )
    if not math.isfinite(count):
        raise RuntimeError("invalid completion count")
    return wilson(count, len(networks))


__all__ = [
    "CONDITIONS",
    "PRESERVATION",
    "PRIMARY",
    "compact_panel",
    "condition_distribution",
    "expected_direction_supported",
    "paired_contrast",
    "stable_complete",
    "study_outcome",
]
