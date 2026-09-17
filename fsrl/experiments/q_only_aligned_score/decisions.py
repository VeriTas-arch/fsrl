"""Frozen pure decision rules for aligned comparator outcomes."""

from __future__ import annotations


def generic_panel_passed(result: dict) -> bool:
    def lower(section: str, group: str) -> float | None:
        return result[section][group]["interval"]["lower"]

    values = [
        lower("competence", "learned"),
        lower("competence", "nonlearned"),
        lower("state_dependence", "learned"),
        lower("state_dependence", "nonlearned"),
        lower("evidence_binding", "learned"),
    ]
    thresholds = (0.5, 0.5, 0.0, 0.0, 0.0)
    return all(
        value is not None and value > threshold
        for value, threshold in zip(values, thresholds, strict=True)
    )


def generic_category(passed_panels: int) -> str:
    if passed_panels == 3:
        return "stable_competent"
    if passed_panels:
        return "panel_variable"
    return "noncompetent"


__all__ = ["generic_category", "generic_panel_passed"]
