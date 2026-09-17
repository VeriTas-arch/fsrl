"""Pure registered gates and stop transitions for the subtraction ladder."""

from __future__ import annotations

from .protocol import LEVELS, validate_level


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


def seed_passed(panels: dict[str, dict]) -> bool:
    return bool(panels) and all(panel_passed(panel) for panel in panels.values())


def classify_level(seed_decisions: dict[str, bool]) -> str:
    passed = sum(seed_decisions.values())
    if passed == 3 and len(seed_decisions) == 3:
        return "clear_continue"
    if passed == 0 and len(seed_decisions) == 3:
        return "structural_collapse"
    if len(seed_decisions) == 3:
        return "mixed_boundary"
    raise ValueError("a level decision requires exactly three registered seeds")


def successor(level: str, outcome: str) -> str | None:
    index = LEVELS.index(validate_level(level))
    if outcome != "clear_continue" or index == len(LEVELS) - 1:
        return None
    return LEVELS[index + 1]


__all__ = ["classify_level", "panel_passed", "seed_passed", "successor"]
