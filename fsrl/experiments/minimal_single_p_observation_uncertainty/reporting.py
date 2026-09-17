"""Canonical compact result and interpretation for the acute M2 intervention."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.minimal_single_p_promotion.decisions import ROWS, wilson
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .decisions import (
    CONDITIONS,
    PRIMARY,
    compact_panel,
    condition_distribution,
    expected_direction_supported,
    paired_contrast,
    stable_complete,
    study_outcome,
)
from .locks import reference, validate_source_input_lock
from .protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    REPORT,
    RESULT,
    RUNS,
    SOURCE_INPUT_LOCK,
    evaluation_directory,
    register,
    specification,
)


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def _network_results(lock: dict) -> dict:
    rows = {}
    for seed in specification()["design"]["network_seeds"]:
        conditions = {}
        for condition in CONDITIONS:
            panels = {
                str(panel): compact_panel(
                    completed(evaluation_directory(seed, panel, condition))["liu"]
                )
                for panel in specification()["design"]["evaluation_panels"]
            }
            conditions[condition] = {
                "stable_complete": stable_complete(panels),
                "panels": panels,
                "panel_mean_primary": {
                    name: float(
                        np.mean([row["primary"][name] for row in panels.values()])
                    )
                    if all(row["primary"][name] is not None for row in panels.values())
                    else None
                    for name in PRIMARY
                },
            }
        rows[str(seed)] = {
            "generic_category": lock["generic_categories"][str(seed)],
            "conditions": conditions,
        }
    return rows


def _contrasts(networks: dict) -> dict:
    settings = specification()["evaluation"]["bootstrap"]
    rows = {}
    for comparison_index, (first, second) in enumerate(
        (("noisy", "clean"), ("noisy", "folded"))
    ):
        endpoints = {}
        for endpoint_index, name in enumerate(PRIMARY):
            endpoints[name] = paired_contrast(
                np.asarray(
                    [
                        row["conditions"][first]["panel_mean_primary"][name]
                        for row in networks.values()
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        row["conditions"][second]["panel_mean_primary"][name]
                        for row in networks.values()
                    ],
                    dtype=np.float64,
                ),
                seed=settings["network_seed_base"]
                + comparison_index * 100
                + endpoint_index,
                samples=settings["samples"],
            )
        rows[f"{first}_minus_{second}"] = endpoints
    return rows


def _row_distributions(networks: dict) -> dict:
    rows = {}
    for condition in CONDITIONS:
        rows[condition] = {}
        for name in ROWS:
            qualitative = sum(
                all(
                    panel["qualitative"][name]
                    for panel in network["conditions"][condition]["panels"].values()
                )
                for network in networks.values()
            )
            calibration = sum(
                all(
                    panel["calibration"][name]
                    for panel in network["conditions"][condition]["panels"].values()
                )
                for network in networks.values()
            )
            rows[condition][name] = {
                "qualitative": wilson(qualitative, len(networks)),
                "calibration": wilson(calibration, len(networks)),
            }
    return rows


def _strata(networks: dict) -> dict:
    result = {}
    for category in ("stable_constructive", "nonconstructive"):
        members = {
            seed: row
            for seed, row in networks.items()
            if row["generic_category"] == category
        }
        result[category] = {
            "networks": sorted(members),
            "stable_complete_counts": {
                condition: sum(
                    row["conditions"][condition]["stable_complete"]
                    for row in members.values()
                )
                for condition in CONDITIONS
            },
            "descriptive_only": category == "nonconstructive",
        }
    return result


def _render(result: dict) -> str:
    lines = [
        "# Acute observation uncertainty in frozen minimal single-P M2",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        "| condition | all-three-panel structured completion | proportion | Wilson 95% |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = result["condition_distributions"][condition]
        lines.append(
            f"| {condition} | {row['count']}/20 | {row['proportion']:.3f} | "
            f"[{row['wilson95']['lower']:.3f}, {row['wilson95']['upper']:.3f}] |"
        )
    lines += [
        "",
        "| contrast | endpoint | mean difference | bootstrap 95% | expected direction supported |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison, endpoints in result["primary_contrasts"].items():
        for name, row in endpoints.items():
            interval = row["bootstrap"]
            lines.append(
                f"| {comparison} | {name} | {row['mean']} | "
                f"[{interval['lower']}, {interval['upper']}] | "
                f"{result['contrast_support'][comparison][name]} |"
            )
    lines += [
        "",
        (
            "All 20 networks, all three panels, and all three conditions are retained. "
            "The 18/2 generic stratification is reported descriptively; the two-network "
            "nonconstructive stratum is not an interaction test."
        ),
        "",
        (
            "Historical quantitative human-interval compatibility is descriptive only. "
            "No sigma, network, panel, checkpoint, or threshold was selected from these outcomes."
        ),
        "",
        (
            "`main_model_promoted` is false. This acute frozen-checkpoint intervention "
            "cannot establish a human observation process, learning mechanism, or neural implementation."
        ),
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    lock = validate_source_input_lock()
    with ProspectiveRun.start(
        RUNS / "summary",
        workflow_id="minimal_single_p_observation_uncertainty_v1",
        execution_id="summary",
        producer={"source_input_lock": reference(SOURCE_INPUT_LOCK)},
        resolved_config=specification()["decision"],
    ):
        networks = _network_results(lock)
        distributions = {
            condition: condition_distribution(networks, condition)
            for condition in CONDITIONS
        }
        contrasts = _contrasts(networks)
        support = {
            comparison: {
                name: expected_direction_supported(name, row)
                for name, row in endpoints.items()
            }
            for comparison, endpoints in contrasts.items()
        }
        clean_support = all(support["noisy_minus_clean"].values())
        direction_support = all(support["noisy_minus_folded"].values())
        outcome = study_outcome(
            distributions["noisy"]["count"], clean_support, direction_support
        )
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "protocol_sha256": PROTOCOL_SHA256,
            "source_input_lock": reference(SOURCE_INPUT_LOCK),
            "networks": networks,
            "condition_distributions": distributions,
            "row_distributions": _row_distributions(networks),
            "primary_contrasts": contrasts,
            "contrast_support": support,
            "clean_completion_support": clean_support,
            "direction_specific_continuous_support": direction_support,
            "baseline_strata": _strata(networks),
            "outcome": outcome,
            "main_model_promoted": False,
            "selection_performed": False,
            "claim_boundary": specification()["claim_boundary"],
            "input_lock_checks": {
                panel: row["checks"] for panel, row in lock["inputs"].items()
            },
        }
        write_json_exclusive(RUNS / "summary" / "result.json", result)
    write_json_exclusive(RESULT, result)
    _write_text(REPORT, _render(result))
    statuses = {
        "no_acute_completion": "valid_negative",
        "direction_specific_acute_completion": "supporting",
        "acute_completion_without_direction_specificity": "mixed",
    }
    counts = {
        condition: result["condition_distributions"][condition]["count"]
        for condition in CONDITIONS
    }
    register(
        status=statuses[outcome],
        finding=(
            f"Outcome {outcome}: all-three-panel structured completion counts were "
            f"clean={counts['clean']}/20, folded={counts['folded']}/20, and "
            f"noisy={counts['noisy']}/20. All networks and panels were retained; "
            "main-model promotion remains false."
        ),
    )
    return {"outcome": outcome, "stable_complete_counts": counts}


__all__ = ["write_report"]
