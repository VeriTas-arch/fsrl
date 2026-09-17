"""Canonical generic and full solution-distribution reports."""

from __future__ import annotations

from collections import Counter

from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .decisions import (
    ROWS,
    generic_category,
    pilot_compatible,
    row_flags,
    study_outcome,
    wilson,
)
from .locks import reference, require_generic_freeze, validate_model_lock
from .protocol import (
    GENERIC_REPORT,
    GENERIC_RESULT,
    MODEL_LOCK,
    PROTOCOL,
    PROTOCOL_SHA256,
    REPORT,
    RESULT,
    RUNS,
    generic_directory,
    liu_directory,
    specification,
)


def _write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _generic_report(result: dict) -> str:
    lines = [
        "# Minimal single-P M2 generic solution distribution",
        "",
        f"Locked networks: {result['networks']}. No Liu outcome was evaluated.",
        "",
        "| category | count | proportion | Wilson 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["distribution"].items():
        interval = row["wilson95"]
        lines.append(
            f"| {name} | {row['count']} | {row['proportion']:.3f} | "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}] |"
        )
    lines += [
        "",
        (
            "All twenty final checkpoints and all sixty generic panels were evaluated. "
            "No network was selected, removed, restarted, or tuned from these outcomes."
        ),
        "",
    ]
    return "\n".join(lines)


def write_generic_report() -> dict:
    _, lock = validate_model_lock()
    spec = specification()
    networks = {}
    with ProspectiveRun.start(
        RUNS / "summary" / "generic",
        workflow_id="minimal_single_p_promotion_v1",
        execution_id="summary-generic",
        producer={"model_lock": reference(MODEL_LOCK)},
        resolved_config=spec["generic_evaluation"],
    ):
        for seed in spec["design"]["network_seeds"]:
            panels = {
                str(panel): completed(generic_directory(seed, panel))
                for panel in spec["design"]["evaluation_panels"]
            }
            networks[str(seed)] = {
                "eta": lock["runs"][str(seed)]["metadata"]["final_eta"],
                "category": generic_category(panels),
                "panels": panels,
            }
        counts = Counter(row["category"] for row in networks.values())
        distribution = {
            name: wilson(counts[name], len(networks))
            for name in (
                "stable_constructive",
                "panel_variable",
                "nonconstructive",
            )
        }
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "protocol_sha256": PROTOCOL_SHA256,
            "model_lock": reference(MODEL_LOCK),
            "networks": len(networks),
            "network_results": networks,
            "distribution": distribution,
            "human_outcomes_exposed": False,
            "selection_performed": False,
        }
        write_json_exclusive(RUNS / "summary" / "generic" / "result.json", result)
    write_json_exclusive(GENERIC_RESULT, result)
    _write_text(GENERIC_REPORT, _generic_report(result))
    return {"networks": len(networks), "distribution": distribution}


def _final_report(result: dict) -> str:
    lines = [
        "# Minimal single-P M2 solution distribution",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        (
            f"Strict pilot-compatible networks: "
            f"{result['pilot_compatibility']['count']}/20 "
            f"({result['pilot_compatibility']['proportion']:.3f})."
        ),
        "",
        "| behavior row | all-panel qualitative | all-panel quantitative |",
        "|---|---:|---:|",
    ]
    for name in ROWS:
        row = result["row_distribution"][name]
        lines.append(
            f"| {name} | {row['qualitative']['count']}/20 | "
            f"{row['calibration']['count']}/20 |"
        )
    lines += [
        "",
        (
            "Every network and panel remains in the machine-readable result. Network "
            "heterogeneity is reported rather than filtered. The historical query-key "
            "shuffle is a structural no-op because this model has no local memory L."
        ),
        "",
        (
            "`main_model_promoted` is false. This study characterizes the selected M2 "
            "candidate; it does not establish a human neural mechanism or global "
            "minimality."
        ),
        "",
    ]
    return "\n".join(lines)


def write_final_report() -> dict:
    generic = require_generic_freeze()
    spec = specification()
    networks = {}
    with ProspectiveRun.start(
        RUNS / "summary" / "final",
        workflow_id="minimal_single_p_promotion_v1",
        execution_id="summary-final",
        producer={"generic_result": reference(GENERIC_RESULT)},
        resolved_config=spec["decision"],
    ):
        for seed in spec["design"]["network_seeds"]:
            generic_panels = generic["network_results"][str(seed)]["panels"]
            liu_panels = {
                str(panel): completed(liu_directory(seed, panel))["liu"]
                for panel in spec["design"]["evaluation_panels"]
            }
            networks[str(seed)] = {
                "generic_category": generic["network_results"][str(seed)]["category"],
                "eta": generic["network_results"][str(seed)]["eta"],
                "pilot_compatible": pilot_compatible(generic_panels, liu_panels),
                "liu_panels": liu_panels,
            }
        compatible = {seed: row["pilot_compatible"] for seed, row in networks.items()}
        categories = {seed: row["generic_category"] for seed, row in networks.items()}
        row_distribution = {}
        for name in ROWS:
            qualitative = sum(
                all(
                    row_flags(panel)[name]["qualitative"]
                    for panel in row["liu_panels"].values()
                )
                for row in networks.values()
            )
            calibration = sum(
                all(
                    row_flags(panel)[name]["calibration"]
                    for panel in row["liu_panels"].values()
                )
                for row in networks.values()
            )
            row_distribution[name] = {
                "qualitative": wilson(qualitative, len(networks)),
                "calibration": wilson(calibration, len(networks)),
            }
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "protocol_sha256": PROTOCOL_SHA256,
            "generic_result": reference(GENERIC_RESULT),
            "networks": networks,
            "generic_distribution": generic["distribution"],
            "row_distribution": row_distribution,
            "pilot_compatibility": wilson(sum(compatible.values()), len(compatible)),
            "outcome": study_outcome(categories, compatible),
            "main_model_promoted": False,
            "selection_performed": False,
            "query_shuffle_interpretation": (
                "structural no-op because M2 has no local memory or local query key"
            ),
            "claim_boundary": spec["claim_boundary"],
        }
        write_json_exclusive(RUNS / "summary" / "final" / "result.json", result)
    write_json_exclusive(RESULT, result)
    _write_text(REPORT, _final_report(result))
    return {
        "outcome": result["outcome"],
        "pilot_compatible": result["pilot_compatibility"]["count"],
        "main_model_promoted": False,
    }


__all__ = ["write_final_report", "write_generic_report"]
