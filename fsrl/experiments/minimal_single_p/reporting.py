"""Canonical level decisions and concise reports for the subtraction ladder."""

from __future__ import annotations

from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .decisions import classify_level, seed_passed, successor
from .locks import reference, validate_model_lock
from .protocol import (
    PROTOCOL,
    RUNS,
    evaluation_directory,
    model_lock_path,
    report_path,
    result_path,
    specification,
    validate_level,
)


def render_report(result: dict) -> str:
    lines = [
        f"# Minimal single-P {result['level']} development result",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        "| seed | all three generic panels pass |",
        "|---:|:---:|",
    ]
    for seed, row in result["seeds"].items():
        lines.append(f"| {seed} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += [
        "",
        "A panel requires learned and nonlearned competence, learned and nonlearned causal P dependence, and globally coherent complete-graph margins. No Liu input, human interval, noisy-training arm, horizon curriculum, or human-facing phenotype was evaluated.",
        "",
        f"Registered successor: `{result['successor'] or 'none'}`.",
        "",
        "This three-seed development result describes only the registered subtraction level. It is not a population success-rate estimate, a global minimality proof, or a human/biological mechanism claim.",
        "",
    ]
    return "\n".join(lines)


def write_report(level: str) -> dict:
    level = validate_level(level)
    _, lock = validate_model_lock(level)
    directory = RUNS / "summary" / level
    seeds = {}
    with ProspectiveRun.start(
        directory,
        workflow_id="minimal_single_p_v1",
        execution_id=f"summary-{level}",
        producer={"model_lock": reference(model_lock_path(level)), "level": level},
        resolved_config=specification()["decision"],
    ):
        for seed in specification()["design"]["network_seeds"]:
            panels = {
                str(panel): completed(evaluation_directory(seed, panel, level))
                for panel in specification()["design"]["evaluation_panels"]
            }
            seeds[str(seed)] = {"panels": panels, "passed": seed_passed(panels)}
        seed_decisions = {seed: row["passed"] for seed, row in seeds.items()}
        outcome = classify_level(seed_decisions)
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "model_lock": reference(model_lock_path(level)),
            "level": level,
            "seeds": seeds,
            "outcome": outcome,
            "successor": successor(level, outcome),
            "claim_boundary": specification()["claim_boundary"],
            "human_outcomes_exposed": False,
            "locked_models": len(lock["runs"]),
        }
        write_json_exclusive(directory / "result.json", result)
    target = result_path(level)
    report = report_path(level)
    write_json_exclusive(target, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(render_report(result))
    return {"level": level, "outcome": outcome, "successor": result["successor"]}


__all__ = ["render_report", "write_report"]
