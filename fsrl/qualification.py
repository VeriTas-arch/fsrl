"""Apply the registered GO/NO-GO criteria to a causal-suite result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_QUALIFICATION_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "qualification_v1.json"
)


def evaluate_qualification(result: dict, specification: dict) -> dict:
    checks = []

    def add(name: str, passed: bool, observed, criterion: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
            }
        )

    required_cue_mode = specification["requires_cue_mode"]
    add(
        "cue_mode",
        result.get("cue_mode") == required_cue_mode,
        result.get("cue_mode"),
        f"== {required_cue_mode}",
    )
    required_encoding = specification["requires_subject_encoding_mode"]
    observed_encoding = result.get("subject_encoding", {}).get("mode")
    add(
        "subject_encoding.mode",
        observed_encoding == required_encoding,
        observed_encoding,
        f"== {required_encoding}",
    )
    provenance = result.get("training_provenance", {})
    add(
        "training_provenance.present",
        provenance.get("present") is True,
        provenance.get("present"),
        "is true",
    )
    add(
        "training_provenance.checkpoint_sha_matches",
        provenance.get("checkpoint_sha_matches") is True,
        provenance.get("checkpoint_sha_matches"),
        "is true",
    )
    task_distribution = provenance.get("task_distribution") or {}
    required_held_out = specification["requires_held_out_liu_graph"]
    observed_held_out = task_distribution.get("liu_graph_held_out")
    add(
        "training_provenance.liu_graph_held_out",
        observed_held_out is required_held_out,
        observed_held_out,
        f"is {str(required_held_out).lower()}",
    )
    conditions = result.get("conditions", {})
    intact = conditions.get("intact")
    add("intact_present", intact is not None, intact is not None, "is true")
    if intact is not None:
        for metric, threshold in specification["intact_minimum"].items():
            observed = intact.get(metric)
            add(
                f"intact.{metric}",
                observed is not None and observed >= threshold,
                observed,
                f">= {threshold}",
            )

    for intervention in specification["required_interventions"]:
        condition = conditions.get(intervention)
        add(
            f"{intervention}.present",
            condition is not None,
            condition is not None,
            "is true",
        )
        if condition is None:
            continue
        for metric, threshold in specification["intervention_maximum"].items():
            observed = condition.get(metric)
            add(
                f"{intervention}.{metric}",
                observed is not None and observed <= threshold,
                observed,
                f"<= {threshold}",
            )

    invariance = result.get("order_invariance", {})
    observed_delta = invariance.get("max_abs_logit_delta")
    threshold = specification["order_invariance_max_abs_logit_delta"]
    add(
        "order_invariance.max_abs_logit_delta",
        observed_delta is not None and observed_delta <= threshold,
        observed_delta,
        f"<= {threshold}",
    )
    return {
        "qualification_id": specification["qualification_id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def load_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the registered GO/NO-GO gate."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_QUALIFICATION_PATH
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    report = evaluate_qualification(
        load_json(parsed.result), load_json(parsed.specification)
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
