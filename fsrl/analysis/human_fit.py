"""Select one global query-choice temperature using overall accuracy only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record

DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/human_fit_v1.json")


def _cohort_signature(result: dict) -> dict:
    return {
        "protocol_id": result.get("protocol_id"),
        "checkpoint_sha256": result.get("checkpoint", {}).get("sha256"),
        "cue_seed": result.get("cue_seed"),
        "support_seed": result.get("support_seed"),
        "subject_encoding_seed": result.get("subject_encoding_seed"),
        "subject_encoding_mode": result.get("subject_encoding_mode"),
        "choice_seed": result.get("sampling", {}).get("seed"),
        "test_blocks": result.get("sampling", {}).get("test_blocks"),
        "trials_per_subject": result.get("sampling", {}).get("trials_per_subject"),
    }


def select_global_temperature(results: list[dict], specification: dict) -> dict:
    """Select the registered temperature closest to approximate overall accuracy."""

    if not results:
        raise ValueError("at least one behavioral result is required")
    expected_grid = [float(value) for value in specification["temperature_grid"]]
    by_temperature = {
        float(result["sampling"]["temperature"]): result for result in results
    }
    if sorted(by_temperature) != sorted(expected_grid):
        raise ValueError("behavioral results must cover the registered grid exactly")
    signature = _cohort_signature(results[0])
    if any(_cohort_signature(result) != signature for result in results[1:]):
        raise ValueError(
            "all behavioral results must use the same cohort and checkpoint"
        )
    if signature["protocol_id"] != specification["protocol_id"]:
        raise ValueError("behavioral protocol does not match fit specification")

    target = float(specification["target_overall_accuracy"])
    rows = []
    for temperature in expected_grid:
        summary = by_temperature[temperature]["summary"]
        observed = float(summary["overall_accuracy"])
        rows.append(
            {
                "temperature": temperature,
                "overall_accuracy": observed,
                "absolute_overall_accuracy_error": abs(observed - target),
                "secondary_metrics": {
                    name: summary[name]
                    for name in specification["secondary_metrics_are_out_of_sample"]
                },
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["absolute_overall_accuracy_error"],
            row["temperature"],
        ),
    )
    for row in rows:
        row["selected"] = row is selected
    return {
        "fit_id": specification["fit_id"],
        "registration_status": specification.get("registration_status"),
        "status": "descriptive_fit_only",
        "status_reason": (
            "the human accuracy target is an approximate figure read and no formal "
            "acceptance tolerance was registered"
        ),
        "selection_metric": specification["selection_metric"],
        "target_overall_accuracy": target,
        "selected_temperature": selected["temperature"],
        "selected_absolute_error": selected["absolute_overall_accuracy_error"],
        "cohort": signature,
        "grid_results": rows,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Select the registered global choice temperature."
    )
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    report = select_global_temperature(
        [load_json(path) for path in parsed.behavior],
        load_json(parsed.specification),
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
