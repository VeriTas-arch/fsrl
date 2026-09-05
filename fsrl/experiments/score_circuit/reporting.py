"""Publish immutable copies and report all fixed circuit cells and boundaries."""

import copy
import shutil

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive

from .evidence import RECORDS, specification
from .verification import verify_result, verify_run

RESULT = RECORDS / "results/score_circuit_v1.json"


def copy_record(ref: dict) -> dict:
    source = verify_reference(ref)
    destination = RECORDS / "results" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if file_sha256(destination) != ref["sha256"]:
        raise RuntimeError("registered circuit copy differs")
    return reference(destination)


def estimate_table(title: str, estimates: dict) -> list[str]:
    lines = [
        "",
        title,
        "",
        "| Endpoint | Mean | 95% lower | 95% upper | N |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in estimates.items():
        interval = value["bootstrap"]
        lines.append(
            f"| {name} | {value['mean']:.7f} | {interval['lower']:.7f} | {interval['upper']:.7f} | {value['subjects']} |"
        )
    return lines


def report_text(result: dict, verification: dict) -> str:
    lines = [
        "# Finite-time opponent score circuit",
        "",
        f"Registered outcome: **`{result['outcome']}`**.",
        "",
        "Three exposed, frozen score-only fits (2111/2112/2113), not new human samples. No training, calibration or added memory trace. All saved generic and Liu inputs were locked together before evaluation.",
        "",
        f"Protocol: `{result['protocol_commit']}`; implementation: `{result['source_commit']}`; evaluation-lock commit: `{result['execution_commit']}`.",
        "",
        "The candidate directly integrates 30 bounded nonnegative efficacies and six effective opponent compartment/error states. Updates read centered presynaptic activity and dynamically generated compartment mismatch. Normalization is a shunting-like error-state dynamic driven by pooled binary-contrast activity. Baseline subtraction, opponent teaching, external stable admission and neutral task initialization are explicit assumptions.",
        "",
        "Time units are dimensionless. A rate-level realization is not a complete conductance/spike model, an anatomical attribution, or evidence for a human neural mechanism. Historical quantitative failures are not repaired by this test.",
        "",
        f"Independent verification: `{verification['passed']}`, {verification['estimates']} estimates reconstructed; maximum error {verification['maximum_estimate_error']:.3g}.",
        "",
        "All paired intervals use 10,000 resamples, separately within each fit/domain. Undefined group subjects are excluded before resampling and their indices are retained in JSON. No pooling across fits.",
    ]
    for seed, fit in result["fits"].items():
        lines += [
            "",
            f"## Fit {seed}",
            "",
            f"Parameters: `{fit['parameters']}`.",
            "",
            f"Decision: `{fit['decision']}`.",
            "",
            "| Cell | Max state error | Max margin error | Bound hits | Minimum efficacy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, case in fit["cases"].items():
            physical = case["physical"].values()
            lines.append(
                f"| {name} | {max(case['trajectory_errors'].values()):.7f} | {max(case['margin_errors'].values()):.7f} | {sum(row['bound_engagements'] for row in physical)} | {min(row['minimum_efficacy'] for row in physical):.7f} |"
            )
        lines += [
            "",
            "### Numerical and query checks",
            "",
            f"`{fit['reference_checks']}`",
            "",
            f"Step refinement: `{fit['refinement']}`",
            "",
            "### Sampled behavior: original definitions",
            "",
            "| Row | Parent qualitative | Circuit qualitative | Parent quantitative | Circuit quantitative |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, flags in fit["behavior"]["flags"].items():
            parent = fit["parent_behavior"]["flags"][name]
            lines.append(
                f"| {name} | {parent['qualitative']} | {flags['qualitative']} | {parent['calibration']} | {flags['calibration']} |"
            )
        for name, case in fit["cases"].items():
            lines += estimate_table(
                f"### {name}: absolute endpoints", case["endpoints"]
            )
            lines += estimate_table(
                f"### {name}: paired circuit minus original", case["paired_differences"]
            )
    lines += ["", "## Preserved boundaries", ""]
    lines += [f"- {boundary}" for boundary in result["boundaries"]]
    lines += ["", "## Stop rule", "", result["stop_rule"], ""]
    return "\n".join(lines)


def publish(directory) -> dict:
    verified = verify_run(directory)
    result = copy.deepcopy(load_json(directory / "result.json"))
    result["runtime_source_result"] = reference(directory / "result.json")
    result["registered_run_manifest"] = copy_record(reference(directory / "run.json"))
    for fit in result["fits"].values():
        fit["reference"]["arrays"] = copy_record(fit["reference"]["arrays"])
        for case in fit["cases"].values():
            case["arrays"] = copy_record(case["arrays"])
        fit["check_arrays"] = copy_record(fit["check_arrays"])
        fit["sampled_behavior"] = copy_record(fit["sampled_behavior"])
    # This second verification uses only registered arrays, not ignored run data.
    registered_validation = verify_result(result)
    if registered_validation != verified:
        raise RuntimeError("published arrays do not reproduce runtime verification")
    write_json_exclusive(RESULT, result)
    validation = RECORDS / "results/score_circuit_v1.validation.json"
    write_json_exclusive(validation, verified)
    report = RECORDS / "reports/score_circuit_v1.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        handle.write(report_text(result, verified))
    return {
        "result": reference(RESULT),
        "validation": reference(validation),
        "report": reference(report),
        "stop_rule": specification()["stop_rule"],
    }
