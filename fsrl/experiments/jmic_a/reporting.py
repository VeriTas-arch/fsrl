"""Deterministic report for the frozen JMIC-A result."""

from __future__ import annotations

from collections import Counter

from fsrl.infra.provenance import load_json

from .locks import validate_source_input_lock
from .protocol import REPORT, RESULT, register


def _cell(value: float) -> str:
    return f"{value:.6f}"


def report() -> dict:
    validate_source_input_lock()
    result = load_json(RESULT)
    if REPORT.exists():
        return result
    lines = [
        "# JMIC-A v1: analytic qualification and frozen predictions",
        "",
        "## Outcome",
        "",
        f"- Implementation: `{result['implementation_status']}`",
        f"- Identification: `{result['identification_status']}`",
        "- Human-model status: not evaluated; no participant responses were read.",
        "- Neural status: not evaluated; no JMIC-P model was trained.",
        "",
        (
            "The result qualifies or limits one fixed continuous Gaussian inference "
            "family. It is not a human rescue result and does not authorize model "
            "promotion."
        ),
        "",
        "## Exact structural predictions",
        "",
        f"- Cycle uncertainty classes: {result['prediction_status']['exact_identities']['cycle_variance_classes']}.",
        "- A/B magnitude placement changes posterior means but not posterior covariance.",
        (
            "- Trialwise marginal and persistent-continuous readouts have identical "
            "pair marginals; their maximum recorded discrepancy is "
            f"`{result['prediction_status']['exact_identities']['trialwise_persistent_pair_marginal_max_abs_error']:.3g}`."
        ),
        "- V1 defines no repetition-count learning law.",
        "",
        "## Identification across all fixed cells",
        "",
        "| sigma/tau0 | theta/tau0 | Jacobian condition | SBC | large recovery |",
        "|---:|---:|---:|:---:|:---:|",
    ]
    for row in result["cells"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(row["sigma_eff_over_tau0"]),
                    _cell(row["theta_dec_over_tau0"]),
                    _cell(row["jacobian"]["condition_number"]),
                    "pass" if row["sbc"]["passed"] else "fail",
                    "pass" if row["large_recovery_passed"] else "fail",
                )
            )
            + " |"
        )
    lines += ["", "## Frozen A/B predictions", ""]
    for readout, estimands in result["prediction_status"][
        "A_B_estimand_directions"
    ].items():
        lines += [
            f"### `{readout}`",
            "",
            "| estimand | direction | range |",
            "|---|---|---:|",
        ]
        for name, values in estimands.items():
            lines.append(
                f"| {name} | {values['direction']} | "
                f"[{values['minimum']:.6f}, {values['maximum']:.6f}] |"
            )
        lines.append("")
    morphology = result["morphology_compatibility"]
    lines += [
        "## Descriptive historical morphology compatibility",
        "",
        (
            "These rows use fixed qualitative definitions without human calibration "
            "intervals or an evidence-binding intervention. They did not select "
            "parameters."
        ),
        "",
        "| readout | all-nine cells | constrained-without-binding cells |",
        "|---|---:|---:|",
    ]
    for readout in result["axes"]["readouts"]:
        selected = [row for row in morphology if row["readout"] == readout]
        lines.append(
            f"| {readout} | {sum(row['all_nine_qualitative'] for row in selected)}/9 | "
            f"{sum(row['constrained_without_evidence_binding'] for row in selected)}/9 |"
        )
    stage_counts = Counter(
        "all_nine" if row["all_nine_qualitative"] else "not_all_nine"
        for row in morphology
    )
    lines += [
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
        (
            "The next scientific step is prediction registration against independent "
            "behavioral conditions or an explicitly computational neural-implementation "
            "study. Synthetic calibration alone does not start JMIC-P."
        ),
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    status = (
        "supporting"
        if result["implementation_status"] == "qualified"
        and result["identification_status"] == "qualified_for_prediction_registration"
        else "inconclusive"
    )
    register(
        status=status,
        finding=(
            "Frozen analytic qualification: "
            f"{result['implementation_status']}; {result['identification_status']}; "
            f"descriptive morphology rows {dict(stage_counts)}. No human responses "
            "or neural models were used."
        ),
    )
    return result


__all__ = ["report"]
