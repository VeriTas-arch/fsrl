"""Freeze and report the completed cross-talk decomposition."""

from __future__ import annotations

from pathlib import Path

from fsrl.infra.provenance import file_sha256, load_json

from .locks import (
    ACTIVE_SOURCE_LOCK_PATH,
    ARRAY_PATH,
    REPORT_PATH,
    RESULT_PATH,
    RUNTIME_ARRAY_PATH,
    RUNTIME_RESULT_PATH,
    reference,
    validate_source_lock,
)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def render_report(result: dict) -> str:
    if result["outcome"] != "exact_cross_talk_decomposed":
        return (
            "# Exact P/L retained cross-talk decomposition\n\n"
            "The frozen diagnostic is non-interpretable because an input or "
            "exact-algebra check failed.\n"
            "No component magnitude, counterfactual, or next-model conclusion "
            "is admitted.\n"
            "The parent functional-replication result remains unchanged.\n"
        )

    per_seed = result["per_seed"]
    factorial = result["gain_by_operating_point_factorial"]
    diagnostics = result["factorial_diagnostics"]
    concentration = result["source_concentration"]["participant_weighted"]
    rows = [
        "# Exact P/L retained cross-talk decomposition",
        "",
        "## Result",
        "",
        "All locked inputs and exact-algebra checks passed. The gain-free omitted-write cross-talk was identical across seeds, and its source sum reconstructed the dual-minus-shared local and probability effects within the frozen tolerances.",
        "",
        "The parent result is unchanged: seed 3006 remains a formal retained-fidelity failure, and the three-seed P/L replication remains `competent_alternative_organization`.",
        "",
        "## Seed decomposition",
        "",
        "| seed | local gain | retained mean effect | bootstrap lower | shared-total sensitivity | P-only sensitivity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in (3004, 3005, 3006):
        row = per_seed[str(seed)]
        contrast = row["reconstructed_retained_contrast"]
        cell = row["retained_cell_summaries"]
        rows.append(
            "| {seed} | {gain:.6f} | {mean} | {lower} | {shared:.6f} | {p_only:.6f} |".format(
                seed=seed,
                gain=row["local_gain"],
                mean=_percent(contrast["mean"]),
                lower=_percent(contrast["bootstrap"]["lower"]),
                shared=cell["shared_total_probability_sensitivity"]["mean"],
                p_only=cell["P_only_probability_sensitivity"]["mean"],
            )
        )

    rows.extend(
        [
            "",
            "The exact probability operating point is the shared total margin (`P + shared L`); the P-only sensitivity is shown separately and was not used for reconstruction.",
            "",
            "## Gain x operating-point factorial",
            "",
            "Each cell is the retained mean effect, with `pass` referring only to the historical -0.005 lower-bound reference. It does not revise the parent decision.",
            "",
            "| operating seed | gain 3004 | gain 3005 | gain 3006 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for operating in (3004, 3005, 3006):
        cells = []
        for gain in (3004, 3005, 3006):
            cell = factorial[str(operating)][str(gain)]
            label = "pass" if cell["frozen_reference_gate_passed"] else "fail"
            cells.append(f"{_percent(cell['summary']['mean'])} ({label})")
        rows.append(f"| {operating} | " + " | ".join(cells) + " |")

    rows.extend(
        [
            "",
            f"Registered factorial classification: `{diagnostics['classification']}`.",
            "",
            "The Shapley splits below attribute the difference in retained mean effect exactly within the fixed factorial; they do not explain how training produced either component.",
            "",
            "| comparison | gain component | operating-point component | total |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostics["mean_shapley_splits"]:
        rows.append(
            "| {seed}->3006 | {gain} | {operating} | {total} |".format(
                seed=row["passing_reference_seed"],
                gain=_percent(row["gain_component"]),
                operating=_percent(row["operating_point_component"]),
                total=_percent(row["total_mean_difference"]),
            )
        )

    rows.extend(
        [
            "",
            "## Source pattern",
            "",
            "The source analysis is descriptive rather than dichotomized. Participant-weighted medians were:",
            "",
            f"- top-one absolute share: {concentration['top_one_share']['median']:.3f}",
            f"- top-two absolute share: {concentration['top_two_share']['median']:.3f}",
            f"- effective source count: {concentration['effective_source_count']['median']:.3f}",
            f"- cancellation ratio: {concentration['cancellation_ratio']['median']:.3f}",
            f"- sources needed for 80% absolute mass: {concentration['source_count_80pct']['median']:.3f}",
            "",
            "## Interpretation boundary",
            "",
            "The retained cost is completely routed through omitted writes and nonorthogonal fixed addresses. Across-network variation is not a change in address geometry: it enters through the learned scalar gain and the frozen shared-margin operating point. This establishes deterministic component attribution, not a causal account of joint training.",
            "",
            "No new training, quantitative-human fitting, or N transport was run. Given that the dual-store computation is already functionally informative but structurally non-necessary, this diagnostic does not justify further P/L repair by itself; the clean P/L model can remain the functional-decomposition model while work returns to the single-P structural-sufficiency mainline.",
            "",
        ]
    )
    return "\n".join(rows)


def freeze_outputs() -> dict:
    validate_source_lock()
    result = load_json(RUNTIME_RESULT_PATH)
    if result["source_lock"] != reference(ACTIVE_SOURCE_LOCK_PATH):
        raise RuntimeError("runtime result cites a different source lock")
    runtime_arrays = reference(RUNTIME_ARRAY_PATH)
    expected_arrays = result["supporting_arrays"]["runtime_source"]
    if runtime_arrays != expected_arrays:
        raise RuntimeError("runtime supporting arrays changed")
    if result["supporting_arrays"]["sha256"] != file_sha256(RUNTIME_ARRAY_PATH):
        raise RuntimeError("runtime result misstates the supporting-array hash")

    _write_bytes_exclusive(ARRAY_PATH, RUNTIME_ARRAY_PATH.read_bytes())
    if reference(ARRAY_PATH) != {
        key: result["supporting_arrays"][key] for key in ("path", "sha256", "bytes")
    }:
        raise RuntimeError("frozen supporting arrays differ from the runtime artifact")
    _write_bytes_exclusive(RESULT_PATH, RUNTIME_RESULT_PATH.read_bytes())
    _write_bytes_exclusive(REPORT_PATH, render_report(result).encode("utf-8"))
    return {
        "result": reference(RESULT_PATH),
        "supporting_arrays": reference(ARRAY_PATH),
        "report": reference(REPORT_PATH),
        "outcome": result["outcome"],
    }
