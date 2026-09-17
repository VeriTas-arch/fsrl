"""Canonical result and report for the historical morphology re-audit."""

from __future__ import annotations

import shutil

from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive

from .locks import reference, validate_source_input_lock
from .protocol import PAIR_TABLE, REPORT, RESULT, RUNS, register, specification


def _copy_exclusive(source, target) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if file_sha256(source) != file_sha256(target):
        raise RuntimeError("registered pair table differs from analysis output")


def _render(result: dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# Current-standard morphology re-audit of historical alpha single-P",
        "",
        f"Registered outcome: `{aggregate['outcome']}`.",
        "",
        (
            f"Across all 18 clean-trained acute-noisy Ae units, "
            f"{aggregate['constrained_units']} met the later constrained morphology "
            f"standard, {aggregate['all_nine_units']} passed all nine qualitative rows, "
            f"and {aggregate['error_inflation_units']} met the paired A0-to-Ae error-"
            "inflation definition."
        ),
        "",
        "## Family-specific results",
        "",
        "| historical family | competent | binding | all nine | constrained | error inflation | seeds with constrained panel | replicated precedent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in aggregate["families"].items():
        lines.append(
            f"| {name} | {row['inherited_competent_units']}/9 | "
            f"{row['evidence_binding_units']}/9 | {row['all_nine_units']}/9 | "
            f"{row['constrained_units']}/9 | {row['error_inflation_units']}/9 | "
            f"{row['seeds_with_constrained_panel']}/3 | "
            f"{str(row['replicated_precedent']).lower()} |"
        )
    lines += [
        "",
        "| historical family | Ae stage distribution | mean latent bimodal pairs | mean sampled bimodal pairs | mean top-5 strong-error share |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in aggregate["families"].items():
        lines.append(
            f"| {name} | `{row['stage_counts']}` | "
            f"{row['latent_bimodal_pairs_mean']:.3f} | "
            f"{row['sampled_bimodal_pairs_mean']:.3f} | "
            f"{row['strong_error_top5_share_mean']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        (
            "The historical five-core and compensation findings remain exactly as "
            "registered. This audit asks the narrower retrospective question of whether "
            "those archived success cells also met a later, stricter pair-morphology "
            "standard; it does not rewrite the original gates."
        ),
        "",
        (
            "All 36 mandatory A0/Ae units were retained. Full and global sampled behavior "
            "was deterministically replayed, and archived fields, probabilities, Hodge "
            "potentials, and decomposition identities passed the frozen tolerances."
        ),
        "",
        (
            "No training, checkpoint loading, model inference, rescaling, extra choices, "
            "participant pooling, or selection was performed. The protocol stops here and "
            "does not authorize another alpha recipe or initialization factorial."
        ),
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    validate_source_input_lock()
    runtime_result = RUNS / "analysis/result.json"
    runtime_table = RUNS / "analysis/pair_table.npz"
    result = load_json(runtime_result)
    if len(result["units"]) != specification()["design"]["mandatory_units"]:
        raise RuntimeError("analysis did not retain all mandatory historical units")
    _copy_exclusive(runtime_table, PAIR_TABLE)
    result["pair_table"] = reference(PAIR_TABLE)
    write_json_exclusive(RESULT, result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(_render(result))
    aggregate = result["aggregate"]
    outcome = aggregate["outcome"]
    status = {
        "no_current_standard_precedent": "valid_negative",
        "isolated_current_standard_precedent": "unresolved",
        "single_family_current_standard_precedent": "supporting",
        "cross_family_current_standard_precedent": "supporting",
    }[outcome]
    register(
        status=status,
        finding=(
            f"Read-only historical re-audit outcome {outcome}: "
            f"{aggregate['constrained_units']}/18 acute-noisy Ae units met constrained "
            f"morphology and {aggregate['all_nine_units']}/18 passed all nine rows; no "
            "training or model execution was performed."
        ),
    )
    return {
        "outcome": outcome,
        "constrained_units": aggregate["constrained_units"],
        "result": reference(RESULT),
        "report": reference(REPORT),
    }


__all__ = ["write_report"]
