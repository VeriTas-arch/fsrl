"""Concise report and compact artifact promotion for the successor."""

from __future__ import annotations

from fsrl.infra.provenance import load_json

from .locks import reference, validate_baseline_artifact_lock
from .protocol import ARRAYS, MECHANISM_RUNS, REPORT, RESULT
from .storage import load_npz, write_npz_exclusive


def render(result: dict) -> str:
    if result["outcome"] != "interpretable":
        return "\n".join(
            [
                "# Single-P direct-authority time-role result",
                "",
                "Registered outcome: `noninterpretable`.",
                "",
                f"Integrity failure: {result['failure']}",
                "",
                "No mechanism label is authorized. The parent remains `time_removal_failure`.",
                "",
            ]
        )
    labels = result["classification"]
    lines = [
        "# Single-P direct-authority time-role result",
        "",
        "Registered outcome: `interpretable`.",
        "",
        "The mechanism phase loaded the frozen bitwise-reproducible 32D direct baseline; it did not replay an adapter internal state or regenerate the original baseline.",
        "",
        f"Seed 3011 scale classification: `{'scale_sufficient' if labels['scale_sufficient'] else 'scale_insufficient'}`.",
        "",
        "| seed | support-time path | query-time path | no-time state dependence | no-time confidence link |",
        "|---:|:---:|:---:|:---:|---|",
    ]
    for seed in ("3011", "3012", "3013"):
        time = labels["per_seed_time_paths"][seed]
        state = labels["per_seed_state_dynamics"][seed]["clean_no_time"]
        lines.append(
            f"| {seed} | {'present' if time['support_time_path_present'] else 'absent'} | "
            f"{'present' if time['query_time_path_present'] else 'absent'} | "
            f"{'present' if state['state_dependent_plasticity_present'] else 'absent'} | "
            f"{state['confidence_link']} |"
        )
    lines += [
        "",
        f"Mechanism heterogeneity: `{labels['mechanism_heterogeneous']}`.",
        "",
        "These are frozen-model, three-network mechanism diagnostics. Scale sufficiency is not parameter equivalence; acute time interventions are not training without time; state dependence is not a human or biological confidence mechanism. The parent outcome remains `time_removal_failure`.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    validate_baseline_artifact_lock()
    result = load_json(RESULT)
    if result["outcome"] == "interpretable":
        arrays = load_npz(MECHANISM_RUNS / "arrays.npz")
        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        write_npz_exclusive(ARRAYS, arrays)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(result), encoding="utf-8")
    return {
        "outcome": result["outcome"],
        "report": reference(REPORT),
        **({"arrays": reference(ARRAYS)} if ARRAYS.exists() else {}),
    }


__all__ = ["render", "write_report"]
