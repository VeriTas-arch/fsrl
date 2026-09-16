"""Freeze and report the completed time-role decomposition."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.infra.provenance import load_json, write_json_exclusive

from .locks import reference, validate_source_lock
from .protocol import ARRAYS, REPORT, RESULT, RUNS


def write_report() -> dict:
    validate_source_lock()
    runtime_result = RUNS / "result.json"
    runtime_arrays = RUNS / "arrays.npz"
    result = load_json(runtime_result)
    with np.load(runtime_arrays, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    write_arrays(ARRAYS, arrays)
    result["sufficient_arrays"] = reference(ARRAYS)
    write_json_exclusive(RESULT, result)
    labels = result["classification"]
    lines = [
        "# Frozen single-P time-role decomposition",
        "",
        f"Parent outcome remains `{result['parent_outcome_unchanged']}`.",
        "",
        "## Mechanistic labels",
        "",
        f"- Scale sufficient for seed 3011: `{labels['scale_sufficient']}`",
        f"- Scale insufficient for seed 3011: `{labels['scale_insufficient']}`",
        f"- Mechanism heterogeneous across the three development instances: `{labels['mechanism_heterogeneous']}`",
        "",
        "| seed | support-time path | query-time path | no-time state dependence | no-time confidence link |",
        "|---:|:---:|:---:|:---:|:---:|",
    ]
    for seed in ("3011", "3012", "3013"):
        paths = labels["per_seed_time_paths"][seed]
        dynamics = labels["per_seed_state_dynamics"][seed]["clean_no_time"]
        lines.append(
            f"| {seed} | {paths['support_time_path_present']} | {paths['query_time_path_present']} | "
            f"{dynamics['state_dependent_plasticity_present']} | {dynamics['confidence_link_supported']} |"
        )
    lines.extend(
        [
            "",
            "The labels diagnose frozen model computations. They do not repair seed 3011, admit a no-time recipe, establish a human confidence mechanism, or turn three networks into a population sample.",
            "",
            "The parent evaluator constructed two legacy blank steps despite the registered zero-blank interface. Algebra and replay show that this path leaves P exactly zero and discards its h/E outputs before support; it is disclosed as an inert implementation deviation, not treated as a scientific intervention.",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "result": reference(RESULT),
        "arrays": reference(ARRAYS),
        "report": reference(REPORT),
        "classification": labels,
    }


__all__ = ["write_report"]
