"""Preserve four-cell behavior and compare architectures on matched subjects."""

import numpy as np

from fsrl.experiments.local_memory_removal.statistics import (
    noninferiority,
    paired_probability,
    probabilities,
)
from fsrl.experiments.observation_replication.reporting import read_raw, summarize
from fsrl.experiments.observation_replication.statistics import panel_draws, panel_mean
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.experiments.write_cost.reporting import copy_artifact
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .compact import export
from .execution import validate_models
from .protocol import (
    PROTOCOL,
    RECORDS,
    RUNS,
    analysis_seed,
    parents,
    recipe,
    specification,
)


def cell_data(seed, panel, source):
    new_rows, new_raw, old_raw = {}, {}, {}
    for cell in specification()["design"]["cells"]:
        directory = RUNS / "evaluation" / str(seed) / str(panel) / cell
        new_rows[cell] = completed(directory)
        new_raw[cell] = read_raw(reference(directory / "raw.npz"))
        old_raw[cell] = read_raw(
            source["parent_artifacts"][f"evaluation/{seed}/{panel}/{cell}/raw.npz"]
        )
        for key in ("signs", "learned", "episode_indices"):
            if not np.array_equal(
                new_raw[cell]["generic"][key], old_raw[cell]["generic"][key]
            ):
                raise RuntimeError("cross-architecture generic identity mismatch")
    return new_rows, new_raw, old_raw


def panel_result(seed, panel, source, prior):
    rows, candidate, hint_present = cell_data(seed, panel, source)
    old_rows = prior["pairs"][str(seed)]["panels"][str(panel)]["single"]["cells"]
    cpu = load_input(source["panels"][str(panel)]["inputs"]["liu-8"])
    sample_seed, spec = analysis_seed(seed, panel), specification()
    summary, endpoints = summarize(
        candidate, rows, cpu, sample_seed, recipe(panel), spec
    )
    _, old_endpoints = summarize(
        hint_present, old_rows, cpu, sample_seed, recipe(panel), spec
    )
    boot = spec["statistics"]["seed_offset"] + sample_seed
    new_draws = panel_draws(
        candidate, endpoints["CE"], endpoints["order_shift"], boot, spec
    )
    old_draws = panel_draws(
        hint_present, old_endpoints["CE"], old_endpoints["order_shift"], boot, spec
    )
    difference = tuple(
        {key: new[key] - old[key] for key in new}
        for new, old in zip(new_draws, old_draws, strict=True)
    )
    ni, ni_draws = {}, {}
    for cell in candidate:
        values, subjects = paired_probability(
            probabilities(candidate[cell]), probabilities(hint_present[cell]), boot
        )
        ni_draws[cell] = values
        intervals, _ = panel_mean([values])
        ni[cell] = {
            "estimates": intervals,
            "subjects": subjects,
            "decision": noninferiority(intervals),
        }
    anchor = read_raw(
        source["anchor_artifacts"][f"evaluation/{seed}/{panel}/Ce/raw.npz"]
    )
    cumulative, cumulative_subjects = paired_probability(
        probabilities(candidate["Ce"]), probabilities(anchor), boot
    )
    cumulative_summary, _ = panel_mean([cumulative])
    return {
        "candidate": summary,
        "noninferiority": ni,
        "cumulative_Ce_probability": {
            "estimates": cumulative_summary,
            "subjects": cumulative_subjects,
        },
    }, (new_draws, difference, ni_draws, cumulative)


def decision(panels, mean, ni):
    result = {
        "all_candidate_cells_competent": all(
            p["candidate"]["decision"]["all_four_competent"] for p in panels.values()
        ),
        "Ce_five_endpoints_noninferior": all(
            row["noninferior"] for row in noninferiority(ni["Ce"]).values()
        ),
        "Ae_and_Ce_core_present": all(
            any(
                p["candidate"]["cells"][cell]["liu"]["routes"]["full"]["core_passed"]
                for p in panels.values()
            )
            for cell in ("Ae", "Ce")
        ),
        "Ae_Ce_binding_retained": all(
            p["candidate"]["evidence_binding"][cell]
            for p in panels.values()
            for cell in ("Ae", "Ce")
        ),
        "positive_I_tau": mean["interaction/global_all77_tau"]["interval"]["lower"] > 0,
        "negative_I_CE": mean["interaction/generic_global"]["interval"]["upper"] < 0,
        "negative_shift_difference": mean["order_shift/C_minus_A"]["interval"]["upper"]
        < 0,
        "both_error_tau_negative": all(
            mean[key + "/global_all77_tau"]["interval"]["upper"] < 0
            for key in ("error_A", "error_C")
        ),
    }
    result["removal_supported_for_pair"] = all(result.values())
    return result


def pair_result(seed, source, prior):
    panels, draws = {}, []
    for panel in specification()["design"]["panels"]:
        panels[str(panel)], values = panel_result(seed, panel, source, prior)
        draws.append(values)
    mean, mean_draws = panel_mean([row[0] for row in draws])
    delta, delta_draws = panel_mean([row[1] for row in draws])
    ni, ni_draws = {}, {}
    for cell in specification()["design"]["cells"]:
        ni[cell], ni_draws[cell] = panel_mean([row[2][cell] for row in draws])
    cumulative, cumulative_draws = panel_mean([row[3] for row in draws])
    result = {
        "panels": panels,
        "cumulative_Ce_probability": cumulative,
        "candidate_equal_panel_mean": mean,
        "candidate_minus_hint_present_effects": delta,
        "noninferiority_equal_panel_mean": ni,
        "decision": decision(panels, mean, ni),
    }
    return result, {
        "candidate": mean_draws,
        "candidate_minus_hint_present": delta_draws,
        "noninferiority": ni_draws,
        "cumulative_Ce_probability": cumulative_draws,
    }


def report():
    source, models = validate_models()
    prior, spec = parents()["result"], specification()
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="admission_hint_removal_v1",
        execution_id="matched-hint-comparison",
        producer={"source_commit": source["source_commit"]},
        resolved_config=spec["noninferiority"],
    ):
        pairs, raw = {}, {}
        for seed in spec["design"]["seeds"]:
            pairs[str(seed)], raw[str(seed)] = pair_result(seed, source, prior)
        result = {
            "protocol": reference(PROTOCOL),
            "hint_present_reference": spec["parents"]["result"],
            "dual_anchor": spec["anchor"],
            "pairs": pairs,
            "removal_supported": all(
                row["decision"]["removal_supported_for_pair"] for row in pairs.values()
            ),
            "claim_boundary": spec["claim_boundary"],
        }
        print(
            {
                "hint_removal_supported": result["removal_supported"],
                "pairs": {k: v["decision"] for k, v in pairs.items()},
            },
            flush=True,
        )
        result["compact_export"] = (
            export(source, models)
            if result["removal_supported"]
            else {"executed": False, "reason": "hint-removal support gate not met"}
        )
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(directory / "result.json", json_ready(result))
    archived = {
        str(p.relative_to(RUNS)): copy_artifact(
            p, RECORDS / "artifacts" / p.relative_to(RUNS)
        )
        for p in sorted(RUNS.rglob("*"))
        if p.is_file() and p.suffix in (".json", ".jsonl", ".npz", ".pth")
    }
    write_json_exclusive(
        RECORDS / "results/result.json", json_ready({**result, "artifacts": archived})
    )
    return {
        "removal_supported": result["removal_supported"],
        "pairs": {seed: row["decision"] for seed, row in pairs.items()},
    }
