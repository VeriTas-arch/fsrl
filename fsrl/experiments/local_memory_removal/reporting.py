"""Preserve four-cell behavior and compare architectures on matched subjects."""

import numpy as np

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
from .statistics import noninferiority, paired_probability, probabilities


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
    rows, single, dual = cell_data(seed, panel, source)
    old_rows = prior["pairs"][str(seed)]["panels"][str(panel)]["cells"]
    cpu = load_input(source["panels"][str(panel)]["inputs"]["liu-8"])
    sample_seed, spec = analysis_seed(seed, panel), specification()
    summary, endpoints = summarize(single, rows, cpu, sample_seed, recipe(panel), spec)
    _, old_endpoints = summarize(dual, old_rows, cpu, sample_seed, recipe(panel), spec)
    boot = spec["statistics"]["seed_offset"] + sample_seed
    new_draws = panel_draws(
        single, endpoints["CE"], endpoints["order_shift"], boot, spec
    )
    old_draws = panel_draws(
        dual, old_endpoints["CE"], old_endpoints["order_shift"], boot, spec
    )
    difference = tuple(
        {key: new[key] - old[key] for key in new}
        for new, old in zip(new_draws, old_draws, strict=True)
    )
    ni, ni_draws = {}, {}
    for cell in single:
        values, subjects = paired_probability(
            probabilities(single[cell]), probabilities(dual[cell]), boot
        )
        ni_draws[cell] = values
        intervals, _ = panel_mean([values])
        ni[cell] = {
            "estimates": intervals,
            "subjects": subjects,
            "decision": noninferiority(intervals),
        }
    return {"single": summary, "noninferiority": ni}, (new_draws, difference, ni_draws)


def decision(panels, mean, ni):
    result = {
        "all_single_cells_competent": all(
            p["single"]["decision"]["all_four_competent"] for p in panels.values()
        ),
        "Ce_five_endpoints_noninferior": all(
            row["noninferior"] for row in noninferiority(ni["Ce"]).values()
        ),
        "Ae_and_Ce_core_present": all(
            any(
                p["single"]["cells"][cell]["liu"]["routes"]["full"]["core_passed"]
                for p in panels.values()
            )
            for cell in ("Ae", "Ce")
        ),
        "Ae_Ce_binding_retained": all(
            p["single"]["evidence_binding"][cell]
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
    result = {
        "panels": panels,
        "single_equal_panel_mean": mean,
        "single_minus_dual_effects": delta,
        "noninferiority_equal_panel_mean": ni,
        "decision": decision(panels, mean, ni),
    }
    return result, {
        "single": mean_draws,
        "single_minus_dual": delta_draws,
        "noninferiority": ni_draws,
    }


def report():
    source, _ = validate_models()
    prior, spec = parents()["result"], specification()
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="local_memory_removal_v1",
        execution_id="matched-memory-comparison",
        producer={"source_commit": source["source_commit"]},
        resolved_config=spec["noninferiority"],
    ):
        pairs, raw = {}, {}
        for seed in spec["design"]["seeds"]:
            pairs[str(seed)], raw[str(seed)] = pair_result(seed, source, prior)
        result = {
            "protocol": reference(PROTOCOL),
            "dual_reference": spec["parents"]["result"],
            "pairs": pairs,
            "removal_supported": all(
                row["decision"]["removal_supported_for_pair"] for row in pairs.values()
            ),
            "claim_boundary": spec["claim_boundary"],
        }
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
