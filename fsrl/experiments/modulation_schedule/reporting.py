"""Prespecified paired intervention effects, without pooling networks."""

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.admission_hint_removal.reporting import decision
from fsrl.experiments.local_memory_removal.statistics import (
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
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate
from .protocol import (
    PROTOCOL,
    RECORDS,
    RUNS,
    SOURCE,
    analysis_seed,
    recipe,
    specification,
)


def write_effects(new, old, seed, samples):
    points, draws = {}, {}
    for cell in new:
        for task in ("generic", "liu"):
            a = new[cell][task]["total_write"].astype(float)
            b = old[cell][task]["total_write"].astype(float)
            counts = bootstrap_counts(np.random.default_rng(seed), samples, len(a))
            key = f"{cell}/{task}/write_difference"
            points[key], draws[key] = (a - b).mean(), counts @ (a - b) / len(a)
            key = f"{cell}/{task}/write_ratio"
            assert b.mean() > 0 and np.all(counts @ b > 0)
            points[key], draws[key] = a.mean() / b.mean(), (counts @ a) / (counts @ b)
    return points, draws


def panel_result(mode, seed, panel, prior):
    rows, new, old = {}, {}, {}
    for cell in prior["protocol"]["design"]["cells"]:
        directory = RUNS / mode / str(seed) / str(panel) / cell
        rows[cell] = completed(directory)
        new[cell] = read_raw(reference(directory / "raw.npz"))
        old[cell] = read_raw(
            prior["result"]["artifacts"][f"evaluation/{seed}/{panel}/{cell}/raw.npz"]
        )
        for key in ("signs", "learned", "episode_indices"):
            np.testing.assert_array_equal(
                new[cell]["generic"][key], old[cell]["generic"][key]
            )
    cpu = load_input(prior["source"]["panels"][str(panel)]["inputs"]["liu-8"])
    spec, sample = prior["protocol"], analysis_seed(seed, panel)
    baseline_rows = prior["result"]["pairs"][str(seed)]["panels"][str(panel)][
        "candidate"
    ]["cells"]
    candidate, endpoints = summarize(new, rows, cpu, sample, recipe(panel), spec)
    baseline, original = summarize(old, baseline_rows, cpu, sample, recipe(panel), spec)
    boot = spec["statistics"]["seed_offset"] + sample
    a = panel_draws(new, endpoints["CE"], endpoints["order_shift"], boot, spec)
    b = panel_draws(old, original["CE"], original["order_shift"], boot, spec)
    delta = tuple(
        {key: x[key] - y[key] for key in x} for x, y in zip(a, b, strict=True)
    )
    probability, ni = {}, {}
    for cell, cell_raw in new.items():
        values, subjects = paired_probability(
            probabilities(cell_raw), probabilities(old[cell]), boot
        )
        probability[cell] = values
        estimates, _ = panel_mean([values])
        ni[cell] = {"estimates": estimates, "subjects": subjects}
    writes = write_effects(new, old, boot, spec["statistics"]["samples"])
    return {"candidate": candidate, "baseline": baseline, "noninferiority": ni}, (
        a,
        b,
        delta,
        probability,
        writes,
    )


def summarize_pair(mode, seed, prior):
    panels, data = {}, []
    for panel in specification()["evaluation_panels"]:
        panels[str(panel)], values = panel_result(mode, seed, panel, prior)
        data.append(values)
    mean, mean_raw = panel_mean([v[0] for v in data])
    baseline, baseline_raw = panel_mean([v[1] for v in data])
    delta, delta_raw = panel_mean([v[2] for v in data])
    writes, writes_raw = panel_mean([v[4] for v in data])
    ni, ni_raw = {}, {}
    for cell in prior["protocol"]["design"]["cells"]:
        ni[cell], ni_raw[cell] = panel_mean([v[3][cell] for v in data])
    flags = decision(panels, mean, ni)
    result = {
        "panels": panels,
        "candidate_equal_panel_mean": mean,
        "baseline_equal_panel_mean": baseline,
        "candidate_minus_affine": delta,
        "write_changes": writes,
        "noninferiority_equal_panel_mean": ni,
        "decision": flags,
    }
    raw = {
        "candidate": mean_raw,
        "baseline": baseline_raw,
        "difference": delta_raw,
        "writes": writes_raw,
        "noninferiority": ni_raw,
    }
    return result, raw


def report(mode):
    prior = validate(calibrated=True, gpu=False)
    directory = RUNS / f"{mode}_comparison"
    pairs, raw = {}, {}
    with ProspectiveRun.start(
        directory,
        workflow_id="modulation_schedule_v1",
        execution_id=f"{mode}-comparison",
        producer={"source": reference(SOURCE)},
        resolved_config=specification()["decision"],
    ):
        for seed in specification()["seeds"]:
            pairs[str(seed)], raw[str(seed)] = summarize_pair(mode, seed, prior)
            print(
                {"mode": mode, "seed": seed, "decision": pairs[str(seed)]["decision"]},
                flush=True,
            )
        result = {
            "mode": mode,
            "pairs": pairs,
            "preserved_in_all_pairs": all(
                v["decision"]["removal_supported_for_pair"] for v in pairs.values()
            ),
        }
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return {"mode": mode, "preserved_in_all_pairs": result["preserved_in_all_pairs"]}


def finalize():
    validate(calibrated=True, gpu=False)
    phase = load_json(RUNS / "phase_comparison/result.json")
    constant = (
        load_json(RUNS / "constant_comparison/result.json")
        if phase["preserved_in_all_pairs"]
        else {"executed": False, "reason": "phase preservation gate not met"}
    )
    result = {
        "protocol": reference(PROTOCOL),
        "audit": reference(RECORDS / "results/audit.json"),
        "phase": phase,
        "constant": constant,
        "compact_diagnostic": load_json(RUNS / "compact_diagnostic/result.json"),
    }
    result["constant_training_triggered"] = bool(
        constant.get("preserved_in_all_pairs", False)
    )
    archived = {
        str(p.relative_to(RUNS)): copy_artifact(
            p, RECORDS / "artifacts" / p.relative_to(RUNS)
        )
        for p in sorted(RUNS.rglob("*"))
        if p.is_file() and p.suffix in (".json", ".npz")
    }
    write_json_exclusive(
        RECORDS / "results/result.json", json_ready({**result, "artifacts": archived})
    )
    return {
        "phase_preserved": phase["preserved_in_all_pairs"],
        "constant_training_triggered": result["constant_training_triggered"],
        "archived": len(archived),
    }
