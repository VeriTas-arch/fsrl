"""Per-panel four-cell results and equal-panel, within-network summaries."""

from itertools import combinations

import numpy as np

from fsrl.experiments.finite_state.reporting import tau_contrast
from fsrl.experiments.memory_structure.measurement import above
from fsrl.experiments.observation_crossover.evaluation import arrays
from fsrl.experiments.observation_crossover.reporting import contrasts, order_shift
from fsrl.experiments.training_strategy.estimands import estimate
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
from fsrl.paths import STUDIES_ROOT

from .execution import validate_models
from .protocol import RECORDS, RUNS, analysis_seed, recipe, specification
from .statistics import panel_mean


def summarize(raw, rows, cpu, seed, parent, spec):
    for first, second in combinations(raw, 2):
        for key in ("signs", "learned", "episode_indices"):
            if not np.array_equal(
                raw[first]["generic"][key], raw[second]["generic"][key]
            ):
                raise RuntimeError("generic participant/query alignment differs")
    n = parent["evaluation"]["liu"]["subjects"]
    signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T
    temperature = parent["evaluation"]["liu"]["temperature"]
    endpoints = {}
    for cell, data in raw.items():
        g, l = data["generic"], data["liu"]
        endpoints[cell] = {}
        for route, prefix, bundle in (
            ("full", "", "intact"),
            ("global", "global_", "local_off"),
        ):
            ce = np.logaddexp(0, -g[prefix + "margins"] * g["signs"]).mean(1)
            if not np.allclose(ce, g[prefix + "ce"], rtol=0, atol=1e-12):
                raise RuntimeError("generic CE cannot be reconstructed")
            endpoints[cell]["generic_" + route] = ce
            endpoints[cell]["liu_" + route] = np.logaddexp(
                0, -l[f"bundles__{bundle}__logits"] * signs / temperature
            ).mean(1)
        if not all(np.isfinite(v).all() for v in endpoints[cell].values()):
            raise RuntimeError("nonfinite complete-panel endpoint")
    boot = spec["statistics"]["seed_offset"] + seed
    absolute = {
        cell: {
            "tau": {
                f"{route}_{scope}": tau_contrast(
                    raw, {cell: 1}, route, scope == "conditional", boot, spec
                )
                for route in ("full", "global")
                for scope in ("all77", "conditional")
            },
            "CE": {
                key: estimate(value, seed=boot, statistics=spec["statistics"])
                for key, value in endpoints[cell].items()
            },
        }
        for cell in raw
    }
    shifted = {
        r: order_shift(
            raw[r + "0"]["liu"]["routes__global__internal__orders"],
            raw[r + "e"]["liu"]["routes__global__internal__orders"],
        )
        for r in ("A", "C")
    }
    shifted["C_minus_A"] = shifted["C"] - shifted["A"]
    shift = {
        key: estimate(value, seed=boot, statistics=spec["statistics"])
        for key, value in shifted.items()
    }
    compared = contrasts(raw, endpoints, seed, spec)
    competent = all(
        row["generic"]["competence"]
        and row["generic"]["global"]["competence"]
        and all(route["competence"] for route in row["liu"]["routes"].values())
        for row in rows.values()
    )
    interaction = compared["interaction"]
    decision = {
        "all_four_competent": competent,
        "positive_population_interaction": interaction["tau"]["global_all77"][
            "interval"
        ]["lower"]
        > 0,
        "negative_generic_global_CE_interaction": interaction["CE"]["generic_global"][
            "bootstrap"
        ]["upper"]
        < 0,
        "negative_order_shift_difference": shift["C_minus_A"]["bootstrap"]["upper"] < 0,
        "both_error_tau_effects_negative": all(
            compared[key]["tau"]["global_all77"]["interval"]["upper"] < 0
            for key in ("error_A", "error_C")
        ),
    }
    decision["joint_output_compensation"] = all(
        decision[key]
        for key in (
            "all_four_competent",
            "positive_population_interaction",
            "negative_generic_global_CE_interaction",
            "negative_order_shift_difference",
        )
    )
    binding = {
        cell: all(
            above(rows[cell]["liu"]["effects"][f"intact_minus_{control}_nonlearned"], 0)
            for control in ("P_off", "evidence_shuffle")
        )
        for cell in rows
    }
    return {
        "cells": rows,
        "absolute": absolute,
        "contrasts": compared,
        "order_shift": shift,
        "evidence_binding": binding,
        "decision": decision,
    }, {"CE": endpoints, "order_shift": shifted}


def read_raw(ref):
    flat = arrays(ref)
    return {
        phase: {
            key.removeprefix(phase + "__"): value
            for key, value in flat.items()
            if key.startswith(phase + "__")
        }
        for phase in ("generic", "liu")
    }


def qualify_estimators():
    oldroot = STUDIES_ROOT / "observation_crossover/records"
    old = load_json(oldroot / "benchmarks/execution_lock.json")
    result = load_json(oldroot / "results/result.json")
    parent_result = load_json(
        STUDIES_ROOT / "observation_uncertainty/records/results/result.json"
    )
    rows, raw = {}, {}
    for cell, arm in (("A0", "clean"), ("Ae", "acute_noisy"), ("Ce", "noisy")):
        rows[cell] = parent_result["pairs"]["2432"]["models"][arm]
        raw[cell] = {
            phase: arrays(ref)
            for phase, ref in old["previous_cells"]["2432"][cell].items()
        }
    directory = oldroot / "artifacts/2432/C0"
    rows["C0"] = load_json(directory / "result.json")
    raw["C0"] = read_raw(reference(directory / "raw.npz"))
    actual, _ = summarize(
        raw, rows, load_input(old["inputs"]["liu-8"]), 2432, recipe(), specification()
    )
    if json_ready(actual) != result["pairs"]["2432"]:
        raise RuntimeError("old four-cell estimator parity failed")
    return {"old_pair": 2432, "complete_summary_exact": True}


def paired_panel(seed, panel, source):
    rows, raw = {}, {}
    for cell in specification()["design"]["cells"]:
        directory = RUNS / "evaluation" / str(seed) / str(panel) / cell
        rows[cell] = completed(directory)
        raw[cell] = read_raw(reference(directory / "raw.npz"))
    cpu = load_input(source["panels"][str(panel)]["inputs"]["liu-8"])
    sample_seed = analysis_seed(seed, panel)
    result, endpoints = summarize(
        raw, rows, cpu, sample_seed, recipe(panel), specification()
    )
    from .statistics import panel_draws

    points, draws = panel_draws(
        raw,
        endpoints["CE"],
        endpoints["order_shift"],
        specification()["statistics"]["seed_offset"] + sample_seed,
        specification(),
    )
    return result, (points, draws)


def report():
    source, _ = validate_models()
    spec = specification()
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="observation_replication_v1",
        execution_id="all-pairs-panels",
        producer={"source": source["source_commit"]},
        resolved_config=spec["statistics"],
    ):
        pairs, raw = {}, {}
        for seed in spec["design"]["seeds"]:
            panels, sampled = {}, []
            for panel in spec["design"]["panels"]:
                panels[str(panel)], values = paired_panel(seed, panel, source)
                sampled.append(values)
            summary, draws = panel_mean(sampled)
            pairs[str(seed)] = {"panels": panels, "equal_panel_mean": summary}
            raw[str(seed)] = draws
        result = {
            "protocol": reference(RECORDS / "benchmarks/protocol.json"),
            "pairs": pairs,
            "independent_training_pairs": 3,
            "crossed_panels": 3,
            "evaluation_units": 36,
            "claim_boundary": spec["claim_boundary"],
        }
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(directory / "result.json", json_ready(result))
    archived = {
        str(path.relative_to(RUNS)): copy_artifact(
            path, RECORDS / "artifacts" / path.relative_to(RUNS)
        )
        for path in sorted(RUNS.rglob("*"))
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".pth")
    }
    write_json_exclusive(
        RECORDS / "results/result.json", json_ready({**result, "artifacts": archived})
    )
    return {seed: row["equal_panel_mean"] for seed, row in pairs.items()}
