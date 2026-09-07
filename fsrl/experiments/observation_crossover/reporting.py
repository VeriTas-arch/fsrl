"""Four-cell paired contrasts and individual order displacement."""

from itertools import combinations

import numpy as np

from fsrl.experiments.finite_state.reporting import tau_contrast
from fsrl.experiments.memory_structure.measurement import above
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
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import arrays
from .protocol import LOCK, RECORDS, RUNS, inherited, specification, validate_lock


def order_shift(first, second):
    """Per-person fraction of item pairs whose relative order has reversed."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("matched two-dimensional orders required")
    n = first.shape[1]
    for value in (first, second):
        if not np.all(np.sort(value, axis=1) == np.arange(n)):
            raise ValueError("orders must be permutations of common item IDs")
    p, q = np.argsort(first, axis=1), np.argsort(second, axis=1)
    i, j = np.triu_indices(n, 1)
    return ((p[:, i] < p[:, j]) != (q[:, i] < q[:, j])).mean(1)


def contrasts(raw, endpoints, seed, spec):
    boot = spec["statistics"]["seed_offset"] + seed
    output = {}
    for label, coefficients in spec["estimands"]["contrasts"].items():
        output[label] = {
            "tau": {
                f"{route}_{scope}": tau_contrast(
                    raw, coefficients, route, scope == "conditional", boot, spec
                )
                for route in ("full", "global")
                for scope in ("all77", "conditional")
            },
            "CE": {
                name: estimate(
                    np.stack(
                        [
                            weight * endpoints[cell][name]
                            for cell, weight in coefficients.items()
                        ]
                    ).sum(0),
                    seed=boot,
                    statistics=spec["statistics"],
                )
                for name in endpoints["A0"]
            },
        }
    return output


def paired(seed, lock, spec):
    parents, rows, raw = inherited(), {}, {}
    for cell, settings in spec["design"]["cells"].items():
        if settings["reuse"] is None:
            directory = RUNS / str(seed) / cell
            rows[cell] = completed(directory)
            flat = arrays(reference(directory / "raw.npz"))
            raw[cell] = {
                phase: {
                    key.removeprefix(phase + "__"): value
                    for key, value in flat.items()
                    if key.startswith(phase + "__")
                }
                for phase in ("generic", "liu")
            }
        else:
            rows[cell] = parents["parent_result"]["pairs"][str(seed)]["models"][
                settings["reuse"]
            ]
            raw[cell] = {
                phase: arrays(ref)
                for phase, ref in lock["previous_cells"][str(seed)][cell].items()
            }
    for first, second in combinations(raw, 2):
        for key in ("signs", "learned", "episode_indices"):
            if not np.array_equal(
                raw[first]["generic"][key], raw[second]["generic"][key]
            ):
                raise RuntimeError("generic participant/query alignment differs")
    cpu = load_input(lock["inputs"]["liu-8"])
    n = parents["parent_protocol"]["evaluation"]["liu"]["subjects"]
    signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T
    temperature = parents["parent_protocol"]["evaluation"]["liu"]["temperature"]
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


def report():
    lock, spec = validate_lock(), specification()
    directory = RUNS / "comparison"
    if directory.exists():
        result = completed(directory)
    else:
        with ProspectiveRun.start(
            directory,
            workflow_id="observation_crossover_v1",
            execution_id="four-cell-comparison",
            producer={"execution_lock": reference(LOCK)},
            resolved_config={
                key: spec[key] for key in ("estimands", "statistics", "decision")
            },
        ):
            pairs, raw = {}, {}
            for seed in spec["design"]["seeds"]:
                pairs[str(seed)], raw[str(seed)] = paired(seed, lock, spec)
            result = {
                "protocol_sha256": lock["protocol_sha256"],
                "source_commit": lock["source_commit"],
                "parent_result": spec["parent_result"],
                "pairs": pairs,
                "claim_boundary": spec["claim_boundary"],
            }
            write_arrays(directory / "raw.npz", flatten_arrays(raw))
            write_json_exclusive(directory / "result.json", json_ready(result))
    archived = {
        str(path.relative_to(RUNS)): copy_artifact(
            path, RECORDS / "artifacts" / path.relative_to(RUNS)
        )
        for path in sorted(RUNS.rglob("*"))
        if path.is_file() and path.suffix in (".json", ".npz")
    }
    write_json_exclusive(
        RECORDS / "results/result.json", json_ready({**result, "artifacts": archived})
    )
    return {seed: row["decision"] for seed, row in result["pairs"].items()}
