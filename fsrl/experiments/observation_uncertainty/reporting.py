"""Paired observation contrasts and closure without automatic promotion."""

import gc

import numpy as np
import torch

from fsrl.experiments.finite_state.evaluation import arrays_at
from fsrl.experiments.finite_state.reporting import tau_contrast
from fsrl.experiments.memory_structure.measurement import above
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.locks import completed
from fsrl.experiments.write_cost.reporting import copy_artifact
from fsrl.infra.provenance import load_json, write_json_exclusive

from .comparator import run as comparator_run
from .diagnostic import diagnostic_run
from .evaluation import generic_run
from .liu import liu_run
from .locks import MODELS, SCREEN, SOURCE, validate_phase
from .protocol import PROTOCOL_SHA256, RECORDS, RUNS, register, specification

CONTRASTS = {
    "folded_minus_clean": {"folded": 1, "clean": -1},
    "noisy_minus_folded": {"noisy": 1, "folded": -1},
    "noisy_minus_clean": {"noisy": 1, "clean": -1},
    "noisy_minus_acute_noisy": {"noisy": 1, "acute_noisy": -1},
}


def paired_seed(seed, spec):
    arms = [*spec["seeds"]["conditions"], "acute_noisy"]
    records = {}
    raw = {}
    endpoints = {}
    for arm in arms:
        paths = {
            phase: RUNS / base / str(seed) / arm
            for phase, base in [
                ("generic", "generic/test"),
                ("diagnostic", "diagnostic"),
                ("liu", "liu"),
            ]
        }
        records[arm] = {phase: completed(path) for phase, path in paths.items()}
        raw[arm] = {phase: arrays_at(path) for phase, path in paths.items()}
        g, d, l = (raw[arm][phase] for phase in ("generic", "diagnostic", "liu"))
        endpoints[arm] = {f"generic_{k}": g[k] for k in ("ce", "global_ce", "cost")}
        for k in (
            "global_ce_benefit",
            "remote_influence",
            "direct_influence",
            "write_fraction",
        ):
            endpoints[arm][f"history_{k}"] = (
                d[f"supported__{k}"] - d[f"conflicting__{k}"]
            )
        for route in ("full", "global"):
            for key in ("stable_error_density", "stable_error_prevalence"):
                endpoints[arm][f"liu_{route}_{key}"] = l[
                    f"routes__{route}__internal__{key}"
                ]
        for label in ("supported", "conflicting"):
            # CE of each full history is separately reconstructable from locked targets.
            endpoints[arm][f"history_{label}_benefit"] = d[
                f"{label}__global_ce_benefit"
            ]
    boot = spec["statistics"]["seed_offset"] + seed
    contrasts = {}
    for name, coeff in CONTRASTS.items():
        contrasts[name] = {
            "estimates": {
                key: estimate(
                    np.stack([v * endpoints[arm][key] for arm, v in coeff.items()]).sum(
                        0
                    ),
                    seed=boot,
                    statistics=spec["statistics"],
                )
                for key in endpoints["clean"]
            },
            "tau": {
                f"{route}_{scope}": tau_contrast(
                    raw, coeff, route, scope == "conditional", boot, spec
                )
                for route in ("full", "global")
                for scope in ("conditional", "all77")
            },
        }
    core = {arm: records[arm]["liu"]["routes"]["full"]["core_passed"] for arm in arms}
    binding = {
        arm: all(
            above(
                records[arm]["liu"]["effects"][f"intact_minus_{control}_nonlearned"], 0
            )
            for control in ("P_off", "evidence_shuffle")
        )
        for arm in arms
    }
    adaptation = (
        contrasts["noisy_minus_acute_noisy"]["estimates"]["generic_ce"]["bootstrap"][
            "upper"
        ]
        < 0
    )
    functional = above(
        contrasts["noisy_minus_clean"]["estimates"]["history_global_ce_benefit"], 0
    )
    return {
        "models": records,
        "contrasts": contrasts,
        "decision": {
            "core_by_model": core,
            "evidence_binding": binding,
            "adaptation_supported": adaptation,
            "added_history_discrimination": functional,
            "folded_sufficient": core["folded"] and binding["folded"],
            "conditional_direction_support": core["noisy"]
            and not core["folded"]
            and binding["noisy"]
            and adaptation,
            "all_trained_core_failed": not any(
                core[a] for a in spec["seeds"]["conditions"]
            ),
        },
    }


def evaluate():
    validate_phase(MODELS)
    source = load_json(SOURCE)
    spec = specification()
    for seed in spec["seeds"]["mandatory"]:
        for arm in [*spec["seeds"]["conditions"], "acute_noisy"]:
            generic_run(seed, arm, "test", source)
            diagnostic_run(seed, arm, source)
            liu_run(seed, arm, source)
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "pairs": {
            str(seed): paired_seed(seed, spec) for seed in spec["seeds"]["mandatory"]
        },
        "comparator": comparator_run(source),
    }
    path = RUNS / "primary.json"
    if path.exists():
        if load_json(path) != json_ready(result):
            raise RuntimeError("primary reconstruction differs")
    else:
        write_json_exclusive(path, json_ready(result))
    return {seed: row["decision"] for seed, row in result["pairs"].items()}


def report():
    screen = validate_phase(SCREEN)
    spec = specification()
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": load_json(SOURCE)["source_commit"],
        "screen": screen,
        "claim_boundary": spec["claim_boundary"],
        "artifacts": {},
    }
    if screen["eligible"]:
        validate_phase(MODELS)
        pairs = {
            str(seed): paired_seed(seed, spec) for seed in spec["seeds"]["mandatory"]
        }
        primary = load_json(RUNS / "primary.json")
        if primary["pairs"] != json_ready(pairs):
            raise RuntimeError("primary reconstruction differs")
        result.update(primary, paired_status="completed")
    else:
        result.update(pairs={}, paired_status="not_triggered_development_incompetent")
    for path in sorted(RUNS.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".pth"):
            result["artifacts"][str(path.relative_to(RUNS))] = copy_artifact(
                path, RECORDS / "artifacts" / path.relative_to(RUNS)
            )
    write_json_exclusive(RECORDS / "results/result.json", json_ready(result))
    register("unresolved", "Execution archived; interpretation pending.")
    return {
        "paired_status": result["paired_status"],
        "artifacts": len(result["artifacts"]),
    }
