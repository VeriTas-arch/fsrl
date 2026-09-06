"""Separate resource, allocation and behavior decisions; archive sufficient evidence."""

import shutil

import numpy as np

from fsrl.experiments.evidence_routing.measurement import paired_tau, selection
from fsrl.experiments.memory_structure.measurement import above
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .locks import SELECTION, SOURCE, completed, validate_phase
from .protocol import PROTOCOL_SHA256, RECORDS, RUNS, register, specification


def arrays_at(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def paired_seed(seed, spec):
    records = {}
    raw = {}
    for arm in ("shared", "cost"):
        records[arm] = {
            phase: completed(RUNS / phase / str(seed) / arm)
            for phase in ("liu", "diagnostic")
        }
        records[arm]["generic"] = completed(RUNS / "generic/test" / str(seed) / arm)
        raw[arm] = {
            phase: arrays_at(RUNS / phase / str(seed) / arm / "raw.npz")
            for phase in ("liu", "diagnostic")
        }
        raw[arm]["generic"] = arrays_at(
            RUNS / "generic/test" / str(seed) / arm / "raw.npz"
        )
    statistics = spec["statistics"]
    boot = statistics["seed_offset"] + seed
    delta = {
        f"generic_{key}": raw["cost"]["generic"][key] - raw["shared"]["generic"][key]
        for key in ("cost", "total_write", "ce")
    }
    for key in (
        "write_fraction",
        "target_write",
        "remote_influence",
        "direct_influence",
        "global_ce_benefit",
    ):
        delta[f"history_interaction_{key}"] = (
            raw["cost"]["diagnostic"][f"novel__{key}"]
            - raw["cost"]["diagnostic"][f"redundant__{key}"]
        ) - (
            raw["shared"]["diagnostic"][f"novel__{key}"]
            - raw["shared"]["diagnostic"][f"redundant__{key}"]
        )
    for group in ("learned", "nonlearned", "omitted"):
        key = f"N8__endpoints__intact__probability__{group}"
        delta[f"liu_probability_{group}"] = (
            raw["cost"]["liu"][key] - raw["shared"]["liu"][key]
        )
    estimates = {
        key: estimate(value, seed=boot, statistics=statistics)
        for key, value in delta.items()
    }
    behavior = [
        selection(load_json(RUNS / "liu" / str(seed) / arm / "behavior-N8.json"))
        for arm in ("shared", "cost")
    ]
    tau = paired_tau(
        behavior[0][0], behavior[1][0], behavior[0][1], behavior[1][1], boot, statistics
    )
    tau = {key.replace("isolated", "cost"): value for key, value in tau.items()}
    full_tau = {}
    for component in ("internal", "global_internal"):
        first, second = [
            raw[arm]["liu"][f"N8__{component}__orders"] for arm in ("shared", "cost")
        ]
        mask = np.ones(len(first), dtype=bool)
        row = paired_tau(first, second, mask, mask, boot, statistics)
        full_tau[component] = {
            key.replace("isolated", "cost"): value for key, value in row.items()
        }
    ratio = (
        records["cost"]["generic"]["mean_cost"]
        / records["shared"]["generic"]["mean_cost"]
    )
    competent = all(records[arm]["generic"]["competence"] for arm in ("shared", "cost"))
    cell = records["cost"]["liu"]["cells"]["8"]
    decision = {
        "both_generic_competent": competent,
        "resource_budget": ratio <= spec["cost"]["budget_ratio"],
        "dynamic_selection": competent
        and ratio <= spec["cost"]["budget_ratio"]
        and above(estimates["history_interaction_write_fraction"], 0.0)
        and above(estimates["history_interaction_remote_influence"], 0.0),
        "cost_liu_core": cell["core_passed"],
    }
    decision["transport_eligible"] = (
        records["cost"]["generic"]["competence"]
        and decision["resource_budget"]
        and cell["core_passed"]
        and all(
            above(cell["effects"][f"intact_minus_{name}_nonlearned"], 0.0)
            for name in ("P_off", "evidence_shuffle")
        )
    )
    return {
        "models": records,
        "write_ratio": ratio,
        "estimates": estimates,
        "conditional_tau": tau,
        "full_panel_tau": full_tau,
        "decision": decision,
    }


def paired_results():
    spec = specification()
    pairs = {str(seed): paired_seed(seed, spec) for seed in spec["seeds"]["mandatory"]}
    return {
        "pairs": pairs,
        "transport_triggered": any(
            row["decision"]["transport_eligible"] for row in pairs.values()
        ),
    }


def copy_artifact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError("archive collision")
    else:
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    return reference(target)


def report():
    selection_lock = validate_phase(SELECTION)
    spec = specification()
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": load_json(SOURCE)["source_commit"],
        "selection": selection_lock,
        "preparation": load_json(RECORDS / "results/preparation.json"),
        "claim_boundary": spec["claim_boundary"],
        "artifacts": {},
    }
    if selection_lock["selected"] is None:
        result.update(
            pairs={},
            transport_triggered=False,
            paired_status="not_triggered_no_feasible_candidate",
            transport_status="not_triggered",
        )
    else:
        primary = paired_results()
        if json_ready(primary) != load_json(RUNS / "primary.json"):
            raise RuntimeError("transport decision changed")
        result.update(
            primary,
            paired_status="completed",
            transport_status="completed"
            if primary["transport_triggered"]
            else "not_triggered",
        )
        result["transport"] = (
            {
                f"{seed}/{arm}": completed(RUNS / "transport" / str(seed) / arm)
                for seed in spec["seeds"]["mandatory"]
                for arm in ("shared", "cost")
            }
            if primary["transport_triggered"]
            else {}
        )
    for path in sorted(RUNS.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".pth"):
            target = RECORDS / "artifacts" / path.relative_to(RUNS)
            result["artifacts"][str(path.relative_to(RUNS))] = copy_artifact(
                path, target
            )
    write_json_exclusive(RECORDS / "results/result.json", json_ready(result))
    register(
        "unresolved",
        "Execution archived; interpretation and bounded validation pending. No automatic promotion.",
    )
    return {
        "paired_status": result["paired_status"],
        "transport_status": result["transport_status"],
        "artifacts": len(result["artifacts"]),
    }
