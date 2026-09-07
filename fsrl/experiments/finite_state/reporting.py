"""Paired factorial estimands, acute adaptation contrasts and immutable closure."""

import gc

import numpy as np
import torch

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.evidence_routing.measurement import tau_matrix, weighted_tau
from fsrl.experiments.memory_structure.measurement import above
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.locks import completed
from fsrl.experiments.write_cost.reporting import copy_artifact
from fsrl.infra.provenance import load_json, write_json_exclusive

from .diagnostic import diagnostic_run
from .evaluation import arrays_at, generic_run
from .liu import liu_run
from .locks import MODELS, SELECTION, SOURCE, validate_phase
from .protocol import PROTOCOL_SHA256, RECORDS, RUNS, register, specification

CONTRASTS = {
    "precision_minus_shared": {"precision": 1, "shared": -1},
    "cost_precision_minus_cost": {"cost_precision": 1, "cost": -1},
    "cost_minus_shared": {"cost": 1, "shared": -1},
    "cost_precision_minus_precision": {"cost_precision": 1, "precision": -1},
    "cost_by_precision": {
        "cost_precision": 1,
        "cost": -1,
        "precision": -1,
        "shared": 1,
    },
    "precision_minus_acute_shared": {"precision": 1, "acute_shared": -1},
    "cost_precision_minus_acute_cost": {"cost_precision": 1, "acute_cost": -1},
}


def tau_contrast(raw, coefficients, route, conditional, boot, spec):
    subjects = len(next(iter(raw.values()))["liu"]["routes__full__sampled_mask"])
    counts = bootstrap_counts(
        np.random.default_rng(boot), spec["statistics"]["samples"], subjects
    )
    draws = np.zeros(len(counts))
    point, sizes = 0.0, {}
    for arm, coefficient in coefficients.items():
        data = raw[arm]["liu"]
        if conditional:
            orders = data[f"routes__{route}__sampled_orders"]
            mask = data[f"routes__{route}__sampled_mask"]
        else:
            orders = data[f"routes__{route}__internal__orders"]
            mask = np.ones(subjects, dtype=bool)
        matrix = tau_matrix(orders)
        draws += coefficient * weighted_tau(matrix, counts * mask)
        point += coefficient * weighted_tau(matrix, mask[None, :].astype(float))[0]
        sizes[arm] = int(mask.sum())
    invalid = int((~np.isfinite(draws)).sum())
    interval = [None, None] if invalid else np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "point": float(point) if np.isfinite(point) else None,
        "interval": {"lower": interval[0], "upper": interval[1]},
        "undefined_draws": invalid,
        "analysis_subjects": sizes,
        "resampled_subjects": subjects,
    }


def scalar_endpoints(raw):
    values = {
        f"generic_{key}": raw["generic"][key]
        for key in (
            "ce",
            "global_ce",
            "cost",
            "total_write",
            "P_abs_mean",
            "A_abs_mean",
        )
    }
    for key in (
        "write_fraction",
        "target_write",
        "remote_influence",
        "direct_influence",
        "global_ce_benefit",
    ):
        values[f"history_{key}"] = (
            raw["diagnostic"][f"novel__{key}"] - raw["diagnostic"][f"redundant__{key}"]
        )
    for route in ("intact", "local_off"):
        for group in ("learned", "nonlearned", "omitted"):
            values[f"liu_{route}_probability_{group}"] = raw["liu"][
                f"endpoints__{route}__probability__{group}"
            ]
    for route in ("full", "global"):
        for key in ("stable_error_density", "stable_error_prevalence"):
            values[f"liu_{route}_{key}"] = raw["liu"][
                f"routes__{route}__internal__{key}"
            ]
        values[f"liu_{route}_coherence"] = raw["liu"][
            f"routes__{route}__behavior__coherence"
        ]
    return values


def paired_seed(seed, spec):
    arms = [*spec["seeds"]["conditions"], "acute_shared", "acute_cost"]
    records, raw, endpoints = {}, {}, {}
    for arm in arms:
        paths = {
            "generic": RUNS / "generic/test" / str(seed) / arm,
            "diagnostic": RUNS / "diagnostic" / str(seed) / arm,
            "liu": RUNS / "liu" / str(seed) / arm,
        }
        records[arm] = {phase: completed(path) for phase, path in paths.items()}
        raw[arm] = {phase: arrays_at(path) for phase, path in paths.items()}
        endpoints[arm] = scalar_endpoints(raw[arm])
    boot = spec["statistics"]["seed_offset"] + seed
    contrasts = {}
    for name, coefficients in CONTRASTS.items():
        values = {
            key: np.stack(
                [
                    coefficient * endpoints[arm][key]
                    for arm, coefficient in coefficients.items()
                ]
            ).sum(axis=0)
            for key in endpoints["shared"]
        }
        contrasts[name] = {
            "estimates": {
                key: estimate(value, seed=boot, statistics=spec["statistics"])
                for key, value in values.items()
            },
            "tau": {
                f"{route}_{scope}": tau_contrast(
                    raw, coefficients, route, scope == "conditional", boot, spec
                )
                for route in ("full", "global")
                for scope in ("conditional", "all77")
            },
            "all_generic_competent": all(
                records[arm]["generic"]["competence"] for arm in coefficients
            ),
        }
    core = {arm: records[arm]["liu"]["routes"]["full"]["core_passed"] for arm in arms}
    functional = {
        name: row["all_generic_competent"]
        and above(row["estimates"]["history_global_ce_benefit"], 0.0)
        for name, row in contrasts.items()
    }
    return {
        "models": records,
        "contrasts": contrasts,
        "decision": {
            "core_by_model": core,
            "functional_allocation_by_contrast": functional,
            "simpler_precision_candidate": core["precision"],
            "cost_precision_only_core": core["cost_precision"]
            and not core["precision"]
            and not core["cost"],
            "complementarity_supported": core["cost_precision"]
            and not core["precision"]
            and not core["cost"]
            and functional["cost_by_precision"]
            and functional["cost_precision_minus_precision"],
        },
    }


def evaluate():
    validate_phase(MODELS)
    source, spec = load_json(SOURCE), specification()
    arms = [*spec["seeds"]["conditions"], "acute_shared", "acute_cost"]
    for seed in spec["seeds"]["mandatory"]:
        for arm in arms:
            generic_run(seed, arm, "test", source)
            diagnostic_run(seed, arm, source)
            liu_run(seed, arm, source)
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "pairs": {
            str(seed): paired_seed(seed, spec) for seed in spec["seeds"]["mandatory"]
        }
    }
    path = RUNS / "primary.json"
    if path.exists():
        if load_json(path) != json_ready(result):
            raise RuntimeError("primary reconstruction differs")
    else:
        write_json_exclusive(path, json_ready(result))
    return {seed: row["decision"] for seed, row in result["pairs"].items()}


def report():
    selected = validate_phase(SELECTION)
    spec = specification()
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": load_json(SOURCE)["source_commit"],
        "selection": selected,
        "claim_boundary": spec["claim_boundary"],
        "artifacts": {},
    }
    if selected["selected_K"] is None:
        result.update(pairs={}, paired_status="not_triggered_no_feasible_candidate")
    else:
        validate_phase(MODELS)
        pairs = {
            str(seed): paired_seed(seed, spec) for seed in spec["seeds"]["mandatory"]
        }
        if json_ready({"pairs": pairs}) != load_json(RUNS / "primary.json"):
            raise RuntimeError("primary reconstruction differs")
        result.update(pairs=pairs, paired_status="completed")
    for path in sorted(RUNS.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".pth"):
            target = RECORDS / "artifacts" / path.relative_to(RUNS)
            result["artifacts"][str(path.relative_to(RUNS))] = copy_artifact(
                path, target
            )
    write_json_exclusive(RECORDS / "results/result.json", json_ready(result))
    register(
        "unresolved",
        "Execution archived; interpretation pending. No automatic promotion.",
    )
    return {
        "paired_status": result["paired_status"],
        "artifacts": len(result["artifacts"]),
    }
