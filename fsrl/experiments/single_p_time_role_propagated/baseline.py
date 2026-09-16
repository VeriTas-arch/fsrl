"""Phase 1: create and validate the frozen direct-authority baseline."""

from __future__ import annotations

import copy
import gc

import numpy as np
import torch

from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.local_memory_removal.statistics import (
    noninferiority,
    paired_probability,
    probabilities,
)
from fsrl.experiments.observation_replication.protocol import (
    specification as analysis_specification,
)
from fsrl.experiments.observation_replication.reporting import summarize
from fsrl.experiments.observation_replication.statistics import panel_draws, panel_mean
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
)
from fsrl.infra.provenance import load_json, write_json_exclusive

from .budgets import audit_unit, merge_caps, reconstructed_error, summary_budget
from .direct import assert_bitwise_equal, load_model
from .external import CELLS, evaluate_unit
from .locks import reference, validate_source_lock, verify_reference
from .protocol import (
    BASELINE_RESULT,
    BASELINE_RUNS,
    PROTOCOL_SHA256,
    SOURCE_LOCK,
    specification,
)
from .storage import write_npz_exclusive


def _inputs(lock: dict, panel: int) -> dict[str, dict]:
    prefix = f"{panel}/"
    return {
        key.removeprefix(prefix): value
        for key, value in lock["parents"]["inputs"].items()
        if key.startswith(prefix)
    }


def _checkpoint(lock: dict, seed: int, condition: str, arm: str) -> dict:
    return lock["parents"]["checkpoints"][f"{seed}/{condition}/{arm}"]


def _parent_raw(lock: dict, identity: str) -> dict:
    path = verify_reference(lock["parents"]["parent_evaluation_raw"][identity])
    with np.load(path, allow_pickle=False) as payload:
        flat = {name: payload[name].copy() for name in payload.files}
    return {
        phase: {
            key.removeprefix(phase + "__"): value
            for key, value in flat.items()
            if key.startswith(phase + "__")
        }
        for phase in ("generic", "liu")
    }


def _raw_cpu(record: dict) -> EpisodeBatch:
    with np.load(verify_reference(record), allow_pickle=False) as payload:
        return EpisodeBatch({name: payload[name] for name in payload.files})


def _decision(panels: dict, candidate_mean: dict, ni: dict) -> dict:
    control_valid = all(
        panel["time_retained_control"]["decision"]["all_four_competent"]
        and all(
            panel["time_retained_control"]["evidence_binding"][cell]
            for cell in ("Ae", "Ce")
        )
        for panel in panels.values()
    )
    candidate_competent = all(
        panel["clean_no_time"]["decision"]["all_four_competent"]
        and all(
            panel["clean_no_time"]["evidence_binding"][cell] for cell in ("Ae", "Ce")
        )
        for panel in panels.values()
    )
    candidate_core = all(
        any(
            panel["clean_no_time"]["cells"][cell]["liu"]["routes"]["full"][
                "core_passed"
            ]
            for panel in panels.values()
        )
        for cell in ("Ae", "Ce")
    )
    ce_noninferior = all(
        row["noninferior"] for row in noninferiority(ni["Ce"]).values()
    )
    compensation = {
        "positive_I_tau": candidate_mean["interaction/global_all77_tau"]["interval"][
            "lower"
        ]
        > 0,
        "negative_I_CE": candidate_mean["interaction/generic_global"]["interval"][
            "upper"
        ]
        < 0,
        "negative_shift_difference": candidate_mean["order_shift/C_minus_A"][
            "interval"
        ]["upper"]
        < 0,
        "both_error_tau_points_negative": all(
            candidate_mean[f"{name}/global_all77_tau"]["point"] < 0
            for name in ("error_A", "error_C")
        ),
    }
    if not control_valid:
        outcome = "training_recipe_failure"
    elif not candidate_competent or not ce_noninferior:
        outcome = "time_removal_failure"
    elif not candidate_core or not all(compensation.values()):
        outcome = "phenotype_incomplete"
    else:
        outcome = "clean_single_p_admitted"
    return {
        "control_valid": control_valid,
        "candidate_competent_and_bound": candidate_competent,
        "Ce_five_endpoints_noninferior": ce_noninferior,
        "Ae_and_Ce_core_present": candidate_core,
        "compensation": compensation,
        "outcome": outcome,
        "admitted": outcome == "clean_single_p_admitted",
    }


def _pair(
    seed: int, panels: dict[str, dict], raws: dict[str, dict]
) -> tuple[dict, dict]:
    condition_draws = {name: [] for name in ("time_retained_control", "clean_no_time")}
    ni_panel = []
    draw_trace = {}
    analysis_spec = analysis_specification()
    for panel in specification()["design"]["evaluation_panels"]:
        panel_key = str(panel)
        condition_raw = {}
        for condition in specification()["design"]["architecture_conditions"]:
            raw = raws[panel_key][condition]
            rows = panels[panel_key][condition]
            sample_seed = seed + panel * 1000000
            result, endpoints = summarize(
                raw,
                rows,
                _raw_cpu(_inputs(load_json(SOURCE_LOCK), panel)["liu-8"]),
                sample_seed,
                inherited_recipe(panel),
                analysis_spec,
            )
            panels[panel_key][condition] = result
            points, draws = panel_draws(
                raw,
                endpoints["CE"],
                endpoints["order_shift"],
                analysis_spec["statistics"]["seed_offset"] + sample_seed,
                analysis_spec,
            )
            condition_draws[condition].append((points, draws))
            for name, value in draws.items():
                draw_trace[f"panel/{panel}/{condition}/{name}"] = value
            condition_raw[condition] = raw
        boot = analysis_spec["statistics"]["seed_offset"] + seed + panel * 1000000
        values = {}
        for cell in CELLS:
            values[cell], _ = paired_probability(
                probabilities(condition_raw["clean_no_time"][cell]),
                probabilities(condition_raw["time_retained_control"][cell]),
                boot,
            )
            for name, value in values[cell][1].items():
                draw_trace[
                    f"panel/{panel}/noninferiority_equal_panel_mean/{cell}/{name}"
                ] = value
        ni_panel.append(values)
    means = {}
    for condition, values in condition_draws.items():
        means[condition], draws = panel_mean(values)
        for name, value in draws.items():
            draw_trace[f"equal_panel/{condition}/{name}"] = value
    ni = {}
    for cell in CELLS:
        ni[cell], draws = panel_mean([values[cell] for values in ni_panel])
        for name, value in draws.items():
            draw_trace[f"noninferiority_equal_panel_mean/{cell}/{name}"] = value
    return {
        "panels": panels,
        "equal_panel_mean": means,
        "noninferiority_equal_panel_mean": ni,
        "decision": _decision(panels, means["clean_no_time"], ni),
    }, draw_trace


def _leaf_values(value, kind, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}/{key}" if prefix else key
            if kind == "point" and key == "point" and isinstance(item, (int, float)):
                result[path] = float(item)
            elif kind == "classification" and isinstance(item, (bool, str)):
                result[path] = item
            else:
                result.update(_leaf_values(item, kind, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_leaf_values(item, kind, f"{prefix}/{index}"))
    return result


def _numeric_leaves(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}/{key}" if prefix else key
            if key in {"point", "lower", "upper"} and isinstance(item, (int, float)):
                result[path] = float(item)
            else:
                result.update(_numeric_leaves(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_numeric_leaves(item, f"{prefix}/{index}"))
    return result


def _compare_parent(
    pairs: dict,
    reconstructed: dict,
    parent: dict,
    outcome: str,
    caps,
    direct_draws: dict,
    parent_draws: dict,
) -> dict:
    saved = _numeric_leaves(parent["pairs"])
    rebuilt = _numeric_leaves(reconstructed)
    if saved.keys() != rebuilt.keys():
        raise RuntimeError("parent summary reconstruction inventory differs")
    reconstruction_error = 0.0
    for name in saved:
        reconstruction_error = max(
            reconstruction_error,
            reconstructed_error(rebuilt[name], saved[name], f"parent summary {name}"),
        )
    direct_values = _numeric_leaves(pairs)
    if direct_values.keys() != rebuilt.keys():
        raise RuntimeError("external summary numeric inventory differs")
    point_error = 0.0
    maximum_ratio = 0.0
    for name in direct_values:
        allowed = summary_budget(name, caps)
        observed = abs(direct_values[name] - rebuilt[name])
        if observed > allowed:
            raise RuntimeError(
                f"external summary exceeds propagated budget: {name}: "
                f"error={observed}, budget={allowed}"
            )
        point_error = max(point_error, observed)
        maximum_ratio = max(maximum_ratio, observed / allowed)
    if direct_draws.keys() != parent_draws.keys():
        raise RuntimeError("external bootstrap draw inventory differs")
    maximum_draw_error = 0.0
    maximum_draw_ratio = 0.0
    draw_values = 0
    for name, value in direct_draws.items():
        left, right = np.asarray(value), np.asarray(parent_draws[name])
        allowed = summary_budget(name, caps)
        error = np.abs(left - right)
        if left.shape != right.shape or np.any(error > allowed):
            raise RuntimeError(
                f"external bootstrap draws exceed propagated budget: {name}: "
                f"error={float(error.max())}, budget={allowed}"
            )
        maximum_draw_error = max(maximum_draw_error, float(error.max()))
        maximum_draw_ratio = max(maximum_draw_ratio, float(error.max()) / allowed)
        draw_values += error.size
    direct_flags = _leaf_values(pairs, "classification")
    parent_flags = _leaf_values(parent["pairs"], "classification")
    if direct_flags != parent_flags:
        differing = sorted(
            name
            for name in direct_flags.keys() | parent_flags.keys()
            if direct_flags.get(name) != parent_flags.get(name)
        )
        raise RuntimeError(
            f"external scientific classifications differ: {differing[:5]}"
        )
    if outcome != parent["outcome"] or outcome != "time_removal_failure":
        raise RuntimeError("external overall parent outcome differs")
    direct_ni = noninferiority(pairs["3011"]["noninferiority_equal_panel_mean"]["Ce"])
    parent_ni = noninferiority(
        parent["pairs"]["3011"]["noninferiority_equal_panel_mean"]["Ce"]
    )
    if direct_ni != parent_ni:
        raise RuntimeError("3011 Ce noninferiority classifications differ")
    return {
        "maximum_summary_numeric_absolute_error": point_error,
        "maximum_summary_budget_fraction": maximum_ratio,
        "maximum_bootstrap_draw_absolute_error": maximum_draw_error,
        "maximum_bootstrap_draw_budget_fraction": maximum_draw_ratio,
        "bootstrap_draw_values_compared": draw_values,
        "maximum_parent_same_input_reconstruction_error": reconstruction_error,
        "classification_leaves_compared": len(direct_flags),
        "summary_numeric_values_compared": len(direct_values),
        "seed_3011_Ce_noninferiority": direct_ni,
        "parent_outcome": outcome,
    }


def _write_unit(identity: str, external: dict, internal: dict) -> list[dict]:
    seed, panel, condition, cell = identity.split("/")
    root = BASELINE_RUNS / seed / panel / condition / cell
    external_path = root / "external.npz"
    write_npz_exclusive(external_path, flatten_arrays(external))
    records = [reference(external_path)]
    if internal["generic"] or internal["liu"]:
        internal_path = root / "internal.npz"
        write_npz_exclusive(internal_path, flatten_arrays(internal))
        records.append(reference(internal_path))
    return records


def _aggregate_outcome(pairs: dict) -> str:
    outcomes = [row["decision"]["outcome"] for row in pairs.values()]
    if all(value == "clean_single_p_admitted" for value in outcomes):
        return "clean_single_p_admitted"
    if "training_recipe_failure" in outcomes:
        return "training_recipe_failure"
    if "time_removal_failure" in outcomes:
        return "time_removal_failure"
    return "phenotype_incomplete"


def run(runtime: dict) -> dict:
    specification()
    lock = validate_source_lock(runtime)
    if BASELINE_RESULT.exists() or BASELINE_RUNS.exists():
        raise RuntimeError("direct-authority baseline is write-once")
    artifacts = []
    panels_by_seed = {}
    raws_by_seed = {}
    parent_raws_by_seed = {}
    replay_errors = {}
    budget_caps = []
    try:
        with torch.no_grad():
            for seed in specification()["design"]["network_seeds"]:
                panels_by_seed[str(seed)] = {}
                raws_by_seed[str(seed)] = {}
                parent_raws_by_seed[str(seed)] = {}
                for panel in specification()["design"]["evaluation_panels"]:
                    panel_key = str(panel)
                    panels_by_seed[str(seed)][panel_key] = {}
                    raws_by_seed[str(seed)][panel_key] = {}
                    parent_raws_by_seed[str(seed)][panel_key] = {}
                    inputs = _inputs(lock, panel)
                    for condition in specification()["design"][
                        "architecture_conditions"
                    ]:
                        panels_by_seed[str(seed)][panel_key][condition] = {}
                        raws_by_seed[str(seed)][panel_key][condition] = {}
                        parent_raws_by_seed[str(seed)][panel_key][condition] = {}
                        for cell, settings in CELLS.items():
                            identity = f"{seed}/{panel}/{condition}/{cell}"
                            checkpoint = _checkpoint(
                                lock, seed, condition, settings["training"]
                            )
                            model = load_model(checkpoint, condition)
                            first = evaluate_unit(
                                model,
                                inputs,
                                observation=settings["observation"],
                                panel=panel,
                                seed=seed,
                                keep_p=cell == "Ce",
                            )
                            del model
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = load_model(checkpoint, condition)
                            second = evaluate_unit(
                                model,
                                inputs,
                                observation=settings["observation"],
                                panel=panel,
                                seed=seed,
                                keep_p=cell == "Ce",
                            )
                            del model
                            first_external, first_result, first_internal = first
                            second_external, _, second_internal = second
                            assert_bitwise_equal(
                                flatten_arrays(first_external),
                                flatten_arrays(second_external),
                            )
                            assert_bitwise_equal(
                                flatten_arrays(first_internal),
                                flatten_arrays(second_internal),
                            )
                            parent_raw = _parent_raw(lock, identity)
                            errors, caps = audit_unit(
                                first_external,
                                parent_raw,
                                _raw_cpu(inputs["liu-8"]),
                            )
                            budget_caps.append(caps)
                            replay_errors[identity] = errors
                            panels_by_seed[str(seed)][panel_key][condition][cell] = (
                                first_result
                            )
                            raws_by_seed[str(seed)][panel_key][condition][cell] = {
                                "generic": first_external["generic"],
                                "liu": first_external["liu"],
                            }
                            parent_raws_by_seed[str(seed)][panel_key][condition][
                                cell
                            ] = parent_raw
                            artifacts += _write_unit(
                                identity, first_external, first_internal
                            )
                            print(
                                {"baseline_unit": identity, "status": "passed"},
                                flush=True,
                            )
                            del first, second, first_internal, second_internal
                            gc.collect()
                            torch.cuda.empty_cache()
        pairs, direct_draws = {}, {}
        for seed in specification()["design"]["network_seeds"]:
            pairs[str(seed)], trace = _pair(
                seed, panels_by_seed[str(seed)], raws_by_seed[str(seed)]
            )
            direct_draws.update(
                {f"{seed}/{key}": value for key, value in trace.items()}
            )
        outcome = _aggregate_outcome(pairs)
        parent = load_json(verify_reference(lock["parents"]["parent_result"]))
        parent_panels = {
            str(seed): {
                str(panel): {
                    condition: copy.deepcopy(
                        parent["pairs"][str(seed)]["panels"][str(panel)][condition][
                            "cells"
                        ]
                    )
                    for condition in specification()["design"][
                        "architecture_conditions"
                    ]
                }
                for panel in specification()["design"]["evaluation_panels"]
            }
            for seed in specification()["design"]["network_seeds"]
        }
        reconstructed, parent_draws = {}, {}
        for seed in specification()["design"]["network_seeds"]:
            reconstructed[str(seed)], trace = _pair(
                seed,
                parent_panels[str(seed)],
                parent_raws_by_seed[str(seed)],
            )
            parent_draws.update(
                {f"{seed}/{key}": value for key, value in trace.items()}
            )
        compatibility = _compare_parent(
            pairs,
            reconstructed,
            parent,
            outcome,
            merge_caps(budget_caps),
            direct_draws,
            parent_draws,
        )
        manifest = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "members": artifacts,
        }
        manifest_path = BASELINE_RUNS / "manifest.json"
        write_json_exclusive(manifest_path, manifest)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "runtime": runtime,
            "outcome": "baseline_compatible",
            "self_reproducibility": "bitwise_identical",
            "external_compatibility": compatibility,
            "maximum_external_errors": {
                name: max(row[name] for row in replay_errors.values())
                for name in next(iter(replay_errors.values()))
            },
            "pairs": json_ready(pairs),
            "evaluation_units": len(replay_errors),
            "artifact_members": len(artifacts),
            "manifest": reference(manifest_path),
            "parent_outcome_unchanged": "time_removal_failure",
            "mechanism_estimates_exposed": False,
        }
        BASELINE_RESULT.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(BASELINE_RESULT, result)
        return {
            "outcome": result["outcome"],
            "evaluation_units": result["evaluation_units"],
            "artifact_members": result["artifact_members"],
            **compatibility,
        }
    except Exception as error:
        BASELINE_RESULT.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(SOURCE_LOCK),
            "runtime": runtime,
            "outcome": "baseline_incompatible",
            "failure": str(error),
            "completed_units": len(replay_errors),
            "mechanism_estimates_exposed": False,
            "parent_outcome_unchanged": "time_removal_failure",
        }
        if not BASELINE_RESULT.exists():
            write_json_exclusive(BASELINE_RESULT, failure)
        raise


__all__ = ["run"]
