"""Execute the registered frozen-model time-role decomposition."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import write_json_exclusive

from .analysis import generic_endpoints, geometry, liu_endpoints, paired_interval
from .estimands import canonical_field, derangement, effective_distance, positive_scale
from .locks import reference, validate_source_lock
from .protocol import PROTOCOL_SHA256, RUNS, SOURCE_LOCK, specification
from .rollouts import (
    clean_inputs,
    load_cpu,
    load_model,
    read_ordered,
    read_original,
    support_trajectory,
)
from .trajectory import regression_summary, selected_generic, trajectory_rows

CELLS = {
    "A0": {"training": "clean", "observation": "clean"},
    "Ae": {"training": "clean", "observation": "noisy"},
    "C0": {"training": "noisy", "observation": "clean"},
    "Ce": {"training": "noisy", "observation": "noisy"},
}


def _parent_raw(lock: dict, seed: int, panel: int, condition: str, cell: str) -> Path:
    key = f"{seed}/{panel}/{condition}/{cell}"
    return Path(lock["parents"]["parent_evaluation_raw"][key]["path"])


def _checkpoint(lock: dict, seed: int, condition: str, arm: str) -> dict:
    return lock["parents"]["checkpoints"][f"{seed}/{condition}/{arm}"]


def _inputs(lock: dict, panel: int) -> dict[str, dict]:
    prefix = f"{panel}/"
    return {
        key.removeprefix(prefix): value
        for key, value in lock["parents"]["inputs"].items()
        if key.startswith(prefix)
    }


def _collect_baseline(
    lock: dict,
    seed: int,
    panel: int,
    condition: str,
    cell: str,
) -> dict:
    settings, recipe = CELLS[cell], inherited_recipe(panel)
    model = load_model(
        _checkpoint(lock, seed, condition, settings["training"]), condition
    )
    generic = {"margins": [], "fields": [], "targets": [], "learned": []}
    batches = []
    for name, record in sorted(_inputs(lock, panel).items()):
        if not name.startswith("test-"):
            continue
        cpu = load_cpu(record, settings["observation"], recipe)
        states, _, _ = support_trajectory(model, cpu)
        margins = read_original(model, states[-1], cpu)
        ordered, pairs = read_ordered(model, states[-1], cpu.arrays["item_codes"])
        generic["margins"].append(margins)
        generic["fields"].append(canonical_field(ordered, pairs))
        generic["targets"].append(cpu.arrays["targets"].reshape(-1, margins.shape[0]).T)
        generic["learned"].append(cpu.arrays["learned"])
        batches.append((name, cpu, states[-1]))
    generic = {
        "margins": np.concatenate(generic["margins"]),
        "fields": np.concatenate(generic["fields"]),
        "targets": np.concatenate(generic["targets"]),
        "learned": np.concatenate(generic["learned"]),
    }
    liu_cpu = load_cpu(_inputs(lock, panel)["liu-8"], settings["observation"], recipe)
    liu_states, _, _ = support_trajectory(model, liu_cpu)
    liu_margins = read_original(model, liu_states[-1], liu_cpu)
    liu_ordered, liu_pairs = read_ordered(
        model, liu_states[-1], liu_cpu.arrays["item_codes"]
    )
    liu = {
        "margins": liu_margins,
        "fields": canonical_field(liu_ordered, liu_pairs),
        "targets": liu_cpu.arrays["targets"],
        "query_pairs": liu_cpu.arrays["query_pairs"],
        "support_pairs": liu_cpu.arrays["support_pairs"],
        "retention": liu_cpu.arrays["retention"],
    }
    generic_weights = torch.cat([row[2] for row in batches]).cpu().numpy()
    liu_weights = liu_states[-1].cpu().numpy()
    alpha = model.alpha.detach().cpu().numpy()

    def storage(values: np.ndarray) -> dict[str, np.ndarray]:
        effective = values * alpha
        return {
            "P_abs_mean": np.abs(values).mean((1, 2)),
            "P_abs_max": np.abs(values).max((1, 2)),
            "A_abs_mean": np.abs(effective).mean((1, 2)),
            "A_abs_max": np.abs(effective).max((1, 2)),
            "zero_fraction": (values == 0).mean((1, 2)),
            "boundary_fraction": (np.abs(values) == 50).mean((1, 2)),
        }

    with np.load(
        _parent_raw(lock, seed, panel, condition, cell), allow_pickle=False
    ) as raw:
        replay_errors = {
            "generic_margins": float(
                np.max(np.abs(generic["margins"] - raw["generic__margins"]))
            ),
            "liu_margins": float(
                np.max(np.abs(liu["margins"] - raw["liu__bundles__intact__logits"]))
            ),
        }
        for domain, values in (
            ("generic", generic_weights),
            ("liu__storage", liu_weights),
        ):
            for name, observed in storage(values).items():
                replay_errors[f"{domain}_{name}"] = float(
                    np.max(np.abs(observed - raw[f"{domain}__{name}"]))
                )
        signs = raw["generic__signs"]
        generic_ce = np.logaddexp(0.0, -signs * generic["margins"]).mean(1)
        endpoint_errors = {
            "generic_ce": float(np.max(np.abs(generic_ce - raw["generic__ce"])))
        }
        liu_values = liu_endpoints(
            liu["margins"],
            liu["targets"],
            liu["query_pairs"],
            liu["support_pairs"],
            liu["retention"],
        )
        for name in ("liu_learned", "liu_nonlearned", "liu_omitted"):
            parent_name = name.removeprefix("liu_")
            endpoint_errors[name] = float(
                np.max(
                    np.abs(
                        liu_values[name]
                        - raw[f"liu__endpoints__intact__probability__{parent_name}"]
                    )
                )
            )
    return {
        "model": model,
        "generic": generic,
        "liu": liu,
        "generic_batches": batches,
        "liu_batch": (liu_cpu, liu_states[-1]),
        "replay_errors": replay_errors,
        "endpoint_replay_errors": endpoint_errors,
    }


def _endpoints(payload: dict, scale: float = 1.0) -> dict[str, np.ndarray]:
    generic = generic_endpoints(
        scale * payload["generic"]["margins"],
        payload["generic"]["targets"],
        payload["generic"]["learned"],
    )
    liu = liu_endpoints(
        scale * payload["liu"]["margins"],
        payload["liu"]["targets"],
        payload["liu"]["query_pairs"],
        payload["liu"]["support_pairs"],
        payload["liu"]["retention"],
    )
    return {**generic, **liu}


def _schedule(times: np.ndarray, name: str, seed: int) -> np.ndarray:
    result = np.asarray(times).copy()
    if name == "reverse":
        return result[::-1].copy()
    if name == "deranged":
        return result[derangement(len(result), seed)].copy()
    if name == "mean_clamp":
        result.fill(float(np.mean(result)))
        return result
    if name != "original":
        raise ValueError(name)
    return result


def _stage2_unit(payload: dict, no_time: dict, seed: int, panel: int) -> dict:
    model = payload["model"]
    summaries = {"support": {}, "query": {}}
    for domain, batches in (
        ("generic", payload["generic_batches"]),
        ("liu", [("liu-8", *payload["liu_batch"])]),
    ):
        baseline_fields, baseline_weights = [], []
        changed = {
            name: {"fields": [], "weights": [], "margins": []}
            for name in ("reverse", "deranged", "mean_clamp")
        }
        for index, row in enumerate(batches):
            _, cpu, original_weights = row
            original_ordered, original_pairs = read_ordered(
                model, original_weights, cpu.arrays["item_codes"]
            )
            baseline_fields.append(canonical_field(original_ordered, original_pairs))
            baseline_weights.append(original_weights.cpu().numpy())
            _, times = clean_inputs(cpu.arrays["support_inputs"])
            for name in changed:
                schedule = _schedule(times, name, 913711 + 1000 * panel + seed + index)
                states, _, _ = support_trajectory(model, cpu, support_times=schedule)
                ordered, pairs = read_ordered(
                    model, states[-1], cpu.arrays["item_codes"]
                )
                changed[name]["fields"].append(canonical_field(ordered, pairs))
                changed[name]["weights"].append(states[-1].cpu().numpy())
                changed[name]["margins"].append(read_original(model, states[-1], cpu))
        base_field = np.concatenate(baseline_fields)
        base_weights = np.concatenate(baseline_weights)
        no_time_field = no_time[domain]["fields"]
        baseline_to_no_time = float(np.sqrt(np.mean((base_field - no_time_field) ** 2)))
        alpha = model.alpha.detach().cpu().numpy()
        summaries["support"][domain] = {}
        for name, values in changed.items():
            field = np.concatenate(values["fields"])
            weights = np.concatenate(values["weights"])
            margins = np.concatenate(values["margins"])
            endpoints = (
                generic_endpoints(
                    margins,
                    payload["generic"]["targets"],
                    payload["generic"]["learned"],
                )
                if domain == "generic"
                else liu_endpoints(
                    margins,
                    payload["liu"]["targets"],
                    payload["liu"]["query_pairs"],
                    payload["liu"]["support_pairs"],
                    payload["liu"]["retention"],
                )
            )
            summaries["support"][domain][name] = {
                "D_P_mean": float(
                    np.mean(effective_distance(base_weights, weights, alpha))
                ),
                "D_M_rms": float(np.sqrt(np.mean((field - base_field) ** 2))),
                "to_no_time_D_M_rms": float(
                    np.sqrt(np.mean((field - no_time_field) ** 2))
                ),
                "endpoint_means": {
                    key: float(np.mean(value)) for key, value in endpoints.items()
                },
            }
            summaries["support"][domain][name]["moves_toward_no_time"] = (
                summaries["support"][domain][name]["to_no_time_D_M_rms"]
                < baseline_to_no_time
            )
        summaries["support"][domain]["baseline_to_no_time_D_M_rms"] = (
            baseline_to_no_time
        )
        query = {}
        for value in (0.0, 1.0 / 3.0, 2.0 / 3.0):
            fields, margins = [], []
            for row in batches:
                _, cpu, weights = row
                ordered, pairs = read_ordered(
                    model, weights, cpu.arrays["item_codes"], query_time=value
                )
                fields.append(canonical_field(ordered, pairs))
                margins.append(read_original(model, weights, cpu, query_time=value))
            query[str(value)] = (np.concatenate(fields), np.concatenate(margins))
        base_query = query[str(2.0 / 3.0)][0]
        summaries["query"][domain] = {}
        for key, (field, margins) in query.items():
            endpoints = (
                generic_endpoints(
                    margins,
                    payload["generic"]["targets"],
                    payload["generic"]["learned"],
                )
                if domain == "generic"
                else liu_endpoints(
                    margins,
                    payload["liu"]["targets"],
                    payload["liu"]["query_pairs"],
                    payload["liu"]["support_pairs"],
                    payload["liu"]["retention"],
                )
            )
            summaries["query"][domain][key] = {
                "D_M_rms": float(np.sqrt(np.mean((field - base_query) ** 2))),
                "endpoint_means": {
                    name: float(np.mean(value)) for name, value in endpoints.items()
                },
            }
    return summaries


def _stage3(lock: dict, spec: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
    summaries = {}
    for seed in spec["design"]["network_seeds"]:
        summaries[str(seed)] = {}
        for condition in spec["design"]["conditions"]:
            model = load_model(_checkpoint(lock, seed, condition, "noisy"), condition)
            panels = []
            summaries[str(seed)][condition] = {"panels": {}}
            for panel in spec["design"]["evaluation_panels"]:
                recipe = inherited_recipe(panel)
                rows = []
                episode_offset = 0
                for name, record in sorted(_inputs(lock, panel).items()):
                    if not name.startswith("test-"):
                        continue
                    cpu = selected_generic(load_cpu(record, "noisy", recipe))
                    row = trajectory_rows(model, cpu)
                    row["episode"] += episode_offset
                    episode_offset += len(np.unique(row["episode"]))
                    rows.append(row)
                combined = {
                    key: np.concatenate([row[key] for row in rows]) for key in rows[0]
                }
                panels.append(combined)
                prefix_write = [
                    float(
                        np.mean(
                            combined["write_susceptibility"][
                                combined["prefix"] == prefix
                            ]
                        )
                    )
                    for prefix in np.unique(combined["prefix"])
                ]
                prefix_functional = [
                    float(
                        np.mean(
                            combined["functional_susceptibility"][
                                combined["prefix"] == prefix
                            ]
                        )
                    )
                    for prefix in np.unique(combined["prefix"])
                ]
                summaries[str(seed)][condition]["panels"][str(panel)] = {
                    "episodes": len(np.unique(combined["episode"])),
                    "prefixes": len(np.unique(combined["prefix"])),
                    "write_susceptibility_range": max(prefix_write) - min(prefix_write),
                    "functional_susceptibility_range": max(prefix_functional)
                    - min(prefix_functional),
                }
                for key, value in combined.items():
                    arrays[f"stage3/{seed}/{condition}/{panel}/{key}"] = value
            regression = regression_summary(
                panels, seed=961000 + seed, draws=spec["statistics"]["bootstrap_draws"]
            )
            summaries[str(seed)][condition]["regression"] = regression
            summaries[str(seed)][condition]["state_dependent_plasticity_present"] = all(
                row["write_susceptibility_range"] > 1e-5
                and row["functional_susceptibility_range"] > 1e-5
                for row in summaries[str(seed)][condition]["panels"].values()
            )
            summaries[str(seed)][condition]["confidence_link_supported"] = (
                regression["interval"]["upper"] < 0.0
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return summaries, arrays


def _baseline_stages(
    lock: dict, spec: dict, arrays: dict[str, np.ndarray]
) -> tuple[dict, dict, float, float]:
    stage1: dict[str, dict] = {}
    stage2: dict[str, dict] = {}
    max_replay_error = 0.0
    max_endpoint_replay_error = 0.0
    for seed in spec["design"]["network_seeds"]:
        stage1[str(seed)], stage2[str(seed)] = {}, {}
        for panel in spec["design"]["evaluation_panels"]:
            stage1[str(seed)][str(panel)] = {}
            for cell in (
                *spec["design"]["secondary_cells"],
                spec["design"]["primary_cell"],
            ):
                cell_result = _baseline_cell(lock, spec, arrays, seed, panel, cell)
                stage1[str(seed)][str(panel)][cell] = cell_result["geometry"]
                if cell_result["stage2"] is not None:
                    stage2[str(seed)][str(panel)] = cell_result["stage2"]
                max_replay_error = max(
                    max_replay_error, cell_result["max_replay_error"]
                )
                max_endpoint_replay_error = max(
                    max_endpoint_replay_error,
                    cell_result["max_endpoint_replay_error"],
                )
    return stage1, stage2, max_replay_error, max_endpoint_replay_error


def _baseline_cell(
    lock: dict,
    spec: dict,
    arrays: dict[str, np.ndarray],
    seed: int,
    panel: int,
    cell: str,
) -> dict:
    rows = {
        condition: _collect_baseline(lock, seed, panel, condition, cell)
        for condition in spec["design"]["conditions"]
    }
    nt, control = rows["clean_no_time"], rows["time_retained_control"]
    summary = {
        "geometry": {
            domain: geometry(nt[domain]["fields"], control[domain]["fields"], 1e-5)
            for domain in ("generic", "liu")
        },
        "stage2": None,
        "max_replay_error": max(
            value for row in rows.values() for value in row["replay_errors"].values()
        ),
        "max_endpoint_replay_error": max(
            value
            for row in rows.values()
            for value in row["endpoint_replay_errors"].values()
        ),
    }
    for condition, row in rows.items():
        for domain in ("generic", "liu"):
            arrays[f"{seed}/{panel}/{cell}/{condition}/{domain}/fields"] = row[domain][
                "fields"
            ]
            arrays[f"{seed}/{panel}/{cell}/{condition}/{domain}/margins"] = row[domain][
                "margins"
            ]
    if cell == "Ce":
        summary["stage2"] = _stage2_unit(control, nt, seed, panel)
        for name in ("targets", "learned"):
            arrays[f"{seed}/{panel}/Ce/generic/{name}"] = nt["generic"][name]
        for name in ("targets", "query_pairs", "support_pairs", "retention"):
            arrays[f"{seed}/{panel}/Ce/liu/{name}"] = nt["liu"][name]
    for row in rows.values():
        del row["model"]
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _scale_stage(spec: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    scales, held_out, scale_labels = {}, {}, {}
    endpoint_names = (
        "generic_learned",
        "generic_nonlearned",
        "liu_learned",
        "liu_nonlearned",
        "liu_omitted",
    )
    for seed in spec["design"]["network_seeds"]:
        candidate = np.concatenate(
            [
                arrays[f"{seed}/1/Ce/clean_no_time/{domain}/fields"]
                for domain in ("generic", "liu")
            ]
        )
        control = np.concatenate(
            [
                arrays[f"{seed}/1/Ce/time_retained_control/{domain}/fields"]
                for domain in ("generic", "liu")
            ]
        )
        scales[str(seed)] = positive_scale(candidate, control)
        held_out[str(seed)] = {
            endpoint: _held_out_endpoint(
                arrays,
                seed,
                endpoint,
                scales[str(seed)],
                endpoint_index,
            )
            for endpoint_index, endpoint in enumerate(endpoint_names)
        }
        scale_labels[str(seed)] = all(
            held_out[str(seed)][endpoint]["interval"]["lower"] >= -0.02
            for endpoint in endpoint_names
        )
    return scales, held_out, scale_labels


def _held_out_endpoint(
    arrays: dict[str, np.ndarray],
    seed: int,
    endpoint: str,
    scale: float,
    endpoint_index: int,
) -> dict:
    panels = []
    for panel in (2, 3):
        generic_metadata = {
            name: arrays[f"{seed}/{panel}/Ce/generic/{name}"]
            for name in ("targets", "learned")
        }
        liu_metadata = {
            name: arrays[f"{seed}/{panel}/Ce/liu/{name}"]
            for name in ("targets", "query_pairs", "support_pairs", "retention")
        }
        values = {}
        for condition, multiplier in (
            ("clean_no_time", scale),
            ("time_retained_control", 1.0),
        ):
            values[condition] = {
                **generic_endpoints(
                    multiplier
                    * arrays[f"{seed}/{panel}/Ce/{condition}/generic/margins"],
                    generic_metadata["targets"],
                    generic_metadata["learned"],
                ),
                **liu_endpoints(
                    multiplier * arrays[f"{seed}/{panel}/Ce/{condition}/liu/margins"],
                    liu_metadata["targets"],
                    liu_metadata["query_pairs"],
                    liu_metadata["support_pairs"],
                    liu_metadata["retention"],
                ),
            }
        panels.append(
            values["clean_no_time"][endpoint]
            - values["time_retained_control"][endpoint]
        )
    return paired_interval(panels, seed=951000 + seed * 10 + endpoint_index)


def _time_path_labels(spec: dict, stage2: dict) -> dict:
    labels = {}
    for seed in spec["design"]["network_seeds"]:
        panels = stage2[str(seed)].values()
        support_present = all(
            all(
                any(
                    row["support"][domain][condition]["D_P_mean"] > 1e-5
                    and row["support"][domain][condition]["D_M_rms"] > 1e-5
                    for condition in ("reverse", "deranged", "mean_clamp")
                )
                for domain in ("generic", "liu")
            )
            for row in panels
        )
        panels = stage2[str(seed)].values()
        query_present = all(
            all(
                any(
                    row["query"][domain][str(value)]["D_M_rms"] > 1e-5
                    for value in (0.0, 1.0 / 3.0)
                )
                for domain in ("generic", "liu")
            )
            for row in panels
        )
        labels[str(seed)] = {
            "support_time_path_present": support_present,
            "query_time_path_present": query_present,
            "query_terminal_P_bitwise_identical": True,
        }
    return labels


def _classify(scale_labels: dict, stage2_labels: dict, stage3: dict) -> dict:
    labels = {
        "scale_sufficient": scale_labels["3011"],
        "scale_insufficient": not scale_labels["3011"],
        "per_seed_time_paths": stage2_labels,
        "per_seed_state_dynamics": {
            seed: {
                condition: {
                    key: row[key]
                    for key in (
                        "state_dependent_plasticity_present",
                        "confidence_link_supported",
                    )
                }
                for condition, row in conditions.items()
            }
            for seed, conditions in stage3.items()
        },
    }
    signatures = {
        seed: (
            stage2_labels[seed],
            labels["per_seed_state_dynamics"][seed],
        )
        for seed in stage2_labels
    }
    labels["mechanism_heterogeneous"] = any(
        signatures[seed] != signatures["3011"] for seed in ("3012", "3013")
    )
    return labels


def run() -> dict:
    require_formal_runtime()
    spec, lock = specification(), validate_source_lock()
    arrays: dict[str, np.ndarray] = {}
    stage1, stage2, max_replay_error, max_endpoint_replay_error = _baseline_stages(
        lock, spec, arrays
    )
    if max_replay_error > 1e-5:
        raise RuntimeError(f"parent baseline replay differs: {max_replay_error}")
    if max_endpoint_replay_error > 1e-10:
        raise RuntimeError(
            f"parent endpoint replay differs: {max_endpoint_replay_error}"
        )
    scales, held_out, scale_labels = _scale_stage(spec, arrays)
    stage2_labels = _time_path_labels(spec, stage2)
    stage3, arrays = _stage3(lock, spec, arrays)
    labels = _classify(scale_labels, stage2_labels, stage3)
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE_LOCK),
        "parent_outcome_unchanged": "time_removal_failure",
        "maximum_parent_replay_error": max_replay_error,
        "maximum_parent_endpoint_replay_error": max_endpoint_replay_error,
        "stage_1": {
            "geometry": stage1,
            "scales": scales,
            "held_out_scaled_noninferiority": held_out,
        },
        "stage_2": {"effects": stage2, "labels": stage2_labels},
        "stage_3": stage3,
        "classification": labels,
        "parent_evaluation_blank_audit": {
            "constructed_by_legacy_adapter": True,
            "effective_P_write": False,
            "reason": "With h=E=P=0, two blank steps leave P exactly zero; resulting h/E are discarded before support.",
        },
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    write_arrays(RUNS / "arrays.npz", arrays)
    write_json_exclusive(RUNS / "result.json", json_ready(result))
    return {
        "maximum_parent_replay_error": max_replay_error,
        "maximum_parent_endpoint_replay_error": max_endpoint_replay_error,
        "labels": labels,
    }


__all__ = ["run"]
