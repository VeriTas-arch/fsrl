"""Phase 2: unchanged time-role estimands from the frozen direct baseline."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.single_p_time_role.analysis import (
    generic_endpoints,
    geometry,
    liu_endpoints,
    paired_interval,
)
from fsrl.experiments.single_p_time_role.estimands import (
    canonical_field,
    derangement,
    effective_distance,
    positive_scale,
)
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.infra.provenance import load_json, write_json_exclusive

from .batches import subset_batch
from .direct import (
    clean_inputs,
    load_cpu,
    load_model,
    read_ordered,
    read_original,
    support_trajectory,
)
from .external import CELLS
from .locks import (
    reference,
    validate_baseline_artifact_lock,
    validate_mechanism_source_lock,
    verify_reference,
)
from .protocol import (
    BASELINE_ARTIFACT_LOCK,
    MECHANISM_RUNS,
    MECHANISM_SOURCE_LOCK,
    PROTOCOL_SHA256,
    RESULT,
    specification,
)
from .storage import load_npz, write_npz_exclusive
from .trajectory import regression_summary, trajectory_rows


def _inputs(lock: dict, panel: int) -> dict[str, dict]:
    prefix = f"{panel}/"
    return {
        key.removeprefix(prefix): value
        for key, value in lock["parents"]["inputs"].items()
        if key.startswith(prefix)
    }


def _checkpoint(lock: dict, seed: int, condition: str, arm: str) -> dict:
    return lock["parents"]["checkpoints"][f"{seed}/{condition}/{arm}"]


def _flat(seed: int, panel: int, condition: str, cell: str, name: str) -> dict:
    return load_npz(
        MECHANISM_RUNS.parent
        / "baseline"
        / str(seed)
        / str(panel)
        / condition
        / cell
        / f"{name}.npz"
    )


def _group(flat: dict, prefix: str) -> dict:
    marker = prefix + "__"
    return {
        key.removeprefix(marker): value
        for key, value in flat.items()
        if key.startswith(marker)
    }


def _bootstrap_draws(lock: dict) -> int:
    authority = load_json(verify_reference(lock["parents"]["scientific_estimands"]))
    draws = authority["statistics"]["bootstrap_draws"]
    if draws != 2000:
        raise RuntimeError("inherited Stage-3 bootstrap count differs")
    return draws


def _validate_batch_metadata(flat: dict, prefix: str, cpu) -> None:
    expected = len(cpu.arrays["item_codes"])
    observed = int(flat[f"{prefix}__source_batch_size"])
    if observed != expected:
        raise RuntimeError(f"locked batch size differs for {prefix}")
    for name in ("support_inputs", "query_inputs"):
        shape = flat[f"{prefix}__source_{name.removesuffix('_inputs')}_shape"]
        if not np.array_equal(shape, cpu.arrays[name].shape):
            raise RuntimeError(f"locked {name} shape differs for {prefix}")


def _stage1(lock: dict, spec: dict, arrays: dict) -> tuple[dict, dict, dict]:
    summaries, payloads = {}, {}
    for seed in spec["design"]["network_seeds"]:
        summaries[str(seed)], payloads[str(seed)] = {}, {}
        for panel in spec["design"]["evaluation_panels"]:
            summaries[str(seed)][str(panel)] = {}
            payloads[str(seed)][str(panel)] = {}
            inputs = _inputs(lock, panel)
            recipe = inherited_recipe(panel)
            for cell, settings in CELLS.items():
                rows = {}
                for condition in spec["design"]["architecture_conditions"]:
                    flat = _flat(seed, panel, condition, cell, "external")
                    generic = _group(flat, "generic")
                    liu = _group(flat, "liu")
                    fields = _group(flat, "stage1")
                    rows[condition] = {
                        "generic": {
                            "fields": fields["generic_fields"],
                            "margins": generic["margins"],
                        },
                        "liu": {
                            "fields": fields["liu_fields"],
                            "margins": liu["bundles__intact__logits"],
                        },
                    }
                    arrays[
                        f"stage1__{seed}__{panel}__{cell}__{condition}__generic_fields"
                    ] = fields["generic_fields"]
                    arrays[
                        f"stage1__{seed}__{panel}__{cell}__{condition}__liu_fields"
                    ] = fields["liu_fields"]
                summaries[str(seed)][str(panel)][cell] = {
                    domain: geometry(
                        rows["clean_no_time"][domain]["fields"],
                        rows["time_retained_control"][domain]["fields"],
                        1e-5,
                    )
                    for domain in ("generic", "liu")
                }
                if cell == "Ce":
                    generic_cpu = [
                        load_cpu(record, settings["observation"], recipe)
                        for name, record in sorted(inputs.items())
                        if name.startswith("test-")
                    ]
                    liu_cpu = load_cpu(inputs["liu-8"], settings["observation"], recipe)
                    payloads[str(seed)][str(panel)] = {
                        **rows,
                        "generic_targets": np.concatenate(
                            [
                                cpu.arrays["targets"]
                                .reshape(-1, len(cpu.arrays["item_codes"]))
                                .T
                                for cpu in generic_cpu
                            ]
                        ),
                        "generic_learned": np.concatenate(
                            [cpu.arrays["learned"] for cpu in generic_cpu]
                        ),
                        "liu_targets": liu_cpu.arrays["targets"],
                        "liu_query_pairs": liu_cpu.arrays["query_pairs"],
                        "liu_support_pairs": liu_cpu.arrays["support_pairs"],
                        "liu_retention": liu_cpu.arrays["retention"],
                    }
    scales, held_out, labels = {}, {}, {}
    names = (
        "generic_learned",
        "generic_nonlearned",
        "liu_learned",
        "liu_nonlearned",
        "liu_omitted",
    )
    for seed in spec["design"]["network_seeds"]:
        row = payloads[str(seed)]["1"]
        candidate = np.concatenate(
            [row["clean_no_time"][domain]["fields"] for domain in ("generic", "liu")]
        )
        control = np.concatenate(
            [
                row["time_retained_control"][domain]["fields"]
                for domain in ("generic", "liu")
            ]
        )
        scale = positive_scale(candidate, control)
        scales[str(seed)] = scale
        held_out[str(seed)] = {}
        for endpoint_index, endpoint in enumerate(names):
            panel_values = []
            for panel in (2, 3):
                value = payloads[str(seed)][str(panel)]
                endpoint_values = {}
                for condition, multiplier in (
                    ("clean_no_time", scale),
                    ("time_retained_control", 1.0),
                ):
                    endpoint_values[condition] = {
                        **generic_endpoints(
                            multiplier * value[condition]["generic"]["margins"],
                            value["generic_targets"],
                            value["generic_learned"],
                        ),
                        **liu_endpoints(
                            multiplier * value[condition]["liu"]["margins"],
                            value["liu_targets"],
                            value["liu_query_pairs"],
                            value["liu_support_pairs"],
                            value["liu_retention"],
                        ),
                    }
                panel_values.append(
                    endpoint_values["clean_no_time"][endpoint]
                    - endpoint_values["time_retained_control"][endpoint]
                )
            held_out[str(seed)][endpoint] = paired_interval(
                panel_values, seed=951000 + seed * 10 + endpoint_index
            )
        labels[str(seed)] = all(
            held_out[str(seed)][name]["interval"]["lower"] >= -0.02 for name in names
        )
    return (
        {
            "geometry": summaries,
            "scales": scales,
            "held_out_scaled_noninferiority": held_out,
        },
        labels,
        payloads,
    )


def _schedule(times: np.ndarray, name: str, seed: int) -> np.ndarray:
    result = np.asarray(times).copy()
    if name == "reverse":
        return result[::-1].copy()
    if name == "deranged":
        return result[derangement(len(result), seed)].copy()
    if name == "mean_clamp":
        result.fill(float(np.mean(result)))
        return result
    raise ValueError(name)


def _stage2_domain(
    model, batches, base_field, base_weights, no_time_field, metadata, seed, panel
):
    changed = {
        name: {"fields": [], "weights": [], "margins": []}
        for name in ("reverse", "deranged", "mean_clamp")
    }
    for index, (_, cpu, _) in enumerate(batches):
        _, times = clean_inputs(cpu.arrays["support_inputs"])
        for name in changed:
            states, _, _, _ = support_trajectory(
                model,
                cpu,
                support_times=_schedule(
                    times, name, 913711 + 1000 * panel + seed + index
                ),
            )
            ordered, pairs = read_ordered(model, states[-1], cpu.arrays["item_codes"])
            changed[name]["fields"].append(canonical_field(ordered, pairs))
            changed[name]["weights"].append(states[-1].cpu().numpy())
            changed[name]["margins"].append(read_original(model, states[-1], cpu))
    baseline_to_no_time = float(np.sqrt(np.mean((base_field - no_time_field) ** 2)))
    alpha = model.alpha.detach().cpu().numpy()
    support = {}
    for name, values in changed.items():
        field = np.concatenate(values["fields"])
        weights = np.concatenate(values["weights"])
        margins = np.concatenate(values["margins"])
        endpoints = (
            generic_endpoints(margins, metadata["targets"], metadata["learned"])
            if "learned" in metadata
            else liu_endpoints(
                margins,
                metadata["targets"],
                metadata["query_pairs"],
                metadata["support_pairs"],
                metadata["retention"],
            )
        )
        support[name] = {
            "D_P_mean": float(
                np.mean(effective_distance(base_weights, weights, alpha))
            ),
            "D_M_rms": float(np.sqrt(np.mean((field - base_field) ** 2))),
            "to_no_time_D_M_rms": float(np.sqrt(np.mean((field - no_time_field) ** 2))),
            "endpoint_means": {
                key: float(np.nanmean(value)) for key, value in endpoints.items()
            },
        }
        support[name]["moves_toward_no_time"] = (
            support[name]["to_no_time_D_M_rms"] < baseline_to_no_time
        )
    support["baseline_to_no_time_D_M_rms"] = baseline_to_no_time
    query_values = {}
    for value in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        fields, margins = [], []
        for _, cpu, weights in batches:
            ordered, pairs = read_ordered(
                model, weights, cpu.arrays["item_codes"], query_time=value
            )
            fields.append(canonical_field(ordered, pairs))
            margins.append(read_original(model, weights, cpu, query_time=value))
        query_values[str(value)] = (np.concatenate(fields), np.concatenate(margins))
    base_query = query_values[str(2.0 / 3.0)][0]
    query = {}
    for key, (field, margins) in query_values.items():
        endpoints = (
            generic_endpoints(margins, metadata["targets"], metadata["learned"])
            if "learned" in metadata
            else liu_endpoints(
                margins,
                metadata["targets"],
                metadata["query_pairs"],
                metadata["support_pairs"],
                metadata["retention"],
            )
        )
        query[key] = {
            "D_M_rms": float(np.sqrt(np.mean((field - base_query) ** 2))),
            "endpoint_means": {
                name: float(np.nanmean(value)) for name, value in endpoints.items()
            },
        }
    return {"support": support, "query": query}


def _stage2(lock: dict, spec: dict, stage1_payloads: dict) -> tuple[dict, dict]:
    effects = {}
    for seed in spec["design"]["network_seeds"]:
        effects[str(seed)] = {}
        model = load_model(
            _checkpoint(lock, seed, "time_retained_control", "noisy"),
            "time_retained_control",
        )
        for panel in spec["design"]["evaluation_panels"]:
            recipe, inputs = inherited_recipe(panel), _inputs(lock, panel)
            internal = _flat(seed, panel, "time_retained_control", "Ce", "internal")
            external = _flat(seed, panel, "time_retained_control", "Ce", "external")
            no_time = _flat(seed, panel, "clean_no_time", "Ce", "external")
            generic_batches = []
            for name, record in sorted(inputs.items()):
                if name.startswith("test-"):
                    cpu = load_cpu(record, "noisy", recipe)
                    weights = torch.from_numpy(
                        internal[f"generic__{name}__terminal_P"]
                    ).to("cuda")
                    _validate_batch_metadata(internal, f"generic__{name}", cpu)
                    generic_batches.append((name, cpu, weights))
            liu_cpu = load_cpu(inputs["liu-8"], "noisy", recipe)
            liu_weights = torch.from_numpy(internal["liu__terminal_P"]).to("cuda")
            _validate_batch_metadata(internal, "liu", liu_cpu)
            payload = stage1_payloads[str(seed)][str(panel)]
            generic = _stage2_domain(
                model,
                generic_batches,
                external["stage1__generic_fields"],
                np.concatenate([row[2].cpu().numpy() for row in generic_batches]),
                no_time["stage1__generic_fields"],
                {
                    "targets": payload["generic_targets"],
                    "learned": payload["generic_learned"],
                },
                seed,
                panel,
            )
            liu = _stage2_domain(
                model,
                [("liu-8", liu_cpu, liu_weights)],
                external["stage1__liu_fields"],
                liu_weights.cpu().numpy(),
                no_time["stage1__liu_fields"],
                {
                    "targets": payload["liu_targets"],
                    "query_pairs": payload["liu_query_pairs"],
                    "support_pairs": payload["liu_support_pairs"],
                    "retention": payload["liu_retention"],
                },
                seed,
                panel,
            )
            effects[str(seed)][str(panel)] = {
                "support": {"generic": generic["support"], "liu": liu["support"]},
                "query": {"generic": generic["query"], "liu": liu["query"]},
            }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    labels = {}
    for seed, panels in effects.items():
        support = all(
            all(
                any(
                    row["support"][domain][name]["D_P_mean"] > 1e-5
                    and row["support"][domain][name]["D_M_rms"] > 1e-5
                    for name in ("reverse", "deranged", "mean_clamp")
                )
                for domain in ("generic", "liu")
            )
            for row in panels.values()
        )
        query = all(
            all(
                any(
                    row["query"][domain][str(value)]["D_M_rms"] > 1e-5
                    for value in (0.0, 1.0 / 3.0)
                )
                for domain in ("generic", "liu")
            )
            for row in panels.values()
        )
        labels[seed] = {
            "support_time_path_present": support,
            "query_time_path_present": query,
            "query_terminal_P_bitwise_identical": True,
        }
    return effects, labels


def _stage3(lock: dict, spec: dict, arrays: dict) -> dict:
    summaries = {}
    for seed in spec["design"]["network_seeds"]:
        summaries[str(seed)] = {}
        for condition in spec["design"]["architecture_conditions"]:
            model = load_model(_checkpoint(lock, seed, condition, "noisy"), condition)
            panels = []
            summaries[str(seed)][condition] = {"panels": {}}
            for panel in spec["design"]["evaluation_panels"]:
                recipe, inputs = inherited_recipe(panel), _inputs(lock, panel)
                internal = _flat(seed, panel, condition, "Ce", "internal")
                rows, episode_offset = [], 0
                for name, record in sorted(inputs.items()):
                    if not name.startswith("test-"):
                        continue
                    cpu = load_cpu(record, "noisy", recipe)
                    positions = internal[f"generic__{name}__selected_positions"]
                    _validate_batch_metadata(internal, f"generic__{name}", cpu)
                    full_prefix = internal[f"generic__{name}__full_prefix_P"]
                    selected_prefix = internal[f"generic__{name}__prefix_P"]
                    if not np.array_equal(full_prefix[:, positions], selected_prefix):
                        raise RuntimeError("selected prefix is not a full-batch slice")
                    selected = subset_batch(cpu, positions)
                    expected = internal[f"generic__{name}__selected_episode_indices"]
                    if not np.array_equal(selected.arrays["episode_indices"], expected):
                        raise RuntimeError("frozen Stage-3 episode selection differs")
                    row = trajectory_rows(
                        model,
                        selected,
                        selected_prefix,
                        internal[f"generic__{name}__natural_modulation"],
                    )
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
                    arrays[f"stage3__{seed}__{condition}__{panel}__{key}"] = value
            regression = regression_summary(
                panels, seed=961000 + seed, draws=_bootstrap_draws(lock)
            )
            row = summaries[str(seed)][condition]
            row["regression"] = regression
            row["state_dependent_plasticity_present"] = all(
                panel["write_susceptibility_range"] > 1e-5
                and panel["functional_susceptibility_range"] > 1e-5
                for panel in row["panels"].values()
            )
            row["confidence_link"] = (
                "supported"
                if regression["interval"]["upper"] < 0.0
                else "not_supported"
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return summaries


def _classify(scale: dict, stage2: dict, stage3: dict) -> dict:
    dynamics = {
        seed: {
            condition: {
                "state_dependent_plasticity_present": row[
                    "state_dependent_plasticity_present"
                ],
                "confidence_link": row["confidence_link"],
            }
            for condition, row in conditions.items()
        }
        for seed, conditions in stage3.items()
    }
    signatures = {seed: (stage2[seed], dynamics[seed]) for seed in stage2}
    return {
        "scale_sufficient": scale["3011"],
        "scale_insufficient": not scale["3011"],
        "per_seed_time_paths": stage2,
        "per_seed_state_dynamics": dynamics,
        "mechanism_heterogeneous": any(
            signatures[seed] != signatures["3011"] for seed in ("3012", "3013")
        ),
    }


def run(runtime: dict) -> dict:
    spec = specification()
    lock = validate_mechanism_source_lock(runtime)
    validate_baseline_artifact_lock()
    if RESULT.exists() or MECHANISM_RUNS.exists():
        raise RuntimeError("direct-authority mechanism execution is write-once")
    arrays = {}
    try:
        with torch.no_grad():
            stage1, scale_labels, payloads = _stage1(lock, spec, arrays)
            stage2, stage2_labels = _stage2(lock, spec, payloads)
            stage3 = _stage3(lock, spec, arrays)
        classification = _classify(scale_labels, stage2_labels, stage3)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(MECHANISM_SOURCE_LOCK),
            "baseline_artifact_lock": reference(BASELINE_ARTIFACT_LOCK),
            "runtime": runtime,
            "outcome": "interpretable",
            "parent_outcome_unchanged": "time_removal_failure",
            "stage_1": stage1,
            "stage_2": {"effects": stage2, "labels": stage2_labels},
            "stage_3": stage3,
            "classification": classification,
        }
        MECHANISM_RUNS.mkdir(parents=True, exist_ok=False)
        write_npz_exclusive(MECHANISM_RUNS / "arrays.npz", arrays)
        write_json_exclusive(MECHANISM_RUNS / "result.json", json_ready(result))
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(
            RESULT,
            json_ready(
                {
                    **result,
                    "runtime_arrays": reference(MECHANISM_RUNS / "arrays.npz"),
                }
            ),
        )
        return {"outcome": "interpretable", "classification": classification}
    except Exception as error:
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_lock": reference(MECHANISM_SOURCE_LOCK),
            "baseline_artifact_lock": reference(BASELINE_ARTIFACT_LOCK),
            "runtime": runtime,
            "outcome": "noninterpretable",
            "failure": str(error),
            "parent_outcome_unchanged": "time_removal_failure",
        }
        if not RESULT.exists():
            write_json_exclusive(RESULT, failure)
        raise


__all__ = ["run"]
