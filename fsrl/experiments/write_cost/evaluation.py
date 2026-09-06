"""Frozen generic tradeoffs and unchanged Liu competence/organization endpoints."""

import json

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import liu_rollouts, margin_bundle
from fsrl.experiments.evidence_routing.measurement import internal_preferences
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.measurement import (
    competence,
    core_flags,
    removal_effects,
    summarize_behavior,
)
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.training_strategy.estimands import estimate, query_endpoints
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    summarize_endpoints,
)
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import configure_execution
from .inputs import load_input
from .locks import MODELS, SOURCE, completed, training_dir, validate_phase
from .model import rollout, sequences
from .protocol import PROTOCOL_SHA256, RUNS, specification


def loaded_model(seed, arm, source):
    runtime = configure_execution()
    if json_ready(runtime) != source["runtime"]:
        raise RuntimeError("evaluation runtime mismatch")
    spec = specification()
    backbone, local = make_model(spec, seed, "dual", "cuda")
    metadata = completed(training_dir(seed, arm))
    for name, module, key in [
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ]:
        module.load_state_dict(
            torch.load(
                training_dir(seed, arm) / f"{name}.pth",
                weights_only=True,
                map_location="cuda",
            )
        )
        if tensor_hashes(module) != metadata[key]:
            raise RuntimeError("loaded parameters differ")
        module.requires_grad_(False).eval()
    return backbone, local, sequences(backbone, compiled=True)


def summarize_generic(arrays, spec, seed):
    endpoints = query_endpoints(
        arrays["margins"][..., None],
        arrays["signs"][..., None],
        {"learned": arrays["learned"], "nonlearned": ~arrays["learned"]},
        temperature=1.0,
    )
    summaries = summarize_endpoints(
        endpoints, spec["statistics"]["seed_offset"] + seed, spec["statistics"]
    )
    ce = np.logaddexp(0.0, -arrays["margins"] * arrays["signs"]).mean(1)
    arrays["ce"] = ce
    summary = {
        "summaries": summaries,
        "competence": competence(summaries),
        "ce": float(ce.mean()),
        "mean_cost": float(arrays["cost"].mean()),
        "mean_total_write": float(arrays["total_write"].mean()),
    }
    return summary


def generic_arrays(backbone, local, seqs, source, split):
    collected = {
        key: []
        for key in (
            "margins",
            "signs",
            "learned",
            "cost",
            "total_write",
            "episode_indices",
        )
    }
    for name, record in sorted(source["inputs"].items()):
        if not name.startswith(split + "-"):
            continue
        cpu = load_input(record)
        batch = cpu.to("cuda")
        result, cost, writes = rollout(backbone, local, seqs[0], seqs[1], batch)
        subjects = batch.support_inputs.shape[2]
        collected["margins"].append(margin_bundle(result.logits, subjects)["logits"])
        collected["signs"].append(
            (2 * cpu.arrays["targets"] - 1).reshape(-1, subjects).T
        )
        collected["learned"].append(cpu.arrays["learned"])
        collected["cost"].append(cost.cpu().numpy().astype(float))
        collected["total_write"].append(writes.sum(0).cpu().numpy().astype(float))
        collected["episode_indices"].append(cpu.arrays["episode_indices"])
    return {key: np.concatenate(values) for key, values in collected.items()}


def generic_run(seed, arm, split, source):
    directory = RUNS / "generic" / split / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs = loaded_model(seed, arm, source)
    identity = {
        "seed": seed,
        "arm": arm,
        "split": split,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="effective_write_cost_v1",
            execution_id=f"{split}-{seed}-{arm}",
            producer=identity,
            resolved_config={"evaluation": spec["evaluation"]["generic"]},
        ),
        torch.no_grad(),
    ):
        arrays = generic_arrays(backbone, local, seqs, source, split)
        result = {**identity, **summarize_generic(arrays, spec, seed)}
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def liu_cell(backbone, local, seqs, source, spec, seed, size, generic_passed):
    cpu = load_input(source["inputs"][f"liu-{size}"])
    protocol = size_protocol(spec, size)
    raw = liu_rollouts(backbone, local, seqs[1], cpu, protocol, spec)
    _, cost, writes = rollout(backbone, local, seqs[0], seqs[1], cpu.to("cuda"))
    raw["cost"] = cost.cpu().numpy()
    raw["total_write"] = writes.sum(0).cpu().numpy()
    endpoints = liu_endpoints(
        raw["bundles"],
        cpu.arrays["retention"],
        protocol,
        spec["evaluation"]["liu"]["temperature"],
    )
    boot = spec["statistics"]["seed_offset"] + seed
    summaries = {
        name: summarize_endpoints(values, boot, spec["statistics"])
        for name, values in endpoints.items()
    }
    behavior, behavior_arrays, sampled = summarize_behavior(
        raw["bundles"]["intact"], protocol, spec, seed
    )
    effects = removal_effects(
        raw["bundles"]["intact"]["logits"], raw["removed"], protocol
    )
    for name in raw["bundles"]:
        if name != "intact":
            for group in ("learned", "nonlearned", "omitted"):
                effects[f"intact_minus_{name}_{group}"] = (
                    endpoints["intact"]["probability"][group]
                    - endpoints[name]["probability"][group]
                )
    internal, internal_arrays = internal_preferences(
        raw["bundles"]["intact"]["logits"], protocol, spec, seed
    )
    global_internal, global_arrays = internal_preferences(
        raw["bundles"]["local_off"]["logits"], protocol, spec, seed
    )
    flags = core_flags(summaries["intact"], behavior, generic_passed)
    result = {
        "summaries": summaries,
        "behavior": behavior,
        "internal": internal,
        "global_internal": global_internal,
        "core_flags": flags,
        "core_passed": all(flags.values()),
        "competence": competence(summaries["intact"]),
        "mean_cost": float(raw["cost"].mean()),
        "mean_total_write": float(raw["total_write"].mean()),
        "effects": {
            key: estimate(value, seed=boot, statistics=spec["statistics"])
            for key, value in effects.items()
        },
    }
    return (
        result,
        {
            **raw,
            "endpoints": endpoints,
            "behavior": behavior_arrays,
            "internal": internal_arrays,
            "global_internal": global_arrays,
            "effects": effects,
        },
        sampled,
    )


def liu_run(seed, arm, sizes, source):
    phase = "liu" if sizes == [8] else "transport"
    directory = RUNS / phase / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    generic = completed(RUNS / "generic/test" / str(seed) / arm)
    backbone, local, seqs = loaded_model(seed, arm, source)
    identity = {"seed": seed, "arm": arm, "protocol_sha256": PROTOCOL_SHA256}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="effective_write_cost_v1",
            execution_id=f"{phase}-{seed}-{arm}",
            producer=identity,
            resolved_config={"evaluation": spec["evaluation"]},
        ),
        torch.no_grad(),
    ):
        result, arrays = {**identity, "cells": {}}, {}
        for size in sizes:
            cell, raw, sampled = liu_cell(
                backbone, local, seqs, source, spec, seed, size, generic["competence"]
            )
            result["cells"][str(size)] = cell
            arrays[f"N{size}"] = raw
            write_json_exclusive(
                directory / f"behavior-N{size}.json", json_ready(sampled)
            )
            print(
                json.dumps(
                    {"seed": seed, "arm": arm, "size": size, "core": cell["core_flags"]}
                ),
                flush=True,
            )
        write_arrays(directory / "raw.npz", flatten_arrays(arrays))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def evaluate():
    from fsrl.infra.provenance import load_json

    from .diagnostic import diagnostic_run
    from .reporting import paired_results

    validate_phase(MODELS)
    source = load_json(SOURCE)
    spec = specification()
    for seed in spec["seeds"]["mandatory"]:
        for arm in spec["seeds"]["conditions"]:
            generic_run(seed, arm, "test", source)
            diagnostic_run(seed, arm, source)
            liu_run(seed, arm, [8], source)
    primary = paired_results()
    write_json_exclusive(RUNS / "primary.json", json_ready(primary))
    if primary["transport_triggered"]:
        for seed in spec["seeds"]["mandatory"]:
            for arm in spec["seeds"]["conditions"]:
                liu_run(seed, arm, [6, 10], source)
    return primary
