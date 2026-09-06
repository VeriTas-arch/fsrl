"""Post-lock paired routing tests and prospectively conditional transport."""

import gc

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import remove_relation
from fsrl.experiments.memory_structure.measurement import (
    competence,
    core_flags,
    removal_effects,
    summarize_behavior,
)
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    make_model,
    read_queries,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.estimands import estimate, query_endpoints
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    summarize_endpoints,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .inputs import route_batch
from .interventions import shuffle_evidence
from .locks import ARTIFACT_LOCK, SOURCE_LOCK, validate_artifacts
from .measurement import internal_preferences
from .protocol import PROTOCOL_SHA256, RUN_ROOT, run_directory, specification


def load_input(name: str, condition: str) -> EpisodeBatch:
    record = load_json(SOURCE_LOCK)["inputs"][name]
    path = verify_reference(record["file"])
    with np.load(path, allow_pickle=False) as data:
        batch = EpisodeBatch({key: data[key] for key in data.files})
    if batch.fingerprint() != record["fingerprint"]:
        raise RuntimeError("evaluation input fingerprint differs")
    return route_batch(batch, condition)


def margin_bundle(logits, subjects: int) -> dict:
    return {
        "logits": (logits[:, 1] - logits[:, 0])
        .reshape(-1, subjects)
        .T.cpu()
        .numpy()
        .astype(np.float64)
    }


def generic_evaluation(
    backbone, local, sequence, spec: dict, seed: int, condition: str
) -> tuple:
    collected = {"margins": [], "signs": [], "learned": []}
    for name in sorted(load_json(SOURCE_LOCK)["inputs"]):
        if not name.startswith("generic-"):
            continue
        cpu = load_input(name, condition)
        batch = cpu.to("cuda")
        result = forward_batch(backbone, local, sequence, batch)
        subjects = batch.support_inputs.shape[2]
        collected["margins"].append(margin_bundle(result.logits, subjects)["logits"])
        collected["signs"].append(
            (2 * cpu.arrays["targets"] - 1).reshape(-1, subjects).T
        )
        collected["learned"].append(cpu.arrays["learned"])
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    endpoints = query_endpoints(
        arrays["margins"][..., None],
        arrays["signs"][..., None],
        {"learned": arrays["learned"], "nonlearned": ~arrays["learned"]},
        temperature=1.0,
    )
    summaries = summarize_endpoints(
        endpoints, spec["statistics"]["seed_offset"] + seed, spec["statistics"]
    )
    return {"summaries": summaries, "competence": competence(summaries)}, {
        **arrays,
        "endpoints": endpoints,
    }


def liu_rollouts(backbone, local, sequence, cpu, protocol, spec: dict) -> dict:
    batch = cpu.to("cuda")
    subjects = batch.support_inputs.shape[2]
    result = forward_batch(backbone, local, sequence, batch)
    p_off, _ = read_queries(
        backbone,
        local,
        sequence,
        batch,
        torch.zeros_like(result.weights),
        result.local_state,
    )
    bundles = {
        "intact": margin_bundle(result.logits, subjects),
        "local_off": margin_bundle(result.global_logits, subjects),
        "P_off": margin_bundle(p_off, subjects),
    }
    settings = spec["evaluation"]["liu"]
    shuffled, route = shuffle_evidence(
        cpu, protocol.support_blocks, settings["evidence_shuffle_seed"]
    )
    shuffled_result = forward_batch(backbone, local, sequence, shuffled.to("cuda"))
    bundles["evidence_shuffle"] = margin_bundle(shuffled_result.logits, subjects)
    query_route = shuffled_pair_indices(
        subjects, protocol.n_items, settings["query_shuffle_seed"]
    )
    if local is not None:
        logits, _ = read_queries(
            backbone,
            local,
            sequence,
            batch,
            result.weights,
            result.local_state,
            torch.as_tensor(query_route, device="cuda"),
        )
        bundles["query_shuffle"] = margin_bundle(logits, subjects)
    removed = []
    for relation in protocol.support_pairs_higher_lower:
        changed = forward_batch(
            backbone, local, sequence, remove_relation(cpu, relation).to("cuda")
        )
        removed.append(margin_bundle(changed.logits, subjects)["logits"])
    return {
        "bundles": bundles,
        "removed": np.stack(removed),
        "evidence_route": route,
        "query_route": query_route,
        "input_fingerprint": cpu.fingerprint(),
    }


def cell_analysis(
    backbone,
    local,
    sequence,
    spec: dict,
    seed: int,
    size: int,
    generic_passed: bool,
    condition: str,
):
    cpu = load_input(f"liu-{size}", condition)
    protocol = size_protocol(spec, size)
    raw = liu_rollouts(backbone, local, sequence, cpu, protocol, spec)
    endpoints = liu_endpoints(
        raw["bundles"],
        cpu.arrays["retention"],
        protocol,
        spec["evaluation"]["liu"]["temperature"],
    )
    bootstrap_seed = spec["statistics"]["seed_offset"] + seed
    summaries = {
        name: summarize_endpoints(values, bootstrap_seed, spec["statistics"])
        for name, values in endpoints.items()
    }
    behavior, behavior_arrays, sampled = summarize_behavior(
        raw["bundles"]["intact"], protocol, spec, seed
    )
    effects = removal_effects(
        raw["bundles"]["intact"]["logits"], raw["removed"], protocol
    )
    for intervention in raw["bundles"]:
        if intervention == "intact":
            continue
        for group in ("learned", "nonlearned", "omitted"):
            effects[f"intact_minus_{intervention}_{group}"] = (
                endpoints["intact"]["probability"][group]
                - endpoints[intervention]["probability"][group]
            )
    internal, internal_arrays = internal_preferences(
        raw["bundles"]["intact"]["logits"], protocol, spec, seed
    )
    flags = core_flags(summaries["intact"], behavior, generic_passed)
    result = {
        "n_items": size,
        "summaries": summaries,
        "behavior": behavior,
        "internal": internal,
        "effects": {
            key: estimate(values, seed=bootstrap_seed, statistics=spec["statistics"])
            for key, values in effects.items()
        },
        "competence": competence(summaries["intact"]),
        "core_flags": flags,
        "core_passed": all(flags.values()),
        "query_shuffle_applicable": local is not None,
        "input_fingerprint": raw["input_fingerprint"],
    }
    return (
        result,
        {
            **raw,
            "endpoints": endpoints,
            "behavior": behavior_arrays,
            "internal": internal_arrays,
            "effects": effects,
        },
        sampled,
    )


def evaluate_one(seed: int, condition: str, lock: dict, spec: dict, *, transport=False):
    phase = "transport" if transport else "evaluation"
    directory = RUN_ROOT / phase / str(seed) / condition
    if directory.exists():
        run = load_json(directory / "run.json")
        if (
            run["lifecycle_state"] != "complete"
            or not validate_run_manifest(directory / "run.json")["passed"]
        ):
            raise RuntimeError("existing evaluation incomplete or modified")
        return load_json(directory / "result.json")
    runtime = configure_execution()
    if json_ready(runtime) != load_json(SOURCE_LOCK)["runtime"]:
        raise RuntimeError("evaluation runtime differs from qualification")
    backbone, local = make_model(spec, seed, "dual", "cuda")
    training = run_directory(seed, condition)
    backbone.load_state_dict(
        torch.load(training / "net.pth", weights_only=True, map_location="cuda")
    )
    if local is not None:
        local.load_state_dict(
            torch.load(training / "local.pth", weights_only=True, map_location="cuda")
        )
        local.requires_grad_(False).eval()
    metadata = lock["runs"][f"{seed}/{condition}"]["metadata"]
    if tensor_hashes(backbone) != metadata["final_backbone"]:
        raise RuntimeError("loaded backbone differs from artifact lock")
    if local is not None and tensor_hashes(local) != metadata["final_local"]:
        raise RuntimeError("loaded local gain differs from artifact lock")
    backbone.requires_grad_(False).eval()
    sequence = compile_module(RecurrentSequence(backbone), PROFILE)
    identity = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "source_commit": lock["source_commit"],
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="weak_evidence_routing_v1",
            execution_id=f"{phase}-{seed}-{condition}",
            producer=identity,
            resolved_config={"runtime": runtime, "evaluation": spec["evaluation"]},
        ),
        torch.no_grad(),
    ):
        generic, raw_generic = generic_evaluation(
            backbone, local, sequence, spec, seed, condition
        )
        result = {**identity, "generic": generic, "cells": {}, "training": metadata}
        arrays = {"generic": raw_generic}
        for size in spec["evaluation"][
            "transport_item_counts" if transport else "item_counts"
        ]:
            cell, raw, behavior = cell_analysis(
                backbone,
                local,
                sequence,
                spec,
                seed,
                size,
                generic["competence"],
                condition,
            )
            result["cells"][str(size)] = cell
            arrays[f"N{size}"] = raw
            write_json_exclusive(
                directory / f"behavior-N{size}.json", json_ready(behavior)
            )
            print(
                {
                    "seed": seed,
                    "condition": condition,
                    "n_items": size,
                    "core_flags": cell["core_flags"],
                },
                flush=True,
            )
        write_arrays(directory / "raw.npz", flatten_arrays(arrays))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def evaluate_all():
    lock = validate_artifacts()
    spec = specification()
    completed = []
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["conditions"]:
            evaluate_one(seed, condition, lock, spec)
            completed.append(f"{seed}/{condition}")
            gc.collect()
            torch.cuda.empty_cache()
    from .reporting import primary_result

    primary = primary_result(lock, spec)
    decision_path = RUN_ROOT / "primary.json"
    if decision_path.exists():
        if load_json(decision_path) != json_ready(primary):
            raise RuntimeError("primary decision changed")
    else:
        write_json_exclusive(decision_path, json_ready(primary))
    if primary["transport_triggered"]:
        for seed in spec["seeds"]["mandatory"]:
            for condition in spec["seeds"]["conditions"]:
                evaluate_one(seed, condition, lock, spec, transport=True)
                gc.collect()
                torch.cuda.empty_cache()
    return {
        "completed": completed,
        "transport_triggered": primary["transport_triggered"],
        "pair_outcomes": primary["pair_outcomes"],
    }
