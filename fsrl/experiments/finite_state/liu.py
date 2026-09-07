"""Full-input Liu policies, fixed-state interventions and encoding replays."""

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.evidence_routing.measurement import (
    internal_preferences,
    selection,
)
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.memory_structure.measurement import (
    competence,
    core_flags,
    removal_effects,
    summarize_behavior,
)
from fsrl.experiments.memory_structure.model import read_queries
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    summarize_endpoints,
)
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import loaded_model, storage_arrays
from .inputs import rounding_seed
from .model import rollout
from .protocol import PROTOCOL_SHA256, RUNS, specification


def primary_rollouts(backbone, local, seqs, states, cpu, protocol, spec):
    batch, draw = cpu.to("cuda"), rounding_seed(4)
    result, cost, writes = rollout(backbone, local, seqs, batch, states, draw)
    subjects = len(cost)
    p_off, _ = read_queries(
        backbone,
        local,
        seqs[1],
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
    changed, _, _ = rollout(backbone, local, seqs, shuffled.to("cuda"), states, draw)
    bundles["evidence_shuffle"] = margin_bundle(changed.logits, subjects)
    query_route = shuffled_pair_indices(
        subjects, protocol.n_items, settings["query_shuffle_seed"]
    )
    logits, _ = read_queries(
        backbone,
        local,
        seqs[1],
        batch,
        result.weights,
        result.local_state,
        torch.as_tensor(query_route, device="cuda"),
    )
    bundles["query_shuffle"] = margin_bundle(logits, subjects)
    removed, removed_global = [], []
    for relation in protocol.support_pairs_higher_lower:
        changed, _, _ = rollout(
            backbone,
            local,
            seqs,
            remove_relation(cpu, relation).to("cuda"),
            states,
            draw,
        )
        removed.append(margin_bundle(changed.logits, subjects)["logits"])
        removed_global.append(margin_bundle(changed.global_logits, subjects)["logits"])
    return {
        "bundles": bundles,
        "removed": np.stack(removed),
        "removed_global": np.stack(removed_global),
        "evidence_route": route,
        "query_route": query_route,
        "cost": cost.cpu().numpy(),
        "total_write": writes.sum(0).cpu().numpy(),
        "storage": storage_arrays(result.weights, backbone.alpha, states),
    }


def primary_analysis(raw, cpu, protocol, spec, seed, generic):
    boot = spec["statistics"]["seed_offset"] + seed
    endpoints = liu_endpoints(
        raw["bundles"],
        cpu.arrays["retention"],
        protocol,
        spec["evaluation"]["liu"]["temperature"],
    )
    summaries = {
        name: summarize_endpoints(values, boot, spec["statistics"])
        for name, values in endpoints.items()
    }
    results, arrays, sampled_records = {}, {}, {}
    for route, bundle in (("full", "intact"), ("global", "local_off")):
        behavior, behavior_arrays, sampled = summarize_behavior(
            raw["bundles"][bundle], protocol, spec, seed
        )
        internal, internal_arrays = internal_preferences(
            raw["bundles"][bundle]["logits"], protocol, spec, seed
        )
        generic_passed = (
            generic["competence"]
            if route == "full"
            else generic["global"]["competence"]
        )
        flags = core_flags(summaries[bundle], behavior, generic_passed)
        results[route] = {
            "behavior": behavior,
            "internal": internal,
            "competence": competence(summaries[bundle]),
            "core_flags": flags,
            "core_passed": all(flags.values()),
        }
        orders, mask = selection(sampled)
        arrays[route] = {
            "behavior": behavior_arrays,
            "internal": internal_arrays,
            "sampled_orders": orders,
            "sampled_mask": mask,
        }
        sampled_records[route] = sampled
    effects = removal_effects(
        raw["bundles"]["intact"]["logits"], raw["removed"], protocol
    )
    for name in raw["bundles"]:
        if name == "intact":
            continue
        for group in ("learned", "nonlearned", "omitted"):
            effects[f"intact_minus_{name}_{group}"] = (
                endpoints["intact"]["probability"][group]
                - endpoints[name]["probability"][group]
            )
    result = {
        "summaries": summaries,
        "routes": results,
        "effects": {
            key: estimate(value, seed=boot, statistics=spec["statistics"])
            for key, value in effects.items()
        },
        "storage": {key: float(value.mean()) for key, value in raw["storage"].items()},
        "mean_cost": float(raw["cost"].mean()),
    }
    return (
        result,
        {"endpoints": endpoints, "routes": arrays, "effects": effects},
        sampled_records,
    )


def replay_analysis(backbone, local, seqs, states, cpu, primary, spec, seed):
    collected = {
        "full": [primary["bundles"]["intact"]["logits"]],
        "global": [primary["bundles"]["local_off"]["logits"]],
    }
    for replay in spec["rounding"]["liu_replays"][1:]:
        result, _, _ = rollout(
            backbone, local, seqs, cpu.to("cuda"), states, rounding_seed(4, replay)
        )
        for route, logits in (
            ("full", result.logits),
            ("global", result.global_logits),
        ):
            collected[route].append(
                margin_bundle(logits, result.weights.shape[0])["logits"]
            )
    arrays, summary = {}, {}
    for route, values in collected.items():
        stack = np.stack(values)
        within = stack.var(axis=0, ddof=1).mean(axis=1)
        between = stack.mean(axis=0).var(axis=0, ddof=1).mean()
        arrays[route] = {"margins": stack, "within_history_variance": within}
        summary[route] = {
            "within_history_variance": estimate(
                within,
                seed=spec["statistics"]["seed_offset"] + seed,
                statistics=spec["statistics"],
            ),
            "between_replay_mean_variance": float(between),
            "between_variance_minus_within_over_replays": float(
                between - within.mean() / len(values)
            ),
            "primary_replay": 0,
            "replays": len(values),
        }
    return summary, arrays


def liu_run(seed, arm, source):
    directory = RUNS / "liu" / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    generic = completed(RUNS / "generic/test" / str(seed) / arm)
    backbone, local, seqs, states = loaded_model(seed, arm, source)
    cpu = load_input(source["inputs"]["liu-8"])
    protocol = size_protocol(spec, 8)
    identity = {
        "seed": seed,
        "arm": arm,
        "states": states,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="finite_state_memory_v1",
            execution_id=f"liu-{seed}-{arm}",
            producer=identity,
            resolved_config={
                "evaluation": spec["evaluation"]["liu"],
                "rounding": spec["rounding"],
            },
        ),
        torch.no_grad(),
    ):
        raw = primary_rollouts(backbone, local, seqs, states, cpu, protocol, spec)
        summary, analysis, sampled = primary_analysis(
            raw, cpu, protocol, spec, seed, generic
        )
        replay, replay_arrays = replay_analysis(
            backbone, local, seqs, states, cpu, raw, spec, seed
        )
        result = {**identity, **summary, "replay": replay}
        write_arrays(
            directory / "raw.npz",
            flatten_arrays({**raw, **analysis, "replay": replay_arrays}),
        )
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
        print(
            {"seed": seed, "arm": arm, "core": result["routes"]["full"]["core_flags"]},
            flush=True,
        )
    return result
