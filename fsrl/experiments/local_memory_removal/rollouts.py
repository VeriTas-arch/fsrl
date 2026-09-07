"""Full and global remain aliases without L; absent local shuffle is not a control."""

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.finite_state.evaluation import storage_arrays
from fsrl.experiments.finite_state.inputs import rounding_seed
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.memory_structure.model import read_queries
from fsrl.experiments.observation_uncertainty.evaluation import observed
from fsrl.experiments.write_cost.inputs import load_input

from .model import rollout


def generic_arrays(backbone, local, seqs, source, split, arm, spec):
    collected = {}
    for name, ref in sorted(source["inputs"].items()):
        if not name.startswith(split + "-"):
            continue
        cpu = observed(load_input(ref), arm, spec)
        result, cost, writes = rollout(backbone, local, seqs, cpu.to("cuda"), None, 0)
        n = len(cost)
        values = {
            "margins": margin_bundle(result.logits, n)["logits"],
            "global_margins": margin_bundle(result.global_logits, n)["logits"],
            "signs": (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T,
            "learned": cpu.arrays["learned"],
            "cost": cost.cpu().numpy().astype(float),
            "total_write": writes.sum(0).cpu().numpy().astype(float),
            "episode_indices": cpu.arrays["episode_indices"],
            **storage_arrays(result.weights, backbone.alpha, None),
        }
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in collected.items()}


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
    query_route = None
    if local is not None:
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
        **({"query_route": query_route} if query_route is not None else {}),
        "cost": cost.cpu().numpy(),
        "total_write": writes.sum(0).cpu().numpy(),
        "storage": storage_arrays(result.weights, backbone.alpha, states),
    }
