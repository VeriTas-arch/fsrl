"""Generic collection with a post-encoding duplicate observation."""

import numpy as np

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.finite_state.evaluation import storage_arrays
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.write_cost.inputs import load_input

from .inputs import observed


def generic_arrays(backbone, local, seqs, source, split, arm, spec, *, keep_hint=False):
    collected = {}
    for name, ref in sorted(source["inputs"].items()):
        if not name.startswith(split + "-"):
            continue
        cpu = observed(load_input(ref), arm, spec, keep_hint=keep_hint)
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
