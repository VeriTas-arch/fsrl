"""Non-Liu mask, additive reconstruction and CUDA compiler qualification."""

import copy

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.memory_structure.inputs import shared_inputs
from fsrl.experiments.memory_structure.interventions import remove_relation
from fsrl.experiments.memory_structure.model import forward_batch, make_model
from fsrl.experiments.training_strategy.batches import EpisodeBatch, input_arrays
from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import compile_module
from fsrl.tasks.protocol import ordered_pairs

from .computation import (
    compare,
    input_checks,
    local_margin,
    relation_mask,
    remove_all_weak,
    remove_global,
)
from .protocol import QUALIFICATION, parent_spec


def synthetic():
    """Nonorthogonal cues, both orientations, four repeats, and a no-weak subject."""
    rng = np.random.default_rng(910005)
    codes = rng.normal(size=(3, 4, 15)).astype(np.float32)
    relations = [(0, 1), (1, 2), (2, 3)]
    trials = [r if block % 2 == 0 else r[::-1] for block in range(4) for r in relations]
    pairs = np.broadcast_to(np.asarray(trials)[:, None], (12, 3, 2)).copy()
    retention = np.asarray([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    z = np.tile(retention.T, (4, 1)).astype(float)
    sign = np.where(pairs[..., 0] < pairs[..., 1], 1.0, -1.0) / 3
    weak = (sign * (z + (1 - z) * 0.35)).astype(np.float32)
    support = input_arrays(codes, pairs, sign * z, np.linspace(0, 2 / 3, 12), 4)
    queries = np.asarray(ordered_pairs(4))
    qpairs = np.broadcast_to(queries[:, None], (12, 3, 2))
    query = input_arrays(codes, qpairs, np.zeros((12, 3)), np.full(12, 2 / 3), 2)
    query = query.transpose(1, 0, 2, 3).reshape(2, 36, -1).copy()
    targets = np.repeat((queries[:, 0] < queries[:, 1]).astype(np.int64), 3)
    return EpisodeBatch(
        {
            "support_inputs": shared_inputs(support, weak),
            "local_evidence": weak,
            "query_inputs": shared_inputs(query),
            "targets": targets,
            "support_pairs": pairs,
            "query_pairs": queries,
            "trial_retention": z,
            "retention": retention,
        }
    ), relations


def independent_local(local, cpu, omitted):
    arrays = cpu.arrays
    cue_size = local.cue_size

    def key(cues):
        left, right = np.split(cues, 2, axis=-1)
        outer = left[..., :, None] * right[..., None, :]
        values = (outer - np.swapaxes(outer, -1, -2)).reshape(*cues.shape[:-1], -1)
        return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)

    support_keys = key(arrays["support_inputs"][:, 0, :, : 2 * cue_size].astype(float))
    evidence = np.where(omitted, 0.0, arrays["local_evidence"])
    state = (support_keys * evidence[..., None]).sum(0)
    queries = arrays["query_inputs"][0, :, : 2 * cue_size].reshape(
        -1, state.shape[0], 2 * cue_size
    )
    raw = (key(queries.astype(float)) * state[None]).sum(-1).T
    gain = np.logaddexp(0, float(local.raw_gain.detach().cpu().item()))
    return gain * raw


def qualify(device="cpu", hidden=8, compiled=False):
    spec = copy.deepcopy(parent_spec())
    spec["architecture"]["hidden_size"] = hidden
    net, local = make_model(spec, 910005, "dual", device)
    cpu, relations = synthetic()
    sequence = RecurrentSequence(net)
    checks = {"input_mask": input_checks(cpu, remove_all_weak(cpu), relations)}
    with torch.no_grad():
        intact = forward_batch(net, local, sequence, cpu.to(device))
        assert intact.local_state is not None
        for i, relation in enumerate(relations):
            mask = relation_mask(cpu, relation)
            expected = independent_local(local, cpu, mask)
            correction = local_margin(local, cpu, mask)
            checks[f"independent_local_{i}"] = compare(expected, correction)
            both = forward_batch(
                net, local, sequence, remove_relation(cpu, relation).to(device)
            )
            global_only = forward_batch(
                net, local, sequence, remove_global(cpu, mask).to(device)
            )
            assert global_only.local_state is not None
            reconstructed = margin_bundle(both.logits, 3)["logits"] - correction
            checks[f"global_reconstruction_{i}"] = compare(
                reconstructed, margin_bundle(global_only.global_logits, 3)["logits"]
            )
            checks[f"local_preserved_{i}"] = compare(
                intact.local_state.cpu().numpy(), global_only.local_state.cpu().numpy()
            )
        joint = forward_batch(net, local, sequence, remove_all_weak(cpu).to(device))
        checks["no_weak_noop"] = compare(
            margin_bundle(intact.logits, 3)["logits"][0],
            margin_bundle(joint.logits, 3)["logits"][0],
        )
        if compiled:
            compiled_sequence = compile_module(RecurrentSequence(net), PROFILE)
            for name, batch, expected in (
                ("intact", cpu, intact),
                ("joint", remove_all_weak(cpu), joint),
            ):
                actual = forward_batch(net, local, compiled_sequence, batch.to(device))
                for field in ("logits", "global_logits", "weights", "local_state"):
                    checks[f"compiled_{name}_{field}"] = compare(
                        getattr(actual, field).cpu().numpy(),
                        getattr(expected, field).cpu().numpy(),
                    )
    return checks


def run():
    from .locks import sources

    runtime = configure_execution()
    checks = qualify("cuda", 200, True)
    result = {
        "passed": all(row["passed"] for row in checks.values()),
        "runtime": runtime,
        "checks": checks,
        "sources": sources(),
        "liu_evaluated": False,
        "trained_models_used": False,
        "seed": 910005,
    }
    write_json_exclusive(QUALIFICATION, result)
    return {"passed": result["passed"], "checks": len(checks)}
