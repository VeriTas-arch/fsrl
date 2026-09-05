"""Batched frozen P/L readouts and all-source removals for the paired study."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.evaluation.contracts import FastWeightIntervention
from fsrl.evaluation.local_access import (
    build_access_trace,
    build_fast_weight_loo,
    relation_reliability,
)
from fsrl.evaluation.sampling import retained_relation_mask
from fsrl.experiments.local_fidelity.evidence_access_pilot import blockwise_derangements
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.tasks.protocol import ordered_pairs

from .batches import input_arrays


def readout_bundle(
    evaluator,
    local,
    fast_weights,
    local_state,
    *,
    local_off=False,
    global_off=False,
    shuffled_indices=None,
) -> dict[str, np.ndarray]:
    """Use the same compiled recurrent sequence and exact two-logit correction."""

    subjects = evaluator.config.bs
    pairs = np.asarray(ordered_pairs(evaluator.protocol.n_items), dtype=np.int64)
    count = len(pairs)
    schedule = np.broadcast_to(pairs[:, None], (count, subjects, 2))
    cpu_inputs = input_arrays(
        evaluator.cue_codes,
        schedule,
        np.zeros((count, subjects), dtype=np.float32),
        np.full(count, evaluator.test_time_value, dtype=np.float32),
        2,
    )
    inputs = torch.from_numpy(
        cpu_inputs.transpose(1, 0, 2, 3).reshape(2, count * subjects, -1).copy()
    ).to(evaluator.device)
    weights = torch.zeros_like(fast_weights) if global_off else fast_weights
    query_weights = weights.repeat(count, 1, 1)
    pair_cues = inputs[0, :, : 2 * local.cue_size]
    if shuffled_indices is not None:
        source_cues = pair_cues.reshape(count, subjects, -1).transpose(0, 1)
        index = torch.as_tensor(shuffled_indices, device=evaluator.device)
        pair_cues = (
            source_cues[torch.arange(subjects, device=evaluator.device)[:, None], index]
            .transpose(0, 1)
            .reshape(count * subjects, -1)
        )
    assert evaluator.sequence_runner is not None
    with torch.no_grad():
        logits, _, _, _, _, _ = evaluator.sequence_runner(
            inputs,
            evaluator.net.initial_hidden(count * subjects),
            evaluator.net.initial_eligibility(count * subjects),
            query_weights,
            False,
        )
        gain_override = logits.new_zeros(count * subjects, 1) if local_off else None
        combined, raw, gain, correction = local(
            logits, local_state.repeat(count, 1), pair_cues, gain_override=gain_override
        )
        values = {
            "logits": combined[:, 1] - combined[:, 0],
            "global_logits": logits[:, 1] - logits[:, 0],
            "raw_local_margins": raw[:, 0],
            "applied_local_margins": correction[:, 0],
            "local_gains": gain[:, 0],
        }
        return {
            name: value.reshape(count, subjects).T.cpu().numpy().astype(np.float64)
            for name, value in values.items()
        }


def rollout_liu(evaluator, local, specification: dict) -> dict:
    settings = specification["evaluation"]["liu"]
    relations = evaluator.protocol.support_pairs_higher_lower
    subjects = evaluator.config.bs
    weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    trace = build_access_trace(evaluator, local, dual_access=True)
    routing = blockwise_derangements(
        subjects,
        evaluator.protocol.support_blocks,
        len(relations),
        settings["evidence_shuffle_seed"],
    )
    shuffled_trace = build_access_trace(
        evaluator, local, dual_access=True, route_maps=routing
    )
    queries = shuffled_pair_indices(
        subjects, evaluator.protocol.n_items, settings["query_shuffle_seed"]
    )
    bundles = {
        "intact": readout_bundle(evaluator, local, weights, trace.state),
        "local_off": readout_bundle(
            evaluator, local, weights, trace.state, local_off=True
        ),
        "P_off": readout_bundle(
            evaluator, local, weights, trace.state, global_off=True
        ),
        "query_shuffle": readout_bundle(
            evaluator, local, weights, trace.state, shuffled_indices=queries
        ),
        "evidence_shuffle": readout_bundle(
            evaluator, local, weights, shuffled_trace.state
        ),
    }
    np.testing.assert_array_equal(
        bundles["local_off"]["logits"], bundles["intact"]["global_logits"]
    )
    loo_weights = build_fast_weight_loo(evaluator, relations)
    loo = {"global": [], "local": [], "combined": []}
    for index, relation in enumerate(relations):
        removed = build_access_trace(
            evaluator, local, dual_access=True, zero_relations=frozenset((relation,))
        )
        for name, kwargs in (
            ("global", {"local_off": True}),
            ("local", {"global_off": True}),
            ("combined", {}),
        ):
            bundle = readout_bundle(
                evaluator, local, loo_weights[index], removed.state, **kwargs
            )
            loo[name].append(bundle["logits"])
        print(f"Liu source removal {index + 1}/{len(relations)} complete", flush=True)
    return {
        "bundles": bundles,
        "loo": {name: np.stack(values) for name, values in loo.items()},
        "retention": retained_relation_mask(evaluator, relations).T,
        "probabilities": np.asarray(
            [
                [
                    relation_reliability(evaluator, subject, *relation)
                    for relation in relations
                ]
                for subject in range(subjects)
            ]
        ),
        "cue_codes": evaluator.cue_codes,
        "support_pairs": np.asarray(
            [
                [(trial.left_item, trial.right_item) for trial in schedule]
                for schedule in evaluator.support_schedules
            ],
            dtype=np.int64,
        ),
        "observed_signed_evidence": np.asarray(
            [
                [trial.signed_magnitude for trial in schedule]
                for schedule in evaluator.support_schedules
            ],
            dtype=np.float64,
        ),
        "natural_local_evidence": trace.natural_scalars,
        "shuffled_local_evidence": shuffled_trace.applied_scalars,
        "evidence_routing": routing,
        "query_routing": queries,
        "fast_weights": weights,
    }
