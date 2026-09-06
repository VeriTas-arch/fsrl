"""Identical union-of-evidence observations for both recurrent candidates."""

import numpy as np

from fsrl.core.config import TrainConfig
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.evaluation.local_access import relation_reliability
from fsrl.evaluation.sampling import retained_relation_mask
from fsrl.experiments.training_strategy.batches import (
    EpisodeBatch,
    input_arrays,
    prepare_batch,
)
from fsrl.experiments.transport.item_count import protocol_for_size
from fsrl.infra.provenance import load_json
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator
from fsrl.training.backbone import registered_excluded_signatures


def generator(spec: dict) -> GenericRankingTaskGenerator:
    task = spec["task"]
    return GenericRankingTaskGenerator(
        cue_size=task["cue_size"],
        min_edges=task["min_edges"],
        max_edges=task["max_edges"],
        support_blocks=task["support_blocks"],
        excluded_signatures=registered_excluded_signatures(),
        subject_encoding_mode=task["subject_encoding_mode"],
    )


def shared_inputs(inputs: np.ndarray, weak: np.ndarray | None = None) -> np.ndarray:
    """Append one scalar; the historical 37-channel ABI stays untouched."""
    result = np.zeros((*inputs.shape[:-1], inputs.shape[-1] + 1), dtype=np.float32)
    result[..., :-1] = inputs
    if weak is not None:
        result[:, 0, :, -1] = weak
    return result


def prepare_shared(episodes: tuple) -> EpisodeBatch:
    arrays = prepare_batch(episodes).arrays
    arrays["support_inputs"] = shared_inputs(
        arrays["support_inputs"], arrays["local_evidence"]
    )
    arrays["query_inputs"] = shared_inputs(arrays["query_inputs"])
    return EpisodeBatch(arrays)


def size_protocol(spec: dict, n_items: int):
    parent = load_json(REPO_ROOT / spec["references"]["item_count_protocol"]["path"])
    graph = next(
        graph
        for graph in parent["size_matched_graph_contract"]["graphs"]
        if graph["n_items"] == n_items
    )
    return protocol_for_size(load_registered_protocol("liu_v2"), graph)


def liu_inputs(spec: dict, n_items: int) -> tuple:
    settings = spec["evaluation"]["liu"]
    protocol = size_protocol(spec, n_items)
    subjects = settings["subjects"]
    evaluator = FrozenFastWeightEvaluator(
        None,
        TrainConfig(bs=subjects, cs=spec["task"]["cue_size"]),
        protocol,
        protocol_only=True,
        required_item_count=None,
        **{
            key: settings[key]
            for key in (
                "cue_seed",
                "support_seed",
                "cue_mode",
                "subject_encoding_mode",
                "subject_encoding_seed",
            )
        },
    )
    schedules = evaluator.support_schedules
    pairs = np.asarray(
        [
            [(trial.left_item, trial.right_item) for trial in schedule]
            for schedule in schedules
        ]
    ).transpose(1, 0, 2)
    signed = np.asarray(
        [[trial.signed_magnitude for trial in schedule] for schedule in schedules]
    ).T
    z = np.asarray(evaluator.subject_trial_gains).T
    p = np.asarray(
        [
            [
                relation_reliability(
                    evaluator, subject, trial.higher_item, trial.lower_item
                )
                for trial in schedule
            ]
            for subject, schedule in enumerate(schedules)
        ]
    ).T
    weak = np.asarray(signed * (z + (1 - z) * p), dtype=np.float32)
    times = np.arange(protocol.support_trials) / (protocol.support_trials - 1) * (2 / 3)
    support = input_arrays(evaluator.cue_codes, pairs, signed * z, times, 4)
    queries = np.asarray(ordered_pairs(n_items))
    query_pairs = np.broadcast_to(queries[:, None], (len(queries), subjects, 2))
    query = input_arrays(
        evaluator.cue_codes,
        query_pairs,
        np.zeros((len(queries), subjects)),
        np.full(len(queries), 2 / 3),
        2,
    )
    query = query.transpose(1, 0, 2, 3).reshape(2, len(queries) * subjects, -1).copy()
    rank = np.argsort(protocol.true_order_high_to_low)
    targets = np.repeat(
        (rank[queries[:, 0]] < rank[queries[:, 1]]).astype(np.int64), subjects
    )
    arrays = {
        "support_inputs": shared_inputs(support, weak),
        "local_evidence": weak,
        "query_inputs": shared_inputs(query),
        "targets": targets,
        "support_pairs": pairs,
        "query_pairs": queries,
        "retention": retained_relation_mask(
            evaluator, protocol.support_pairs_higher_lower
        ).T,
        "signed_magnitudes": signed,
        "trial_retention": z,
        "probabilities": p,
        "item_codes": evaluator.cue_codes,
    }
    return protocol, EpisodeBatch(arrays)
