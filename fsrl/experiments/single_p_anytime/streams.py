"""Deterministic max-stream generation followed by prefix truncation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsrl.experiments.clean_single_p.batches import (
    SinglePEpisodeBatch,
    prepare_single_p,
)
from fsrl.experiments.pl_direct_training.batches import prepare_batch
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import graph_bucket
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator, RankingEpisode

SUPPORT_ARRAYS = frozenset(
    {
        "support_inputs",
        "support_times",
        "local_evidence",
        "signed_magnitudes",
        "retention",
        "probabilities",
        "support_pairs",
        "realized_q",
    }
)


@dataclass(frozen=True)
class MaxStream:
    episodes: tuple[RankingEpisode, ...]
    task_fingerprint: str
    clean: SinglePEpisodeBatch
    noisy: SinglePEpisodeBatch


def task_seed(network_seed: int, update: int) -> int:
    return 710000000 + network_seed * 10000 + update


def observation_seed(network_seed: int, update: int) -> int:
    return 810000000 + network_seed * 10000 + update


def make_generator(task: dict, *, support_blocks: int) -> GenericRankingTaskGenerator:
    return make_task_generator({"task": {**task, "support_blocks": support_blocks}})


def sample_fixed_edge_episodes(
    generator: GenericRankingTaskGenerator,
    *,
    seed: int,
    edge_count: int,
    batch_size: int,
    validation: bool,
) -> tuple[RankingEpisode, ...]:
    rng = np.random.default_rng(seed)
    episodes = []
    while len(episodes) < batch_size:
        episode = generator.sample(rng, n_edges=edge_count)
        is_validation = graph_bucket(episode.graph_rank_pairs, generator.n_items) == 0
        if is_validation == validation:
            episodes.append(episode)
    return tuple(episodes)


def generate_max_stream(
    generator: GenericRankingTaskGenerator,
    *,
    network_seed: int,
    update: int,
    edge_count: int,
    batch_size: int,
    validation: bool = False,
) -> MaxStream:
    episodes = sample_fixed_edge_episodes(
        generator,
        seed=task_seed(network_seed, update),
        edge_count=edge_count,
        batch_size=batch_size,
        validation=validation,
    )
    task_fingerprint = prepare_batch(episodes).fingerprint()
    clean = prepare_single_p(
        episodes,
        "clean",
        observation_seed=observation_seed(network_seed, update),
    )
    noisy = prepare_single_p(
        episodes,
        "noisy",
        observation_seed=observation_seed(network_seed, update),
    )
    return MaxStream(episodes, task_fingerprint, clean, noisy)


def prefix_batch(
    batch: SinglePEpisodeBatch, *, edge_count: int, blocks: int
) -> SinglePEpisodeBatch:
    support_trials = edge_count * blocks
    available = batch.arrays["support_inputs"].shape[0]
    if support_trials > available or blocks < 1:
        raise ValueError("requested support prefix is unavailable")
    arrays = {
        name: value[:support_trials].copy() if name in SUPPORT_ARRAYS else value.copy()
        for name, value in batch.arrays.items()
    }
    # Time is out-of-band and unused, but preserve the inherited within-prefix scale.
    arrays["support_times"] = np.broadcast_to(
        np.linspace(0.0, 2.0 / 3.0, support_trials, dtype=np.float32)[
            :, None, None, None
        ],
        arrays["support_times"].shape,
    ).copy()
    return SinglePEpisodeBatch(arrays)


def learned_mask(batch: SinglePEpisodeBatch) -> np.ndarray:
    query_pairs = batch.arrays["query_pairs"]
    graphs = batch.arrays["graphs"]
    orders = batch.arrays["orders"]
    n_queries, subjects = query_pairs.shape[:2]
    mask = np.empty((n_queries, subjects), dtype=bool)
    for subject in range(subjects):
        order = orders[subject]
        graph_items = {
            tuple(sorted((int(order[first]), int(order[second]))))
            for first, second in graphs[subject]
        }
        for query in range(n_queries):
            mask[query, subject] = (
                tuple(sorted(map(int, query_pairs[query, subject]))) in graph_items
            )
    return mask


__all__ = [
    "SUPPORT_ARRAYS",
    "MaxStream",
    "generate_max_stream",
    "learned_mask",
    "make_generator",
    "observation_seed",
    "prefix_batch",
    "sample_fixed_edge_episodes",
    "task_seed",
]
