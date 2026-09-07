"""Matched latent worlds; only item cues, time and realized q enter the learner."""

from itertools import pairwise

import numpy as np

from fsrl.evaluation.sampling import deterministic_cue_codes
from fsrl.experiments.training_strategy.batches import input_arrays

QUERIES = np.asarray([(i, j) for i in range(8) for j in range(8) if i != j])
GROUPS = (
    ((0, 1, 4, 5), (2, 3, 6, 7)),
    ((1, 3, 5, 7), (0, 2, 4, 6)),
)


def initial_world(seed, count):
    """Shared scores/cues/noise; role labels never appear in numerical inputs."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-0.35, 0.35, (count, 8))
    # A/D start near the decision boundary, independently of their group assignment.
    scores[:, 0] = rng.uniform(-0.04, 0.04, count)
    scores[:, 3] = -scores[:, 0]
    codes = np.stack(
        [
            deterministic_cue_codes(1, 8, 15, seed=seed * 1000 + i)[0]
            for i in range(count)
        ]
    )
    return {
        "scores": scores,
        "codes": codes,
        "noise": rng.standard_normal((32, count)),
        "order": np.argsort(rng.random((count, 24)), axis=1),
        "orientation": rng.choice([-1, 1], (32, count)),
    }


def edges_for(history, topology):
    groups = GROUPS[history]
    edges = []
    for group in groups:
        if topology == "chain":
            edges.extend(pairwise(group))
        elif topology == "star":
            anchor = 1 if 1 in group else 2
            edges.extend((anchor, item) for item in group if item != anchor)
        else:
            raise ValueError("unknown topology")
    return np.asarray(edges, dtype=np.int64)


def physical_inputs(codes, pairs, q):
    times = np.arange(len(pairs)) / 31 * (2 / 3)
    original = input_arrays(codes, pairs, q, times, 4)
    support = np.concatenate((original, np.zeros((*original.shape[:-1], 1))), -1)
    support[:, 0, :, 37] = q
    subjects = len(codes)
    query_pairs = np.broadcast_to(QUERIES[:, None], (56, subjects, 2))
    original_query = input_arrays(
        codes, query_pairs, np.zeros((56, subjects)), np.full(56, 2 / 3), 2
    )
    query = np.concatenate(
        (original_query, np.zeros((*original_query.shape[:-1], 1))), -1
    )
    return support.astype(np.float32), query.astype(np.float32)


def make_panel(seed, count, history, topology, condition, sigma=0.05, shift=0.4):
    """24 old-world observations followed by eight matched challenge events."""
    world = initial_world(seed, count)
    edges = edges_for(history, topology)
    old_pairs = np.concatenate((np.tile(edges, (3, 1)), np.tile([[1, 2]], (6, 1))))
    old_pairs = old_pairs[world["order"]].transpose(1, 0, 2)
    group_b, group_c = GROUPS[history]
    cross = [(i, j) for i in group_b for j in group_c if {i, j} != {0, 3}]
    cross = [pair for pair in cross if pair != (1, 2)]
    post = np.asarray(
        [(1, 2), edges[0], cross[0], edges[2], cross[-1], edges[3], cross[2], edges[5]]
    )
    pairs = np.concatenate((old_pairs, np.broadcast_to(post[:, None], (8, count, 2))))
    reverse = world["orientation"] < 0
    pairs[reverse] = pairs[reverse, ::-1]
    old = world["scores"]
    revised = old.copy()
    revised[:, group_b] += shift / 2
    revised[:, group_c] -= shift / 2
    scores = np.broadcast_to(old, (32, count, 8)).copy()
    if condition == "revision":
        scores[24:] = revised
    elif condition not in ("stable", "outlier"):
        raise ValueError("unknown condition")
    subject = np.arange(count)[None]
    q = (
        scores[np.arange(32)[:, None], subject, pairs[:, :, 0]]
        - scores[np.arange(32)[:, None], subject, pairs[:, :, 1]]
    )
    if condition == "outlier":
        q[24] = (
            revised[np.arange(count), pairs[24, :, 0]]
            - revised[np.arange(count), pairs[24, :, 1]]
        )
    q = (q + sigma * world["noise"]).astype(np.float32)
    support, query = physical_inputs(world["codes"], pairs, q)
    final = scores[-1]
    old_difference = old[:, QUERIES[:, 0]] - old[:, QUERIES[:, 1]]
    new_difference = final[:, QUERIES[:, 0]] - final[:, QUERIES[:, 1]]
    structural_change = (revised - old)[:, QUERIES[:, 0]] - (revised - old)[
        :, QUERIES[:, 1]
    ]
    post_seen = np.zeros((count, 56), dtype=bool)
    for k, (i, j) in enumerate(QUERIES):
        post_seen[:, k] = np.any(
            ((pairs[24:, :, 0] == i) & (pairs[24:, :, 1] == j))
            | ((pairs[24:, :, 0] == j) & (pairs[24:, :, 1] == i)),
            axis=0,
        )
    return {
        "support": support,
        "query": query,
        "pairs": pairs,
        "q": q,
        "codes": world["codes"],
        "old_scores": old,
        "final_scores": final,
        "old_sign": np.sign(old_difference),
        "final_sign": np.sign(new_difference),
        "changed_remote": (abs(structural_change) > 1e-10) & ~post_seen,
        "unchanged": abs(structural_change) < 1e-10,
        "post_seen": post_seen,
        "query_pairs": QUERIES,
    }
