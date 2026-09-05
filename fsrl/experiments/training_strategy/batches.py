"""Paired generic streams with explicit original retention probabilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from fsrl.core.inputs import RelationalInputLayout
from fsrl.tasks.sparse_ranking import (
    GenericRankingTaskGenerator,
    GraphSignature,
    RankingEpisode,
)


def graph_bucket(graph: GraphSignature, n_items: int = 8) -> int:
    reflection = tuple(sorted((n_items - 1 - j, n_items - 1 - i) for i, j in graph))
    canonical = min(tuple(sorted(graph)), reflection)
    payload = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 16


def sample_episodes(
    generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
    batch_size: int,
    *,
    validation: bool = False,
) -> tuple[RankingEpisode, ...]:
    n_edges = int(rng.integers(generator.min_edges, generator.max_edges + 1))
    episodes = []
    while len(episodes) < batch_size:
        episode = generator.sample(rng, n_edges=n_edges)
        in_validation = graph_bucket(episode.graph_rank_pairs, generator.n_items) == 0
        if in_validation == validation:
            episodes.append(episode)
    return tuple(episodes)


@dataclass(frozen=True)
class TensorBatch:
    support_inputs: torch.Tensor
    local_evidence: torch.Tensor
    query_inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    """CPU arrays retain the input/label provenance; only four tensors enter rollout."""

    arrays: dict[str, np.ndarray]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays.items()):
            array = np.ascontiguousarray(value)
            header = json.dumps([name, array.dtype.str, list(array.shape)])
            digest.update(header.encode("ascii") + b"\0")
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to(self, device: str | torch.device) -> TensorBatch:
        return TensorBatch(
            **{
                name: torch.from_numpy(self.arrays[name]).to(device)
                for name in (
                    "support_inputs",
                    "local_evidence",
                    "query_inputs",
                    "targets",
                )
            }
        )


def input_arrays(
    codes: np.ndarray,
    pairs: np.ndarray,
    evidence: np.ndarray,
    times: np.ndarray,
    steps: int,
) -> np.ndarray:
    trials, subjects = pairs.shape[:2]
    layout = RelationalInputLayout(codes.shape[-1])
    inputs = np.zeros((trials, steps, subjects, layout.input_size), dtype=np.float32)
    inputs[:, :, :, layout.bias_index] = 1.0
    inputs[:, :, :, layout.time_index] = times[:, None, None]
    indices = np.arange(subjects)[None, :]
    inputs[:, 0, :, : layout.cue_size] = codes[indices, pairs[:, :, 0]]
    inputs[:, 0, :, layout.cue_size : layout.pair_cue_width] = codes[
        indices, pairs[:, :, 1]
    ]
    inputs[:, 0, :, layout.evidence_index] = evidence
    inputs[:, 1, :, layout.pair_cue_width] = 1.0
    return inputs


def _support_values(episodes: tuple[RankingEpisode, ...]) -> dict[str, np.ndarray]:
    n_support = len(episodes[0].support_trials)
    shape = (n_support, len(episodes))
    signed = np.empty(shape, dtype=np.float64)
    retained = np.empty(shape, dtype=np.float64)
    probabilities = np.empty(shape, dtype=np.float64)
    pairs = np.empty((*shape, 2), dtype=np.int64)
    for subject, episode in enumerate(episodes):
        for step, trial in enumerate(episode.support_trials):
            signed[step, subject] = trial.signed_magnitude
            retained[step, subject] = trial.encoding_reliability
            pairs[step, subject] = (trial.left_item, trial.right_item)
            distance = round(
                abs(trial.signed_magnitude) * (len(episode.item_codes) - 1)
            )
            probabilities[step, subject] = (
                episode.subject_encoding.relation_reliability(
                    trial.left_item, trial.right_item, distance
                )
            )
    if not np.all((retained == 0.0) | (retained == 1.0)):
        raise ValueError("the joint protocol requires binary stable retention masks")
    return {
        "signed_magnitudes": signed,
        "retention": retained,
        "probabilities": probabilities,
        "support_pairs": pairs,
    }


def prepare_batch(
    episodes: tuple[RankingEpisode, ...], *, support_query_time: float = 2.0 / 3.0
) -> EpisodeBatch:
    values = _support_values(episodes)
    codes = np.stack([episode.item_codes for episode in episodes])
    signed, retained, probabilities = (
        values[name] for name in ("signed_magnitudes", "retention", "probabilities")
    )
    n_support, batch_size = signed.shape
    n_queries = len(episodes[0].query_trials)
    query_pairs = np.asarray(
        [
            [
                (e.query_trials[q].left_item, e.query_trials[q].right_item)
                for e in episodes
            ]
            for q in range(n_queries)
        ],
        dtype=np.int64,
    )
    targets = np.asarray(
        [
            [e.query_trials[q].correct_action for e in episodes]
            for q in range(n_queries)
        ],
        dtype=np.int64,
    ).reshape(-1)
    support = input_arrays(
        codes,
        values["support_pairs"],
        signed * retained,
        np.arange(n_support) / max(1, n_support - 1) * support_query_time,
        4,
    )
    query = input_arrays(
        codes,
        query_pairs,
        np.zeros((n_queries, batch_size)),
        np.full(n_queries, support_query_time),
        2,
    )
    query = np.ascontiguousarray(query.transpose(1, 0, 2, 3)).reshape(
        2, n_queries * batch_size, -1
    )
    return EpisodeBatch(
        {
            **values,
            "support_inputs": support,
            "local_evidence": np.asarray(
                signed * (retained + (1.0 - retained) * probabilities), dtype=np.float32
            ),
            "query_inputs": query,
            "query_pairs": query_pairs,
            "targets": targets,
            "item_codes": codes,
            "orders": np.asarray([e.true_order_high_to_low for e in episodes]),
            "graphs": np.asarray([e.graph_rank_pairs for e in episodes]),
        }
    )
