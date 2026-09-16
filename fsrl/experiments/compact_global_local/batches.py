"""Compact two-step inputs over the maintained generic ranking task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.tasks.sparse_ranking import RankingEpisode

from .model import CompactModelConfig


@dataclass(frozen=True)
class CompactTensorBatch:
    support_inputs: torch.Tensor
    local_evidence: torch.Tensor
    query_inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class CompactEpisodeBatch:
    arrays: dict[str, np.ndarray]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays.items()):
            array = np.ascontiguousarray(value)
            header = json.dumps([name, array.dtype.str, list(array.shape)])
            digest.update(header.encode("ascii") + b"\0")
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to(self, device: str | torch.device) -> CompactTensorBatch:
        return CompactTensorBatch(
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


def compact_input_arrays(
    codes: np.ndarray,
    pairs: np.ndarray,
    evidence: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    if steps < 2:
        raise ValueError("compact trials require at least two steps")
    trials, subjects = pairs.shape[:2]
    config = CompactModelConfig(cue_size=codes.shape[-1])
    inputs = np.zeros((trials, steps, subjects, config.input_size), dtype=np.float32)
    indices = np.arange(subjects)[None, :]
    inputs[:, 0, :, : config.cue_size] = codes[indices, pairs[:, :, 0]]
    inputs[:, 0, :, config.cue_size : config.pair_cue_width] = codes[
        indices, pairs[:, :, 1]
    ]
    inputs[:, 0, :, config.evidence_index] = evidence
    inputs[:, 1, :, config.response_index] = 1.0
    return inputs


def _support_values(episodes: tuple[RankingEpisode, ...]) -> dict[str, np.ndarray]:
    n_support = len(episodes[0].support_trials)
    shape = (n_support, len(episodes))
    signed = np.empty(shape, dtype=np.float64)
    retained = np.empty(shape, dtype=np.float64)
    probabilities = np.empty(shape, dtype=np.float64)
    pairs = np.empty((*shape, 2), dtype=np.int64)
    for subject, episode in enumerate(episodes):
        if len(episode.support_trials) != n_support:
            raise ValueError("support lengths differ within a batch")
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
        raise ValueError("compact training requires binary stable retention")
    return {
        "signed_magnitudes": signed,
        "retention": retained,
        "probabilities": probabilities,
        "support_pairs": pairs,
    }


def prepare_batch(episodes: tuple[RankingEpisode, ...]) -> CompactEpisodeBatch:
    values = _support_values(episodes)
    codes = np.stack([episode.item_codes for episode in episodes])
    signed = values["signed_magnitudes"]
    retained = values["retention"]
    probabilities = values["probabilities"]
    n_queries = len(episodes[0].query_trials)
    batch_size = len(episodes)
    query_pairs = np.asarray(
        [
            [
                (
                    episode.query_trials[index].left_item,
                    episode.query_trials[index].right_item,
                )
                for episode in episodes
            ]
            for index in range(n_queries)
        ],
        dtype=np.int64,
    )
    targets = np.asarray(
        [
            [episode.query_trials[index].correct_action for episode in episodes]
            for index in range(n_queries)
        ],
        dtype=np.int64,
    ).reshape(-1)
    support = compact_input_arrays(
        codes,
        values["support_pairs"],
        signed * retained,
        steps=3,
    )
    query = compact_input_arrays(
        codes,
        query_pairs,
        np.zeros((n_queries, batch_size), dtype=np.float32),
        steps=2,
    )
    query = np.ascontiguousarray(query.transpose(1, 0, 2, 3)).reshape(
        2, n_queries * batch_size, -1
    )
    return CompactEpisodeBatch(
        {
            **values,
            "support_inputs": support,
            "local_evidence": np.asarray(
                signed * (retained + (1.0 - retained) * probabilities),
                dtype=np.float32,
            ),
            "query_inputs": query,
            "query_pairs": query_pairs,
            "targets": targets,
            "item_codes": codes,
            "orders": np.asarray(
                [episode.true_order_high_to_low for episode in episodes]
            ),
            "graphs": np.asarray([episode.graph_rank_pairs for episode in episodes]),
        }
    )


__all__ = [
    "CompactEpisodeBatch",
    "CompactTensorBatch",
    "compact_input_arrays",
    "prepare_batch",
    "sample_episodes",
]
