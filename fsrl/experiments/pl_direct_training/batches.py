"""Paired 32-channel task streams with time stored out of band."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.tasks.sparse_ranking import RankingEpisode

from .model import FactorizedPlasticRNNConfig


@dataclass(frozen=True)
class DirectTensorBatch:
    support_inputs: torch.Tensor
    support_times: torch.Tensor
    local_evidence: torch.Tensor
    query_inputs: torch.Tensor
    query_times: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class DirectEpisodeBatch:
    arrays: dict[str, np.ndarray]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays.items()):
            array = np.ascontiguousarray(value)
            header = json.dumps([name, array.dtype.str, list(array.shape)])
            digest.update(header.encode("ascii") + b"\0")
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to(self, device: str | torch.device) -> DirectTensorBatch:
        return DirectTensorBatch(
            **{
                name: torch.from_numpy(self.arrays[name]).to(device)
                for name in (
                    "support_inputs",
                    "support_times",
                    "local_evidence",
                    "query_inputs",
                    "query_times",
                    "targets",
                )
            }
        )


def direct_input_arrays(
    codes: np.ndarray,
    pairs: np.ndarray,
    evidence: np.ndarray,
    times: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if steps < 2:
        raise ValueError("direct trials require at least two steps")
    trials, subjects = pairs.shape[:2]
    config = FactorizedPlasticRNNConfig(cue_size=codes.shape[-1])
    inputs = np.zeros((trials, steps, subjects, config.input_size), dtype=np.float32)
    time_values = np.broadcast_to(
        np.asarray(times, dtype=np.float32)[:, None, None, None],
        (trials, steps, subjects, 1),
    ).copy()
    indices = np.arange(subjects)[None, :]
    inputs[:, 0, :, : config.cue_size] = codes[indices, pairs[:, :, 0]]
    inputs[:, 0, :, config.cue_size : config.pair_cue_width] = codes[
        indices, pairs[:, :, 1]
    ]
    inputs[:, 0, :, config.evidence_index] = evidence
    inputs[:, 1, :, config.response_index] = 1.0
    return inputs, time_values


def expand_legacy_inputs(
    inputs: torch.Tensor, time_values: torch.Tensor, cue_size: int
) -> torch.Tensor:
    if inputs.shape[:-1] != time_values.shape[:-1] or time_values.shape[-1] != 1:
        raise ValueError("task inputs and time values must share leading dimensions")
    config = FactorizedPlasticRNNConfig(cue_size=cue_size)
    if inputs.shape[-1] != config.input_size:
        raise ValueError("task inputs have the wrong width")
    legacy = inputs.new_zeros(*inputs.shape[:-1], 2 * cue_size + 7)
    legacy[..., : config.evidence_index] = inputs[..., : config.evidence_index]
    legacy[..., 2 * cue_size + 1] = 1.0
    legacy[..., 2 * cue_size + 2 : 2 * cue_size + 3] = time_values
    legacy[..., 2 * cue_size + 4] = inputs[..., config.evidence_index]
    return legacy


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
        for trial_index, trial in enumerate(episode.support_trials):
            signed[trial_index, subject] = trial.signed_magnitude
            retained[trial_index, subject] = trial.encoding_reliability
            pairs[trial_index, subject] = (trial.left_item, trial.right_item)
            distance = round(
                abs(trial.signed_magnitude) * (len(episode.item_codes) - 1)
            )
            probabilities[trial_index, subject] = (
                episode.subject_encoding.relation_reliability(
                    trial.left_item, trial.right_item, distance
                )
            )
    if not np.all((retained == 0.0) | (retained == 1.0)):
        raise ValueError("direct training requires binary stable retention")
    return {
        "signed_magnitudes": signed,
        "retention": retained,
        "probabilities": probabilities,
        "support_pairs": pairs,
    }


def prepare_batch(
    episodes: tuple[RankingEpisode, ...], *, support_query_time: float = 2.0 / 3.0
) -> DirectEpisodeBatch:
    values = _support_values(episodes)
    codes = np.stack([episode.item_codes for episode in episodes])
    signed = values["signed_magnitudes"]
    retained = values["retention"]
    probabilities = values["probabilities"]
    n_support, batch_size = signed.shape
    n_queries = len(episodes[0].query_trials)
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
    support_inputs, support_times = direct_input_arrays(
        codes,
        values["support_pairs"],
        signed * retained,
        np.arange(n_support) / max(1, n_support - 1) * support_query_time,
        steps=4,
    )
    query_inputs, query_times = direct_input_arrays(
        codes,
        query_pairs,
        np.zeros((n_queries, batch_size), dtype=np.float32),
        np.full(n_queries, support_query_time, dtype=np.float32),
        steps=2,
    )
    query_inputs = np.ascontiguousarray(query_inputs.transpose(1, 0, 2, 3)).reshape(
        2, n_queries * batch_size, -1
    )
    query_times = np.ascontiguousarray(query_times.transpose(1, 0, 2, 3)).reshape(
        2, n_queries * batch_size, 1
    )
    return DirectEpisodeBatch(
        {
            **values,
            "support_inputs": support_inputs,
            "support_times": support_times,
            "local_evidence": np.asarray(
                signed * (retained + (1.0 - retained) * probabilities),
                dtype=np.float32,
            ),
            "query_inputs": query_inputs,
            "query_times": query_times,
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
    "DirectEpisodeBatch",
    "DirectTensorBatch",
    "direct_input_arrays",
    "expand_legacy_inputs",
    "prepare_batch",
    "sample_episodes",
]
