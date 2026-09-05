"""Task-only adapters; no recurrent checkpoint or query label enters a state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.experiments.training_strategy.batches import EpisodeBatch, prepare_batch
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol
from fsrl.tasks.sparse_ranking import RankingEpisode


@dataclass(frozen=True)
class ModelBatch:
    arrays: dict[str, np.ndarray]

    def fingerprint(self) -> str:
        return EpisodeBatch(self.arrays).fingerprint()

    def tensors(self, device: str, dtype: torch.dtype = torch.float32) -> tuple:
        width = self.arrays["support_cues"].shape[-1]
        packed = np.concatenate(
            [
                self.arrays["support_cues"],
                *(
                    self.arrays[key][..., None]
                    for key in ("signed", "retention", "local_evidence")
                ),
            ],
            axis=-1,
        )
        support = torch.as_tensor(
            np.ascontiguousarray(packed), dtype=dtype, device=device
        )
        query = torch.as_tensor(
            np.ascontiguousarray(self.arrays["query_cues"]), dtype=dtype, device=device
        )
        return (
            support[..., :width],
            support[..., width],
            support[..., width + 1],
            support[..., width + 2],
            query,
        )


def pair_cues(codes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    subjects = np.arange(codes.shape[0])[:, None]
    return np.concatenate(
        (codes[subjects, pairs[..., 0]], codes[subjects, pairs[..., 1]]),
        axis=-1,
    )


def generic_batch(episodes: tuple[RankingEpisode, ...]) -> ModelBatch:
    original = prepare_batch(episodes).arrays
    codes = original["item_codes"]
    query_pairs = original["query_pairs"].transpose(1, 0, 2)
    learned = np.asarray(
        [
            [
                tuple(sorted(pair))
                in {
                    tuple(sorted((t.left_item, t.right_item)))
                    for t in episode.support_trials
                }
                for pair in pairs
            ]
            for episode, pairs in zip(episodes, query_pairs, strict=True)
        ]
    )
    return ModelBatch(
        {
            "support_cues": original["support_inputs"][:, 0, :, : 2 * codes.shape[-1]],
            "signed": original["signed_magnitudes"],
            "retention": original["retention"],
            "probabilities": original["probabilities"],
            "local_evidence": original["local_evidence"],
            "query_cues": pair_cues(codes, query_pairs),
            "targets": original["targets"].reshape(-1, len(episodes)).T,
            "learned": learned,
            "codes": codes,
            "query_pairs": query_pairs,
            "support_pairs": original["support_pairs"].transpose(1, 0, 2),
            "graphs": original["graphs"],
        }
    )


def liu_batch(spec: dict):
    settings = spec["evaluation"]["liu"]
    protocol = load_registered_protocol(settings["protocol_id"])
    evaluator = FrozenFastWeightEvaluator(
        None,
        TrainConfig(bs=settings["subjects"], cs=spec["task"]["cue_size"]),
        protocol,
        protocol_only=True,
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
    assert evaluator.subject_encoding_states is not None
    assert evaluator.subject_trial_gains is not None
    pairs = np.asarray(
        [
            [(t.left_item, t.right_item) for t in schedule]
            for schedule in evaluator.support_schedules
        ]
    )
    signed = np.asarray(
        [
            [t.signed_magnitude for t in schedule]
            for schedule in evaluator.support_schedules
        ]
    )
    z = np.asarray(evaluator.subject_trial_gains)
    p = np.asarray(
        [
            [
                state.relation_reliability(
                    t.left_item,
                    t.right_item,
                    round(abs(t.signed_magnitude) * (protocol.n_items - 1)),
                )
                for t in schedule
            ]
            for state, schedule in zip(
                evaluator.subject_encoding_states,
                evaluator.support_schedules,
                strict=True,
            )
        ]
    )
    queries = np.broadcast_to(
        np.asarray(ordered_pairs(protocol.n_items))[None],
        (settings["subjects"], protocol.n_items * (protocol.n_items - 1), 2),
    )
    return protocol, ModelBatch(
        {
            "support_cues": pair_cues(evaluator.cue_codes, pairs).transpose(1, 0, 2),
            "signed": signed.T,
            "retention": z.T,
            "probabilities": p.T,
            "local_evidence": (signed * (z + (1 - z) * p)).T,
            "query_cues": pair_cues(evaluator.cue_codes, queries),
            "codes": evaluator.cue_codes,
            "support_pairs": pairs,
            "query_pairs": queries,
        }
    )
