"""Paired clean single-P batches with one realized observation channel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from fsrl.experiments.pl_direct_training.batches import prepare_batch
from fsrl.tasks.sparse_ranking import RankingEpisode


@dataclass(frozen=True)
class SinglePTensorBatch:
    support_inputs: torch.Tensor
    query_inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class TimeMetadata:
    support: torch.Tensor
    query: torch.Tensor


@dataclass(frozen=True)
class SinglePEpisodeBatch:
    arrays: dict[str, np.ndarray]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays.items()):
            array = np.ascontiguousarray(value)
            digest.update(
                json.dumps([name, array.dtype.str, list(array.shape)]).encode("ascii")
                + b"\0"
            )
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to(self, device: str | torch.device) -> tuple[SinglePTensorBatch, TimeMetadata]:
        return (
            SinglePTensorBatch(
                support_inputs=torch.from_numpy(self.arrays["support_inputs"]).to(
                    device
                ),
                query_inputs=torch.from_numpy(self.arrays["query_inputs"]).to(device),
                targets=torch.from_numpy(self.arrays["targets"]).to(device),
            ),
            TimeMetadata(
                support=torch.from_numpy(self.arrays["support_times"]).to(device),
                query=torch.from_numpy(self.arrays["query_times"]).to(device),
            ),
        )


def prepare_single_p(
    episodes: tuple[RankingEpisode, ...],
    arm: str,
    *,
    observation_seed: int,
    sigma: float = 1.0 / 7.0,
) -> SinglePEpisodeBatch:
    direct = prepare_batch(episodes)
    arrays = {name: value.copy() for name, value in direct.arrays.items()}
    q = arrays["local_evidence"].copy()
    if arm == "noisy":
        q = (
            q.astype(np.float64)
            + sigma * np.random.default_rng(observation_seed).standard_normal(q.shape)
        ).astype(np.float32)
    elif arm != "clean":
        raise ValueError(f"unknown observation arm: {arm}")
    arrays["realized_q"] = q
    arrays["support_inputs"][:, 0, :, -1] = q
    return SinglePEpisodeBatch(arrays)


__all__ = [
    "SinglePEpisodeBatch",
    "SinglePTensorBatch",
    "TimeMetadata",
    "prepare_single_p",
]
