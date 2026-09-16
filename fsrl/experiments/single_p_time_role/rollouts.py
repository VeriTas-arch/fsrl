"""No-training rollouts for the frozen single-P time-role diagnostic."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch

from fsrl.experiments.clean_single_p.model import (
    AffineSingleP,
    AffineSinglePSequence,
    CleanSinglePConfig,
)
from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.locks import verify_reference


def load_cpu(record: dict, arm: str, recipe: dict) -> EpisodeBatch:
    with np.load(verify_reference(record), allow_pickle=False) as raw:
        cpu = EpisodeBatch({name: raw[name] for name in raw.files})
    return observed(cpu, arm, recipe)


def load_model(checkpoint: dict, condition: str) -> AffineSingleP:
    payload = torch.load(
        verify_reference(checkpoint), map_location="cuda", weights_only=True
    )
    model = AffineSingleP(
        CleanSinglePConfig(**payload["config"]),
        retain_time=condition == "time_retained_control",
        device="cuda",
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False).eval()
    return model


def clean_inputs(legacy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((*legacy.shape[:-1], 32), dtype=np.float32)
    inputs[..., :31] = legacy[..., :31]
    inputs[..., 31] = legacy[..., 37]
    times = np.asarray(legacy[..., 32:33], dtype=np.float32)
    return inputs, times


def ordered_queries(
    codes: np.ndarray, *, query_time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = len(codes)
    pairs = np.asarray(tuple(permutations(range(8), 2)), dtype=np.int64)
    inputs = np.zeros((2, len(pairs), subjects, 32), dtype=np.float32)
    indices = np.arange(subjects)[None, :]
    tiled = np.broadcast_to(pairs[:, None, :], (len(pairs), subjects, 2))
    inputs[0, :, :, :15] = codes[indices, tiled[:, :, 0]]
    inputs[0, :, :, 15:30] = codes[indices, tiled[:, :, 1]]
    inputs[1, :, :, 30] = 1.0
    flat = np.ascontiguousarray(inputs.transpose(0, 1, 2, 3)).reshape(
        2, len(pairs) * subjects, 32
    )
    times = np.full((2, len(pairs) * subjects, 1), query_time, dtype=np.float32)
    return flat, times, np.broadcast_to(pairs, (subjects, *pairs.shape))


def _sequence(
    model: AffineSingleP,
    sequence: AffineSinglePSequence,
    inputs: torch.Tensor,
    times: torch.Tensor,
    weights: torch.Tensor,
    *,
    update: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = inputs.shape[1]
    kwargs = {"time_values": times} if model.retains_time else {}
    margins, _, _, _, result = sequence(
        inputs,
        model.initial_hidden(count),
        model.initial_eligibility(count),
        weights,
        update,
        **kwargs,
    )
    return margins[:, 0], result


def support_trajectory(
    model: AffineSingleP,
    cpu: EpisodeBatch,
    *,
    support_times: np.ndarray | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    clean, original_times = clean_inputs(cpu.arrays["support_inputs"])
    times = original_times if support_times is None else support_times
    inputs = torch.from_numpy(clean).to("cuda")
    time_tensor = torch.from_numpy(np.asarray(times, dtype=np.float32)).to("cuda")
    count = inputs.shape[2]
    weights = model.initial_fast_weights(count)
    sequence = AffineSinglePSequence(model)
    states = [weights.clone()]
    writes = []
    for trial, trial_times in zip(inputs.unbind(0), time_tensor.unbind(0), strict=True):
        before = weights
        _, weights = _sequence(
            model, sequence, trial, trial_times, weights, update=True
        )
        writes.append(weights - before)
        states.append(weights.clone())
    return states, writes


def read_ordered(
    model: AffineSingleP,
    weights: torch.Tensor,
    codes: np.ndarray,
    *,
    query_time: float = 2.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    inputs, times, pairs = ordered_queries(codes, query_time=query_time)
    subjects, queries = len(codes), 56
    expanded = (
        weights.unsqueeze(0)
        .expand(queries, -1, -1, -1)
        .reshape(queries * subjects, weights.shape[-2], weights.shape[-1])
    )
    margins, _ = _sequence(
        model,
        AffineSinglePSequence(model),
        torch.from_numpy(inputs).to("cuda"),
        torch.from_numpy(times).to("cuda"),
        expanded,
        update=False,
    )
    return margins.reshape(queries, subjects).T.cpu().numpy(), pairs


def read_original(
    model: AffineSingleP,
    weights: torch.Tensor,
    cpu: EpisodeBatch,
    *,
    query_time: float | None = None,
) -> np.ndarray:
    clean, times = clean_inputs(cpu.arrays["query_inputs"])
    if query_time is not None:
        times.fill(query_time)
    subjects = weights.shape[0]
    queries = cpu.arrays["targets"].size // subjects
    expanded = (
        weights.unsqueeze(0)
        .expand(queries, -1, -1, -1)
        .reshape(queries * subjects, weights.shape[-2], weights.shape[-1])
    )
    margins, _ = _sequence(
        model,
        AffineSinglePSequence(model),
        torch.from_numpy(clean).to("cuda"),
        torch.from_numpy(times).to("cuda"),
        expanded,
        update=False,
    )
    return margins.reshape(queries, subjects).T.cpu().numpy()


__all__ = [
    "clean_inputs",
    "load_cpu",
    "load_model",
    "ordered_queries",
    "read_ordered",
    "read_original",
    "support_trajectory",
]
