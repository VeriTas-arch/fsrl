"""The sole 32D eager arithmetic path for direct baseline and mechanisms."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch

from fsrl.experiments.clean_single_p.model import AffineSingleP, CleanSinglePConfig
from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.infra.provenance import tensor_hashes


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
    expected = checkpoint.get("tensor_hashes")
    if expected is not None and tensor_hashes(model) != expected:
        raise RuntimeError("loaded direct model differs from locked tensors")
    model.requires_grad_(False).eval()
    return model


def clean_inputs(legacy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((*legacy.shape[:-1], 32), dtype=np.float32)
    inputs[..., :31] = legacy[..., :31]
    inputs[..., 31] = legacy[..., 37]
    return inputs, np.asarray(legacy[..., 32:33], dtype=np.float32)


def _sequence(
    model: AffineSingleP,
    inputs: torch.Tensor,
    times: torch.Tensor,
    weights: torch.Tensor,
    *,
    update: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = model.initial_hidden(inputs.shape[1])
    eligibility = model.initial_eligibility(inputs.shape[1])
    total_write = weights.new_zeros(inputs.shape[1])
    margin = modulation = None
    for step_inputs, step_time in zip(inputs.unbind(0), times.unbind(0), strict=True):
        margin, modulation, hidden, eligibility, proposal = model.step(
            step_inputs,
            hidden,
            eligibility,
            weights,
            time_values=step_time if model.retains_time else None,
        )
        if update:
            total_write += ((proposal - weights) * model.alpha).abs().mean((-2, -1))
            weights = proposal
    if margin is None or modulation is None:
        raise ValueError("a direct sequence must contain at least one step")
    return margin[:, 0], modulation[:, 0], weights, total_write


def support_trajectory(
    model: AffineSingleP,
    cpu: EpisodeBatch,
    *,
    support_times: np.ndarray | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    clean, original_times = clean_inputs(cpu.arrays["support_inputs"])
    times = original_times if support_times is None else support_times
    inputs = torch.from_numpy(clean).to("cuda")
    time_tensor = torch.from_numpy(np.asarray(times, dtype=np.float32)).to("cuda")
    weights = model.initial_fast_weights(inputs.shape[2])
    states = [weights.clone()]
    writes: list[torch.Tensor] = []
    modulations: list[torch.Tensor] = []
    total_write = weights.new_zeros(inputs.shape[2])
    for trial, trial_times in zip(inputs.unbind(0), time_tensor.unbind(0), strict=True):
        before = weights
        _, modulation, weights, amount = _sequence(
            model, trial, trial_times, weights, update=True
        )
        writes.append(weights - before)
        modulations.append(modulation)
        total_write += amount
        states.append(weights.clone())
    return states, writes, modulations, total_write


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
    flat = np.ascontiguousarray(inputs).reshape(2, len(pairs) * subjects, 32)
    times = np.full((2, len(pairs) * subjects, 1), query_time, dtype=np.float32)
    return flat, times, np.broadcast_to(pairs, (subjects, *pairs.shape))


def _expanded_weights(weights: torch.Tensor, queries: int) -> torch.Tensor:
    subjects = weights.shape[0]
    return (
        weights.unsqueeze(0)
        .expand(queries, -1, -1, -1)
        .reshape(queries * subjects, weights.shape[-2], weights.shape[-1])
    )


def read_ordered(
    model: AffineSingleP,
    weights: torch.Tensor,
    codes: np.ndarray,
    *,
    query_time: float = 2.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    inputs, times, pairs = ordered_queries(codes, query_time=query_time)
    subjects, queries = len(codes), 56
    expanded = _expanded_weights(weights, queries)
    before = expanded.clone()
    margins, _, result, _ = _sequence(
        model,
        torch.from_numpy(inputs).to("cuda"),
        torch.from_numpy(times).to("cuda"),
        expanded,
        update=False,
    )
    if not torch.equal(before, result):
        raise RuntimeError("read-only ordered query changed terminal P")
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
    expanded = _expanded_weights(weights, queries)
    before = expanded.clone()
    margins, _, result, _ = _sequence(
        model,
        torch.from_numpy(clean).to("cuda"),
        torch.from_numpy(times).to("cuda"),
        expanded,
        update=False,
    )
    if not torch.equal(before, result):
        raise RuntimeError("read-only original query changed terminal P")
    return margins.reshape(queries, subjects).T.cpu().numpy()


def apply_support_probe(
    model: AffineSingleP,
    inputs: np.ndarray,
    times: np.ndarray,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply one probe through the same direct step loop as natural support."""
    _, _, changed, _ = _sequence(
        model,
        torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to("cuda"),
        torch.from_numpy(np.asarray(times, dtype=np.float32)).to("cuda"),
        weights,
        update=True,
    )
    return changed


def assert_bitwise_equal(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> None:
    if first.keys() != second.keys():
        raise RuntimeError("direct replay array inventory differs")
    for name, value in first.items():
        left, right = np.asarray(value), np.asarray(second[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise RuntimeError(f"direct replay metadata differs: {name}")
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise RuntimeError(f"direct replay is not bitwise identical: {name}")


__all__ = [
    "apply_support_probe",
    "assert_bitwise_equal",
    "clean_inputs",
    "load_cpu",
    "load_model",
    "ordered_queries",
    "read_ordered",
    "read_original",
    "support_trajectory",
]
