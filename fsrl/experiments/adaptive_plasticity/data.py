"""Cue-derived relation slots and matched schedule transformations."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import canonical_addresses


def relation_slots(support_cues: np.ndarray) -> np.ndarray:
    keys, _ = canonical_addresses(support_cues)
    slots = np.empty(keys.shape, dtype=np.int64)
    for subject in range(keys.shape[1]):
        address: dict[int, int] = {}
        for trial in range(keys.shape[0]):
            key = int(keys[trial, subject])
            if key not in address:
                address[key] = len(address)
            slots[trial, subject] = address[key]
    return slots


def model_tensors(
    batch: ModelBatch, device: str, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    support, signed, retention, _, query = batch.tensors(device, dtype)
    slots = torch.as_tensor(
        np.ascontiguousarray(relation_slots(batch.arrays["support_cues"])),
        dtype=torch.int64,
        device=device,
    )
    return support, signed, retention, slots, query


def occurrence_indices(support_cues: np.ndarray) -> np.ndarray:
    slots = relation_slots(support_cues)
    occurrences = np.empty(slots.shape, dtype=np.int8)
    for subject in range(slots.shape[1]):
        counts: dict[int, int] = {}
        for trial in range(slots.shape[0]):
            slot = int(slots[trial, subject])
            occurrences[trial, subject] = counts.get(slot, 0)
            counts[slot] = int(occurrences[trial, subject]) + 1
    return occurrences


def clustered_batch(
    batch: ModelBatch, uniforms: np.ndarray, rng: np.random.Generator
) -> tuple[ModelBatch, np.ndarray, np.ndarray]:
    """Group each relation's four existing occurrences without changing content."""
    cues = batch.arrays["support_cues"]
    keys, _ = canonical_addresses(cues)
    trials, subjects = keys.shape
    permutations = np.empty((trials, subjects), dtype=np.int64)
    for subject in range(subjects):
        unique = list(dict.fromkeys(int(value) for value in keys[:, subject]))
        order = [unique[index] for index in rng.permutation(len(unique))]
        permutations[:, subject] = np.concatenate(
            [np.flatnonzero(keys[:, subject] == key) for key in order]
        )
    arrays = dict(batch.arrays)
    subject_axis = np.arange(subjects)[None, :]
    for name in (
        "support_cues",
        "signed",
        "retention",
        "probabilities",
        "local_evidence",
    ):
        values = batch.arrays[name]
        arrays[name] = values[permutations, subject_axis]
    if "support_pairs" in arrays:
        pairs = batch.arrays["support_pairs"]
        arrays["support_pairs"] = pairs[np.arange(subjects)[:, None], permutations.T]
    clustered_uniforms = uniforms[permutations, subject_axis]
    return ModelBatch(arrays), clustered_uniforms, permutations
