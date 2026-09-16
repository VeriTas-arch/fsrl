"""Deterministic batch slicing used by the frozen Stage-3 selection."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.training_strategy.batches import EpisodeBatch


def subset_batch(cpu: EpisodeBatch, selected: np.ndarray) -> EpisodeBatch:
    arrays = {}
    subjects = cpu.arrays["item_codes"].shape[0]
    for name, value in cpu.arrays.items():
        if name == "support_inputs":
            arrays[name] = value[:, :, selected].copy()
        elif name in {
            "local_evidence",
            "signed_magnitudes",
            "retention",
            "probabilities",
            "support_pairs",
        }:
            arrays[name] = value[:, selected].copy()
        elif name == "query_inputs":
            queries = value.shape[1] // subjects
            arrays[name] = (
                value.reshape(value.shape[0], queries, subjects, value.shape[-1])[
                    :, :, selected
                ]
                .copy()
                .reshape(value.shape[0], queries * len(selected), value.shape[-1])
            )
        elif name == "targets":
            arrays[name] = value.reshape(-1, subjects)[:, selected].copy().reshape(-1)
        elif value.ndim and value.shape[0] == subjects:
            arrays[name] = value[selected].copy()
        elif value.ndim >= 2 and value.shape[1] == subjects:
            arrays[name] = value[:, selected].copy()
        else:
            arrays[name] = value.copy()
    return EpisodeBatch(arrays)


def selected_generic(cpu: EpisodeBatch) -> tuple[EpisodeBatch, np.ndarray]:
    selected = np.argsort(cpu.arrays["episode_indices"])[:2]
    return subset_batch(cpu, selected), selected


__all__ = ["selected_generic", "subset_batch"]
