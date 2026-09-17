"""Extract only model-visible cue and realized-q tensors from parent batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class VisibleBatch:
    support_cues: np.ndarray
    realized_q: np.ndarray
    query_cues: np.ndarray
    targets: np.ndarray

    def tensors(self, device: str) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.as_tensor(np.ascontiguousarray(value), device=device)
            for value in (
                self.support_cues,
                self.realized_q,
                self.query_cues,
                self.targets,
            )
        )


def visible_batch(arrays: dict[str, np.ndarray]) -> VisibleBatch:
    support = np.asarray(arrays["support_inputs"])
    query = np.asarray(arrays["query_inputs"])
    targets = np.asarray(arrays["targets"])
    if support.ndim != 4 or support.shape[1] != 4 or support.shape[-1] not in {32, 38}:
        raise ValueError("unexpected parent support tensor")
    if query.ndim != 3 or query.shape[0] != 2 or query.shape[-1] != support.shape[-1]:
        raise ValueError("unexpected parent query tensor")
    subjects = support.shape[2]
    if targets.size % subjects:
        raise ValueError("query targets do not factor by subject")
    queries = targets.size // subjects
    cues = support[:, 0, :, :30]
    q = support[:, 0, :, -1]
    query_cues = query[0, :, :30].reshape(queries, subjects, 30).transpose(1, 0, 2)
    target_matrix = targets.reshape(queries, subjects).T
    if not np.array_equal(cues, support[:, 0, :, :30]):
        raise RuntimeError("cue extraction changed values")
    return VisibleBatch(cues, q, query_cues, target_matrix)


def margin_signs(batch: VisibleBatch) -> np.ndarray:
    return 2 * batch.targets - 1


__all__ = ["VisibleBatch", "margin_signs", "visible_batch"]
