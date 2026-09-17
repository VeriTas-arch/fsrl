"""Nested-prefix support rollout and read-only query evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fsrl.experiments.clean_single_p.batches import SinglePTensorBatch, TimeMetadata
from fsrl.experiments.clean_single_p.model import AffineSingleP, AffineSinglePSequence


@dataclass(frozen=True)
class NestedResult:
    margins: torch.Tensor
    p_norm: torch.Tensor
    effective_p_norm: torch.Tensor
    clamp_fraction: torch.Tensor
    prefix_weights: tuple[torch.Tensor, ...]


def nested_forward(
    backbone: AffineSingleP,
    sequence: AffineSinglePSequence,
    batch: SinglePTensorBatch,
    times: TimeMetadata,
    *,
    edge_count: int,
    blocks: int = 7,
) -> NestedResult:
    if backbone.retains_time:
        raise ValueError("anytime study accepts only the clean no-time model")
    expected_trials = edge_count * blocks
    if batch.support_inputs.shape[0] != expected_trials:
        raise ValueError("nested support length differs from B*E")
    batch_size = batch.support_inputs.shape[2]
    weights = backbone.initial_fast_weights(batch_size)
    saved = []
    for trial, (inputs, trial_times) in enumerate(
        zip(batch.support_inputs.unbind(0), times.support.unbind(0), strict=True), 1
    ):
        del trial_times
        _, _, _, _, weights = sequence(
            inputs,
            backbone.initial_hidden(batch_size),
            backbone.initial_eligibility(batch_size),
            weights,
            True,
        )
        if trial % edge_count == 0:
            saved.append(weights.clone())
    if len(saved) != blocks:
        raise RuntimeError("nested rollout did not save every block prefix")
    query_size = batch.targets.numel()
    query_count = query_size // batch_size
    margins = []
    p_norm = []
    effective_p_norm = []
    clamp = []
    for prefix in saved:
        query_weights = (
            prefix.unsqueeze(0)
            .expand(query_count, -1, -1, -1)
            .reshape(
                query_size,
                backbone.model_config.hidden_size,
                backbone.model_config.hidden_size,
            )
        )
        margin, _, _, _, returned = sequence(
            batch.query_inputs,
            backbone.initial_hidden(query_size),
            backbone.initial_eligibility(query_size),
            query_weights,
            False,
        )
        if not torch.equal(returned, query_weights):
            raise RuntimeError("anytime query wrote back to P")
        margins.append(margin[:, 0])
        p_norm.append(torch.linalg.matrix_norm(prefix).mean())
        effective_p_norm.append(
            torch.linalg.matrix_norm(backbone.alpha * prefix).mean()
        )
        clamp.append((prefix.abs() == 50.0).to(torch.float32).mean())
    return NestedResult(
        torch.stack(margins),
        torch.stack(p_norm),
        torch.stack(effective_p_norm),
        torch.stack(clamp),
        tuple(saved),
    )


__all__ = ["NestedResult", "nested_forward"]
