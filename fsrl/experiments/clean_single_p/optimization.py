"""Single-P rollout and effective-coordinate optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .batches import SinglePTensorBatch, TimeMetadata
from .model import AffineSingleP, AffineSinglePSequence, margin_loss


@dataclass(frozen=True)
class SinglePResult:
    loss: torch.Tensor
    query_loss: torch.Tensor
    margins: torch.Tensor
    fast_weights: torch.Tensor
    first_support_write: torch.Tensor


def _sequence(
    condition: str,
    sequence: AffineSinglePSequence,
    inputs: torch.Tensor,
    times: torch.Tensor,
    hidden: torch.Tensor,
    eligibility: torch.Tensor,
    fast_weights: torch.Tensor,
    update: bool,
):
    if condition == "time_retained_control":
        return sequence(
            inputs,
            hidden,
            eligibility,
            fast_weights,
            update,
            time_values=times,
        )
    if condition == "clean_no_time":
        return sequence(inputs, hidden, eligibility, fast_weights, update)
    raise ValueError(f"unknown clean single-P condition: {condition}")


def forward_batch(
    condition: str,
    backbone: AffineSingleP,
    sequence: AffineSinglePSequence,
    batch: SinglePTensorBatch,
    times: TimeMetadata,
    *,
    penalty: float,
) -> SinglePResult:
    batch_size = batch.support_inputs.shape[2]
    weights = backbone.initial_fast_weights(batch_size)
    first = None
    for inputs, trial_times in zip(
        batch.support_inputs.unbind(0), times.support.unbind(0), strict=True
    ):
        _, _, _, _, weights = _sequence(
            condition,
            sequence,
            inputs,
            trial_times,
            backbone.initial_hidden(batch_size),
            backbone.initial_eligibility(batch_size),
            weights,
            True,
        )
        if first is None:
            first = weights
    if first is None:
        raise ValueError("single-P batch has no support trials")
    query_size = batch.targets.numel()
    query_count = query_size // batch_size
    query_weights = (
        weights.unsqueeze(0)
        .expand(query_count, -1, -1, -1)
        .reshape(
            query_size,
            backbone.model_config.hidden_size,
            backbone.model_config.hidden_size,
        )
    )
    margins, _, _, _, _ = _sequence(
        condition,
        sequence,
        batch.query_inputs,
        times.query,
        backbone.initial_hidden(query_size),
        backbone.initial_eligibility(query_size),
        query_weights,
        False,
    )
    query_loss = margin_loss(margins, batch.targets)
    loss = query_loss + penalty * weights.square().mean()
    return SinglePResult(loss, query_loss, margins, weights, first)


def folded_parameters(backbone: AffineSingleP) -> tuple[nn.Parameter, ...]:
    cue_bias = backbone.cue_projection.bias
    margin_bias = backbone.h2margin.bias
    if cue_bias is None or margin_bias is None:
        raise TypeError("clean single-P folded layers require biases")
    return (
        cast(nn.Parameter, cue_bias),
        backbone.evidence_weight,
        cast(nn.Parameter, backbone.h2margin.weight),
        cast(nn.Parameter, margin_bias),
    )


def make_optimizer(backbone: AffineSingleP, spec: dict) -> torch.optim.Adam:
    folded = folded_parameters(backbone)
    folded_ids = {id(value) for value in folded}
    ordinary = [value for value in backbone.parameters() if id(value) not in folded_ids]
    rate = spec["training"]["base_learning_rate"]
    return torch.optim.Adam(
        [
            {"params": ordinary, "lr": rate, "name": "ordinary"},
            {"params": list(folded), "lr": 2.0 * rate, "name": "folded"},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    )


def clip_gradients(backbone: AffineSingleP, maximum: float) -> torch.Tensor:
    folded_ids = {id(value) for value in folded_parameters(backbone)}
    norms = []
    parameters = []
    for parameter in backbone.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("nonfinite clean single-P gradient")
        parameters.append(parameter)
        norm = torch.linalg.vector_norm(parameter.grad)
        norms.append(norm * (2.0**0.5) if id(parameter) in folded_ids else norm)
    if not norms:
        raise RuntimeError("clean single-P backbone has no gradients")
    total = torch.linalg.vector_norm(torch.stack(norms))
    coefficient = torch.clamp(maximum / (total + 1e-6), max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(coefficient)
    return total


def training_step(
    condition: str,
    backbone: AffineSingleP,
    sequence: AffineSinglePSequence,
    batch: SinglePTensorBatch,
    times: TimeMetadata,
    optimizer: torch.optim.Adam,
    spec: dict,
) -> SinglePResult:
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(
        condition,
        backbone,
        sequence,
        batch,
        times,
        penalty=spec["training"]["fast_weight_penalty"],
    )
    result.loss.backward()
    clip_gradients(backbone, 2.0)
    optimizer.step()
    return result


__all__ = [
    "SinglePResult",
    "clip_gradients",
    "folded_parameters",
    "forward_batch",
    "make_optimizer",
    "training_step",
]
