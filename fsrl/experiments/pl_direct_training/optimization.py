"""Joint direct-training objective with effective-coordinate Adam geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.factorized_plastic_rnn import FactorizedPlasticRNN
from fsrl.core.local_trace import PackedConjunctiveLocalTrace

from .batches import DirectTensorBatch
from .model import NoTimePlasticRNN

DirectBackbone = FactorizedPlasticRNN | NoTimePlasticRNN


@dataclass(frozen=True)
class DirectBatchResult:
    loss: torch.Tensor
    query_loss: torch.Tensor
    margins: torch.Tensor
    global_margins: torch.Tensor
    fast_weights: torch.Tensor
    first_support_write: torch.Tensor
    local_state: torch.Tensor


def margin_loss(margins: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    signs = 2.0 * targets.to(margins.dtype).reshape_as(margins) - 1.0
    return F.softplus(-signs * margins).mean()


def run_sequence(
    condition: str,
    sequence: nn.Module,
    inputs: torch.Tensor,
    times: torch.Tensor,
    hidden: torch.Tensor,
    eligibility: torch.Tensor,
    fast_weights: torch.Tensor,
    update_fast_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition == "time_retained_control":
        return sequence(
            inputs,
            times,
            hidden,
            eligibility,
            fast_weights,
            update_fast_weights,
        )
    if condition == "no_time_candidate":
        return sequence(
            inputs,
            hidden,
            eligibility,
            fast_weights,
            update_fast_weights,
        )
    raise ValueError(f"unknown direct-training condition: {condition}")


def query_from_state(
    condition: str,
    backbone: DirectBackbone,
    local: PackedConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: DirectTensorBatch,
    fast_weights: torch.Tensor,
    local_state: torch.Tensor,
    *,
    local_active: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = fast_weights.shape[0]
    query_size = batch.targets.numel()
    n_queries = query_size // batch_size
    query_weights = (
        fast_weights.unsqueeze(0)
        .expand(n_queries, -1, -1, -1)
        .reshape(
            query_size,
            backbone.model_config.hidden_size,
            backbone.model_config.hidden_size,
        )
    )
    global_margin, _, _, _, _ = run_sequence(
        condition,
        sequence,
        batch.query_inputs,
        batch.query_times,
        backbone.initial_hidden(query_size),
        backbone.initial_eligibility(query_size),
        query_weights,
        False,
    )
    query_trace = local_state.repeat(n_queries, 1)
    pair_cues = batch.query_inputs[0, :, : 2 * local.cue_size]
    gain_override = None if local_active else torch.zeros_like(global_margin)
    margins, _, _, _ = local(
        global_margin,
        query_trace,
        pair_cues,
        gain_override=gain_override,
    )
    return margins, global_margin


def forward_batch(
    condition: str,
    backbone: DirectBackbone,
    local: PackedConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: DirectTensorBatch,
    *,
    fast_weight_penalty: float,
) -> DirectBatchResult:
    batch_size = batch.support_inputs.shape[2]
    fast_weights = backbone.initial_fast_weights(batch_size)
    local_state = local.initial_state(batch_size)
    first_support_write = None
    for trial, (inputs, times) in enumerate(
        zip(batch.support_inputs.unbind(), batch.support_times.unbind(), strict=True)
    ):
        _, _, _, _, fast_weights = run_sequence(
            condition,
            sequence,
            inputs,
            times,
            backbone.initial_hidden(batch_size),
            backbone.initial_eligibility(batch_size),
            fast_weights,
            True,
        )
        local_state = local.write(
            local_state,
            inputs[0, :, : 2 * local.cue_size],
            batch.local_evidence[trial],
        )
        if trial == 0:
            first_support_write = fast_weights
    if first_support_write is None:
        raise ValueError("a batch must contain at least one support trial")
    margins, global_margins = query_from_state(
        condition,
        backbone,
        local,
        sequence,
        batch,
        fast_weights,
        local_state,
        local_active=True,
    )
    query_loss = margin_loss(margins, batch.targets)
    loss = query_loss + fast_weight_penalty * fast_weights.square().mean()
    return DirectBatchResult(
        loss,
        query_loss,
        margins,
        global_margins,
        fast_weights,
        first_support_write,
        local_state,
    )


def folded_parameters(backbone: DirectBackbone) -> tuple[nn.Parameter, ...]:
    input_bias = backbone.input_projection.bias
    margin_bias = backbone.h2margin.bias
    if input_bias is None or margin_bias is None:
        raise TypeError("direct-training folded layers require bias parameters")
    return (
        cast(nn.Parameter, input_bias),
        cast(nn.Parameter, backbone.h2margin.weight),
        cast(nn.Parameter, margin_bias),
    )


def make_optimizer(
    backbone: DirectBackbone,
    local: PackedConjunctiveLocalTrace,
    optimization: dict,
) -> torch.optim.Adam:
    folded = folded_parameters(backbone)
    folded_ids = {id(parameter) for parameter in folded}
    ordinary = [
        parameter
        for parameter in backbone.parameters()
        if id(parameter) not in folded_ids
    ]
    return torch.optim.Adam(
        [
            {
                "params": ordinary,
                "lr": optimization["base_backbone_learning_rate"],
                "name": "backbone",
            },
            {
                "params": list(folded),
                "lr": 2.0 * optimization["base_backbone_learning_rate"],
                "name": "folded_backbone",
            },
            {
                "params": [local.raw_gain],
                "lr": optimization["local_learning_rate"],
                "name": "local",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )


def clip_effective_backbone_gradients(
    backbone: DirectBackbone, maximum: float
) -> torch.Tensor:
    folded_ids = {id(parameter) for parameter in folded_parameters(backbone)}
    norms = []
    parameters = []
    for parameter in backbone.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("nonfinite direct-training gradient")
        parameters.append(parameter)
        norm = torch.linalg.vector_norm(parameter.grad)
        if id(parameter) in folded_ids:
            norm = norm * (2.0**0.5)
        norms.append(norm)
    if not norms:
        raise RuntimeError("direct backbone has no gradients")
    total = torch.linalg.vector_norm(torch.stack(norms))
    coefficient = torch.clamp(maximum / (total + 1e-6), max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(coefficient)
    return total


def training_step(
    condition: str,
    backbone: DirectBackbone,
    local: PackedConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: DirectTensorBatch,
    optimizer: torch.optim.Adam,
    *,
    optimization: dict,
) -> DirectBatchResult:
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(
        condition,
        backbone,
        local,
        sequence,
        batch,
        fast_weight_penalty=optimization["fast_weight_penalty"],
    )
    result.loss.backward()
    clip_effective_backbone_gradients(backbone, optimization["gradient_clip"])
    torch.nn.utils.clip_grad_norm_(
        [local.raw_gain],
        optimization["gradient_clip"],
        error_if_nonfinite=True,
    )
    optimizer.step()
    return result


__all__ = [
    "DirectBackbone",
    "DirectBatchResult",
    "clip_effective_backbone_gradients",
    "folded_parameters",
    "forward_batch",
    "make_optimizer",
    "margin_loss",
    "query_from_state",
    "run_sequence",
    "training_step",
]
