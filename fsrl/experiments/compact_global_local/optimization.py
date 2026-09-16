"""Registered staged objective for the compact P/L candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .batches import CompactTensorBatch
from .model import CompactPlasticRNN, PackedLocalTrace


@dataclass(frozen=True)
class CompactBatchResult:
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


def configure_phase(
    backbone: CompactPlasticRNN, local: PackedLocalTrace, phase: str
) -> None:
    if phase not in {"global", "local"}:
        raise ValueError(f"unknown optimization phase: {phase}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(phase == "global")
        parameter.grad = None
    local.raw_gain.requires_grad_(phase == "local")
    local.raw_gain.grad = None


def make_optimizer(
    backbone: CompactPlasticRNN,
    local: PackedLocalTrace,
    optimization: dict,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": list(backbone.parameters()),
                "lr": optimization["backbone_learning_rate"],
                "name": "backbone",
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


def query_from_state(
    backbone: CompactPlasticRNN,
    local: PackedLocalTrace,
    sequence: nn.Module,
    batch: CompactTensorBatch,
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
    global_margin, _, _, _, _ = sequence(
        batch.query_inputs,
        backbone.initial_hidden(query_size),
        backbone.initial_eligibility(query_size),
        query_weights,
        False,
    )
    query_trace = local_state.repeat(n_queries, 1)
    pair_cues = batch.query_inputs[0, :, : 2 * local.cue_size]
    margins, _, _ = local(global_margin, query_trace, pair_cues, active=local_active)
    return margins, global_margin


def forward_batch(
    backbone: CompactPlasticRNN,
    local: PackedLocalTrace,
    sequence: nn.Module,
    batch: CompactTensorBatch,
    *,
    local_active: bool,
    fast_weight_penalty: float,
) -> CompactBatchResult:
    batch_size = batch.support_inputs.shape[2]
    hidden = backbone.initial_hidden(batch_size)
    eligibility = backbone.initial_eligibility(batch_size)
    fast_weights = backbone.initial_fast_weights(batch_size)
    local_state = local.initial_state(batch_size)
    first_support_write = None
    for trial, inputs in enumerate(batch.support_inputs.unbind()):
        _, _, _, _, fast_weights = sequence(
            inputs, hidden, eligibility, fast_weights, True
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
        backbone,
        local,
        sequence,
        batch,
        fast_weights,
        local_state,
        local_active=local_active,
    )
    query_loss = margin_loss(margins, batch.targets)
    loss = query_loss + fast_weight_penalty * fast_weights.square().mean()
    return CompactBatchResult(
        loss,
        query_loss,
        margins,
        global_margins,
        fast_weights,
        first_support_write,
        local_state,
    )


def training_step(
    backbone: CompactPlasticRNN,
    local: PackedLocalTrace,
    sequence: nn.Module,
    batch: CompactTensorBatch,
    optimizer: torch.optim.Adam,
    *,
    phase: str,
    optimization: dict,
) -> CompactBatchResult:
    configure_phase(backbone, local, phase)
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(
        backbone,
        local,
        sequence,
        batch,
        local_active=phase == "local",
        fast_weight_penalty=(
            optimization["fast_weight_penalty"] if phase == "global" else 0.0
        ),
    )
    result.loss.backward()
    backbone_gradients = [
        parameter for parameter in backbone.parameters() if parameter.grad is not None
    ]
    if backbone_gradients:
        torch.nn.utils.clip_grad_norm_(
            backbone_gradients,
            optimization["gradient_clip"],
            error_if_nonfinite=True,
        )
    if local.raw_gain.grad is not None:
        torch.nn.utils.clip_grad_norm_(
            [local.raw_gain],
            optimization["gradient_clip"],
            error_if_nonfinite=True,
        )
    optimizer.step()
    return result
