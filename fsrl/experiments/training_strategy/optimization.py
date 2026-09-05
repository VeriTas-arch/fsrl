"""One query objective with the two prospectively registered update schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.plastic_rnn import RetroModulRNN

from .batches import TensorBatch


@dataclass(frozen=True)
class BatchResult:
    loss: torch.Tensor
    query_loss: torch.Tensor
    logits: torch.Tensor
    global_logits: torch.Tensor
    fast_weights: torch.Tensor
    first_support_write: torch.Tensor
    local_state: torch.Tensor


def configure_phase(
    backbone: RetroModulRNN, local: ConjunctiveLocalTrace, phase: str
) -> None:
    if phase not in {"global", "local", "joint"}:
        raise ValueError(f"unknown optimization phase: {phase}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(phase != "local")
        parameter.grad = None
    local.raw_gain.requires_grad_(phase != "global")
    local.raw_gain.grad = None


def make_optimizer(
    backbone: RetroModulRNN, local: ConjunctiveLocalTrace, optimization: dict
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
    backbone: RetroModulRNN,
    local: ConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: TensorBatch,
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
    global_logits, _, _, _, _, _ = sequence(
        batch.query_inputs,
        backbone.initial_hidden(query_size),
        backbone.initial_eligibility(query_size),
        query_weights,
        False,
    )
    if not local_active:
        return global_logits, global_logits
    query_trace = local_state.repeat(n_queries, 1)
    pair_cues = batch.query_inputs[0, :, : 2 * local.cue_size]
    logits, _, _, _ = local(global_logits, query_trace, pair_cues)
    return logits, global_logits


def forward_batch(
    backbone: RetroModulRNN,
    local: ConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: TensorBatch,
    *,
    local_active: bool,
    fast_weight_penalty: float,
) -> BatchResult:
    batch_size = batch.support_inputs.shape[2]
    hidden = backbone.initial_hidden(batch_size)
    eligibility = backbone.initial_eligibility(batch_size)
    fast_weights = backbone.initial_fast_weights(batch_size)
    blank = batch.support_inputs.new_zeros(
        2, batch_size, backbone.model_config.input_size
    )
    _, _, _, _, _, fast_weights = sequence(
        blank, hidden, eligibility, fast_weights, True
    )
    local_state = local.initial_state(batch_size)
    first_support_write = None
    for trial, inputs in enumerate(batch.support_inputs.unbind()):
        _, _, _, _, _, fast_weights = sequence(
            inputs, hidden, eligibility, fast_weights, True
        )
        local_state = local.write(
            local_state, inputs[0, :, : 2 * local.cue_size], batch.local_evidence[trial]
        )
        if trial == 0:
            first_support_write = fast_weights
    assert first_support_write is not None
    logits, global_logits = query_from_state(
        backbone,
        local,
        sequence,
        batch,
        fast_weights,
        local_state,
        local_active=local_active,
    )
    query_loss = F.cross_entropy(logits, batch.targets)
    loss = query_loss + fast_weight_penalty * fast_weights.square().mean()
    return BatchResult(
        loss,
        query_loss,
        logits,
        global_logits,
        fast_weights,
        first_support_write,
        local_state,
    )


def training_step(
    backbone: RetroModulRNN,
    local: ConjunctiveLocalTrace,
    sequence: nn.Module,
    batch: TensorBatch,
    optimizer: torch.optim.Adam,
    *,
    phase: str,
    optimization: dict,
) -> BatchResult:
    configure_phase(backbone, local, phase)
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(
        backbone,
        local,
        sequence,
        batch,
        local_active=phase != "global",
        fast_weight_penalty=(
            0.0 if phase == "local" else optimization["fast_weight_penalty"]
        ),
    )
    result.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        backbone.parameters(), optimization["gradient_clip"], error_if_nonfinite=True
    )
    torch.nn.utils.clip_grad_norm_(
        [local.raw_gain], optimization["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return result
