"""Single-rate optimization and rollout for the minimal single-P ladder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fsrl.experiments.clean_single_p.batches import SinglePTensorBatch

from .model import MinimalSingleP, MinimalSinglePSequence


@dataclass(frozen=True)
class MinimalSinglePResult:
    loss: torch.Tensor
    query_loss: torch.Tensor
    margins: torch.Tensor
    fast_weights: torch.Tensor


def margin_loss(margins: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    signs = 2.0 * targets.to(margins.dtype).reshape_as(margins) - 1.0
    return F.softplus(-signs * margins).mean()


def query_from_weights(
    model: MinimalSingleP,
    sequence: MinimalSinglePSequence,
    query_inputs: torch.Tensor,
    fast_weights: torch.Tensor,
) -> torch.Tensor:
    query_size = query_inputs.shape[1]
    margins, _, _, _, _ = sequence(
        query_inputs,
        model.initial_hidden(query_size),
        model.initial_eligibility(query_size),
        fast_weights,
        False,
    )
    return margins


def forward_batch(
    model: MinimalSingleP,
    sequence: MinimalSinglePSequence,
    batch: SinglePTensorBatch,
    *,
    penalty: float,
) -> MinimalSinglePResult:
    batch_size = batch.support_inputs.shape[2]
    weights = model.initial_fast_weights(batch_size)
    for inputs in batch.support_inputs.unbind(0):
        _, _, _, _, weights = sequence(
            inputs,
            model.initial_hidden(batch_size),
            model.initial_eligibility(batch_size),
            weights,
            True,
        )
    query_size = batch.targets.numel()
    query_count = query_size // batch_size
    query_weights = (
        weights.unsqueeze(0)
        .expand(query_count, -1, -1, -1)
        .reshape(
            query_size,
            model.model_config.hidden_size,
            model.model_config.hidden_size,
        )
    )
    margins = query_from_weights(model, sequence, batch.query_inputs, query_weights)
    query_loss = margin_loss(margins, batch.targets)
    loss = query_loss + penalty * weights.square().mean()
    return MinimalSinglePResult(loss, query_loss, margins, weights)


def make_optimizer(model: MinimalSingleP, learning_rate: float) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    )


def global_gradient_norm(model: MinimalSingleP) -> torch.Tensor:
    norms = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("minimal single-P parameter has no gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("nonfinite minimal single-P gradient")
        norms.append(torch.linalg.vector_norm(parameter.grad))
    return torch.linalg.vector_norm(torch.stack(norms))


def training_step(
    model: MinimalSingleP,
    sequence: MinimalSinglePSequence,
    batch: SinglePTensorBatch,
    optimizer: torch.optim.Adam,
    *,
    penalty: float,
) -> tuple[MinimalSinglePResult, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(model, sequence, batch, penalty=penalty)
    result.loss.backward()
    norm = global_gradient_norm(model)
    optimizer.step()
    return result, norm


__all__ = [
    "MinimalSinglePResult",
    "forward_batch",
    "global_gradient_norm",
    "make_optimizer",
    "margin_loss",
    "query_from_weights",
    "training_step",
]
