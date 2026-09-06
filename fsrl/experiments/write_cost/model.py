"""Account for accepted effective writes without changing recurrent equations."""

import torch
from torch import nn

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.memory_structure.model import Rollout, objective, read_queries


class WriteSequence(nn.Module):
    """A support trial returns cumulative per-episode modification intensity."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, inputs, hidden, eligibility, weights):
        cost = weights.new_zeros(weights.shape[0])
        for current in inputs.unbind(0):
            _, _, _, hidden, eligibility, updated = self.cell(
                current, hidden, eligibility, weights
            )
            cost = cost + ((updated - weights) * self.cell.alpha).abs().mean((1, 2))
            weights = updated
        return weights, cost


def rollout(backbone, local, support_sequence, query_sequence, batch):
    subjects = batch.support_inputs.shape[2]
    hidden = backbone.initial_hidden(subjects)
    eligibility = backbone.initial_eligibility(subjects)
    weights = backbone.initial_fast_weights(subjects)
    blank = batch.support_inputs.new_zeros(
        2, subjects, backbone.model_config.input_size
    )
    # Preserve the historical initial blank computation; its accepted writes are zero.
    _, _, _, _, _, weights = query_sequence(blank, hidden, eligibility, weights, True)
    state = local.initial_state(subjects)
    writes = []
    first = weights
    for index, inputs in enumerate(batch.support_inputs.unbind(0)):
        weights, amount = support_sequence(inputs, hidden, eligibility, weights)
        writes.append(amount)
        state = local.write(
            state, inputs[0, :, : 2 * local.cue_size], batch.local_evidence[index]
        )
        if index == 0:
            first = weights
    logits, global_logits = read_queries(
        backbone, local, query_sequence, batch, weights, state
    )
    per_trial = torch.stack(writes)
    cost = per_trial.sum(0) / (
        batch.support_inputs.shape[0] * batch.support_inputs.shape[1]
    )
    return Rollout(logits, global_logits, weights, state, first), cost, per_trial


def loss_for(result, cost, batch, spec, coefficient):
    loss, ce = objective(result, batch, spec["optimization"]["fast_weight_penalty"])
    return loss + coefficient * cost.mean(), ce


def update(backbone, local, support, query, batch, optimizer, spec, coefficient):
    optimizer.zero_grad(set_to_none=True)
    result, cost, _ = rollout(backbone, local, support, query, batch)
    loss, ce = loss_for(result, cost, batch, spec, coefficient)
    loss.backward()
    limit = spec["optimization"]["gradient_clip"]
    torch.nn.utils.clip_grad_norm_(
        backbone.parameters(), limit, error_if_nonfinite=True
    )
    torch.nn.utils.clip_grad_norm_([local.raw_gain], limit, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "cost": float(cost.detach().mean()),
    }


def sequences(backbone, *, compiled=False):
    support, query = WriteSequence(backbone), RecurrentSequence(backbone)
    if compiled:
        from fsrl.experiments.training_strategy.execution import PROFILE
        from fsrl.infra.runtime import compile_module

        support = compile_module(support, PROFILE)
        query = compile_module(query, PROFILE)
    return support, query
