"""Replace only support modulation; preserve old eligibility and query computation."""

import torch
from torch import nn

from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.infra.runtime import compile_module


class ScheduledSupport(nn.Module):
    def __init__(self, cell, values):
        super().__init__()
        self.cell = cell
        self.values = values

    def forward(self, inputs, hidden, eligibility, weights, uniforms):
        cost = weights.new_zeros(weights.shape[0])
        for phase, current in enumerate(inputs.unbind(0)):
            _, _, _, hidden, updated_e, proposal = self.cell(
                current, hidden, eligibility, weights
            )
            if self.values is not None:
                proposal = torch.clamp(
                    weights + self.values[phase] * eligibility, -50, 50
                )
            cost = cost + ((proposal - weights) * self.cell.alpha).abs().mean((1, 2))
            eligibility, weights = updated_e, proposal
        return weights, cost


def scheduled_sequences(net, original, values, *, compiled=True):
    support = ScheduledSupport(net, values)
    if compiled:
        support = compile_module(support, PROFILE)
    return support, original[1]
