"""The one-factor M2 postsynaptic vector-modulation candidate."""

from __future__ import annotations

import torch
from torch import nn

from fsrl.experiments.minimal_single_p.model import (
    MinimalSingleP,
    MinimalSinglePConfig,
)


class VectorModulatedSingleP(MinimalSingleP):
    """M2 with one instantaneous modulation value per postsynaptic row."""

    def step(self, inputs, hidden, eligibility, fast_weights):
        batch_size, hidden_size = hidden.shape
        next_hidden = torch.tanh(
            self.input_drive(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                self.w + fast_weights, hidden.view(batch_size, hidden_size, 1)
            )
        ).view(batch_size, hidden_size)
        margin = self.h2margin(next_hidden)
        modulation = self.h2modulation(next_hidden)
        next_fast_weights = torch.clamp(
            fast_weights + modulation.view(batch_size, hidden_size, 1) * eligibility,
            min=-50.0,
            max=50.0,
        )
        increment = torch.tanh(
            torch.bmm(
                next_hidden.view(batch_size, hidden_size, 1),
                hidden.view(batch_size, 1, hidden_size),
            )
        )
        next_eligibility = (1.0 - self.etaet) * eligibility + self.etaet * increment
        return margin, modulation, next_hidden, next_eligibility, next_fast_weights


def make_model(
    seed: int, *, device: str | torch.device = "cpu", hidden_size: int = 200
) -> VectorModulatedSingleP:
    torch.manual_seed(seed)
    model = VectorModulatedSingleP(
        MinimalSinglePConfig(level="M2-vector-modulation", hidden_size=hidden_size),
        dense_alpha=False,
        learned_eta=True,
        zero_modulation=True,
        device=device,
    )
    head = nn.Linear(hidden_size, hidden_size, device="meta")
    head.weight = nn.Parameter(model.w.new_zeros(hidden_size, hidden_size))
    head.bias = nn.Parameter(model.w.new_zeros(hidden_size))
    model.h2modulation = head
    if hidden_size == 200 and sum(p.numel() for p in model.parameters()) != 87002:
        raise RuntimeError("M2 vector-modulation parameter count differs")
    return model


__all__ = ["VectorModulatedSingleP", "make_model"]
