"""Reusable recurrent sequence execution for compiled training and evaluation."""

from __future__ import annotations

import torch
from torch import nn

from .plastic_rnn import RetroModulRNN


class RecurrentSequence(nn.Module):
    """Execute one complete trial while preserving its fast-weight policy."""

    def __init__(self, cell: RetroModulRNN):
        super().__init__()
        self.cell = cell

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        update_fast_weights: bool,
    ):
        logits = value = dopamine = None
        for inputs in input_sequence.unbind(0):
            (
                logits,
                value,
                dopamine,
                hidden,
                eligibility,
                proposed_fast_weights,
            ) = self.cell(inputs, hidden, eligibility, fast_weights)
            if update_fast_weights:
                fast_weights = proposed_fast_weights
        return logits, value, dopamine, hidden, eligibility, fast_weights
