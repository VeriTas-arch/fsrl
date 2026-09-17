"""Read-only legacy-shaped adapter preserving vector row modulation."""

from __future__ import annotations

import torch
from torch import nn

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.linear_modulation.model import LinearModulationRNN

from .model import VectorModulatedSingleP


class VectorModulationRNN(LinearModulationRNN):
    def forward(self, inputs, hidden, eligibility, fast_weights):
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        next_hidden = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                self.w + self.alpha * fast_weights,
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)
        logits, value = self.h2o(next_hidden), self.h2v(next_hidden)
        modulation = self.h2DA(next_hidden)
        proposal = torch.clamp(
            fast_weights + modulation.view(batch_size, hidden_size, 1) * eligibility,
            -50.0,
            50.0,
        )
        increment = torch.tanh(
            torch.bmm(
                next_hidden.view(batch_size, hidden_size, 1),
                hidden.view(batch_size, 1, hidden_size),
            )
        )
        next_eligibility = (1.0 - self.etaet) * eligibility + self.etaet * increment
        return logits, value, modulation, next_hidden, next_eligibility, proposal


def evaluation_adapter(model: VectorModulatedSingleP) -> VectorModulationRNN:
    config = model.model_config
    adapter = VectorModulationRNN(
        RetroModelConfig(38, config.hidden_size, 2, 32), device=model.w.device
    )
    head = nn.Linear(config.hidden_size, config.hidden_size, device="meta")
    head.weight = nn.Parameter(model.h2modulation.weight.detach().clone())
    head.bias = nn.Parameter(model.h2modulation.bias.detach().clone())
    adapter.h2DA = head
    with torch.no_grad():
        adapter.i2h.weight.zero_()
        adapter.i2h.weight[:, : config.evidence_index].copy_(
            model.cue_projection.weight
        )
        adapter.i2h.weight[:, 37].copy_(model.evidence_weight)
        adapter.i2h.bias.copy_(model.cue_projection.bias)
        adapter.w.copy_(model.w)
        adapter.alpha.fill_(1.0)
        adapter.etaet.copy_(model.etaet)
        adapter.h2o.weight[0].copy_(-0.5 * model.h2margin.weight[0])
        adapter.h2o.weight[1].copy_(0.5 * model.h2margin.weight[0])
        adapter.h2o.bias[0].copy_(-0.5 * model.h2margin.bias[0])
        adapter.h2o.bias[1].copy_(0.5 * model.h2margin.bias[0])
        adapter.h2v.weight.zero_()
        adapter.h2v.bias.zero_()
    return adapter.requires_grad_(False).eval()


__all__ = ["VectorModulationRNN", "evaluation_adapter"]
