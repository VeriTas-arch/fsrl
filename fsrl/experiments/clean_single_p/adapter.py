"""Read-only legacy-shaped evaluation adapter for clean single-margin weights."""

from __future__ import annotations

import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.linear_modulation.model import LinearModulationRNN

from .model import AffineSingleP


def evaluation_adapter(model: AffineSingleP) -> LinearModulationRNN:
    config = model.model_config
    adapter = LinearModulationRNN(
        RetroModelConfig(38, config.hidden_size, 2, 32), device=model.w.device
    )
    with torch.no_grad():
        adapter.i2h.weight.zero_()
        adapter.i2h.weight[:, : config.evidence_index].copy_(
            model.cue_projection.weight
        )
        adapter.i2h.weight[:, 37].copy_(model.evidence_weight)
        if model.time_weight is not None:
            adapter.i2h.weight[:, 32].copy_(model.time_weight)
        adapter.i2h.bias.copy_(model.cue_projection.bias)
        adapter.w.copy_(model.w)
        adapter.alpha.copy_(model.alpha)
        adapter.etaet.copy_(model.etaet)
        adapter.h2DA.weight.copy_(model.h2modulation.weight)
        adapter.h2DA.bias.copy_(model.h2modulation.bias)
        adapter.h2o.weight[0].copy_(-0.5 * model.h2margin.weight[0])
        adapter.h2o.weight[1].copy_(0.5 * model.h2margin.weight[0])
        adapter.h2o.bias[0].copy_(-0.5 * model.h2margin.bias[0])
        adapter.h2o.bias[1].copy_(0.5 * model.h2margin.bias[0])
        adapter.h2v.weight.zero_()
        adapter.h2v.bias.zero_()
    adapter.requires_grad_(False).eval()
    return adapter


__all__ = ["evaluation_adapter"]
