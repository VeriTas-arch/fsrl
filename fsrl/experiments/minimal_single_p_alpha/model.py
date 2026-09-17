"""The one-factor M2-alpha candidate."""

from __future__ import annotations

import torch

from fsrl.experiments.minimal_single_p.model import (
    MinimalSingleP,
    MinimalSinglePConfig,
)


def make_model(seed: int, *, device="cpu", hidden_size: int = 200) -> MinimalSingleP:
    torch.manual_seed(seed)
    model = MinimalSingleP(
        MinimalSinglePConfig(level="M2-alpha", hidden_size=hidden_size),
        dense_alpha=True,
        learned_eta=True,
        zero_modulation=True,
        device=device,
    )
    with torch.no_grad():
        model.alpha.fill_(1.0)
    if hidden_size == 200 and sum(p.numel() for p in model.parameters()) != 87003:
        raise RuntimeError("M2-alpha parameter count differs")
    return model


__all__ = ["make_model"]
