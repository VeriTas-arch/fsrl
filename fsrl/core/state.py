"""Typed recurrent and episode-level model states."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlasticRNNState:
    hidden: torch.Tensor
    eligibility: torch.Tensor
    fast_weights: torch.Tensor


@dataclass(frozen=True)
class RelationalEpisodeState:
    global_fast_weights: torch.Tensor
    local_trace: torch.Tensor

    def detached_clone(self) -> RelationalEpisodeState:
        return RelationalEpisodeState(
            self.global_fast_weights.detach().clone(),
            self.local_trace.detach().clone(),
        )
