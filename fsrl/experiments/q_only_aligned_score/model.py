"""Two-scalar q-only normalized delta rule."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.local_trace import inverse_softplus


class QOnlyScore(nn.Module):
    def __init__(self, *, cue_size: int = 15, epsilon: float = 1e-8, device="cpu"):
        super().__init__()
        self.cue_size = cue_size
        self.epsilon = epsilon
        self.raw_eta = nn.Parameter(torch.tensor([0.0], device=device))
        self.raw_gamma = nn.Parameter(
            torch.tensor([inverse_softplus(1.0)], device=device)
        )

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma)

    def forward(
        self,
        support_cues: torch.Tensor,
        realized_q: torch.Tensor,
        query_cues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subjects = support_cues.shape[1]
        w = support_cues.new_zeros(subjects, self.cue_size)
        eta = self.eta
        for cues, q in zip(support_cues.unbind(), realized_q.unbind(), strict=True):
            d = cues[:, : self.cue_size] - cues[:, self.cue_size :]
            error = q - (w * d).sum(-1)
            step = eta * error / (self.epsilon + d.square().sum(-1))
            w = w + step[:, None] * d
        d_query = query_cues[..., : self.cue_size] - query_cues[..., self.cue_size :]
        margins = self.gamma * (w[:, None] * d_query).sum(-1)
        return margins, w


def physical_parameters(model: QOnlyScore) -> dict[str, float]:
    return {"eta": float(model.eta.detach()), "gamma": float(model.gamma.detach())}


def initial_raw_parameters() -> dict[str, float]:
    return {"raw_eta": 0.0, "raw_gamma": float(math.log(math.expm1(1.0)))}


__all__ = ["QOnlyScore", "initial_raw_parameters", "physical_parameters"]
