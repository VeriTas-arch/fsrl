"""Normalized metric-error score state with an optional unchanged local trace."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.local_trace import ConjunctiveLocalTrace, inverse_softplus


class MetricScoreLearner(nn.Module):
    def __init__(
        self,
        cue_size: int,
        *,
        with_local: bool,
        initial_eta: float,
        initial_global_gain: float,
        initial_local_gain: float,
        epsilon: float,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if not 0 < initial_eta < 1 or epsilon <= 0:
            raise ValueError("eta must lie in (0,1) and epsilon must be positive")
        self.cue_size = cue_size
        self.epsilon = epsilon
        self.raw_eta = nn.Parameter(
            torch.tensor(
                [math.log(initial_eta / (1 - initial_eta))],
                device=device,
            )
        )
        self.raw_global_gain = nn.Parameter(
            torch.tensor(
                [inverse_softplus(initial_global_gain)],
                device=device,
            )
        )
        self.local = (
            ConjunctiveLocalTrace(
                cue_size,
                initial_gain=initial_local_gain,
                epsilon=epsilon,
                device=device,
            )
            if with_local
            else None
        )

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def global_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_global_gain)

    def forward(
        self,
        support_cues: torch.Tensor,
        signed: torch.Tensor,
        retention: torch.Tensor,
        local_evidence: torch.Tensor,
        query_cues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Axes: support (trial, subject, cue); queries (subject, query, cue).
        subjects = support_cues.shape[1]
        w = support_cues.new_zeros(subjects, self.cue_size)
        trace = (
            self.local.initial_state(subjects)
            if self.local is not None
            else support_cues.new_empty(subjects, 0)
        )
        eta = self.eta
        for t, cues in enumerate(support_cues.unbind()):
            x = cues[:, : self.cue_size] - cues[:, self.cue_size :]
            error = signed[t] - (w * x).sum(dim=-1)
            update = eta * retention[t] * error / (self.epsilon + x.square().sum(-1))
            w = w + update[:, None] * x
            if self.local is not None:
                trace = self.local.write(trace, cues, local_evidence[t])
        q = query_cues[..., : self.cue_size] - query_cues[..., self.cue_size :]
        global_margin = self.global_gain * (w[:, None] * q).sum(dim=-1)
        local_margin = torch.zeros_like(global_margin)
        if self.local is not None:
            keys = self.local.key(query_cues.reshape(-1, 2 * self.cue_size))
            keys = keys.reshape(subjects, query_cues.shape[1], -1)
            local_margin = self.local.gain * (trace[:, None] * keys).sum(-1)
        return global_margin + local_margin, global_margin, local_margin, w, trace


def make_model(condition: str, spec: dict, device: str = "cpu") -> MetricScoreLearner:
    if condition not in spec["seeds"]["conditions"]:
        raise ValueError("unregistered minimal learner condition")
    settings = spec["optimization"]
    return MetricScoreLearner(
        spec["task"]["cue_size"],
        with_local=condition == "score_trace",
        initial_eta=settings["initial_eta"],
        initial_global_gain=settings["initial_global_gain"],
        initial_local_gain=settings["initial_local_gain"],
        epsilon=spec["model"]["epsilon"],
        device=device,
    )
