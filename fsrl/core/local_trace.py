"""Content-addressable persistent trace for directly experienced relations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.infra.runtime import default_device


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


def antisymmetric_conjunctive_key(
    pair_cues: torch.Tensor,
    cue_size: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Bind two normal item cues without using item or relation identifiers."""

    if cue_size < 2:
        raise ValueError("cue_size must be at least two")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if pair_cues.ndim != 2 or pair_cues.shape[1] != 2 * cue_size:
        raise ValueError("pair_cues must contain one left and one right cue")
    left = pair_cues[:, :cue_size]
    right = pair_cues[:, cue_size:]
    conjunction = left[:, :, None] * right[:, None, :]
    conjunction = conjunction - right[:, :, None] * left[:, None, :]
    flat = conjunction.flatten(start_dim=1)
    denominator = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(epsilon)
    return flat / denominator


class ConjunctiveLocalTrace(nn.Module):
    """One-scalar-value antisymmetric tensor-product associative trace."""

    def __init__(
        self,
        cue_size: int,
        *,
        initial_gain: float = 0.1,
        epsilon: float = 1e-8,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if cue_size < 2:
            raise ValueError("cue_size must be at least two")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.cue_size = int(cue_size)
        self.epsilon = float(epsilon)
        execution_device = torch.device(device or default_device())
        self.raw_gain = nn.Parameter(
            torch.tensor(
                [inverse_softplus(initial_gain)],
                dtype=torch.float32,
                device=execution_device,
            )
        )

    @property
    def gain(self) -> torch.Tensor:
        return F.softplus(self.raw_gain)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.cue_size * self.cue_size,
            dtype=self.raw_gain.dtype,
            device=self.raw_gain.device,
        )

    def key(self, pair_cues: torch.Tensor) -> torch.Tensor:
        return antisymmetric_conjunctive_key(pair_cues, self.cue_size, self.epsilon)

    def write(
        self,
        trace: torch.Tensor,
        pair_cues: torch.Tensor,
        encoded_signed_value: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_signed_value.ndim == 1:
            encoded_signed_value = encoded_signed_value[:, None]
        if encoded_signed_value.shape != (trace.shape[0], 1):
            raise ValueError("encoded_signed_value must be one scalar per subject")
        return trace + encoded_signed_value * self.key(pair_cues)

    def read(
        self, trace: torch.Tensor, pair_cues: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_margin = torch.sum(trace * self.key(pair_cues), dim=1, keepdim=True)
        return raw_margin, self.gain * raw_margin

    def forward(
        self,
        logits: torch.Tensor,
        trace: torch.Tensor,
        pair_cues: torch.Tensor,
        gain_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_margin = torch.sum(trace * self.key(pair_cues), dim=1, keepdim=True)
        gain = self.gain.expand_as(raw_margin)
        applied_gain = gain if gain_override is None else gain_override
        correction = applied_gain * raw_margin
        corrected = logits.clone()
        corrected[:, 0] = logits[:, 0] - 0.5 * correction[:, 0]
        corrected[:, 1] = logits[:, 1] + 0.5 * correction[:, 0]
        return corrected, raw_margin, gain, correction
