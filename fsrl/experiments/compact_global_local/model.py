"""Compact recurrent P state and exact packed direct-relation L state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fsrl.infra.runtime import default_device


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class CompactModelConfig:
    cue_size: int = 15
    hidden_size: int = 200

    def __post_init__(self) -> None:
        if self.cue_size < 2:
            raise ValueError("cue_size must be at least two")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")

    @property
    def pair_cue_width(self) -> int:
        return 2 * self.cue_size

    @property
    def response_index(self) -> int:
        return self.pair_cue_width

    @property
    def evidence_index(self) -> int:
        return self.pair_cue_width + 1

    @property
    def input_size(self) -> int:
        return self.pair_cue_width + 2


class CompactPlasticRNN(nn.Module):
    """Two-step recurrent cell with one margin and one modulation output."""

    def __init__(
        self,
        config: CompactModelConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model_config = config
        execution_device = torch.device(device or default_device())
        self.i2h = nn.Linear(config.input_size, config.hidden_size).to(execution_device)
        self.w = nn.Parameter(
            (
                (1.0 / np.sqrt(config.hidden_size))
                * (2.0 * torch.rand(config.hidden_size, config.hidden_size) - 1.0)
            ).to(execution_device)
        )
        self.alpha = nn.Parameter(
            (
                0.01 * (2.0 * torch.rand(config.hidden_size, config.hidden_size) - 1.0)
            ).to(execution_device)
        )
        self.etaet = nn.Parameter(torch.tensor([0.7], device=execution_device))
        self.modulation_scale = nn.Parameter(
            torch.tensor([1.0], device=execution_device)
        )
        self.h2modulation = nn.Linear(config.hidden_size, 1).to(execution_device)
        self.h2margin = nn.Linear(config.hidden_size, 1).to(execution_device)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        if hidden.shape != (batch_size, hidden_size):
            raise ValueError("hidden state has the wrong shape")
        expected_matrix = (batch_size, hidden_size, hidden_size)
        if (
            eligibility.shape != expected_matrix
            or fast_weights.shape != expected_matrix
        ):
            raise ValueError("plastic state has the wrong shape")

        hidden_column = hidden.view(batch_size, hidden_size, 1)
        next_hidden = torch.tanh(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(self.w + self.alpha * fast_weights, hidden_column)
        ).view(batch_size, hidden_size)
        margin = self.h2margin(next_hidden)
        modulation = self.modulation_scale * torch.tanh(self.h2modulation(next_hidden))
        next_fast_weights = torch.clamp(
            fast_weights + modulation.view(batch_size, 1, 1) * eligibility,
            min=-50.0,
            max=50.0,
        )
        eligibility_increment = torch.tanh(
            torch.bmm(
                next_hidden.view(batch_size, hidden_size, 1),
                hidden.view(batch_size, 1, hidden_size),
            )
        )
        next_eligibility = (
            1.0 - self.etaet
        ) * eligibility + self.etaet * eligibility_increment
        return (
            margin,
            modulation,
            next_hidden,
            next_eligibility,
            next_fast_weights,
        )

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(batch_size, self.model_config.hidden_size)

    def initial_eligibility(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(
            batch_size,
            self.model_config.hidden_size,
            self.model_config.hidden_size,
        )

    def initial_fast_weights(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(
            batch_size,
            self.model_config.hidden_size,
            self.model_config.hidden_size,
        )


class CompactRecurrentSequence(nn.Module):
    """Execute one complete two-step trial."""

    def __init__(self, cell: CompactPlasticRNN) -> None:
        super().__init__()
        self.cell = cell

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        update_fast_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        margin = modulation = None
        for step_inputs in inputs.unbind(0):
            margin, modulation, hidden, eligibility, proposed = self.cell(
                step_inputs, hidden, eligibility, fast_weights
            )
            if update_fast_weights:
                fast_weights = proposed
        if margin is None or modulation is None:
            raise ValueError("a trial must contain at least one recurrent step")
        return margin, modulation, hidden, eligibility, fast_weights


class PackedLocalTrace(nn.Module):
    """Strict-upper-triangle form of the normalized antisymmetric local trace."""

    upper_rows: torch.Tensor
    upper_columns: torch.Tensor

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
        upper = torch.triu_indices(
            cue_size, cue_size, offset=1, device=execution_device
        )
        self.register_buffer("upper_rows", upper[0], persistent=False)
        self.register_buffer("upper_columns", upper[1], persistent=False)
        self.raw_gain = nn.Parameter(
            torch.tensor(
                [_inverse_softplus(initial_gain)],
                dtype=torch.float32,
                device=execution_device,
            )
        )

    @property
    def state_size(self) -> int:
        return self.cue_size * (self.cue_size - 1) // 2

    @property
    def gain(self) -> torch.Tensor:
        return F.softplus(self.raw_gain)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.raw_gain.new_zeros(batch_size, self.state_size)

    def key(self, pair_cues: torch.Tensor) -> torch.Tensor:
        if pair_cues.ndim != 2 or pair_cues.shape[1] != 2 * self.cue_size:
            raise ValueError("pair_cues must contain one left and one right cue")
        left = pair_cues[:, : self.cue_size]
        right = pair_cues[:, self.cue_size :]
        upper = (
            left[:, self.upper_rows] * right[:, self.upper_columns]
            - right[:, self.upper_rows] * left[:, self.upper_columns]
        )
        denominator = torch.linalg.vector_norm(upper, dim=1, keepdim=True).clamp_min(
            self.epsilon
        )
        return upper / denominator

    def write(
        self,
        state: torch.Tensor,
        pair_cues: torch.Tensor,
        encoded_signed_value: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_signed_value.ndim == 1:
            encoded_signed_value = encoded_signed_value[:, None]
        if encoded_signed_value.shape != (state.shape[0], 1):
            raise ValueError("encoded_signed_value must be one scalar per subject")
        return state + encoded_signed_value * self.key(pair_cues)

    def read(self, state: torch.Tensor, pair_cues: torch.Tensor) -> torch.Tensor:
        return torch.sum(state * self.key(pair_cues), dim=1, keepdim=True)

    def forward(
        self,
        global_margin: torch.Tensor,
        state: torch.Tensor,
        pair_cues: torch.Tensor,
        *,
        active: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.read(state, pair_cues)
        correction = self.gain * raw if active else torch.zeros_like(raw)
        return global_margin + correction, raw, correction
