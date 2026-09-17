"""Directly initialized cells for the registered minimal single-P ladder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .protocol import specification, validate_level


@dataclass(frozen=True)
class MinimalSinglePConfig:
    level: str
    cue_size: int = 15
    hidden_size: int = 200

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


class MinimalSingleP(nn.Module):
    """One recurrent state, one persistent P, and one scalar write modulator."""

    def __init__(
        self,
        config: MinimalSinglePConfig,
        *,
        dense_alpha: bool,
        learned_eta: bool,
        zero_modulation: bool,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model_config = config
        execution_device = torch.device(device)
        self.cue_projection = nn.Linear(
            config.evidence_index, config.hidden_size, device=execution_device
        )
        self.evidence_weight = nn.Parameter(
            torch.empty(config.hidden_size, device=execution_device)
        )
        self.w = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        # Construct the full canonical draw before deleting absent components. This
        # keeps shared H=200 tensors paired across the registered ladder.
        self.alpha = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        self.etaet = nn.Parameter(torch.empty(1, device=execution_device))
        self.h2modulation = nn.Linear(config.hidden_size, 1, device=execution_device)
        self.h2margin = nn.Linear(config.hidden_size, 1, device=execution_device)
        self.reset_direct_parameters()
        if not dense_alpha:
            del self.alpha
            self.register_parameter("alpha", None)
        if not learned_eta:
            del self.etaet
            self.register_parameter("etaet", None)
        if zero_modulation:
            with torch.no_grad():
                self.h2modulation.weight.zero_()
                assert self.h2modulation.bias is not None
                self.h2modulation.bias.zero_()

    def reset_direct_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.model_config.evidence_index)
        recurrent_bound = 1.0 / math.sqrt(self.model_config.hidden_size)
        with torch.no_grad():
            self.evidence_weight.uniform_(-bound, bound)
            self.w.uniform_(-recurrent_bound, recurrent_bound)
            self.alpha.uniform_(-0.01, 0.01)
            self.etaet.fill_(0.7)

    @property
    def has_dense_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def has_learned_eta(self) -> bool:
        return self.etaet is not None

    def input_drive(self, inputs: torch.Tensor) -> torch.Tensor:
        config = self.model_config
        if inputs.shape[-1] != config.input_size:
            raise ValueError("minimal single-P inputs have the wrong width")
        return (
            self.cue_projection(inputs[:, : config.evidence_index])
            + inputs[:, config.evidence_index : config.input_size]
            * self.evidence_weight
        )

    def effective_fast_weights(self, fast_weights: torch.Tensor) -> torch.Tensor:
        return self.alpha * fast_weights if self.alpha is not None else fast_weights

    def step(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, hidden_size = hidden.shape
        hidden_column = hidden.view(batch_size, hidden_size, 1)
        next_hidden = torch.tanh(
            self.input_drive(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                self.w + self.effective_fast_weights(fast_weights), hidden_column
            )
        ).view(batch_size, hidden_size)
        margin = self.h2margin(next_hidden)
        modulation = self.h2modulation(next_hidden)
        next_fast_weights = torch.clamp(
            fast_weights + modulation.view(batch_size, 1, 1) * eligibility,
            min=-50.0,
            max=50.0,
        )
        increment = torch.tanh(
            torch.bmm(
                next_hidden.view(batch_size, hidden_size, 1),
                hidden.view(batch_size, 1, hidden_size),
            )
        )
        next_eligibility = (
            increment
            if self.etaet is None
            else (1.0 - self.etaet) * eligibility + self.etaet * increment
        )
        return margin, modulation, next_hidden, next_eligibility, next_fast_weights

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(batch_size, self.model_config.hidden_size)

    def initial_eligibility(self, batch_size: int) -> torch.Tensor:
        size = self.model_config.hidden_size
        return self.w.new_zeros(batch_size, size, size)

    def initial_fast_weights(self, batch_size: int) -> torch.Tensor:
        size = self.model_config.hidden_size
        return self.w.new_zeros(batch_size, size, size)


class MinimalSinglePSequence(nn.Module):
    def __init__(self, cell: MinimalSingleP) -> None:
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
            margin, modulation, hidden, eligibility, proposal = self.cell.step(
                step_inputs, hidden, eligibility, fast_weights
            )
            if update_fast_weights:
                fast_weights = proposal
        if margin is None or modulation is None:
            raise ValueError("a trial must contain at least one recurrent step")
        return margin, modulation, hidden, eligibility, fast_weights


def level_settings(level: str) -> dict:
    validate_level(level)
    architecture = specification()["architecture"]["levels"][level]
    return {
        "hidden_size": int(architecture["hidden_size"]),
        "dense_alpha": level in {"C0", "M1"},
        "learned_eta": level in {"C0", "M1", "M2"},
        "zero_modulation": level not in {"C0", "M1"},
        "parameter_count": int(architecture["parameter_count"]),
    }


def make_model(
    level: str,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    hidden_size: int | None = None,
) -> MinimalSingleP:
    settings = level_settings(level)
    size = settings["hidden_size"] if hidden_size is None else hidden_size
    torch.manual_seed(seed)
    model = MinimalSingleP(
        MinimalSinglePConfig(level=level, hidden_size=size),
        dense_alpha=settings["dense_alpha"],
        learned_eta=settings["learned_eta"],
        zero_modulation=settings["zero_modulation"],
        device=device,
    )
    if hidden_size is None:
        observed = sum(parameter.numel() for parameter in model.parameters())
        if observed != settings["parameter_count"]:
            raise RuntimeError(
                f"minimal single-P parameter count differs for {level}: {observed}"
            )
    return model


__all__ = [
    "MinimalSingleP",
    "MinimalSinglePConfig",
    "MinimalSinglePSequence",
    "level_settings",
    "make_model",
]
