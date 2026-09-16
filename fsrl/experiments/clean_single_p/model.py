"""Clean 32-channel single-P cells with affine modulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.inputs import RelationalInputLayout
from fsrl.experiments.linear_modulation.model import LinearModulationRNN


@dataclass(frozen=True)
class CleanSinglePConfig:
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


class AffineSingleP(nn.Module):
    """One dense fast weight with one affine write-modulation scalar."""

    def __init__(
        self,
        config: CleanSinglePConfig,
        *,
        retain_time: bool,
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
        if retain_time:
            self.time_weight = nn.Parameter(
                torch.empty(config.hidden_size, device=execution_device)
            )
        else:
            self.register_parameter("time_weight", None)
        self.w = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        self.alpha = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        self.etaet = nn.Parameter(torch.empty(1, device=execution_device))
        self.h2modulation = nn.Linear(config.hidden_size, 1, device=execution_device)
        self.h2margin = nn.Linear(config.hidden_size, 1, device=execution_device)

    @property
    def retains_time(self) -> bool:
        return self.time_weight is not None

    def input_drive(
        self, inputs: torch.Tensor, time_values: torch.Tensor | None
    ) -> torch.Tensor:
        config = self.model_config
        if inputs.shape[-1] != config.input_size:
            raise ValueError("clean single-P inputs have the wrong width")
        drive = self.cue_projection(inputs[:, : config.evidence_index])
        drive = drive + inputs[:, config.evidence_index : config.input_size] * (
            self.evidence_weight
        )
        if self.retains_time:
            if time_values is None or time_values.shape != (inputs.shape[0], 1):
                raise ValueError("the time control requires one scalar per input")
            assert self.time_weight is not None
            drive = drive + time_values * self.time_weight
        elif time_values is not None:
            raise ValueError("clean_no_time does not accept time")
        return drive

    def step(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        *,
        time_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, hidden_size = hidden.shape
        hidden_column = hidden.view(batch_size, hidden_size, 1)
        next_hidden = torch.tanh(
            self.input_drive(inputs, time_values).view(batch_size, hidden_size, 1)
            + torch.matmul(self.w + self.alpha * fast_weights, hidden_column)
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
        next_eligibility = (1.0 - self.etaet) * eligibility + self.etaet * increment
        return margin, modulation, next_hidden, next_eligibility, next_fast_weights

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(batch_size, self.model_config.hidden_size)

    def initial_eligibility(self, batch_size: int) -> torch.Tensor:
        size = self.model_config.hidden_size
        return self.w.new_zeros(batch_size, size, size)

    def initial_fast_weights(self, batch_size: int) -> torch.Tensor:
        size = self.model_config.hidden_size
        return self.w.new_zeros(batch_size, size, size)


class AffineSinglePSequence(nn.Module):
    def __init__(self, cell: AffineSingleP) -> None:
        super().__init__()
        self.cell = cell

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        update_fast_weights: bool,
        time_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        margin = modulation = None
        time_steps = (
            (None,) * len(inputs)
            if time_values is None
            else tuple(time_values.unbind(0))
        )
        for step_inputs, step_time in zip(inputs.unbind(0), time_steps, strict=True):
            margin, modulation, hidden, eligibility, proposal = self.cell.step(
                step_inputs,
                hidden,
                eligibility,
                fast_weights,
                time_values=step_time,
            )
            if update_fast_weights:
                fast_weights = proposal
        if margin is None or modulation is None:
            raise ValueError("a trial must contain at least one recurrent step")
        return margin, modulation, hidden, eligibility, fast_weights


def map_shadow(shadow: LinearModulationRNN, condition: str) -> AffineSingleP:
    retain_time = condition == "time_retained_control"
    if not retain_time and condition != "clean_no_time":
        raise ValueError(f"unknown clean single-P condition: {condition}")
    config = CleanSinglePConfig(
        cue_size=(shadow.model_config.input_size - 8) // 2,
        hidden_size=shadow.model_config.hidden_size,
    )
    layout = RelationalInputLayout(config.cue_size)
    target = AffineSingleP(config, retain_time=retain_time, device=shadow.w.device)
    with torch.no_grad():
        target.cue_projection.weight.copy_(
            shadow.i2h.weight[:, : config.evidence_index]
        )
        target.cue_projection.bias.copy_(
            shadow.i2h.bias + shadow.i2h.weight[:, layout.bias_index]
        )
        target.evidence_weight.copy_(
            shadow.i2h.weight[:, layout.evidence_index]
            + shadow.i2h.weight[:, shadow.model_config.input_size - 1]
        )
        if target.time_weight is not None:
            target.time_weight.copy_(shadow.i2h.weight[:, layout.time_index])
        target.w.copy_(shadow.w)
        target.alpha.copy_(shadow.alpha)
        target.etaet.copy_(shadow.etaet)
        target.h2modulation.weight.copy_(shadow.h2DA.weight)
        target.h2modulation.bias.copy_(shadow.h2DA.bias)
        target.h2margin.weight.copy_(shadow.h2o.weight[1:2] - shadow.h2o.weight[0:1])
        target.h2margin.bias.copy_(shadow.h2o.bias[1:2] - shadow.h2o.bias[0:1])
    return target


def expand_shadow_inputs(
    inputs: torch.Tensor, time_values: torch.Tensor, cue_size: int
) -> torch.Tensor:
    config = CleanSinglePConfig(cue_size=cue_size)
    layout = RelationalInputLayout(cue_size)
    legacy = inputs.new_zeros(*inputs.shape[:-1], layout.input_size + 1)
    legacy[..., : config.evidence_index] = inputs[..., : config.evidence_index]
    legacy[..., layout.bias_index] = 1.0
    legacy[..., layout.time_index : layout.time_index + 1] = time_values
    q = inputs[..., config.evidence_index]
    legacy[..., layout.evidence_index] = q
    legacy[..., -1] = q
    return legacy


def margin_loss(margins: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    signs = 2.0 * targets.to(margins.dtype).reshape_as(margins) - 1.0
    return F.softplus(-signs * margins).mean()


def shared_parameters(model: AffineSingleP) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in model.named_parameters() if name != "time_weight"
    }


__all__ = [
    "AffineSingleP",
    "AffineSinglePSequence",
    "CleanSinglePConfig",
    "expand_shadow_inputs",
    "map_shadow",
    "margin_loss",
    "shared_parameters",
]
