"""Checkpoint-preserving factorization of the maintained plastic RNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.infra.runtime import default_device

from .inputs import RelationalInputLayout
from .plastic_rnn import RetroModulRNN


@dataclass(frozen=True)
class FactorizedPlasticRNNConfig:
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


def factorize_legacy_inputs(
    inputs: torch.Tensor, cue_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one active v1 input into task channels and deterministic time."""

    layout = RelationalInputLayout(cue_size)
    layout.validate_width(inputs.shape[-1])
    if torch.count_nonzero(inputs[..., layout.bias_index] - 1.0):
        raise ValueError("active legacy inputs must keep the constant channel at one")
    if torch.count_nonzero(inputs[..., layout.reward_index]):
        raise ValueError("the passive task requires a zero reward channel")
    if torch.count_nonzero(inputs[..., layout.action_start :]):
        raise ValueError("the passive task requires zero previous-action channels")
    external = torch.cat(
        (
            inputs[..., : layout.stimulus_width],
            inputs[..., layout.evidence_index : layout.evidence_index + 1],
        ),
        dim=-1,
    )
    return external, inputs[..., layout.time_index : layout.time_index + 1]


class FactorizedPlasticRNN(nn.Module):
    """Exact active-task form with external time separated from 32 task channels."""

    legacy_constant_weight: torch.Tensor | None
    legacy_input_bias: torch.Tensor | None
    legacy_output_weight: torch.Tensor | None
    legacy_output_bias: torch.Tensor | None

    def __init__(
        self,
        config: FactorizedPlasticRNNConfig,
        *,
        device: str | torch.device | None = None,
        legacy_numerical_compatibility: bool = False,
    ) -> None:
        super().__init__()
        self.model_config = config
        execution_device = torch.device(device or default_device())
        self.input_projection = nn.Linear(config.input_size, config.hidden_size).to(
            execution_device
        )
        self.time_weight = nn.Parameter(
            torch.empty(config.hidden_size, device=execution_device)
        )
        self.w = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        self.alpha = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, device=execution_device)
        )
        self.etaet = nn.Parameter(torch.empty(1, device=execution_device))
        self.modulation_scale = nn.Parameter(torch.empty(1, device=execution_device))
        self.h2modulation = nn.Linear(config.hidden_size, 2).to(execution_device)
        self.h2margin = nn.Linear(config.hidden_size, 1).to(execution_device)
        if legacy_numerical_compatibility:
            self.register_buffer(
                "legacy_constant_weight",
                torch.empty(config.hidden_size, device=execution_device),
            )
            self.register_buffer(
                "legacy_input_bias",
                torch.empty(config.hidden_size, device=execution_device),
            )
            self.register_buffer(
                "legacy_output_weight",
                torch.empty(2, config.hidden_size, device=execution_device),
            )
            self.register_buffer(
                "legacy_output_bias", torch.empty(2, device=execution_device)
            )
        else:
            self.register_buffer("legacy_constant_weight", None)
            self.register_buffer("legacy_input_bias", None)
            self.register_buffer("legacy_output_weight", None)
            self.register_buffer("legacy_output_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        nn.init.uniform_(self.time_weight, -0.01, 0.01)
        nn.init.uniform_(self.w, -0.01, 0.01)
        nn.init.uniform_(self.alpha, -0.01, 0.01)
        nn.init.constant_(self.etaet, 0.7)
        nn.init.ones_(self.modulation_scale)
        self.h2modulation.reset_parameters()
        self.h2margin.reset_parameters()

    @classmethod
    def from_legacy(cls, legacy: RetroModulRNN) -> FactorizedPlasticRNN:
        config = FactorizedPlasticRNNConfig(
            cue_size=legacy.model_config.cue_size,
            hidden_size=legacy.model_config.hidden_size,
        )
        result = cls(
            config,
            device=legacy.w.device,
            legacy_numerical_compatibility=True,
        )
        layout = RelationalInputLayout(config.cue_size)
        with torch.no_grad():
            result.input_projection.weight[:, : layout.stimulus_width].copy_(
                legacy.i2h.weight[:, : layout.stimulus_width]
            )
            result.input_projection.weight[:, config.evidence_index].copy_(
                legacy.i2h.weight[:, layout.evidence_index]
            )
            result.input_projection.bias.copy_(
                legacy.i2h.bias + legacy.i2h.weight[:, layout.bias_index]
            )
            result.time_weight.copy_(legacy.i2h.weight[:, layout.time_index])
            result.w.copy_(legacy.w)
            result.alpha.copy_(legacy.alpha)
            result.etaet.copy_(legacy.etaet)
            result.modulation_scale.copy_(legacy.DAmult)
            result.h2modulation.weight.copy_(legacy.h2DA.weight)
            result.h2modulation.bias.copy_(legacy.h2DA.bias)
            result.h2margin.weight.copy_(
                legacy.h2o.weight[1:2] - legacy.h2o.weight[0:1]
            )
            result.h2margin.bias.copy_(legacy.h2o.bias[1:2] - legacy.h2o.bias[0:1])
            assert result.legacy_constant_weight is not None
            assert result.legacy_input_bias is not None
            assert result.legacy_output_weight is not None
            assert result.legacy_output_bias is not None
            result.legacy_constant_weight.copy_(legacy.i2h.weight[:, layout.bias_index])
            result.legacy_input_bias.copy_(legacy.i2h.bias)
            result.legacy_output_weight.copy_(legacy.h2o.weight)
            result.legacy_output_bias.copy_(legacy.h2o.bias)
        return result

    @property
    def uses_legacy_numerics(self) -> bool:
        buffers = (
            self.legacy_constant_weight,
            self.legacy_input_bias,
            self.legacy_output_weight,
            self.legacy_output_bias,
        )
        if any(value is None for value in buffers):
            if not all(value is None for value in buffers):
                raise RuntimeError(
                    "legacy numerical buffers must be all present or absent"
                )
            return False
        return True

    def _input_drive(
        self, inputs: torch.Tensor, time_values: torch.Tensor
    ) -> torch.Tensor:
        if not self.uses_legacy_numerics:
            return self.input_projection(inputs) + time_values * self.time_weight
        assert self.legacy_constant_weight is not None
        assert self.legacy_input_bias is not None
        batch_size = inputs.shape[0]
        evidence_index = self.model_config.evidence_index
        zeros = inputs.new_zeros(batch_size, 1)
        expanded_inputs = torch.cat(
            (
                inputs[:, :evidence_index],
                torch.ones_like(zeros),
                time_values,
                zeros,
                inputs[:, evidence_index : evidence_index + 1],
                zeros,
                zeros,
            ),
            dim=1,
        )
        hidden_zeros = self.time_weight.new_zeros(self.model_config.hidden_size, 1)
        expanded_weight = torch.cat(
            (
                self.input_projection.weight[:, :evidence_index],
                self.legacy_constant_weight[:, None],
                self.time_weight[:, None],
                hidden_zeros,
                self.input_projection.weight[:, evidence_index : evidence_index + 1],
                hidden_zeros,
                hidden_zeros,
            ),
            dim=1,
        )
        return F.linear(expanded_inputs, expanded_weight, self.legacy_input_bias)

    def _margin(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.uses_legacy_numerics:
            return self.h2margin(hidden)
        assert self.legacy_output_weight is not None
        assert self.legacy_output_bias is not None
        logits = F.linear(hidden, self.legacy_output_weight, self.legacy_output_bias)
        return logits[:, 1:2] - logits[:, 0:1]

    def forward(
        self,
        inputs: torch.Tensor,
        time_values: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        if inputs.shape != (batch_size, self.model_config.input_size):
            raise ValueError("inputs have the wrong shape")
        if time_values.shape != (batch_size, 1):
            raise ValueError("time_values must contain one scalar per batch row")
        if hidden.shape != (batch_size, hidden_size):
            raise ValueError("hidden state has the wrong shape")
        expected_matrix = (batch_size, hidden_size, hidden_size)
        if (
            eligibility.shape != expected_matrix
            or fast_weights.shape != expected_matrix
        ):
            raise ValueError("plastic state has the wrong shape")

        hidden_column = hidden.view(batch_size, hidden_size, 1)
        input_drive = self._input_drive(inputs, time_values)
        next_hidden = torch.tanh(
            input_drive.view(batch_size, hidden_size, 1)
            + torch.matmul(self.w + self.alpha * fast_weights, hidden_column)
        ).view(batch_size, hidden_size)
        margin = self._margin(next_hidden)
        modulation_units = torch.tanh(self.h2modulation(next_hidden))
        modulation = self.modulation_scale * (
            modulation_units[:, 0:1] - modulation_units[:, 1:2]
        )
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


class FactorizedRecurrentSequence(nn.Module):
    """Execute one active support or query trial without writing queries back."""

    def __init__(self, cell: FactorizedPlasticRNN) -> None:
        super().__init__()
        self.cell = cell

    def forward(
        self,
        inputs: torch.Tensor,
        time_values: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        update_fast_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        margin = modulation = None
        for step_inputs, step_times in zip(inputs.unbind(0), time_values.unbind(0)):
            margin, modulation, hidden, eligibility, proposed = self.cell(
                step_inputs, step_times, hidden, eligibility, fast_weights
            )
            if update_fast_weights:
                fast_weights = proposed
        if margin is None or modulation is None:
            raise ValueError("a trial must contain at least one recurrent step")
        return margin, modulation, hidden, eligibility, fast_weights


__all__ = [
    "FactorizedPlasticRNN",
    "FactorizedPlasticRNNConfig",
    "FactorizedRecurrentSequence",
    "factorize_legacy_inputs",
]
