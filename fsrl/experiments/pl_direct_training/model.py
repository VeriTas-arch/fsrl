"""Fresh clean time-control and structurally time-free P backbones."""

from __future__ import annotations

import torch

from fsrl.core.factorized_plastic_rnn import (
    FactorizedPlasticRNN,
    FactorizedPlasticRNNConfig,
)
from fsrl.core.inputs import RelationalInputLayout
from fsrl.core.plastic_rnn import RetroModulRNN


class NoTimePlasticRNN(FactorizedPlasticRNN):
    """The clean active-task cell with no time parameter or time argument."""

    def __init__(
        self,
        config: FactorizedPlasticRNNConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            legacy_numerical_compatibility=False,
        )
        del self.time_weight

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        if inputs.shape != (batch_size, self.model_config.input_size):
            raise ValueError("inputs have the wrong shape")
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
            self.input_projection(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(self.w + self.alpha * fast_weights, hidden_column)
        ).view(batch_size, hidden_size)
        margin = self.h2margin(next_hidden)
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


class NoTimeRecurrentSequence(torch.nn.Module):
    """Execute one trial without accepting time metadata."""

    def __init__(self, cell: NoTimePlasticRNN) -> None:
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


def _copy_shadow_parameters(
    target: FactorizedPlasticRNN, shadow: RetroModulRNN
) -> None:
    config = target.model_config
    layout = RelationalInputLayout(config.cue_size)
    with torch.no_grad():
        target.input_projection.weight[:, : layout.stimulus_width].copy_(
            shadow.i2h.weight[:, : layout.stimulus_width]
        )
        target.input_projection.weight[:, config.evidence_index].copy_(
            shadow.i2h.weight[:, layout.evidence_index]
        )
        target.input_projection.bias.copy_(
            shadow.i2h.bias + shadow.i2h.weight[:, layout.bias_index]
        )
        target.w.copy_(shadow.w)
        target.alpha.copy_(shadow.alpha)
        target.etaet.copy_(shadow.etaet)
        target.modulation_scale.copy_(shadow.DAmult)
        target.h2modulation.weight.copy_(shadow.h2DA.weight)
        target.h2modulation.bias.copy_(shadow.h2DA.bias)
        target.h2margin.weight.copy_(shadow.h2o.weight[1:2] - shadow.h2o.weight[0:1])
        target.h2margin.bias.copy_(shadow.h2o.bias[1:2] - shadow.h2o.bias[0:1])


def map_shadow_model(
    shadow: RetroModulRNN, condition: str
) -> FactorizedPlasticRNN | NoTimePlasticRNN:
    config = FactorizedPlasticRNNConfig(
        cue_size=shadow.model_config.cue_size,
        hidden_size=shadow.model_config.hidden_size,
    )
    if condition == "time_retained_control":
        target: FactorizedPlasticRNN | NoTimePlasticRNN = FactorizedPlasticRNN(
            config,
            device=shadow.w.device,
            legacy_numerical_compatibility=False,
        )
        with torch.no_grad():
            target.time_weight.copy_(
                shadow.i2h.weight[:, RelationalInputLayout(config.cue_size).time_index]
            )
    elif condition == "no_time_candidate":
        target = NoTimePlasticRNN(config, device=shadow.w.device)
    else:
        raise ValueError(f"unknown direct-training condition: {condition}")
    _copy_shadow_parameters(target, shadow)
    if target.uses_legacy_numerics:
        raise RuntimeError("fresh direct models must not retain compatibility buffers")
    return target


def shared_parameter_tensors(
    model: FactorizedPlasticRNN | NoTimePlasticRNN,
) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in model.named_parameters() if name != "time_weight"
    }


__all__ = [
    "NoTimePlasticRNN",
    "NoTimeRecurrentSequence",
    "map_shadow_model",
    "shared_parameter_tensors",
]
