"""Low-capacity first-order residual correction at the policy margin."""

from __future__ import annotations

import math

import torch
from torch import nn

from fsrl.core.config import DEVICE
from fsrl.core.plastic_rnn import RetroModulRNN


def inverse_sigmoid(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("sigmoid target must lie strictly between zero and one")
    return math.log(value / (1.0 - value))


def policy_residual_statistics(
    baseline: torch.Tensor,
    fast_weight_drive: torch.Tensor,
    output_margin: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return registered exact, linear, and residual policy increments."""

    baseline_hidden = torch.tanh(baseline)
    exact_increment = torch.tanh(baseline + fast_weight_drive) - baseline_hidden
    linear_increment = (1.0 - baseline_hidden.square()) * fast_weight_drive
    hidden_residual = linear_increment - exact_increment
    margin = output_margin.view(1, -1)
    exact_policy = torch.sum(margin * exact_increment, dim=1, keepdim=True)
    linear_policy = torch.sum(margin * linear_increment, dim=1, keepdim=True)
    policy_residual = linear_policy - exact_policy
    residual_norm = torch.linalg.vector_norm(hidden_residual, dim=1, keepdim=True)
    return policy_residual, exact_policy, linear_policy, residual_norm


class PolicyResidualTransition(nn.Module):
    """Restore the first-order residual only in the two-class policy margin."""

    def __init__(
        self,
        backbone: RetroModulRNN,
        *,
        initial_eta: float = 0.5,
    ) -> None:
        super().__init__()
        if backbone.h2o.out_features != 2:
            raise ValueError("policy residual requires exactly two output classes")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.raw_eta = nn.Parameter(
            torch.tensor(
                [inverse_sigmoid(initial_eta)], dtype=torch.float32, device=DEVICE
            )
        )

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def output_margin(self) -> torch.Tensor:
        return self.backbone.h2o.weight[1] - self.backbone.h2o.weight[0]

    def statistics(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = inputs.shape[0]
        hidden_size = self.backbone.GG["hs"]
        hidden_column = hidden.view(batch_size, hidden_size, 1)
        baseline = (
            self.backbone.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(self.backbone.w, hidden_column)
        ).view(batch_size, hidden_size)
        drive = torch.matmul(self.backbone.alpha * fast_weights, hidden_column).view(
            batch_size, hidden_size
        )
        return policy_residual_statistics(baseline, drive, self.output_margin)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        eta_override: torch.Tensor | None = None,
        residual_override: torch.Tensor | None = None,
    ):
        original = self.backbone(inputs, hidden, eligibility, fast_weights)
        policy_residual, _, _, _ = self.statistics(inputs, hidden, fast_weights)
        batch_size = inputs.shape[0]
        natural_eta = self.eta.expand(batch_size, 1)
        applied_eta = natural_eta if eta_override is None else eta_override
        residual_basis = (
            policy_residual if residual_override is None else residual_override
        )
        applied_correction = applied_eta * residual_basis
        correction_logits = torch.cat(
            (-0.5 * applied_correction, 0.5 * applied_correction), dim=1
        )
        corrected_output = original[0] + correction_logits
        if eta_override is not None:
            corrected_output = torch.where(
                eta_override == 0.0, original[0], corrected_output
            )
        return (
            corrected_output,
            *original[1:],
            policy_residual,
            natural_eta,
            applied_correction,
        )
