"""Policy-relative signed-curvature query expression gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import DEVICE
from .curvature_gate import inverse_softplus
from .model import RetroModulRNN


def policy_opposition_statistics(
    baseline: torch.Tensor,
    fast_weight_drive: torch.Tensor,
    output_margin: torch.Tensor,
    *,
    tau: float,
    epsilon: float,
) -> tuple[torch.Tensor, ...]:
    """Return the two sign-matched risks and their registered online terms."""

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    baseline_hidden = torch.tanh(baseline)
    derivative = 1.0 - baseline_hidden.square()
    jacobian_drive = derivative * fast_weight_drive
    quadratic = -baseline_hidden * derivative * fast_weight_drive.square()
    margin = output_margin.view(1, -1)
    first_order_value = torch.sum(margin * jacobian_drive, dim=1, keepdim=True)
    quadratic_value = torch.sum(margin * quadratic, dim=1, keepdim=True)
    scale_squared = torch.sum(output_margin.square()) * torch.sum(
        jacobian_drive.square(), dim=1, keepdim=True
    )
    denominator = first_order_value.square() + tau * scale_squared + epsilon
    signed_product = first_order_value * quadratic_value
    opposition_risk = torch.relu(-signed_product) / denominator
    support_risk = torch.relu(signed_product) / denominator
    return (
        opposition_risk,
        support_risk,
        first_order_value,
        quadratic_value,
        scale_squared,
        denominator,
    )


class PolicyOppositionGateTransition(nn.Module):
    """Attenuate only curvature opposing first-order output-margin value."""

    def __init__(
        self,
        backbone: RetroModulRNN,
        *,
        tau: float = 0.01,
        epsilon: float = 1e-8,
        initial_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if backbone.h2o.out_features != 2:
            raise ValueError("policy margin requires exactly two output classes")
        self.backbone = backbone
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.raw_beta = nn.Parameter(
            torch.tensor(
                [inverse_softplus(initial_beta)], dtype=torch.float32, device=DEVICE
            )
        )

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

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
        return policy_opposition_statistics(
            baseline,
            drive,
            self.output_margin,
            tau=self.tau,
            epsilon=self.epsilon,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        gamma_override: torch.Tensor | None = None,
        use_support_risk: bool = False,
    ):
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
        opposition_risk, support_risk, _, _, _, _ = policy_opposition_statistics(
            baseline,
            drive,
            self.output_margin,
            tau=self.tau,
            epsilon=self.epsilon,
        )
        risk = support_risk if use_support_risk else opposition_risk
        conditioned_gamma = 1.0 / (1.0 + self.beta * risk)
        applied_gamma = conditioned_gamma if gamma_override is None else gamma_override
        preactivation = baseline + applied_gamma * drive
        if gamma_override is not None:
            original_preactivation = (
                self.backbone.i2h(inputs).view(batch_size, hidden_size, 1)
                + torch.matmul(
                    self.backbone.w + self.backbone.alpha * fast_weights,
                    hidden_column,
                )
            ).view(batch_size, hidden_size)
            preactivation = torch.where(
                gamma_override == 1.0, original_preactivation, preactivation
            )
        hidden_activation = torch.tanh(preactivation)

        output = self.backbone.h2o(hidden_activation)
        value = self.backbone.h2v(hidden_activation)
        da_pair = torch.tanh(self.backbone.h2DA(hidden_activation))
        da = self.backbone.DAmult * (da_pair[:, 0] - da_pair[:, 1])[:, None]
        proposed_fast_weights = fast_weights + da.view(batch_size, 1, 1) * eligibility
        proposed_fast_weights = torch.clip(proposed_fast_weights, min=-50.0, max=50.0)
        delta_eligibility = torch.bmm(
            hidden_activation.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        delta_eligibility = torch.tanh(delta_eligibility)
        proposed_eligibility = (
            1.0 - self.backbone.etaet
        ) * eligibility + self.backbone.etaet * delta_eligibility
        return (
            output,
            value,
            da,
            hidden_activation,
            proposed_eligibility,
            proposed_fast_weights,
            risk,
            conditioned_gamma,
            applied_gamma,
        )
