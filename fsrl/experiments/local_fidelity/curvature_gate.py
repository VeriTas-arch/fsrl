"""Low-capacity curvature-conditioned query expression gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.config import NUMRESPONSESTEP, TrainConfig
from fsrl.core.local_trace import inverse_softplus
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.tasks.holdouts import registered_holdout_signatures
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator, RankingEpisode
from fsrl.training.backbone import MetaTrainConfig, build_meta_input_sequence


def curvature_risk(
    baseline: torch.Tensor, fast_weight_drive: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    baseline_hidden = torch.tanh(baseline)
    derivative = 1.0 - baseline_hidden.square()
    jacobian_drive = derivative * fast_weight_drive
    quadratic = -baseline_hidden * derivative * fast_weight_drive.square()
    return torch.linalg.vector_norm(quadratic, dim=1, keepdim=True) / (
        torch.linalg.vector_norm(jacobian_drive, dim=1, keepdim=True) + epsilon
    )


class CurvatureGateTransition(nn.Module):
    """Apply the registered gate to one query response transition."""

    def __init__(
        self,
        backbone: RetroModulRNN,
        *,
        epsilon: float = 1e-8,
        initial_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.backbone = backbone
        self.epsilon = float(epsilon)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        device = next(self.backbone.parameters()).device
        self.raw_beta = nn.Parameter(
            torch.tensor(
                [inverse_softplus(initial_beta)], dtype=torch.float32, device=device
            )
        )

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        gamma_override: torch.Tensor | None = None,
    ):
        batch_size = inputs.shape[0]
        hidden_size = self.backbone.model_config.hidden_size
        hidden_column = hidden.view(batch_size, hidden_size, 1)
        baseline = (
            self.backbone.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(self.backbone.w, hidden_column)
        ).view(batch_size, hidden_size)
        drive = torch.matmul(self.backbone.alpha * fast_weights, hidden_column).view(
            batch_size, hidden_size
        )
        risk = curvature_risk(baseline, drive, self.epsilon)
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


@dataclass(frozen=True)
class GateBatchStats:
    loss: torch.Tensor
    query_cross_entropy: float
    query_accuracy: float
    gamma_sum: float
    gamma_count: int
    risk_sum: float


def run_gate_batch(
    training_config: MetaTrainConfig,
    model_config: TrainConfig,
    backbone,
    gate,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
) -> GateBatchStats:
    """Run one generic batch while keeping support and fast-weight formation v1."""

    n_edges = int(
        rng.integers(training_config.min_edges, training_config.max_edges + 1)
    )
    episodes: tuple[RankingEpisode, ...] = tuple(
        task_generator.sample(rng, n_edges=n_edges)
        for _ in range(training_config.batch_size)
    )
    device = next(backbone.parameters()).device
    hidden = backbone.initial_hidden(model_config.bs)
    eligibility = backbone.initial_eligibility(model_config.bs)
    fast_weights = backbone.initial_fast_weights(model_config.bs)
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=device)
    for _ in range(2):
        _, _, _, hidden, eligibility, fast_weights = backbone(
            blank, hidden, eligibility, fast_weights
        )

    n_support = len(episodes[0].support_trials)
    zero_hidden = backbone.initial_hidden(model_config.bs)
    zero_eligibility = backbone.initial_eligibility(model_config.bs)
    for trial_index in range(n_support):
        hidden = zero_hidden
        eligibility = zero_eligibility
        trials = [episode.support_trials[trial_index] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [trial.signed_magnitude * trial.encoding_reliability for trial in trials],
            dtype=np.float32,
        )
        time_value = (
            trial_index / max(1, n_support - 1) * training_config.support_query_time
        )
        input_sequence = build_meta_input_sequence(
            model_config,
            episodes,
            left,
            right,
            signed,
            num_steps=model_config.triallen,
            time_value=time_value,
            support_trial=True,
            device=device,
        )
        for inputs in input_sequence.unbind():
            _, _, _, hidden, eligibility, fast_weights = backbone(
                inputs, hidden, eligibility, fast_weights
            )

    query_loss = torch.zeros((), device=device)
    correct = 0
    total = 0
    gamma_sum = torch.zeros((), device=device)
    risk_sum = torch.zeros((), device=device)
    gamma_count = 0
    n_queries = len(episodes[0].query_trials)
    for query_index in range(n_queries):
        hidden = zero_hidden
        eligibility = zero_eligibility
        trials = [episode.query_trials[query_index] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        targets = torch.tensor(
            [trial.correct_action for trial in trials], dtype=torch.long, device=device
        )
        signed = np.zeros(model_config.bs, dtype=np.float32)
        input_sequence = build_meta_input_sequence(
            model_config,
            episodes,
            left,
            right,
            signed,
            num_steps=NUMRESPONSESTEP + 1,
            time_value=training_config.support_query_time,
            support_trial=False,
            device=device,
        )
        step0, response = input_sequence.unbind()
        _, _, _, hidden, eligibility, _proposed = backbone(
            step0, hidden, eligibility, fast_weights
        )
        (
            response_logits,
            _,
            _,
            _,
            _,
            _,
            risk,
            conditioned_gamma,
            _,
        ) = gate(response, hidden, eligibility, fast_weights)
        query_loss = query_loss + F.cross_entropy(response_logits, targets)
        correct += int(torch.sum(torch.argmax(response_logits, dim=1) == targets))
        total += model_config.bs
        gamma_sum = gamma_sum + torch.sum(conditioned_gamma)
        risk_sum = risk_sum + torch.sum(risk)
        gamma_count += model_config.bs

    query_loss = query_loss / n_queries
    return GateBatchStats(
        loss=query_loss,
        query_cross_entropy=float(query_loss.detach()),
        query_accuracy=correct / total,
        gamma_sum=float(gamma_sum.detach()),
        gamma_count=gamma_count,
        risk_sum=float(risk_sum.detach()),
    )


def make_gate_tasks(training_config: MetaTrainConfig) -> GenericRankingTaskGenerator:
    return GenericRankingTaskGenerator(
        cue_size=training_config.cue_size,
        min_edges=training_config.min_edges,
        max_edges=training_config.max_edges,
        support_blocks=training_config.support_blocks,
        excluded_signatures=registered_holdout_signatures(),
        subject_encoding_mode=training_config.subject_encoding_mode,
    )
