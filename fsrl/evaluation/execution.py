"""Execution helpers for frozen fast-weight evaluation."""

from __future__ import annotations

import torch

from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.infra.runtime import ExecutionProfile

from .contracts import FrozenEvaluationBackend


class RecurrentTrajectory(torch.nn.Module):
    """Execute one trial while retaining hidden and margin trajectories."""

    def __init__(self, cell: RetroModulRNN):
        super().__init__()
        self.cell = cell

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_steps: list[torch.Tensor] = []
        margin_steps: list[torch.Tensor] = []
        for inputs in input_sequence.unbind(0):
            logits, _, _, hidden, eligibility, _ = self.cell(
                inputs, hidden, eligibility, fast_weights
            )
            hidden_steps.append(hidden)
            margin_steps.append(logits[:, 1] - logits[:, 0])
        return torch.stack(hidden_steps, dim=1), torch.stack(margin_steps, dim=1)


def evaluation_execution_record(
    backend: FrozenEvaluationBackend,
    execution_profile: ExecutionProfile | None,
) -> dict[str, object]:
    if backend == FrozenEvaluationBackend.LEGACY_STEPWISE:
        return {"execution_schema_version": 1, "backend": backend.value}
    assert execution_profile is not None
    return {
        "execution_schema_version": 2,
        "backend": backend.value,
        "profile": execution_profile.to_dict(),
        "compile_scope": "complete_recurrent_trial_sequence",
        "support_batching": "sequential_trials_one_transfer_each",
        "query_batching": "all_query_pairs_by_subject",
        "metric_transfer": "one_batched_device_to_cpu_transfer",
        "trajectory_transfer": (
            "one_hidden_and_one_logit_batched_device_to_cpu_transfer"
        ),
    }
