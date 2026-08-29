"""Exact support-time plasticity traces for frozen evaluators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .frozen_fast_weight import FrozenFastWeightEvaluator


@dataclass(frozen=True)
class SupportTrialTrace:
    final_fast_weights: torch.Tensor
    da: torch.Tensor
    eligibility_before: torch.Tensor
    intended_increment: torch.Tensor
    actual_increment: torch.Tensor
    clip_fraction: torch.Tensor
    clip_excess_mean: torch.Tensor
    forward_max_abs_error: float


def _trial_inputs(
    evaluator: FrozenFastWeightEvaluator,
    trial_index: int,
    evidence_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
    left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
    right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
    signed = np.asarray(
        [
            trial.signed_magnitude
            * evaluator._encoding_reliability(subject, trial_index)
            * evidence_scales[subject]
            for subject, trial in enumerate(trials)
        ],
        dtype=np.float32,
    )
    time_value = (
        trial_index
        / max(1, evaluator.protocol.support_trials - 1)
        * evaluator.test_time_value
    )
    return left, right, signed, time_value


def trace_support_trial(
    evaluator: FrozenFastWeightEvaluator,
    fast_weights: torch.Tensor,
    trial_index: int,
    *,
    evidence_scales: np.ndarray | None = None,
) -> SupportTrialTrace:
    """Advance one support slot and expose the exact pre-update write tensors."""

    if evidence_scales is None:
        scales = np.ones(evaluator.config.bs, dtype=np.float32)
    else:
        scales = np.asarray(evidence_scales, dtype=np.float32)
        if scales.shape != (evaluator.config.bs,):
            raise ValueError("evidence_scales must have one value per subject")
    left, right, signed, time_value = _trial_inputs(evaluator, trial_index, scales)
    hidden = evaluator.net.initial_hidden(evaluator.config.bs)
    eligibility = evaluator.net.initial_eligibility(evaluator.config.bs)
    current = fast_weights.detach().clone()
    da_steps = []
    eligibility_steps = []
    intended_steps = []
    actual_steps = []
    clip_fraction_steps = []
    clip_excess_steps = []
    max_error = 0.0

    with torch.no_grad():
        for step in range(evaluator.config.triallen):
            inputs = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=step,
                time_value=time_value,
                support_trial=True,
            )
            before_eligibility = eligibility.detach().clone()
            before_fast_weights = current.detach().clone()
            _, _, da, hidden, eligibility, proposed = evaluator.net(
                inputs, hidden, eligibility, current
            )
            intended = da.view(evaluator.config.bs, 1, 1) * before_eligibility
            preclip = before_fast_weights + intended
            expected = torch.clamp(preclip, min=-50.0, max=50.0)
            max_error = max(
                max_error, float(torch.max(torch.abs(expected - proposed)).item())
            )
            actual = proposed - before_fast_weights
            excess = torch.clamp(torch.abs(preclip) - 50.0, min=0.0)
            da_steps.append(da[:, 0].detach().clone())
            eligibility_steps.append(before_eligibility)
            intended_steps.append(intended.detach().clone())
            actual_steps.append(actual.detach().clone())
            clip_fraction_steps.append(
                torch.mean((excess > 0.0).to(torch.float32), dim=(1, 2))
            )
            clip_excess_steps.append(torch.mean(excess, dim=(1, 2)))
            current = proposed

    return SupportTrialTrace(
        final_fast_weights=current.detach().clone(),
        da=torch.stack(da_steps, dim=1),
        eligibility_before=torch.stack(eligibility_steps, dim=1),
        intended_increment=torch.stack(intended_steps, dim=1),
        actual_increment=torch.stack(actual_steps, dim=1),
        clip_fraction=torch.stack(clip_fraction_steps, dim=1),
        clip_excess_mean=torch.stack(clip_excess_steps, dim=1),
        forward_max_abs_error=max_error,
    )
