"""Stable query-bundle readout for the combined global and local system."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.relational_system import (
    GlobalLocalRelationalSystem,
    RelationalIntervention,
)
from fsrl.core.state import RelationalEpisodeState
from fsrl.tasks.protocol import ordered_pairs

from .frozen_fast_weight import FrozenFastWeightEvaluator

Pair = tuple[int, int]
PairSchedules = Sequence[Sequence[Pair]]


def readout_relational_query_bundle(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    fast_weights: torch.Tensor,
    local_state: torch.Tensor,
    pair_schedules: PairSchedules,
    *,
    local_off: bool,
    global_off: bool,
    shuffled_indices: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Read all scheduled P/L queries through the maintained system boundary."""

    subjects = evaluator.config.bs
    pair_count = len(pair_schedules[0])
    arrays = {
        name: np.empty((subjects, pair_count), dtype=np.float64)
        for name in (
            "logits",
            "global_logits",
            "raw_local_margins",
            "applied_local_margins",
            "local_gains",
            "policy_residuals",
        )
    }
    system = GlobalLocalRelationalSystem(evaluator.net, local)
    state = RelationalEpisodeState(fast_weights, local_state)
    all_pairs = ordered_pairs(evaluator.protocol.n_items)
    with torch.no_grad():
        for pair_index in range(pair_count):
            left = np.asarray(
                [schedule[pair_index][0] for schedule in pair_schedules],
                dtype=np.int64,
            )
            right = np.asarray(
                [schedule[pair_index][1] for schedule in pair_schedules],
                dtype=np.int64,
            )
            signed = np.zeros(subjects, dtype=np.float32)
            step0 = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=0,
                time_value=evaluator.test_time_value,
                support_trial=False,
            )
            response = evaluator._step_inputs(
                left,
                right,
                signed,
                numstep=1,
                time_value=evaluator.test_time_value,
                support_trial=False,
            )
            local_step0 = step0
            if shuffled_indices is not None:
                mapped = [
                    all_pairs[int(shuffled_indices[subject, pair_index])]
                    for subject in range(subjects)
                ]
                local_step0 = evaluator._step_inputs(
                    np.asarray([pair[0] for pair in mapped], dtype=np.int64),
                    np.asarray([pair[1] for pair in mapped], dtype=np.int64),
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
            query_state = state
            intervention = RelationalIntervention.INTACT
            if global_off and local_off:
                query_state = RelationalEpisodeState(
                    torch.zeros_like(fast_weights), local_state
                )
                intervention = RelationalIntervention.LOCAL_OFF
            elif global_off:
                intervention = RelationalIntervention.GLOBAL_OFF
            elif local_off:
                intervention = RelationalIntervention.LOCAL_OFF
            readout = system.query(
                query_state,
                torch.stack((step0, response)),
                pair_cues=local_step0[:, : 2 * evaluator.config.cs],
                intervention=intervention,
            )
            arrays["logits"][:, pair_index] = (
                (readout.logits[:, 1] - readout.logits[:, 0]).cpu().numpy()
            )
            arrays["global_logits"][:, pair_index] = (
                (readout.global_logits[:, 1] - readout.global_logits[:, 0])
                .cpu()
                .numpy()
            )
            arrays["raw_local_margins"][:, pair_index] = (
                readout.raw_local_margin[:, 0].cpu().numpy()
            )
            arrays["applied_local_margins"][:, pair_index] = (
                readout.local_correction[:, 0].cpu().numpy()
            )
            arrays["local_gains"][:, pair_index] = (
                readout.local_gain[:, 0].cpu().numpy()
            )
            arrays["policy_residuals"][:, pair_index] = (
                readout.policy_residual[:, 0].cpu().numpy()
            )
    return arrays
