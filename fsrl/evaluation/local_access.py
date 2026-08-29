"""Frozen local-access rollout and intervention helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypedDict

import numpy as np
import torch

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.tasks.evidence import broader_local_admission

from .contracts import FastWeightIntervention
from .frozen_fast_weight import FrozenFastWeightEvaluator
from .relational_query import readout_relational_query_bundle

Relation = tuple[int, int]
PairSchedules = Sequence[Sequence[Relation]]
QueryBundle = dict[str, np.ndarray]


class DualAccessQueryReadout(TypedDict):
    intact_trace: AccessTrace
    condition_bundles: dict[str, QueryBundle]
    global_loo_bundles: list[QueryBundle]
    local_loo_bundles: list[QueryBundle]


def access_factor(global_admission: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Broaden local admission while preserving every retained write exactly."""

    admission = np.asarray(global_admission, dtype=np.float64)
    probability = np.asarray(reliability, dtype=np.float64)
    if admission.shape != probability.shape:
        raise ValueError("global admission and reliability must have the same shape")
    if not np.all((admission == 0.0) | (admission == 1.0)):
        raise ValueError("global admission must be binary")
    if not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("reliability must lie in [0, 1]")
    return broader_local_admission(admission, probability)


def apply_blockwise_route(values: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Route donor support scalars to recipient slots under fixed block maps."""

    scalars = np.asarray(values)
    if scalars.ndim != 2 or maps.ndim != 3:
        raise ValueError("values or maps have the wrong rank")
    subjects, trials = scalars.shape
    if maps.shape[0] != subjects or maps.shape[1] * maps.shape[2] != trials:
        raise ValueError("values do not match block maps")
    block_size = maps.shape[2]
    routed = np.empty_like(scalars)
    for subject in range(subjects):
        for block in range(maps.shape[1]):
            start = block * block_size
            routed[subject, start : start + block_size] = scalars[
                subject, start + maps[subject, block]
            ]
    return routed


@dataclass(frozen=True)
class AccessTrace:
    state: torch.Tensor
    natural_scalars: np.ndarray
    applied_scalars: np.ndarray
    route_maps: np.ndarray | None


def relation_reliability(
    evaluator: FrozenFastWeightEvaluator, subject: int, higher: int, lower: int
) -> float:
    if evaluator.subject_encoding_states is None:
        raise RuntimeError("dual access requires the frozen subject encoding state")
    distance = evaluator.item_rank[lower] - evaluator.item_rank[higher]
    return evaluator.subject_encoding_states[subject].relation_reliability(
        higher, lower, distance
    )


def _natural_local_scalars(
    evaluator: FrozenFastWeightEvaluator,
    *,
    dual_access: bool,
    zero_relations: frozenset[Relation],
) -> np.ndarray:
    values = np.empty(
        (evaluator.config.bs, evaluator.protocol.support_trials), dtype=np.float32
    )
    for subject, schedule in enumerate(evaluator.support_schedules):
        for trial_index, trial in enumerate(schedule):
            relation = (trial.higher_item, trial.lower_item)
            if relation in zero_relations:
                values[subject, trial_index] = 0.0
                continue
            admission = evaluator._encoding_reliability(subject, trial_index)
            if dual_access:
                probability = relation_reliability(
                    evaluator, subject, trial.higher_item, trial.lower_item
                )
                admission = float(
                    access_factor(np.asarray([admission]), np.asarray([probability]))[0]
                )
            values[subject, trial_index] = trial.signed_magnitude * admission
    return values


def build_access_trace(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    *,
    dual_access: bool,
    route_maps: np.ndarray | None = None,
    zero_relations: frozenset[Relation] = frozenset(),
) -> AccessTrace:
    """Replay frozen support slots into the confirmed local trace."""

    natural = _natural_local_scalars(
        evaluator, dual_access=dual_access, zero_relations=zero_relations
    )
    applied = (
        natural if route_maps is None else apply_blockwise_route(natural, route_maps)
    )
    state = local.initial_state(evaluator.config.bs)
    with torch.no_grad():
        for trial_index in range(evaluator.protocol.support_trials):
            trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
            left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
            right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
            step0 = evaluator._step_inputs(
                left,
                right,
                applied[:, trial_index],
                numstep=0,
                time_value=(
                    trial_index
                    / max(1, evaluator.protocol.support_trials - 1)
                    * evaluator.test_time_value
                ),
                support_trial=True,
            )
            state = local.write(
                state,
                step0[:, : 2 * evaluator.config.cs],
                torch.from_numpy(applied[:, trial_index]).to(state.device),
            )
    return AccessTrace(state.detach().clone(), natural, applied, route_maps)


def build_fast_weight_loo(
    evaluator: FrozenFastWeightEvaluator, relations: Iterable[Relation]
) -> torch.Tensor:
    """Replay one global fast-weight leave-one-relation-out state per relation."""

    rows = []
    for relation in relations:
        state = evaluator.initialize_fast_weights()
        for trial_index in range(evaluator.protocol.support_trials):
            state = evaluator.advance_support_trial(
                state, trial_index, zero_relations=frozenset((relation,))
            )
        rows.append(state)
    return torch.stack(rows)


def readout_dual_access_query_conditions(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    pair_schedules: PairSchedules,
) -> DualAccessQueryReadout:
    """Read intact, component-off, and relation-LOO dual-access conditions."""

    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    loo_fast_weights = build_fast_weight_loo(evaluator, relations)
    intact_trace = build_access_trace(evaluator, local, dual_access=True)
    loo_traces = [
        build_access_trace(
            evaluator, local, dual_access=True, zero_relations=frozenset((relation,))
        )
        for relation in relations
    ]
    condition_bundles = {
        "intact": readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            intact_trace.state,
            pair_schedules,
            local_off=False,
            global_off=False,
            shuffled_indices=None,
        ),
        "a_off": readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            intact_trace.state,
            pair_schedules,
            local_off=True,
            global_off=False,
            shuffled_indices=None,
        ),
        "P_off_a_on": readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            intact_trace.state,
            pair_schedules,
            local_off=False,
            global_off=True,
            shuffled_indices=None,
        ),
    }
    global_loo_bundles = [
        readout_relational_query_bundle(
            evaluator,
            local,
            loo_fast_weights[index],
            loo_traces[index].state,
            pair_schedules,
            local_off=True,
            global_off=False,
            shuffled_indices=None,
        )
        for index in range(len(relations))
    ]
    local_loo_bundles = [
        readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            loo_traces[index].state,
            pair_schedules,
            local_off=False,
            global_off=True,
            shuffled_indices=None,
        )
        for index in range(len(relations))
    ]
    return {
        "intact_trace": intact_trace,
        "condition_bundles": condition_bundles,
        "global_loo_bundles": global_loo_bundles,
        "local_loo_bundles": local_loo_bundles,
    }


def measure_presentation_invariance(
    evaluator: FrozenFastWeightEvaluator,
    local: ConjunctiveLocalTrace,
    natural_scalars: np.ndarray,
) -> dict[str, float]:
    """Measure reversal identities for support writes and query keys."""

    support_error = 0.0
    with torch.no_grad():
        for trial_index in range(evaluator.protocol.support_trials):
            trials = [schedule[trial_index] for schedule in evaluator.support_schedules]
            left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
            right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
            zero = np.zeros(evaluator.config.bs, dtype=np.float32)
            forward = evaluator._step_inputs(
                left,
                right,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=True,
            )
            reverse = evaluator._step_inputs(
                right,
                left,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=True,
            )
            scalar = torch.from_numpy(natural_scalars[:, trial_index]).to(
                forward.device
            )[:, None]
            support_error = max(
                support_error,
                float(
                    torch.max(
                        torch.abs(
                            scalar * local.key(forward[:, : 2 * evaluator.config.cs])
                            - (-scalar)
                            * local.key(reverse[:, : 2 * evaluator.config.cs])
                        )
                    ).cpu()
                ),
            )

        query_error = 0.0
        zero = np.zeros(evaluator.config.bs, dtype=np.float32)
        for left_item, right_item in combinations(range(evaluator.protocol.n_items), 2):
            left = np.full(evaluator.config.bs, left_item, dtype=np.int64)
            right = np.full(evaluator.config.bs, right_item, dtype=np.int64)
            forward = evaluator._step_inputs(
                left,
                right,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=False,
            )
            reverse = evaluator._step_inputs(
                right,
                left,
                zero,
                numstep=0,
                time_value=0.0,
                support_trial=False,
            )
            query_error = max(
                query_error,
                float(
                    torch.max(
                        torch.abs(
                            local.key(forward[:, : 2 * evaluator.config.cs])
                            + local.key(reverse[:, : 2 * evaluator.config.cs])
                        )
                    ).cpu()
                ),
            )
    return {
        "support_write_reversal_max_abs_error": support_error,
        "query_key_reversal_max_abs_error": query_error,
    }
