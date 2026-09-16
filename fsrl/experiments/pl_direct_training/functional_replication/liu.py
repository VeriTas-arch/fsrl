"""Matched shared/dual Liu-v2 rollout for the clean no-time candidate."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    blockwise_derangements,
)
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices

from ..batches import direct_input_arrays
from ..liu import DirectLiuEvaluator

CONDITION_CONFIGURATIONS = {
    "dual_access": ("dual", False, False, False),
    "shared_access": ("shared", False, False, False),
    "dual_evidence_shuffle": ("evidence_shuffle", False, False, False),
    "dual_query_shuffle": ("dual", False, False, True),
    "local_off": ("shared", True, False, False),
    "P_off_dual": ("dual", False, True, False),
    "P_off_shared": ("shared", False, True, False),
}


def admitted_local_evidence(
    signed_magnitude: float,
    retained: float,
    probability: float,
    access: str,
) -> float:
    if access == "shared":
        admission = retained
    elif access in {"dual", "evidence_shuffle"}:
        admission = retained + (1.0 - retained) * probability
    else:
        raise ValueError(f"unknown local access rule: {access}")
    return float(signed_magnitude * admission)


class FunctionalLiuEvaluator(DirectLiuEvaluator):
    def local_evidence(
        self,
        access: str,
        *,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
    ) -> np.ndarray:
        values = np.empty(
            (self.subjects, self.protocol.support_trials), dtype=np.float32
        )
        for subject, schedule in enumerate(self.support_schedules):
            for trial_index, trial in enumerate(schedule):
                relation = (trial.higher_item, trial.lower_item)
                if relation in zero_relations:
                    values[subject, trial_index] = 0.0
                    continue
                values[subject, trial_index] = admitted_local_evidence(
                    trial.signed_magnitude,
                    self.trial_gain(subject, trial_index),
                    self.relation_probability(subject, *relation),
                    access,
                )
        return values

    def build_access_state(
        self,
        access: str,
        *,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
        route_maps: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        natural = self.local_evidence(access, zero_relations=zero_relations)
        applied = (
            natural if route_maps is None else self.route_evidence(natural, route_maps)
        )
        state = self.local.initial_state(self.subjects)
        with torch.no_grad():
            for trial_index in range(self.protocol.support_trials):
                trials = [schedule[trial_index] for schedule in self.support_schedules]
                pairs = np.asarray(
                    [[(trial.left_item, trial.right_item) for trial in trials]],
                    dtype=np.int64,
                )
                inputs, _ = direct_input_arrays(
                    self.cue_codes,
                    pairs,
                    applied[:, trial_index][None, :],
                    np.asarray([0.0], dtype=np.float32),
                    steps=4,
                )
                state = self.local.write(
                    state,
                    torch.from_numpy(inputs[0, 0, :, : 2 * self.local.cue_size]).to(
                        self.device
                    ),
                    torch.from_numpy(applied[:, trial_index]).to(self.device),
                )
        return state.detach().clone(), natural, applied


def _condition_bundle(
    evaluator: FunctionalLiuEvaluator,
    condition: str,
    fast_weights: torch.Tensor,
    states: dict[str, torch.Tensor],
    query_maps: np.ndarray,
) -> dict[str, np.ndarray]:
    trace, local_off, global_off, query_shuffled = CONDITION_CONFIGURATIONS[condition]
    return evaluator.query_bundle(
        fast_weights,
        states[trace],
        local_off=local_off,
        global_off=global_off,
        shuffled_indices=query_maps if query_shuffled else None,
    )


def rollout_functional(evaluator: FunctionalLiuEvaluator, settings: dict) -> dict:
    protocol = evaluator.protocol
    relations = tuple(protocol.support_pairs_higher_lower)
    fast_weights = evaluator.learn_fast_weights()
    evidence_maps = blockwise_derangements(
        evaluator.subjects,
        protocol.support_blocks,
        len(relations),
        int(settings["evidence_shuffle_seed"]),
    )
    query_maps = shuffled_pair_indices(
        evaluator.subjects,
        protocol.n_items,
        int(settings["query_shuffle_seed"]),
    )
    shared_state, shared_natural, _ = evaluator.build_access_state("shared")
    dual_state, dual_natural, _ = evaluator.build_access_state("dual")
    shuffled_state, _, shuffled_applied = evaluator.build_access_state(
        "evidence_shuffle", route_maps=evidence_maps
    )
    states = {
        "shared": shared_state,
        "dual": dual_state,
        "evidence_shuffle": shuffled_state,
    }
    bundles = {
        condition: _condition_bundle(
            evaluator, condition, fast_weights, states, query_maps
        )
        for condition in CONDITION_CONFIGURATIONS
    }
    condition_loo = {condition: [] for condition in CONDITION_CONFIGURATIONS}
    for relation in relations:
        removed = frozenset((relation,))
        loo_weights = evaluator.learn_fast_weights(zero_relations=removed)
        loo_shared, _, _ = evaluator.build_access_state(
            "shared", zero_relations=removed
        )
        loo_dual, _, _ = evaluator.build_access_state("dual", zero_relations=removed)
        loo_shuffle, _, _ = evaluator.build_access_state(
            "evidence_shuffle",
            zero_relations=removed,
            route_maps=evidence_maps,
        )
        loo_states = {
            "shared": loo_shared,
            "dual": loo_dual,
            "evidence_shuffle": loo_shuffle,
        }
        for condition in CONDITION_CONFIGURATIONS:
            condition_loo[condition].append(
                _condition_bundle(
                    evaluator, condition, loo_weights, loo_states, query_maps
                )["logits"]
            )
    global_reference = bundles["shared_access"]["global_logits"]
    global_identity_error = max(
        float(np.max(np.abs(bundle["global_logits"] - global_reference)))
        for condition, bundle in bundles.items()
        if not CONDITION_CONFIGURATIONS[condition][2]
    )
    local_identity_error = max(
        float(
            np.max(
                np.abs(
                    bundle["logits"]
                    - bundle["global_logits"]
                    - bundle["applied_local_margins"]
                )
            )
        )
        for bundle in bundles.values()
    )
    if not np.array_equal(
        bundles["local_off"]["logits"], bundles["local_off"]["global_logits"]
    ):
        raise RuntimeError("functional local-off identity failed")
    return {
        "bundles": bundles,
        "loo": {condition: np.stack(rows) for condition, rows in condition_loo.items()},
        "retention": evaluator.retention(),
        "shared_local_evidence": shared_natural,
        "dual_local_evidence": dual_natural,
        "shuffled_local_evidence": shuffled_applied,
        "evidence_routing": evidence_maps,
        "query_routing": query_maps,
        "integrity": {
            "global_condition_logit_max_abs_error": global_identity_error,
            "local_margin_identity_max_abs_error": local_identity_error,
        },
    }
