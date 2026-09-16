"""Frozen Liu-v2 rollout for the compact P/L candidate."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.evaluation.contracts import FastWeightIntervention
from fsrl.evaluation.sampling import deterministic_cue_codes
from fsrl.evaluation.subject_encoding import build_frozen_subject_encoding
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    blockwise_derangements,
)
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.infra.runtime import ExecutionProfile, compile_module
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs

from .batches import compact_input_arrays
from .model import CompactPlasticRNN, CompactRecurrentSequence, PackedLocalTrace


class CompactLiuEvaluator:
    def __init__(
        self,
        backbone: CompactPlasticRNN,
        local: PackedLocalTrace,
        protocol: RankingProtocol,
        *,
        subjects: int,
        cue_seed: int,
        support_seed: int,
        subject_encoding_seed: int,
        cue_mode: str,
        subject_encoding_mode: str,
        execution_profile: ExecutionProfile = PROFILE,
    ) -> None:
        self.backbone = backbone
        self.local = local
        self.protocol = protocol
        self.subjects = subjects
        self.item_rank = {
            item: rank for rank, item in enumerate(protocol.true_order_high_to_low)
        }
        self.cue_codes = deterministic_cue_codes(
            subjects,
            protocol.n_items,
            backbone.model_config.cue_size,
            cue_seed,
            mode=cue_mode,
        )
        self.support_schedules = tuple(
            protocol.support_schedule(np.random.default_rng(support_seed + subject))
            for subject in range(subjects)
        )
        legacy_config = TrainConfig(
            bs=subjects,
            cs=backbone.model_config.cue_size,
            hs=backbone.model_config.hidden_size,
            triallen=3,
        )
        encoding = build_frozen_subject_encoding(
            legacy_config,
            protocol,
            self.item_rank,
            self.support_schedules,
            mode=subject_encoding_mode,
            seed=subject_encoding_seed,
        )
        self.subject_encoding_states = encoding.states
        self.subject_relation_gains = encoding.relation_gains
        self.subject_trial_gains = encoding.trial_gains
        self.sequence = compile_module(
            CompactRecurrentSequence(backbone), execution_profile
        )

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @contextmanager
    def alpha_zeroed(self, enabled: bool):
        if not enabled:
            yield
            return
        saved = self.backbone.alpha.detach().clone()
        with torch.no_grad():
            self.backbone.alpha.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                self.backbone.alpha.copy_(saved)

    def relation_probability(self, subject: int, higher: int, lower: int) -> float:
        if self.subject_encoding_states is None:
            return 1.0
        distance = self.item_rank[lower] - self.item_rank[higher]
        return self.subject_encoding_states[subject].relation_reliability(
            higher, lower, distance
        )

    def trial_gain(self, subject: int, trial_index: int) -> float:
        if self.subject_trial_gains is None:
            return 1.0
        return self.subject_trial_gains[subject][trial_index]

    def _support_inputs(
        self,
        trial_index: int,
        *,
        zero_evidence: bool,
        zero_relations: frozenset[tuple[int, int]],
    ) -> torch.Tensor:
        trials = [schedule[trial_index] for schedule in self.support_schedules]
        pairs = np.asarray(
            [[(trial.left_item, trial.right_item) for trial in trials]], dtype=np.int64
        )
        evidence = np.asarray(
            [
                0.0
                if zero_evidence
                or (trial.higher_item, trial.lower_item) in zero_relations
                else trial.signed_magnitude * self.trial_gain(subject, trial_index)
                for subject, trial in enumerate(trials)
            ],
            dtype=np.float32,
        )[None, :]
        return torch.from_numpy(
            compact_input_arrays(self.cue_codes, pairs, evidence, steps=3)[0]
        ).to(self.device)

    def advance_support(
        self,
        fast_weights: torch.Tensor,
        trial_index: int,
        *,
        intervention: FastWeightIntervention = FastWeightIntervention.INTACT,
        zero_evidence: bool = False,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
    ) -> torch.Tensor:
        inputs = self._support_inputs(
            trial_index,
            zero_evidence=zero_evidence,
            zero_relations=zero_relations,
        )
        with (
            torch.no_grad(),
            self.alpha_zeroed(intervention == FastWeightIntervention.ALPHA_ZERO),
        ):
            outputs = self.sequence(
                inputs,
                self.backbone.initial_hidden(self.subjects),
                self.backbone.initial_eligibility(self.subjects),
                fast_weights,
                intervention != FastWeightIntervention.WRITE_OFF,
            )
        return outputs[-1].detach().clone()

    def learn_fast_weights(
        self,
        intervention: FastWeightIntervention = FastWeightIntervention.INTACT,
        *,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
    ) -> torch.Tensor:
        state = self.backbone.initial_fast_weights(self.subjects)
        for trial in range(self.protocol.support_trials):
            state = self.advance_support(
                state,
                trial,
                intervention=intervention,
                zero_relations=zero_relations,
            )
        if intervention == FastWeightIntervention.RESET:
            state = torch.zeros_like(state)
        elif intervention == FastWeightIntervention.SHUFFLE:
            state = torch.roll(state, shifts=1, dims=0)
        return state

    def local_evidence(
        self, *, zero_relations: frozenset[tuple[int, int]] = frozenset()
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
                retained = self.trial_gain(subject, trial_index)
                probability = self.relation_probability(subject, *relation)
                admission = retained + (1.0 - retained) * probability
                values[subject, trial_index] = trial.signed_magnitude * admission
        return values

    @staticmethod
    def route_evidence(values: np.ndarray, maps: np.ndarray) -> np.ndarray:
        routed = np.empty_like(values)
        block_size = maps.shape[2]
        for subject in range(values.shape[0]):
            for block in range(maps.shape[1]):
                start = block * block_size
                routed[subject, start : start + block_size] = values[
                    subject, start + maps[subject, block]
                ]
        return routed

    def build_local_state(
        self,
        *,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
        route_maps: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        natural = self.local_evidence(zero_relations=zero_relations)
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
                cues = compact_input_arrays(
                    self.cue_codes,
                    pairs,
                    applied[:, trial_index][None, :],
                    steps=3,
                )[0, 0, :, : 2 * self.local.cue_size]
                state = self.local.write(
                    state,
                    torch.from_numpy(cues).to(self.device),
                    torch.from_numpy(applied[:, trial_index]).to(self.device),
                )
        return state.detach().clone(), natural, applied

    def query_bundle(
        self,
        fast_weights: torch.Tensor,
        local_state: torch.Tensor,
        *,
        local_off: bool = False,
        global_off: bool = False,
        alpha_zero: bool = False,
        shuffled_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        pairs = np.asarray(ordered_pairs(self.protocol.n_items), dtype=np.int64)
        count = len(pairs)
        schedule = np.broadcast_to(pairs[:, None], (count, self.subjects, 2))
        inputs = compact_input_arrays(
            self.cue_codes,
            schedule,
            np.zeros((count, self.subjects), dtype=np.float32),
            steps=2,
        )
        inputs = torch.from_numpy(
            inputs.transpose(1, 0, 2, 3).reshape(2, count * self.subjects, -1).copy()
        ).to(self.device)
        weights = torch.zeros_like(fast_weights) if global_off else fast_weights
        query_weights = weights.repeat(count, 1, 1)
        pair_cues = inputs[0, :, : 2 * self.local.cue_size]
        if shuffled_indices is not None:
            source = pair_cues.reshape(count, self.subjects, -1).transpose(0, 1)
            indices = torch.as_tensor(shuffled_indices, device=self.device)
            pair_cues = (
                source[
                    torch.arange(self.subjects, device=self.device)[:, None],
                    indices,
                ]
                .transpose(0, 1)
                .reshape(count * self.subjects, -1)
            )
        with torch.no_grad(), self.alpha_zeroed(alpha_zero):
            global_margin, _, _, _, _ = self.sequence(
                inputs,
                self.backbone.initial_hidden(count * self.subjects),
                self.backbone.initial_eligibility(count * self.subjects),
                query_weights,
                False,
            )
            margins, raw, correction = self.local(
                global_margin,
                local_state.repeat(count, 1),
                pair_cues,
                active=not local_off,
            )
        values = {
            "logits": margins[:, 0],
            "global_logits": global_margin[:, 0],
            "raw_local_margins": raw[:, 0],
            "applied_local_margins": correction[:, 0],
            "local_gains": self.local.gain.expand_as(raw)[:, 0],
        }
        return {
            name: value.detach()
            .reshape(count, self.subjects)
            .T.cpu()
            .numpy()
            .astype(np.float64)
            for name, value in values.items()
        }

    def retention(self) -> np.ndarray:
        relations = self.protocol.support_pairs_higher_lower
        if self.subject_relation_gains is None:
            return np.ones((self.subjects, len(relations)), dtype=bool)
        return np.asarray(
            [
                [
                    self.subject_relation_gains[subject][relation] > 0.0
                    for relation in relations
                ]
                for subject in range(self.subjects)
            ],
            dtype=bool,
        )


def rollout_liu(evaluator: CompactLiuEvaluator) -> dict:
    protocol = evaluator.protocol
    relations = protocol.support_pairs_higher_lower
    weights = evaluator.learn_fast_weights()
    local_state, natural, _ = evaluator.build_local_state()
    routing = blockwise_derangements(
        evaluator.subjects, protocol.support_blocks, len(relations), 32301
    )
    shuffled_state, _, shuffled = evaluator.build_local_state(route_maps=routing)
    queries = shuffled_pair_indices(evaluator.subjects, protocol.n_items, 31801)
    bundles = {
        "intact": evaluator.query_bundle(weights, local_state),
        "local_off": evaluator.query_bundle(weights, local_state, local_off=True),
        "P_off": evaluator.query_bundle(weights, local_state, global_off=True),
        "query_shuffle": evaluator.query_bundle(
            weights, local_state, shuffled_indices=queries
        ),
        "evidence_shuffle": evaluator.query_bundle(weights, shuffled_state),
    }
    if not np.array_equal(
        bundles["local_off"]["logits"], bundles["intact"]["global_logits"]
    ):
        raise RuntimeError("local-off identity failed")
    loo = {"global": [], "local": [], "combined": []}
    for relation in relations:
        loo_weights = evaluator.learn_fast_weights(
            zero_relations=frozenset((relation,))
        )
        loo_local, _, _ = evaluator.build_local_state(
            zero_relations=frozenset((relation,))
        )
        loo["global"].append(
            evaluator.query_bundle(loo_weights, loo_local, local_off=True)["logits"]
        )
        loo["local"].append(
            evaluator.query_bundle(loo_weights, loo_local, global_off=True)["logits"]
        )
        loo["combined"].append(evaluator.query_bundle(loo_weights, loo_local)["logits"])
    controls = {}
    for intervention in (
        FastWeightIntervention.WRITE_OFF,
        FastWeightIntervention.ALPHA_ZERO,
        FastWeightIntervention.RESET,
        FastWeightIntervention.SHUFFLE,
    ):
        control_weights = evaluator.learn_fast_weights(intervention)
        controls[intervention.value] = evaluator.query_bundle(
            control_weights,
            local_state,
            local_off=True,
            alpha_zero=intervention == FastWeightIntervention.ALPHA_ZERO,
        )["logits"]
    return {
        "bundles": bundles,
        "loo": {name: np.stack(rows) for name, rows in loo.items()},
        "retention": evaluator.retention(),
        "natural_local_evidence": natural,
        "shuffled_local_evidence": shuffled,
        "evidence_routing": routing,
        "query_routing": queries,
        "legacy_controls": controls,
    }
