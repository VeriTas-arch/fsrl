"""Strict frozen-state and fast-weight causal evaluation for Liu-style tasks."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.core.config import NUMRESPONSESTEP, TrainConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.runtime import (
    ExecutionProfile,
    compile_module,
    configure_runtime,
    default_device,
)
from fsrl.tasks.protocol import RankingProtocol, load_ranking_protocol
from fsrl.tasks.protocol_catalog import LIU_V1_PROTOCOL_PATH
from fsrl.tasks.subject_encoding import (
    SubjectEncodingConfig,
    SubjectEncodingState,
    sample_subject_encoding_states,
)

from ..core.inputs import EVIDENCE_AUXILIARY_OFFSET
from ..core.plastic_rnn import RetroModulRNN
from ..training.checkpoints import (
    CheckpointInfo,
    checkpoint_sha256,
    load_retro_checkpoint,
    load_training_provenance,
)

DEFAULT_PROTOCOL_PATH = LIU_V1_PROTOCOL_PATH
DISTANCE_INPUT_OFFSET = EVIDENCE_AUXILIARY_OFFSET

__all__ = [
    "DISTANCE_INPUT_OFFSET",
    "CheckpointInfo",
    "ConditionMetrics",
    "FastWeightIntervention",
    "FrozenEvaluationBackend",
    "FrozenFastWeightEvaluator",
    "OrderInvarianceMetrics",
    "checkpoint_sha256",
    "deterministic_cue_codes",
    "load_retro_checkpoint",
    "load_training_provenance",
    "main",
    "parse_args",
    "retained_relation_mask",
    "run_causal_suite",
]


class FastWeightIntervention(str, Enum):
    INTACT = "intact"
    WRITE_OFF = "write_off"
    ALPHA_ZERO = "alpha_zero"
    RESET = "reset"
    SHUFFLE = "shuffle"


class FrozenEvaluationBackend(str, Enum):
    """Explicit execution backend for frozen causal evaluation."""

    LEGACY_STEPWISE = "legacy_stepwise"
    BATCHED_SEQUENCE = "batched_sequence"


class _RecurrentTrajectory(torch.nn.Module):
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
        hidden_steps = []
        margin_steps = []
        for inputs in input_sequence.unbind(0):
            logits, _, _, hidden, eligibility, _ = self.cell(
                inputs, hidden, eligibility, fast_weights
            )
            hidden_steps.append(hidden)
            margin_steps.append(logits[:, 1] - logits[:, 0])
        return torch.stack(hidden_steps, dim=1), torch.stack(margin_steps, dim=1)


@dataclass(frozen=True)
class ConditionMetrics:
    intervention: str
    overall_accuracy: float
    learned_accuracy: float
    nonlearned_accuracy: float
    mean_probability_correct: float
    mean_abs_fast_weight: float
    mean_circular_triads: float
    mean_transitive_triplet_fraction: float


@dataclass(frozen=True)
class OrderInvarianceMetrics:
    schedules: int
    pairs: int
    max_abs_logit_delta: float
    mean_abs_logit_delta: float


def retained_relation_mask(
    evaluator: FrozenFastWeightEvaluator, relations
) -> np.ndarray:
    """Return relation-by-subject retention under the evaluator's encoding."""

    if evaluator.subject_relation_gains is None:
        return np.ones((len(relations), evaluator.config.bs), dtype=bool)
    return np.asarray(
        [
            [
                evaluator.subject_relation_gains[subject][relation] > 0.0
                for subject in range(evaluator.config.bs)
            ]
            for relation in relations
        ],
        dtype=bool,
    )


def deterministic_cue_codes(
    n_subjects: int,
    n_items: int,
    cue_size: int,
    seed: int,
    *,
    mode: str = "shared",
) -> np.ndarray:
    """Generate a shared cue set with optional subject-specific item mappings."""

    if cue_size > 20:
        raise ValueError("cue_size > 20 is not supported by exhaustive code generation")
    rng = np.random.default_rng(seed)
    values = np.arange(1 << cue_size, dtype=np.uint32)
    bit_positions = np.arange(cue_size, dtype=np.uint32)
    bits = ((values[:, None] >> bit_positions) & 1).astype(np.int8)
    candidates = (bits * 2 - 1).astype(np.float32)
    for _ in range(100):
        codes: list[np.ndarray] = []
        for candidate_index in rng.permutation(len(candidates)):
            candidate = candidates[int(candidate_index)]
            if all(np.mean(previous == candidate) <= 0.66 for previous in codes):
                codes.append(candidate)
                if len(codes) == n_items:
                    shared = np.stack(codes)
                    if mode == "shared":
                        return np.repeat(shared[None, :, :], n_subjects, axis=0)
                    if mode == "permuted_shared":
                        return np.stack(
                            [
                                shared[rng.permutation(n_items)]
                                for _ in range(n_subjects)
                            ]
                        )
                    raise ValueError(f"unknown cue mode: {mode}")
    raise ValueError(
        f"Could not construct {n_items} sufficiently distinct {cue_size}-bit cues"
    )


class FrozenFastWeightEvaluator:
    """Evaluate one shared network under explicit fast-weight interventions."""

    def __init__(
        self,
        net: RetroModulRNN,
        config: TrainConfig,
        protocol: RankingProtocol,
        *,
        cue_seed: int = 0,
        support_seed: int = 0,
        cue_mode: str = "shared",
        subject_encoding_mode: str = "none",
        subject_encoding_seed: int = 0,
        test_time_value: float = 2.0 / 3.0,
        backend: FrozenEvaluationBackend
        | str = FrozenEvaluationBackend.LEGACY_STEPWISE,
        execution_profile: ExecutionProfile | None = None,
    ) -> None:
        if config.bs < 1:
            raise ValueError("batch size must be positive")
        if protocol.n_items != 8:
            raise ValueError("The current neural evaluator requires eight items")
        self.net = net
        self.config = config
        self.protocol = protocol
        self.item_rank = {
            item: position
            for position, item in enumerate(protocol.true_order_high_to_low)
        }
        self.test_time_value = float(test_time_value)
        self.backend = FrozenEvaluationBackend(backend)
        if self.backend == FrozenEvaluationBackend.LEGACY_STEPWISE:
            if execution_profile is not None:
                raise ValueError(
                    "execution_profile is only valid for the batched evaluator"
                )
            self.execution_profile = None
            self.sequence_runner = None
            self.trajectory_runner = None
        else:
            profile = execution_profile or ExecutionProfile(
                device=self.device.type,
                compile=self.device.type == "cuda",
                require_cuda=self.device.type == "cuda",
            )
            if profile.device != self.device.type:
                raise ValueError("evaluation profile and network device differ")
            self.execution_profile = profile
            self.sequence_runner = compile_module(RecurrentSequence(net), profile)
            self.trajectory_runner = compile_module(_RecurrentTrajectory(net), profile)
        self.cue_codes = deterministic_cue_codes(
            config.bs, protocol.n_items, config.cs, cue_seed, mode=cue_mode
        )
        self.cue_mode = cue_mode
        supported_encoding_modes = {
            "none",
            "stable_bottleneck",
            "stable_omission",
            "presentationwise_omission",
            "blockwise_omission",
            "uniform_no_bottleneck",
        }
        if subject_encoding_mode not in supported_encoding_modes:
            raise ValueError(f"unknown subject encoding mode: {subject_encoding_mode}")
        self.support_schedules = tuple(
            protocol.support_schedule(np.random.default_rng(support_seed + subject))
            for subject in range(config.bs)
        )
        self.subject_relation_gains: tuple[dict[tuple[int, int], float], ...] | None
        self.subject_trial_gains: tuple[tuple[float, ...], ...] | None
        if subject_encoding_mode == "none":
            self.subject_encoding_states: tuple[SubjectEncodingState, ...] | None = None
            self.subject_relation_gains = None
            self.subject_trial_gains = None
        else:
            encoding_rng = np.random.default_rng(subject_encoding_seed)
            self.subject_encoding_states = sample_subject_encoding_states(
                encoding_rng,
                config.bs,
                protocol.n_items,
            )
            probabilities = []
            for state in self.subject_encoding_states:
                subject_probabilities = {}
                for higher, lower in protocol.support_pairs_higher_lower:
                    symbolic_distance = self.item_rank[lower] - self.item_rank[higher]
                    subject_probabilities[(higher, lower)] = state.relation_reliability(
                        higher, lower, symbolic_distance
                    )
                probabilities.append(subject_probabilities)
            if subject_encoding_mode == "stable_bottleneck":
                relation_gains = probabilities
            elif subject_encoding_mode == "stable_omission":
                relation_gains = [
                    {
                        pair: float(encoding_rng.random() < probability)
                        for pair, probability in subject_probabilities.items()
                    }
                    for subject_probabilities in probabilities
                ]
            elif subject_encoding_mode == "uniform_no_bottleneck":
                uniform_gain = float(
                    np.mean(
                        [
                            probability
                            for subject_probabilities in probabilities
                            for probability in subject_probabilities.values()
                        ]
                    )
                )
                relation_gains = [
                    {pair: uniform_gain for pair in subject_probabilities}
                    for subject_probabilities in probabilities
                ]
            else:
                relation_gains = probabilities
            self.subject_relation_gains = tuple(relation_gains)

            trial_gains = []
            for subject, schedule in enumerate(self.support_schedules):
                if subject_encoding_mode == "presentationwise_omission":
                    values = tuple(
                        float(
                            encoding_rng.random()
                            < probabilities[subject][
                                (trial.higher_item, trial.lower_item)
                            ]
                        )
                        for trial in schedule
                    )
                elif subject_encoding_mode == "blockwise_omission":
                    block_relation_gains = {
                        (block, pair): float(
                            encoding_rng.random() < probabilities[subject][pair]
                        )
                        for block in range(protocol.support_blocks)
                        for pair in protocol.support_pairs_higher_lower
                    }
                    values = tuple(
                        block_relation_gains[
                            (
                                trial.block_index,
                                (trial.higher_item, trial.lower_item),
                            )
                        ]
                        for trial in schedule
                    )
                else:
                    values = tuple(
                        relation_gains[subject][(trial.higher_item, trial.lower_item)]
                        for trial in schedule
                    )
                trial_gains.append(values)
            self.subject_trial_gains = tuple(trial_gains)
        self.subject_encoding_mode = subject_encoding_mode
        self.subject_encoding_seed = subject_encoding_seed

    @property
    def device(self) -> torch.device:
        """Resolve the execution device only when a rollout actually needs it."""

        return next(self.net.parameters()).device

    def _step_inputs(
        self,
        left_items: np.ndarray,
        right_items: np.ndarray,
        signed_magnitudes: np.ndarray,
        *,
        numstep: int,
        time_value: float,
        support_trial: bool,
    ) -> torch.Tensor:
        inputs = np.zeros((self.config.bs, self.config.inputsize), dtype=np.float32)
        for subject in range(self.config.bs):
            if numstep == 0:
                inputs[subject, : 2 * self.config.cs] = np.concatenate(
                    (
                        self.cue_codes[subject, left_items[subject]],
                        self.cue_codes[subject, right_items[subject]],
                    )
                )
            elif numstep == NUMRESPONSESTEP:
                inputs[subject, self.config.nbstimbits - 1] = 1.0
            inputs[subject, self.config.nbstimbits] = 1.0
            inputs[subject, self.config.nbstimbits + 1] = time_value
            if support_trial and numstep == 0:
                inputs[subject, self.config.nbstimbits + DISTANCE_INPUT_OFFSET] = (
                    signed_magnitudes[subject]
                )
        return torch.from_numpy(inputs).to(self.device)

    def _batched_input_sequence(
        self,
        left_items: np.ndarray,
        right_items: np.ndarray,
        signed_magnitudes: np.ndarray,
        *,
        num_steps: int,
        time_value: float,
        support_trial: bool,
        subject_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        batch_size = len(left_items)
        if right_items.shape != (batch_size,) or signed_magnitudes.shape != (
            batch_size,
        ):
            raise ValueError("batched evaluator inputs do not align")
        if subject_indices is None:
            subject_indices = np.arange(batch_size, dtype=np.int64)
        if subject_indices.shape != (batch_size,):
            raise ValueError("subject indices do not align with batched inputs")
        inputs = np.zeros(
            (num_steps, batch_size, self.config.inputsize), dtype=np.float32
        )
        inputs[:, :, self.config.nbstimbits] = 1.0
        inputs[:, :, self.config.nbstimbits + 1] = time_value
        inputs[0, :, : self.config.cs] = self.cue_codes[subject_indices, left_items]
        inputs[0, :, self.config.cs : 2 * self.config.cs] = self.cue_codes[
            subject_indices, right_items
        ]
        if support_trial:
            inputs[0, :, self.config.nbstimbits + DISTANCE_INPUT_OFFSET] = (
                signed_magnitudes
            )
        if num_steps > NUMRESPONSESTEP:
            inputs[NUMRESPONSESTEP, :, self.config.nbstimbits - 1] = 1.0
        return torch.from_numpy(inputs).to(self.device)

    def evaluation_execution_record(self) -> dict:
        if self.backend == FrozenEvaluationBackend.LEGACY_STEPWISE:
            return {"execution_schema_version": 1, "backend": self.backend.value}
        assert self.execution_profile is not None
        return {
            "execution_schema_version": 2,
            "backend": self.backend.value,
            "profile": self.execution_profile.to_dict(),
            "compile_scope": "complete_recurrent_trial_sequence",
            "support_batching": "sequential_trials_one_transfer_each",
            "query_batching": "all_query_pairs_by_subject",
            "metric_transfer": "one_batched_device_to_cpu_transfer",
            "trajectory_transfer": (
                "one_hidden_and_one_logit_batched_device_to_cpu_transfer"
            ),
        }

    @contextmanager
    def _alpha_zeroed(self, enabled: bool):
        if not enabled:
            yield
            return
        saved = self.net.alpha.detach().clone()
        with torch.no_grad():
            self.net.alpha.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                self.net.alpha.copy_(saved)

    def learn_fast_weights(self, intervention: FastWeightIntervention) -> torch.Tensor:
        fast_weights = self.initialize_fast_weights(intervention)
        for trial_index in range(self.protocol.support_trials):
            fast_weights = self.advance_support_trial(
                fast_weights,
                trial_index,
                intervention=intervention,
            )

        if intervention == FastWeightIntervention.RESET:
            fast_weights = torch.zeros_like(fast_weights)
        elif intervention == FastWeightIntervention.SHUFFLE:
            if self.config.bs < 2:
                raise ValueError("shuffle intervention requires batch_size >= 2")
            fast_weights = torch.roll(fast_weights, shifts=1, dims=0)
        return fast_weights.detach().clone()

    def initialize_fast_weights(
        self, intervention: FastWeightIntervention = FastWeightIntervention.INTACT
    ) -> torch.Tensor:
        """Return P_0 after the registered two blank initialization steps."""

        if self.backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
            assert self.sequence_runner is not None
            batch_size = self.config.bs
            blank_sequence = torch.zeros(
                2,
                batch_size,
                self.config.inputsize,
                device=self.device,
            )
            with (
                torch.no_grad(),
                self._alpha_zeroed(intervention == FastWeightIntervention.ALPHA_ZERO),
            ):
                outputs = self.sequence_runner(
                    blank_sequence,
                    self.net.initial_hidden(batch_size),
                    self.net.initial_eligibility(batch_size),
                    self.net.initial_fast_weights(batch_size),
                    intervention != FastWeightIntervention.WRITE_OFF,
                )
            return outputs[-1].detach().clone()

        hidden = self.net.initial_hidden(self.config.bs)
        eligibility = self.net.initial_eligibility(self.config.bs)
        fast_weights = self.net.initial_fast_weights(self.config.bs)
        blank = torch.zeros(self.config.bs, self.config.inputsize, device=self.device)

        with (
            torch.no_grad(),
            self._alpha_zeroed(intervention == FastWeightIntervention.ALPHA_ZERO),
        ):
            for _ in range(2):
                _, _, _, hidden, eligibility, proposed = self.net(
                    blank, hidden, eligibility, fast_weights
                )
                fast_weights = (
                    fast_weights
                    if intervention == FastWeightIntervention.WRITE_OFF
                    else proposed
                )
        return fast_weights.detach().clone()

    def advance_support_trial(
        self,
        fast_weights: torch.Tensor,
        trial_index: int,
        *,
        intervention: FastWeightIntervention = FastWeightIntervention.INTACT,
        zero_evidence: bool = False,
        zero_relations: frozenset[tuple[int, int]] = frozenset(),
    ) -> torch.Tensor:
        """Advance one registered support slot while optionally zeroing evidence."""

        if not 0 <= trial_index < self.protocol.support_trials:
            raise IndexError("support trial index is outside the registered schedule")
        if fast_weights.shape != (
            self.config.bs,
            self.config.hs,
            self.config.hs,
        ):
            raise ValueError("fast_weights has the wrong shape")

        hidden = self.net.initial_hidden(self.config.bs)
        eligibility = self.net.initial_eligibility(self.config.bs)
        trials = [schedule[trial_index] for schedule in self.support_schedules]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [
                0.0
                if zero_evidence
                or (trial.higher_item, trial.lower_item) in zero_relations
                else trial.signed_magnitude
                * self._encoding_reliability(subject, trial_index)
                for subject, trial in enumerate(trials)
            ],
            dtype=np.float32,
        )
        trial_time = (
            trial_index
            / max(1, self.protocol.support_trials - 1)
            * self.test_time_value
        )
        if self.backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
            assert self.sequence_runner is not None
            input_sequence = self._batched_input_sequence(
                left,
                right,
                signed,
                num_steps=self.config.triallen,
                time_value=trial_time,
                support_trial=True,
            )
            with (
                torch.no_grad(),
                self._alpha_zeroed(intervention == FastWeightIntervention.ALPHA_ZERO),
            ):
                outputs = self.sequence_runner(
                    input_sequence,
                    hidden,
                    eligibility,
                    fast_weights,
                    intervention != FastWeightIntervention.WRITE_OFF,
                )
            return outputs[-1].detach().clone()
        with (
            torch.no_grad(),
            self._alpha_zeroed(intervention == FastWeightIntervention.ALPHA_ZERO),
        ):
            for numstep in range(self.config.triallen):
                inputs = self._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=numstep,
                    time_value=trial_time,
                    support_trial=True,
                )
                _, _, _, hidden, eligibility, proposed = self.net(
                    inputs, hidden, eligibility, fast_weights
                )
                fast_weights = (
                    fast_weights
                    if intervention == FastWeightIntervention.WRITE_OFF
                    else proposed
                )
        return fast_weights.detach().clone()

    def _encoding_reliability(self, subject: int, trial_index: int) -> float:
        if self.subject_trial_gains is None:
            return 1.0
        return self.subject_trial_gains[subject][trial_index]

    def realized_support_evidence(self) -> tuple[tuple[dict, ...], ...]:
        """Return the exact support evidence presented to each virtual subject."""

        rows = []
        for subject, schedule in enumerate(self.support_schedules):
            rows.append(
                tuple(
                    {
                        "higher_item": trial.higher_item,
                        "lower_item": trial.lower_item,
                        "magnitude": abs(trial.signed_magnitude),
                        "reliability": self._encoding_reliability(subject, trial_index),
                        "block_index": trial.block_index,
                    }
                    for trial_index, trial in enumerate(schedule)
                )
            )
        return tuple(rows)

    def readout_logits(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        alpha_zero: bool = False,
    ) -> tuple[dict[tuple[int, int], float], ...]:
        if len(pair_schedules) != self.config.bs:
            raise ValueError("one pair schedule is required per batch subject")
        schedule_lengths = {len(schedule) for schedule in pair_schedules}
        if len(schedule_lengths) != 1:
            raise ValueError("all pair schedules must have equal length")
        if self.backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
            return self._readout_logits_batched(
                fast_weights, pair_schedules, alpha_zero=alpha_zero
            )
        outputs = [{} for _ in range(self.config.bs)]

        with torch.no_grad(), self._alpha_zeroed(alpha_zero):
            for pair_index in range(next(iter(schedule_lengths))):
                hidden = self.net.initial_hidden(self.config.bs)
                eligibility = self.net.initial_eligibility(self.config.bs)
                left = np.asarray(
                    [schedule[pair_index][0] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                right = np.asarray(
                    [schedule[pair_index][1] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                signed = np.zeros(self.config.bs, dtype=np.float32)
                response_logits = None
                for numstep in range(self.config.triallen):
                    inputs = self._step_inputs(
                        left,
                        right,
                        signed,
                        numstep=numstep,
                        time_value=self.test_time_value,
                        support_trial=False,
                    )
                    logits, _, _, hidden, eligibility, _proposed = self.net(
                        inputs, hidden, eligibility, fast_weights
                    )
                    if numstep == NUMRESPONSESTEP:
                        response_logits = logits[:, 1] - logits[:, 0]
                assert response_logits is not None
                values = response_logits.detach().cpu().numpy()
                for subject, value in enumerate(values):
                    outputs[subject][(int(left[subject]), int(right[subject]))] = float(
                        value
                    )
        return tuple(outputs)

    def _readout_logits_batched(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        alpha_zero: bool,
    ) -> tuple[dict[tuple[int, int], float], ...]:
        assert self.sequence_runner is not None
        pair_count = len(pair_schedules[0])
        outputs = [{} for _ in range(self.config.bs)]
        if pair_count == 0:
            return tuple(outputs)
        input_sequence, query_fast_weights = self._prepare_batched_queries(
            fast_weights,
            pair_schedules,
            num_steps=NUMRESPONSESTEP + 1,
        )
        query_batch_size = pair_count * self.config.bs
        with torch.no_grad(), self._alpha_zeroed(alpha_zero):
            logits, _, _, _, _, _ = self.sequence_runner(
                input_sequence,
                self.net.initial_hidden(query_batch_size),
                self.net.initial_eligibility(query_batch_size),
                query_fast_weights,
                False,
            )
        values = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        values = values.reshape(pair_count, self.config.bs)
        for pair_index, row in enumerate(values):
            for subject, value in enumerate(row):
                pair = pair_schedules[subject][pair_index]
                outputs[subject][pair] = float(value)
        return tuple(outputs)

    def _prepare_batched_queries(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_count = len(pair_schedules[0])
        left = np.asarray(
            [
                schedule[pair_index][0]
                for pair_index in range(pair_count)
                for schedule in pair_schedules
            ],
            dtype=np.int64,
        )
        right = np.asarray(
            [
                schedule[pair_index][1]
                for pair_index in range(pair_count)
                for schedule in pair_schedules
            ],
            dtype=np.int64,
        )
        subject_indices = np.tile(np.arange(self.config.bs, dtype=np.int64), pair_count)
        query_batch_size = pair_count * self.config.bs
        input_sequence = self._batched_input_sequence(
            left,
            right,
            np.zeros(query_batch_size, dtype=np.float32),
            num_steps=num_steps,
            time_value=self.test_time_value,
            support_trial=False,
            subject_indices=subject_indices,
        )
        query_fast_weights = (
            fast_weights.unsqueeze(0)
            .expand(pair_count, -1, -1, -1)
            .reshape(query_batch_size, self.config.hs, self.config.hs)
        )
        return input_sequence, query_fast_weights

    def readout_hidden_states(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[dict[tuple[int, int], np.ndarray], ...]:
        """Return response-step hidden states under the strict frozen query protocol."""

        trajectories = self.readout_hidden_trajectories(fast_weights, pair_schedules)
        return tuple(
            {pair: states[NUMRESPONSESTEP].copy() for pair, states in subject.items()}
            for subject in trajectories
        )

    def readout_hidden_trajectories(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        alpha_zero: bool = False,
    ) -> tuple[dict[tuple[int, int], np.ndarray], ...]:
        """Return every hidden step under the strict frozen query protocol."""

        hidden, _logits = self.readout_hidden_and_logit_trajectories(
            fast_weights, pair_schedules, alpha_zero=alpha_zero
        )
        return hidden

    def readout_hidden_and_logit_trajectories(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        alpha_zero: bool = False,
    ) -> tuple[
        tuple[dict[tuple[int, int], np.ndarray], ...],
        tuple[dict[tuple[int, int], np.ndarray], ...],
    ]:
        """Return hidden states and output margins from each frozen query step."""

        if len(pair_schedules) != self.config.bs:
            raise ValueError("one pair schedule is required per batch subject")
        schedule_lengths = {len(schedule) for schedule in pair_schedules}
        if len(schedule_lengths) != 1:
            raise ValueError("all pair schedules must have equal length")
        if self.backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
            return self._readout_trajectories_batched(
                fast_weights, pair_schedules, alpha_zero=alpha_zero
            )
        hidden_outputs = [{} for _ in range(self.config.bs)]
        logit_outputs = [{} for _ in range(self.config.bs)]
        with torch.no_grad(), self._alpha_zeroed(alpha_zero):
            for pair_index in range(next(iter(schedule_lengths))):
                hidden = self.net.initial_hidden(self.config.bs)
                eligibility = self.net.initial_eligibility(self.config.bs)
                left = np.asarray(
                    [schedule[pair_index][0] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                right = np.asarray(
                    [schedule[pair_index][1] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                signed = np.zeros(self.config.bs, dtype=np.float32)
                hidden_steps = []
                logit_steps = []
                for numstep in range(self.config.triallen):
                    inputs = self._step_inputs(
                        left,
                        right,
                        signed,
                        numstep=numstep,
                        time_value=self.test_time_value,
                        support_trial=False,
                    )
                    logits, _, _, hidden, eligibility, _proposed = self.net(
                        inputs, hidden, eligibility, fast_weights
                    )
                    hidden_steps.append(hidden.detach().cpu().numpy())
                    logit_steps.append(
                        (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
                    )
                stacked_hidden = np.stack(hidden_steps, axis=1)
                stacked_logits = np.stack(logit_steps, axis=1)
                for subject, states in enumerate(stacked_hidden):
                    pair = (int(left[subject]), int(right[subject]))
                    hidden_outputs[subject][pair] = states.copy()
                    logit_outputs[subject][pair] = stacked_logits[subject].copy()
        return tuple(hidden_outputs), tuple(logit_outputs)

    def _readout_trajectories_batched(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
        *,
        alpha_zero: bool,
    ) -> tuple[
        tuple[dict[tuple[int, int], np.ndarray], ...],
        tuple[dict[tuple[int, int], np.ndarray], ...],
    ]:
        assert self.trajectory_runner is not None
        pair_count = len(pair_schedules[0])
        hidden_outputs = [{} for _ in range(self.config.bs)]
        logit_outputs = [{} for _ in range(self.config.bs)]
        if pair_count == 0:
            return tuple(hidden_outputs), tuple(logit_outputs)
        input_sequence, query_fast_weights = self._prepare_batched_queries(
            fast_weights,
            pair_schedules,
            num_steps=self.config.triallen,
        )
        query_batch_size = pair_count * self.config.bs
        with torch.no_grad(), self._alpha_zeroed(alpha_zero):
            hidden, logits = self.trajectory_runner(
                input_sequence,
                self.net.initial_hidden(query_batch_size),
                self.net.initial_eligibility(query_batch_size),
                query_fast_weights,
            )
        hidden_values = (
            hidden.detach()
            .cpu()
            .numpy()
            .reshape(pair_count, self.config.bs, self.config.triallen, self.config.hs)
        )
        logit_values = (
            logits.detach()
            .cpu()
            .numpy()
            .reshape(pair_count, self.config.bs, self.config.triallen)
        )
        for pair_index in range(pair_count):
            for subject in range(self.config.bs):
                pair = pair_schedules[subject][pair_index]
                hidden_outputs[subject][pair] = hidden_values[
                    pair_index, subject
                ].copy()
                logit_outputs[subject][pair] = logit_values[pair_index, subject].copy()
        return tuple(hidden_outputs), tuple(logit_outputs)

    def condition_evaluation(
        self, intervention: FastWeightIntervention
    ) -> tuple[ConditionMetrics, tuple[dict[tuple[int, int], int], ...]]:
        fast_weights = self.learn_fast_weights(intervention)
        canonical = tuple(combinations(range(self.protocol.n_items), 2))
        ordered_pairs = tuple(
            oriented
            for first, second in canonical
            for oriented in ((first, second), (second, first))
        )
        schedules = tuple(ordered_pairs for _ in range(self.config.bs))
        logits = self.readout_logits(
            fast_weights,
            schedules,
            alpha_zero=intervention == FastWeightIntervention.ALPHA_ZERO,
        )
        learned = self.protocol.learned_pairs
        subject_overall = []
        subject_learned = []
        subject_nonlearned = []
        subject_probability = []
        subject_cycles = []
        subject_winners = []
        for subject_logits in logits:
            correct = []
            correct_learned = []
            correct_nonlearned = []
            probabilities = []
            winners = {}
            for pair in canonical:
                forward_logit = subject_logits[pair]
                reverse_logit = subject_logits[(pair[1], pair[0])]
                first_is_higher = self.item_rank[pair[0]] < self.item_rank[pair[1]]
                if first_is_higher:
                    pair_correct = (
                        float(forward_logit > 0.0),
                        float(reverse_logit < 0.0),
                    )
                    pair_probability = (
                        float(1.0 / (1.0 + np.exp(-forward_logit))),
                        float(1.0 / (1.0 + np.exp(reverse_logit))),
                    )
                else:
                    pair_correct = (
                        float(forward_logit < 0.0),
                        float(reverse_logit > 0.0),
                    )
                    pair_probability = (
                        float(1.0 / (1.0 + np.exp(forward_logit))),
                        float(1.0 / (1.0 + np.exp(-reverse_logit))),
                    )
                correct.extend(pair_correct)
                probabilities.extend(pair_probability)
                target = correct_learned if pair in learned else correct_nonlearned
                target.extend(pair_correct)
                canonical_margin = 0.5 * (forward_logit - reverse_logit)
                winners[pair] = pair[0] if canonical_margin > 0.0 else pair[1]
            cycles = _count_circular_triads(winners, self.protocol.n_items)
            subject_overall.append(np.mean(correct))
            subject_learned.append(np.mean(correct_learned))
            subject_nonlearned.append(np.mean(correct_nonlearned))
            subject_probability.append(np.mean(probabilities))
            subject_cycles.append(cycles)
            subject_winners.append(winners)
        n_triads = len(tuple(combinations(range(self.protocol.n_items), 3)))
        return (
            ConditionMetrics(
                intervention=intervention.value,
                overall_accuracy=float(np.mean(subject_overall)),
                learned_accuracy=float(np.mean(subject_learned)),
                nonlearned_accuracy=float(np.mean(subject_nonlearned)),
                mean_probability_correct=float(np.mean(subject_probability)),
                mean_abs_fast_weight=float(torch.mean(torch.abs(fast_weights)).cpu()),
                mean_circular_triads=float(np.mean(subject_cycles)),
                mean_transitive_triplet_fraction=float(
                    1.0 - np.mean(subject_cycles) / n_triads
                ),
            ),
            tuple(subject_winners),
        )

    def condition_metrics(
        self, intervention: FastWeightIntervention
    ) -> ConditionMetrics:
        metrics, _winners = self.condition_evaluation(intervention)
        return metrics

    def order_invariance(
        self, fast_weights: torch.Tensor, schedules: int, seed: int
    ) -> OrderInvarianceMetrics:
        if schedules < 2:
            raise ValueError("order invariance requires at least two schedules")
        canonical = tuple(combinations(range(self.protocol.n_items), 2))
        ordered_pairs = tuple(
            oriented
            for first, second in canonical
            for oriented in ((first, second), (second, first))
        )
        rng = np.random.default_rng(seed)
        runs = []
        for _ in range(schedules):
            order = rng.permutation(len(ordered_pairs))
            schedule = tuple(ordered_pairs[int(index)] for index in order)
            batch_schedules = tuple(schedule for _ in range(self.config.bs))
            runs.append(self.readout_logits(fast_weights, batch_schedules))

        deltas = []
        reference = runs[0]
        for run in runs[1:]:
            for subject in range(self.config.bs):
                for pair in ordered_pairs:
                    deltas.append(abs(reference[subject][pair] - run[subject][pair]))
        return OrderInvarianceMetrics(
            schedules=schedules,
            pairs=len(ordered_pairs),
            max_abs_logit_delta=float(max(deltas, default=0.0)),
            mean_abs_logit_delta=float(np.mean(deltas)) if deltas else 0.0,
        )


def _count_circular_triads(winners: dict[tuple[int, int], int], n_items: int) -> int:
    cycles = 0
    for a, b, c in combinations(range(n_items), 3):
        ab = winners[(a, b)]
        ac = winners[(a, c)]
        bc = winners[(b, c)]
        if (ab == a and bc == b and ac == c) or (ab == b and bc == c and ac == a):
            cycles += 1
    return cycles


def run_causal_suite(
    checkpoint: Path | str,
    *,
    batch_size: int,
    cue_seed: int,
    support_seed: int,
    order_seed: int,
    order_schedules: int,
    cue_mode: str,
    subject_encoding_mode: str,
    subject_encoding_seed: int,
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
    evaluation_backend: FrozenEvaluationBackend | str = (
        FrozenEvaluationBackend.BATCHED_SEQUENCE
    ),
    execution_profile: ExecutionProfile | None = None,
) -> dict:
    protocol_path = Path(protocol_path)
    protocol = load_ranking_protocol(protocol_path)
    backend = FrozenEvaluationBackend(evaluation_backend)
    runtime = None
    if backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
        selected_device = default_device()
        execution_profile = execution_profile or ExecutionProfile(
            device=selected_device,
            compile=selected_device == "cuda",
            require_cuda=selected_device == "cuda",
        )
        runtime = configure_runtime(execution_profile)
    net, config, checkpoint_info = load_retro_checkpoint(
        checkpoint,
        batch_size,
        device=(execution_profile.device if execution_profile is not None else None),
    )
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=cue_seed,
        support_seed=support_seed,
        cue_mode=cue_mode,
        subject_encoding_mode=subject_encoding_mode,
        subject_encoding_seed=subject_encoding_seed,
        backend=backend,
        execution_profile=execution_profile,
    )
    conditions = {}
    condition_winners = {}
    for intervention in FastWeightIntervention:
        metrics, winners = evaluator.condition_evaluation(intervention)
        conditions[intervention.value] = asdict(metrics)
        condition_winners[intervention.value] = winners
    intact_winners = condition_winners[FastWeightIntervention.INTACT.value]
    for intervention, winners_by_subject in condition_winners.items():
        agreements = []
        for subject, winners in enumerate(winners_by_subject):
            agreements.extend(
                int(winner == intact_winners[subject][pair])
                for pair, winner in winners.items()
            )
        conditions[intervention]["mean_pair_decision_agreement_to_intact"] = float(
            np.mean(agreements)
        )
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    invariance = evaluator.order_invariance(
        intact_fast_weights, schedules=order_schedules, seed=order_seed
    )
    provenance = load_training_provenance(Path(checkpoint), checkpoint_info.sha256)
    result = {
        "protocol_id": protocol.protocol_id,
        "protocol_path": str(protocol_path.resolve()),
        "checkpoint": asdict(checkpoint_info),
        "batch_size": batch_size,
        "cue_seed": cue_seed,
        "cue_mode": cue_mode,
        "subject_encoding": {
            "mode": subject_encoding_mode,
            "seed": subject_encoding_seed,
            "configuration": SubjectEncodingConfig().to_dict(),
        },
        "training_provenance": provenance,
        "support_seed": support_seed,
        "conditions": conditions,
        "order_invariance": asdict(invariance),
    }
    if evaluator.backend != FrozenEvaluationBackend.LEGACY_STEPWISE:
        result["evaluation_execution"] = evaluator.evaluation_execution_record()
        result["evaluation_execution"]["runtime"] = runtime
    return result


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered fast-weight causal qualification suite."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cue-seed", type=int, default=1)
    parser.add_argument(
        "--cue-mode", choices=["shared", "permuted_shared"], default="permuted_shared"
    )
    parser.add_argument(
        "--subject-encoding",
        choices=[
            "none",
            "stable_bottleneck",
            "stable_omission",
            "presentationwise_omission",
            "blockwise_omission",
            "uniform_no_bottleneck",
        ],
        default="stable_omission",
    )
    parser.add_argument("--subject-encoding-seed", type=int, default=300)
    parser.add_argument("--support-seed", type=int, default=100)
    parser.add_argument("--order-seed", type=int, default=200)
    parser.add_argument("--order-schedules", type=int, default=8)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--evaluation-backend",
        choices=[backend.value for backend in FrozenEvaluationBackend],
        default=FrozenEvaluationBackend.BATCHED_SEQUENCE.value,
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_causal_suite(
        parsed.checkpoint,
        batch_size=parsed.batch_size,
        cue_seed=parsed.cue_seed,
        support_seed=parsed.support_seed,
        order_seed=parsed.order_seed,
        order_schedules=parsed.order_schedules,
        cue_mode=parsed.cue_mode,
        subject_encoding_mode=parsed.subject_encoding,
        subject_encoding_seed=parsed.subject_encoding_seed,
        protocol_path=parsed.protocol,
        evaluation_backend=parsed.evaluation_backend,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
