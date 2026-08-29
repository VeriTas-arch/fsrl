"""Strict frozen-state and fast-weight causal evaluation for Liu-style tasks."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch

from fsrl.core.config import NUMRESPONSESTEP, TrainConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.runtime import (
    ExecutionProfile,
    compile_module,
)
from fsrl.tasks.protocol import RankingProtocol

from ..core.inputs import EVIDENCE_AUXILIARY_OFFSET
from ..core.plastic_rnn import RetroModulRNN
from ..training.checkpoints import (
    CheckpointInfo,
    checkpoint_sha256,
    load_training_provenance,
)
from ..training.legacy_checkpoints import load_frozen_retro_checkpoint
from .contracts import (
    ConditionMetrics,
    FastWeightIntervention,
    FrozenEvaluationBackend,
    OrderInvarianceMetrics,
)
from .execution import RecurrentTrajectory, evaluation_execution_record
from .metrics import count_circular_triads
from .sampling import deterministic_cue_codes, retained_relation_mask
from .subject_encoding import (
    build_frozen_subject_encoding,
    realized_support_evidence,
)

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
    "load_frozen_retro_checkpoint",
    "load_training_provenance",
    "retained_relation_mask",
]


class FrozenFastWeightEvaluator:
    """Evaluate one shared network under explicit fast-weight interventions."""

    def __init__(
        self,
        net: RetroModulRNN | None,
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
        protocol_only: bool = False,
        required_item_count: int | None = 8,
    ) -> None:
        if config.bs < 1:
            raise ValueError("batch size must be positive")
        if net is None and not protocol_only:
            raise ValueError("a neural network is required outside protocol-only mode")
        if required_item_count is not None and required_item_count < 2:
            raise ValueError("required_item_count must be at least two or None")
        if required_item_count is not None and protocol.n_items != required_item_count:
            raise ValueError(
                f"the evaluator requires {required_item_count} protocol items"
            )
        self._net = net
        self.config = config
        self.protocol = protocol
        self.item_rank = {
            item: position
            for position, item in enumerate(protocol.true_order_high_to_low)
        }
        self.test_time_value = float(test_time_value)
        self.backend = FrozenEvaluationBackend(backend)
        if net is None and self.backend != FrozenEvaluationBackend.LEGACY_STEPWISE:
            raise ValueError("protocol-only mode requires the legacy backend")
        if self.backend == FrozenEvaluationBackend.LEGACY_STEPWISE:
            if execution_profile is not None:
                raise ValueError(
                    "execution_profile is only valid for the batched evaluator"
                )
            self.execution_profile = None
            self.sequence_runner = None
            self.trajectory_runner = None
        else:
            if net is None:
                raise ValueError("batched execution requires a neural network")
            profile = execution_profile or ExecutionProfile(
                device=self.device.type,
                compile=self.device.type == "cuda",
                require_cuda=self.device.type == "cuda",
            )
            if profile.device != self.device.type:
                raise ValueError("evaluation profile and network device differ")
            self.execution_profile = profile
            self.sequence_runner = compile_module(RecurrentSequence(net), profile)
            self.trajectory_runner = compile_module(RecurrentTrajectory(net), profile)
        self.cue_codes = deterministic_cue_codes(
            config.bs, protocol.n_items, config.cs, cue_seed, mode=cue_mode
        )
        self.cue_mode = cue_mode
        self.support_schedules = tuple(
            protocol.support_schedule(np.random.default_rng(support_seed + subject))
            for subject in range(config.bs)
        )
        encoding = build_frozen_subject_encoding(
            config,
            protocol,
            self.item_rank,
            self.support_schedules,
            mode=subject_encoding_mode,
            seed=subject_encoding_seed,
        )
        self.subject_encoding_states = encoding.states
        self.subject_relation_gains = encoding.relation_gains
        self.subject_trial_gains = encoding.trial_gains
        self.subject_encoding_mode = subject_encoding_mode
        self.subject_encoding_seed = subject_encoding_seed

    @property
    def net(self) -> RetroModulRNN:
        """Return the neural module, rejecting rollouts in protocol-only mode."""

        if self._net is None:
            raise RuntimeError(
                "protocol-only evaluators cannot execute neural rollouts"
            )
        return self._net

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
        return evaluation_execution_record(self.backend, self.execution_profile)

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

        return realized_support_evidence(
            self.support_schedules, self.subject_trial_gains
        )

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
            cycles = count_circular_triads(winners, self.protocol.n_items)
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
