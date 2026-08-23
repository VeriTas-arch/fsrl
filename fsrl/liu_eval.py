"""Strict frozen-state and fast-weight causal evaluation for Liu-style tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .config import ADDINPUT, DEVICE, NUMRESPONSESTEP, TrainConfig
from .model import RetroModulRNN
from .ranking_protocol import RankingProtocol, SupportTrial, load_ranking_protocol
from .subject_encoding import (
    SubjectEncodingConfig,
    SubjectEncodingState,
    sample_subject_encoding_states,
)

DISTANCE_INPUT_OFFSET = 3


class FastWeightIntervention(str, Enum):
    INTACT = "intact"
    WRITE_OFF = "write_off"
    ALPHA_ZERO = "alpha_zero"
    RESET = "reset"
    SHUFFLE = "shuffle"


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    sha256: str
    hidden_size: int
    cue_size: int
    input_size: int


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


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_provenance(checkpoint: Path, checkpoint_hash: str) -> dict:
    metadata_path = checkpoint.parent / "config.json"
    if not metadata_path.is_file():
        return {"present": False}
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    registered_hash = metadata.get("checkpoint", {}).get("sha256")
    return {
        "present": True,
        "path": str(metadata_path.resolve()),
        "checkpoint_sha_matches": registered_hash == checkpoint_hash,
        "task_distribution": metadata.get("task_distribution"),
    }


def load_retro_checkpoint(
    path: Path | str, batch_size: int
) -> tuple[RetroModulRNN, TrainConfig, CheckpointInfo]:
    path = Path(path)
    try:
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=DEVICE)
    hidden_size, input_size = state_dict["i2h.weight"].shape
    cue_remainder = int(input_size) - (1 + ADDINPUT + 2)
    if cue_remainder <= 0 or cue_remainder % 2:
        raise ValueError(f"Checkpoint input size {input_size} has no valid cue size")
    cue_size = cue_remainder // 2
    config = TrainConfig(
        bs=batch_size,
        hs=int(hidden_size),
        cs=cue_size,
        nbcues_min=8,
        nbcues_max=8,
    )
    net = RetroModulRNN(config.to_model_dict())
    net.load_state_dict(state_dict)
    net.eval()
    info = CheckpointInfo(
        path=str(path.resolve()),
        sha256=checkpoint_sha256(path),
        hidden_size=int(hidden_size),
        cue_size=cue_size,
        input_size=int(input_size),
    )
    return net, config, info


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
        self.cue_codes = deterministic_cue_codes(
            config.bs, protocol.n_items, config.cs, cue_seed, mode=cue_mode
        )
        self.cue_mode = cue_mode
        self.subject_relation_gains: tuple[dict[tuple[int, int], float], ...] | None
        if subject_encoding_mode == "none":
            self.subject_encoding_states: tuple[SubjectEncodingState, ...] | None = None
            self.subject_relation_gains = None
        elif subject_encoding_mode in {"stable_bottleneck", "stable_omission"}:
            encoding_rng = np.random.default_rng(subject_encoding_seed)
            self.subject_encoding_states = sample_subject_encoding_states(
                encoding_rng,
                config.bs,
                protocol.n_items,
            )
            gains = []
            for state in self.subject_encoding_states:
                subject_gains = {}
                for higher, lower in protocol.support_pairs_higher_lower:
                    symbolic_distance = self.item_rank[lower] - self.item_rank[higher]
                    probability = state.relation_reliability(
                        higher, lower, symbolic_distance
                    )
                    subject_gains[(higher, lower)] = (
                        probability
                        if subject_encoding_mode == "stable_bottleneck"
                        else float(encoding_rng.random() < probability)
                    )
                gains.append(subject_gains)
            self.subject_relation_gains = tuple(gains)
        else:
            raise ValueError(f"unknown subject encoding mode: {subject_encoding_mode}")
        self.subject_encoding_mode = subject_encoding_mode
        self.subject_encoding_seed = subject_encoding_seed
        self.support_schedules = tuple(
            protocol.support_schedule(np.random.default_rng(support_seed + subject))
            for subject in range(config.bs)
        )

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
        return torch.from_numpy(inputs).to(DEVICE)

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
        hidden = self.net.initialZeroState(self.config.bs)
        eligibility = self.net.initialZeroET(self.config.bs)
        fast_weights = self.net.initialZeroPlasticWeights(self.config.bs)
        blank = torch.zeros(self.config.bs, self.config.inputsize, device=DEVICE)

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

            n_trials = self.protocol.support_trials
            for trial_index in range(n_trials):
                hidden = self.net.initialZeroState(self.config.bs)
                eligibility = self.net.initialZeroET(self.config.bs)
                trials = [schedule[trial_index] for schedule in self.support_schedules]
                left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
                right = np.asarray(
                    [trial.right_item for trial in trials], dtype=np.int64
                )
                signed = np.asarray(
                    [
                        trial.signed_magnitude
                        * self._encoding_reliability(subject, trial)
                        for subject, trial in enumerate(trials)
                    ],
                    dtype=np.float32,
                )
                trial_time = trial_index / max(1, n_trials - 1) * self.test_time_value
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

        if intervention == FastWeightIntervention.RESET:
            fast_weights = torch.zeros_like(fast_weights)
        elif intervention == FastWeightIntervention.SHUFFLE:
            if self.config.bs < 2:
                raise ValueError("shuffle intervention requires batch_size >= 2")
            fast_weights = torch.roll(fast_weights, shifts=1, dims=0)
        return fast_weights.detach().clone()

    def _encoding_reliability(self, subject: int, trial: SupportTrial) -> float:
        if self.subject_encoding_states is None:
            return 1.0
        assert self.subject_relation_gains is not None
        return self.subject_relation_gains[subject][
            (trial.higher_item, trial.lower_item)
        ]

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
        outputs = [{} for _ in range(self.config.bs)]

        with torch.no_grad(), self._alpha_zeroed(alpha_zero):
            for pair_index in range(next(iter(schedule_lengths))):
                hidden = self.net.initialZeroState(self.config.bs)
                eligibility = self.net.initialZeroET(self.config.bs)
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

    def readout_hidden_states(
        self,
        fast_weights: torch.Tensor,
        pair_schedules: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[dict[tuple[int, int], np.ndarray], ...]:
        """Return response-step hidden states under the strict frozen query protocol."""

        if len(pair_schedules) != self.config.bs:
            raise ValueError("one pair schedule is required per batch subject")
        schedule_lengths = {len(schedule) for schedule in pair_schedules}
        if len(schedule_lengths) != 1:
            raise ValueError("all pair schedules must have equal length")
        outputs = [{} for _ in range(self.config.bs)]
        with torch.no_grad():
            for pair_index in range(next(iter(schedule_lengths))):
                hidden = self.net.initialZeroState(self.config.bs)
                eligibility = self.net.initialZeroET(self.config.bs)
                left = np.asarray(
                    [schedule[pair_index][0] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                right = np.asarray(
                    [schedule[pair_index][1] for schedule in pair_schedules],
                    dtype=np.int64,
                )
                signed = np.zeros(self.config.bs, dtype=np.float32)
                response_hidden = None
                for numstep in range(self.config.triallen):
                    inputs = self._step_inputs(
                        left,
                        right,
                        signed,
                        numstep=numstep,
                        time_value=self.test_time_value,
                        support_trial=False,
                    )
                    _, _, _, hidden, eligibility, _proposed = self.net(
                        inputs, hidden, eligibility, fast_weights
                    )
                    if numstep == NUMRESPONSESTEP:
                        response_hidden = hidden.detach().cpu().numpy()
                assert response_hidden is not None
                for subject, state in enumerate(response_hidden):
                    outputs[subject][(int(left[subject]), int(right[subject]))] = (
                        state.copy()
                    )
        return tuple(outputs)

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
                pair_correct = (float(forward_logit > 0.0), float(reverse_logit < 0.0))
                pair_probability = (
                    float(1.0 / (1.0 + np.exp(-forward_logit))),
                    float(1.0 / (1.0 + np.exp(reverse_logit))),
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
) -> dict:
    protocol = load_ranking_protocol()
    net, config, checkpoint_info = load_retro_checkpoint(checkpoint, batch_size)
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=cue_seed,
        support_seed=support_seed,
        cue_mode=cue_mode,
        subject_encoding_mode=subject_encoding_mode,
        subject_encoding_seed=subject_encoding_seed,
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
    return {
        "protocol_id": protocol.protocol_id,
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
        choices=["none", "stable_bottleneck", "stable_omission"],
        default="stable_omission",
    )
    parser.add_argument("--subject-encoding-seed", type=int, default=300)
    parser.add_argument("--support-seed", type=int, default=100)
    parser.add_argument("--order-seed", type=int, default=200)
    parser.add_argument("--order-schedules", type=int, default=8)
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
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
