"""Supervised outer-loop meta-training on generic sparse ranking graphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.config import NUMRESPONSESTEP, TrainConfig
from fsrl.infra.provenance import file_sha256
from fsrl.infra.runtime import (
    DEFAULT_COMPILED_PROFILE,
    ExecutionProfile,
    compile_module,
    configure_runtime,
    default_device,
)
from fsrl.tasks.subject_encoding import SubjectEncodingConfig

from ..core.inputs import RelationalInputLayout
from ..core.plastic_rnn import RetroModulRNN
from ..tasks.sparse_ranking import (
    GenericRankingTaskGenerator,
    GraphSignature,
    RankingEpisode,
)


@dataclass(frozen=True)
class MetaTrainConfig:
    seed: int = 1
    outer_steps: int = 30000
    batch_size: int = 32
    hidden_size: int = 200
    cue_size: int = 15
    min_edges: int = 7
    max_edges: int = 10
    support_blocks: int = 4
    learning_rate: float = 1e-4
    gradient_clip: float = 2.0
    fast_weight_penalty: float = 1e-4
    support_query_time: float = 2.0 / 3.0
    save_every: int = 500
    subject_encoding_mode: str = "stable_omission"


@dataclass(frozen=True)
class MetaBatchStats:
    loss: torch.Tensor
    query_cross_entropy: float
    query_accuracy: float
    mean_abs_fast_weight: float
    n_edges: int


# Frozen execution record for the already registered backbones. Do not rewrite it
# when prospective training execution changes.
COMPILED_TRAINING_EXECUTION = {
    "torch_compile": {
        "enabled": True,
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
    },
    "trial_input_transfer": "one_contiguous_cpu_to_gpu_transfer_per_trial",
}

OPTIMIZED_COMPILED_TRAINING_EXECUTION = {
    "execution_schema_version": 2,
    "torch_compile": {
        "enabled": True,
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
    },
    "compile_scope": "complete_recurrent_trial_sequence",
    "query_batching": "all_queries_vectorized_with_frozen_fast_weights",
    "input_transfer": (
        "one_support_batch_and_one_query_batch_cpu_to_cuda_transfer_per_meta_batch"
    ),
    "target_transfer": "one_query_target_cpu_to_cuda_transfer_per_meta_batch",
    "metric_transfer": "one_batched_cuda_to_cpu_transfer_per_meta_batch",
}


def registered_excluded_signatures() -> frozenset[GraphSignature]:
    """Resolve the historical two-graph holdout only at the workflow boundary."""

    from fsrl.tasks.meta_tasks import held_out_liu_graph_signatures

    return held_out_liu_graph_signatures()


def compiled_execution_record(profile: ExecutionProfile) -> dict:
    transfer_device = "gpu" if profile.device == "cuda" else profile.device
    return {
        "torch_compile": {
            "enabled": True,
            "backend": profile.compile_backend,
            "fullgraph": profile.compile_fullgraph,
            "mode": profile.compile_mode,
        },
        "trial_input_transfer": (
            f"one_contiguous_cpu_to_{transfer_device}_transfer_per_trial"
        ),
    }


def optimized_compiled_execution_record(profile: ExecutionProfile) -> dict:
    return {
        "execution_schema_version": 2,
        "torch_compile": {
            "enabled": True,
            "backend": profile.compile_backend,
            "fullgraph": profile.compile_fullgraph,
            "mode": profile.compile_mode,
        },
        "compile_scope": "complete_recurrent_trial_sequence",
        "query_batching": "all_queries_vectorized_with_frozen_fast_weights",
        "input_transfer": (
            "one_support_batch_and_one_query_batch_cpu_to_"
            f"{profile.device}_transfer_per_meta_batch"
        ),
        "target_transfer": (
            f"one_query_target_cpu_to_{profile.device}_transfer_per_meta_batch"
        ),
        "metric_transfer": (
            f"one_batched_{profile.device}_to_cpu_transfer_per_meta_batch"
        ),
    }


class RecurrentSequence(nn.Module):
    """Execute one complete trial while preserving its fast-weight policy."""

    def __init__(self, cell: RetroModulRNN):
        super().__init__()
        self.cell = cell

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden: torch.Tensor,
        eligibility: torch.Tensor,
        fast_weights: torch.Tensor,
        update_fast_weights: bool,
    ):
        logits = value = dopamine = None
        for inputs in input_sequence.unbind(0):
            (
                logits,
                value,
                dopamine,
                hidden,
                eligibility,
                proposed_fast_weights,
            ) = self.cell(inputs, hidden, eligibility, fast_weights)
            if update_fast_weights:
                fast_weights = proposed_fast_weights
        return logits, value, dopamine, hidden, eligibility, fast_weights


def build_meta_inputs(
    model_config: TrainConfig,
    episodes: tuple[RankingEpisode, ...],
    left_items: np.ndarray,
    right_items: np.ndarray,
    signed_magnitudes: np.ndarray,
    *,
    numstep: int,
    time_value: float,
    support_trial: bool,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Build passive inputs; query targets are deliberately absent from this API."""

    inputs = _build_meta_input_array(
        model_config,
        episodes,
        left_items,
        right_items,
        signed_magnitudes,
        numstep=numstep,
        time_value=time_value,
        support_trial=support_trial,
    )
    return torch.from_numpy(inputs).to(device or default_device())


def _build_meta_input_array(
    model_config: TrainConfig,
    episodes: tuple[RankingEpisode, ...],
    left_items: np.ndarray,
    right_items: np.ndarray,
    signed_magnitudes: np.ndarray,
    *,
    numstep: int,
    time_value: float,
    support_trial: bool,
) -> np.ndarray:
    inputs = np.zeros((len(episodes), model_config.inputsize), dtype=np.float32)
    inputs[:, model_config.nbstimbits] = 1.0
    inputs[:, model_config.nbstimbits + 1] = time_value
    if numstep == 0:
        for subject, episode in enumerate(episodes):
            inputs[subject, : 2 * model_config.cs] = np.concatenate(
                (
                    episode.item_codes[left_items[subject]],
                    episode.item_codes[right_items[subject]],
                )
            )
        if support_trial:
            layout = RelationalInputLayout(model_config.cs)
            inputs[:, layout.evidence_index] = signed_magnitudes
    elif numstep == NUMRESPONSESTEP:
        inputs[:, model_config.nbstimbits - 1] = 1.0
    return inputs


def build_meta_input_sequence(
    model_config: TrainConfig,
    episodes: tuple[RankingEpisode, ...],
    left_items: np.ndarray,
    right_items: np.ndarray,
    signed_magnitudes: np.ndarray,
    *,
    num_steps: int,
    time_value: float,
    support_trial: bool,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    inputs = np.stack(
        [
            _build_meta_input_array(
                model_config,
                episodes,
                left_items,
                right_items,
                signed_magnitudes,
                numstep=numstep,
                time_value=time_value,
                support_trial=support_trial,
            )
            for numstep in range(num_steps)
        ]
    )
    return torch.from_numpy(inputs).to(device or default_device())


def run_meta_batch(
    training_config: MetaTrainConfig,
    model_config: TrainConfig,
    net: RetroModulRNN,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
) -> MetaBatchStats:
    """Run the frozen stepwise batch execution used by registered backbones."""

    n_edges = int(
        rng.integers(training_config.min_edges, training_config.max_edges + 1)
    )
    episodes = tuple(
        task_generator.sample(rng, n_edges=n_edges)
        for _ in range(training_config.batch_size)
    )
    hidden = net.initialZeroState(model_config.bs)
    eligibility = net.initialZeroET(model_config.bs)
    fast_weights = net.initialZeroPlasticWeights(model_config.bs)
    device = next(net.parameters()).device
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=device)
    for _ in range(2):
        _, _, _, hidden, eligibility, fast_weights = net(
            blank, hidden, eligibility, fast_weights
        )

    n_support = len(episodes[0].support_trials)
    zero_hidden = net.initialZeroState(model_config.bs)
    zero_eligibility = net.initialZeroET(model_config.bs)
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
            _, _, _, hidden, eligibility, fast_weights = net(
                inputs, hidden, eligibility, fast_weights
            )

    query_loss = torch.zeros((), device=device)
    correct = 0
    total = 0
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
        response_logits = None
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
        for numstep, inputs in enumerate(input_sequence.unbind()):
            logits, _, _, hidden, eligibility, _proposed = net(
                inputs, hidden, eligibility, fast_weights
            )
            if numstep == NUMRESPONSESTEP:
                response_logits = logits
        assert response_logits is not None
        query_loss = query_loss + F.cross_entropy(response_logits, targets)
        correct += int(torch.sum(torch.argmax(response_logits, dim=1) == targets))
        total += model_config.bs

    query_loss = query_loss / n_queries
    loss = query_loss + training_config.fast_weight_penalty * torch.mean(
        fast_weights**2
    )
    return MetaBatchStats(
        loss=loss,
        query_cross_entropy=float(query_loss.detach()),
        query_accuracy=correct / total,
        mean_abs_fast_weight=float(torch.mean(torch.abs(fast_weights)).detach()),
        n_edges=n_edges,
    )


def run_optimized_meta_batch(
    training_config: MetaTrainConfig,
    model_config: TrainConfig,
    net: RetroModulRNN,
    sequence_runner: nn.Module,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
) -> MetaBatchStats:
    """Run the prospective vectorized batch execution recorded by schema v2."""

    n_edges = int(
        rng.integers(training_config.min_edges, training_config.max_edges + 1)
    )
    episodes = tuple(
        task_generator.sample(rng, n_edges=n_edges)
        for _ in range(training_config.batch_size)
    )
    hidden = net.initialZeroState(model_config.bs)
    eligibility = net.initialZeroET(model_config.bs)
    fast_weights = net.initialZeroPlasticWeights(model_config.bs)
    device = next(net.parameters()).device
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=device)
    blank_sequence = blank.unsqueeze(0).expand(2, -1, -1)
    _, _, _, hidden, eligibility, fast_weights = sequence_runner(
        blank_sequence, hidden, eligibility, fast_weights, True
    )

    n_support = len(episodes[0].support_trials)
    zero_hidden = net.initialZeroState(model_config.bs)
    zero_eligibility = net.initialZeroET(model_config.bs)
    support_input_sequences = []
    for trial_index in range(n_support):
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
        support_input_sequences.append(
            build_meta_input_sequence(
                model_config,
                episodes,
                left,
                right,
                signed,
                num_steps=model_config.triallen,
                time_value=time_value,
                support_trial=True,
                device="cpu",
            )
        )
    support_input_batch = torch.stack(support_input_sequences).to(device)
    for input_sequence in support_input_batch.unbind(0):
        _, _, _, hidden, eligibility, fast_weights = sequence_runner(
            input_sequence,
            zero_hidden,
            zero_eligibility,
            fast_weights,
            True,
        )

    n_queries = len(episodes[0].query_trials)
    query_episodes = episodes * n_queries
    query_trials = [
        episode.query_trials[query_index]
        for query_index in range(n_queries)
        for episode in episodes
    ]
    left = np.asarray([trial.left_item for trial in query_trials], dtype=np.int64)
    right = np.asarray([trial.right_item for trial in query_trials], dtype=np.int64)
    targets = torch.tensor(
        [trial.correct_action for trial in query_trials],
        dtype=torch.long,
        device=device,
    )
    query_batch_size = n_queries * model_config.bs
    input_sequence = build_meta_input_sequence(
        model_config,
        query_episodes,
        left,
        right,
        np.zeros(query_batch_size, dtype=np.float32),
        num_steps=NUMRESPONSESTEP + 1,
        time_value=training_config.support_query_time,
        support_trial=False,
        device=device,
    )
    query_fast_weights = (
        fast_weights.unsqueeze(0)
        .expand(n_queries, -1, -1, -1)
        .reshape(query_batch_size, model_config.hs, model_config.hs)
    )
    response_logits, _, _, _, _, _ = sequence_runner(
        input_sequence,
        net.initialZeroState(query_batch_size),
        net.initialZeroET(query_batch_size),
        query_fast_weights,
        False,
    )
    query_loss = F.cross_entropy(response_logits, targets)
    correct = torch.sum(torch.argmax(response_logits, dim=1) == targets)
    loss = query_loss + training_config.fast_weight_penalty * torch.mean(
        fast_weights**2
    )
    diagnostics = torch.stack(
        (
            query_loss.detach(),
            correct.detach().to(query_loss.dtype),
            torch.mean(torch.abs(fast_weights)).detach(),
        )
    ).cpu()
    query_cross_entropy, correct_count, mean_abs_fast_weight = diagnostics.tolist()
    return MetaBatchStats(
        loss=loss,
        query_cross_entropy=query_cross_entropy,
        query_accuracy=correct_count / query_batch_size,
        mean_abs_fast_weight=mean_abs_fast_weight,
        n_edges=n_edges,
    )


def make_model_and_tasks(
    training_config: MetaTrainConfig,
    *,
    excluded_signatures: frozenset[GraphSignature] | None = None,
    device: str | torch.device | None = None,
):
    model_config = TrainConfig(
        rngseed=training_config.seed,
        bs=training_config.batch_size,
        hs=training_config.hidden_size,
        cs=training_config.cue_size,
        nbtraintrials=training_config.support_blocks * training_config.max_edges,
        nbtesttrials=28,
        nbcues_min=8,
        nbcues_max=8,
    )
    net = RetroModulRNN(model_config.to_model_dict(), device=device)
    if excluded_signatures is None:
        excluded_signatures = registered_excluded_signatures()
    task_generator = GenericRankingTaskGenerator(
        cue_size=training_config.cue_size,
        min_edges=training_config.min_edges,
        max_edges=training_config.max_edges,
        support_blocks=training_config.support_blocks,
        excluded_signatures=excluded_signatures,
        subject_encoding_mode=training_config.subject_encoding_mode,
    )
    return model_config, net, task_generator


def compile_meta_model(net: RetroModulRNN):
    """Compile the frozen single-cell execution used by registered backbones."""

    return compile_module(net, DEFAULT_COMPILED_PROFILE)


def compile_meta_sequence(
    net: RetroModulRNN,
    profile: ExecutionProfile = DEFAULT_COMPILED_PROFILE,
):
    """Compile the prospective complete-trial execution."""

    return compile_module(RecurrentSequence(net), profile)


def save_meta_checkpoint(
    output_dir: Path,
    net: RetroModulRNN,
    training_config: MetaTrainConfig,
    step: int,
    *,
    execution: dict | None = None,
    excluded_signatures: frozenset[GraphSignature] | None = None,
) -> None:
    if excluded_signatures is None:
        excluded_signatures = registered_excluded_signatures()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "net.dat"
    torch.save(net.state_dict(), checkpoint_path)
    metadata = {
        "schema_version": 1,
        "training": asdict(training_config),
        "completed_outer_steps": step + 1,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
        },
        "task_distribution": {
            "n_items": 8,
            "connected_sparse_graph": True,
            "liu_graph_held_out": True,
            "held_out_rank_graph_signatures": [
                [list(pair) for pair in signature]
                for signature in sorted(excluded_signatures)
            ],
            "held_out_graph_scope": (
                "source-correct Liu graph and its rank-axis reflection"
            ),
            "query_labels_enter_episode_inputs": False,
            "query_fast_weights": "frozen",
            "query_time_channel": "constant_at_support_query_boundary",
            "subject_encoding": {
                "state_scope": "fixed_for_entire_episode",
                "mode": training_config.subject_encoding_mode,
                "acts_on": "support_relation_retention",
                "contains_rank_label": False,
                "configuration": SubjectEncodingConfig().to_dict(),
            },
        },
    }
    if execution is not None:
        metadata["execution"] = execution
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def train_meta_model(
    training_config: MetaTrainConfig,
    output_dir: Path,
    *,
    compile_model: bool = False,
    optimized_execution: bool = False,
    execution_profile: ExecutionProfile | None = None,
    excluded_signatures: frozenset[GraphSignature] | None = None,
) -> None:
    profile = execution_profile or ExecutionProfile(
        device=default_device(),
        compile=compile_model or optimized_execution,
        require_cuda=False,
    )
    configure_runtime(profile)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model_config, net, task_generator = make_model_and_tasks(
        training_config,
        excluded_signatures=excluded_signatures,
        device=profile.device,
    )
    resolved_exclusions = task_generator.excluded_signatures
    if optimized_execution:
        if not profile.compile:
            raise ValueError("optimized execution requires torch.compile")
        training_net = net
        sequence_runner = compile_meta_sequence(net, profile)
    else:
        training_net = compile_module(net, profile) if profile.compile else net
        sequence_runner = None
    optimizer = torch.optim.Adam(
        training_net.parameters(), lr=training_config.learning_rate
    )
    log_path = output_dir / "train_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(training_config.outer_steps):
        optimizer.zero_grad()
        if sequence_runner is None:
            stats = run_meta_batch(
                training_config, model_config, training_net, task_generator, rng
            )
        else:
            stats = run_optimized_meta_batch(
                training_config,
                model_config,
                net,
                sequence_runner,
                task_generator,
                rng,
            )
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), training_config.gradient_clip)
        optimizer.step()
        record = {
            "outer_step": step,
            "loss": float(stats.loss.detach()),
            "query_cross_entropy": stats.query_cross_entropy,
            "query_accuracy": stats.query_accuracy,
            "mean_abs_fast_weight": stats.mean_abs_fast_weight,
            "n_edges": stats.n_edges,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if training_config.save_every > 0 and (
            (step + 1) % training_config.save_every == 0
            or step + 1 == training_config.outer_steps
        ):
            save_meta_checkpoint(
                output_dir,
                net,
                training_config,
                step,
                execution=(
                    (
                        optimized_compiled_execution_record(profile)
                        if optimized_execution
                        else compiled_execution_record(profile)
                    )
                    if profile.compile
                    else None
                ),
                excluded_signatures=resolved_exclusions,
            )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Meta-train a plastic RNN on generic sparse ranking graphs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--cue-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--optimized-execution",
        action="store_true",
        help=(
            "use the prospective vectorized trial-sequence execution; implies "
            "--compile-model"
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device())
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--subject-encoding",
        choices=["stable_attenuation", "stable_omission"],
        default="stable_omission",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    training_config = MetaTrainConfig(
        seed=parsed.seed,
        outer_steps=parsed.outer_steps,
        batch_size=parsed.batch_size,
        hidden_size=parsed.hidden_size,
        cue_size=parsed.cue_size,
        learning_rate=parsed.learning_rate,
        save_every=parsed.save_every,
        subject_encoding_mode=parsed.subject_encoding,
    )
    compile_model = parsed.compile_model or parsed.optimized_execution
    profile = ExecutionProfile(
        device=parsed.device,
        cpu_threads=parsed.cpu_threads,
        compile=compile_model,
        require_cuda=parsed.device == "cuda",
    )
    configure_runtime(profile)
    train_meta_model(
        training_config,
        parsed.output_dir,
        compile_model=compile_model,
        optimized_execution=parsed.optimized_execution,
        execution_profile=profile,
    )


if __name__ == "__main__":
    main()
