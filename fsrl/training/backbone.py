"""Supervised outer-loop meta-training on generic sparse ranking graphs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.config import NUMRESPONSESTEP, TrainConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.provenance import file_sha256
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import (
    DEFAULT_COMPILED_PROFILE,
    ExecutionProfile,
    begin_compiled_iteration,
    compile_module,
    configure_runtime,
    default_device,
    uses_cuda_graphs,
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
class MetaBatchMetrics:
    loss: float
    query_cross_entropy: float
    query_accuracy: float
    mean_abs_fast_weight: float
    n_edges: int


@dataclass(frozen=True)
class MetaBatchStats:
    loss: torch.Tensor
    diagnostics: torch.Tensor
    query_count: int
    n_edges: int

    def materialize_metrics(self) -> MetaBatchMetrics:
        loss, query_cross_entropy, correct_count, mean_abs_fast_weight = (
            self.diagnostics.detach().cpu().tolist()
        )
        return MetaBatchMetrics(
            loss=loss,
            query_cross_entropy=query_cross_entropy,
            query_accuracy=correct_count / self.query_count,
            mean_abs_fast_weight=mean_abs_fast_weight,
            n_edges=self.n_edges,
        )


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

OPTIMIZED_TRAINING_PROFILE = replace(
    DEFAULT_COMPILED_PROFILE,
    compile_mode="reduce-overhead",
)

OPTIMIZED_COMPILED_TRAINING_EXECUTION = {
    "execution_schema_version": 3,
    "runtime_profile": OPTIMIZED_TRAINING_PROFILE.to_dict(),
    "torch_compile": {
        "enabled": True,
        "backend": "inductor",
        "fullgraph": True,
        "mode": "reduce-overhead",
    },
    "cuda_graph_iteration_boundary": "explicit_outer_step",
    "compile_scope": "complete_recurrent_trial_sequence",
    "query_batching": "all_queries_vectorized_with_frozen_fast_weights",
    "item_code_sampling": "sequential_candidates_vectorized_similarity_check",
    "host_trial_sequence_assembly": "single_preallocated_numpy_array",
    "input_transfer": (
        "one_support_batch_and_one_query_batch_cpu_to_cuda_transfer_per_meta_batch"
    ),
    "target_transfer": "one_query_target_cpu_to_cuda_transfer_per_meta_batch",
    "metric_transfer": "one_batched_cuda_to_cpu_transfer_per_meta_batch",
    "metric_transfer_phase": "after_optimizer_step",
    "training_log_writer": "one_exclusive_buffered_open_per_execution",
}


def registered_excluded_signatures() -> frozenset[GraphSignature]:
    """Resolve the explicit current two-protocol holdout contract."""

    from fsrl.tasks.holdouts import registered_holdout_signatures

    return registered_holdout_signatures()


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
        "execution_schema_version": 3,
        "runtime_profile": profile.to_dict(),
        "torch_compile": {
            "enabled": True,
            "backend": profile.compile_backend,
            "fullgraph": profile.compile_fullgraph,
            "mode": profile.compile_mode,
        },
        "cuda_graph_iteration_boundary": (
            "explicit_outer_step" if uses_cuda_graphs(profile) else "not_applicable"
        ),
        "compile_scope": "complete_recurrent_trial_sequence",
        "query_batching": "all_queries_vectorized_with_frozen_fast_weights",
        "item_code_sampling": "sequential_candidates_vectorized_similarity_check",
        "host_trial_sequence_assembly": "single_preallocated_numpy_array",
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
        "metric_transfer_phase": "after_optimizer_step",
        "training_log_writer": "one_exclusive_buffered_open_per_execution",
    }


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
    inputs = np.zeros(
        (num_steps, len(episodes), model_config.inputsize), dtype=np.float32
    )
    inputs[:, :, model_config.nbstimbits] = 1.0
    inputs[:, :, model_config.nbstimbits + 1] = time_value
    for subject, episode in enumerate(episodes):
        inputs[0, subject, : model_config.cs] = episode.item_codes[left_items[subject]]
        inputs[0, subject, model_config.cs : 2 * model_config.cs] = episode.item_codes[
            right_items[subject]
        ]
    if support_trial:
        layout = RelationalInputLayout(model_config.cs)
        inputs[0, :, layout.evidence_index] = signed_magnitudes
    if num_steps > NUMRESPONSESTEP:
        inputs[NUMRESPONSESTEP, :, model_config.nbstimbits - 1] = 1.0
    return torch.from_numpy(inputs).to(device or default_device())


def _build_meta_input_sequence_from_codes(
    model_config: TrainConfig,
    item_codes: np.ndarray,
    left_items: np.ndarray,
    right_items: np.ndarray,
    signed_magnitudes: np.ndarray,
    *,
    num_steps: int,
    time_value: float,
    support_trial: bool,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    batch_size = len(left_items)
    if item_codes.ndim != 3 or item_codes.shape[0] != batch_size:
        raise ValueError("item codes do not align with the meta-batch")
    inputs = np.zeros((num_steps, batch_size, model_config.inputsize), dtype=np.float32)
    inputs[:, :, model_config.nbstimbits] = 1.0
    inputs[:, :, model_config.nbstimbits + 1] = time_value
    subjects = np.arange(batch_size)
    inputs[0, :, : model_config.cs] = item_codes[subjects, left_items]
    inputs[0, :, model_config.cs : 2 * model_config.cs] = item_codes[
        subjects, right_items
    ]
    if support_trial:
        layout = RelationalInputLayout(model_config.cs)
        inputs[0, :, layout.evidence_index] = signed_magnitudes
    if num_steps > NUMRESPONSESTEP:
        inputs[NUMRESPONSESTEP, :, model_config.nbstimbits - 1] = 1.0
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
    hidden = net.initial_hidden(model_config.bs)
    eligibility = net.initial_eligibility(model_config.bs)
    fast_weights = net.initial_fast_weights(model_config.bs)
    device = next(net.parameters()).device
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=device)
    for _ in range(2):
        _, _, _, hidden, eligibility, fast_weights = net(
            blank, hidden, eligibility, fast_weights
        )

    n_support = len(episodes[0].support_trials)
    zero_hidden = net.initial_hidden(model_config.bs)
    zero_eligibility = net.initial_eligibility(model_config.bs)
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
    diagnostics = torch.stack(
        (
            loss.detach(),
            query_loss.detach(),
            torch.tensor(correct, dtype=query_loss.dtype, device=device),
            torch.mean(torch.abs(fast_weights)).detach(),
        )
    )
    return MetaBatchStats(
        loss=loss,
        diagnostics=diagnostics,
        query_count=total,
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
    """Run the prospective vectorized batch execution recorded by schema v3."""

    n_edges = int(
        rng.integers(training_config.min_edges, training_config.max_edges + 1)
    )
    episodes = tuple(
        task_generator.sample(rng, n_edges=n_edges)
        for _ in range(training_config.batch_size)
    )
    episode_item_codes = np.stack([episode.item_codes for episode in episodes])
    hidden = net.initial_hidden(model_config.bs)
    eligibility = net.initial_eligibility(model_config.bs)
    fast_weights = net.initial_fast_weights(model_config.bs)
    device = next(net.parameters()).device
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=device)
    blank_sequence = blank.unsqueeze(0).expand(2, -1, -1)
    _, _, _, hidden, eligibility, fast_weights = sequence_runner(
        blank_sequence, hidden, eligibility, fast_weights, True
    )

    n_support = len(episodes[0].support_trials)
    zero_hidden = net.initial_hidden(model_config.bs)
    zero_eligibility = net.initial_eligibility(model_config.bs)
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
            _build_meta_input_sequence_from_codes(
                model_config,
                episode_item_codes,
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
    query_item_codes = np.tile(episode_item_codes, (n_queries, 1, 1))
    input_sequence = _build_meta_input_sequence_from_codes(
        model_config,
        query_item_codes,
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
        net.initial_hidden(query_batch_size),
        net.initial_eligibility(query_batch_size),
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
            loss.detach(),
            query_loss.detach(),
            correct.detach().to(query_loss.dtype),
            torch.mean(torch.abs(fast_weights)).detach(),
        )
    )
    return MetaBatchStats(
        loss=loss,
        diagnostics=diagnostics,
        query_count=query_batch_size,
        n_edges=n_edges,
    )


def apply_meta_batch_update(
    training_config: MetaTrainConfig,
    model_config: TrainConfig,
    net: RetroModulRNN,
    training_net: RetroModulRNN,
    sequence_runner: nn.Module | None,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
) -> MetaBatchStats:
    """Apply one optimizer update without synchronizing diagnostics to the host."""

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
    return stats


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
    profile: ExecutionProfile = OPTIMIZED_TRAINING_PROFILE,
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
    runtime: dict | None = None,
    excluded_signatures: frozenset[GraphSignature] | None = None,
    checkpoint_filename: str = "net.pth",
) -> None:
    if excluded_signatures is None:
        excluded_signatures = registered_excluded_signatures()
    checkpoint_name = Path(checkpoint_filename)
    if checkpoint_name.name != checkpoint_filename or checkpoint_name.suffix != ".pth":
        raise ValueError("new checkpoints must use a basename ending in .pth")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name
    torch.save(net.state_dict(), checkpoint_path)
    metadata = {
        "schema_version": 2,
        "training": asdict(training_config),
        "completed_outer_steps": step + 1,
        "model": asdict(net.model_config),
        "checkpoint": {
            "schema_version": 1,
            "format": "pytorch_state_dict",
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
    if runtime is not None:
        metadata["runtime"] = runtime
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
    checkpoint_filename: str = "net.pth",
) -> None:
    if execution_profile is None:
        execution_device = default_device()
        profile = ExecutionProfile(
            device=execution_device,
            compile=compile_model or optimized_execution,
            compile_mode=(
                OPTIMIZED_TRAINING_PROFILE.compile_mode
                if optimized_execution and execution_device == "cuda"
                else DEFAULT_COMPILED_PROFILE.compile_mode
            ),
            require_cuda=False,
        )
    else:
        profile = execution_profile
    runtime = configure_runtime(profile)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model_config, net, task_generator = make_model_and_tasks(
        training_config,
        excluded_signatures=excluded_signatures,
        device=profile.device,
    )
    resolved_exclusions = task_generator.excluded_signatures
    run = ProspectiveRun.start(
        output_dir,
        workflow_id="relational_model",
        execution_id=output_dir.name,
        producer={
            "module": "fsrl.training.backbone",
            "callable": "train_meta_model",
        },
        resolved_config={
            "training": asdict(training_config),
            "execution_profile": profile.to_dict(),
            "execution_schema": "current" if optimized_execution else "historical",
            "checkpoint_filename": checkpoint_filename,
            "held_out_rank_graph_signatures": [
                [list(pair) for pair in signature]
                for signature in sorted(resolved_exclusions)
            ],
        },
    )
    with run:
        if optimized_execution:
            training_net = net
            sequence_runner = (
                compile_meta_sequence(net, profile)
                if profile.compile
                else RecurrentSequence(net)
            )
        else:
            training_net = compile_module(net, profile) if profile.compile else net
            sequence_runner = None
        optimizer = torch.optim.Adam(
            training_net.parameters(), lr=training_config.learning_rate
        )
        log_path = output_dir / "train_log.jsonl"

        with log_path.open("x", encoding="utf-8") as log_handle:
            for step in range(training_config.outer_steps):
                begin_compiled_iteration(profile)
                stats = apply_meta_batch_update(
                    training_config,
                    model_config,
                    net,
                    training_net,
                    sequence_runner,
                    task_generator,
                    rng,
                    optimizer,
                )
                metrics = stats.materialize_metrics()
                record = {
                    "outer_step": step,
                    "loss": metrics.loss,
                    "query_cross_entropy": metrics.query_cross_entropy,
                    "query_accuracy": metrics.query_accuracy,
                    "mean_abs_fast_weight": metrics.mean_abs_fast_weight,
                    "n_edges": metrics.n_edges,
                }
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
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
                        runtime=runtime if optimized_execution else None,
                        excluded_signatures=resolved_exclusions,
                        checkpoint_filename=checkpoint_filename,
                    )
