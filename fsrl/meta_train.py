"""Supervised outer-loop meta-training on generic sparse ranking graphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, NUMRESPONSESTEP, TrainConfig
from .liu_eval import DISTANCE_INPUT_OFFSET, checkpoint_sha256
from .meta_tasks import GenericRankingTaskGenerator, RankingEpisode
from .model import RetroModulRNN
from .subject_encoding import SubjectEncodingConfig


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


@dataclass(frozen=True)
class MetaBatchStats:
    loss: torch.Tensor
    query_cross_entropy: float
    query_accuracy: float
    mean_abs_fast_weight: float
    n_edges: int


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
) -> torch.Tensor:
    """Build passive inputs; query targets are deliberately absent from this API."""

    inputs = np.zeros((model_config.bs, model_config.inputsize), dtype=np.float32)
    for subject, episode in enumerate(episodes):
        if numstep == 0:
            inputs[subject, : 2 * model_config.cs] = np.concatenate(
                (
                    episode.item_codes[left_items[subject]],
                    episode.item_codes[right_items[subject]],
                )
            )
        elif numstep == NUMRESPONSESTEP:
            inputs[subject, model_config.nbstimbits - 1] = 1.0
        inputs[subject, model_config.nbstimbits] = 1.0
        inputs[subject, model_config.nbstimbits + 1] = time_value
        if support_trial and numstep == 0:
            inputs[subject, model_config.nbstimbits + DISTANCE_INPUT_OFFSET] = (
                signed_magnitudes[subject]
            )
    return torch.from_numpy(inputs).to(DEVICE)


def run_meta_batch(
    training_config: MetaTrainConfig,
    model_config: TrainConfig,
    net: RetroModulRNN,
    task_generator: GenericRankingTaskGenerator,
    rng: np.random.Generator,
) -> MetaBatchStats:
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
    blank = torch.zeros(model_config.bs, model_config.inputsize, device=DEVICE)
    for _ in range(2):
        _, _, _, hidden, eligibility, fast_weights = net(
            blank, hidden, eligibility, fast_weights
        )

    n_support = len(episodes[0].support_trials)
    for trial_index in range(n_support):
        hidden = net.initialZeroState(model_config.bs)
        eligibility = net.initialZeroET(model_config.bs)
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
        for numstep in range(model_config.triallen):
            inputs = build_meta_inputs(
                model_config,
                episodes,
                left,
                right,
                signed,
                numstep=numstep,
                time_value=time_value,
                support_trial=True,
            )
            _, _, _, hidden, eligibility, fast_weights = net(
                inputs, hidden, eligibility, fast_weights
            )

    query_loss = torch.zeros((), device=DEVICE)
    correct = 0
    total = 0
    n_queries = len(episodes[0].query_trials)
    for query_index in range(n_queries):
        hidden = net.initialZeroState(model_config.bs)
        eligibility = net.initialZeroET(model_config.bs)
        trials = [episode.query_trials[query_index] for episode in episodes]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        targets = torch.tensor(
            [trial.correct_action for trial in trials], dtype=torch.long, device=DEVICE
        )
        signed = np.zeros(model_config.bs, dtype=np.float32)
        response_logits = None
        for numstep in range(NUMRESPONSESTEP + 1):
            inputs = build_meta_inputs(
                model_config,
                episodes,
                left,
                right,
                signed,
                numstep=numstep,
                time_value=training_config.support_query_time,
                support_trial=False,
            )
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


def make_model_and_tasks(training_config: MetaTrainConfig):
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
    net = RetroModulRNN(model_config.to_model_dict())
    task_generator = GenericRankingTaskGenerator(
        cue_size=training_config.cue_size,
        min_edges=training_config.min_edges,
        max_edges=training_config.max_edges,
        support_blocks=training_config.support_blocks,
        exclude_liu_graph=True,
    )
    return model_config, net, task_generator


def save_meta_checkpoint(
    output_dir: Path,
    net: RetroModulRNN,
    training_config: MetaTrainConfig,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "net.dat"
    torch.save(net.state_dict(), checkpoint_path)
    metadata = {
        "schema_version": 1,
        "training": asdict(training_config),
        "completed_outer_steps": step + 1,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256(checkpoint_path),
        },
        "task_distribution": {
            "n_items": 8,
            "connected_sparse_graph": True,
            "liu_graph_held_out": True,
            "query_labels_enter_episode_inputs": False,
            "query_fast_weights": "frozen",
            "query_time_channel": "constant_at_support_query_boundary",
            "subject_encoding": {
                "state_scope": "fixed_for_entire_episode",
                "acts_on": "support_relation_reliability",
                "contains_rank_label": False,
                "configuration": SubjectEncodingConfig().to_dict(),
            },
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def train_meta_model(training_config: MetaTrainConfig, output_dir: Path) -> None:
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model_config, net, task_generator = make_model_and_tasks(training_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=training_config.learning_rate)
    log_path = output_dir / "train_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(training_config.outer_steps):
        optimizer.zero_grad()
        stats = run_meta_batch(training_config, model_config, net, task_generator, rng)
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
            save_meta_checkpoint(output_dir, net, training_config, step)


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
    )
    train_meta_model(training_config, parsed.output_dir)


if __name__ == "__main__":
    main()
