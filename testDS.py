"""Evaluate a trained FSRL model across many random seeds.

Protocol:
- 8 stimuli, indexed A..H -> 0..7.
- Training phase: 8 fixed pairs, repeated 4 times.
- Test phase: all 28 unordered pairs, repeated 10 times.
- Pair order is randomized within each trial.
- Training trials carry a signed pair-distance signal in the former reward slot.
- Test trials leave that slot at 0.

The script evaluates the same model under many random seeds, collects the
28-pair test accuracies for each seed, and saves a 28-panel density plot.
"""

from __future__ import annotations
import csv
import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from fsrl.config import ADDINPUT, DEVICE, NUMRESPONSESTEP, TrainConfig
from fsrl.model import RetroModulRNN
from fsrl.task import generate_cue_data


TRAIN_PAIR_LABELS = [
    (0, 5),  # A-F
    (1, 2),  # B-C
    (1, 4),  # B-E
    (2, 6),  # C-G
    (3, 5),  # D-F
    (3, 6),  # D-G
    (4, 7),  # E-H
    (0, 7),  # A-H
]

PAIR_LIST = list(combinations(range(8), 2))
PAIR_LABELS = {
    pair: f"{chr(ord('A') + pair[0])}-{chr(ord('A') + pair[1])}" for pair in PAIR_LIST
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FSRL model across many random seeds."
    )
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().with_name("net.dat")),
        help="Path to the saved model state dict.",
    )
    parser.add_argument("--seed", type=int, default=460, help="Random seed.")
    parser.add_argument(
        "--cs",
        type=int,
        default=15,
        help="Stimulus code length used by the trained model.",
    )
    parser.add_argument(
        "--hs",
        type=int,
        default=200,
        help="Hidden size used by the trained model.",
    )
    parser.add_argument(
        "--triallen",
        type=int,
        default=4,
        help="Number of time steps per trial.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using argmax.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of random seeds to evaluate.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=123,
        help="Offset added to each evaluation seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory used to save the CSV and figure.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def build_trial_schedule(
    train_pairs: list[tuple[int, int]],
    train_repeats: int,
    test_repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    test_pairs = list(combinations(range(8), 2))
    trials: list[dict[str, object]] = []

    for _ in range(train_repeats):
        block = train_pairs.copy()
        rng.shuffle(block)
        for pair in block:
            trials.append({"pair": pair, "phase": "train"})

    for _ in range(test_repeats):
        block = test_pairs.copy()
        rng.shuffle(block)
        for pair in block:
            trials.append({"pair": pair, "phase": "test"})

    return trials


def randomize_pair(pair: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    if rng.integers(2) == 0:
        return pair
    return pair[1], pair[0]


def build_inputs(
    config: TrainConfig,
    nbcues: int,
    cue_data,
    cues,
    previous_actions: np.ndarray,
    numstep: int,
    numstep_ep: int,
    is_train_trial: bool,
) -> torch.Tensor:
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")

    for batch_index in range(config.bs):
        cue = cues[batch_index][numstep]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.nbstimbits - 1] = np.concatenate(
                (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
            )
        elif cue == nbcues:
            inputs[batch_index, config.nbstimbits - 1] = 1.0

        inputs[batch_index, config.nbstimbits + 0] = 1.0
        inputs[batch_index, config.nbstimbits + 1] = numstep_ep / config.eplen

        if is_train_trial:
            cue_pair = cues[batch_index][0]
            signed_distance = (cue_pair[1] - cue_pair[0]) / (nbcues - 1) * (np.random.random() + 1)
            inputs[batch_index, config.nbstimbits + 2] = signed_distance
        else:
            inputs[batch_index, config.nbstimbits + 2] = 0.0

        if numstep == NUMRESPONSESTEP + 1:
            inputs[
                batch_index,
                config.nbstimbits + ADDINPUT + previous_actions[batch_index],
            ] = 1.0

    return torch.from_numpy(inputs).detach().to(DEVICE)


def choose_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    if deterministic:
        return torch.argmax(probs, dim=1)
    return torch.distributions.Categorical(probs).sample()


def evaluate_episode(
    net: RetroModulRNN,
    config: TrainConfig,
    trials: list[dict[str, object]],
    stochastic: bool,
) -> dict[tuple[int, int], dict[str, int]]:
    rng = np.random.default_rng(config.rngseed if config.rngseed >= 0 else 0)
    cue_data = generate_cue_data(config, 8)

    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)
    pw = net.initialZeroPlasticWeights(config.bs)
    previous_actions = np.zeros(config.bs, dtype="int32")

    blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(
        DEVICE
    )
    with torch.no_grad():
        for _ in range(2):
            _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

    pair_results: dict[tuple[int, int], dict[str, int]] = {
        pair: {"correct": 0, "total": 0}
        for pair in combinations(range(8), 2)
    }

    numstep_ep = 0
    with torch.no_grad():
        for trial in trials:
            hidden = net.initialZeroState(config.bs)
            et = net.initialZeroET(config.bs)

            base_pair = trial["pair"]
            phase = trial["phase"]
            assert isinstance(base_pair, tuple)
            is_train_trial = phase == "train"
            presented_pair = randomize_pair(base_pair, rng)
            cues = [[presented_pair, 8, -1, -1]]

            action_at_response = None
            for numstep in range(config.triallen):
                inputs = build_inputs(
                    config=config,
                    nbcues=8,
                    cue_data=cue_data,
                    cues=cues,
                    previous_actions=previous_actions,
                    numstep=numstep,
                    numstep_ep=numstep_ep,
                    is_train_trial=is_train_trial,
                )
                logits, _, _, hidden, et, pw = net(inputs, hidden, et, pw)
                actions = choose_action(logits, deterministic=not stochastic)
                previous_actions = actions.detach().cpu().numpy().astype("int32")
                if numstep == NUMRESPONSESTEP:
                    action_at_response = int(previous_actions[0])
                numstep_ep += 1

            assert action_at_response is not None
            correct_action = 1 if presented_pair[0] < presented_pair[1] else 0
            pair_key = tuple(sorted(base_pair))
            if is_train_trial:
                continue
            pair_results[pair_key]["total"] += 1
            pair_results[pair_key]["correct"] += int(action_at_response == correct_action)

    return pair_results


def evaluate_many_seeds(
    net: RetroModulRNN,
    base_config: TrainConfig,
    train_pairs: list[tuple[int, int]],
    train_repeats: int,
    test_repeats: int,
    num_seeds: int,
    seed_offset: int,
    stochastic: bool,
) -> tuple[dict[tuple[int, int], list[float]], list[dict[str, object]]]:
    pair_distributions: dict[tuple[int, int], list[float]] = {
        pair: [] for pair in PAIR_LIST
    }
    rows: list[dict[str, object]] = []

    for seed_index in range(num_seeds):
        seed = seed_offset + seed_index
        set_seed(seed)
        config = replace(base_config, rngseed=seed)
        trial_rng = np.random.default_rng(seed)
        trials = build_trial_schedule(
            train_pairs=train_pairs,
            train_repeats=train_repeats,
            test_repeats=test_repeats,
            rng=trial_rng,
        )

        pair_results = evaluate_episode(
            net=net,
            config=config,
            trials=trials,
            stochastic=stochastic,
        )

        for pair in PAIR_LIST:
            stats = pair_results[pair]
            total = stats["total"]
            correct = stats["correct"]
            accuracy = correct / total if total else float("nan")
            pair_distributions[pair].append(accuracy)
            rows.append(
                {
                    "seed": seed,
                    "pair": PAIR_LABELS[pair],
                    "total": total,
                    "correct": correct,
                    "accuracy": accuracy,
                }
            )

    return pair_distributions, rows


def plot_accuracy_distributions(
    pair_distributions: dict[tuple[int, int], list[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 7, figsize=(21, 12), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 11)

    for axis, pair in zip(axes.flat, PAIR_LIST):
        values = np.asarray(pair_distributions[pair], dtype=float)
        values = values[~np.isnan(values)]
        axis.hist(
            values,
            bins=bins,
            density=True,
            color="#2563eb",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_title(PAIR_LABELS[pair], fontsize=9)
        axis.set_xlim(0, 1)
        axis.grid(True, alpha=0.2)

    for axis in axes[-1, :]:
        axis.set_xlabel("Accuracy")
    for axis in axes[:, 0]:
        axis.set_ylabel("Density")

    fig.suptitle("Per-pair accuracy distributions across seeds", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    train_pairs = TRAIN_PAIR_LABELS
    train_repeats = 4
    test_repeats = 10

    config = TrainConfig(
        rngseed=args.seed,
        bs=1,
        cs=args.cs,
        hs=args.hs,
        triallen=args.triallen,
        nbtraintrials=len(train_pairs) * train_repeats,
        nbtesttrials=(8 * 7 // 2) * test_repeats,
    )

    net = RetroModulRNN(config.to_model_dict())
    net.load_state_dict(load_state_dict(model_path))
    net.eval()

    pair_distributions, rows = evaluate_many_seeds(
        net=net,
        base_config=config,
        train_pairs=train_pairs,
        train_repeats=train_repeats,
        test_repeats=test_repeats,
        num_seeds=args.num_seeds,
        seed_offset=args.seed + args.seed_offset,
        stochastic=args.stochastic,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{model_path.stem}_seed_distributions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "pair", "total", "correct", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / f"{model_path.stem}_seed_distributions.png"
    plot_accuracy_distributions(pair_distributions, figure_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved figure to {figure_path}")

if __name__ == "__main__":
    main()