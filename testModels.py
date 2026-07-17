"""Evaluate all trained FSRL models on a fixed 8-item transitive-inference episode.

Protocol:
- 8 stimuli, indexed A..H -> 0..7.
- Training phase: 8 fixed pairs, repeated 4 times.
- Test phase: all 28 unordered pairs, repeated 10 times.
- Pair order is randomized within each trial.
- Training trials carry a signed pair-distance signal in the former reward slot.
- Test trials leave that slot at 0.

The script prints and saves a single CSV table with one column per model file
and one row per test pair. It also saves one large summary figure containing
all 28 pair-wise accuracy distributions.
"""

from __future__ import annotations
import csv
import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import beta as scipy_beta

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained FSRL models on a fixed 8-item TI episode."
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing .dat model files.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory to save one density plot per test pair.",
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed.")
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def discover_model_paths(models_dir: Path) -> list[Path]:
    model_paths = sorted(models_dir.glob("*.dat"))
    if not model_paths:
        raise FileNotFoundError(f"No .dat model files found in: {models_dir}")
    return model_paths


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
    signed_distance: float = 0.0,
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

            # Pre-compute signed_distance once per trial for reproducibility
            if is_train_trial:
                cue_pair = presented_pair
                signed_distance = (cue_pair[1] - cue_pair[0]) / (8 - 1) * (rng.random() + 1)
            else:
                signed_distance = 0.0

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
                    signed_distance=signed_distance,
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


def evaluate_model(
    model_path: Path,
    config: TrainConfig,
    trials: list[dict[str, object]],
    stochastic: bool,
    eval_seed: int,
) -> dict[tuple[int, int], dict[str, int]]:
    eval_config = replace(config, rngseed=eval_seed)
    set_seed(eval_seed)
    net = RetroModulRNN(eval_config.to_model_dict())
    net.load_state_dict(load_state_dict(model_path))
    net.eval()
    return evaluate_episode(
        net=net,
        config=eval_config,
        trials=trials,
        stochastic=stochastic,
    )


def generate_test_seeds(master_seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(master_seed)
    return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=count)]


def plot_pair_distributions(
    results_by_model: dict[str, dict[str, float]],
    pair_names: list[str],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results_by_model)
    n_pairs = len(pair_names)
    ncols = 7
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, 12), dpi=150, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for index, pair_name in enumerate(pair_names):
        ax = axes[index]
        accuracies = np.array(
            [results_by_model[model_name][pair_name] for model_name in model_names],
            dtype=float,
        )
        accuracies = accuracies[np.isfinite(accuracies)]

        if accuracies.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_xlim(0, 1)
        elif np.unique(accuracies).size == 1:
            value = float(accuracies[0])
            ax.axvline(value, color="#1f77b4", linewidth=8)
            ax.set_xlim(0, 1)
        else:
            bins = np.linspace(0.0, 1.0, 11)
            ax.hist(accuracies, bins=bins, density=False, color="#1f77b4", alpha=0.75, edgecolor="white")
            ax.set_xlim(0, 1)

        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Frequency")
        ax.set_title(pair_name)
        ax.grid(alpha=0.2, linewidth=0.5)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.supxlabel("Accuracy")
    fig.supylabel("Frequency")
    fig.tight_layout()

    output_path = figures_dir / "pair_accuracy_densities.png"
    fig.savefig(output_path)
    plt.close(fig)


def estimate_beta_params(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")

    clipped = np.clip(values, 1e-4, 1 - 1e-4)

    try:
        alpha, beta, _, _ = scipy_beta.fit(clipped, floc=0, fscale=1)
    except Exception:
        mean = float(clipped.mean())
        variance = float(clipped.var(ddof=0))
        if variance <= 0.0:
            variance = 1e-4
        common = mean * (1.0 - mean) / variance - 1.0
        if not np.isfinite(common) or common <= 0.0:
            alpha, beta = 1.0, 1.0
        else:
            alpha = mean * common
            beta = (1.0 - mean) * common

    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        mean = float(clipped.mean())
        variance = float(clipped.var(ddof=0))
        if variance <= 0.0:
            variance = 1e-4
        common = mean * (1.0 - mean) / variance - 1.0
        if not np.isfinite(common) or common <= 0.0:
            return 1.0, 1.0
        alpha = mean * common
        beta = (1.0 - mean) * common

    return float(alpha), float(beta)


def plot_pair_beta_fits(
    results_by_model: dict[str, dict[str, float]],
    pair_names: list[str],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results_by_model)
    n_pairs = len(pair_names)
    ncols = 7
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, 12), dpi=150, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    x_values = np.linspace(0.001, 0.999, 400)
    for index, pair_name in enumerate(pair_names):
        ax = axes[index]
        accuracies = np.array(
            [results_by_model[model_name][pair_name] for model_name in model_names],
            dtype=float,
        )
        accuracies = accuracies[np.isfinite(accuracies)]

        if accuracies.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(pair_name)
            ax.axis("off")
            continue

        bins = np.linspace(0.0, 1.0, 11)
        ax.hist(accuracies, bins=bins, density=True, color="#7f8c8d", alpha=0.45, edgecolor="white")

        alpha, beta = estimate_beta_params(accuracies)
        if np.isfinite(alpha) and np.isfinite(beta):
            pdf = scipy_beta.pdf(x_values, alpha, beta)
            ax.plot(x_values, pdf, color="#d62728", linewidth=2)
            text = f"$\\alpha$={alpha:.2f}\n$\\beta$={beta:.2f}"
        else:
            text = "$\\alpha$=nan\n$\\beta$=nan"

        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set_xlim(0, 1)
        ax.set_title(pair_name)
        ax.grid(alpha=0.2, linewidth=0.5)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.supxlabel("Accuracy")
    fig.supylabel("Density")
    fig.tight_layout()

    output_path = figures_dir / "pair_accuracy_beta_fits.png"
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    model_paths = discover_model_paths(models_dir)

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
    
    trial_rng = np.random.default_rng(args.seed)
    trials = build_trial_schedule(
        train_pairs=train_pairs,
        train_repeats=train_repeats,
        test_repeats=test_repeats,
        rng=trial_rng,
    )

    test_seeds = generate_test_seeds(args.seed, len(model_paths))

    letters = "ABCDEFGH"
    pair_order = list(combinations(range(8), 2))
    pair_names = [f"{letters[pair[0]]}-{letters[pair[1]]}" for pair in pair_order]

    results_by_model: dict[str, dict[str, float]] = {}
    print("Test pair accuracies by model:")
    for model_path, eval_seed in zip(model_paths, test_seeds, strict=True):
        pair_results = evaluate_model(
            model_path=model_path,
            config=config,
            trials=trials,
            stochastic=args.stochastic,
            eval_seed=eval_seed,
        )

        model_name = model_path.name
        model_rows: dict[str, float] = {}
        correct_total = 0
        trial_total = 0
        for pair, pair_name in zip(pair_order, pair_names, strict=True):
            stats = pair_results[pair]
            total = stats["total"]
            correct = stats["correct"]
            accuracy = correct / total if total else float("nan")
            model_rows[pair_name] = accuracy
            correct_total += correct
            trial_total += total

        overall_accuracy = correct_total / trial_total if trial_total else float("nan")
        results_by_model[model_name] = model_rows
        print(f"  {model_name} (seed={eval_seed}): {overall_accuracy:.3f}")

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = str(Path("models") / "results" / f"batch_test_results_seed_{args.seed}.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair_name in pair_names:
        row = {"pair": pair_name}
        for model_path in model_paths:
            row[model_path.name] = results_by_model[model_path.name][pair_name]
        rows.append(row)

    fieldnames = ["pair"] + [model_path.name for model_path in model_paths]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {output_path}")

    figures_dir = (
        Path(args.figures_dir)
        if args.figures_dir is not None
        else Path("models") / "results" / f"figures_seed_{args.seed}"
    )
    plot_pair_distributions(
        results_by_model=results_by_model,
        pair_names=pair_names,
        figures_dir=figures_dir,
    )
    print(f"Saved pair distribution figures to {figures_dir}")

    plot_pair_beta_fits(
        results_by_model=results_by_model,
        pair_names=pair_names,
        figures_dir=figures_dir,
    )
    print(f"Saved beta fit figure to {figures_dir / 'pair_accuracy_beta_fits.png'}")

if __name__ == "__main__":
    main()