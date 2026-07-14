"""Evaluate a trained FSRL model on a fixed 8-item transitive-inference episode.

Protocol:
- 8 stimuli, indexed A..H -> 0..7.
- Training phase: 8 fixed pairs, repeated 4 times.
- Test phase: all 28 unordered pairs, repeated 10 times.
- Pair order is randomized within each trial.
- Training trials carry a signed pair-distance signal in the former reward slot.
- Test trials leave that slot at 0.

The script prints the accuracy for all 28 test pairs.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FSRL model on a fixed 8-item TI episode."
    )
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().with_name("net.dat")),
        help="Path to the saved model state dict.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    trial_rng = np.random.default_rng(args.seed)
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
        stochastic=args.stochastic,
    )

    letters = "ABCDEFGH"
    correct_total = 0
    trial_total = 0

    print("Test pair accuracies:")
    for pair in combinations(range(8), 2):
        stats = pair_results[pair]
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else float("nan")
        correct_total += correct
        trial_total += total
        print(
            f"  {letters[pair[0]]}-{letters[pair[1]]}: "
            f"{accuracy:.3f} ({correct}/{total})"
        )

    overall_accuracy = correct_total / trial_total if trial_total else float("nan")
    print(f"Overall test accuracy: {overall_accuracy:.3f}")


if __name__ == "__main__":
    main()