"""
Standard full all-pair evaluation for single_neo_mutants_distance_input_v3 checkpoints.

This script runs the trained Simple-NEO mutant model through its learning phase,
then probes every unordered N=8 pair in a no-feedback test phase. It is designed
for the behavior-graph and observational no-feedback mutants, especially:

  - behavior_graph_no_test_feedback
  - observational_learning_no_test_feedback_with_test_loss

Important design choices:
  1. Test reward is always zero in this evaluator.
  2. Labels are never fed back as episode inputs during evaluation.
  3. Hidden state and eligibility trace reset at every trial, while plastic
     weights persist across the episode, matching simple_neo.
  4. By default, the time input is clamped to the training episode range. This
     avoids out-of-distribution time values when evaluating 28 test trials even
     though the model was trained with 10 test trials.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_VERSION = "simple_neo_full28_eval_2026-07-09_distance_input_v3"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import single_neo_mutants_distance_input_v3 as neo  # noqa: E402

Pair = Tuple[int, int]


def log(msg: str) -> None:
    print(msg, flush=True)


def all_unordered_pairs(n_items: int = 8) -> List[Pair]:
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def unordered_pair(pair: Pair) -> Pair:
    a, b = pair
    return (a, b) if a < b else (b, a)


def is_learned_pair(pair: Pair) -> bool:
    return unordered_pair(pair) in {unordered_pair(p) for p in neo.BEHAVIOR_GRAPH_N8}


def load_training_config(checkpoint_dir: Path) -> dict:
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text().strip()
    if not text:
        return {}
    # single_neo_mutants_distance_input_v3 wrote a Python dict string rather than strict JSON.
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception as exc:
            raise ValueError(f"Could not parse config file: {cfg_path}") from exc


def build_eval_config(args: argparse.Namespace) -> neo.TrainConfig:
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    file_cfg = load_training_config(ckpt_dir) if ckpt_dir else {}

    def get_value(name: str, default):
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        return file_cfg.get(name, default)

    variant = args.variant or file_cfg.get("variant", "behavior_graph_no_test_feedback")
    nmin = int(file_cfg.get("nbcues_min", 8))
    nmax = int(file_cfg.get("nbcues_max", 8))
    if variant == "exact_simple_neo":
        nmin = int(args.nmin if args.nmin is not None else file_cfg.get("nbcues_min", 4))
        nmax = int(args.nmax if args.nmax is not None else file_cfg.get("nbcues_max", 8))
    else:
        nmin = nmax = 8

    config = neo.TrainConfig(
        rngseed=args.seed,
        variant=variant,
        bs=args.eval_batch_size,
        nbiter=0,
        save_every=0,
        pe=0,
        hs=int(get_value("hidden_size", file_cfg.get("hs", 200))),
        cs=int(get_value("cue_size", file_cfg.get("cs", 15))),
        lr=float(file_cfg.get("lr", 1e-4)),
        lpw=float(file_cfg.get("lpw", 1e-4)),
        nbtraintrials=int(args.nbtraintrials if args.nbtraintrials is not None else file_cfg.get("nbtraintrials", 20)),
        # Keep nbtesttrials at the value used during training so config.eplen and
        # the time input match the trained model. Full-28 testing is handled by
        # the evaluator schedule, not by changing the training config length.
        nbtesttrials=int(file_cfg.get("nbtesttrials", 10)),
        testlmult=float(file_cfg.get("testlmult", 3.0)),
        train_supervised_loss_weight=float(file_cfg.get("train_supervised_loss_weight", 1.0)),
        test_supervised_loss_weight=float(file_cfg.get("test_supervised_loss_weight", 1.0)),
        nbcues_min=nmin,
        nbcues_max=nmax,
        write_csv=False,
        num_threads=args.num_threads,
        distance_input=bool(file_cfg.get("distance_input", True)) and not args.no_distance_input,
        distance_input_train_only=(False if args.distance_input_all_phases else bool(file_cfg.get("distance_input_train_only", True))),
    )
    return config


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not args.checkpoint_dir:
        raise ValueError("Provide --checkpoint or --checkpoint-dir")
    ckpt_dir = Path(args.checkpoint_dir)
    candidates = [ckpt_dir / "net.dat", ckpt_dir / f"netAE{args.train_seed_for_filename}.dat"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find net.dat in {ckpt_dir}")


def make_distance_observation(config: neo.TrainConfig, cue_pairs: List[Pair], nbcues: int, is_train_trial: bool) -> np.ndarray:
    """Signed rank-distance input matching single_neo_mutants_distance_input_v3."""
    distance_observation = np.zeros(config.bs, dtype="float32")
    if neo.should_provide_distance_input(config, is_train_trial):
        for b, pair in enumerate(cue_pairs):
            distance_observation[b] = neo.signed_rank_distance(list(pair), nbcues)
    return distance_observation


def build_step_inputs_eval(
    config: neo.TrainConfig,
    nbcues: int,
    cue_data,
    cues,
    reward: np.ndarray,
    previous_actions: np.ndarray,
    distance_observation: np.ndarray,
    numstep: int,
    numstep_ep: int,
    include_previous_action: np.ndarray,
    time_mode: str,
    eval_total_steps: int,
):
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")
    include_previous_action = np.asarray(include_previous_action, dtype="float32")

    if time_mode == "original":
        time_value = numstep_ep / config.eplen
    elif time_mode == "clamp":
        # Preserve training-time scaling until 1.0, then keep the time input in range.
        time_value = min(numstep_ep / config.eplen, (config.eplen - 1) / config.eplen)
    elif time_mode == "rescale_full_eval":
        time_value = numstep_ep / max(1, eval_total_steps)
    else:
        raise ValueError(f"Unknown time_mode: {time_mode}")

    for batch_index in range(config.bs):
        cue = cues[batch_index][numstep]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.nbstimbits - 1] = np.concatenate(
                (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
            )
        elif cue == nbcues:
            inputs[batch_index, config.nbstimbits - 1] = 1

        inputs[batch_index, config.nbstimbits + 0] = 1.0
        inputs[batch_index, config.nbstimbits + 1] = time_value
        inputs[batch_index, config.nbstimbits + 2] = reward[batch_index]
        if isinstance(cue, (list, tuple, np.ndarray)) and hasattr(neo, "DISTANCE_INPUT_OFFSET"):
            inputs[batch_index, config.nbstimbits + neo.DISTANCE_INPUT_OFFSET] = distance_observation[batch_index]

        if numstep == neo.NUMRESPONSESTEP + 1 and include_previous_action[batch_index] > 0.5:
            inputs[batch_index, config.nbstimbits + neo.ADDINPUT + previous_actions[batch_index]] = 1

    return torch.from_numpy(inputs).detach().to(neo.DEVICE)


def choose_action(y_raw: torch.Tensor, action_mode: str) -> torch.Tensor:
    if action_mode == "greedy":
        return torch.argmax(y_raw, dim=1)
    if action_mode == "sample":
        probs = torch.softmax(y_raw, dim=1)
        return torch.distributions.Categorical(probs).sample()
    raise ValueError(f"Unknown action_mode: {action_mode}")


def run_one_trial(
    config: neo.TrainConfig,
    net: neo.RetroModulRNN,
    nbcues: int,
    cue_data,
    hidden: torch.Tensor,
    et: torch.Tensor,
    pw: torch.Tensor,
    cue_pairs: List[Pair],
    requires_choice: np.ndarray,
    reward_available: np.ndarray,
    distance_observation: np.ndarray,
    numstep_ep: int,
    action_mode: str,
    time_mode: str,
    eval_total_steps: int,
    collect_choice: bool,
):
    # Per-trial reset, matching simple_neo. Plastic weights persist.
    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)

    cues = [[list(pair), nbcues, -1, -1] for pair in cue_pairs]
    previous_actions = np.zeros(config.bs, dtype="int32")
    reward = np.zeros(config.bs, dtype="float32")
    chosen_winner = np.full(config.bs, -1, dtype="int32")
    correct = np.full(config.bs, np.nan, dtype="float32")
    prob_correct = np.full(config.bs, np.nan, dtype="float32")

    for numstep in range(config.triallen):
        inputs = build_step_inputs_eval(
            config,
            nbcues,
            cue_data,
            cues,
            reward,
            previous_actions,
            distance_observation,
            numstep,
            numstep_ep,
            requires_choice,
            time_mode,
            eval_total_steps,
        )
        y_raw, _value, _daout, hidden, et, pw_new = net(inputs, hidden, et, pw)
        if collect_choice and args_global.freeze_test_plastic:
            # Optional diagnostic: let hidden/et update but keep episode memory fixed during test.
            pw = pw
        else:
            pw = pw_new

        reward = np.zeros(config.bs, dtype="float32")
        if numstep == neo.NUMRESPONSESTEP:
            actions = choose_action(y_raw, action_mode).detach().cpu().numpy().astype("int32")
            previous_actions = actions.copy()
            probs = torch.softmax(y_raw, dim=1).detach().cpu().numpy()

            for b, pair in enumerate(cue_pairs):
                if requires_choice[b] <= 0.5:
                    previous_actions[b] = 0
                    continue
                first, second = int(pair[0]), int(pair[1])
                correct_action = 1 if first < second else 0
                prob_correct[b] = probs[b, correct_action]
                is_correct = actions[b] == correct_action
                correct[b] = 1.0 if is_correct else 0.0
                winner = first if actions[b] == 1 else second
                chosen_winner[b] = winner
                if reward_available[b] > 0.5:
                    reward[b] = config.rew if is_correct else -config.rew

        numstep_ep += 1

    return hidden, et, pw, numstep_ep, chosen_winner, correct, prob_correct


# Module-level bridge used by run_one_trial for optional freeze diagnostic.
args_global = None


def make_learning_pairs(config: neo.TrainConfig, nbcues: int) -> List[Pair]:
    pairs: List[Pair] = []
    if config.variant in ("exact_simple_neo", "n8_fixed"):
        for _ in range(config.bs):
            pairs.append(tuple(neo.sample_adjacent_pair(nbcues)))
    elif config.variant.startswith("behavior_graph"):
        for _ in range(config.bs):
            pairs.append(tuple(neo.sample_behavior_graph_pair(random_orientation=True)))
    elif neo.is_passive_observational_variant(config.variant):
        random_orientation = (
            neo.should_provide_distance_input(config, is_train_trial=True)
            or neo.use_train_aux_loss(config.variant)
            or not config.observational_train_orients_correct_first
        )
        for _ in range(config.bs):
            pairs.append(tuple(neo.sample_behavior_graph_pair(random_orientation=random_orientation)))
    else:
        raise ValueError(f"Unknown variant: {config.variant}")
    return pairs


def make_full_test_schedules(n_eval: int, n_items: int, orientation: str, rng: np.random.Generator) -> List[List[Pair]]:
    base_pairs = all_unordered_pairs(n_items)
    schedules: List[List[Pair]] = []
    for _ in range(n_eval):
        if orientation == "both":
            pairs = []
            for p in base_pairs:
                pairs.append(p)
                pairs.append((p[1], p[0]))
            rng.shuffle(pairs)
        else:
            pairs = base_pairs.copy()
            rng.shuffle(pairs)
            if orientation == "random":
                pairs = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
            elif orientation == "canonical":
                pass
            else:
                raise ValueError(f"Unknown test_orientation: {orientation}")
        schedules.append(pairs)
    return schedules


def count_circular_triads(choice_winner: Dict[Pair, int], n_items: int) -> int:
    cycles = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            for k in range(j + 1, n_items):
                triples = [(i, j), (i, k), (j, k)]
                wins = {i: 0, j: 0, k: 0}
                valid = True
                for a, b in triples:
                    key = unordered_pair((a, b))
                    if key not in choice_winner:
                        valid = False
                        break
                    wins[choice_winner[key]] += 1
                if not valid:
                    continue
                # In a 3-node tournament, a directed cycle means every item has one win.
                if wins[i] == wins[j] == wins[k] == 1:
                    cycles += 1
    return cycles


def kendall_tau_from_copeland(choice_winner: Dict[Pair, int], n_items: int) -> float:
    wins = np.zeros(n_items, dtype=np.float64)
    for pair, winner in choice_winner.items():
        if 0 <= winner < n_items:
            wins[winner] += 1.0
    concordant = 0.0
    discordant = 0.0
    comparable = 0.0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # True ranking: lower index should be higher, therefore score_i > score_j.
            if wins[i] == wins[j]:
                continue
            comparable += 1.0
            if wins[i] > wins[j]:
                concordant += 1.0
            else:
                discordant += 1.0
    if comparable == 0:
        return 0.0
    return (concordant - discordant) / comparable


def evaluate(config: neo.TrainConfig, net: neo.RetroModulRNN, args: argparse.Namespace):
    global args_global
    args_global = args
    rng = np.random.default_rng(args.seed if args.seed >= 0 else None)
    n_items = 8
    full_pairs = all_unordered_pairs(n_items)
    learned_set = {unordered_pair(p) for p in neo.BEHAVIOR_GRAPH_N8}

    total_requested = args.eval_episodes
    batch_size = config.bs
    n_batches = math.ceil(total_requested / batch_size)

    aggregate = {
        "correct": 0.0,
        "count": 0.0,
        "learned_correct": 0.0,
        "learned_count": 0.0,
        "nonlearned_correct": 0.0,
        "nonlearned_count": 0.0,
        "adjacent_correct": 0.0,
        "adjacent_count": 0.0,
        "nonadjacent_correct": 0.0,
        "nonadjacent_count": 0.0,
        "first10_correct": 0.0,
        "first10_count": 0.0,
        "later_correct": 0.0,
        "later_count": 0.0,
        "prob_correct_sum": 0.0,
        "prob_correct_count": 0.0,
        "mean_abs_pw_learning_sum": 0.0,
        "mean_abs_pw_final_sum": 0.0,
        "eval_subjects": 0.0,
        "circular_triads_sum": 0.0,
        "kendall_tau_sum": 0.0,
    }
    distance_correct = {d: 0.0 for d in range(1, n_items)}
    distance_count = {d: 0.0 for d in range(1, n_items)}
    pair_correct = {p: 0.0 for p in full_pairs}
    pair_count = {p: 0.0 for p in full_pairs}

    max_test_len = 56 if args.test_orientation == "both" else 28
    eval_total_steps = (config.nbtraintrials + max_test_len) * config.triallen

    net.eval()
    done = 0
    with torch.no_grad():
        for batch_index in range(n_batches):
            current_bs = min(batch_size, total_requested - done)
            if current_bs <= 0:
                break
            # Keep model config batch size consistent for this final partial batch.
            if current_bs != config.bs:
                old_bs = config.bs
                config.bs = current_bs
                net.GG["bs"] = current_bs
            else:
                old_bs = config.bs

            nbcues = 8
            cue_data = neo.generate_cue_data(config, nbcues)
            hidden = net.initialZeroState(config.bs)
            et = net.initialZeroET(config.bs)
            pw = net.initialZeroPlasticWeights(config.bs)
            blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(neo.DEVICE)
            for _ in range(2):
                _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

            numstep_ep = 0
            for _ in range(config.nbtraintrials):
                cue_pairs = make_learning_pairs(config, nbcues)
                if neo.is_passive_observational_variant(config.variant):
                    requires_choice = np.zeros(config.bs, dtype="float32")
                    reward_available = np.zeros(config.bs, dtype="float32")
                else:
                    requires_choice = np.ones(config.bs, dtype="float32")
                    reward_available = np.ones(config.bs, dtype="float32")
                distance_observation = make_distance_observation(config, cue_pairs, nbcues, is_train_trial=True)
                hidden, et, pw, numstep_ep, _winner, _correct, _prob = run_one_trial(
                    config,
                    net,
                    nbcues,
                    cue_data,
                    hidden,
                    et,
                    pw,
                    cue_pairs,
                    requires_choice,
                    reward_available,
                    distance_observation,
                    numstep_ep,
                    args.action_mode,
                    args.time_mode,
                    eval_total_steps,
                    collect_choice=False,
                )

            mean_abs_pw_learning_by_subject = torch.mean(torch.abs(pw), dim=(1, 2)).detach().cpu().numpy()
            schedules = make_full_test_schedules(config.bs, n_items, args.test_orientation, rng)
            max_len = len(schedules[0])
            subject_choices: List[Dict[Pair, int]] = [dict() for _ in range(config.bs)]

            for test_idx in range(max_len):
                cue_pairs = [schedules[b][test_idx] for b in range(config.bs)]
                requires_choice = np.ones(config.bs, dtype="float32")
                reward_available = np.zeros(config.bs, dtype="float32")  # standard no-feedback test
                distance_observation = make_distance_observation(config, cue_pairs, nbcues, is_train_trial=False)
                hidden, et, pw, numstep_ep, winners, correct, prob_correct = run_one_trial(
                    config,
                    net,
                    nbcues,
                    cue_data,
                    hidden,
                    et,
                    pw,
                    cue_pairs,
                    requires_choice,
                    reward_available,
                    distance_observation,
                    numstep_ep,
                    args.action_mode,
                    args.time_mode,
                    eval_total_steps,
                    collect_choice=True,
                )
                for b, pair in enumerate(cue_pairs):
                    if np.isnan(correct[b]):
                        continue
                    upair = unordered_pair(pair)
                    dist = abs(upair[0] - upair[1])
                    is_learned = upair in learned_set
                    c = float(correct[b])
                    aggregate["correct"] += c
                    aggregate["count"] += 1.0
                    aggregate["prob_correct_sum"] += float(prob_correct[b])
                    aggregate["prob_correct_count"] += 1.0
                    distance_correct[dist] += c
                    distance_count[dist] += 1.0
                    pair_correct[upair] += c
                    pair_count[upair] += 1.0
                    if is_learned:
                        aggregate["learned_correct"] += c
                        aggregate["learned_count"] += 1.0
                    else:
                        aggregate["nonlearned_correct"] += c
                        aggregate["nonlearned_count"] += 1.0
                    if dist == 1:
                        aggregate["adjacent_correct"] += c
                        aggregate["adjacent_count"] += 1.0
                    else:
                        aggregate["nonadjacent_correct"] += c
                        aggregate["nonadjacent_count"] += 1.0
                    if test_idx < 10:
                        aggregate["first10_correct"] += c
                        aggregate["first10_count"] += 1.0
                    else:
                        aggregate["later_correct"] += c
                        aggregate["later_count"] += 1.0
                    # For self-consistency, keep one choice per unordered pair. If both
                    # orientations are evaluated, later presentations overwrite earlier ones;
                    # use orientation=random for a strict 28-pair tournament.
                    if winners[b] >= 0:
                        subject_choices[b][upair] = int(winners[b])

            mean_abs_pw_final_by_subject = torch.mean(torch.abs(pw), dim=(1, 2)).detach().cpu().numpy()
            for b in range(config.bs):
                aggregate["circular_triads_sum"] += count_circular_triads(subject_choices[b], n_items)
                aggregate["kendall_tau_sum"] += kendall_tau_from_copeland(subject_choices[b], n_items)
                aggregate["mean_abs_pw_learning_sum"] += float(mean_abs_pw_learning_by_subject[b])
                aggregate["mean_abs_pw_final_sum"] += float(mean_abs_pw_final_by_subject[b])
                aggregate["eval_subjects"] += 1.0

            done += current_bs
            if current_bs != old_bs:
                config.bs = old_bs
                net.GG["bs"] = old_bs

    def ratio(num_key: str, den_key: str):
        den = aggregate[den_key]
        return None if den == 0 else aggregate[num_key] / den

    subjects = max(1.0, aggregate["eval_subjects"])
    total_triads = math.comb(n_items, 3)
    summary = {
        "script_version": SCRIPT_VERSION,
        "model_script_version": getattr(neo, "SCRIPT_VERSION", "unknown"),
        "variant": config.variant,
        "checkpoint": str(args.checkpoint or Path(args.checkpoint_dir) / "net.dat"),
        "eval_episodes": int(aggregate["eval_subjects"]),
        "n_items": n_items,
        "nbtraintrials": config.nbtraintrials,
        "training_nbtesttrials_for_time_input": config.nbtesttrials,
        "training_eplen_for_time_input": config.eplen,
        "test_pairs_per_episode": max_test_len,
        "test_reward_forced_zero": True,
        "test_orientation": args.test_orientation,
        "action_mode": args.action_mode,
        "time_mode": args.time_mode,
        "freeze_test_plastic": bool(args.freeze_test_plastic),
        "distance_input": bool(config.distance_input),
        "distance_input_train_only": bool(config.distance_input_train_only),
        "distance_slot": f"nbstimbits+{getattr(neo, 'DISTANCE_INPUT_OFFSET', 'NA')}",
        "overall_accuracy": ratio("correct", "count"),
        "learned_pairs_accuracy": ratio("learned_correct", "learned_count"),
        "nonlearned_pairs_accuracy": ratio("nonlearned_correct", "nonlearned_count"),
        "adjacent_accuracy": ratio("adjacent_correct", "adjacent_count"),
        "nonadjacent_accuracy": ratio("nonadjacent_correct", "nonadjacent_count"),
        "first10_test_accuracy": ratio("first10_correct", "first10_count"),
        "later_test_accuracy": ratio("later_correct", "later_count"),
        "mean_prob_correct": ratio("prob_correct_sum", "prob_correct_count"),
        "distance_accuracy": {
            str(d): (None if distance_count[d] == 0 else distance_correct[d] / distance_count[d])
            for d in range(1, n_items)
        },
        "mean_circular_triads": aggregate["circular_triads_sum"] / subjects,
        "mean_transitive_triplet_fraction": 1.0 - (aggregate["circular_triads_sum"] / subjects) / total_triads,
        "mean_kendall_tau_copeland_to_true": aggregate["kendall_tau_sum"] / subjects,
        "mean_abs_pw_after_learning": aggregate["mean_abs_pw_learning_sum"] / subjects,
        "mean_abs_pw_after_full_test": aggregate["mean_abs_pw_final_sum"] / subjects,
    }

    pair_rows = []
    for pair in full_pairs:
        cnt = pair_count[pair]
        pair_rows.append(
            {
                "pair": f"{pair[0]}-{pair[1]}",
                "i": pair[0],
                "j": pair[1],
                "distance": abs(pair[0] - pair[1]),
                "is_learned_pair": pair in learned_set,
                "accuracy": None if cnt == 0 else pair_correct[pair] / cnt,
                "count": int(cnt),
            }
        )
    return summary, pair_rows


def save_outputs(summary: dict, pair_rows: list, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "full28_eval_summary.json"
    pair_path = output_dir / "full28_pair_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with pair_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "i", "j", "distance", "is_learned_pair", "accuracy", "count"])
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)
    log(f"[save] summary: {summary_path}")
    log(f"[save] pair metrics: {pair_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full 28-pair no-feedback evaluator for single_neo_mutants_distance_input_v3 checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Training output dir containing net.dat and config.json.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path. Overrides --checkpoint-dir/net.dat.")
    parser.add_argument("--variant", type=str, default=None, help="Override variant; defaults to config.json variant.")
    parser.add_argument("--eval-episodes", type=int, default=512, help="Number of independent eval subjects/episodes.")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--cue-size", type=int, default=None)
    parser.add_argument("--nbtraintrials", type=int, default=None)
    parser.add_argument("--nmin", type=int, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--train-seed-for-filename", type=int, default=1, help="Fallback for netAE<seed>.dat if net.dat is missing.")
    parser.add_argument("--test-orientation", choices=["random", "canonical", "both"], default="random")
    parser.add_argument("--action-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--time-mode", choices=["clamp", "original", "rescale_full_eval"], default="clamp")
    parser.add_argument("--freeze-test-plastic", action="store_true", help="Diagnostic only: do not update plastic weights during test trials.")
    parser.add_argument("--no-distance-input", action="store_true", help="Disable signed rank-distance input during behavioral learning evaluation.")
    parser.add_argument("--distance-input-all-phases", action="store_true", help="Also expose signed distance during test trials; normally keep this off to preserve no-feedback test.")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <checkpoint-dir>/full28_eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    ckpt_path = resolve_checkpoint_path(args)
    config = build_eval_config(args)
    model_config = config.to_model_dict()
    net = neo.RetroModulRNN(model_config)
    state = torch.load(ckpt_path, map_location=neo.DEVICE)
    net.load_state_dict(state)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_dir or ckpt_path.parent) / "full28_eval"

    log(f"[setup] evaluator version: {SCRIPT_VERSION}")
    log(f"[setup] model script version: {getattr(neo, 'SCRIPT_VERSION', 'unknown')}")
    log(f"[setup] device: {neo.DEVICE}; torch_num_threads={torch.get_num_threads()}")
    log(f"[setup] checkpoint: {ckpt_path}")
    log(f"[setup] variant: {config.variant}; eval_episodes={args.eval_episodes}; batch={config.bs}")
    log(f"[setup] full test orientation={args.test_orientation}; action_mode={args.action_mode}; time_mode={args.time_mode}")
    log(f"[setup] distance_input={config.distance_input}; train_only={config.distance_input_train_only}; distance_slot=nbstimbits+{getattr(neo, 'DISTANCE_INPUT_OFFSET', 'NA')}")
    log("[setup] Test reward is forced to zero for all full-28 test trials.")

    summary, pair_rows = evaluate(config, net, args)
    save_outputs(summary, pair_rows, output_dir)
    log("[summary]")
    for key in [
        "overall_accuracy",
        "learned_pairs_accuracy",
        "nonlearned_pairs_accuracy",
        "adjacent_accuracy",
        "nonadjacent_accuracy",
        "first10_test_accuracy",
        "later_test_accuracy",
        "mean_prob_correct",
        "mean_circular_triads",
        "mean_transitive_triplet_fraction",
        "mean_kendall_tau_copeland_to_true",
        "mean_abs_pw_after_learning",
        "mean_abs_pw_after_full_test",
    ]:
        log(f"  {key}: {summary[key]}")
    log(f"  distance_accuracy: {summary['distance_accuracy']}")


if __name__ == "__main__":
    main()
