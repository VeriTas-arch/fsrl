"""
Paper-aligned behavioral evaluation for simple_neo_mutants_v2 checkpoints.

This evaluator targets the behavioral phenomena reported in:
Liu, Wang & Luo (2026), "Human brains construct individualized global rankings
from identical few-shot learning input".

It goes beyond a single full-28 pass by simulating a cohort of model "subjects":
  - paper-style few-shot learning: 8 behavior-graph pairs, once per block, 4 blocks
    by default;
  - paper-style no-feedback testing: all 28 unordered pairs, once per block, 10
    blocks by default;
  - repeated choices per pair, enabling subject-level error consistency, beta
    distribution fitting across subjects, HodgeRank reconstruction, circular triads,
    inter-subject ranking similarity, serial-position and symbolic-distance effects.

Critical safety/interpretation choices:
  1. Test reward is always forced to zero.
  2. Correct labels are never fed back as inputs during evaluation.
  3. The time input uses the training episode length and is clamped by default.
  4. For passive observational variants, learning relations are encoded by
     correct-first pair presentation, matching simple_neo_mutants_v2.
  5. For paper-style idiosyncrasy analyses, metrics are reported both on all
     subjects and on the subset excluding globally correct-ranking subjects.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

SCRIPT_VERSION = "simple_neo_paper_behavior_eval_2026-07-09_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import simple_neo_mutants_v2 as neo  # noqa: E402

Pair = Tuple[int, int]
N_ITEMS = 8
MAX_CIRCULAR_TRIADS_N8 = 20.0


def log(msg: str) -> None:
    print(msg, flush=True)


def all_unordered_pairs(n_items: int = N_ITEMS) -> List[Pair]:
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def unordered_pair(pair: Pair) -> Pair:
    a, b = int(pair[0]), int(pair[1])
    return (a, b) if a < b else (b, a)


def all_behavior_pairs_unordered() -> List[Pair]:
    return [unordered_pair(p) for p in neo.BEHAVIOR_GRAPH_N8]


def load_training_config(checkpoint_dir: Optional[Path]) -> dict:
    if checkpoint_dir is None:
        return {}
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text().strip()
    if not text:
        return {}
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception as exc:
            raise ValueError(f"Could not parse config file: {cfg_path}") from exc


def build_eval_config(args: argparse.Namespace) -> neo.TrainConfig:
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    file_cfg = load_training_config(ckpt_dir)

    def get_cli_or_file(cli_name: str, file_names: Iterable[str], default):
        cli_value = getattr(args, cli_name, None)
        if cli_value is not None:
            return cli_value
        for name in file_names:
            if name in file_cfg:
                return file_cfg[name]
        return default

    variant = args.variant or file_cfg.get("variant", "observational_learning_no_test_feedback_with_test_loss")
    if variant == "exact_simple_neo":
        nmin = int(args.nmin if args.nmin is not None else file_cfg.get("nbcues_min", 4))
        nmax = int(args.nmax if args.nmax is not None else file_cfg.get("nbcues_max", 8))
    else:
        nmin = nmax = N_ITEMS

    config = neo.TrainConfig(
        rngseed=args.seed,
        variant=variant,
        bs=args.eval_batch_size,
        nbiter=0,
        save_every=0,
        pe=0,
        hs=int(get_cli_or_file("hidden_size", ["hidden_size", "hs"], 200)),
        cs=int(get_cli_or_file("cue_size", ["cue_size", "cs"], 15)),
        lr=float(file_cfg.get("lr", 1e-4)),
        lpw=float(file_cfg.get("lpw", 1e-4)),
        # Keep training config values for model/time input. Paper schedules are
        # handled by the evaluator, not by changing config.eplen.
        nbtraintrials=int(file_cfg.get("nbtraintrials", 20)),
        nbtesttrials=int(file_cfg.get("nbtesttrials", 10)),
        testlmult=float(file_cfg.get("testlmult", 3.0)),
        train_supervised_loss_weight=float(file_cfg.get("train_supervised_loss_weight", 1.0)),
        test_supervised_loss_weight=float(file_cfg.get("test_supervised_loss_weight", 1.0)),
        nbcues_min=nmin,
        nbcues_max=nmax,
        write_csv=False,
        num_threads=args.num_threads,
    )
    return config


def resolve_checkpoint_path(args: argparse.Namespace) -> Optional[Path]:
    if args.allow_random_init:
        return None
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not args.checkpoint_dir:
        raise ValueError("Provide --checkpoint or --checkpoint-dir, or use --allow-random-init for smoke tests.")
    ckpt_dir = Path(args.checkpoint_dir)
    candidates = [ckpt_dir / "net.dat", ckpt_dir / f"netAE{args.train_seed_for_filename}.dat"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find net.dat in {ckpt_dir}")


def build_step_inputs_eval(
    config: neo.TrainConfig,
    nbcues: int,
    cue_data,
    cues,
    reward: np.ndarray,
    previous_actions: np.ndarray,
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
        time_value = min(numstep_ep / config.eplen, (config.eplen - 1) / config.eplen)
    elif time_mode == "rescale_paper_eval":
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
    numstep_ep: int,
    action_mode: str,
    time_mode: str,
    eval_total_steps: int,
    collect_choice: bool,
    freeze_test_plastic: bool,
):
    # Per-trial reset, matching simple_neo. Plastic weights persist across trials.
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
            numstep,
            numstep_ep,
            requires_choice,
            time_mode,
            eval_total_steps,
        )
        y_raw, _value, _daout, hidden, et, pw_new = net(inputs, hidden, et, pw)
        if collect_choice and freeze_test_plastic:
            # Diagnostic/paper choice: keep learned episode memory fixed during test.
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


def orient_pair_for_learning(config: neo.TrainConfig, pair: Pair, rng: np.random.Generator) -> Pair:
    upair = unordered_pair(pair)
    if config.variant.startswith("behavior_graph"):
        # In active rewarded variants, the network can learn either orientation.
        return (upair[1], upair[0]) if rng.random() < 0.5 else upair
    if neo.is_passive_observational_variant(config.variant):
        # No response/feedback during learning; relation sign is encoded by
        # correct-first pair presentation unless the variant explicitly uses train_aux.
        random_orientation = neo.use_train_aux_loss(config.variant) or not config.observational_train_orients_correct_first
        if random_orientation and rng.random() < 0.5:
            return (upair[1], upair[0])
        return upair
    return upair


def make_learning_schedule(config: neo.TrainConfig, args: argparse.Namespace, rng: np.random.Generator) -> List[List[Pair]]:
    """Return per-subject learning schedule: list of batch-size lists of pairs."""
    if args.learning_schedule == "paper_blocks":
        block_pairs = all_behavior_pairs_unordered()
        schedules: List[List[Pair]] = [[] for _ in range(config.bs)]
        for b in range(config.bs):
            for _block in range(args.learning_blocks):
                pairs = block_pairs.copy()
                rng.shuffle(pairs)
                schedules[b].extend([orient_pair_for_learning(config, p, rng) for p in pairs])
        return schedules

    if args.learning_schedule == "train_like":
        n_trials = config.nbtraintrials
        schedules = [[] for _ in range(config.bs)]
        for b in range(config.bs):
            for _ in range(n_trials):
                if config.variant.startswith("behavior_graph") or neo.is_passive_observational_variant(config.variant):
                    pair = neo.sample_behavior_graph_pair(random_orientation=False)
                    schedules[b].append(orient_pair_for_learning(config, tuple(pair), rng))
                elif config.variant in ("exact_simple_neo", "n8_fixed"):
                    pair = neo.sample_adjacent_pair(N_ITEMS)
                    if rng.random() < 0.5:
                        pair = (pair[1], pair[0])
                    schedules[b].append(tuple(pair))
                else:
                    raise ValueError(f"Unknown variant: {config.variant}")
        return schedules

    raise ValueError(f"Unknown learning_schedule: {args.learning_schedule}")


def make_test_schedule(args: argparse.Namespace, rng: np.random.Generator) -> List[List[Pair]]:
    base_pairs = all_unordered_pairs(N_ITEMS)
    schedules: List[List[Pair]] = [[] for _ in range(args.eval_batch_size)]
    for b in range(args.eval_batch_size):
        for _block in range(args.test_blocks):
            pairs = base_pairs.copy()
            rng.shuffle(pairs)
            if args.test_orientation == "random":
                pairs = [(j, i) if rng.random() < 0.5 else (i, j) for i, j in pairs]
            elif args.test_orientation == "canonical":
                pass
            else:
                raise ValueError(f"Unknown test_orientation: {args.test_orientation}")
            schedules[b].extend(pairs)
    return schedules


def count_circular_triads_from_winners(choice_winner: Dict[Pair, int], n_items: int = N_ITEMS) -> int:
    cycles = 0
    for i, j, k in combinations(range(n_items), 3):
        wins = {i: 0, j: 0, k: 0}
        valid = True
        for a, b in [(i, j), (i, k), (j, k)]:
            key = unordered_pair((a, b))
            if key not in choice_winner:
                valid = False
                break
            wins[int(choice_winner[key])] += 1
        if valid and wins[i] == wins[j] == wins[k] == 1:
            cycles += 1
    return cycles


def hodge_scores_from_y(y_mat: np.ndarray) -> np.ndarray:
    rows = []
    vals = []
    for i in range(N_ITEMS):
        for j in range(i + 1, N_ITEMS):
            row = np.zeros(N_ITEMS, dtype=np.float64)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
            vals.append(y_mat[i, j])
    # Gauge constraint: sum scores = 0.
    rows.append(np.ones(N_ITEMS, dtype=np.float64))
    vals.append(0.0)
    A = np.vstack(rows)
    y = np.asarray(vals, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    # rank_position[item] = 0 for highest/true rank A-like item, 7 for lowest.
    order = np.argsort(-scores)
    rank_position = np.empty_like(order)
    rank_position[order] = np.arange(len(order))
    return rank_position


def kendall_tau_rank_positions(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    for i, j in combinations(range(len(rank_a)), 2):
        da = rank_a[i] - rank_a[j]
        db = rank_b[i] - rank_b[j]
        prod = da * db
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def majority_winners_and_y(correct_counts: np.ndarray, choice_counts_for_i: np.ndarray, total_counts: np.ndarray) -> Tuple[Dict[Pair, int], np.ndarray, int]:
    """
    Build majority winner dictionary and continuous preference matrix Y.
    For i<j, Y[i,j] = mean preference for i over j in [-1, 1].
    """
    y_mat = np.zeros((N_ITEMS, N_ITEMS), dtype=np.float64)
    winners: Dict[Pair, int] = {}
    ties = 0
    for i, j in all_unordered_pairs(N_ITEMS):
        total = total_counts[i, j]
        if total <= 0:
            continue
        # choice_counts_for_i counts times item i was chosen over j for i<j.
        c_i = choice_counts_for_i[i, j]
        c_j = total - c_i
        y = (c_i - c_j) / total
        y_mat[i, j] = y
        y_mat[j, i] = -y
        if c_i > c_j:
            winners[(i, j)] = i
        elif c_j > c_i:
            winners[(i, j)] = j
        else:
            ties += 1
            # Tie broken by true order as conservative fallback; tie count is reported.
            winners[(i, j)] = i
    return winners, y_mat, ties


def fit_beta_distribution(values: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3 or stats is None:
        return None, None, "not_fit"
    eps = 1e-3
    clipped = np.clip(values, eps, 1.0 - eps)
    try:
        alpha, beta, _loc, _scale = stats.beta.fit(clipped, floc=0, fscale=1)
    except Exception:
        # Fallback method-of-moments estimate.
        m = float(np.mean(clipped))
        v = float(np.var(clipped, ddof=1))
        if v <= 0 or m <= 0 or m >= 1:
            return None, None, "not_fit"
        common = m * (1 - m) / v - 1
        alpha = m * common
        beta = (1 - m) * common
    if alpha > 1 and beta > 1:
        cls = "unimodal"
    elif alpha > 1 and beta < 1:
        cls = "high_accuracy"
    elif alpha < 1 and beta > 1:
        cls = "low_accuracy"
    elif alpha < 1 and beta < 1:
        cls = "bimodal"
    else:
        cls = "boundary"
    return float(alpha), float(beta), cls


def slope_and_t(subject_distance_acc: np.ndarray) -> dict:
    # subject_distance_acc shape: n_subjects x 7, NaNs allowed.
    xs = np.arange(1, N_ITEMS, dtype=np.float64)
    slopes = []
    for row in subject_distance_acc:
        mask = np.isfinite(row)
        if np.sum(mask) >= 2:
            slope, _intercept = np.polyfit(xs[mask], row[mask], 1)
            slopes.append(float(slope))
    if not slopes:
        return {"mean_slope": None, "t_vs_zero": None, "p_vs_zero": None, "n": 0}
    slopes_arr = np.asarray(slopes)
    if stats is not None and len(slopes_arr) > 1:
        t_stat, p_val = stats.ttest_1samp(slopes_arr, 0.0)
        return {"mean_slope": float(np.mean(slopes_arr)), "t_vs_zero": float(t_stat), "p_vs_zero": float(p_val), "n": int(len(slopes_arr))}
    return {"mean_slope": float(np.mean(slopes_arr)), "t_vs_zero": None, "p_vs_zero": None, "n": int(len(slopes_arr))}


def evaluate(config: neo.TrainConfig, net: neo.RetroModulRNN, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed if args.seed >= 0 else None)
    learned_set = set(all_behavior_pairs_unordered())
    full_pairs = all_unordered_pairs(N_ITEMS)
    n_pairs = len(full_pairs)
    total_requested = args.eval_subjects
    n_batches = math.ceil(total_requested / config.bs)

    subject_rows = []
    # Accumulators with subject x pair style data.
    pair_correct_by_subject = []
    pair_count_by_subject = []
    pair_choose_i_by_subject = []
    distance_acc_by_subject = []
    position_acc_by_subject = []
    hodge_rank_positions = []

    # max schedule length determines optional time rescaling; training eplen remains unchanged.
    learning_len = args.learning_blocks * len(learned_set) if args.learning_schedule == "paper_blocks" else config.nbtraintrials
    test_len = args.test_blocks * n_pairs
    eval_total_steps = (learning_len + test_len) * config.triallen

    net.eval()
    done = 0
    with torch.no_grad():
        for _batch_idx in range(n_batches):
            current_bs = min(config.bs, total_requested - done)
            if current_bs <= 0:
                break
            old_bs = config.bs
            if current_bs != config.bs:
                config.bs = current_bs
                net.GG["bs"] = current_bs

            nbcues = N_ITEMS
            cue_data = neo.generate_cue_data(config, nbcues)
            hidden = net.initialZeroState(config.bs)
            et = net.initialZeroET(config.bs)
            pw = net.initialZeroPlasticWeights(config.bs)
            blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(neo.DEVICE)
            for _ in range(2):
                _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

            numstep_ep = 0
            learning_schedules = make_learning_schedule(config, args, rng)
            for t in range(len(learning_schedules[0])):
                cue_pairs = [learning_schedules[b][t] for b in range(config.bs)]
                if neo.is_passive_observational_variant(config.variant):
                    requires_choice = np.zeros(config.bs, dtype="float32")
                    reward_available = np.zeros(config.bs, dtype="float32")
                else:
                    requires_choice = np.ones(config.bs, dtype="float32")
                    reward_available = np.ones(config.bs, dtype="float32")
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
                    numstep_ep,
                    args.action_mode,
                    args.time_mode,
                    eval_total_steps,
                    collect_choice=False,
                    freeze_test_plastic=False,
                )

            pw_after_learning = torch.mean(torch.abs(pw), dim=(1, 2)).detach().cpu().numpy()
            test_schedules = make_test_schedule(args, rng)

            # Per-subject matrices indexed by true-rank item indices i<j.
            correct_counts = np.zeros((config.bs, N_ITEMS, N_ITEMS), dtype=np.float64)
            total_counts = np.zeros((config.bs, N_ITEMS, N_ITEMS), dtype=np.float64)
            choose_i_counts = np.zeros((config.bs, N_ITEMS, N_ITEMS), dtype=np.float64)
            prob_correct_sum = np.zeros(config.bs, dtype=np.float64)
            prob_correct_count = np.zeros(config.bs, dtype=np.float64)

            for test_idx in range(test_len):
                cue_pairs = [test_schedules[b][test_idx] for b in range(config.bs)]
                requires_choice = np.ones(config.bs, dtype="float32")
                reward_available = np.zeros(config.bs, dtype="float32")
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
                    numstep_ep,
                    args.action_mode,
                    args.time_mode,
                    eval_total_steps,
                    collect_choice=True,
                    freeze_test_plastic=args.freeze_test_plastic,
                )
                for b, pair in enumerate(cue_pairs):
                    upair = unordered_pair(pair)
                    i, j = upair
                    total_counts[b, i, j] += 1.0
                    correct_counts[b, i, j] += float(correct[b])
                    prob_correct_sum[b] += float(prob_correct[b])
                    prob_correct_count[b] += 1.0
                    if winners[b] == i:
                        choose_i_counts[b, i, j] += 1.0

            pw_after_test = torch.mean(torch.abs(pw), dim=(1, 2)).detach().cpu().numpy()

            for b in range(config.bs):
                pair_acc_vec = []
                pair_count_vec = []
                pair_choose_i_vec = []
                for i, j in full_pairs:
                    total = total_counts[b, i, j]
                    acc = np.nan if total <= 0 else correct_counts[b, i, j] / total
                    pair_acc_vec.append(acc)
                    pair_count_vec.append(total)
                    pair_choose_i_vec.append(choose_i_counts[b, i, j])
                pair_correct_by_subject.append(pair_acc_vec)
                pair_count_by_subject.append(pair_count_vec)
                pair_choose_i_by_subject.append(pair_choose_i_vec)

                winners, y_mat, tie_count = majority_winners_and_y(correct_counts[b], choose_i_counts[b], total_counts[b])
                circular = count_circular_triads_from_winners(winners, N_ITEMS)
                self_consistency_coeff = 1.0 - circular / MAX_CIRCULAR_TRIADS_N8
                hodge_scores = hodge_scores_from_y(y_mat)
                hodge_rank = ranks_from_scores(hodge_scores)
                true_rank = np.arange(N_ITEMS)
                tau_true = kendall_tau_rank_positions(hodge_rank, true_rank)
                hodge_rank_positions.append(hodge_rank)

                # Ranking class based on majority tournament.
                majority_correct_all = all(winners.get((i, j), None) == i for i, j in full_pairs)
                if majority_correct_all:
                    ranking_class = "correct"
                elif circular == 0:
                    ranking_class = "self_consistent_incorrect"
                else:
                    ranking_class = "self_inconsistent"

                # Accuracy groups.
                learned_accs = []
                nonlearned_accs = []
                adjacent_accs = []
                nonadjacent_accs = []
                dist_acc_sum = np.zeros(N_ITEMS, dtype=np.float64)
                dist_acc_count = np.zeros(N_ITEMS, dtype=np.float64)
                pos_acc_sum = np.zeros(N_ITEMS, dtype=np.float64)
                pos_acc_count = np.zeros(N_ITEMS, dtype=np.float64)
                pair_index = 0
                for i, j in full_pairs:
                    total = total_counts[b, i, j]
                    if total <= 0:
                        pair_index += 1
                        continue
                    acc = correct_counts[b, i, j] / total
                    dist = abs(i - j)
                    dist_acc_sum[dist] += acc
                    dist_acc_count[dist] += 1.0
                    pos_acc_sum[i] += acc
                    pos_acc_count[i] += 1.0
                    pos_acc_sum[j] += acc
                    pos_acc_count[j] += 1.0
                    if (i, j) in learned_set:
                        learned_accs.append(acc)
                    else:
                        nonlearned_accs.append(acc)
                    if dist == 1:
                        adjacent_accs.append(acc)
                    else:
                        nonadjacent_accs.append(acc)
                    pair_index += 1

                dist_row = np.full(N_ITEMS - 1, np.nan, dtype=np.float64)
                for d in range(1, N_ITEMS):
                    if dist_acc_count[d] > 0:
                        dist_row[d - 1] = dist_acc_sum[d] / dist_acc_count[d]
                distance_acc_by_subject.append(dist_row)

                pos_row = np.full(N_ITEMS, np.nan, dtype=np.float64)
                for p in range(N_ITEMS):
                    if pos_acc_count[p] > 0:
                        pos_row[p] = pos_acc_sum[p] / pos_acc_count[p]
                position_acc_by_subject.append(pos_row)

                # Error consistency thresholds: subject has at least one pair with
                # error proportion >= threshold, and number/proportion of such pairs.
                threshold_metrics = {}
                for thr in [0.60, 0.70, 0.80, 0.90, 1.00]:
                    err_pairs = 0
                    for i, j in full_pairs:
                        total = total_counts[b, i, j]
                        if total <= 0:
                            continue
                        err_rate = 1.0 - correct_counts[b, i, j] / total
                        if err_rate >= thr - 1e-9:
                            err_pairs += 1
                    label = int(round(thr * 100))
                    threshold_metrics[f"n_consistent_error_pairs_{label}"] = err_pairs
                    threshold_metrics[f"has_consistent_error_pair_{label}"] = int(err_pairs > 0)

                subject_rows.append(
                    {
                        "subject": done + b,
                        "overall_accuracy": float(np.nanmean(pair_acc_vec)),
                        "learned_accuracy": float(np.nanmean(learned_accs)) if learned_accs else np.nan,
                        "nonlearned_accuracy": float(np.nanmean(nonlearned_accs)) if nonlearned_accs else np.nan,
                        "adjacent_accuracy": float(np.nanmean(adjacent_accs)) if adjacent_accs else np.nan,
                        "nonadjacent_accuracy": float(np.nanmean(nonadjacent_accs)) if nonadjacent_accs else np.nan,
                        "mean_prob_correct": float(prob_correct_sum[b] / max(1.0, prob_correct_count[b])),
                        "circular_triads": int(circular),
                        "self_consistency_coefficient": float(self_consistency_coeff),
                        "ranking_class": ranking_class,
                        "kendall_tau_hodge_to_true": float(tau_true),
                        "tie_pairs_majority": int(tie_count),
                        "mean_abs_pw_after_learning": float(pw_after_learning[b]),
                        "mean_abs_pw_after_test": float(pw_after_test[b]),
                        **threshold_metrics,
                    }
                )

            done += current_bs
            if current_bs != old_bs:
                config.bs = old_bs
                net.GG["bs"] = old_bs

    # Convert accumulated arrays.
    pair_acc_by_subject = np.asarray(pair_correct_by_subject, dtype=np.float64)  # subjects x 28 accuracies
    pair_count_by_subject = np.asarray(pair_count_by_subject, dtype=np.float64)
    pair_choose_i_by_subject = np.asarray(pair_choose_i_by_subject, dtype=np.float64)
    distance_acc_by_subject = np.asarray(distance_acc_by_subject, dtype=np.float64)
    position_acc_by_subject = np.asarray(position_acc_by_subject, dtype=np.float64)
    hodge_rank_positions = np.asarray(hodge_rank_positions, dtype=np.int64)

    # Subject filter used in paper for idiosyncratic/pair-level analyses.
    correct_mask = np.asarray([r["ranking_class"] == "correct" for r in subject_rows], dtype=bool)
    if args.paper_exclude_correct_rankers:
        analysis_mask = ~correct_mask
    else:
        analysis_mask = np.ones(len(subject_rows), dtype=bool)

    # Pair-level rows and beta fits.
    pair_rows = []
    beta_rows = []
    beta_class_counts = {"unimodal": 0, "high_accuracy": 0, "low_accuracy": 0, "bimodal": 0, "boundary": 0, "not_fit": 0}
    for pair_idx, (i, j) in enumerate(full_pairs):
        vals_all = pair_acc_by_subject[:, pair_idx]
        vals_analysis = pair_acc_by_subject[analysis_mask, pair_idx]
        alpha, beta, cls = fit_beta_distribution(vals_analysis)
        beta_class_counts[cls] = beta_class_counts.get(cls, 0) + 1
        row_common = {
            "pair": f"{i}-{j}",
            "i": i,
            "j": j,
            "distance": abs(i - j),
            "is_learned_pair": (i, j) in learned_set,
            "mean_accuracy_all_subjects": float(np.nanmean(vals_all)),
            "mean_accuracy_analysis_subjects": float(np.nanmean(vals_analysis)) if vals_analysis.size else np.nan,
            "n_subjects_all": int(len(vals_all)),
            "n_subjects_analysis": int(np.sum(analysis_mask)),
        }
        pair_rows.append(row_common)
        beta_rows.append({**row_common, "beta_alpha": alpha, "beta_beta": beta, "beta_class": cls})

    # Inter-subject ranking similarity based on Hodge reconstructed ranks.
    tau_vals = []
    idxs = np.where(analysis_mask)[0]
    for a_idx_pos in range(len(idxs)):
        for b_idx_pos in range(a_idx_pos + 1, len(idxs)):
            a = idxs[a_idx_pos]
            b = idxs[b_idx_pos]
            tau_vals.append(kendall_tau_rank_positions(hodge_rank_positions[a], hodge_rank_positions[b]))
    tau_vals = np.asarray(tau_vals, dtype=np.float64)

    # Aggregate subject rows.
    def mean_col(name: str, mask: Optional[np.ndarray] = None):
        vals = np.asarray([r[name] for r in subject_rows], dtype=np.float64)
        if mask is not None:
            vals = vals[mask]
        return None if len(vals) == 0 else float(np.nanmean(vals))

    def count_class(cls: str):
        return int(sum(1 for r in subject_rows if r["ranking_class"] == cls))

    def has_count(threshold_label: int, mask: Optional[np.ndarray] = None):
        vals = np.asarray([r[f"has_consistent_error_pair_{threshold_label}"] for r in subject_rows], dtype=np.float64)
        if mask is not None:
            vals = vals[mask]
        return int(np.nansum(vals)), float(np.nanmean(vals)) if len(vals) else None

    distance_mean = np.nanmean(distance_acc_by_subject, axis=0)
    position_mean = np.nanmean(position_acc_by_subject, axis=0)
    slope_stats = slope_and_t(distance_acc_by_subject)

    summary = {
        "script_version": SCRIPT_VERSION,
        "model_script_version": getattr(neo, "SCRIPT_VERSION", "unknown"),
        "variant": config.variant,
        "checkpoint": str(args.checkpoint or (Path(args.checkpoint_dir) / "net.dat" if args.checkpoint_dir else "random_init")),
        "eval_subjects": int(len(subject_rows)),
        "analysis_subjects_after_correct_ranker_filter": int(np.sum(analysis_mask)),
        "paper_exclude_correct_rankers": bool(args.paper_exclude_correct_rankers),
        "learning_schedule": args.learning_schedule,
        "learning_blocks": args.learning_blocks,
        "test_blocks": args.test_blocks,
        "test_trials_per_pair": args.test_blocks,
        "test_pairs_total_per_subject": int(args.test_blocks * n_pairs),
        "test_reward_forced_zero": True,
        "freeze_test_plastic": bool(args.freeze_test_plastic),
        "action_mode": args.action_mode,
        "time_mode": args.time_mode,
        "training_eplen_for_time_input": config.eplen,
        "overall_accuracy": mean_col("overall_accuracy"),
        "learned_pairs_accuracy": mean_col("learned_accuracy"),
        "nonlearned_pairs_accuracy": mean_col("nonlearned_accuracy"),
        "adjacent_accuracy": mean_col("adjacent_accuracy"),
        "nonadjacent_accuracy": mean_col("nonadjacent_accuracy"),
        "mean_prob_correct": mean_col("mean_prob_correct"),
        "distance_accuracy": {str(d): float(distance_mean[d - 1]) for d in range(1, N_ITEMS)},
        "serial_position_accuracy": {str(pos + 1): float(position_mean[pos]) for pos in range(N_ITEMS)},
        "symbolic_distance_slope": slope_stats,
        "mean_circular_triads": mean_col("circular_triads"),
        "mean_self_consistency_coefficient": mean_col("self_consistency_coefficient"),
        "mean_kendall_tau_hodge_to_true": mean_col("kendall_tau_hodge_to_true"),
        "ranking_class_counts": {
            "correct": count_class("correct"),
            "self_consistent_incorrect": count_class("self_consistent_incorrect"),
            "self_inconsistent": count_class("self_inconsistent"),
        },
        "consistent_error_subject_counts_all": {},
        "consistent_error_subject_counts_analysis": {},
        "beta_pair_class_counts_analysis_subjects": beta_class_counts,
        "mean_inter_subject_kendall_tau_hodge": None if len(tau_vals) == 0 else float(np.nanmean(tau_vals)),
        "std_inter_subject_kendall_tau_hodge": None if len(tau_vals) == 0 else float(np.nanstd(tau_vals)),
        "mean_abs_pw_after_learning": mean_col("mean_abs_pw_after_learning"),
        "mean_abs_pw_after_test": mean_col("mean_abs_pw_after_test"),
    }
    for label in [60, 70, 80, 90, 100]:
        n_all, prop_all = has_count(label)
        n_analysis, prop_analysis = has_count(label, analysis_mask)
        summary["consistent_error_subject_counts_all"][str(label)] = {"n": n_all, "proportion": prop_all}
        summary["consistent_error_subject_counts_analysis"][str(label)] = {"n": n_analysis, "proportion": prop_analysis}

    distance_rows = [
        {"distance": d, "accuracy": float(distance_mean[d - 1])} for d in range(1, N_ITEMS)
    ]
    position_rows = [
        {"rank_position": pos + 1, "accuracy": float(position_mean[pos])} for pos in range(N_ITEMS)
    ]
    return summary, subject_rows, pair_rows, beta_rows, distance_rows, position_rows


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(summary: dict, subject_rows: list, pair_rows: list, beta_rows: list, distance_rows: list, position_rows: list, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_behavior_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    save_csv(output_dir / "subject_metrics.csv", subject_rows)
    save_csv(output_dir / "pair_accuracy_matrix.csv", pair_rows)
    save_csv(output_dir / "beta_pair_fits.csv", beta_rows)
    save_csv(output_dir / "distance_effect.csv", distance_rows)
    save_csv(output_dir / "serial_position_effect.csv", position_rows)
    log(f"[save] summary: {output_dir / 'paper_behavior_summary.json'}")
    log(f"[save] subject metrics: {output_dir / 'subject_metrics.csv'}")
    log(f"[save] pair/beta/distance/serial CSVs written under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned behavioral evaluator for simple_neo_mutants_v2 checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Training output dir containing net.dat and config.json.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path. Overrides --checkpoint-dir/net.dat.")
    parser.add_argument("--variant", type=str, default=None, help="Override variant; defaults to config.json variant.")
    parser.add_argument("--eval-subjects", type=int, default=512, help="Number of model subjects/episodes.")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--cue-size", type=int, default=None)
    parser.add_argument("--nmin", type=int, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--train-seed-for-filename", type=int, default=1)
    parser.add_argument("--learning-schedule", choices=["paper_blocks", "train_like"], default="paper_blocks")
    parser.add_argument("--learning-blocks", type=int, default=4, help="Paper uses 4 blocks, each behavior pair once per block.")
    parser.add_argument("--test-blocks", type=int, default=10, help="Paper uses 10 blocks, each all 28 pairs once per block.")
    parser.add_argument("--test-orientation", choices=["random", "canonical"], default="random")
    parser.add_argument("--action-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--time-mode", choices=["clamp", "original", "rescale_paper_eval"], default="clamp")
    parser.add_argument("--freeze-test-plastic", action="store_true", help="Keep plastic weights fixed during no-feedback test blocks.")
    parser.add_argument("--paper-exclude-correct-rankers", action="store_true", help="For beta/error/idiosyncrasy summaries, exclude subjects with perfectly correct majority rankings, matching the paper's pair-level analysis logic.")
    parser.add_argument("--allow-random-init", action="store_true", help="Smoke test mode: evaluate a random untrained model.")
    parser.add_argument("--output-dir", type=str, default=None)
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
    net = neo.RetroModulRNN(config.to_model_dict())
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=neo.DEVICE)
        net.load_state_dict(state)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_dir or "/mnt/data") / "paper_behavior_eval"

    log(f"[setup] paper evaluator version: {SCRIPT_VERSION}")
    log(f"[setup] model script version: {getattr(neo, 'SCRIPT_VERSION', 'unknown')}")
    log(f"[setup] device: {neo.DEVICE}; torch_num_threads={torch.get_num_threads()}")
    log(f"[setup] checkpoint: {ckpt_path if ckpt_path is not None else 'random_init'}")
    log(f"[setup] variant: {config.variant}; eval_subjects={args.eval_subjects}; batch={config.bs}")
    log(f"[setup] learning_schedule={args.learning_schedule}; learning_blocks={args.learning_blocks}; test_blocks={args.test_blocks}")
    log(f"[setup] test_orientation={args.test_orientation}; action_mode={args.action_mode}; time_mode={args.time_mode}; freeze_test_plastic={args.freeze_test_plastic}")
    log("[setup] Test reward is forced to zero for all paper-style test trials.")

    summary, subject_rows, pair_rows, beta_rows, distance_rows, position_rows = evaluate(config, net, args)
    save_outputs(summary, subject_rows, pair_rows, beta_rows, distance_rows, position_rows, output_dir)

    log("[summary]")
    for key in [
        "overall_accuracy",
        "learned_pairs_accuracy",
        "nonlearned_pairs_accuracy",
        "adjacent_accuracy",
        "nonadjacent_accuracy",
        "mean_circular_triads",
        "mean_self_consistency_coefficient",
        "mean_kendall_tau_hodge_to_true",
        "mean_inter_subject_kendall_tau_hodge",
    ]:
        log(f"  {key}: {summary[key]}")
    log(f"  distance_accuracy: {summary['distance_accuracy']}")
    log(f"  serial_position_accuracy: {summary['serial_position_accuracy']}")
    log(f"  symbolic_distance_slope: {summary['symbolic_distance_slope']}")
    log(f"  ranking_class_counts: {summary['ranking_class_counts']}")
    log(f"  consistent_error_subject_counts_analysis: {summary['consistent_error_subject_counts_analysis']}")
    log(f"  beta_pair_class_counts_analysis_subjects: {summary['beta_pair_class_counts_analysis_subjects']}")


if __name__ == "__main__":
    main()
