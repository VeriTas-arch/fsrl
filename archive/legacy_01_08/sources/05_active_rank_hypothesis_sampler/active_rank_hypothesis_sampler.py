"""
Active rank-hypothesis attractor sampler for the few-shot ranking task.

This file is deliberately separate from ``simple_neo.py``.  The previous models in
this package update a continuous score vector.  Here the structural change is that
a virtual subject commits to one *discrete global ranking hypothesis* after sparse
local learning.  This is meant to model a human-like constructive mechanism:

1. Local pair observations are stored with subject-specific reliability.
2. An active reinstatement / replay stage evaluates many possible global rankings
   against the remembered edges.
3. Limited cognitive resource and neural noise make the system sample one attractor
   state, rather than average over the whole posterior.
4. The no-feedback test phase reads out the committed ranking, so errors are stable
   and self-consistent.

The model uses exactly the Liu/Wang/Luo 8-item task used elsewhere in this package:
8 fixed non-adjacent learning pairs, 4 presentations per pair, and testing on all
28 pairs for 10 no-feedback repetitions.

Run a single evaluation:
    python active_rank_hypothesis_sampler.py --output-dir outputs_active_sampler

Run a compact parameter sweep and then the best preset:
    python active_rank_hypothesis_sampler.py --sweep --output-dir outputs_active_sampler_sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.stats import beta as beta_dist
except Exception:  # pragma: no cover - scipy is listed in requirements, this is just defensive.
    beta_dist = None


PAPER_LEARNING_PAIRS_RANK: list[tuple[int, int]] = [
    (0, 5),  # A-F, distance 5
    (1, 2),  # B-C, distance 1
    (1, 4),  # B-E, distance 3
    (2, 6),  # C-G, distance 4
    (3, 5),  # D-F, distance 2
    (3, 6),  # D-G, distance 3
    (4, 7),  # E-H, distance 3
    (0, 7),  # A-H, distance 7
]


@dataclass
class SamplerConfig:
    seed: int = 42
    eval_subjects: int = 77
    eval_repetitions: int = 10
    n_items: int = 8

    # Evidence model. sigma_distance is intentionally large enough that exact
    # distances do not force the unique true ranking; the learner treats distance
    # as fuzzy magnitude information.
    sigma_distance: float = 2.0
    order_bonus: float = 1.5
    posterior_temperature: float = 1.0

    # Subject-specific edge-memory reliability. Larger distances and anchors tend
    # to be encoded more reliably, but each subject gets a different reliability
    # profile. This is the main place where identical input becomes individualized.
    edge_weight_concentration: float = 8.0
    distance_salience: float = 1.2
    anchor_bias: float = 0.0
    encoding_dropout: float = 0.0

    # Subject-level cognitive resource / posterior precision. High-resource
    # subjects sample near-MAP rankings; low-resource subjects sample broader
    # but still globally coherent attractors.
    resource_log_mean: float = 0.0
    resource_log_sd: float = 0.6

    # Choice readout from the committed ranking during no-feedback testing.
    choice_beta: float = 2.2
    lapse: float = 0.03

    output_dir: str = "outputs_active_sampler"

    @property
    def all_rank_pairs(self) -> list[tuple[int, int]]:
        return list(combinations(range(self.n_items), 2))


@dataclass
class PrecomputedHypotheses:
    permutations: np.ndarray  # [P, N], low-to-high item order.
    positions: np.ndarray  # [P, N], item -> subjective rank position.
    learning_diffs: np.ndarray  # [P, E], pos[b] - pos[a] for learned edges.
    learning_order_ok: np.ndarray  # [P, E], whether learned edge order is correct.
    test_diffs: np.ndarray  # [P, 28], pos[b] - pos[a] for all true rank pairs.
    all_pairs: list[tuple[int, int]]
    learned_pair_indices: np.ndarray
    nonlearned_pair_indices: np.ndarray


def precompute_hypotheses(config: SamplerConfig) -> PrecomputedHypotheses:
    n_items = config.n_items
    all_pairs = config.all_rank_pairs
    permutations = np.array(list(itertools.permutations(range(n_items))), dtype=np.int16)
    positions = np.empty_like(permutations)
    for idx, perm in enumerate(permutations):
        positions[idx, perm] = np.arange(n_items, dtype=np.int16)

    learning_diffs = np.stack(
        [positions[:, b] - positions[:, a] for a, b in PAPER_LEARNING_PAIRS_RANK], axis=1
    ).astype(np.float32)
    learning_order_ok = (learning_diffs > 0).astype(np.float32)
    test_diffs = np.stack([positions[:, b] - positions[:, a] for a, b in all_pairs], axis=1).astype(np.float32)
    learned_pair_indices = np.array([all_pairs.index(pair) for pair in PAPER_LEARNING_PAIRS_RANK], dtype=np.int64)
    nonlearned_pair_indices = np.array(
        [idx for idx, pair in enumerate(all_pairs) if pair not in set(PAPER_LEARNING_PAIRS_RANK)], dtype=np.int64
    )
    return PrecomputedHypotheses(
        permutations=permutations,
        positions=positions,
        learning_diffs=learning_diffs,
        learning_order_ok=learning_order_ok,
        test_diffs=test_diffs,
        all_pairs=all_pairs,
        learned_pair_indices=learned_pair_indices,
        nonlearned_pair_indices=nonlearned_pair_indices,
    )


def kendall_tau_order(order_a: Iterable[int], order_b: Iterable[int]) -> float:
    a = list(order_a)
    b = list(order_b)
    pos_a = {item: idx for idx, item in enumerate(a)}
    pos_b = {item: idx for idx, item in enumerate(b)}
    concordant = 0
    discordant = 0
    for i, j in combinations(a, 2):
        sign_a = pos_a[i] - pos_a[j]
        sign_b = pos_b[i] - pos_b[j]
        if sign_a * sign_b > 0:
            concordant += 1
        elif sign_a * sign_b < 0:
            discordant += 1
    denom = concordant + discordant
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def self_consistency_from_majority(pair_acc: np.ndarray, pair_list: list[tuple[int, int]], n_items: int) -> tuple[float, int]:
    """Circular-triad consistency from majority choices.

    pair_acc is accuracy for rank-ordered pairs (a,b) where b is truly higher.
    If accuracy >= 0.5, majority preference is b > a; otherwise a > b.
    """
    pref = np.zeros((n_items, n_items), dtype=bool)
    for idx, (a, b) in enumerate(pair_list):
        if pair_acc[idx] >= 0.5:
            pref[b, a] = True
        else:
            pref[a, b] = True
    circular = 0
    for a, b, c in combinations(range(n_items), 3):
        if (pref[a, b] and pref[b, c] and pref[c, a]) or (pref[a, c] and pref[c, b] and pref[b, a]):
            circular += 1
    max_triads = (n_items**3 - 4 * n_items) // 24 if n_items % 2 == 0 else (n_items**3 - n_items) // 24
    return 1.0 - circular / max_triads, circular


def beta_fit_category(values: np.ndarray) -> tuple[float, float, str]:
    """Fit beta distribution and classify as in the behavioral paper.

    Accuracy values are discrete proportions and can be exactly 0/1, so we apply
    a tiny clipping before fitting.
    """
    if beta_dist is None:
        return float("nan"), float("nan"), "scipy_unavailable"
    clipped = np.clip(values.astype(float), 1e-3, 1.0 - 1e-3)
    alpha, beta, _loc, _scale = beta_dist.fit(clipped, floc=0.0, fscale=1.0)
    if alpha < 1.0 and beta < 1.0:
        category = "bimodal"
    elif alpha > 1.0 and beta < 1.0:
        category = "high_accuracy"
    elif alpha < 1.0 and beta > 1.0:
        category = "low_accuracy"
    else:
        category = "unimodal"
    return float(alpha), float(beta), category


def simulate(config: SamplerConfig, pre: PrecomputedHypotheses | None = None, fit_beta: bool = True) -> dict:
    if pre is None:
        pre = precompute_hypotheses(config)
    rng = np.random.default_rng(config.seed)
    n_subj = config.eval_subjects
    reps = config.eval_repetitions
    n_edges = len(PAPER_LEARNING_PAIRS_RANK)
    true_learning_diffs = np.array([b - a for a, b in PAPER_LEARNING_PAIRS_RANK], dtype=np.float32)
    anchor_flags = np.array(
        [1.0 if (a in (0, config.n_items - 1) or b in (0, config.n_items - 1)) else 0.0 for a, b in PAPER_LEARNING_PAIRS_RANK],
        dtype=np.float32,
    )
    base_reliability = (true_learning_diffs / (config.n_items - 1)) ** config.distance_salience
    base_reliability = base_reliability + config.anchor_bias * anchor_flags + 1e-3
    base_reliability = base_reliability / base_reliability.mean()

    pair_acc = np.zeros((n_subj, len(pre.all_pairs)), dtype=np.float32)
    chosen_orders = np.zeros((n_subj, config.n_items), dtype=np.int16)
    chosen_indices = np.zeros(n_subj, dtype=np.int64)
    resources = np.zeros(n_subj, dtype=np.float32)
    reliabilities = np.zeros((n_subj, n_edges), dtype=np.float32)
    posterior_entropy = np.zeros(n_subj, dtype=np.float32)
    chosen_logp = np.zeros(n_subj, dtype=np.float32)

    for subj in range(n_subj):
        if config.edge_weight_concentration > 0:
            rel = rng.gamma(
                shape=config.edge_weight_concentration * base_reliability,
                scale=1.0 / config.edge_weight_concentration,
            ).astype(np.float32)
        else:
            rel = np.ones(n_edges, dtype=np.float32)
        if config.encoding_dropout > 0:
            rel = rel * (rng.random(n_edges) >= config.encoding_dropout).astype(np.float32)
        reliabilities[subj] = rel

        resource = float(math.exp(config.resource_log_mean + config.resource_log_sd * rng.normal()))
        resources[subj] = resource

        distance_sse = ((pre.learning_diffs - true_learning_diffs) ** 2 / (2.0 * config.sigma_distance**2)) * rel
        log_posterior = -distance_sse.sum(axis=1) + config.order_bonus * (pre.learning_order_ok * rel).sum(axis=1)
        log_posterior = log_posterior * resource / max(config.posterior_temperature, 1e-6)
        log_posterior = log_posterior - float(log_posterior.max())
        probs = np.exp(log_posterior)
        probs = probs / probs.sum()
        # Entropy in nats; low entropy means strong commitment to a small set of attractors.
        posterior_entropy[subj] = float(-(probs * np.log(probs + 1e-12)).sum())
        chosen_idx = int(rng.choice(len(pre.permutations), p=probs))
        chosen_indices[subj] = chosen_idx
        chosen_orders[subj] = pre.permutations[chosen_idx]
        chosen_logp[subj] = float(log_posterior[chosen_idx])

        subjective_diffs = pre.test_diffs[chosen_idx]
        p_correct = config.lapse * 0.5 + (1.0 - config.lapse) / (1.0 + np.exp(-config.choice_beta * subjective_diffs))
        pair_acc[subj] = rng.binomial(reps, p_correct) / reps

    summary, pair_rows, subject_rows = evaluate_outputs(config, pre, pair_acc, chosen_orders, resources, reliabilities, posterior_entropy, chosen_logp, fit_beta=fit_beta)
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "subject_rows": subject_rows,
        "pair_accuracy_matrix": pair_acc,
        "chosen_orders": chosen_orders,
        "reliabilities": reliabilities,
        "resources": resources,
        "chosen_indices": chosen_indices,
    }


def evaluate_outputs(
    config: SamplerConfig,
    pre: PrecomputedHypotheses,
    pair_acc: np.ndarray,
    chosen_orders: np.ndarray,
    resources: np.ndarray,
    reliabilities: np.ndarray,
    posterior_entropy: np.ndarray,
    chosen_logp: np.ndarray,
    fit_beta: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    pair_error_prop = 1.0 - pair_acc
    learned_acc = float(pair_acc[:, pre.learned_pair_indices].mean())
    nonlearned_acc = float(pair_acc[:, pre.nonlearned_pair_indices].mean())
    overall_acc = float(pair_acc.mean())

    consistent_subjects_80 = int((pair_error_prop >= 0.8).any(axis=1).sum())
    consistent_subjects_100 = int((pair_error_prop >= 1.0).any(axis=1).sum())

    choice_self_consistency = []
    choice_circular_counts = []
    correct_ranking_count = 0
    self_consistent_incorrect_count = 0
    self_inconsistent_count = 0
    true_order = tuple(range(config.n_items))

    for subj in range(config.eval_subjects):
        sc, circ = self_consistency_from_majority(pair_acc[subj], pre.all_pairs, config.n_items)
        choice_self_consistency.append(sc)
        choice_circular_counts.append(circ)
        if tuple(int(x) for x in chosen_orders[subj]) == true_order:
            correct_ranking_count += 1
        elif circ == 0:
            self_consistent_incorrect_count += 1
        else:
            self_inconsistent_count += 1

    taus = []
    for i, j in combinations(range(config.eval_subjects), 2):
        taus.append(kendall_tau_order(chosen_orders[i], chosen_orders[j]))

    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pre.all_pairs) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())

    # Serial position effect proxy: accuracy averaged over all comparisons containing each item.
    serial_position_acc = {}
    for item in range(config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pre.all_pairs) if a == item or b == item]
        serial_position_acc[str(item + 1)] = float(pair_acc[:, idx].mean())

    pair_rows: list[dict] = []
    beta_counts = {"bimodal": 0, "high_accuracy": 0, "low_accuracy": 0, "unimodal": 0, "scipy_unavailable": 0}
    learned_set = set(PAPER_LEARNING_PAIRS_RANK)
    for pidx, (a, b) in enumerate(pre.all_pairs):
        if fit_beta:
            alpha, beta, category = beta_fit_category(pair_acc[:, pidx])
        else:
            # Fast sweep-time proxy: enough to rank parameter sets without thousands
            # of maximum-likelihood beta fits. The final selected run still uses
            # the exact beta fit.
            vals = pair_acc[:, pidx]
            low_mass = float((vals <= 0.2).mean())
            high_mass = float((vals >= 0.8).mean())
            if low_mass > 0.15 and high_mass > 0.15:
                alpha, beta, category = float("nan"), float("nan"), "bimodal"
            elif high_mass > 0.65:
                alpha, beta, category = float("nan"), float("nan"), "high_accuracy"
            elif low_mass > 0.65:
                alpha, beta, category = float("nan"), float("nan"), "low_accuracy"
            else:
                alpha, beta, category = float("nan"), float("nan"), "unimodal"
        beta_counts[category] = beta_counts.get(category, 0) + 1
        pair_rows.append(
            {
                "pair": f"{chr(65 + a)}-{chr(65 + b)}",
                "rank_a": a,
                "rank_b": b,
                "rank_distance": b - a,
                "learned_pair": (a, b) in learned_set,
                "mean_accuracy": float(pair_acc[:, pidx].mean()),
                "beta_alpha": alpha,
                "beta_beta": beta,
                "beta_category": category,
                "subjects_majority_error": int((pair_acc[:, pidx] < 0.5).sum()),
            }
        )

    subject_rows: list[dict] = []
    for subj in range(config.eval_subjects):
        order = [chr(65 + int(x)) for x in chosen_orders[subj]]
        tau_true = kendall_tau_order(chosen_orders[subj], true_order)
        consistent_error_pairs_80 = int((pair_error_prop[subj] >= 0.8).sum())
        consistent_error_pairs_100 = int((pair_error_prop[subj] >= 1.0).sum())
        subject_rows.append(
            {
                "subject": subj + 1,
                "order_low_to_high": "".join(order),
                "kendall_tau_with_true": float(tau_true),
                "pair_accuracy_mean": float(pair_acc[subj].mean()),
                "learned_accuracy": float(pair_acc[subj, pre.learned_pair_indices].mean()),
                "nonlearned_accuracy": float(pair_acc[subj, pre.nonlearned_pair_indices].mean()),
                "self_consistency": float(choice_self_consistency[subj]),
                "circular_triads": int(choice_circular_counts[subj]),
                "consistent_error_pairs_80pct": consistent_error_pairs_80,
                "consistent_error_pairs_100pct": consistent_error_pairs_100,
                "resource": float(resources[subj]),
                "posterior_entropy": float(posterior_entropy[subj]),
                "chosen_relative_log_posterior": float(chosen_logp[subj]),
            }
        )

    summary = {
        "model": "active_rank_hypothesis_attractor_sampler",
        "config": asdict(config),
        "eval_subjects": config.eval_subjects,
        "eval_repetitions": config.eval_repetitions,
        "overall_accuracy": overall_acc,
        "learned_pairs_accuracy": learned_acc,
        "nonlearned_pairs_accuracy": nonlearned_acc,
        "consistent_error_subjects_80pct": consistent_subjects_80,
        "consistent_error_subjects_80pct_ratio": consistent_subjects_80 / config.eval_subjects,
        "consistent_error_subjects_100pct": consistent_subjects_100,
        "consistent_error_subjects_100pct_ratio": consistent_subjects_100 / config.eval_subjects,
        "mean_self_consistency_from_majority_choices": float(np.mean(choice_self_consistency)),
        "mean_circular_triads_from_majority_choices": float(np.mean(choice_circular_counts)),
        "correct_ranking_subjects": correct_ranking_count,
        "self_consistent_incorrect_subjects": self_consistent_incorrect_count,
        "self_inconsistent_subjects": self_inconsistent_count,
        "mean_inter_subject_kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "distance_accuracy": distance_acc,
        "serial_position_accuracy": serial_position_acc,
        "beta_pair_category_counts": beta_counts,
        "mean_resource": float(resources.mean()),
        "std_resource": float(resources.std()),
        "mean_posterior_entropy": float(posterior_entropy.mean()),
    }
    return summary, pair_rows, subject_rows


def write_outputs(output_dir: Path, config: SamplerConfig, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_active_sampler.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    with open(output_dir / "active_sampler_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)
    with open(output_dir / "pair_accuracy_beta_fits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["pair_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["pair_rows"])
    with open(output_dir / "subject_rankings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["subject_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["subject_rows"])
    np.save(output_dir / "pair_accuracy_matrix.npy", result["pair_accuracy_matrix"])
    np.save(output_dir / "chosen_orders.npy", result["chosen_orders"])
    np.save(output_dir / "edge_reliabilities.npy", result["reliabilities"])


def compact_score(summary: dict) -> float:
    """Heuristic score for selecting promising parameter settings.

    Targets are approximate behavioral desiderata, not hard-fitted human data.
    """
    targets = {
        "overall_accuracy": 0.84,
        "learned_pairs_accuracy": 0.92,
        "nonlearned_pairs_accuracy": 0.81,
        "consistent_error_subjects_80pct_ratio": 0.91,
        "consistent_error_subjects_100pct_ratio": 0.70,
        "mean_self_consistency_from_majority_choices": 1.00,
        "mean_inter_subject_kendall_tau": 0.55,
    }
    score = 0.0
    for key, target in targets.items():
        score += abs(summary[key] - target)
    # Mild preference for a few fully correct subjects, as in the behavioral paper.
    score += abs(summary["correct_ranking_subjects"] - 8) / 100.0
    if summary["learned_pairs_accuracy"] < summary["nonlearned_pairs_accuracy"]:
        score += 0.25
    return score


def run_sweep(base_config: SamplerConfig, output_dir: Path) -> list[dict]:
    pre = precompute_hypotheses(base_config)
    rows: list[dict] = []
    # Compact but meaningful sweep. It varies structural uncertainty, ordinal evidence,
    # subject resource variability, and choice stochasticity.
    # Kept intentionally compact so a normal 600 s timeout is sufficient.
    sigma_values = [1.7, 2.0, 2.3]
    order_bonus_values = [1.0, 1.5, 2.0]
    resource_sd_values = [0.0, 0.6]
    beta_values = [2.2]
    lapse_values = [0.03, 0.05]
    seed_values = [base_config.seed]
    for sigma in sigma_values:
        for order_bonus in order_bonus_values:
            for resource_sd in resource_sd_values:
                for choice_beta in beta_values:
                    for lapse in lapse_values:
                        for seed in seed_values:
                            config = SamplerConfig(**asdict(base_config))
                            config.sigma_distance = sigma
                            config.order_bonus = order_bonus
                            config.resource_log_sd = resource_sd
                            config.choice_beta = choice_beta
                            config.lapse = lapse
                            config.seed = seed
                            result = simulate(config, pre, fit_beta=False)
                            summary = result["summary"]
                            row = {
                                "score": compact_score(summary),
                                "seed": seed,
                                "sigma_distance": sigma,
                                "order_bonus": order_bonus,
                                "resource_log_sd": resource_sd,
                                "choice_beta": choice_beta,
                                "lapse": lapse,
                                "overall_accuracy": summary["overall_accuracy"],
                                "learned_pairs_accuracy": summary["learned_pairs_accuracy"],
                                "nonlearned_pairs_accuracy": summary["nonlearned_pairs_accuracy"],
                                "consistent80_ratio": summary["consistent_error_subjects_80pct_ratio"],
                                "consistent100_ratio": summary["consistent_error_subjects_100pct_ratio"],
                                "choice_self_consistency": summary["mean_self_consistency_from_majority_choices"],
                                "choice_circular_triads": summary["mean_circular_triads_from_majority_choices"],
                                "inter_subject_tau": summary["mean_inter_subject_kendall_tau"],
                                "correct_ranking_subjects": summary["correct_ranking_subjects"],
                                "self_consistent_incorrect_subjects": summary["self_consistent_incorrect_subjects"],
                                "beta_bimodal_pairs": summary["beta_pair_category_counts"].get("bimodal", 0),
                                "beta_high_accuracy_pairs": summary["beta_pair_category_counts"].get("high_accuracy", 0),
                            }
                            rows.append(row)
    rows.sort(key=lambda r: r["score"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "active_sampler_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "active_sampler_sweep_top10.json", "w", encoding="utf-8") as f:
        json.dump(rows[:10], f, indent=2, ensure_ascii=False)
    return rows


def make_report(output_dir: Path, summary: dict, sweep_rows: list[dict] | None = None) -> None:
    cfg = summary["config"]
    beta_counts = summary["beta_pair_category_counts"]
    dist = summary["distance_accuracy"]
    lines = [
        "# Active rank-hypothesis attractor sampler report",
        "",
        "## 核心机制",
        "",
        "该模型把学习后的内部状态从连续分数平均值改成一个离散的全局排序吸引子。每个虚拟被试先以个体化可靠性编码 8 条局部边，随后在 8! 个可能全局排序中进行主动重激活/重放式后验采样，并在无反馈测试中从这个已承诺的主观排序读出选择。",
        "",
        "关键结构差异是：模型不再把不确定性保留为每个 item 的独立噪声，而是在测试前 collapse 到一个整体 ranking hypothesis。因此，同一个被试的错误天然是稳定且传递自洽的；不同被试因为可靠性和资源不同，会落入不同的吸引子。",
        "",
        "## 单次运行结果",
        "",
        f"- seed = {cfg['seed']}; virtual subjects = {summary['eval_subjects']}; repetitions = {summary['eval_repetitions']}",
        f"- overall accuracy = {summary['overall_accuracy']:.3f}",
        f"- learned / non-learned accuracy = {summary['learned_pairs_accuracy']:.3f} / {summary['nonlearned_pairs_accuracy']:.3f}",
        f"- subjects with >=80% stable error on at least one pair = {summary['consistent_error_subjects_80pct']} / {summary['eval_subjects']} ({summary['consistent_error_subjects_80pct_ratio']:.3f})",
        f"- subjects with 100% stable error on at least one pair = {summary['consistent_error_subjects_100pct']} / {summary['eval_subjects']} ({summary['consistent_error_subjects_100pct_ratio']:.3f})",
        f"- majority-choice self-consistency = {summary['mean_self_consistency_from_majority_choices']:.3f}; circular triads = {summary['mean_circular_triads_from_majority_choices']:.3f}",
        f"- inter-subject Kendall tau = {summary['mean_inter_subject_kendall_tau']:.3f}",
        f"- correct / self-consistent incorrect / self-inconsistent subjects = {summary['correct_ranking_subjects']} / {summary['self_consistent_incorrect_subjects']} / {summary['self_inconsistent_subjects']}",
        f"- beta pair categories = {beta_counts}",
        f"- distance accuracy = {', '.join([k + ':' + format(v, '.3f') for k, v in dist.items()])}",
        "",
        "## 推荐参数",
        "",
        f"```text\nsigma_distance={cfg['sigma_distance']}\norder_bonus={cfg['order_bonus']}\nresource_log_sd={cfg['resource_log_sd']}\nchoice_beta={cfg['choice_beta']}\nlapse={cfg['lapse']}\nedge_weight_concentration={cfg['edge_weight_concentration']}\ndistance_salience={cfg['distance_salience']}\n```",
    ]
    if sweep_rows:
        lines.extend(["", "## Sweep top 5", ""])
        for idx, row in enumerate(sweep_rows[:5], 1):
            lines.append(
                f"{idx}. score={row['score']:.3f}, sigma={row['sigma_distance']}, order_bonus={row['order_bonus']}, "
                f"resource_sd={row['resource_log_sd']}, beta={row['choice_beta']}, lapse={row['lapse']}, "
                f"overall={row['overall_accuracy']:.3f}, learned={row['learned_pairs_accuracy']:.3f}, "
                f"nonlearned={row['nonlearned_pairs_accuracy']:.3f}, c80={row['consistent80_ratio']:.3f}, "
                f"c100={row['consistent100_ratio']:.3f}, self={row['choice_self_consistency']:.3f}, "
                f"tau={row['inter_subject_tau']:.3f}, bimodal_pairs={row['beta_bimodal_pairs']}"
            )
    (output_dir / "ACTIVE_SAMPLER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active rank-hypothesis attractor sampler for the Liu/Wang/Luo task.")
    parser.add_argument("--output-dir", default="outputs_active_sampler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-subjects", type=int, default=77)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--sigma-distance", type=float, default=2.0)
    parser.add_argument("--order-bonus", type=float, default=1.5)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--edge-weight-concentration", type=float, default=8.0)
    parser.add_argument("--distance-salience", type=float, default=1.2)
    parser.add_argument("--anchor-bias", type=float, default=0.0)
    parser.add_argument("--encoding-dropout", type=float, default=0.0)
    parser.add_argument("--resource-log-mean", type=float, default=0.0)
    parser.add_argument("--resource-log-sd", type=float, default=0.6)
    parser.add_argument("--choice-beta", type=float, default=2.2)
    parser.add_argument("--lapse", type=float, default=0.03)
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SamplerConfig(
        seed=args.seed,
        eval_subjects=args.eval_subjects,
        eval_repetitions=args.eval_repetitions,
        sigma_distance=args.sigma_distance,
        order_bonus=args.order_bonus,
        posterior_temperature=args.posterior_temperature,
        edge_weight_concentration=args.edge_weight_concentration,
        distance_salience=args.distance_salience,
        anchor_bias=args.anchor_bias,
        encoding_dropout=args.encoding_dropout,
        resource_log_mean=args.resource_log_mean,
        resource_log_sd=args.resource_log_sd,
        choice_beta=args.choice_beta,
        lapse=args.lapse,
        output_dir=args.output_dir,
    )
    output_dir = Path(config.output_dir)
    sweep_rows = None
    if args.sweep:
        print("[sweep] running compact sweep...")
        sweep_rows = run_sweep(config, output_dir)
        print(f"[sweep] best score = {sweep_rows[0]['score']:.4f}; wrote active_sampler_sweep_results.csv")
    pre = precompute_hypotheses(config)
    result = simulate(config, pre)
    write_outputs(output_dir, config, result)
    make_report(output_dir, result["summary"], sweep_rows=sweep_rows)
    print("[eval] active rank-hypothesis sampler summary:")
    for key, value in result["summary"].items():
        if key == "config":
            continue
        print(f"  {key}: {value}")
    print(f"[done] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
