"""
Constructive-ranking baseline for the few-shot relational-learning task.

This script replaces the original adjacent-pair reinforcement-learning episode in
``simple_neo.py`` with a deliberately constrained neural mechanism:

1. Each item has an internal scalar score stored in an episode-local memory.
2. A learning observation (item_i, item_j, signed distance) is treated as a
   constraint on the two scores, not as a reward.
3. A small trainable update network learns how strongly to write each constraint
   into the score memory.
4. Test choices are forced to use one global score vector: choose by comparing
   score(item_i) and score(item_j). This makes pairwise choices come from a
   shared latent ranking rather than from independent pair classifiers.

The meta-training distribution is intentionally broader than the behavioral
paper's exact task. During training, item identities, rank-to-item mappings and
sparse learning graphs are randomized. The behavioral paper's fixed 8-pair task
is used in ``evaluate_paper_task`` as a held-out-style evaluation protocol.

There is no within-episode reward. The only optimization signal is the outer-loop
loss computed after running a full episode, matching an observation-only learning
phase followed by no-feedback testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rank-space pairs from Liu/Wang/Luo behavioral task.
# We use ranks 0..7 from low to high, so (0, 5) means A-F and the signed
# distance for the oriented pair (A, F) is +5/7.
PAPER_LEARNING_PAIRS_RANK = [
    (0, 5),  # A-F
    (1, 2),  # B-C
    (1, 4),  # B-E
    (2, 6),  # C-G
    (3, 5),  # D-F
    (3, 6),  # D-G
    (4, 7),  # E-H
    (0, 7),  # A-H
]


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class TrainConfig:
    # Small defaults are intentional: they make the training script quick to run
    # and suitable for smoke tests. Increase nbiter/bs/hidden_size for real runs.
    rngseed: int = 1
    nbiter: int = 4
    bs: int = 2
    hidden_size: int = 8
    item_dim: int = 15
    subject_dim: int = 8
    n_items: int = 8
    n_learning_pairs: int = 8
    n_learning_blocks: int = 4
    relation_noise: float = 0.03
    eval_relation_noise: float = 0.0
    edge_dropout: float = 0.0
    eval_edge_dropout: float = 0.0
    init_score_noise: float = 0.0
    subject_scale: float = 1.0
    update_scale: float = 1.0
    eval_beta_override: float = 0.0
    lr: float = 3e-4
    eps: float = 1e-8
    l2: float = 0.0
    gc: float = 2.0
    lambda_recon: float = 0.5
    lambda_entropy: float = 0.005
    lambda_score_l2: float = 1e-4
    save_every: int = 50
    pe: int = 10
    output_dir: str = "outputs_constructive"
    eval_subjects: int = 8
    eval_repetitions: int = 5

    @property
    def all_rank_pairs(self) -> list[tuple[int, int]]:
        return list(combinations(range(self.n_items), 2))

    @property
    def rank_values(self) -> torch.Tensor:
        # Centering removes an arbitrary global offset from the score coordinate.
        values = torch.arange(self.n_items, dtype=torch.float32)
        values = (values - values.mean()) / (self.n_items - 1)
        return values


class ConstructiveRankingNet(nn.Module):
    """Trainable score-memory updater for sparse pairwise ranking evidence."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        z = config.subject_dim
        d = config.item_dim
        h = config.hidden_size

        # Initial item score = function(item identity, subject latent state).
        self.score_init = nn.Sequential(
            nn.Linear(d + z, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

        # Constraint writing gate. It does not output the answer; it only decides
        # how strongly the current relation-error should update the two item scores.
        self.update_gate = nn.Sequential(
            nn.Linear(2 * d + z + 4, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

        # Global trainable learning-rate and choice inverse-temperature.
        self.logit_step = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

    def initial_scores(self, item_vecs: torch.Tensor, subject_z: torch.Tensor) -> torch.Tensor:
        """Return episode-local initial scores, shape [B, N]."""
        bsz, n_items, _ = item_vecs.shape
        z_scaled = self.config.subject_scale * subject_z
        z_rep = z_scaled[:, None, :].expand(bsz, n_items, subject_z.shape[-1])
        raw = self.score_init(torch.cat([item_vecs, z_rep], dim=-1)).squeeze(-1)
        if self.config.init_score_noise > 0:
            raw = raw + self.config.init_score_noise * torch.randn_like(raw)
        return self.center_scores(raw)

    @staticmethod
    def center_scores(scores: torch.Tensor) -> torch.Tensor:
        return scores - scores.mean(dim=1, keepdim=True)

    def update_scores(
        self,
        scores: torch.Tensor,
        item_vecs: torch.Tensor,
        subject_z: torch.Tensor,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        observed_diff_j_minus_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one relation-error-driven update.

        The observation says score(j) - score(i) should equal observed_diff.
        The model learns a gate, but the sign and conservation structure of the
        update are fixed so that learning has to operate on a shared score axis.
        """
        bsz = scores.shape[0]
        batch_idx = torch.arange(bsz, device=scores.device)

        score_i = scores[batch_idx, item_i]
        score_j = scores[batch_idx, item_j]
        pred_diff = score_j - score_i
        error = observed_diff_j_minus_i - pred_diff

        vec_i = item_vecs[batch_idx, item_i]
        vec_j = item_vecs[batch_idx, item_j]
        features = torch.cat(
            [
                vec_i,
                vec_j,
                self.config.subject_scale * subject_z,
                observed_diff_j_minus_i[:, None],
                observed_diff_j_minus_i.abs()[:, None],
                pred_diff[:, None],
                error[:, None],
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.update_gate(features)).squeeze(1)
        step = self.config.update_scale * torch.sigmoid(self.logit_step) * gate
        step = step.clamp(0.0, 1.0)
        delta = 0.5 * step * error

        next_scores = scores.clone()
        next_scores[batch_idx, item_i] = next_scores[batch_idx, item_i] - delta
        next_scores[batch_idx, item_j] = next_scores[batch_idx, item_j] + delta
        next_scores = self.center_scores(next_scores)
        return next_scores, pred_diff, error, gate

    def choice_logits(self, scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
        """Two-class logits: class 0 means item_i higher; class 1 means item_j higher."""
        batch_idx = torch.arange(scores.shape[0], device=scores.device)
        beta = F.softplus(self.log_beta) + 1e-3
        return beta * torch.stack([scores[batch_idx, item_i], scores[batch_idx, item_j]], dim=1)


def set_seed(seed: int) -> None:
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_item_vectors(batch_size: int, n_items: int, item_dim: int) -> torch.Tensor:
    # Random identity-like binary codes. They do not reveal rank.
    return (torch.randint(0, 2, (batch_size, n_items, item_dim), device=DEVICE).float() * 2.0) - 1.0


def sample_subject_latents(batch_size: int, subject_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, subject_dim, device=DEVICE)


def rank_to_item_maps(batch_size: int, n_items: int) -> torch.Tensor:
    maps = [torch.randperm(n_items, device=DEVICE) for _ in range(batch_size)]
    return torch.stack(maps, dim=0)


def true_scores_by_item(rank_to_item: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    values = config.rank_values.to(DEVICE)
    bsz = rank_to_item.shape[0]
    scores = torch.empty(bsz, config.n_items, device=DEVICE)
    for rank in range(config.n_items):
        scores[:, rank_to_item[:, rank]] = values[rank]
    return scores


def _is_connected(edges: list[tuple[int, int]], n_items: int) -> bool:
    neighbors = {i: set() for i in range(n_items)}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nxt in neighbors[node]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return len(seen) == n_items


def sample_sparse_rank_graph(config: TrainConfig) -> list[tuple[int, int]]:
    """Sample a connected sparse graph in rank space."""
    all_pairs = config.all_rank_pairs
    for _ in range(1000):
        idx = np.random.choice(len(all_pairs), size=config.n_learning_pairs, replace=False)
        edges = [all_pairs[int(i)] for i in idx]
        degrees = np.zeros(config.n_items, dtype=int)
        for a, b in edges:
            degrees[a] += 1
            degrees[b] += 1
        if np.all(degrees > 0) and _is_connected(edges, config.n_items):
            return edges
    # Fallback: a chain plus one random edge.
    edges = [(i, i + 1) for i in range(config.n_items - 1)]
    remaining = [p for p in all_pairs if p not in edges]
    edges.append(random.choice(remaining))
    return edges[: config.n_learning_pairs]


def rank_edges_to_item_edges(
    rank_edges: list[tuple[int, int]],
    rank_to_item: torch.Tensor,
    random_orientation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rank-space edges to item-index edges for each batch element."""
    bsz = rank_to_item.shape[0]
    n_edges = len(rank_edges)
    item_edges = torch.empty(bsz, n_edges, 2, dtype=torch.long, device=DEVICE)
    rank_edges_tensor = torch.tensor(rank_edges, dtype=torch.long, device=DEVICE)
    for b in range(bsz):
        item_edges[b, :, 0] = rank_to_item[b, rank_edges_tensor[:, 0]]
        item_edges[b, :, 1] = rank_to_item[b, rank_edges_tensor[:, 1]]
    if random_orientation:
        swap = torch.rand(bsz, n_edges, device=DEVICE) < 0.5
        first = item_edges[:, :, 0].clone()
        second = item_edges[:, :, 1].clone()
        item_edges[:, :, 0] = torch.where(swap, second, first)
        item_edges[:, :, 1] = torch.where(swap, first, second)
    return item_edges, rank_edges_tensor


def all_item_pairs(rank_to_item: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    rank_pairs = torch.tensor(config.all_rank_pairs, dtype=torch.long, device=DEVICE)
    bsz = rank_to_item.shape[0]
    pairs = torch.empty(bsz, len(config.all_rank_pairs), 2, dtype=torch.long, device=DEVICE)
    for b in range(bsz):
        pairs[b, :, 0] = rank_to_item[b, rank_pairs[:, 0]]
        pairs[b, :, 1] = rank_to_item[b, rank_pairs[:, 1]]
    return pairs


def gather_true_diff(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(true_scores.shape[0], device=true_scores.device)
    return true_scores[batch_idx, item_j] - true_scores[batch_idx, item_i]


def run_learning_phase(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    true_scores: torch.Tensor,
    learning_item_edges: torch.Tensor,
    train_mode: bool,
    noise_std: float | None = None,
    edge_dropout: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run observation-only sparse-pair learning.

    Returns final scores, accumulated reconstruction loss and mean update gate.
    """
    bsz, n_edges, _ = learning_item_edges.shape
    scores = net.initial_scores(item_vecs, subject_z)
    if noise_std is None:
        noise_std = config.relation_noise if train_mode else config.eval_relation_noise
    if edge_dropout is None:
        edge_dropout = config.edge_dropout if train_mode else config.eval_edge_dropout
    recon_losses = []
    gates = []

    for _block in range(config.n_learning_blocks):
        orders = torch.stack([torch.randperm(n_edges, device=DEVICE) for _ in range(bsz)], dim=0)
        for k in range(n_edges):
            edge_idx = orders[:, k]
            item_i = learning_item_edges[torch.arange(bsz, device=DEVICE), edge_idx, 0]
            item_j = learning_item_edges[torch.arange(bsz, device=DEVICE), edge_idx, 1]
            true_diff = gather_true_diff(true_scores, item_i, item_j)
            if noise_std > 0:
                obs_diff = true_diff + noise_std * torch.randn_like(true_diff)
                obs_diff = obs_diff.clamp(-1.0, 1.0)
            else:
                obs_diff = true_diff
            if edge_dropout > 0:
                keep = (torch.rand_like(obs_diff) >= edge_dropout).float()
                # If internally unattended, replace this observation by the current prediction, yielding no score update.
                pred_current = scores[torch.arange(bsz, device=DEVICE), item_j] - scores[torch.arange(bsz, device=DEVICE), item_i]
                obs_diff = keep * obs_diff + (1.0 - keep) * pred_current.detach()
            scores, pred_diff, error, gate = net.update_scores(scores, item_vecs, subject_z, item_i, item_j, obs_diff)
            recon_losses.append(error.pow(2).mean())
            gates.append(gate.mean())

    if recon_losses:
        recon_loss = torch.stack(recon_losses).mean()
        mean_gate = torch.stack(gates).mean()
    else:
        recon_loss = torch.tensor(0.0, device=DEVICE)
        mean_gate = torch.tensor(0.0, device=DEVICE)
    return scores, recon_loss, mean_gate


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    choice_loss: float
    recon_loss: float
    entropy_loss: float
    test_acc: float
    learned_acc: float
    nonlearned_acc: float
    mean_gate: float
    mean_abs_score: float


def run_training_episode(config: TrainConfig, net: ConstructiveRankingNet) -> EpisodeStats:
    bsz = config.bs
    item_vecs = sample_item_vectors(bsz, config.n_items, config.item_dim)
    subject_z = sample_subject_latents(bsz, config.subject_dim)
    r2i = rank_to_item_maps(bsz, config.n_items)
    true_scores = true_scores_by_item(r2i, config)

    # A broader task family: random connected sparse evidence graphs.
    rank_edges_per_batch = [sample_sparse_rank_graph(config) for _ in range(bsz)]
    learning_item_edges = torch.empty(bsz, config.n_learning_pairs, 2, dtype=torch.long, device=DEVICE)
    for b, rank_edges in enumerate(rank_edges_per_batch):
        edge_items, _ = rank_edges_to_item_edges(rank_edges, r2i[b : b + 1], random_orientation=True)
        learning_item_edges[b] = edge_items[0]

    scores, recon_loss, mean_gate = run_learning_phase(
        config, net, item_vecs, subject_z, true_scores, learning_item_edges, train_mode=True
    )

    # Outer-loop query loss over all pairwise comparisons. No feedback is fed
    # back into the episode; this is only the meta-training objective.
    pairs = all_item_pairs(r2i, config)
    choice_losses = []
    entropies = []
    correct = []
    learned_correct = []
    nonlearned_correct = []

    learned_sets = [set(tuple(sorted(map(int, edge.tolist()))) for edge in learning_item_edges[b]) for b in range(bsz)]

    for k in range(pairs.shape[1]):
        item_i = pairs[:, k, 0]
        item_j = pairs[:, k, 1]
        logits = net.choice_logits(scores, item_i, item_j)
        true_i = true_scores[torch.arange(bsz, device=DEVICE), item_i]
        true_j = true_scores[torch.arange(bsz, device=DEVICE), item_j]
        labels = (true_j > true_i).long()  # 0 if item_i higher, 1 if item_j higher.
        choice_losses.append(F.cross_entropy(logits, labels))
        probs = F.softmax(logits, dim=1)
        entropies.append((-(probs * (probs + 1e-8).log()).sum(dim=1)).mean())
        pred = torch.argmax(logits, dim=1)
        corr = (pred == labels).float()
        correct.append(corr)
        for b in range(bsz):
            pair_key = tuple(sorted([int(item_i[b]), int(item_j[b])]))
            if pair_key in learned_sets[b]:
                learned_correct.append(corr[b])
            else:
                nonlearned_correct.append(corr[b])

    choice_loss = torch.stack(choice_losses).mean()
    entropy_loss = torch.stack(entropies).mean()
    correct_tensor = torch.stack(correct, dim=1)
    learned_acc = torch.stack(learned_correct).mean() if learned_correct else torch.tensor(float("nan"), device=DEVICE)
    nonlearned_acc = torch.stack(nonlearned_correct).mean() if nonlearned_correct else torch.tensor(float("nan"), device=DEVICE)

    loss = (
        choice_loss
        + config.lambda_recon * recon_loss
        + config.lambda_entropy * entropy_loss
        + config.lambda_score_l2 * scores.pow(2).mean()
    )

    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        choice_loss=float(choice_loss.detach()),
        recon_loss=float(recon_loss.detach()),
        entropy_loss=float(entropy_loss.detach()),
        test_acc=float(correct_tensor.mean().detach()),
        learned_acc=float(learned_acc.detach()),
        nonlearned_acc=float(nonlearned_acc.detach()),
        mean_gate=float(mean_gate.detach()),
        mean_abs_score=float(scores.abs().mean().detach()),
    )


def train(config: TrainConfig) -> ConstructiveRankingNet:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = ConstructiveRankingNet(config).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2)
    log(f"[setup] Device: {DEVICE}")
    log(f"[setup] Batch size: {config.bs}; episodes: {config.nbiter}; output: {output_dir}")
    log(f"[setup] Parameter count: {sum(p.numel() for p in net.parameters())}")
    log("[task] Inner-loop reward: none (reward_input = 0). Learning trials provide relation constraints only.")

    log_path = output_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "loss",
                "choice_loss",
                "recon_loss",
                "entropy_loss",
                "test_acc",
                "learned_acc",
                "nonlearned_acc",
                "mean_gate",
                "mean_abs_score",
            ],
        )
        writer.writeheader()

    start = time.time()
    episode_iter = tqdm(range(config.nbiter), desc="training episodes", unit="episode", dynamic_ncols=True, file=sys.stdout)
    last_stats: EpisodeStats | None = None
    for episode in episode_iter:
        optimizer.zero_grad()
        stats = run_training_episode(config, net)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        optimizer.step()
        last_stats = stats

        if episode % config.pe == 0 or episode == config.nbiter - 1:
            elapsed = time.time() - start
            log(
                f"Episode {episode} ==== {elapsed:.2f}s | "
                f"loss={stats.loss_value:.4f} choice={stats.choice_loss:.4f} recon={stats.recon_loss:.4f} | "
                f"acc={stats.test_acc:.3f} learned={stats.learned_acc:.3f} nonlearned={stats.nonlearned_acc:.3f} | "
                f"gate={stats.mean_gate:.3f} | |score|={stats.mean_abs_score:.3f}"
            )
            start = time.time()

        with open(log_path, "a", newline="") as f:
            # Write manually because stats.loss is a tensor and not CSV friendly.
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    stats.loss_value,
                    stats.choice_loss,
                    stats.recon_loss,
                    stats.entropy_loss,
                    stats.test_acc,
                    stats.learned_acc,
                    stats.nonlearned_acc,
                    stats.mean_gate,
                    stats.mean_abs_score,
                ]
            )

        if config.save_every > 0 and episode > 0 and episode % config.save_every == 0:
            torch.save({"model_state": net.state_dict(), "config": asdict(config)}, output_dir / "net_constructive.pt")
            log(f"[save] checkpoint: {output_dir / 'net_constructive.pt'}")

    torch.save({"model_state": net.state_dict(), "config": asdict(config)}, output_dir / "net_constructive.pt")
    log(f"[save] final checkpoint: {output_dir / 'net_constructive.pt'}")
    if last_stats is not None:
        log(f"[done] final train acc={last_stats.test_acc:.3f}, loss={last_stats.loss_value:.4f}")
    return net


def majority_preference_matrix(pair_correctness: np.ndarray, pair_choices_high_rank: np.ndarray, n_items: int) -> np.ndarray:
    """Build a directed preference matrix pref[i,j]=1 if i is judged higher than j."""
    pref = np.zeros((n_items, n_items), dtype=int)
    for a in range(n_items):
        for b in range(a + 1, n_items):
            # Stored index follows combinations order.
            pass
    return pref


def self_consistency_from_scores(scores: np.ndarray) -> tuple[float, int]:
    """Score-derived rankings are transitive; this returns 1.0 unless ties are exact."""
    n_items = scores.shape[0]
    circular = 0
    for a, b, c in combinations(range(n_items), 3):
        # Majority preferences induced by scalar scores.
        ab = scores[a] > scores[b]
        bc = scores[b] > scores[c]
        ac = scores[a] > scores[c]
        # Cycles cannot occur for strict scalar scores. Ties are counted as non-cycles.
        if (ab and bc and not ac) or ((not ab) and (not bc) and ac):
            circular += 1
    max_triads = (n_items**3 - 4 * n_items) // 24 if n_items % 2 == 0 else (n_items**3 - n_items) // 24
    return 1.0 - circular / max_triads, circular




def self_consistency_from_majority(pair_acc: np.ndarray, pair_list: list[tuple[int, int]], n_items: int) -> tuple[float, int]:
    """Compute circular-triad consistency from majority choices.

    pair_acc is accuracy for rank-ordered pairs (a,b) where b is ground-truth higher.
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
        # Count a cycle if all three directed preferences form either orientation.
        if (pref[a, b] and pref[b, c] and pref[c, a]) or (pref[a, c] and pref[c, b] and pref[b, a]):
            circular += 1
    max_triads = (n_items**3 - 4 * n_items) // 24 if n_items % 2 == 0 else (n_items**3 - n_items) // 24
    return 1.0 - circular / max_triads, circular

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


@torch.no_grad()
def evaluate_paper_task(config: TrainConfig, net: ConstructiveRankingNet) -> dict:
    """Evaluate the behavioral paper's exact few-shot graph with virtual subjects."""
    net.eval()
    old_log_beta = None
    if config.eval_beta_override and config.eval_beta_override > 0:
        old_log_beta = net.log_beta.detach().clone()
        net.log_beta.data.fill_(math.log(config.eval_beta_override))
    n_subj = config.eval_subjects
    reps = config.eval_repetitions
    # Same item identities for all virtual subjects, matching identical external input.
    one_item_set = sample_item_vectors(1, config.n_items, config.item_dim)
    item_vecs = one_item_set.expand(n_subj, -1, -1).contiguous()
    subject_z = sample_subject_latents(n_subj, config.subject_dim)
    r2i = torch.arange(config.n_items, device=DEVICE)[None, :].expand(n_subj, -1).contiguous()
    true_scores = true_scores_by_item(r2i, config)
    learning_edges, _ = rank_edges_to_item_edges(PAPER_LEARNING_PAIRS_RANK, r2i, random_orientation=False)
    scores, recon_loss, mean_gate = run_learning_phase(config, net, item_vecs, subject_z, true_scores, learning_edges, train_mode=False)

    pair_list = config.all_rank_pairs
    paper_pair_set = {tuple(p) for p in PAPER_LEARNING_PAIRS_RANK}
    pair_acc = np.zeros((n_subj, len(pair_list)), dtype=float)
    pair_error_prop = np.zeros_like(pair_acc)

    beta_logits_all = []
    for pidx, (ra, rb) in enumerate(pair_list):
        item_i = torch.full((n_subj,), ra, dtype=torch.long, device=DEVICE)
        item_j = torch.full((n_subj,), rb, dtype=torch.long, device=DEVICE)
        labels = torch.ones(n_subj, dtype=torch.long, device=DEVICE)  # rb is higher than ra.
        logits = net.choice_logits(scores, item_i, item_j)
        beta_logits_all.append(float((logits[:, 1] - logits[:, 0]).mean().cpu()))
        probs = F.softmax(logits, dim=1)
        # Repeated no-feedback test choices.
        choices = []
        for _ in range(reps):
            sampled = torch.distributions.Categorical(probs).sample()
            choices.append((sampled == labels).float().cpu().numpy())
        arr = np.stack(choices, axis=1)
        pair_acc[:, pidx] = arr.mean(axis=1)
        pair_error_prop[:, pidx] = 1.0 - pair_acc[:, pidx]

    learned_idx = np.array([idx for idx, p in enumerate(pair_list) if p in paper_pair_set], dtype=int)
    nonlearned_idx = np.array([idx for idx, p in enumerate(pair_list) if p not in paper_pair_set], dtype=int)
    learned_acc = float(pair_acc[:, learned_idx].mean())
    nonlearned_acc = float(pair_acc[:, nonlearned_idx].mean())
    overall_acc = float(pair_acc.mean())

    consistent_subjects_80 = int((pair_error_prop >= 0.8).any(axis=1).sum())
    consistent_subjects_100 = int((pair_error_prop >= 1.0).any(axis=1).sum())

    subject_orders = []
    self_consistency = []
    circular_counts = []
    choice_self_consistency = []
    choice_circular_counts = []
    for b in range(n_subj):
        score_np = scores[b].detach().cpu().numpy()
        # Low-to-high subjective order.
        order = list(np.argsort(score_np))
        subject_orders.append(order)
        sc, circ = self_consistency_from_scores(score_np)
        self_consistency.append(sc)
        circular_counts.append(circ)
        csc, ccirc = self_consistency_from_majority(pair_acc[b], pair_list, config.n_items)
        choice_self_consistency.append(csc)
        choice_circular_counts.append(ccirc)

    taus = []
    for i, j in combinations(range(n_subj), 2):
        taus.append(kendall_tau_order(subject_orders[i], subject_orders[j]))

    # Distance effect proxy: mean accuracy by true rank distance.
    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())

    summary = {
        "eval_subjects": n_subj,
        "eval_repetitions": reps,
        "overall_accuracy": overall_acc,
        "learned_pairs_accuracy": learned_acc,
        "nonlearned_pairs_accuracy": nonlearned_acc,
        "consistent_error_subjects_80pct": consistent_subjects_80,
        "consistent_error_subjects_80pct_ratio": consistent_subjects_80 / n_subj,
        "consistent_error_subjects_100pct": consistent_subjects_100,
        "consistent_error_subjects_100pct_ratio": consistent_subjects_100 / n_subj,
        "mean_self_consistency_from_scores": float(np.mean(self_consistency)),
        "mean_circular_triads_from_scores": float(np.mean(circular_counts)),
        "mean_self_consistency_from_majority_choices": float(np.mean(choice_self_consistency)),
        "mean_circular_triads_from_majority_choices": float(np.mean(choice_circular_counts)),
        "mean_inter_subject_kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "distance_accuracy": distance_acc,
        "reconstruction_loss_after_learning": float(recon_loss.cpu()),
        "mean_update_gate": float(mean_gate.cpu()),
        "mean_abs_final_score": float(scores.abs().mean().cpu()),
    }
    if old_log_beta is not None:
        net.log_beta.data.copy_(old_log_beta)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constructive ranking baseline for few-shot relational learning.")
    parser.add_argument("--nbiter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs_constructive")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--eval-subjects", type=int, default=8)
    parser.add_argument("--eval-repetitions", type=int, default=5)
    parser.add_argument("--relation-noise", type=float, default=0.03)
    parser.add_argument("--eval-relation-noise", type=float, default=0.0)
    parser.add_argument("--edge-dropout", type=float, default=0.0)
    parser.add_argument("--eval-edge-dropout", type=float, default=0.0)
    parser.add_argument("--init-score-noise", type=float, default=0.0)
    parser.add_argument("--subject-scale", type=float, default=1.0)
    parser.add_argument("--update-scale", type=float, default=1.0)
    parser.add_argument("--eval-beta-override", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        rngseed=args.seed,
        nbiter=args.nbiter,
        bs=args.batch_size,
        hidden_size=args.hidden_size,
        output_dir=args.output_dir,
        save_every=args.save_every,
        pe=args.print_every,
        eval_subjects=args.eval_subjects,
        eval_repetitions=args.eval_repetitions,
        relation_noise=args.relation_noise,
        eval_relation_noise=args.eval_relation_noise,
        edge_dropout=args.edge_dropout,
        eval_edge_dropout=args.eval_edge_dropout,
        init_score_noise=args.init_score_noise,
        subject_scale=args.subject_scale,
        update_scale=args.update_scale,
        eval_beta_override=args.eval_beta_override,
        lr=args.lr,
    )
    set_seed(config.rngseed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_constructive.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    net = train(config)
    summary = evaluate_paper_task(config, net)
    with open(output_dir / "paper_task_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("[eval] behavioral-paper task summary:")
    for key, value in summary.items():
        log(f"  {key}: {value}")


if __name__ == "__main__":
    main()
