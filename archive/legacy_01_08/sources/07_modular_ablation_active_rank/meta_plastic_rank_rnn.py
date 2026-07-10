"""
Meta-trained plastic RNN with a differentiable active-rank posterior module.

This script is intentionally separate from ``simple_neo.py`` and the earlier
``active_rank_hypothesis_sampler.py``.  It keeps the cognitive idea that the
learner commits to a global ranking hypothesis, but makes the hypothesis stage a
fully differentiable neural module trained end-to-end through an observation-only
few-shot episode.

Architecture
------------
1. A plastic RNN controller processes the learning sequence.  It has slow
   recurrent weights plus episode-local Hebbian fast weights.
2. The controller writes local edge memories: an encoded relation and a learned
   reliability for each observed pair.
3. A differentiable active-rank module evaluates all 8! possible global rankings
   against the remembered edges.  During meta-training, losses are computed from
   the soft posterior over rankings; no test feedback is fed into the episode.
4. At behavioral evaluation, each virtual subject samples/commits to one ranking
   attractor and then performs repeated no-feedback choices from that ranking.

The implementation uses randomized meta-training episodes and evaluates on the
exact 8 fixed learning pairs used elsewhere in the package.
"""

from __future__ import annotations

import argparse
import csv
import os
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
# The 8! posterior uses many medium-sized tensor ops.  On CPU, OpenMP
# oversubscription makes it dramatically slower, so keep the default single-threaded.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from scipy.stats import beta as beta_dist
except Exception:  # pragma: no cover
    beta_dist = None

ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rank-space pairs from the behavioral task.  Ranks are low-to-high: (0,5)
# means the second item should be judged higher than the first.
PAPER_LEARNING_PAIRS_RANK: list[tuple[int, int]] = [
    (0, 5),
    (1, 2),
    (1, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (0, 7),
]


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class MetaPlasticConfig:
    seed: int = 1301
    nbiter: int = 300
    batch_size: int = 12
    hidden_size: int = 32
    item_dim: int = 16
    subject_dim: int = 10
    n_items: int = 8
    n_learning_pairs: int = 8
    n_learning_blocks: int = 4
    relation_noise: float = 0.10
    eval_relation_noise: float = 0.00
    train_edge_dropout: float = 0.03
    eval_edge_dropout: float = 0.00
    lr: float = 4e-4
    weight_decay: float = 1e-5
    grad_clip: float = 2.0
    print_every: int = 25
    save_every: int = 200
    output_dir: str = "outputs_meta_plastic_rank_rnn"

    # Plastic RNN fast weights.
    plastic_eta: float = 0.075
    plastic_decay: float = 0.92
    plastic_norm: float = 1.0
    fast_weight_gain: float = 0.75

    # Differentiable active-rank posterior.  These are initialized near the
    # earlier non-neural sampler, but are trainable/bounded in the module.
    init_sigma_distance: float = 2.0
    init_order_bonus: float = 1.45
    init_global_precision: float = 1.0
    posterior_temperature: float = 1.0
    posterior_entropy_weight: float = 0.005
    strength_l1_weight: float = 0.002
    edge_recon_weight: float = 0.03
    sigma_prior_weight: float = 0.01
    precision_prior_weight: float = 0.003

    # Human-like evaluation: sample one attractor per subject, then choices are
    # repeated from that attractor.  The resource noise creates participant-level
    # posterior precision differences without changing the external input.
    eval_subjects: int = 77
    eval_repetitions: int = 10
    eval_choice_beta: float = 2.2
    eval_lapse: float = 0.03
    eval_resource_log_sd: float = 0.55
    eval_resource_log_mean: float = 0.0
    eval_commit_temperature: float = 1.0

    # Meta-training objective can include a small stochastic commitment loss to
    # train the network under attractor-style readout, while keeping the main
    # posterior loss differentiable.
    straight_through_commit_weight: float = 0.15
    train_commit_temperature: float = 0.75

    # Modular ablation / input switches.  Defaults implement the requested
    # RNN-derived reliability setting: no hand-coded distance salience enters
    # write strength.
    observation_mode: str = "distance"  # {distance, raw_bars}
    reliability_mode: str = "rnn"  # {rnn, feature_rnn, manual_distance, constant, oracle_distance}
    relation_encoding_mode: str = "rnn"  # {residual_observation, rnn}
    rank_readout: str = "commit"  # {commit, top1_commit, posterior_mean}
    ablate_rnn: bool = False
    ablate_plasticity: bool = False
    ablate_subject_latent: bool = False
    ablate_item_vectors: bool = False
    ablate_resource: bool = False
    manual_distance_salience: float = 0.9
    manual_subject_salience: float = 0.15
    constant_reliability: float = 0.5
    raw_bar_noise: float = 0.02

    @property
    def all_rank_pairs(self) -> list[tuple[int, int]]:
        return list(combinations(range(self.n_items), 2))

    @property
    def rank_values(self) -> torch.Tensor:
        values = torch.arange(self.n_items, dtype=torch.float32)
        values = values - values.mean()
        return values


@dataclass
class RankHypotheses:
    permutations: torch.Tensor  # [P,N] low-to-high item order
    positions: torch.Tensor  # [P,N], item -> position
    pos_diff: torch.Tensor  # [P,N,N], pos[j]-pos[i]
    order_ok: torch.Tensor  # [P,N,N], 1 if j is higher than i
    all_pairs: list[tuple[int, int]]
    learned_pair_indices: np.ndarray
    nonlearned_pair_indices: np.ndarray


def set_seed(seed: int) -> None:
    if seed < 0:
        log("[setup] No fixed seed.")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rank_hypotheses(config: MetaPlasticConfig, device: str = DEVICE) -> RankHypotheses:
    perms_np = np.array(list(itertools.permutations(range(config.n_items))), dtype=np.int64)
    positions_np = np.empty_like(perms_np)
    for idx, perm in enumerate(perms_np):
        positions_np[idx, perm] = np.arange(config.n_items, dtype=np.int64)
    permutations = torch.tensor(perms_np, dtype=torch.long, device=device)
    positions = torch.tensor(positions_np, dtype=torch.float32, device=device)
    pos_diff = positions[:, None, :] - positions[:, :, None]  # [P,i,j] = pos[j]-pos[i]
    order_ok = (pos_diff > 0).float()
    all_pairs = config.all_rank_pairs
    learned_pair_indices = np.array([all_pairs.index(p) for p in PAPER_LEARNING_PAIRS_RANK], dtype=np.int64)
    nonlearned_pair_indices = np.array([i for i, p in enumerate(all_pairs) if p not in set(PAPER_LEARNING_PAIRS_RANK)], dtype=np.int64)
    return RankHypotheses(permutations, positions, pos_diff, order_ok, all_pairs, learned_pair_indices, nonlearned_pair_indices)


def sample_item_vectors(batch_size: int, n_items: int, item_dim: int) -> torch.Tensor:
    # Random identity-like vectors; they do not encode rank.
    return (torch.randint(0, 2, (batch_size, n_items, item_dim), device=DEVICE).float() * 2.0) - 1.0


def sample_subject_latents(batch_size: int, subject_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, subject_dim, device=DEVICE)


def rank_to_item_maps(batch_size: int, n_items: int) -> torch.Tensor:
    return torch.stack([torch.randperm(n_items, device=DEVICE) for _ in range(batch_size)], dim=0)


def true_rank_position_by_item(rank_to_item: torch.Tensor, config: MetaPlasticConfig) -> torch.Tensor:
    bsz = rank_to_item.shape[0]
    pos = torch.empty(bsz, config.n_items, device=DEVICE)
    for rank in range(config.n_items):
        pos[:, rank_to_item[:, rank]] = float(rank)
    return pos


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


def sample_sparse_rank_graph(config: MetaPlasticConfig) -> list[tuple[int, int]]:
    all_pairs = config.all_rank_pairs
    for _ in range(1000):
        idx = np.random.choice(len(all_pairs), size=config.n_learning_pairs, replace=False)
        edges = [all_pairs[int(i)] for i in idx]
        degree = np.zeros(config.n_items, dtype=int)
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        if np.all(degree > 0) and _is_connected(edges, config.n_items):
            return edges
    chain = [(i, i + 1) for i in range(config.n_items - 1)]
    remaining = [p for p in all_pairs if p not in chain]
    return (chain + [random.choice(remaining)])[: config.n_learning_pairs]


def rank_edges_to_item_edges(rank_edges: Sequence[tuple[int, int]], rank_to_item: torch.Tensor, random_orientation: bool = True) -> torch.Tensor:
    bsz = rank_to_item.shape[0]
    edge_rank = torch.tensor(rank_edges, dtype=torch.long, device=DEVICE)
    edges = torch.empty(bsz, len(rank_edges), 2, dtype=torch.long, device=DEVICE)
    for b in range(bsz):
        edges[b, :, 0] = rank_to_item[b, edge_rank[:, 0]]
        edges[b, :, 1] = rank_to_item[b, edge_rank[:, 1]]
    if random_orientation:
        swap = torch.rand(bsz, len(rank_edges), device=DEVICE) < 0.5
        first = edges[:, :, 0].clone()
        second = edges[:, :, 1].clone()
        edges[:, :, 0] = torch.where(swap, second, first)
        edges[:, :, 1] = torch.where(swap, first, second)
    return edges


def all_item_pairs_by_rank(rank_to_item: torch.Tensor, config: MetaPlasticConfig) -> torch.Tensor:
    rank_pairs = torch.tensor(config.all_rank_pairs, dtype=torch.long, device=DEVICE)
    bsz = rank_to_item.shape[0]
    out = torch.empty(bsz, len(rank_pairs), 2, dtype=torch.long, device=DEVICE)
    for b in range(bsz):
        out[b, :, 0] = rank_to_item[b, rank_pairs[:, 0]]
        out[b, :, 1] = rank_to_item[b, rank_pairs[:, 1]]
    return out


def gather_true_diff(true_pos: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(true_pos.shape[0], device=true_pos.device)
    return true_pos[batch, item_j] - true_pos[batch, item_i]


class PlasticRankRNN(nn.Module):
    def __init__(self, config: MetaPlasticConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.item_dim
        z = config.subject_dim
        input_dim = 2 * d + z + 5

        self.init_h = nn.Sequential(nn.Linear(z, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.input_proj = nn.Linear(input_dim, h)
        self.rec_proj = nn.Linear(h, h, bias=False)
        self.hebb_gate = nn.Sequential(nn.Linear(input_dim + h, h), nn.Tanh(), nn.Linear(h, 1))
        self.layer_norm = nn.LayerNorm(h)

        edge_dim = input_dim + h
        self.edge_head = nn.Sequential(
            nn.Linear(edge_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 2),
        )
        self.resource_head = nn.Sequential(nn.Linear(h + z, h), nn.Tanh(), nn.Linear(h, 1))
        # Reliability can be forced to arise from recurrent activity alone.
        # This is the default mechanism used in the new ablation suite.
        self.reliability_head = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

        # Bounded trainable likelihood parameters.  The bounds keep the rank
        # module cognitively fuzzy rather than becoming an exact solver.
        self.raw_sigma = nn.Parameter(self._inv_sigmoid_for_range(config.init_sigma_distance, 0.6, 3.2))
        self.raw_order_bonus = nn.Parameter(self._inv_sigmoid_for_range(config.init_order_bonus, 0.1, 3.0))
        self.raw_precision = nn.Parameter(torch.tensor(math.log(math.exp(config.init_global_precision) - 1.0)))
        self.raw_strength_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_choice_beta_train = nn.Parameter(torch.tensor(math.log(math.exp(2.0) - 1.0)))

    @staticmethod
    def _inv_sigmoid_for_range(value: float, lo: float, hi: float) -> torch.Tensor:
        x = min(max((value - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return torch.tensor(math.log(x / (1.0 - x)), dtype=torch.float32)

    def sigma_distance(self) -> torch.Tensor:
        return 0.6 + 2.6 * torch.sigmoid(self.raw_sigma)

    def order_bonus(self) -> torch.Tensor:
        return 0.1 + 2.9 * torch.sigmoid(self.raw_order_bonus)

    def global_precision(self) -> torch.Tensor:
        return F.softplus(self.raw_precision) + 1e-4

    def strength_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_strength_gain) + 1e-4

    def train_choice_beta(self) -> torch.Tensor:
        return F.softplus(self.raw_choice_beta_train) + 1e-4

    def plastic_step(self, h: torch.Tensor, plastic: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.bmm(plastic, h.unsqueeze(2)).squeeze(2)
        pre = self.input_proj(x) + self.rec_proj(h) + self.config.fast_weight_gain * fast
        h_new = torch.tanh(self.layer_norm(pre))
        gate = torch.sigmoid(self.hebb_gate(torch.cat([x, h_new], dim=1))).squeeze(1)
        eta = self.config.plastic_eta * gate
        hebb = torch.bmm(h_new.unsqueeze(2), h.unsqueeze(1))
        plastic = self.config.plastic_decay * plastic + eta[:, None, None] * hebb
        if self.config.plastic_norm > 0:
            plastic = torch.clamp(plastic, -self.config.plastic_norm, self.config.plastic_norm)
        return h_new, plastic, gate

    def run_learning(
        self,
        item_vecs: torch.Tensor,
        subject_z: torch.Tensor,
        true_pos: torch.Tensor,
        learning_edges: torch.Tensor,
        train_mode: bool,
        noise_std: float | None = None,
        edge_dropout: float | None = None,
    ) -> dict:
        cfg = self.config
        bsz, n_edges, _ = learning_edges.shape
        if noise_std is None:
            noise_std = cfg.relation_noise if train_mode else cfg.eval_relation_noise
        if edge_dropout is None:
            edge_dropout = cfg.train_edge_dropout if train_mode else cfg.eval_edge_dropout

        subject_input = torch.zeros_like(subject_z) if cfg.ablate_subject_latent else subject_z
        item_vecs_input = torch.zeros_like(item_vecs) if cfg.ablate_item_vectors else item_vecs
        h = self.init_h(subject_input)
        if cfg.ablate_rnn:
            h = torch.zeros_like(h)
        plastic = torch.zeros(bsz, cfg.hidden_size, cfg.hidden_size, device=item_vecs.device)
        mem_sum = torch.zeros(bsz, cfg.n_items, cfg.n_items, device=item_vecs.device)
        strength = torch.zeros_like(mem_sum)
        gate_values: list[torch.Tensor] = []
        edge_strength_values: list[torch.Tensor] = []
        edge_recon_losses: list[torch.Tensor] = []
        batch = torch.arange(bsz, device=item_vecs.device)

        for block in range(cfg.n_learning_blocks):
            # Per-subject order is important: it lets identical evidence form
            # different transient attractors through plastic fast weights.
            orders = torch.stack([torch.randperm(n_edges, device=item_vecs.device) for _ in range(bsz)], dim=0)
            for k in range(n_edges):
                edge_idx = orders[:, k]
                item_i = learning_edges[batch, edge_idx, 0]
                item_j = learning_edges[batch, edge_idx, 1]
                true_diff = gather_true_diff(true_pos, item_i, item_j)
                obs_diff = true_diff
                if noise_std > 0:
                    obs_diff = (obs_diff + noise_std * torch.randn_like(obs_diff)).clamp(-(cfg.n_items - 1), cfg.n_items - 1)
                if edge_dropout > 0:
                    keep = (torch.rand_like(obs_diff) > edge_dropout).float()
                    # Dropout means weak/no observation; the RNN receives the old
                    # memory estimate rather than the ground-truth edge.
                    old_strength = strength[batch, item_i, item_j]
                    old_est = torch.where(old_strength > 1e-5, mem_sum[batch, item_i, item_j] / old_strength.clamp_min(1e-5), torch.zeros_like(obs_diff))
                    obs_diff = keep * obs_diff + (1.0 - keep) * old_est.detach()

                vec_i = item_vecs_input[batch, item_i]
                vec_j = item_vecs_input[batch, item_j]
                obs_norm = obs_diff / (cfg.n_items - 1)
                if cfg.observation_mode == "distance":
                    obs_ch1 = obs_norm
                    obs_ch2 = torch.zeros_like(obs_norm)
                    encoded_observation = obs_diff
                elif cfg.observation_mode == "raw_bars":
                    # Raw sensory-style bars: the network sees two noisy absolute bar heights,
                    # not a hand-provided symbolic distance/salience channel.  The relational
                    # evidence available to memory is the observed bar difference.
                    center = torch.empty_like(obs_norm).uniform_(-0.45, 0.45)
                    obs_ch1 = center - 0.5 * obs_norm
                    obs_ch2 = center + 0.5 * obs_norm
                    if cfg.raw_bar_noise > 0:
                        obs_ch1 = obs_ch1 + cfg.raw_bar_noise * torch.randn_like(obs_ch1)
                        obs_ch2 = obs_ch2 + cfg.raw_bar_noise * torch.randn_like(obs_ch2)
                    encoded_observation = ((obs_ch2 - obs_ch1) * (cfg.n_items - 1)).clamp(-(cfg.n_items - 1), cfg.n_items - 1)
                else:
                    raise ValueError(f"Unknown observation_mode: {cfg.observation_mode}")
                features = torch.cat(
                    [
                        vec_i,
                        vec_j,
                        subject_input,
                        obs_ch1[:, None],
                        obs_ch2[:, None],
                        torch.full((bsz, 1), block / max(1, cfg.n_learning_blocks - 1), device=item_vecs.device),
                        strength[batch, item_i, item_j][:, None],
                        torch.ones(bsz, 1, device=item_vecs.device),
                    ],
                    dim=1,
                )
                if cfg.ablate_rnn:
                    h = torch.zeros_like(h)
                    gate = torch.zeros(bsz, device=item_vecs.device)
                    plastic = torch.zeros_like(plastic)
                else:
                    if cfg.ablate_plasticity:
                        plastic = torch.zeros_like(plastic)
                    h, plastic, gate = self.plastic_step(h, plastic, features)
                    if cfg.ablate_plasticity:
                        plastic = torch.zeros_like(plastic)
                head = self.edge_head(torch.cat([features, h], dim=1))

                if cfg.reliability_mode == "rnn":
                    raw_strength = self.reliability_head(h).squeeze(1)
                    write_strength = torch.sigmoid(raw_strength) * self.strength_gain().clamp(0.05, 3.0)
                elif cfg.reliability_mode == "feature_rnn":
                    raw_strength = head[:, 0]
                    write_strength = torch.sigmoid(raw_strength) * self.strength_gain().clamp(0.05, 3.0)
                elif cfg.reliability_mode == "manual_distance":
                    raw_strength = head[:, 0] + cfg.manual_distance_salience * obs_norm.abs() + cfg.manual_subject_salience * subject_input.mean(dim=1)
                    write_strength = torch.sigmoid(raw_strength) * self.strength_gain().clamp(0.05, 3.0)
                elif cfg.reliability_mode == "oracle_distance":
                    raw_strength = cfg.manual_distance_salience * obs_norm.abs() + cfg.manual_subject_salience * subject_input.mean(dim=1)
                    write_strength = torch.sigmoid(raw_strength) * self.strength_gain().clamp(0.05, 3.0)
                elif cfg.reliability_mode == "constant":
                    write_strength = torch.full_like(obs_norm, cfg.constant_reliability)
                else:
                    raise ValueError(f"Unknown reliability_mode: {cfg.reliability_mode}")
                write_strength = write_strength.clamp(1e-4, 1.0)

                if cfg.relation_encoding_mode == "residual_observation":
                    diff_adjust = 0.35 * torch.tanh(head[:, 1])
                    encoded = (encoded_observation + diff_adjust).clamp(-(cfg.n_items - 1), cfg.n_items - 1)
                elif cfg.relation_encoding_mode == "rnn":
                    encoded = ((cfg.n_items - 1) * torch.tanh(head[:, 1])).clamp(-(cfg.n_items - 1), cfg.n_items - 1)
                else:
                    raise ValueError(f"Unknown relation_encoding_mode: {cfg.relation_encoding_mode}")

                prev_s = strength[batch, item_i, item_j]
                prev_sum = mem_sum[batch, item_i, item_j]
                new_s = (prev_s + write_strength).clamp(0.0, 4.0)
                new_sum = prev_sum + write_strength * encoded
                mem_sum = mem_sum.clone()
                strength = strength.clone()
                mem_sum[batch, item_i, item_j] = new_sum
                mem_sum[batch, item_j, item_i] = -new_sum
                strength[batch, item_i, item_j] = new_s
                strength[batch, item_j, item_i] = new_s
                gate_values.append(gate.mean())
                edge_strength_values.append(write_strength.mean())
                target_est = new_sum / new_s.clamp_min(1e-5)
                edge_recon_losses.append((target_est - true_diff).pow(2).mean())

        # Convert accumulated weighted sums to estimated relation memory.
        memory = torch.where(strength > 1e-5, mem_sum / strength.clamp_min(1e-5), torch.zeros_like(mem_sum))
        if cfg.ablate_resource:
            resource = torch.ones(bsz, device=item_vecs.device)
        else:
            resource_raw = self.resource_head(torch.cat([h, subject_input], dim=1)).squeeze(1)
            resource = 0.5 + F.softplus(resource_raw)
        return {
            "memory": memory,
            "strength": strength.clamp(0.0, 4.0),
            "h": h,
            "resource": resource,
            "mean_hebb_gate": torch.stack(gate_values).mean() if gate_values else torch.tensor(0.0, device=item_vecs.device),
            "mean_edge_strength": torch.stack(edge_strength_values).mean() if edge_strength_values else torch.tensor(0.0, device=item_vecs.device),
            "edge_recon_loss": torch.stack(edge_recon_losses).mean() if edge_recon_losses else torch.tensor(0.0, device=item_vecs.device),
        }

    def posterior_logits(
        self,
        hypo: RankHypotheses,
        memory: torch.Tensor,
        strength: torch.Tensor,
        learning_edges: torch.Tensor,
        resource: torch.Tensor,
        resource_multiplier: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        bsz = memory.shape[0]
        pcount = hypo.pos_diff.shape[0]
        logits = torch.empty(bsz, pcount, device=memory.device)
        sigma = self.sigma_distance().clamp_min(0.2)
        order_bonus = self.order_bonus()
        precision = self.global_precision()
        for b in range(bsz):
            edges = learning_edges[b]
            ii = edges[:, 0]
            jj = edges[:, 1]
            target = memory[b, ii, jj]  # [E]
            w = strength[b, ii, jj].clamp(0.0, 4.0)
            cand = hypo.pos_diff[:, ii, jj]  # [P,E]
            signed_ok = (cand * target.sign()[None, :] > 0).float()
            dist_cost = ((cand - target[None, :]) ** 2) / (2.0 * sigma.pow(2))
            edge_logit = (-dist_cost + order_bonus * signed_ok) * w[None, :]
            logits[b] = edge_logit.sum(dim=1)
        mult = resource
        if resource_multiplier is not None:
            mult = mult * resource_multiplier
        logits = logits * precision * mult[:, None]
        return logits

    def posterior_probs(
        self,
        hypo: RankHypotheses,
        memory: torch.Tensor,
        strength: torch.Tensor,
        learning_edges: torch.Tensor,
        resource: torch.Tensor,
        temperature: float,
        resource_multiplier: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.posterior_logits(hypo, memory, strength, learning_edges, resource, resource_multiplier=resource_multiplier)
        logits = logits / max(float(temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values
        return torch.softmax(logits, dim=1)

    def expected_correct_prob(
        self,
        hypo: RankHypotheses,
        posterior: torch.Tensor,
        query_pairs: torch.Tensor,
        choice_beta: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return p(correct) for query pairs supplied as [B,Q,2].

        Query pairs are ordered low-to-high in the true ranking, so a correct
        candidate ranking puts the second item above the first.
        """
        if choice_beta is None:
            beta = self.train_choice_beta()
        elif isinstance(choice_beta, torch.Tensor):
            beta = choice_beta
        else:
            beta = torch.tensor(float(choice_beta), device=posterior.device)
        bsz, qcount, _ = query_pairs.shape
        probs = torch.empty(bsz, qcount, device=posterior.device)
        for b in range(bsz):
            ii = query_pairs[b, :, 0]
            jj = query_pairs[b, :, 1]
            diffs = hypo.pos_diff[:, ii, jj]  # [P,Q]
            p_ok_given_perm = torch.sigmoid(beta * diffs)
            probs[b] = torch.matmul(posterior[b], p_ok_given_perm)
        return probs.clamp(1e-6, 1.0 - 1e-6)

    def straight_through_commit_probs(
        self,
        hypo: RankHypotheses,
        logits: torch.Tensor,
        query_pairs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        # Differentiable hard-ish attractor readout using Gumbel-Softmax.
        y = F.gumbel_softmax(logits, tau=max(temperature, 1e-5), hard=True, dim=1)
        return self.expected_correct_prob(hypo, y, query_pairs, choice_beta=self.train_choice_beta())


@dataclass
class TrainStats:
    loss: torch.Tensor
    loss_value: float
    choice_loss: float
    commit_loss: float
    entropy: float
    edge_recon: float
    accuracy_proxy: float
    sigma: float
    order_bonus: float
    precision: float
    mean_hebb_gate: float
    mean_edge_strength: float


def run_training_episode(config: MetaPlasticConfig, model: PlasticRankRNN, hypo: RankHypotheses) -> TrainStats:
    bsz = config.batch_size
    item_vecs = sample_item_vectors(bsz, config.n_items, config.item_dim)
    subject_z = sample_subject_latents(bsz, config.subject_dim)
    r2i = rank_to_item_maps(bsz, config.n_items)
    true_pos = true_rank_position_by_item(r2i, config)
    rank_edges_per_batch = [sample_sparse_rank_graph(config) for _ in range(bsz)]
    learning_edges = torch.empty(bsz, config.n_learning_pairs, 2, dtype=torch.long, device=DEVICE)
    for b, rank_edges in enumerate(rank_edges_per_batch):
        learning_edges[b] = rank_edges_to_item_edges(rank_edges, r2i[b : b + 1], random_orientation=True)[0]
    query_pairs = all_item_pairs_by_rank(r2i, config)

    state = model.run_learning(item_vecs, subject_z, true_pos, learning_edges, train_mode=True)
    # Mild stochastic resource during training helps the module learn under
    # participant-level posterior precision variability.
    resource_noise = torch.exp(0.35 * torch.randn(bsz, device=DEVICE))
    logits = model.posterior_logits(hypo, state["memory"], state["strength"], learning_edges, state["resource"], resource_multiplier=resource_noise)
    posterior = torch.softmax((logits - logits.max(dim=1, keepdim=True).values) / config.posterior_temperature, dim=1)
    p_correct = model.expected_correct_prob(hypo, posterior, query_pairs)
    choice_loss = -(p_correct + 1e-8).log().mean()
    with torch.no_grad():
        acc_proxy = float((p_correct > 0.5).float().mean().detach())
    entropy = (-(posterior * (posterior + 1e-8).log()).sum(dim=1)).mean()

    commit_loss = torch.tensor(0.0, device=DEVICE)
    if config.straight_through_commit_weight > 0:
        p_commit = model.straight_through_commit_probs(hypo, logits, query_pairs, temperature=config.train_commit_temperature)
        commit_loss = -(p_commit + 1e-8).log().mean()

    # Priors prevent a degenerate exact symbolic solver; the mechanism should
    # remain a bounded, fuzzy biological-style inference system.
    sigma_prior = (model.sigma_distance() - config.init_sigma_distance).pow(2)
    precision_prior = (model.global_precision() - config.init_global_precision).pow(2)
    strength_l1 = state["strength"].mean()
    loss = (
        choice_loss
        + config.straight_through_commit_weight * commit_loss
        - config.posterior_entropy_weight * entropy
        + config.edge_recon_weight * state["edge_recon_loss"]
        + config.strength_l1_weight * strength_l1
        + config.sigma_prior_weight * sigma_prior
        + config.precision_prior_weight * precision_prior
    )
    return TrainStats(
        loss=loss,
        loss_value=float(loss.detach()),
        choice_loss=float(choice_loss.detach()),
        commit_loss=float(commit_loss.detach()),
        entropy=float(entropy.detach()),
        edge_recon=float(state["edge_recon_loss"].detach()),
        accuracy_proxy=acc_proxy,
        sigma=float(model.sigma_distance().detach()),
        order_bonus=float(model.order_bonus().detach()),
        precision=float(model.global_precision().detach()),
        mean_hebb_gate=float(state["mean_hebb_gate"].detach()),
        mean_edge_strength=float(state["mean_edge_strength"].detach()),
    )


def train(config: MetaPlasticConfig, hypo: RankHypotheses) -> PlasticRankRNN:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = PlasticRankRNN(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    log(f"[setup] Device={DEVICE}; parameters={sum(p.numel() for p in model.parameters())}; output={out}")
    rows = []
    t0 = time.time()
    last = None
    disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
    for it in tqdm(range(1, config.nbiter + 1), desc="meta-training", disable=disable_tqdm):
        model.train()
        opt.zero_grad(set_to_none=True)
        stats = run_training_episode(config, model, hypo)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        last = stats
        if it == 1 or it % config.print_every == 0 or it == config.nbiter:
            row = {
                "iter": it,
                "loss": stats.loss_value,
                "choice_loss": stats.choice_loss,
                "commit_loss": stats.commit_loss,
                "entropy": stats.entropy,
                "edge_recon": stats.edge_recon,
                "accuracy_proxy": stats.accuracy_proxy,
                "sigma": stats.sigma,
                "order_bonus": stats.order_bonus,
                "precision": stats.precision,
                "mean_hebb_gate": stats.mean_hebb_gate,
                "mean_edge_strength": stats.mean_edge_strength,
                "elapsed_sec": time.time() - t0,
            }
            rows.append(row)
            log(
                f"[train {it:04d}] loss={row['loss']:.3f} p_acc={row['accuracy_proxy']:.3f} "
                f"H={row['entropy']:.2f} sigma={row['sigma']:.2f} bonus={row['order_bonus']:.2f} "
                f"prec={row['precision']:.2f} edge={row['mean_edge_strength']:.2f}"
            )
        if config.save_every > 0 and it % config.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, out / f"checkpoint_iter_{it}.pt")
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, out / "meta_plastic_rank_rnn.pt")
    if last is not None:
        log(f"[done] final loss={last.loss_value:.4f}, proxy acc={last.accuracy_proxy:.3f}")
    return model


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
    if beta_dist is None:
        return float("nan"), float("nan"), "scipy_unavailable"
    clipped = np.clip(values.astype(float), 1e-3, 1.0 - 1e-3)
    try:
        alpha, beta, _loc, _scale = beta_dist.fit(clipped, floc=0.0, fscale=1.0)
    except Exception:
        # Discrete 0/1-heavy proportions can make beta MLE fail.  Fall back to
        # a robust mass-based category that preserves the behavioral diagnostic.
        low = float((values <= 0.2).mean())
        high = float((values >= 0.8).mean())
        if low > 0.15 and high > 0.15:
            return float("nan"), float("nan"), "bimodal"
        if high > 0.65:
            return float("nan"), float("nan"), "high_accuracy"
        if low > 0.65:
            return float("nan"), float("nan"), "low_accuracy"
        return float("nan"), float("nan"), "unimodal"
    if alpha < 1.0 and beta < 1.0:
        cat = "bimodal"
    elif alpha > 1.0 and beta < 1.0:
        cat = "high_accuracy"
    elif alpha < 1.0 and beta > 1.0:
        cat = "low_accuracy"
    else:
        cat = "unimodal"
    return float(alpha), float(beta), cat


@torch.no_grad()
def evaluate_paper_task(config: MetaPlasticConfig, model: PlasticRankRNN, hypo: RankHypotheses, seed_offset: int = 10000, fit_beta: bool = True) -> dict:
    model.eval()
    rng = np.random.default_rng(config.seed + seed_offset)
    n_subj = config.eval_subjects
    reps = config.eval_repetitions
    # Same external item identities for all subjects; only subject latent / internal
    # noise differs, matching the identical-input behavioral design.
    one_item_set = sample_item_vectors(1, config.n_items, config.item_dim)
    item_vecs = one_item_set.expand(n_subj, -1, -1).contiguous()
    subject_z = sample_subject_latents(n_subj, config.subject_dim)
    r2i = torch.arange(config.n_items, device=DEVICE)[None, :].expand(n_subj, -1).contiguous()
    true_pos = true_rank_position_by_item(r2i, config)
    learning_edges = rank_edges_to_item_edges(PAPER_LEARNING_PAIRS_RANK, r2i, random_orientation=False)
    state = model.run_learning(item_vecs, subject_z, true_pos, learning_edges, train_mode=False)
    resource_mult = torch.tensor(
        np.exp(config.eval_resource_log_mean + config.eval_resource_log_sd * rng.normal(size=n_subj)),
        dtype=torch.float32,
        device=DEVICE,
    )
    logits = model.posterior_logits(hypo, state["memory"], state["strength"], learning_edges, state["resource"], resource_multiplier=resource_mult)
    logits = logits / max(config.eval_commit_temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values
    posterior = torch.softmax(logits, dim=1)
    posterior_np = posterior.cpu().numpy()
    entropy = (-(posterior * (posterior + 1e-8).log()).sum(dim=1)).cpu().numpy()

    chosen_indices = np.zeros(n_subj, dtype=np.int64)
    chosen_orders = np.zeros((n_subj, config.n_items), dtype=np.int64)
    for s in range(n_subj):
        if config.rank_readout == "top1_commit":
            chosen_indices[s] = int(np.argmax(posterior_np[s]))
        else:
            chosen_indices[s] = int(rng.choice(posterior_np.shape[1], p=posterior_np[s]))
        chosen_orders[s] = hypo.permutations[chosen_indices[s]].cpu().numpy()

    pair_list = config.all_rank_pairs
    pair_acc = np.zeros((n_subj, len(pair_list)), dtype=np.float32)
    pos_np = hypo.positions.cpu().numpy()
    if config.rank_readout in {"commit", "top1_commit"}:
        for pidx, (a, b) in enumerate(pair_list):
            subjective_diff = pos_np[chosen_indices, b] - pos_np[chosen_indices, a]
            p_correct = config.eval_lapse * 0.5 + (1.0 - config.eval_lapse) / (1.0 + np.exp(-config.eval_choice_beta * subjective_diff))
            pair_acc[:, pidx] = rng.binomial(reps, p_correct) / reps
    elif config.rank_readout == "posterior_mean":
        for pidx, (a, b) in enumerate(pair_list):
            diffs = pos_np[:, b] - pos_np[:, a]
            p_ok_given_perm = 1.0 / (1.0 + np.exp(-config.eval_choice_beta * diffs))
            p_correct = config.eval_lapse * 0.5 + (1.0 - config.eval_lapse) * (posterior_np @ p_ok_given_perm)
            pair_acc[:, pidx] = rng.binomial(reps, p_correct) / reps
    else:
        raise ValueError(f"Unknown rank_readout: {config.rank_readout}")

    learned_idx = hypo.learned_pair_indices
    nonlearned_idx = hypo.nonlearned_pair_indices
    pair_error_prop = 1.0 - pair_acc
    learned_acc = float(pair_acc[:, learned_idx].mean())
    nonlearned_acc = float(pair_acc[:, nonlearned_idx].mean())
    overall_acc = float(pair_acc.mean())
    consistent_80 = int((pair_error_prop >= 0.8).any(axis=1).sum())
    consistent_100 = int((pair_error_prop >= 1.0).any(axis=1).sum())

    scs, circs = [], []
    correct_ranking = 0
    self_consistent_incorrect = 0
    self_inconsistent = 0
    true_order = tuple(range(config.n_items))
    for s in range(n_subj):
        sc, circ = self_consistency_from_majority(pair_acc[s], pair_list, config.n_items)
        scs.append(sc)
        circs.append(circ)
        if tuple(int(x) for x in chosen_orders[s]) == true_order:
            correct_ranking += 1
        elif circ == 0:
            self_consistent_incorrect += 1
        else:
            self_inconsistent += 1

    taus = [kendall_tau_order(chosen_orders[i], chosen_orders[j]) for i, j in combinations(range(n_subj), 2)]
    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())
    serial_position_acc = {}
    for item in range(config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if a == item or b == item]
        serial_position_acc[str(item + 1)] = float(pair_acc[:, idx].mean())

    beta_counts = {"bimodal": 0, "high_accuracy": 0, "low_accuracy": 0, "unimodal": 0, "scipy_unavailable": 0}
    pair_rows = []
    learned_set = set(PAPER_LEARNING_PAIRS_RANK)
    for pidx, (a, b) in enumerate(pair_list):
        if fit_beta:
            alpha, beta, cat = beta_fit_category(pair_acc[:, pidx])
        else:
            vals = pair_acc[:, pidx]
            low = float((vals <= 0.2).mean())
            high = float((vals >= 0.8).mean())
            if low > 0.15 and high > 0.15:
                alpha, beta, cat = float("nan"), float("nan"), "bimodal"
            elif high > 0.65:
                alpha, beta, cat = float("nan"), float("nan"), "high_accuracy"
            elif low > 0.65:
                alpha, beta, cat = float("nan"), float("nan"), "low_accuracy"
            else:
                alpha, beta, cat = float("nan"), float("nan"), "unimodal"
        beta_counts[cat] = beta_counts.get(cat, 0) + 1
        pair_rows.append(
            {
                "pair": f"{chr(65 + a)}-{chr(65 + b)}",
                "rank_a": a,
                "rank_b": b,
                "rank_distance": b - a,
                "learned_pair": (a, b) in learned_set,
                "mean_accuracy": float(pair_acc[:, pidx].mean()),
                "subjects_majority_error": int((pair_acc[:, pidx] < 0.5).sum()),
                "beta_alpha": alpha,
                "beta_beta": beta,
                "beta_category": cat,
            }
        )

    subject_rows = []
    for s in range(n_subj):
        order_string = "".join(chr(65 + int(x)) for x in chosen_orders[s])
        subject_rows.append(
            {
                "subject": s + 1,
                "order_low_to_high": order_string,
                "kendall_tau_with_true": float(kendall_tau_order(chosen_orders[s], true_order)),
                "pair_accuracy_mean": float(pair_acc[s].mean()),
                "learned_accuracy": float(pair_acc[s, learned_idx].mean()),
                "nonlearned_accuracy": float(pair_acc[s, nonlearned_idx].mean()),
                "self_consistency": float(scs[s]),
                "circular_triads": int(circs[s]),
                "consistent_error_pairs_80pct": int((pair_error_prop[s] >= 0.8).sum()),
                "consistent_error_pairs_100pct": int((pair_error_prop[s] >= 1.0).sum()),
                "posterior_entropy": float(entropy[s]),
                "resource_multiplier": float(resource_mult[s].cpu()),
                "model_resource": float(state["resource"][s].cpu()),
            }
        )

    summary = {
        "model": "meta_trained_plastic_rnn_differentiable_rank_attractor",
        "ablation_signature": {
            "observation_mode": config.observation_mode,
            "reliability_mode": config.reliability_mode,
            "relation_encoding_mode": config.relation_encoding_mode,
            "rank_readout": config.rank_readout,
            "ablate_rnn": config.ablate_rnn,
            "ablate_plasticity": config.ablate_plasticity,
            "ablate_subject_latent": config.ablate_subject_latent,
            "ablate_item_vectors": config.ablate_item_vectors,
            "ablate_resource": config.ablate_resource,
        },
        "config": asdict(config),
        "eval_subjects": n_subj,
        "eval_repetitions": reps,
        "overall_accuracy": overall_acc,
        "learned_pairs_accuracy": learned_acc,
        "nonlearned_pairs_accuracy": nonlearned_acc,
        "consistent_error_subjects_80pct": consistent_80,
        "consistent_error_subjects_80pct_ratio": consistent_80 / n_subj,
        "consistent_error_subjects_100pct": consistent_100,
        "consistent_error_subjects_100pct_ratio": consistent_100 / n_subj,
        "mean_self_consistency_from_majority_choices": float(np.mean(scs)),
        "mean_circular_triads_from_majority_choices": float(np.mean(circs)),
        "correct_ranking_subjects": correct_ranking,
        "self_consistent_incorrect_subjects": self_consistent_incorrect,
        "self_inconsistent_subjects": self_inconsistent,
        "mean_inter_subject_kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "distance_accuracy": distance_acc,
        "serial_position_accuracy": serial_position_acc,
        "beta_pair_category_counts": beta_counts,
        "mean_posterior_entropy": float(np.mean(entropy)),
        "mean_model_resource": float(state["resource"].mean().cpu()),
        "mean_eval_resource_multiplier": float(resource_mult.mean().cpu()),
        "mean_hebb_gate": float(state["mean_hebb_gate"].cpu()),
        "mean_edge_strength": float(state["mean_edge_strength"].cpu()),
        "edge_recon_loss": float(state["edge_recon_loss"].cpu()),
        "learned_likelihood_sigma": float(model.sigma_distance().cpu()),
        "learned_order_bonus": float(model.order_bonus().cpu()),
        "learned_global_precision": float(model.global_precision().cpu()),
        "learned_train_choice_beta": float(model.train_choice_beta().cpu()),
    }
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "subject_rows": subject_rows,
        "pair_accuracy_matrix": pair_acc,
        "chosen_orders": chosen_orders,
        "posterior_entropy": entropy,
        "posterior_top1_prob": posterior_np.max(axis=1),
    }


def write_eval_outputs(output_dir: Path, result: dict, prefix: str = "meta_plastic") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{prefix}_paper_task_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)
    with open(output_dir / f"{prefix}_pair_accuracy_beta_fits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["pair_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["pair_rows"])
    with open(output_dir / f"{prefix}_subject_rankings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["subject_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["subject_rows"])
    np.save(output_dir / f"{prefix}_pair_accuracy_matrix.npy", result["pair_accuracy_matrix"])
    np.save(output_dir / f"{prefix}_chosen_orders.npy", result["chosen_orders"])


def compact_score(summary: dict) -> float:
    # Approximate human-like targets used only for comparing model variants.
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
    for k, t in targets.items():
        score += abs(summary[k] - t)
    score += abs(summary["correct_ranking_subjects"] - 8) / 100.0
    if summary["learned_pairs_accuracy"] < summary["nonlearned_pairs_accuracy"]:
        score += 0.25
    return score


def run_eval_sweep(config: MetaPlasticConfig, model: PlasticRankRNN, hypo: RankHypotheses, output_dir: Path) -> list[dict]:
    rows = []
    # Small post-training mechanism sweep.  This is not fitting weights; it checks
    # whether the neural posterior needs a different commitment/readout regime.
    betas = [1.8, 2.2, 2.6]
    lapses = [0.02, 0.03, 0.05]
    resource_sds = [0.35, 0.55, 0.75]
    temps = [0.85, 1.0, 1.25]
    base = asdict(config)
    for beta in betas:
        for lapse in lapses:
            for rsd in resource_sds:
                for temp in temps:
                    cfg = MetaPlasticConfig(**base)
                    cfg.eval_choice_beta = beta
                    cfg.eval_lapse = lapse
                    cfg.eval_resource_log_sd = rsd
                    cfg.eval_commit_temperature = temp
                    result = evaluate_paper_task(cfg, model, hypo, seed_offset=20000, fit_beta=False)
                    s = result["summary"]
                    rows.append(
                        {
                            "score": compact_score(s),
                            "eval_choice_beta": beta,
                            "eval_lapse": lapse,
                            "eval_resource_log_sd": rsd,
                            "eval_commit_temperature": temp,
                            "overall_accuracy": s["overall_accuracy"],
                            "learned_pairs_accuracy": s["learned_pairs_accuracy"],
                            "nonlearned_pairs_accuracy": s["nonlearned_pairs_accuracy"],
                            "consistent80_ratio": s["consistent_error_subjects_80pct_ratio"],
                            "consistent100_ratio": s["consistent_error_subjects_100pct_ratio"],
                            "choice_self_consistency": s["mean_self_consistency_from_majority_choices"],
                            "choice_circular_triads": s["mean_circular_triads_from_majority_choices"],
                            "inter_subject_tau": s["mean_inter_subject_kendall_tau"],
                            "correct_ranking_subjects": s["correct_ranking_subjects"],
                            "self_consistent_incorrect_subjects": s["self_consistent_incorrect_subjects"],
                            "mean_posterior_entropy": s["mean_posterior_entropy"],
                            "beta_bimodal_pairs": s["beta_pair_category_counts"].get("bimodal", 0),
                            "beta_high_accuracy_pairs": s["beta_pair_category_counts"].get("high_accuracy", 0),
                        }
                    )
    rows.sort(key=lambda r: r["score"])
    with open(output_dir / "eval_commitment_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "eval_commitment_sweep_top10.json", "w", encoding="utf-8") as f:
        json.dump(rows[:10], f, indent=2, ensure_ascii=False)
    return rows


def make_report(output_dir: Path, summary: dict, sweep_rows: list[dict] | None = None) -> None:
    cfg = summary["config"]
    lines = [
        "# Meta-trained plastic RNN with differentiable rank-attractor module",
        "",
        "## 机制",
        "",
        "该版本把之前离散枚举采样器改造成可微模块，并接到完整的 observation-only meta-trained plastic RNN：学习阶段由带 Hebbian fast weights 的 RNN 逐个处理 pair observation，写入局部 edge memory；学习结束后，可微 active-rank posterior 对全部 8! 个全局排序假设计算 soft posterior；训练时从 soft posterior 反传，评估时每个 virtual subject 抽样/承诺到一个 ranking attractor，再进行重复无反馈选择。",
        "",
        "## 行为学固定任务结果",
        "",
        f"- seed = {cfg['seed']}; meta-training episodes = {cfg['nbiter']}; virtual subjects = {summary['eval_subjects']}; repetitions = {summary['eval_repetitions']}",
        f"- overall accuracy = {summary['overall_accuracy']:.3f}",
        f"- learned / non-learned accuracy = {summary['learned_pairs_accuracy']:.3f} / {summary['nonlearned_pairs_accuracy']:.3f}",
        f"- >=80% stable-error subjects = {summary['consistent_error_subjects_80pct']} / {summary['eval_subjects']} ({summary['consistent_error_subjects_80pct_ratio']:.3f})",
        f"- 100% stable-error subjects = {summary['consistent_error_subjects_100pct']} / {summary['eval_subjects']} ({summary['consistent_error_subjects_100pct_ratio']:.3f})",
        f"- majority-choice self-consistency = {summary['mean_self_consistency_from_majority_choices']:.3f}; circular triads = {summary['mean_circular_triads_from_majority_choices']:.3f}",
        f"- correct / self-consistent incorrect / self-inconsistent subjects = {summary['correct_ranking_subjects']} / {summary['self_consistent_incorrect_subjects']} / {summary['self_inconsistent_subjects']}",
        f"- inter-subject Kendall tau = {summary['mean_inter_subject_kendall_tau']:.3f}",
        f"- beta pair categories = {summary['beta_pair_category_counts']}",
        f"- distance accuracy = {', '.join(k + ':' + format(v, '.3f') for k, v in summary['distance_accuracy'].items())}",
        "",
        "## 学到的神经模块参数",
        "",
        f"- likelihood sigma = {summary['learned_likelihood_sigma']:.3f}",
        f"- order bonus = {summary['learned_order_bonus']:.3f}",
        f"- global posterior precision = {summary['learned_global_precision']:.3f}",
        f"- train-time choice beta = {summary['learned_train_choice_beta']:.3f}",
        f"- mean Hebbian gate = {summary['mean_hebb_gate']:.3f}",
        f"- mean edge write strength = {summary['mean_edge_strength']:.3f}",
        "",
        "## 解释",
        "",
        "这个模型与旧版连续 score/update 模型的核心差异不是多加噪声，而是把学习后的内部表征组织成一个 subject-level global-ranking attractor。RNN 和 fast weights 负责把相同局部输入变成个体化 edge reliability；active-rank posterior 负责把局部证据压缩为全局排序假设；测试读出来自同一个已承诺排序，因此稳定错误和传递自洽可以同时出现。",
    ]
    if sweep_rows:
        lines.extend(["", "## Post-training commitment/readout sweep top 5", ""])
        for i, r in enumerate(sweep_rows[:5], 1):
            lines.append(
                f"{i}. score={r['score']:.3f}, beta={r['eval_choice_beta']}, lapse={r['eval_lapse']}, "
                f"resource_sd={r['eval_resource_log_sd']}, temp={r['eval_commit_temperature']}, "
                f"overall={r['overall_accuracy']:.3f}, learned={r['learned_pairs_accuracy']:.3f}, "
                f"nonlearned={r['nonlearned_pairs_accuracy']:.3f}, c80={r['consistent80_ratio']:.3f}, "
                f"c100={r['consistent100_ratio']:.3f}, self={r['choice_self_consistency']:.3f}, "
                f"tau={r['inter_subject_tau']:.3f}, correct={r['correct_ranking_subjects']}, "
                f"bimodal={r['beta_bimodal_pairs']}"
            )
    (output_dir / "META_PLASTIC_RNN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-trained plastic RNN with differentiable rank-attractor posterior.")
    p.add_argument("--seed", type=int, default=1301)
    p.add_argument("--nbiter", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--item-dim", type=int, default=16)
    p.add_argument("--subject-dim", type=int, default=10)
    p.add_argument("--relation-noise", type=float, default=0.10)
    p.add_argument("--eval-relation-noise", type=float, default=0.00)
    p.add_argument("--train-edge-dropout", type=float, default=0.03)
    p.add_argument("--eval-edge-dropout", type=float, default=0.00)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--output-dir", default="outputs_meta_plastic_rank_rnn")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-subjects", type=int, default=77)
    p.add_argument("--eval-repetitions", type=int, default=10)
    p.add_argument("--eval-choice-beta", type=float, default=2.2)
    p.add_argument("--eval-lapse", type=float, default=0.03)
    p.add_argument("--eval-resource-log-sd", type=float, default=0.55)
    p.add_argument("--eval-commit-temperature", type=float, default=1.0)
    p.add_argument("--straight-through-commit-weight", type=float, default=0.15)
    p.add_argument("--posterior-entropy-weight", type=float, default=0.005)
    p.add_argument("--edge-recon-weight", type=float, default=0.03)
    p.add_argument("--plastic-eta", type=float, default=0.075)
    p.add_argument("--plastic-decay", type=float, default=0.92)
    p.add_argument("--fast-weight-gain", type=float, default=0.75)
    p.add_argument("--observation-mode", choices=["distance", "raw_bars"], default="distance")
    p.add_argument("--reliability-mode", choices=["rnn", "feature_rnn", "manual_distance", "constant", "oracle_distance"], default="rnn")
    p.add_argument("--relation-encoding-mode", choices=["residual_observation", "rnn"], default="rnn")
    p.add_argument("--rank-readout", choices=["commit", "top1_commit", "posterior_mean"], default="commit")
    p.add_argument("--ablate-rnn", action="store_true")
    p.add_argument("--ablate-plasticity", action="store_true")
    p.add_argument("--ablate-subject-latent", action="store_true")
    p.add_argument("--ablate-item-vectors", action="store_true")
    p.add_argument("--ablate-resource", action="store_true")
    p.add_argument("--manual-distance-salience", type=float, default=0.9)
    p.add_argument("--manual-subject-salience", type=float, default=0.15)
    p.add_argument("--constant-reliability", type=float, default=0.5)
    p.add_argument("--raw-bar-noise", type=float, default=0.02)
    p.add_argument("--load-checkpoint", default=None, help="Load a checkpoint and evaluate or continue training from it.")
    p.add_argument("--eval-only", action="store_true", help="Evaluate the loaded/untrained model without optimizer steps.")
    p.add_argument("--eval-sweep", action="store_true")
    p.add_argument("--no-train", action="store_true", help="Evaluate an untrained model; useful as a sanity check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MetaPlasticConfig(
        seed=args.seed,
        nbiter=args.nbiter,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        item_dim=args.item_dim,
        subject_dim=args.subject_dim,
        relation_noise=args.relation_noise,
        eval_relation_noise=args.eval_relation_noise,
        train_edge_dropout=args.train_edge_dropout,
        eval_edge_dropout=args.eval_edge_dropout,
        lr=args.lr,
        output_dir=args.output_dir,
        print_every=args.print_every,
        save_every=args.save_every,
        eval_subjects=args.eval_subjects,
        eval_repetitions=args.eval_repetitions,
        eval_choice_beta=args.eval_choice_beta,
        eval_lapse=args.eval_lapse,
        eval_resource_log_sd=args.eval_resource_log_sd,
        eval_commit_temperature=args.eval_commit_temperature,
        straight_through_commit_weight=args.straight_through_commit_weight,
        posterior_entropy_weight=args.posterior_entropy_weight,
        edge_recon_weight=args.edge_recon_weight,
        plastic_eta=args.plastic_eta,
        plastic_decay=args.plastic_decay,
        fast_weight_gain=args.fast_weight_gain,
        observation_mode=args.observation_mode,
        reliability_mode=args.reliability_mode,
        relation_encoding_mode=args.relation_encoding_mode,
        rank_readout=args.rank_readout,
        ablate_rnn=args.ablate_rnn,
        ablate_plasticity=args.ablate_plasticity,
        ablate_subject_latent=args.ablate_subject_latent,
        ablate_item_vectors=args.ablate_item_vectors,
        ablate_resource=args.ablate_resource,
        manual_distance_salience=args.manual_distance_salience,
        manual_subject_salience=args.manual_subject_salience,
        constant_reliability=args.constant_reliability,
        raw_bar_noise=args.raw_bar_noise,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    with open(out / "config_meta_plastic_rank_rnn.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    hypo = make_rank_hypotheses(cfg)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=DEVICE)
        # Command-line ablation switches deliberately override the stored config,
        # so the same trained network can be re-evaluated under independent lesions.
        model = PlasticRankRNN(cfg).to(DEVICE)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            log(f"[load] non-strict checkpoint load: missing={list(missing)}, unexpected={list(unexpected)}")
    elif args.no_train:
        model = PlasticRankRNN(cfg).to(DEVICE)
    else:
        model = train(cfg, hypo)
    if args.load_checkpoint and (not args.eval_only) and (not args.no_train):
        # Continue training from the loaded checkpoint under the current switches.
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rows = []
        t0 = time.time()
        for it in tqdm(range(1, cfg.nbiter + 1), desc="continued-meta-training", disable=os.environ.get("DISABLE_TQDM", "0") == "1"):
            model.train()
            opt.zero_grad(set_to_none=True)
            stats = run_training_episode(cfg, model, hypo)
            stats.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if it == 1 or it % cfg.print_every == 0 or it == cfg.nbiter:
                row = {
                    "iter": it, "loss": stats.loss_value, "choice_loss": stats.choice_loss,
                    "commit_loss": stats.commit_loss, "entropy": stats.entropy, "edge_recon": stats.edge_recon,
                    "accuracy_proxy": stats.accuracy_proxy, "sigma": stats.sigma, "order_bonus": stats.order_bonus,
                    "precision": stats.precision, "mean_hebb_gate": stats.mean_hebb_gate,
                    "mean_edge_strength": stats.mean_edge_strength, "elapsed_sec": time.time() - t0,
                }
                rows.append(row)
                log(f"[cont {it:04d}] loss={row['loss']:.3f} p_acc={row['accuracy_proxy']:.3f} H={row['entropy']:.2f} edge={row['mean_edge_strength']:.2f}")
        if rows:
            with open(out / "continued_train_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out / "meta_plastic_rank_rnn.pt")
    result = evaluate_paper_task(cfg, model, hypo, fit_beta=True)
    write_eval_outputs(out, result)
    sweep_rows = run_eval_sweep(cfg, model, hypo, out) if args.eval_sweep else None
    # If sweep found a better readout regime, also write the best-run outputs.
    if sweep_rows:
        best = sweep_rows[0]
        best_cfg = MetaPlasticConfig(**asdict(cfg))
        best_cfg.eval_choice_beta = float(best["eval_choice_beta"])
        best_cfg.eval_lapse = float(best["eval_lapse"])
        best_cfg.eval_resource_log_sd = float(best["eval_resource_log_sd"])
        best_cfg.eval_commit_temperature = float(best["eval_commit_temperature"])
        best_result = evaluate_paper_task(best_cfg, model, hypo, seed_offset=20000, fit_beta=True)
        write_eval_outputs(out, best_result, prefix="meta_plastic_best_sweep")
        make_report(out, best_result["summary"], sweep_rows=sweep_rows)
        summary_to_print = best_result["summary"]
    else:
        make_report(out, result["summary"], sweep_rows=None)
        summary_to_print = result["summary"]
    log("[eval] behavioral-paper task summary:")
    for key, value in summary_to_print.items():
        if key == "config":
            continue
        log(f"  {key}: {value}")
    log(f"[done] wrote outputs to {out}")


if __name__ == "__main__":
    main()
