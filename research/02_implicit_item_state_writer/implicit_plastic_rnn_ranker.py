"""
Implicit plastic recurrent ranking model for no-feedback transitive/rank inference.

V3 fixed from V2 baseline: query-pair orientation randomization plus AMP-safe binary entropy to prevent NaN loss.

Design goal
-----------
This is a main-model alternative to the explicit score / edge-table models used in
stages 01--04. It deliberately does NOT maintain:
  - scalar item scores
  - an explicit pairwise edge-memory table
  - an enumerated 8! rank posterior

Instead it maintains episode-local neural state:
  - dynamic item representations e_i(t)
  - optional global recurrent hidden state h_t
  - optional Hebbian fast weights A_t

Learning trials provide passive observed relations, not reward. Test trials provide
no feedback, but can still update internal recurrent / plastic / item state from
self-generated choice traces.

Outer-loop training is supervised after the full episode, using the true ranking
only to compute a meta-loss on test choices. Ground truth is never used during the
no-feedback test updates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DEVICE = torch.device("cpu")

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
class Config:
    # run control
    rngseed: int = 1
    nbiter: int = 200
    bs: int = 128
    lr: float = 3e-4
    eps: float = 1e-8
    l2: float = 0.0
    gc: float = 2.0
    pe: int = 50
    save_every: int = 1000
    output_dir: str = "outputs_implicit_plastic"
    device: str = "auto"  # auto|cpu|cuda
    amp: bool = False
    compile_model: bool = False
    num_threads: int = 1

    # task
    n_items: int = 8
    n_learning_pairs: int = 8
    n_learning_blocks: int = 4
    paper_graph_train_prob: float = 0.3
    relation_noise: float = 0.03
    eval_relation_noise: float = 0.0
    edge_dropout: float = 0.0
    eval_edge_dropout: float = 0.0
    relation_input: str = "magnitude"  # magnitude|sign|raw_bars

    # representation sizes
    item_input_dim: int = 32
    item_state_dim: int = 64
    subject_dim: int = 16
    hidden_size: int = 192
    rnn_hidden_size: int = 128

    # optional mechanisms
    use_rnn: bool = True
    use_hebbian: bool = True
    use_subject_modulation: bool = True
    use_reliability_gate: bool = True
    use_item_state_updates: bool = True

    # recurrent / plastic dynamics
    rnn_input_gain: float = 1.0
    rnn_rec_gain: float = 1.0
    fast_weight_gain: float = 0.45
    hebb_eta: float = 0.08
    plastic_decay: float = 0.97
    plastic_clip: float = 3.0
    detach_plastic_state: bool = False
    rnn_dropout: float = 0.0
    item_state_decay: float = 0.0
    item_update_scale: float = 0.35
    item_norm_clip: float = 5.0

    # blank/consolidation updates between learning blocks
    consolidation_steps: int = 0
    consolidation_eta_scale: float = 0.25

    # test phase
    test_update_mode: str = "self_reconsolidate"  # frozen|hidden|self_reconsolidate
    train_test_repetitions: int = 1
    eval_repetitions: int = 10
    test_order_shuffle: bool = True
    train_query_random_orientation: bool = True
    eval_query_random_orientation: bool = True
    test_eta_scale: float = 0.35
    test_item_update_scale: float = 0.10
    test_pseudo_confidence_power: float = 1.0
    detach_test_pseudo: bool = False
    eval_sample_choices: bool = True

    # readout / loss
    choice_beta_init: float = 2.0
    eval_beta_override: float = 0.0
    lambda_state_norm: float = 1e-4
    lambda_plastic_l2: float = 1e-5
    lambda_eta_l2: float = 1e-4
    lambda_gate_l1: float = 0.0
    lambda_entropy: float = 0.002

    # eval
    eval_subjects: int = 128

    @property
    def all_rank_pairs(self) -> list[tuple[int, int]]:
        return list(combinations(range(self.n_items), 2))

    @property
    def rank_values(self) -> torch.Tensor:
        values = torch.arange(self.n_items, dtype=torch.float32)
        return (values - values.mean()) / (self.n_items - 1)


@dataclass
class State:
    item_state: torch.Tensor  # [B,N,D]
    h: torch.Tensor           # [B,H]
    plastic: torch.Tensor     # [B,H,H]


@dataclass
class PhaseStats:
    mean_gate: torch.Tensor
    mean_eta: torch.Tensor
    mean_abs_plastic: torch.Tensor
    state_norm: torch.Tensor
    plastic_l2: torch.Tensor


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    choice_loss: float
    entropy_loss: float
    acc: float
    learned_acc: float
    nonlearned_acc: float
    mean_gate_learning: float
    mean_eta_learning: float
    mean_abs_plastic_learning: float
    mean_gate_test: float
    mean_eta_test: float
    mean_abs_plastic_test: float
    state_norm: float


def sample_subject_latents(batch_size: int, subject_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, subject_dim, device=DEVICE)


def rank_to_item_maps(batch_size: int, n_items: int) -> torch.Tensor:
    return torch.stack([torch.randperm(n_items, device=DEVICE) for _ in range(batch_size)], dim=0)


def true_scores_by_item(rank_to_item: torch.Tensor, config: Config) -> torch.Tensor:
    values = config.rank_values.to(DEVICE)
    bsz = rank_to_item.shape[0]
    scores = torch.empty(bsz, config.n_items, device=DEVICE)
    batch_idx = torch.arange(bsz, device=DEVICE)
    for rank in range(config.n_items):
        # Per-subject assignment: rank_to_item[b, rank] is the item occupying this
        # rank for subject b. Using scores[:, rank_to_item[:, rank]] is Cartesian
        # advanced indexing and overwrites columns across subjects.
        scores[batch_idx, rank_to_item[:, rank]] = values[rank]
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


def sample_sparse_rank_graph(config: Config) -> list[tuple[int, int]]:
    if config.paper_graph_train_prob > 0 and random.random() < config.paper_graph_train_prob:
        return list(PAPER_LEARNING_PAIRS_RANK)
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
    edges = [(i, i + 1) for i in range(config.n_items - 1)]
    remaining = [p for p in all_pairs if p not in edges]
    edges.append(random.choice(remaining))
    return edges[: config.n_learning_pairs]


def rank_edges_to_item_edges(rank_edges: list[tuple[int, int]], rank_to_item: torch.Tensor, random_orientation: bool = True) -> torch.Tensor:
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
    return item_edges


def all_item_pairs(rank_to_item: torch.Tensor, config: Config) -> torch.Tensor:
    rank_pairs = torch.tensor(config.all_rank_pairs, dtype=torch.long, device=DEVICE)
    bsz = rank_to_item.shape[0]
    pairs = torch.empty(bsz, len(config.all_rank_pairs), 2, dtype=torch.long, device=DEVICE)
    for b in range(bsz):
        pairs[b, :, 0] = rank_to_item[b, rank_pairs[:, 0]]
        pairs[b, :, 1] = rank_to_item[b, rank_pairs[:, 1]]
    return pairs


def randomize_query_pair_orientation(pairs: torch.Tensor) -> torch.Tensor:
    """Randomly swap the two displayed items for each query pair.

    This removes a position-label shortcut in the paper-style evaluation where
    all rank pairs are enumerated as (lower_rank, higher_rank), making class 1
    always correct. The unordered pair identity is unchanged, and labels are
    still computed from true_scores after orientation is chosen.
    """
    swapped_pairs = pairs.clone()
    swap = torch.rand(pairs.shape[0], pairs.shape[1], device=pairs.device) < 0.5
    first = swapped_pairs[:, :, 0].clone()
    second = swapped_pairs[:, :, 1].clone()
    swapped_pairs[:, :, 0] = torch.where(swap, second, first)
    swapped_pairs[:, :, 1] = torch.where(swap, first, second)
    return swapped_pairs


def gather_true_diff(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    bidx = torch.arange(true_scores.shape[0], device=DEVICE)
    return true_scores[bidx, item_j] - true_scores[bidx, item_i]


def pair_labels(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    return (gather_true_diff(true_scores, item_i, item_j) > 0).float()


def binary_entropy_from_logits(logit: torch.Tensor) -> torch.Tensor:
    """Numerically stable Bernoulli entropy H(sigmoid(logit)).

    The previous implementation used prob * log(prob + 1e-8). Under CUDA AMP,
    1e-8 can underflow in fp16 and saturated probabilities can produce 0 * -inf,
    which made the total loss become NaN while choice_loss stayed finite.
    Compute the entropy in fp32 using the identity:
        H(sigmoid(x)) = softplus(x) - sigmoid(x) * x
    """
    logit32 = logit.float()
    prob32 = torch.sigmoid(logit32)
    return F.softplus(logit32) - prob32 * logit32


def assert_finite_loss(loss: torch.Tensor, components: dict[str, torch.Tensor | float]) -> None:
    """Fail fast with component values if the scalar training loss is non-finite."""
    if torch.isfinite(loss).all():
        return
    parts = []
    for name, value in components.items():
        if torch.is_tensor(value):
            val = value.detach()
            if val.numel() == 1:
                parts.append(f"{name}={float(val.cpu())}")
            else:
                parts.append(f"{name}_finite={bool(torch.isfinite(val).all().cpu())}")
        else:
            parts.append(f"{name}={value}")
    raise FloatingPointError("Non-finite training loss: " + ", ".join(parts))


class ImplicitPlasticRanker(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        N = config.n_items
        D0 = config.item_input_dim
        D = config.item_state_dim
        S = config.subject_dim
        H = config.rnn_hidden_size
        M = config.hidden_size

        self.base_item = nn.Parameter(torch.randn(N, D0) / math.sqrt(D0))
        init_in = D0 + (S if config.use_subject_modulation else 0)
        self.item_init = nn.Sequential(nn.Linear(init_in, M), nn.Tanh(), nn.Linear(M, D), nn.LayerNorm(D))
        self.h_init = nn.Sequential(nn.Linear(S, H), nn.Tanh(), nn.Linear(H, H)) if config.use_subject_modulation else None

        # relation features: signed relation, absolute magnitude, phase flag, confidence/self flag, optional raw bars two channels
        self.rel_feat_dim = 6
        pair_dim = 4 * D + self.rel_feat_dim + (S if config.use_subject_modulation else 0)
        rnn_in_dim = pair_dim
        if config.use_rnn:
            self.rnn_input = nn.Linear(rnn_in_dim, H)
            self.rnn_rec = nn.Linear(H, H, bias=False)
            self.rnn_ln = nn.LayerNorm(H)
            self.rnn_dropout = nn.Dropout(config.rnn_dropout)
            eta_in = rnn_in_dim + H + (S if config.use_subject_modulation else 0)
            self.eta_head = nn.Sequential(nn.Linear(eta_in, M), nn.Tanh(), nn.Linear(M, 1))
        else:
            self.register_parameter("rnn_input", None)

        writer_in = pair_dim + H
        self.write_head = nn.Sequential(nn.Linear(writer_in, M), nn.Tanh(), nn.Linear(M, M), nn.Tanh(), nn.Linear(M, 2 * D))
        self.gate_head = nn.Sequential(nn.Linear(writer_in, M), nn.Tanh(), nn.Linear(M, 1))

        # Antisymmetric pair comparator. Positive logit means item_j is judged higher than item_i.
        phi_in = 2 * D + D + H + (S if config.use_subject_modulation else 0)
        self.phi = nn.Sequential(nn.Linear(phi_in, M), nn.Tanh(), nn.Linear(M, M), nn.Tanh(), nn.Linear(M, 1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(config.choice_beta_init), dtype=torch.float32))

    def initial_state(self, bsz: int, subject_z: torch.Tensor) -> State:
        cfg = self.config
        base = self.base_item[None, :, :].expand(bsz, -1, -1)
        if cfg.use_subject_modulation:
            subj = subject_z[:, None, :].expand(-1, cfg.n_items, -1)
            init_x = torch.cat([base, subj], dim=-1)
        else:
            init_x = base
        item_state = self.item_init(init_x)
        if cfg.use_rnn:
            if self.h_init is not None:
                h = self.h_init(subject_z)
            else:
                h = torch.zeros(bsz, cfg.rnn_hidden_size, device=DEVICE)
            plastic = torch.zeros(bsz, cfg.rnn_hidden_size, cfg.rnn_hidden_size, device=DEVICE)
        else:
            h = torch.zeros(bsz, cfg.rnn_hidden_size, device=DEVICE)
            plastic = torch.zeros(bsz, cfg.rnn_hidden_size, cfg.rnn_hidden_size, device=DEVICE)
        return State(item_state=item_state, h=h, plastic=plastic)

    def relation_features(self, relation: torch.Tensor, phase: str, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.config
        if confidence is None:
            confidence = torch.ones_like(relation)
        if cfg.relation_input == "sign":
            r = relation.sign()
            mag = torch.zeros_like(relation)
        elif cfg.relation_input == "raw_bars":
            # Two pseudo sensory channels; difference carries relation, center is irrelevant nuisance.
            center = torch.zeros_like(relation)
            bar_i = center - 0.5 * relation
            bar_j = center + 0.5 * relation
            r = bar_j - bar_i
            mag = torch.stack([bar_i, bar_j], dim=1).mean(dim=1).abs()
        else:
            r = relation
            mag = relation.abs()
        is_test = torch.ones_like(relation) if phase == "test" else torch.zeros_like(relation)
        is_self = torch.ones_like(relation) if phase == "test" else torch.zeros_like(relation)
        return torch.stack([r, mag, confidence, is_test, is_self, torch.ones_like(relation)], dim=1)

    def _gather_items(self, item_state: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bidx = torch.arange(item_state.shape[0], device=DEVICE)
        return item_state[bidx, item_i], item_state[bidx, item_j]

    def pair_context(self, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor,
                     relation: torch.Tensor, phase: str, confidence: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ei, ej = self._gather_items(state.item_state, item_i, item_j)
        rel = self.relation_features(relation, phase, confidence)
        parts = [ei, ej, ej - ei, ei * ej, rel]
        if self.config.use_subject_modulation:
            parts.append(subject_z)
        return torch.cat(parts, dim=1), ei, ej

    def rnn_step(self, x: torch.Tensor, state: State, subject_z: torch.Tensor, eta_scale: float) -> tuple[State, torch.Tensor]:
        cfg = self.config
        if not cfg.use_rnn:
            return state, torch.zeros(x.shape[0], device=DEVICE)
        h_prev = state.h.detach() if cfg.detach_plastic_state else state.h
        plastic_prev = state.plastic.detach() if cfg.detach_plastic_state else state.plastic
        fast = torch.bmm(plastic_prev, h_prev.unsqueeze(2)).squeeze(2)
        pre = cfg.rnn_input_gain * self.rnn_input(x) + cfg.rnn_rec_gain * self.rnn_rec(state.h) + cfg.fast_weight_gain * fast
        h_new = torch.tanh(self.rnn_ln(pre))
        h_new = self.rnn_dropout(h_new)
        eta_parts = [x, h_new]
        if cfg.use_subject_modulation:
            eta_parts.append(subject_z)
        eta = cfg.hebb_eta * eta_scale * torch.sigmoid(self.eta_head(torch.cat(eta_parts, dim=1)).squeeze(1))
        if not cfg.use_hebbian:
            eta = torch.zeros_like(eta)
            plastic_new = torch.zeros_like(state.plastic)
        else:
            hebb = torch.bmm(h_new.unsqueeze(2), h_prev.unsqueeze(1))
            plastic_new = cfg.plastic_decay * state.plastic + eta[:, None, None] * hebb
            if cfg.plastic_clip > 0:
                plastic_new = plastic_new.clamp(-cfg.plastic_clip, cfg.plastic_clip)
        return State(state.item_state, h_new, plastic_new), eta

    def write_items(self, x: torch.Tensor, state: State, item_i: torch.Tensor, item_j: torch.Tensor, scale: float) -> tuple[State, torch.Tensor]:
        cfg = self.config
        if not cfg.use_item_state_updates or scale <= 0:
            return state, torch.zeros(x.shape[0], device=DEVICE)
        wh = torch.cat([x, state.h], dim=1)
        delta = torch.tanh(self.write_head(wh)).view(x.shape[0], 2, cfg.item_state_dim)
        if cfg.use_reliability_gate:
            gate = torch.sigmoid(self.gate_head(wh).squeeze(1))
        else:
            gate = torch.ones(x.shape[0], device=DEVICE)
        step = scale * gate
        item_state = state.item_state.clone()
        bidx = torch.arange(item_state.shape[0], device=DEVICE)
        old_i = item_state[bidx, item_i]
        old_j = item_state[bidx, item_j]
        new_i = (1.0 - cfg.item_state_decay) * old_i + step[:, None] * delta[:, 0, :]
        new_j = (1.0 - cfg.item_state_decay) * old_j + step[:, None] * delta[:, 1, :]
        if cfg.item_norm_clip > 0:
            new_i = torch.clamp(new_i, -cfg.item_norm_clip, cfg.item_norm_clip)
            new_j = torch.clamp(new_j, -cfg.item_norm_clip, cfg.item_norm_clip)
        item_state[bidx, item_i] = new_i
        item_state[bidx, item_j] = new_j
        return State(item_state, state.h, state.plastic), gate

    def observe_pair(self, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor,
                     relation: torch.Tensor, phase: str, eta_scale: float, item_update_scale: float,
                     confidence: Optional[torch.Tensor] = None) -> tuple[State, torch.Tensor, torch.Tensor]:
        x, _, _ = self.pair_context(state, subject_z, item_i, item_j, relation, phase, confidence)
        state, eta = self.rnn_step(x, state, subject_z, eta_scale)
        state, gate = self.write_items(x, state, item_i, item_j, item_update_scale)
        return state, gate, eta

    def comparator_logit(self, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
        ei, ej = self._gather_items(state.item_state, item_i, item_j)
        def phi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            parts = [a, b, b - a, state.h]
            if self.config.use_subject_modulation:
                parts.append(subject_z)
            return self.phi(torch.cat(parts, dim=1)).squeeze(1)
        beta = self.log_beta.exp().clamp(0.05, 30.0)
        return beta * (phi(ei, ej) - phi(ej, ei))

    def blank_consolidation(self, state: State, subject_z: torch.Tensor, eta_scale: float) -> tuple[State, torch.Tensor]:
        cfg = self.config
        if not cfg.use_rnn or cfg.consolidation_steps <= 0:
            return state, torch.zeros(state.item_state.shape[0], device=DEVICE)
        # Use pooled item state as an internally generated blank/retrieval context.
        pooled = state.item_state.mean(dim=1)
        zeros_rel = torch.zeros(state.item_state.shape[0], self.rel_feat_dim, device=DEVICE)
        parts = [pooled, pooled, torch.zeros_like(pooled), pooled * pooled, zeros_rel]
        if cfg.use_subject_modulation:
            parts.append(subject_z)
        x = torch.cat(parts, dim=1)
        eta_vals = []
        for _ in range(cfg.consolidation_steps):
            state, eta = self.rnn_step(x, state, subject_z, eta_scale)
            eta_vals.append(eta)
        return state, torch.stack(eta_vals).mean() if eta_vals else torch.zeros((), device=DEVICE)


def run_learning_phase(config: Config, net: ImplicitPlasticRanker, subject_z: torch.Tensor,
                       true_scores: torch.Tensor, learning_edges: torch.Tensor,
                       train_mode: bool) -> tuple[State, PhaseStats]:
    bsz, n_edges, _ = learning_edges.shape
    state = net.initial_state(bsz, subject_z)
    gate_vals, eta_vals, plastic_vals = [], [], []
    for _block in range(config.n_learning_blocks):
        orders = torch.stack([torch.randperm(n_edges, device=DEVICE) for _ in range(bsz)], dim=0)
        bidx = torch.arange(bsz, device=DEVICE)
        for k in range(n_edges):
            edge_idx = orders[:, k]
            item_i = learning_edges[bidx, edge_idx, 0]
            item_j = learning_edges[bidx, edge_idx, 1]
            rel = gather_true_diff(true_scores, item_i, item_j)
            noise_std = config.relation_noise if train_mode else config.eval_relation_noise
            if noise_std > 0:
                rel = (rel + noise_std * torch.randn_like(rel)).clamp(-1.0, 1.0)
            dropout = config.edge_dropout if train_mode else config.eval_edge_dropout
            if dropout > 0:
                keep = (torch.rand_like(rel) >= dropout).float()
                rel = keep * rel
            state, gate, eta = net.observe_pair(
                state, subject_z, item_i, item_j, rel, phase="learn",
                eta_scale=1.0, item_update_scale=config.item_update_scale,
            )
            gate_vals.append(gate.mean())
            eta_vals.append(eta.mean())
            plastic_vals.append(state.plastic.abs().mean())
        state, c_eta = net.blank_consolidation(state, subject_z, eta_scale=config.consolidation_eta_scale)
        if torch.is_tensor(c_eta) and c_eta.numel() > 0:
            eta_vals.append(c_eta.mean())
    zero = torch.zeros((), device=DEVICE)
    stats = PhaseStats(
        mean_gate=torch.stack(gate_vals).mean() if gate_vals else zero,
        mean_eta=torch.stack(eta_vals).mean() if eta_vals else zero,
        mean_abs_plastic=torch.stack(plastic_vals).mean() if plastic_vals else zero,
        state_norm=state.item_state.pow(2).mean().sqrt(),
        plastic_l2=state.plastic.pow(2).mean(),
    )
    return state, stats


def run_no_feedback_test(config: Config, net: ImplicitPlasticRanker, state: State, subject_z: torch.Tensor,
                         true_scores: torch.Tensor, pairs: torch.Tensor, train_mode: bool,
                         repetitions: int, learned_sets: Optional[list[set[tuple[int, int]]]] = None,
                         sample_choices: bool = False) -> dict:
    bsz, n_pairs, _ = pairs.shape
    bidx = torch.arange(bsz, device=DEVICE)
    choice_losses, entropies, corrects, learned_corrects, nonlearned_corrects = [], [], [], [], []
    pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    label_one_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    gate_vals, eta_vals, plastic_vals = [], [], []
    for rep in range(repetitions):
        if config.test_order_shuffle:
            orders = torch.stack([torch.randperm(n_pairs, device=DEVICE) for _ in range(bsz)], dim=0)
        else:
            orders = torch.arange(n_pairs, device=DEVICE)[None, :].expand(bsz, -1)
        for k in range(n_pairs):
            pidx = orders[:, k]
            item_i = pairs[bidx, pidx, 0]
            item_j = pairs[bidx, pidx, 1]
            label = pair_labels(true_scores, item_i, item_j)
            label_one_sum[bidx, pidx] += label
            logit = net.comparator_logit(state, subject_z, item_i, item_j)
            loss = F.binary_cross_entropy_with_logits(logit.float(), label.float(), reduction="none")
            choice_losses.append(loss.mean())
            prob = torch.sigmoid(logit.float())
            entropies.append(binary_entropy_from_logits(logit).mean())
            if sample_choices:
                choice = torch.bernoulli(prob).float()
                correct = (choice == label).float()
                pseudo_sign = choice * 2.0 - 1.0
            else:
                correct = ((prob >= 0.5).float() == label).float()
                pseudo_sign = torch.tanh(logit.float())
            pair_correct_sum[bidx, pidx] += correct
            corrects.append(correct.mean())
            if learned_sets is not None:
                learned_mask = torch.tensor([tuple(sorted((int(item_i[b].item()), int(item_j[b].item())))) in learned_sets[b] for b in range(bsz)], device=DEVICE)
                if learned_mask.any():
                    learned_corrects.append(correct[learned_mask].mean())
                if (~learned_mask).any():
                    nonlearned_corrects.append(correct[~learned_mask].mean())
            confidence = torch.tanh(logit.float()).abs().clamp(0.0, 1.0).pow(config.test_pseudo_confidence_power)
            pseudo_relation = pseudo_sign * confidence
            if config.detach_test_pseudo or not train_mode:
                pseudo_relation = pseudo_relation.detach()
                confidence = confidence.detach()
            if config.test_update_mode != "frozen":
                # No ground-truth feedback: relation is generated from the model's own current choice/logit.
                upd_scale = 0.0 if config.test_update_mode == "hidden" else config.test_item_update_scale
                state, gate, eta = net.observe_pair(
                    state, subject_z, item_i, item_j, pseudo_relation, phase="test",
                    eta_scale=config.test_eta_scale, item_update_scale=upd_scale, confidence=confidence,
                )
                gate_vals.append(gate.mean())
                eta_vals.append(eta.mean())
                plastic_vals.append(state.plastic.abs().mean())
    zero = torch.zeros((), device=DEVICE)
    return {
        "state": state,
        "choice_loss": torch.stack(choice_losses).mean() if choice_losses else zero,
        "entropy": torch.stack(entropies).mean() if entropies else zero,
        "acc": torch.stack(corrects).mean() if corrects else zero,
        "learned_acc": torch.stack(learned_corrects).mean() if learned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "nonlearned_acc": torch.stack(nonlearned_corrects).mean() if nonlearned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "mean_gate": torch.stack(gate_vals).mean() if gate_vals else zero,
        "mean_eta": torch.stack(eta_vals).mean() if eta_vals else zero,
        "mean_abs_plastic": torch.stack(plastic_vals).mean() if plastic_vals else state.plastic.abs().mean(),
        "pair_correct_rate": pair_correct_sum / max(1, repetitions),
        "label_one_rate": label_one_sum / max(1, repetitions),
    }


def run_training_episode(config: Config, net: ImplicitPlasticRanker) -> EpisodeStats:
    bsz = config.bs
    subject_z = sample_subject_latents(bsz, config.subject_dim)
    r2i = rank_to_item_maps(bsz, config.n_items)
    true_scores = true_scores_by_item(r2i, config)
    rank_edges_per_batch = [sample_sparse_rank_graph(config) for _ in range(bsz)]
    learning_edges = torch.empty(bsz, config.n_learning_pairs, 2, dtype=torch.long, device=DEVICE)
    for b, rank_edges in enumerate(rank_edges_per_batch):
        learning_edges[b] = rank_edges_to_item_edges(rank_edges, r2i[b:b+1], random_orientation=True)[0]
    state, learn_stats = run_learning_phase(config, net, subject_z, true_scores, learning_edges, train_mode=True)
    pairs = all_item_pairs(r2i, config)
    if config.train_query_random_orientation:
        pairs = randomize_query_pair_orientation(pairs)
    learned_sets = [set(tuple(sorted(map(int, edge.tolist()))) for edge in learning_edges[b]) for b in range(bsz)]
    test = run_no_feedback_test(
        config, net, state, subject_z, true_scores, pairs, train_mode=True,
        repetitions=max(1, config.train_test_repetitions), learned_sets=learned_sets, sample_choices=False,
    )
    final_state = test["state"]
    state_norm_loss = final_state.item_state.pow(2).mean()
    plastic_loss = learn_stats.plastic_l2 + final_state.plastic.pow(2).mean()
    eta_loss = learn_stats.mean_eta.pow(2) + test["mean_eta"].pow(2)
    gate_loss = learn_stats.mean_gate + test["mean_gate"]
    loss = (
        test["choice_loss"]
        + config.lambda_entropy * test["entropy"]
        + config.lambda_state_norm * state_norm_loss
        + config.lambda_plastic_l2 * plastic_loss
        + config.lambda_eta_l2 * eta_loss
        + config.lambda_gate_l1 * gate_loss
    )
    assert_finite_loss(loss, {
        "choice_loss": test["choice_loss"],
        "entropy": test["entropy"],
        "state_norm_loss": state_norm_loss,
        "plastic_loss": plastic_loss,
        "eta_loss": eta_loss,
        "gate_loss": gate_loss,
        "lambda_entropy": config.lambda_entropy,
    })
    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        choice_loss=float(test["choice_loss"].detach()),
        entropy_loss=float(test["entropy"].detach()),
        acc=float(test["acc"].detach()),
        learned_acc=float(test["learned_acc"].detach()) if torch.isfinite(test["learned_acc"]) else float("nan"),
        nonlearned_acc=float(test["nonlearned_acc"].detach()) if torch.isfinite(test["nonlearned_acc"]) else float("nan"),
        mean_gate_learning=float(learn_stats.mean_gate.detach()),
        mean_eta_learning=float(learn_stats.mean_eta.detach()),
        mean_abs_plastic_learning=float(learn_stats.mean_abs_plastic.detach()),
        mean_gate_test=float(test["mean_gate"].detach()),
        mean_eta_test=float(test["mean_eta"].detach()),
        mean_abs_plastic_test=float(test["mean_abs_plastic"].detach()),
        state_norm=float(final_state.item_state.pow(2).mean().sqrt().detach()),
    )


def kendall_tau_order(order_a: Iterable[int], order_b: Iterable[int]) -> float:
    a = list(order_a); b = list(order_b)
    pos_a = {item: idx for idx, item in enumerate(a)}
    pos_b = {item: idx for idx, item in enumerate(b)}
    concordant = discordant = 0
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


def choice_order_from_pair_acc(pair_acc: np.ndarray, pair_list: list[tuple[int, int]], n_items: int) -> list[int]:
    wins = np.zeros(n_items, dtype=float)
    for idx, (a, b) in enumerate(pair_list):
        # pair_acc is correctness for true b>a in paper task. For paper identity ranks, true b>a for all listed rank pairs.
        if pair_acc[idx] >= 0.5:
            wins[b] += 1.0
        else:
            wins[a] += 1.0
    return list(np.argsort(wins))


def pc1_order_and_tau(item_state: np.ndarray, reference_order: list[int]) -> tuple[list[int], float]:
    X = item_state - item_state.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        pc = X @ vt[0]
    except np.linalg.LinAlgError:
        pc = X[:, 0]
    order1 = list(np.argsort(pc))
    order2 = list(np.argsort(-pc))
    tau1 = kendall_tau_order(order1, reference_order)
    tau2 = kendall_tau_order(order2, reference_order)
    return (order1, tau1) if abs(tau1) >= abs(tau2) else (order2, tau2)


@torch.no_grad()
def evaluate_paper_task(config: Config, net: ImplicitPlasticRanker) -> dict:
    net.eval()
    old_beta = None
    if config.eval_beta_override and config.eval_beta_override > 0:
        old_beta = net.log_beta.detach().clone()
        net.log_beta.data.fill_(math.log(config.eval_beta_override))
    n = config.eval_subjects
    subject_z = sample_subject_latents(n, config.subject_dim)
    r2i = torch.arange(config.n_items, device=DEVICE)[None, :].expand(n, -1).contiguous()
    true_scores = true_scores_by_item(r2i, config)
    learning_edges = rank_edges_to_item_edges(PAPER_LEARNING_PAIRS_RANK, r2i, random_orientation=False)
    state, learn_stats = run_learning_phase(config, net, subject_z, true_scores, learning_edges, train_mode=False)
    pairs = torch.empty(n, len(config.all_rank_pairs), 2, dtype=torch.long, device=DEVICE)
    for idx, (a, b) in enumerate(config.all_rank_pairs):
        pairs[:, idx, 0] = a
        pairs[:, idx, 1] = b
    if config.eval_query_random_orientation:
        pairs = randomize_query_pair_orientation(pairs)
    test = run_no_feedback_test(
        config, net, state, subject_z, true_scores, pairs, train_mode=False,
        repetitions=config.eval_repetitions, learned_sets=None, sample_choices=config.eval_sample_choices,
    )
    pair_acc = test["pair_correct_rate"].cpu().numpy()
    item_state_np = test["state"].item_state.detach().cpu().numpy()
    pair_list = config.all_rank_pairs
    paper_pair_set = {tuple(p) for p in PAPER_LEARNING_PAIRS_RANK}
    learned_idx = np.array([idx for idx, p in enumerate(pair_list) if p in paper_pair_set], dtype=int)
    nonlearned_idx = np.array([idx for idx, p in enumerate(pair_list) if p not in paper_pair_set], dtype=int)
    pair_error_prop = 1.0 - pair_acc
    self_cons, circulars, choice_orders = [], [], []
    pc1_tau_true, pc1_tau_choice, subject_orders_pc1 = [], [], []
    true_order = list(range(config.n_items))
    for b in range(n):
        sc, circ = self_consistency_from_majority(pair_acc[b], pair_list, config.n_items)
        self_cons.append(sc); circulars.append(circ)
        c_order = choice_order_from_pair_acc(pair_acc[b], pair_list, config.n_items)
        choice_orders.append(c_order)
        pc_order, tau_true = pc1_order_and_tau(item_state_np[b], true_order)
        subject_orders_pc1.append(pc_order)
        pc1_tau_true.append(tau_true)
        pc1_tau_choice.append(kendall_tau_order(pc_order, c_order))
    taus_choice = [kendall_tau_order(choice_orders[i], choice_orders[j]) for i, j in combinations(range(n), 2)]
    taus_pc1 = [kendall_tau_order(subject_orders_pc1[i], subject_orders_pc1[j]) for i, j in combinations(range(n), 2)]
    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())
    summary = {
        "eval_subjects": n,
        "eval_repetitions": config.eval_repetitions,
        "model_type": "implicit_plastic_rnn_no_scores_no_edge_table",
        "use_rnn": config.use_rnn,
        "use_hebbian": config.use_hebbian,
        "use_subject_modulation": config.use_subject_modulation,
        "use_reliability_gate": config.use_reliability_gate,
        "test_update_mode": config.test_update_mode,
        "train_query_random_orientation": config.train_query_random_orientation,
        "eval_query_random_orientation": config.eval_query_random_orientation,
        "relation_input": config.relation_input,
        "mean_true_label_is_class1_j_higher": float(test["label_one_rate"].detach().cpu().numpy().mean()),
        "overall_accuracy": float(pair_acc.mean()),
        "learned_pairs_accuracy": float(pair_acc[:, learned_idx].mean()),
        "nonlearned_pairs_accuracy": float(pair_acc[:, nonlearned_idx].mean()),
        "consistent_error_subjects_80pct": int((pair_error_prop >= 0.8).any(axis=1).sum()),
        "consistent_error_subjects_80pct_ratio": float((pair_error_prop >= 0.8).any(axis=1).mean()),
        "consistent_error_subjects_100pct": int((pair_error_prop >= 1.0).any(axis=1).sum()),
        "consistent_error_subjects_100pct_ratio": float((pair_error_prop >= 1.0).any(axis=1).mean()),
        "mean_self_consistency_from_majority_choices": float(np.mean(self_cons)),
        "mean_circular_triads_from_majority_choices": float(np.mean(circulars)),
        "mean_inter_subject_kendall_tau_choice_orders": float(np.mean(taus_choice)) if taus_choice else 0.0,
        "mean_inter_subject_kendall_tau_pc1_orders": float(np.mean(taus_pc1)) if taus_pc1 else 0.0,
        "pc1_rank_axis_tau_true_mean_abs": float(np.mean(np.abs(pc1_tau_true))),
        "pc1_rank_axis_tau_choice_mean_abs": float(np.mean(np.abs(pc1_tau_choice))),
        "distance_accuracy": distance_acc,
        "mean_gate_learning": float(learn_stats.mean_gate.detach().cpu()),
        "mean_hebb_eta_learning": float(learn_stats.mean_eta.detach().cpu()),
        "mean_abs_plastic_learning": float(learn_stats.mean_abs_plastic.detach().cpu()),
        "mean_gate_test": float(test["mean_gate"].detach().cpu()),
        "mean_hebb_eta_test": float(test["mean_eta"].detach().cpu()),
        "mean_abs_plastic_test": float(test["mean_abs_plastic"].detach().cpu()),
        "mean_item_state_norm": float(test["state"].item_state.pow(2).mean().sqrt().detach().cpu()),
        "choice_beta": float(net.log_beta.exp().detach().cpu()),
    }
    if old_beta is not None:
        net.log_beta.data.copy_(old_beta)
    return summary


def train(config: Config) -> ImplicitPlasticRanker:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_implicit.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    net = ImplicitPlasticRanker(config).to(DEVICE)
    if config.compile_model and hasattr(torch, "compile"):
        log("[setup] torch.compile enabled")
        net = torch.compile(net)  # type: ignore[assignment]
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2)
    use_amp = bool(config.amp and DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log(f"[setup] Device={DEVICE}; amp={use_amp}; output={out}")
    log(f"[setup] implicit model: no scalar score state, no explicit edge-memory table, no 8! posterior")
    log(f"[setup] use_rnn={config.use_rnn}; use_hebbian={config.use_hebbian}; test_update_mode={config.test_update_mode}")
    log(f"[setup] bs={config.bs}; nbiter={config.nbiter}; item_state_dim={config.item_state_dim}; rnn_hidden={config.rnn_hidden_size}")
    log(f"[setup] parameters={sum(p.numel() for p in net.parameters())}")
    log("[task] inner loop: passive observed relations only; test: no feedback, internal self-updates only")
    fields = [
        "episode", "loss", "choice_loss", "entropy_loss", "acc", "learned_acc", "nonlearned_acc",
        "mean_gate_learning", "mean_eta_learning", "mean_abs_plastic_learning",
        "mean_gate_test", "mean_eta_test", "mean_abs_plastic_test", "state_norm",
    ]
    with open(out / "train_log.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    start = time.time()
    last = None
    iterator = tqdm(range(config.nbiter), desc="training episodes", unit="episode", dynamic_ncols=True, file=sys.stdout)
    for ep in iterator:
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            stats = run_training_episode(config, net)
        scaler.scale(stats.loss).backward()
        if config.gc > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        scaler.step(opt); scaler.update()
        last = stats
        with open(out / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                ep, stats.loss_value, stats.choice_loss, stats.entropy_loss, stats.acc, stats.learned_acc, stats.nonlearned_acc,
                stats.mean_gate_learning, stats.mean_eta_learning, stats.mean_abs_plastic_learning,
                stats.mean_gate_test, stats.mean_eta_test, stats.mean_abs_plastic_test, stats.state_norm,
            ])
        if ep % config.pe == 0 or ep == config.nbiter - 1:
            elapsed = time.time() - start
            log(
                f"Episode {ep} ==== {elapsed:.2f}s | loss={stats.loss_value:.4f} choice={stats.choice_loss:.4f} "
                f"acc={stats.acc:.3f} learned={stats.learned_acc:.3f} nonlearned={stats.nonlearned_acc:.3f} | "
                f"gateL={stats.mean_gate_learning:.3f} etaL={stats.mean_eta_learning:.4f} |pL|={stats.mean_abs_plastic_learning:.4f} "
                f"gateT={stats.mean_gate_test:.3f} etaT={stats.mean_eta_test:.4f} |state|={stats.state_norm:.3f}"
            )
            start = time.time()
        if config.save_every > 0 and ep > 0 and ep % config.save_every == 0:
            torch.save({"model_state": net.state_dict(), "config": asdict(config)}, out / "net_implicit.pt")
            log(f"[save] checkpoint: {out / 'net_implicit.pt'}")
    torch.save({"model_state": net.state_dict(), "config": asdict(config)}, out / "net_implicit.pt")
    if last is not None:
        log(f"[done] final train acc={last.acc:.3f}, loss={last.loss_value:.4f}")
    return net


def configure_device(config: Config) -> None:
    global DEVICE
    if config.num_threads and config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(config.num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(config.num_threads))
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        DEVICE = torch.device("cuda:0")
    elif config.device == "cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Implicit plastic RNN ranker: no explicit scores, no edge table.")
    p.add_argument("--rngseed", type=int, default=1)
    p.add_argument("--nbiter", type=int, default=200)
    p.add_argument("--batch-size", "--bs", dest="bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--grad-clip", "--gc", dest="gc", type=float, default=2.0)
    p.add_argument("--print-every", "--pe", dest="pe", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default="outputs_implicit_plastic")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--num-threads", type=int, default=1)

    p.add_argument("--n-learning-blocks", type=int, default=4)
    p.add_argument("--paper-graph-train-prob", type=float, default=0.3)
    p.add_argument("--relation-noise", type=float, default=0.03)
    p.add_argument("--eval-relation-noise", type=float, default=0.0)
    p.add_argument("--edge-dropout", type=float, default=0.0)
    p.add_argument("--eval-edge-dropout", type=float, default=0.0)
    p.add_argument("--relation-input", choices=["magnitude", "sign", "raw_bars"], default="magnitude")

    p.add_argument("--item-input-dim", type=int, default=32)
    p.add_argument("--item-state-dim", type=int, default=64)
    p.add_argument("--subject-dim", type=int, default=16)
    p.add_argument("--hidden-size", type=int, default=192)
    p.add_argument("--rnn-hidden-size", type=int, default=128)

    p.add_argument("--use-rnn", dest="use_rnn", action="store_true")
    p.add_argument("--no-rnn", dest="use_rnn", action="store_false")
    p.set_defaults(use_rnn=True)
    p.add_argument("--use-hebbian", dest="use_hebbian", action="store_true")
    p.add_argument("--no-hebbian", dest="use_hebbian", action="store_false")
    p.set_defaults(use_hebbian=True)
    p.add_argument("--use-subject-modulation", dest="use_subject_modulation", action="store_true")
    p.add_argument("--no-subject-modulation", dest="use_subject_modulation", action="store_false")
    p.set_defaults(use_subject_modulation=True)
    p.add_argument("--use-reliability-gate", dest="use_reliability_gate", action="store_true")
    p.add_argument("--no-reliability-gate", dest="use_reliability_gate", action="store_false")
    p.set_defaults(use_reliability_gate=True)
    p.add_argument("--use-item-state-updates", dest="use_item_state_updates", action="store_true")
    p.add_argument("--no-item-state-updates", dest="use_item_state_updates", action="store_false")
    p.set_defaults(use_item_state_updates=True)

    p.add_argument("--rnn-input-gain", type=float, default=1.0)
    p.add_argument("--rnn-rec-gain", type=float, default=1.0)
    p.add_argument("--fast-weight-gain", type=float, default=0.45)
    p.add_argument("--hebb-eta", type=float, default=0.08)
    p.add_argument("--plastic-decay", type=float, default=0.97)
    p.add_argument("--plastic-clip", type=float, default=3.0)
    p.add_argument("--detach-plastic-state", action="store_true")
    p.add_argument("--rnn-dropout", type=float, default=0.0)
    p.add_argument("--item-state-decay", type=float, default=0.0)
    p.add_argument("--item-update-scale", type=float, default=0.35)
    p.add_argument("--item-norm-clip", type=float, default=5.0)
    p.add_argument("--consolidation-steps", type=int, default=0)
    p.add_argument("--consolidation-eta-scale", type=float, default=0.25)

    p.add_argument("--test-update-mode", choices=["frozen", "hidden", "self_reconsolidate"], default="self_reconsolidate")
    p.add_argument("--train-test-repetitions", type=int, default=1)
    p.add_argument("--eval-repetitions", type=int, default=10)
    p.add_argument("--test-order-shuffle", dest="test_order_shuffle", action="store_true")
    p.add_argument("--no-test-order-shuffle", dest="test_order_shuffle", action="store_false")
    p.set_defaults(test_order_shuffle=True)
    p.add_argument("--train-query-random-orientation", dest="train_query_random_orientation", action="store_true")
    p.add_argument("--no-train-query-random-orientation", dest="train_query_random_orientation", action="store_false")
    p.set_defaults(train_query_random_orientation=True)
    p.add_argument("--eval-query-random-orientation", dest="eval_query_random_orientation", action="store_true")
    p.add_argument("--no-eval-query-random-orientation", dest="eval_query_random_orientation", action="store_false")
    p.set_defaults(eval_query_random_orientation=True)
    p.add_argument("--test-eta-scale", type=float, default=0.35)
    p.add_argument("--test-item-update-scale", type=float, default=0.10)
    p.add_argument("--test-pseudo-confidence-power", type=float, default=1.0)
    p.add_argument("--detach-test-pseudo", action="store_true")
    p.add_argument("--eval-sample-choices", dest="eval_sample_choices", action="store_true")
    p.add_argument("--no-eval-sample-choices", dest="eval_sample_choices", action="store_false")
    p.set_defaults(eval_sample_choices=True)

    p.add_argument("--choice-beta-init", type=float, default=2.0)
    p.add_argument("--eval-beta-override", type=float, default=0.0)
    p.add_argument("--lambda-state-norm", type=float, default=1e-4)
    p.add_argument("--lambda-plastic-l2", type=float, default=1e-5)
    p.add_argument("--lambda-eta-l2", type=float, default=1e-4)
    p.add_argument("--lambda-gate-l1", type=float, default=0.0)
    p.add_argument("--lambda-entropy", type=float, default=0.002)
    p.add_argument("--eval-subjects", type=int, default=128)
    args = p.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    configure_device(config)
    random.seed(config.rngseed)
    np.random.seed(config.rngseed)
    torch.manual_seed(config.rngseed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(config.rngseed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log(f"[setup] seed={config.rngseed}; torch_num_threads={torch.get_num_threads()}")
    net = train(config)
    summary = evaluate_paper_task(config, net)
    out = Path(config.output_dir)
    with open(out / "paper_task_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("[eval] behavioral-paper task summary:")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
