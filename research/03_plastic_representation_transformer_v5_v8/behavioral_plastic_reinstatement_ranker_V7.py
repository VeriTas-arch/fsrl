"""
V7 behavioral plastic-reinstatement ranker.

Goal
----
This version keeps the behavioral-paper task format (fixed sparse non-adjacent
learning graph, four learning blocks, no-feedback all-pair testing), but changes
the mechanism toward the Miconi/Kay-style plastic representation solution.

Important design choices
------------------------
1. No explicit scalar item scores.
2. No explicit edge-memory table.
3. No direct high-capacity item-state writer.
4. Episode-local memory is carried mainly by recurrent plastic weights P(t),
   gated by self-generated neuromodulatory signals.
5. Pair decisions are made by comparing plastic-transformed single-item
   representations, so plasticity must shape item representations rather than
   directly writing labels into a table.
6. The V5 free reinstatement path is kept optional, but the default learning
   scaffold uses only an observed-pair replay buffer: previously observed
   learning relations can be internally replayed through the same plastic
   controller. No unobserved test-pair labels are ever replayed.
7. Learning-pair auxiliary and weak margin losses are applied only to relations
   actually shown during the learning phase. They provide a local gradient for
   the plastic rank-axis without leaking answers for unobserved test pairs.

Outer-loop training is still supervised by the true ranking for the final
no-feedback test choices. True labels are never used for no-feedback test
updates, and learning-phase auxiliary targets are restricted to observed
learning relations.
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
    output_dir: str = "outputs_behavioral_plastic_v7"
    device: str = "auto"
    amp: bool = False
    compile_model: bool = False
    num_threads: int = 1

    # task
    n_items: int = 8
    n_learning_pairs: int = 8
    n_learning_blocks: int = 4
    paper_graph_train_prob: float = 1.0
    relation_noise: float = 0.03
    eval_relation_noise: float = 0.0
    edge_dropout: float = 0.0
    eval_edge_dropout: float = 0.0
    relation_input: str = "magnitude"  # magnitude|sign

    # representation sizes
    item_input_dim: int = 32
    item_code_dim: int = 128
    subject_dim: int = 16
    hidden_size: int = 192
    rnn_hidden_size: int = 128

    # mechanisms
    use_rnn: bool = True
    use_hebbian: bool = True
    use_subject_modulation: bool = True
    use_hidden_readout: bool = True
    hidden_readout_gain: float = 0.05

    # recurrent/plastic dynamics
    rnn_input_gain: float = 1.0
    rnn_rec_gain: float = 1.0
    fast_weight_gain: float = 0.50
    hebb_eta: float = 0.08
    plastic_decay: float = 0.97
    trace_decay: float = 0.70
    plastic_clip: float = 3.0
    detach_plastic_state: bool = False
    rnn_dropout: float = 0.0
    neuromod_clip: float = 1.0

    # active reinstatement / internal replay
    reinstatement_steps: int = 0
    reinstatement_eta_scale: float = 0.30
    test_reinstatement_steps: int = 0
    replay_temperature: float = 1.0
    observed_replay_steps: int = 1
    observed_replay_eta_scale: float = 0.50
    replay_aux_scale: float = 1.00

    # test phase
    test_update_mode: str = "frozen"  # frozen|hidden|self_reconsolidate
    train_test_repetitions: int = 1
    eval_repetitions: int = 10
    test_order_shuffle: bool = True
    train_query_random_orientation: bool = True
    eval_query_random_orientation: bool = True
    test_eta_scale: float = 0.35
    test_pseudo_confidence_power: float = 1.0
    detach_test_pseudo: bool = False
    eval_sample_choices: bool = True

    # readout/loss
    choice_beta_init: float = 2.0
    eval_beta_override: float = 0.0
    lambda_plastic_l2: float = 1e-5
    lambda_trace_l2: float = 1e-6
    lambda_eta_l2: float = 1e-4
    lambda_entropy: float = 0.001
    lambda_score_var: float = 0.0
    lambda_learning_aux: float = 1.00
    lambda_axis_margin: float = 0.10
    learning_margin: float = 0.20
    lambda_score_std_floor: float = 0.0
    score_std_floor: float = 0.05

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
    h: torch.Tensor        # [B,H]
    plastic: torch.Tensor  # [B,H,H]
    trace: torch.Tensor    # [B,H,H]


@dataclass
class PhaseStats:
    mean_mod: torch.Tensor
    mean_eta_abs: torch.Tensor
    mean_abs_plastic: torch.Tensor
    mean_abs_trace: torch.Tensor
    plastic_l2: torch.Tensor
    trace_l2: torch.Tensor
    aux_loss: torch.Tensor
    margin_loss: torch.Tensor
    aux_acc: torch.Tensor
    observed_score_std: torch.Tensor


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    choice_loss: float
    entropy_loss: float
    acc: float
    learned_acc: float
    nonlearned_acc: float
    mean_mod_learning: float
    mean_eta_abs_learning: float
    mean_abs_plastic_learning: float
    mean_mod_test: float
    mean_eta_abs_test: float
    mean_abs_plastic_test: float
    final_score_std: float
    learning_aux_loss: float
    learning_margin_loss: float
    learning_aux_acc: float
    observed_score_std: float


def sample_subject_latents(batch_size: int, subject_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, subject_dim, device=DEVICE)


def rank_to_item_maps(batch_size: int, n_items: int) -> torch.Tensor:
    return torch.stack([torch.randperm(n_items, device=DEVICE) for _ in range(batch_size)], dim=0)


def true_scores_by_item(rank_to_item: torch.Tensor, config: Config) -> torch.Tensor:
    values = config.rank_values.to(DEVICE)
    bsz = rank_to_item.shape[0]
    scores = torch.empty(bsz, config.n_items, device=DEVICE)
    bidx = torch.arange(bsz, device=DEVICE)
    for rank in range(config.n_items):
        scores[bidx, rank_to_item[:, rank]] = values[rank]
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
    rank_edges_tensor = torch.tensor(rank_edges, dtype=torch.long, device=DEVICE)
    rank_batch = rank_edges_tensor[None, :, :].expand(bsz, -1, -1)
    item_i = torch.gather(rank_to_item, 1, rank_batch[:, :, 0])
    item_j = torch.gather(rank_to_item, 1, rank_batch[:, :, 1])
    edges = torch.stack([item_i, item_j], dim=2)
    if random_orientation:
        edges = randomize_pair_orientation(edges)
    return edges


def randomize_pair_orientation(pairs: torch.Tensor) -> torch.Tensor:
    swap = torch.rand(pairs.shape[0], pairs.shape[1], device=DEVICE) < 0.5
    first = pairs[:, :, 0].clone()
    second = pairs[:, :, 1].clone()
    out = pairs.clone()
    out[:, :, 0] = torch.where(swap, second, first)
    out[:, :, 1] = torch.where(swap, first, second)
    return out


def all_item_pairs(rank_to_item: torch.Tensor, config: Config, random_orientation: bool = False) -> torch.Tensor:
    rank_pairs = torch.tensor(config.all_rank_pairs, dtype=torch.long, device=DEVICE)
    bsz = rank_to_item.shape[0]
    rank_batch = rank_pairs[None, :, :].expand(bsz, -1, -1)
    item_i = torch.gather(rank_to_item, 1, rank_batch[:, :, 0])
    item_j = torch.gather(rank_to_item, 1, rank_batch[:, :, 1])
    pairs = torch.stack([item_i, item_j], dim=2)
    if random_orientation:
        pairs = randomize_pair_orientation(pairs)
    return pairs


def gather_true_diff(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    bidx = torch.arange(true_scores.shape[0], device=DEVICE)
    return true_scores[bidx, item_j] - true_scores[bidx, item_i]


def pair_labels(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    return (gather_true_diff(true_scores, item_i, item_j) > 0).float()


def learned_pair_mask_from_edges(learning_edges: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    le = torch.sort(learning_edges, dim=2).values
    pp = torch.sort(pairs, dim=2).values
    return ((pp[:, :, None, :] == le[:, None, :, :]).all(dim=3)).any(dim=2)


class PlasticReinstatementRanker(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        N = config.n_items
        D0 = config.item_input_dim
        H = config.rnn_hidden_size
        S = config.subject_dim
        M = config.hidden_size

        self.base_item = nn.Parameter(torch.randn(N, D0) / math.sqrt(D0))
        item_in = D0 + (S if config.use_subject_modulation else 0)
        self.item_encoder = nn.Sequential(nn.Linear(item_in, M), nn.Tanh(), nn.Linear(M, H), nn.LayerNorm(H))
        self.subject_to_item_bias = nn.Linear(S, H) if config.use_subject_modulation else None
        self.h_init = nn.Sequential(nn.Linear(S, H), nn.Tanh(), nn.Linear(H, H)) if config.use_subject_modulation else None

        # Plastic recurrent representation transformer.
        self.item_self = nn.Linear(H, H, bias=False)
        self.rec_item = nn.Parameter((1.0 / math.sqrt(H)) * (2.0 * torch.rand(H, H) - 1.0))
        self.alpha_item = nn.Parameter(0.01 * (2.0 * torch.rand(H, H) - 1.0))
        self.item_ln = nn.LayerNorm(H)

        # Recurrent controller for neuromodulated plasticity.
        self.rel_feat_dim = 6
        context_dim = 4 * H + self.rel_feat_dim + (S if config.use_subject_modulation else 0)
        self.context_proj = nn.Sequential(nn.Linear(context_dim, M), nn.Tanh(), nn.Linear(M, H))
        self.rnn_rec = nn.Linear(H, H, bias=False)
        self.rnn_ln = nn.LayerNorm(H)
        self.rnn_dropout = nn.Dropout(config.rnn_dropout)
        self.mod_head = nn.Sequential(nn.Linear(context_dim + H + (S if config.use_subject_modulation else 0), M), nn.Tanh(), nn.Linear(M, 1))

        # Active item reinstatement controller.
        self.replay_query = nn.Sequential(nn.Linear(H + (S if config.use_subject_modulation else 0), M), nn.Tanh(), nn.Linear(M, H))
        self.replay_context = nn.Sequential(nn.Linear(2 * H + (S if config.use_subject_modulation else 0), M), nn.Tanh(), nn.Linear(M, H))

        self.score_head = nn.Linear(H, 1)
        self.hidden_axis = nn.Linear(H + (S if config.use_subject_modulation else 0), H)
        self.log_beta = nn.Parameter(torch.tensor(math.log(config.choice_beta_init), dtype=torch.float32))

    def initial_state(self, bsz: int, subject_z: torch.Tensor) -> State:
        H = self.config.rnn_hidden_size
        if self.config.use_rnn:
            h = self.h_init(subject_z) if self.h_init is not None else torch.zeros(bsz, H, device=DEVICE)
        else:
            h = torch.zeros(bsz, H, device=DEVICE)
        plastic = torch.zeros(bsz, H, H, device=DEVICE)
        trace = torch.zeros_like(plastic)
        return State(h=h, plastic=plastic, trace=trace)

    def item_codes(self, bsz: int, subject_z: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        base = self.base_item[None, :, :].expand(bsz, -1, -1)
        if cfg.use_subject_modulation:
            subj = subject_z[:, None, :].expand(-1, cfg.n_items, -1)
            x = torch.cat([base, subj], dim=-1)
        else:
            x = base
        return self.item_encoder(x)

    def relation_features(self, relation: torch.Tensor, phase: str, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.config
        if confidence is None:
            confidence = torch.ones_like(relation)
        if cfg.relation_input == "sign":
            r = relation.sign()
            mag = torch.zeros_like(relation)
        else:
            r = relation
            mag = relation.abs()
        is_test = torch.ones_like(relation) if phase == "test" else torch.zeros_like(relation)
        is_replay = torch.ones_like(relation) if phase == "replay" else torch.zeros_like(relation)
        return torch.stack([r, mag, confidence, is_test, is_replay, torch.ones_like(relation)], dim=1)

    def transform_all_items(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        fixed = torch.matmul(codes, self.rec_item.t())
        if cfg.use_hebbian and cfg.use_rnn:
            # alpha_item gates which fast-weight entries affect item transformation.
            fast_matrix = state.plastic * self.alpha_item[None, :, :]
            fast = torch.bmm(codes, fast_matrix.transpose(1, 2))
        else:
            fast = torch.zeros_like(fixed)
        pre = self.item_self(codes) + cfg.rnn_rec_gain * fixed + cfg.fast_weight_gain * fast
        if self.subject_to_item_bias is not None:
            pre = pre + self.subject_to_item_bias(subject_z)[:, None, :]
        return torch.tanh(self.item_ln(pre))

    def item_scores(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor) -> torch.Tensor:
        psi = self.transform_all_items(codes, state, subject_z)
        scores = self.score_head(psi).squeeze(-1)
        if self.config.use_rnn and self.config.use_hidden_readout and self.config.hidden_readout_gain != 0:
            if self.config.use_subject_modulation:
                axis_in = torch.cat([state.h, subject_z], dim=1)
            else:
                axis_in = state.h
            axis = torch.tanh(self.hidden_axis(axis_in))
            scores = scores + self.config.hidden_readout_gain * (psi * axis[:, None, :]).sum(dim=2) / math.sqrt(psi.shape[-1])
        return scores - scores.mean(dim=1, keepdim=True)

    def comparator_logit(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
        bidx = torch.arange(codes.shape[0], device=DEVICE)
        scores = self.item_scores(codes, state, subject_z)
        beta = self.log_beta.exp().clamp(0.05, 30.0)
        return beta * (scores[bidx, item_j] - scores[bidx, item_i])

    def pair_context(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor,
                     relation: torch.Tensor, phase: str, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        bidx = torch.arange(codes.shape[0], device=DEVICE)
        psi = self.transform_all_items(codes, state, subject_z)
        pi = psi[bidx, item_i]
        pj = psi[bidx, item_j]
        rel = self.relation_features(relation, phase, confidence)
        parts = [pi, pj, pj - pi, pi * pj, rel]
        if self.config.use_subject_modulation:
            parts.append(subject_z)
        return torch.cat(parts, dim=1)

    def controller_step(self, context: torch.Tensor, state: State, subject_z: torch.Tensor, eta_scale: float) -> tuple[State, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if not cfg.use_rnn:
            z = torch.zeros(context.shape[0], device=DEVICE)
            return state, z, z
        h_prev = state.h.detach() if cfg.detach_plastic_state else state.h
        plastic_prev = state.plastic.detach() if cfg.detach_plastic_state else state.plastic
        fast = torch.bmm(plastic_prev, h_prev.unsqueeze(2)).squeeze(2)
        pre = cfg.rnn_input_gain * self.context_proj(context) + cfg.rnn_rec_gain * self.rnn_rec(state.h) + cfg.fast_weight_gain * fast
        h_new = torch.tanh(self.rnn_ln(pre))
        if self.training and cfg.rnn_dropout > 0:
            h_new = self.rnn_dropout(h_new)
        if cfg.use_subject_modulation:
            mod_in = torch.cat([context, h_new, subject_z], dim=1)
        else:
            mod_in = torch.cat([context, h_new], dim=1)
        raw_mod = torch.tanh(self.mod_head(mod_in).squeeze(1))
        if cfg.neuromod_clip > 0:
            raw_mod = raw_mod.clamp(-cfg.neuromod_clip, cfg.neuromod_clip)
        if not cfg.use_hebbian:
            mod = torch.zeros_like(raw_mod)
            trace_new = torch.zeros_like(state.trace)
            plastic_new = torch.zeros_like(state.plastic)
        else:
            mod = cfg.hebb_eta * eta_scale * raw_mod
            hebb = torch.bmm(h_new.unsqueeze(2), h_prev.unsqueeze(1))
            hebb = torch.tanh(hebb)
            trace_new = cfg.trace_decay * state.trace + (1.0 - cfg.trace_decay) * hebb
            plastic_new = cfg.plastic_decay * state.plastic + mod[:, None, None] * trace_new
            if cfg.plastic_clip > 0:
                plastic_new = plastic_new.clamp(-cfg.plastic_clip, cfg.plastic_clip)
        return State(h=h_new, plastic=plastic_new, trace=trace_new), mod, mod.abs()

    def observe_pair(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor,
                     relation: torch.Tensor, phase: str, eta_scale: float, confidence: Optional[torch.Tensor] = None) -> tuple[State, torch.Tensor, torch.Tensor]:
        context = self.pair_context(codes, state, subject_z, item_i, item_j, relation, phase, confidence)
        return self.controller_step(context, state, subject_z, eta_scale)

    def reinstate(self, codes: torch.Tensor, state: State, subject_z: torch.Tensor, steps: int, eta_scale: float) -> tuple[State, list[torch.Tensor], list[torch.Tensor]]:
        cfg = self.config
        mods: list[torch.Tensor] = []
        etas: list[torch.Tensor] = []
        if steps <= 0 or not cfg.use_rnn:
            return state, mods, etas
        for _ in range(steps):
            if cfg.use_subject_modulation:
                q_in = torch.cat([state.h, subject_z], dim=1)
            else:
                q_in = state.h
            query = self.replay_query(q_in)
            logits = (codes * query[:, None, :]).sum(dim=2) / max(1e-6, math.sqrt(codes.shape[-1]))
            attn = torch.softmax(logits / max(1e-3, cfg.replay_temperature), dim=1)
            retrieved = (attn[:, :, None] * codes).sum(dim=1)
            pooled = codes.mean(dim=1)
            if cfg.use_subject_modulation:
                ctx = self.replay_context(torch.cat([retrieved, pooled, subject_z], dim=1))
            else:
                ctx = self.replay_context(torch.cat([retrieved, pooled], dim=1))
            zero_rel = torch.zeros(codes.shape[0], device=DEVICE)
            rel = self.relation_features(zero_rel, phase="replay", confidence=torch.zeros_like(zero_rel))
            # Match context_dim: [pi, pj, diff, prod, rel, subject]. Use retrieved/pooled as pseudo-items.
            parts = [ctx, pooled, pooled - ctx, ctx * pooled, rel]
            if cfg.use_subject_modulation:
                parts.append(subject_z)
            full_ctx = torch.cat(parts, dim=1)
            state, mod, eta_abs = self.controller_step(full_ctx, state, subject_z, eta_scale)
            mods.append(mod.mean())
            etas.append(eta_abs.mean())
        return state, mods, etas


def run_learning_phase(config: Config, net: PlasticReinstatementRanker, subject_z: torch.Tensor, true_scores: torch.Tensor,
                       learning_edges: torch.Tensor, train_mode: bool) -> tuple[State, torch.Tensor, PhaseStats]:
    """Run the behaviorally grounded learning phase.

    V7 uses a predict -> observed feedback -> plastic update loop. For each
    shown learning pair, the model first predicts using only the current episode
    state. Only after that prediction is the actually observed relation revealed
    and passed to the recurrent plastic controller. Local prediction and margin
    losses are computed only for pairs that are actually shown during learning.

    The observed replay buffer stores only previously shown learning relations in
    the current episode. Replay also uses the same predict -> feedback/update
    loop. No unobserved all-pair test labels are inserted into the learning
    phase, replay buffer, or no-feedback test updates.
    """
    bsz, n_edges, _ = learning_edges.shape
    codes = net.item_codes(bsz, subject_z)
    state = net.initial_state(bsz, subject_z)
    mod_vals: list[torch.Tensor] = []
    eta_vals: list[torch.Tensor] = []
    plastic_vals: list[torch.Tensor] = []
    trace_vals: list[torch.Tensor] = []
    aux_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    aux_accs: list[torch.Tensor] = []
    observed_signed_logits: list[torch.Tensor] = []

    # Each entry is a batch tensor for one relation that was actually observed.
    # There is no entry for unobserved test pairs.
    observed_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    noise_std = config.relation_noise if train_mode else config.eval_relation_noise
    dropout = config.edge_dropout if train_mode else config.eval_edge_dropout
    bidx = torch.arange(bsz, device=DEVICE)

    def predict_observed_pair(item_i: torch.Tensor, item_j: torch.Tensor, relation: torch.Tensor, scale: float) -> None:
        """Predict a shown relation before feedback is applied.

        This is the local behavioral learning signal. It uses only a pair that has
        been presented in the learning phase. The target is the shown relation,
        not an unobserved test-pair answer.
        """
        if scale <= 0:
            return
        if config.lambda_learning_aux <= 0 and config.lambda_axis_margin <= 0:
            return
        label = (relation > 0).float()
        logit = net.comparator_logit(codes, state, subject_z, item_i, item_j).float()
        if config.lambda_learning_aux > 0:
            aux_losses.append(scale * F.binary_cross_entropy_with_logits(logit, label, reduction="none").mean())
            pred = (torch.sigmoid(logit) >= 0.5).float()
            aux_accs.append((pred == label).float().mean())
        if config.lambda_axis_margin > 0:
            signed_logit = logit * (label * 2.0 - 1.0)
            margin_losses.append(scale * F.relu(config.learning_margin - signed_logit).mean())
            observed_signed_logits.append(signed_logit.detach())

    def feedback_update(item_i: torch.Tensor, item_j: torch.Tensor, relation: torch.Tensor, phase: str, eta_scale: float) -> None:
        """Reveal the shown relation and apply neuromodulated plastic update."""
        nonlocal state
        state, mod, eta_abs = net.observe_pair(codes, state, subject_z, item_i, item_j, relation, phase=phase, eta_scale=eta_scale)
        mod_vals.append(mod.mean())
        eta_vals.append(eta_abs.mean())
        plastic_vals.append(state.plastic.abs().mean())
        trace_vals.append(state.trace.abs().mean())

    for _block in range(config.n_learning_blocks):
        orders = torch.argsort(torch.rand(bsz, n_edges, device=DEVICE), dim=1)
        for k in range(n_edges):
            edge_idx = orders[:, k]
            item_i = learning_edges[bidx, edge_idx, 0]
            item_j = learning_edges[bidx, edge_idx, 1]
            rel = gather_true_diff(true_scores, item_i, item_j)
            if noise_std > 0:
                rel = (rel + noise_std * torch.randn_like(rel)).clamp(-1.0, 1.0)
            if dropout > 0:
                keep = (torch.rand_like(rel) >= dropout).float()
                rel = keep * rel

            # Step 1: prediction before feedback. This is the critical V7 change.
            predict_observed_pair(item_i, item_j, rel, scale=1.0)

            # Step 2: observed feedback drives the recurrent/plastic update.
            feedback_update(item_i, item_j, rel, phase="learn", eta_scale=1.0)
            observed_buffer.append((item_i.detach(), item_j.detach(), rel.detach()))

            # Observed-pair replay/rehearsal. It is still behaviorally grounded:
            # replayed relations have all been shown earlier in this same episode.
            if config.observed_replay_steps > 0 and observed_buffer:
                for _ in range(config.observed_replay_steps):
                    rb = random.randrange(len(observed_buffer))
                    ri, rj, rr = observed_buffer[rb]
                    predict_observed_pair(ri, rj, rr, scale=config.replay_aux_scale)
                    feedback_update(ri, rj, rr, phase="replay", eta_scale=config.observed_replay_eta_scale)

            # Optional free-form reinstatement from earlier versions. The default
            # is zero; the behaviorally grounded replay above is the main route.
            state, rmods, retas = net.reinstate(codes, state, subject_z, steps=config.reinstatement_steps, eta_scale=config.reinstatement_eta_scale)
            mod_vals.extend(rmods)
            eta_vals.extend(retas)
            if rmods:
                plastic_vals.append(state.plastic.abs().mean())
                trace_vals.append(state.trace.abs().mean())

    zero = torch.zeros((), device=DEVICE)
    aux_loss = torch.stack(aux_losses).mean() if aux_losses else zero
    margin_loss = torch.stack(margin_losses).mean() if margin_losses else zero
    aux_acc = torch.stack(aux_accs).mean() if aux_accs else zero
    if observed_signed_logits:
        observed_score_std = torch.cat([x.reshape(-1) for x in observed_signed_logits]).std()
    else:
        observed_score_std = zero
    stats = PhaseStats(
        mean_mod=torch.stack(mod_vals).mean() if mod_vals else zero,
        mean_eta_abs=torch.stack(eta_vals).mean() if eta_vals else zero,
        mean_abs_plastic=torch.stack(plastic_vals).mean() if plastic_vals else state.plastic.abs().mean(),
        mean_abs_trace=torch.stack(trace_vals).mean() if trace_vals else state.trace.abs().mean(),
        plastic_l2=state.plastic.pow(2).mean(),
        trace_l2=state.trace.pow(2).mean(),
        aux_loss=aux_loss,
        margin_loss=margin_loss,
        aux_acc=aux_acc,
        observed_score_std=observed_score_std,
    )
    return state, codes, stats

def run_no_feedback_test(config: Config, net: PlasticReinstatementRanker, state: State, codes: torch.Tensor, subject_z: torch.Tensor,
                         true_scores: torch.Tensor, pairs: torch.Tensor, train_mode: bool, repetitions: int,
                         learned_pair_mask: Optional[torch.Tensor] = None, sample_choices: bool = False) -> dict:
    bsz, n_pairs, _ = pairs.shape
    bidx = torch.arange(bsz, device=DEVICE)
    choice_losses: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    correct_means: list[torch.Tensor] = []
    learned_corrects: list[torch.Tensor] = []
    nonlearned_corrects: list[torch.Tensor] = []
    mod_vals: list[torch.Tensor] = []
    eta_vals: list[torch.Tensor] = []
    plastic_vals: list[torch.Tensor] = []
    pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    label_one_sum = torch.zeros(bsz, n_pairs, device=DEVICE)

    for _rep in range(repetitions):
        if config.test_order_shuffle:
            orders = torch.stack([torch.randperm(n_pairs, device=DEVICE) for _ in range(bsz)], dim=0)
        else:
            orders = torch.arange(n_pairs, device=DEVICE)[None, :].expand(bsz, -1)
        for k in range(n_pairs):
            pidx = orders[:, k]
            item_i = pairs[bidx, pidx, 0]
            item_j = pairs[bidx, pidx, 1]
            label = pair_labels(true_scores, item_i, item_j)
            logit = net.comparator_logit(codes, state, subject_z, item_i, item_j)
            logit32 = logit.float()
            label32 = label.float()
            choice_losses.append(F.binary_cross_entropy_with_logits(logit32, label32, reduction="none").mean())
            prob32 = torch.sigmoid(logit32).clamp(1e-6, 1.0 - 1e-6)
            ent = -(prob32 * prob32.log() + (1.0 - prob32) * (1.0 - prob32).log())
            entropies.append(ent.mean())
            if sample_choices:
                choice = torch.bernoulli(prob32).float()
                correct = (choice == label32).float()
                pseudo_sign = choice * 2.0 - 1.0
            else:
                choice = (prob32 >= 0.5).float()
                correct = (choice == label32).float()
                pseudo_sign = torch.tanh(logit32)
            pair_correct_sum[bidx, pidx] += correct
            label_one_sum[bidx, pidx] += label32
            correct_means.append(correct.mean())
            if learned_pair_mask is not None:
                mask = learned_pair_mask[bidx, pidx].bool()
                if mask.any():
                    learned_corrects.append(correct[mask])
                if (~mask).any():
                    nonlearned_corrects.append(correct[~mask])
            if config.test_update_mode != "frozen":
                confidence = torch.tanh(logit32).abs().clamp(0.0, 1.0).pow(config.test_pseudo_confidence_power)
                pseudo_relation = (pseudo_sign * confidence).clamp(-1.0, 1.0)
                if config.detach_test_pseudo or not train_mode:
                    pseudo_relation = pseudo_relation.detach()
                    confidence = confidence.detach()
                eta_scale = config.test_eta_scale if config.test_update_mode == "self_reconsolidate" else 0.0
                state, mod, eta_abs = net.observe_pair(codes, state, subject_z, item_i, item_j, pseudo_relation, phase="test", eta_scale=eta_scale, confidence=confidence)
                mod_vals.append(mod.mean())
                eta_vals.append(eta_abs.mean())
                plastic_vals.append(state.plastic.abs().mean())
                if config.test_update_mode == "self_reconsolidate":
                    state, rmods, retas = net.reinstate(codes, state, subject_z, steps=config.test_reinstatement_steps, eta_scale=config.reinstatement_eta_scale)
                    mod_vals.extend(rmods)
                    eta_vals.extend(retas)
                    if rmods:
                        plastic_vals.append(state.plastic.abs().mean())
    zero = torch.zeros((), device=DEVICE)
    return {
        "state": state,
        "choice_loss": torch.stack(choice_losses).mean() if choice_losses else zero,
        "entropy": torch.stack(entropies).mean() if entropies else zero,
        "acc": torch.stack(correct_means).mean() if correct_means else zero,
        "learned_acc": torch.cat(learned_corrects).mean() if learned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "nonlearned_acc": torch.cat(nonlearned_corrects).mean() if nonlearned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "mean_mod": torch.stack(mod_vals).mean() if mod_vals else zero,
        "mean_eta_abs": torch.stack(eta_vals).mean() if eta_vals else zero,
        "mean_abs_plastic": torch.stack(plastic_vals).mean() if plastic_vals else state.plastic.abs().mean(),
        "pair_correct_rate": pair_correct_sum / max(1, repetitions),
        "label_one_rate": label_one_sum / max(1, repetitions),
    }


def run_training_episode(config: Config, net: PlasticReinstatementRanker) -> EpisodeStats:
    bsz = config.bs
    subject_z = sample_subject_latents(bsz, config.subject_dim)
    r2i = rank_to_item_maps(bsz, config.n_items)
    true_scores = true_scores_by_item(r2i, config)
    rank_edges_per_batch = [sample_sparse_rank_graph(config) for _ in range(bsz)]
    learning_edges = torch.empty(bsz, config.n_learning_pairs, 2, dtype=torch.long, device=DEVICE)
    for b, rank_edges in enumerate(rank_edges_per_batch):
        learning_edges[b] = rank_edges_to_item_edges(rank_edges, r2i[b:b+1], random_orientation=True)[0]
    state, codes, learn_stats = run_learning_phase(config, net, subject_z, true_scores, learning_edges, train_mode=True)
    pairs = all_item_pairs(r2i, config, random_orientation=config.train_query_random_orientation)
    learned_mask = learned_pair_mask_from_edges(learning_edges, pairs)
    test = run_no_feedback_test(config, net, state, codes, subject_z, true_scores, pairs,
                                train_mode=True, repetitions=max(1, config.train_test_repetitions),
                                learned_pair_mask=learned_mask, sample_choices=False)
    final_state: State = test["state"]
    final_scores = net.item_scores(codes, final_state, subject_z)
    score_std = final_scores.std(dim=1).mean()
    score_std_floor_loss = F.relu(torch.tensor(config.score_std_floor, device=DEVICE) - score_std).pow(2)
    loss = (
        test["choice_loss"]
        + config.lambda_learning_aux * learn_stats.aux_loss
        + config.lambda_axis_margin * learn_stats.margin_loss
        + config.lambda_entropy * test["entropy"]
        + config.lambda_plastic_l2 * (learn_stats.plastic_l2 + final_state.plastic.pow(2).mean())
        + config.lambda_trace_l2 * (learn_stats.trace_l2 + final_state.trace.pow(2).mean())
        + config.lambda_eta_l2 * (learn_stats.mean_eta_abs.pow(2) + test["mean_eta_abs"].pow(2))
        + config.lambda_score_var * final_scores.var(dim=1).mean()
        + config.lambda_score_std_floor * score_std_floor_loss
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite loss: choice={float(test['choice_loss'].detach())} aux={float(learn_stats.aux_loss.detach())} "
            f"margin={float(learn_stats.margin_loss.detach())} entropy={float(test['entropy'].detach())} "
            f"plastic={float(final_state.plastic.pow(2).mean().detach())} trace={float(final_state.trace.pow(2).mean().detach())}"
        )
    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        choice_loss=float(test["choice_loss"].detach()),
        entropy_loss=float(test["entropy"].detach()),
        acc=float(test["acc"].detach()),
        learned_acc=float(test["learned_acc"].detach()) if torch.isfinite(test["learned_acc"]) else float("nan"),
        nonlearned_acc=float(test["nonlearned_acc"].detach()) if torch.isfinite(test["nonlearned_acc"]) else float("nan"),
        mean_mod_learning=float(learn_stats.mean_mod.detach()),
        mean_eta_abs_learning=float(learn_stats.mean_eta_abs.detach()),
        mean_abs_plastic_learning=float(learn_stats.mean_abs_plastic.detach()),
        mean_mod_test=float(test["mean_mod"].detach()),
        mean_eta_abs_test=float(test["mean_eta_abs"].detach()),
        mean_abs_plastic_test=float(test["mean_abs_plastic"].detach()),
        final_score_std=float(score_std.detach()),
        learning_aux_loss=float(learn_stats.aux_loss.detach()),
        learning_margin_loss=float(learn_stats.margin_loss.detach()),
        learning_aux_acc=float(learn_stats.aux_acc.detach()),
        observed_score_std=float(learn_stats.observed_score_std.detach()),
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
        if pair_acc[idx] >= 0.5:
            wins[b] += 1.0
        else:
            wins[a] += 1.0
    return list(np.argsort(wins))


@torch.no_grad()
def evaluate_paper_task(config: Config, net: PlasticReinstatementRanker) -> dict:
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
    state, codes, learn_stats = run_learning_phase(config, net, subject_z, true_scores, learning_edges, train_mode=False)
    pairs = all_item_pairs(r2i, config, random_orientation=config.eval_query_random_orientation)
    learned_mask = learned_pair_mask_from_edges(learning_edges, pairs)
    test = run_no_feedback_test(config, net, state, codes, subject_z, true_scores, pairs,
                                train_mode=False, repetitions=config.eval_repetitions,
                                learned_pair_mask=learned_mask, sample_choices=config.eval_sample_choices)
    pair_acc = test["pair_correct_rate"].cpu().numpy()
    label_one = test["label_one_rate"].cpu().numpy()
    pair_list = config.all_rank_pairs
    paper_pair_set = {tuple(p) for p in PAPER_LEARNING_PAIRS_RANK}
    learned_idx = np.array([idx for idx, p in enumerate(pair_list) if p in paper_pair_set], dtype=int)
    nonlearned_idx = np.array([idx for idx, p in enumerate(pair_list) if p not in paper_pair_set], dtype=int)
    pair_error_prop = 1.0 - pair_acc
    self_cons, circulars, choice_orders = [], [], []
    for b in range(n):
        sc, circ = self_consistency_from_majority(pair_acc[b], pair_list, config.n_items)
        self_cons.append(sc)
        circulars.append(circ)
        choice_orders.append(choice_order_from_pair_acc(pair_acc[b], pair_list, config.n_items))
    taus_choice = [kendall_tau_order(choice_orders[i], choice_orders[j]) for i, j in combinations(range(n), 2)]

    final_scores = net.item_scores(codes, test["state"], subject_z).detach().cpu().numpy()
    score_orders = [list(np.argsort(final_scores[b])) for b in range(n)]
    taus_score = [kendall_tau_order(score_orders[i], score_orders[j]) for i, j in combinations(range(n), 2)]
    true_order = list(range(config.n_items))
    score_tau_true = [kendall_tau_order(score_orders[b], true_order) for b in range(n)]
    score_tau_choice = [kendall_tau_order(score_orders[b], choice_orders[b]) for b in range(n)]

    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())
    summary = {
        "eval_subjects": n,
        "eval_repetitions": config.eval_repetitions,
        "model_type": "v7_predict_feedback_plastic_rank_axis_no_direct_item_writer",
        "use_rnn": config.use_rnn,
        "use_hebbian": config.use_hebbian,
        "use_hidden_readout": config.use_hidden_readout,
        "hidden_readout_gain": config.hidden_readout_gain,
        "reinstatement_steps": config.reinstatement_steps,
        "observed_replay_steps": config.observed_replay_steps,
        "observed_replay_eta_scale": config.observed_replay_eta_scale,
        "lambda_learning_aux": config.lambda_learning_aux,
        "lambda_axis_margin": config.lambda_axis_margin,
        "learning_margin": config.learning_margin,
        "test_reinstatement_steps": config.test_reinstatement_steps,
        "test_update_mode": config.test_update_mode,
        "relation_input": config.relation_input,
        "train_query_random_orientation": config.train_query_random_orientation,
        "eval_query_random_orientation": config.eval_query_random_orientation,
        "mean_true_label_is_class1_j_higher": float(label_one.mean()),
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
        "mean_inter_subject_kendall_tau_score_orders": float(np.mean(taus_score)) if taus_score else 0.0,
        "score_order_tau_true_mean_abs": float(np.mean(np.abs(score_tau_true))),
        "score_order_tau_choice_mean_abs": float(np.mean(np.abs(score_tau_choice))),
        "distance_accuracy": distance_acc,
        "mean_mod_learning": float(learn_stats.mean_mod.detach().cpu()),
        "mean_eta_abs_learning": float(learn_stats.mean_eta_abs.detach().cpu()),
        "mean_abs_plastic_learning": float(learn_stats.mean_abs_plastic.detach().cpu()),
        "mean_abs_trace_learning": float(learn_stats.mean_abs_trace.detach().cpu()),
        "learning_aux_loss": float(learn_stats.aux_loss.detach().cpu()),
        "learning_margin_loss": float(learn_stats.margin_loss.detach().cpu()),
        "learning_aux_accuracy": float(learn_stats.aux_acc.detach().cpu()),
        "observed_score_std": float(learn_stats.observed_score_std.detach().cpu()),
        "mean_mod_test": float(test["mean_mod"].detach().cpu()),
        "mean_eta_abs_test": float(test["mean_eta_abs"].detach().cpu()),
        "mean_abs_plastic_test": float(test["mean_abs_plastic"].detach().cpu()),
        "mean_final_score_std": float(torch.tensor(final_scores).std(dim=1).mean()),
        "choice_beta": float(net.log_beta.exp().detach().cpu()),
    }
    if old_beta is not None:
        net.log_beta.data.copy_(old_beta)
    return summary


def train(config: Config) -> PlasticReinstatementRanker:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_v7.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    net = PlasticReinstatementRanker(config).to(DEVICE)
    if config.compile_model and hasattr(torch, "compile"):
        log("[setup] torch.compile enabled")
        net = torch.compile(net)  # type: ignore[assignment]
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2)
    use_amp = bool(config.amp and DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log(f"[setup] Device={DEVICE}; amp={use_amp}; output={out}")
    log("[setup] V7: behavioral task + predict-feedback learning loop + plastic representation transformer; no direct item-state writer")
    log(f"[setup] use_rnn={config.use_rnn}; use_hebbian={config.use_hebbian}; test_update_mode={config.test_update_mode}")
    log(f"[setup] bs={config.bs}; nbiter={config.nbiter}; item_code_dim={config.item_code_dim}; rnn_hidden={config.rnn_hidden_size}")
    log(f"[setup] reinstatement_steps={config.reinstatement_steps}; observed_replay_steps={config.observed_replay_steps}; hidden_readout_gain={config.hidden_readout_gain}")
    log(f"[setup] parameters={sum(p.numel() for p in net.parameters())}")
    fields = [
        "episode", "loss", "choice_loss", "entropy_loss", "acc", "learned_acc", "nonlearned_acc",
        "mean_mod_learning", "mean_eta_abs_learning", "mean_abs_plastic_learning",
        "mean_mod_test", "mean_eta_abs_test", "mean_abs_plastic_test", "final_score_std", "learning_aux_loss", "learning_margin_loss", "learning_aux_acc", "observed_score_std",
    ]
    with open(out / "train_log.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    start = time.time()
    last: EpisodeStats | None = None
    iterator = tqdm(range(config.nbiter), desc="training episodes", unit="episode", dynamic_ncols=True, file=sys.stdout)
    for ep in iterator:
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            stats = run_training_episode(config, net)
        scaler.scale(stats.loss).backward()
        if config.gc > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        scaler.step(opt)
        scaler.update()
        last = stats
        with open(out / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                ep, stats.loss_value, stats.choice_loss, stats.entropy_loss, stats.acc, stats.learned_acc, stats.nonlearned_acc,
                stats.mean_mod_learning, stats.mean_eta_abs_learning, stats.mean_abs_plastic_learning,
                stats.mean_mod_test, stats.mean_eta_abs_test, stats.mean_abs_plastic_test, stats.final_score_std, stats.learning_aux_loss, stats.learning_margin_loss, stats.learning_aux_acc, stats.observed_score_std,
            ])
        if ep % config.pe == 0 or ep == config.nbiter - 1:
            elapsed = time.time() - start
            log(
                f"Episode {ep} ==== {elapsed:.2f}s | loss={stats.loss_value:.4f} choice={stats.choice_loss:.4f} "
                f"acc={stats.acc:.3f} learned={stats.learned_acc:.3f} nonlearned={stats.nonlearned_acc:.3f} | "
                f"modL={stats.mean_mod_learning:.4f} etaL={stats.mean_eta_abs_learning:.4f} |pL|={stats.mean_abs_plastic_learning:.4f} "
                f"modT={stats.mean_mod_test:.4f} etaT={stats.mean_eta_abs_test:.4f} |pT|={stats.mean_abs_plastic_test:.4f} "
                f"score_std={stats.final_score_std:.3f} aux={stats.learning_aux_loss:.4f} aux_acc={stats.learning_aux_acc:.3f} margin={stats.learning_margin_loss:.4f} obs_std={stats.observed_score_std:.3f}"
            )
            start = time.time()
        if config.save_every > 0 and ep > 0 and ep % config.save_every == 0:
            torch.save({"model_state": net.state_dict(), "config": asdict(config)}, out / "net_v7.pt")
            log(f"[save] checkpoint: {out / 'net_v7.pt'}")
    torch.save({"model_state": net.state_dict(), "config": asdict(config)}, out / "net_v7.pt")
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
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="V7 behavioral plastic ranker: predict-feedback learning loop, observed-pair replay, no direct item-state writer.")
    p.add_argument("--rngseed", type=int, default=1)
    p.add_argument("--nbiter", type=int, default=200)
    p.add_argument("--batch-size", "--bs", dest="bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--grad-clip", "--gc", dest="gc", type=float, default=2.0)
    p.add_argument("--print-every", "--pe", dest="pe", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="outputs_behavioral_plastic_v7")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--num-threads", type=int, default=1)

    p.add_argument("--n-learning-blocks", type=int, default=4)
    p.add_argument("--paper-graph-train-prob", type=float, default=1.0)
    p.add_argument("--relation-noise", type=float, default=0.03)
    p.add_argument("--eval-relation-noise", type=float, default=0.0)
    p.add_argument("--edge-dropout", type=float, default=0.0)
    p.add_argument("--eval-edge-dropout", type=float, default=0.0)
    p.add_argument("--relation-input", choices=["magnitude", "sign"], default="magnitude")

    p.add_argument("--item-input-dim", type=int, default=32)
    p.add_argument("--item-code-dim", type=int, default=128)
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
    p.add_argument("--use-hidden-readout", dest="use_hidden_readout", action="store_true")
    p.add_argument("--no-hidden-readout", dest="use_hidden_readout", action="store_false")
    p.set_defaults(use_hidden_readout=True)
    p.add_argument("--hidden-readout-gain", type=float, default=0.05)

    p.add_argument("--rnn-input-gain", type=float, default=1.0)
    p.add_argument("--rnn-rec-gain", type=float, default=1.0)
    p.add_argument("--fast-weight-gain", type=float, default=0.50)
    p.add_argument("--hebb-eta", type=float, default=0.08)
    p.add_argument("--plastic-decay", type=float, default=0.97)
    p.add_argument("--trace-decay", type=float, default=0.70)
    p.add_argument("--plastic-clip", type=float, default=3.0)
    p.add_argument("--detach-plastic-state", action="store_true")
    p.add_argument("--rnn-dropout", type=float, default=0.0)
    p.add_argument("--neuromod-clip", type=float, default=1.0)

    p.add_argument("--reinstatement-steps", type=int, default=0)
    p.add_argument("--reinstatement-eta-scale", type=float, default=0.30)
    p.add_argument("--test-reinstatement-steps", type=int, default=0)
    p.add_argument("--replay-temperature", type=float, default=1.0)
    p.add_argument("--observed-replay-steps", type=int, default=1)
    p.add_argument("--observed-replay-eta-scale", type=float, default=0.50)
    p.add_argument("--replay-aux-scale", type=float, default=1.00)

    p.add_argument("--test-update-mode", choices=["frozen", "hidden", "self_reconsolidate"], default="frozen")
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
    p.add_argument("--test-pseudo-confidence-power", type=float, default=1.0)
    p.add_argument("--detach-test-pseudo", action="store_true")
    p.add_argument("--eval-sample-choices", dest="eval_sample_choices", action="store_true")
    p.add_argument("--no-eval-sample-choices", dest="eval_sample_choices", action="store_false")
    p.set_defaults(eval_sample_choices=True)

    p.add_argument("--choice-beta-init", type=float, default=2.0)
    p.add_argument("--eval-beta-override", type=float, default=0.0)
    p.add_argument("--lambda-plastic-l2", type=float, default=1e-5)
    p.add_argument("--lambda-trace-l2", type=float, default=1e-6)
    p.add_argument("--lambda-eta-l2", type=float, default=1e-4)
    p.add_argument("--lambda-entropy", type=float, default=0.001)
    p.add_argument("--lambda-score-var", type=float, default=0.0)
    p.add_argument("--lambda-learning-aux", type=float, default=1.00)
    p.add_argument("--lambda-axis-margin", type=float, default=0.10)
    p.add_argument("--learning-margin", type=float, default=0.20)
    p.add_argument("--lambda-score-std-floor", type=float, default=0.0)
    p.add_argument("--score-std-floor", type=float, default=0.05)
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
