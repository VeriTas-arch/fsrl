"""
GPU-fast unified constructive-ranking learner for stages 01--04, with optional plastic Hebbian RNN.

V3 fixed: V2 fixes plus random train/eval query orientation to remove class-position shortcut leakage.

This script is intended to replace the earlier 01/02/03/04 command-line variants:
  01: direct score-coordinate learner
  02: edge memory + forgetting / capacity interference
  03: subject/item/pair/distance reliability switches
  04: schema encoding, replay, reconsolidation

New in this integrated version:
  - Optional RNN core: --use-rnn / --no-rnn
  - Optional Hebbian fast weights inside that RNN: --use-hebbian / --no-hebbian
  - CPU/GPU selection: --device auto|cpu|cuda, with optional --amp on CUDA
  - Human-like test default: no-feedback but internally plastic.
    Test trials can update hidden/fast weights and weakly reconsolidate memory/scores
    from self-generated pseudo evidence, never from true labels/reward.

Outer loop: supervised meta-loss after a full episode.
Inner loop: passive observation learning; no reward.
Test phase: no feedback; optional internal state update only.
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

# GPU-fast caches. They are rebuilt after DEVICE is configured.
RANK_GRAPH_POOL: torch.Tensor | None = None
ALL_RANK_PAIRS_TENSOR: torch.Tensor | None = None
PAPER_RANK_EDGES_TENSOR: torch.Tensor | None = None


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    """Return True when a cached tensor is already on the intended device.

    torch.device("cuda") and tensor.device "cuda:0" do not compare equal in PyTorch,
    which caused the GPU-fast graph pool to be rebuilt and logged every episode.
    """
    actual = torch.device(actual)
    expected = torch.device(expected)
    if actual.type != expected.type:
        return False
    if actual.type != "cuda":
        return True
    actual_index = 0 if actual.index is None else actual.index
    if expected.index is None:
        try:
            expected_index = torch.cuda.current_device()
        except Exception:
            expected_index = 0
    else:
        expected_index = expected.index
    return actual_index == expected_index


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class TrainConfig:
    # Run control
    rngseed: int = 1
    nbiter: int = 4
    bs: int = 2
    lr: float = 3e-4
    eps: float = 1e-8
    l2: float = 0.0
    gc: float = 2.0
    save_every: int = 50
    pe: int = 10
    output_dir: str = "outputs_unified"
    num_threads: int = 1
    device: str = "auto"
    amp: bool = False
    compile_model: bool = False
    gpu_fast: bool = True
    graph_pool_size: int = 8192
    fast_batch_graph_sampling: bool = True
    fast_per_subject_orders: bool = True
    timing_breakdown: bool = False

    # Task
    n_items: int = 8
    n_learning_pairs: int = 8
    n_learning_blocks: int = 4
    paper_graph_train_prob: float = 0.0

    # Stage 01 score-coordinate baseline
    hidden_size: int = 8
    item_dim: int = 15
    subject_dim: int = 8
    relation_noise: float = 0.03
    eval_relation_noise: float = 0.0
    edge_dropout: float = 0.0
    eval_edge_dropout: float = 0.0
    init_score_noise: float = 0.0
    subject_scale: float = 1.0
    update_scale: float = 1.0
    eval_beta_override: float = 0.0
    rnn_choice_gain: float = 0.25

    # Stage 02 memory mechanisms
    memory_mode: str = "direct"  # direct | edge_online | edge_block | edge_hybrid
    forget_rate: float = 0.0
    memory_capacity: float = 10.0
    memory_encoding_noise: float = 0.0
    eval_memory_encoding_noise: float = 0.0
    memory_attention_bias: float = 0.0
    distance_salience: float = 0.0

    # Stage 03 subject-specific reliability
    subject_attention_scale: float = 0.0
    item_attention_scale: float = 0.0
    pair_attention_scale: float = 0.0
    distance_attention_scale: float = 0.0
    reliability_temperature: float = 1.0

    # Stage 04 modular memory mechanisms
    reconsolidation_strength: float = 0.0
    reconsolidation_power: float = 1.0
    reconsolidation_refresh: float = 0.0
    schema_encoding_bias: float = 0.0
    replay_steps: int = 0
    replay_strength: float = 1.0
    replay_temperature: float = 1.0
    schema_sweeps: int = 1

    # Optional plastic Hebbian RNN core
    use_rnn: bool = False
    use_hebbian: bool = True
    rnn_hidden_size: int = 64
    rnn_input_gain: float = 1.0
    rnn_rec_gain: float = 1.0
    fast_weight_gain: float = 0.35
    hebb_eta: float = 0.06
    plastic_decay: float = 0.96
    plastic_clip: float = 3.0
    detach_plastic_state: bool = False
    rnn_dropout: float = 0.0

    # Test-phase no-feedback internal update
    test_update_mode: str = "self_reconsolidate"  # frozen | hidden | self_reconsolidate
    train_test_repetitions: int = 1
    test_eta_scale: float = 0.35
    test_score_update_scale: float = 0.10
    test_memory_update_strength: float = 0.08
    test_pseudo_confidence_power: float = 1.0
    test_order_shuffle: bool = True
    detach_test_state: bool = False

    # DBG / anti-shortcut diagnostics
    dbg_eval_components: bool = False
    # Randomize left/right orientation of query pairs. This prevents models from
    # exploiting the previous shortcut that class-1 / second item was always the
    # higher-rank item during all-pair train/eval queries.
    train_query_random_orientation: bool = True
    eval_query_random_orientation: bool = True
    eval_only_checkpoint: str = ""

    # Losses
    lambda_recon: float = 0.5
    lambda_entropy: float = 0.005
    lambda_score_l2: float = 1e-4
    lambda_plastic_l2: float = 1e-5
    lambda_eta_l2: float = 1e-4

    # Evaluation
    eval_subjects: int = 8
    eval_repetitions: int = 5

    @property
    def all_rank_pairs(self) -> list[tuple[int, int]]:
        return list(combinations(range(self.n_items), 2))

    @property
    def rank_values(self) -> torch.Tensor:
        values = torch.arange(self.n_items, dtype=torch.float32)
        values = (values - values.mean()) / (self.n_items - 1)
        return values


@dataclass
class EpisodeState:
    scores: torch.Tensor
    memory: Optional[torch.Tensor]
    strength: Optional[torch.Tensor]
    rnn_h: Optional[torch.Tensor]
    plastic: Optional[torch.Tensor]


@dataclass
class PhaseStats:
    recon_loss: torch.Tensor
    mean_gate: torch.Tensor
    mean_eta: torch.Tensor
    mean_abs_plastic: torch.Tensor
    plastic_l2: torch.Tensor


class PlasticHebbianCore(nn.Module):
    def __init__(self, config: TrainConfig, obs_dim: int):
        super().__init__()
        self.config = config
        h = config.rnn_hidden_size
        self.input_proj = nn.Linear(obs_dim, h)
        self.rec_proj = nn.Linear(h, h, bias=False)
        self.layer_norm = nn.LayerNorm(h)
        self.eta_head = nn.Sequential(
            nn.Linear(obs_dim + h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.dropout = nn.Dropout(config.rnn_dropout)

    def initial_state(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(bsz, self.config.rnn_hidden_size, device=device)
        plastic = torch.zeros(bsz, self.config.rnn_hidden_size, self.config.rnn_hidden_size, device=device)
        return h, plastic

    def step(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        plastic: torch.Tensor,
        train_mode: bool,
        eta_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if cfg.detach_plastic_state:
            h_fast = h.detach()
            plastic_fast = plastic.detach()
        else:
            h_fast = h
            plastic_fast = plastic
        # Keep recurrent episode-local state in fp32 under AMP.
        # CUDA autocast may make linear outputs fp16, but fast-weight bmm and
        # later state assignments are safer and more stable in fp32.
        plastic_fast = plastic_fast.float()
        h_fast = h_fast.float()
        h = h.float()
        fast = torch.bmm(plastic_fast, h_fast.unsqueeze(2)).squeeze(2)
        pre = cfg.rnn_input_gain * self.input_proj(x) + cfg.rnn_rec_gain * self.rec_proj(h) + cfg.fast_weight_gain * fast
        h_new = torch.tanh(self.layer_norm(pre)).float()
        if train_mode and cfg.rnn_dropout > 0:
            h_new = self.dropout(h_new)
        if not cfg.use_hebbian:
            eta = torch.zeros(h_new.shape[0], device=h_new.device, dtype=torch.float32)
            plastic_new = plastic.float()
        else:
            eta = ((cfg.hebb_eta * eta_scale) * torch.sigmoid(self.eta_head(torch.cat([x, h_new], dim=1))).squeeze(1)).float()
            hebb = torch.bmm(h_new.unsqueeze(2), h.unsqueeze(1)).float()
            plastic_new = cfg.plastic_decay * plastic.float() + eta[:, None, None] * hebb
            if cfg.plastic_clip > 0:
                plastic_new = plastic_new.clamp(-cfg.plastic_clip, cfg.plastic_clip)
        return h_new, plastic_new, eta


class ConstructiveRankingNet(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        z = config.subject_dim
        d = config.item_dim
        h = config.hidden_size
        self.base_feature_dim = 2 * d + z + 4
        self.rnn_core: PlasticHebbianCore | None = None
        rnn_dim = 0
        if config.use_rnn:
            self.rnn_core = PlasticHebbianCore(config, obs_dim=self.base_feature_dim)
            rnn_dim = config.rnn_hidden_size

        self.score_init = nn.Sequential(nn.Linear(d + z, h), nn.Tanh(), nn.Linear(h, 1))
        self.update_gate = nn.Sequential(
            nn.Linear(self.base_feature_dim + rnn_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)
        )
        self.memory_attention = nn.Sequential(
            nn.Linear(2 * d + z + 2 + rnn_dim, h), nn.Tanh(), nn.Linear(h, 1)
        )
        self.item_reliability = nn.Sequential(nn.Linear(d + z, h), nn.Tanh(), nn.Linear(h, 1))
        self.pair_reliability = nn.Sequential(nn.Linear(2 * d + z, h), nn.Tanh(), nn.Linear(h, 1))
        self.distance_reliability = nn.Linear(z, 1)
        self.choice_bias: nn.Module | None = None
        if config.use_rnn:
            self.choice_bias = nn.Sequential(
                nn.Linear(self.base_feature_dim + rnn_dim, h), nn.Tanh(), nn.Linear(h, 2)
            )
        self.logit_step = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

    @staticmethod
    def center_scores(scores: torch.Tensor) -> torch.Tensor:
        return scores - scores.mean(dim=1, keepdim=True)

    def initial_scores(self, item_vecs: torch.Tensor, subject_z: torch.Tensor) -> torch.Tensor:
        bsz, n_items, _ = item_vecs.shape
        z_scaled = self.config.subject_scale * subject_z
        z_rep = z_scaled[:, None, :].expand(bsz, n_items, subject_z.shape[-1])
        raw = self.score_init(torch.cat([item_vecs, z_rep], dim=-1)).squeeze(-1)
        if self.config.init_score_noise > 0:
            raw = raw + self.config.init_score_noise * torch.randn_like(raw)
        # Keep episode-local scalar state in fp32 even when AMP is enabled.
        # Linear layers may emit fp16 under autocast; indexed state updates later
        # require source/destination dtypes to match and are more stable in fp32.
        return self.center_scores(raw.float())

    def initial_recurrent_state(self, bsz: int, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.rnn_core is None:
            return None, None
        return self.rnn_core.initial_state(bsz, device)

    def build_base_features(
        self,
        scores: torch.Tensor,
        item_vecs: torch.Tensor,
        subject_z: torch.Tensor,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        observed_diff_j_minus_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = scores.shape[0]
        batch_idx = torch.arange(bsz, device=scores.device)
        score_i = scores[batch_idx, item_i]
        score_j = scores[batch_idx, item_j]
        pred_diff = score_j - score_i
        error = observed_diff_j_minus_i - pred_diff
        vec_i = item_vecs[batch_idx, item_i]
        vec_j = item_vecs[batch_idx, item_j]
        base = torch.cat([
            vec_i,
            vec_j,
            self.config.subject_scale * subject_z,
            observed_diff_j_minus_i[:, None],
            observed_diff_j_minus_i.abs()[:, None],
            pred_diff[:, None],
            error[:, None],
        ], dim=1)
        return base, pred_diff, error

    def step_rnn(
        self,
        base_features: torch.Tensor,
        rnn_h: torch.Tensor | None,
        plastic: torch.Tensor | None,
        train_mode: bool,
        eta_scale: float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.rnn_core is None:
            return rnn_h, plastic, torch.zeros(base_features.shape[0], device=base_features.device, dtype=base_features.dtype)
        assert rnn_h is not None and plastic is not None
        rnn_h, plastic, eta = self.rnn_core.step(base_features, rnn_h, plastic, train_mode=train_mode, eta_scale=eta_scale)
        return rnn_h, plastic, eta

    def gate_features(self, base_features: torch.Tensor, rnn_h: torch.Tensor | None) -> torch.Tensor:
        if self.config.use_rnn:
            assert rnn_h is not None
            return torch.cat([base_features, rnn_h], dim=1)
        return base_features

    def update_scores(
        self,
        scores: torch.Tensor,
        item_vecs: torch.Tensor,
        subject_z: torch.Tensor,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        observed_diff_j_minus_i: torch.Tensor,
        rnn_h: torch.Tensor | None,
        plastic: torch.Tensor | None,
        train_mode: bool,
        external_weight: torch.Tensor | None = None,
        rnn_update: bool = False,
        eta_scale: float = 1.0,
        update_scale_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        bsz = scores.shape[0]
        batch_idx = torch.arange(bsz, device=scores.device)
        base, pred_diff, error = self.build_base_features(scores, item_vecs, subject_z, item_i, item_j, observed_diff_j_minus_i)
        eta = torch.zeros(bsz, device=scores.device, dtype=scores.dtype)
        if rnn_update:
            rnn_h, plastic, eta = self.step_rnn(base, rnn_h, plastic, train_mode=train_mode, eta_scale=eta_scale)
        gate = torch.sigmoid(self.update_gate(self.gate_features(base, rnn_h))).squeeze(1)
        scale = self.config.update_scale if update_scale_override is None else update_scale_override
        step = scale * torch.sigmoid(self.logit_step) * gate
        if external_weight is not None:
            step = step * external_weight.clamp(0.0, 1.0)
        step = step.clamp(0.0, 1.0)
        delta = 0.5 * step * error
        next_scores = scores.clone()
        # AMP safety: advanced indexing assignment requires exact dtype match.
        delta = delta.to(dtype=next_scores.dtype)
        next_scores[batch_idx, item_i] = (next_scores[batch_idx, item_i] - delta).to(dtype=next_scores.dtype)
        next_scores[batch_idx, item_j] = (next_scores[batch_idx, item_j] + delta).to(dtype=next_scores.dtype)
        next_scores = self.center_scores(next_scores)
        return next_scores, pred_diff, error, gate, rnn_h, plastic, eta

    def encode_attention(
        self,
        item_vecs: torch.Tensor,
        subject_z: torch.Tensor,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        observed_diff_j_minus_i: torch.Tensor,
        rnn_h: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz = item_vecs.shape[0]
        batch_idx = torch.arange(bsz, device=item_vecs.device)
        vec_i = item_vecs[batch_idx, item_i]
        vec_j = item_vecs[batch_idx, item_j]
        features = torch.cat([
            vec_i,
            vec_j,
            self.config.subject_scale * subject_z,
            observed_diff_j_minus_i[:, None],
            observed_diff_j_minus_i.abs()[:, None],
        ], dim=1)
        if self.config.use_rnn:
            assert rnn_h is not None
            features_for_attn = torch.cat([features, rnn_h], dim=1)
        else:
            features_for_attn = features
        raw = self.memory_attention(features_for_attn).squeeze(1)
        if self.config.item_attention_scale != 0.0 or self.config.subject_attention_scale != 0.0:
            zi = self.config.subject_scale * subject_z
            item_i_raw = self.item_reliability(torch.cat([vec_i, zi], dim=1)).squeeze(1)
            item_j_raw = self.item_reliability(torch.cat([vec_j, zi], dim=1)).squeeze(1)
            raw = raw + self.config.item_attention_scale * 0.5 * (item_i_raw + item_j_raw)
            raw = raw + self.config.subject_attention_scale * zi.mean(dim=1)
        if self.config.pair_attention_scale != 0.0:
            pair_raw = self.pair_reliability(torch.cat([vec_i, vec_j, self.config.subject_scale * subject_z], dim=1)).squeeze(1)
            raw = raw + self.config.pair_attention_scale * pair_raw
        if self.config.distance_attention_scale != 0.0:
            dist_pref = self.distance_reliability(self.config.subject_scale * subject_z).squeeze(1)
            raw = raw + self.config.distance_attention_scale * dist_pref * (observed_diff_j_minus_i.abs() - 0.5)
        raw = raw + self.config.memory_attention_bias + self.config.distance_salience * observed_diff_j_minus_i.abs()
        temp = max(1e-3, float(self.config.reliability_temperature))
        return torch.sigmoid(raw / temp)

    def choice_logits(
        self,
        scores: torch.Tensor,
        item_i: torch.Tensor,
        item_j: torch.Tensor,
        base_features: torch.Tensor | None = None,
        rnn_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_idx = torch.arange(scores.shape[0], device=scores.device)
        beta = F.softplus(self.log_beta) + 1e-3
        logits = beta * torch.stack([scores[batch_idx, item_i], scores[batch_idx, item_j]], dim=1)
        if self.choice_bias is not None and base_features is not None and rnn_h is not None and self.config.rnn_choice_gain != 0:
            logits = logits + self.config.rnn_choice_gain * self.choice_bias(torch.cat([base_features, rnn_h], dim=1))
        return logits


def configure_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_arg}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def configure_threads(num_threads: int, device: torch.device) -> None:
    if device.type == "cpu" and num_threads and num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(num_threads, 4)))
        except RuntimeError:
            pass
        log(f"[setup] torch num_threads={torch.get_num_threads()}")


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
    return (torch.randint(0, 2, (batch_size, n_items, item_dim), device=DEVICE).float() * 2.0) - 1.0


def sample_subject_latents(batch_size: int, subject_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, subject_dim, device=DEVICE)


def rank_to_item_maps(batch_size: int, n_items: int) -> torch.Tensor:
    return torch.stack([torch.randperm(n_items, device=DEVICE) for _ in range(batch_size)], dim=0)


def true_scores_by_item(rank_to_item: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    values = config.rank_values.to(DEVICE)
    bsz = rank_to_item.shape[0]
    scores = torch.empty(bsz, config.n_items, device=DEVICE)
    batch_idx = torch.arange(bsz, device=DEVICE)
    for rank in range(config.n_items):
        # Per-subject assignment: rank_to_item[b, rank] is the item occupying this rank
        # for subject b. The previous scores[:, rank_to_item[:, rank]] form uses
        # Cartesian advanced indexing and overwrote many columns for every subject.
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


def sample_sparse_rank_graph(config: TrainConfig) -> list[tuple[int, int]]:
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


def _all_rank_pairs_tensor(config: TrainConfig) -> torch.Tensor:
    """Cached [P, 2] rank-pair tensor on DEVICE."""
    global ALL_RANK_PAIRS_TENSOR
    if ALL_RANK_PAIRS_TENSOR is None or not _same_device(ALL_RANK_PAIRS_TENSOR.device, DEVICE):
        ALL_RANK_PAIRS_TENSOR = torch.tensor(config.all_rank_pairs, dtype=torch.long, device=DEVICE)
    return ALL_RANK_PAIRS_TENSOR


def _sample_valid_sparse_rank_graph_no_paper(config: TrainConfig) -> list[tuple[int, int]]:
    """Same validity constraints as sample_sparse_rank_graph(), but never returns the paper graph by probability."""
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


def build_rank_graph_pool(config: TrainConfig) -> torch.Tensor:
    """Pre-generate valid sparse rank graphs once, avoiding per-episode Python graph sampling.

    The pool contains rank-index edges [low_rank, high_rank]. During training we select graph ids
    on DEVICE and then apply rank_to_item_maps plus optional random orientation. This preserves the
    original task logic but removes the per-subject Python loop from every episode.
    """
    graphs = []
    size = max(1, int(config.graph_pool_size))
    for _ in range(size):
        graphs.append(_sample_valid_sparse_rank_graph_no_paper(config))
    return torch.tensor(graphs, dtype=torch.long, device=DEVICE)


def ensure_rank_graph_pool(config: TrainConfig) -> torch.Tensor | None:
    global RANK_GRAPH_POOL
    if not config.fast_batch_graph_sampling or config.graph_pool_size <= 0:
        return None
    if RANK_GRAPH_POOL is None or not _same_device(RANK_GRAPH_POOL.device, DEVICE) or RANK_GRAPH_POOL.shape[1] != config.n_learning_pairs:
        RANK_GRAPH_POOL = build_rank_graph_pool(config)
        log(f"[gpu-fast] graph_pool_size={RANK_GRAPH_POOL.shape[0]} graphs on {RANK_GRAPH_POOL.device}")
    return RANK_GRAPH_POOL


def _paper_rank_edges_tensor() -> torch.Tensor:
    """Cached paper graph tensor on DEVICE."""
    global PAPER_RANK_EDGES_TENSOR
    if PAPER_RANK_EDGES_TENSOR is None or not _same_device(PAPER_RANK_EDGES_TENSOR.device, DEVICE):
        PAPER_RANK_EDGES_TENSOR = torch.tensor(PAPER_LEARNING_PAIRS_RANK, dtype=torch.long, device=DEVICE)
    return PAPER_RANK_EDGES_TENSOR


def sample_rank_edges_batch(config: TrainConfig, bsz: int) -> torch.Tensor:
    """Return [B, E, 2] rank-edge tensor.

    If graph-pool sampling is enabled, this is a fully tensorized DEVICE operation after the one-time
    pool build. Paper-graph mixing is implemented with a DEVICE-side Bernoulli mask, preserving
    --paper-graph-train-prob semantics.
    """
    pool = ensure_rank_graph_pool(config)
    if pool is None:
        # Correct fallback, slower: same semantics as the previous source.
        edges_cpu = [sample_sparse_rank_graph(config) for _ in range(bsz)]
        return torch.tensor(edges_cpu, dtype=torch.long, device=DEVICE)
    idx = torch.randint(0, pool.shape[0], (bsz,), device=DEVICE)
    rank_edges = pool[idx]
    if config.paper_graph_train_prob > 0:
        paper = _paper_rank_edges_tensor()[None, :, :].expand(bsz, -1, -1)
        use_paper = (torch.rand(bsz, device=DEVICE) < config.paper_graph_train_prob)[:, None, None]
        rank_edges = torch.where(use_paper, paper, rank_edges)
    return rank_edges


def rank_edges_batch_to_item_edges(rank_edges: torch.Tensor, rank_to_item: torch.Tensor, random_orientation: bool = True) -> torch.Tensor:
    """Vectorized rank-edge -> item-edge conversion.

    rank_edges: [B, E, 2] rank ids. rank_to_item: [B, N].
    Returns item_edges [B, E, 2]. This replaces a Python loop over batch subjects.
    """
    item_a = torch.gather(rank_to_item, 1, rank_edges[:, :, 0])
    item_b = torch.gather(rank_to_item, 1, rank_edges[:, :, 1])
    item_edges = torch.stack([item_a, item_b], dim=2)
    if random_orientation:
        swap = torch.rand(item_edges.shape[0], item_edges.shape[1], device=DEVICE) < 0.5
        first = item_edges[:, :, 0].clone()
        second = item_edges[:, :, 1].clone()
        item_edges[:, :, 0] = torch.where(swap, second, first)
        item_edges[:, :, 1] = torch.where(swap, first, second)
    return item_edges


def rank_edges_to_item_edges(rank_edges: list[tuple[int, int]], rank_to_item: torch.Tensor, random_orientation: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper for fixed shared rank_edges, now vectorized."""
    bsz = rank_to_item.shape[0]
    rank_edges_tensor = torch.tensor(rank_edges, dtype=torch.long, device=DEVICE)
    rank_edges_batch = rank_edges_tensor[None, :, :].expand(bsz, -1, -1)
    item_edges = rank_edges_batch_to_item_edges(rank_edges_batch, rank_to_item, random_orientation=random_orientation)
    return item_edges, rank_edges_tensor


def all_item_pairs(rank_to_item: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    rank_pairs = _all_rank_pairs_tensor(config)
    bsz = rank_to_item.shape[0]
    rank_pairs_batch = rank_pairs[None, :, :].expand(bsz, -1, -1)
    return rank_edges_batch_to_item_edges(rank_pairs_batch, rank_to_item, random_orientation=False)


def randomize_query_pair_orientation(pairs: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """Randomly swap left/right query order without changing pair identity.

    The previous all-pair query tensor was always generated as [lower_rank, higher_rank],
    so the true label was always class 1 (the second item). That let the RNN choice-bias
    head solve the task by learning a class-position shortcut. This helper preserves the
    same set and index order of rank pairs but flips the displayed/query orientation with
    p=0.5, so labels are approximately balanced and must be inferred from item state.
    """
    if not enabled:
        return pairs
    pairs = pairs.clone()
    swap = torch.rand(pairs.shape[0], pairs.shape[1], device=pairs.device) < 0.5
    first = pairs[:, :, 0].clone()
    second = pairs[:, :, 1].clone()
    pairs[:, :, 0] = torch.where(swap, second, first)
    pairs[:, :, 1] = torch.where(swap, first, second)
    return pairs


def learned_pair_mask_from_edges(learning_item_edges: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Vectorized learned/nonlearned mask for training diagnostics.

    learning_item_edges: [B,E,2], pairs: [B,P,2]. Order-insensitive membership.
    Returns bool mask [B,P]. This replaces Python set construction and a per-subject loop in testing.
    """
    le = torch.sort(learning_item_edges, dim=2).values
    pp = torch.sort(pairs, dim=2).values
    return ((pp[:, :, None, :] == le[:, None, :, :]).all(dim=3)).any(dim=2)


def gather_true_diff(true_scores: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor) -> torch.Tensor:
    return true_scores[torch.arange(true_scores.shape[0], device=true_scores.device), item_j] - true_scores[torch.arange(true_scores.shape[0], device=true_scores.device), item_i]


def init_episode_state(config: TrainConfig, net: ConstructiveRankingNet, item_vecs: torch.Tensor, subject_z: torch.Tensor) -> EpisodeState:
    bsz = item_vecs.shape[0]
    scores = net.initial_scores(item_vecs, subject_z)
    memory = None
    strength = None
    if config.memory_mode != "direct":
        memory = torch.zeros(bsz, config.n_items, config.n_items, device=DEVICE)
        strength = torch.zeros_like(memory)
    rnn_h, plastic = net.initial_recurrent_state(bsz, DEVICE)
    return EpisodeState(scores=scores, memory=memory, strength=strength, rnn_h=rnn_h, plastic=plastic)


def apply_memory_decay_and_interference(strength: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.forget_rate > 0:
        strength = strength * math.exp(-config.forget_rate)
    if config.memory_capacity > 0 and config.memory_capacity < 1000:
        row_sum = strength.sum(dim=2, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(config.memory_capacity / row_sum, max=1.0)
        strength = strength * scale
        strength = 0.5 * (strength + strength.transpose(1, 2))
    return strength


def encode_edge_memory(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    state: EpisodeState,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    item_i: torch.Tensor,
    item_j: torch.Tensor,
    observed_diff_j_minus_i: torch.Tensor,
    train_mode: bool,
) -> tuple[EpisodeState, torch.Tensor]:
    assert state.memory is not None and state.strength is not None
    bsz = state.memory.shape[0]
    batch_idx = torch.arange(bsz, device=DEVICE)
    attn = net.encode_attention(item_vecs, subject_z, item_i, item_j, observed_diff_j_minus_i, state.rnn_h)
    noise_std = config.memory_encoding_noise if train_mode else config.eval_memory_encoding_noise
    encoded = observed_diff_j_minus_i
    if noise_std > 0:
        encoded = (encoded + noise_std * torch.randn_like(encoded)).clamp(-1.0, 1.0)
    if config.schema_encoding_bias > 0:
        schema_diff = state.scores[batch_idx, item_j] - state.scores[batch_idx, item_i]
        mix = (config.schema_encoding_bias * (1.0 - attn)).clamp(0.0, 1.0)
        encoded = ((1.0 - mix) * encoded + mix * schema_diff).clamp(-1.0, 1.0)
    memory = state.memory.clone()
    strength = state.strength.clone()
    old_ij = memory[batch_idx, item_i, item_j]
    old_ji = memory[batch_idx, item_j, item_i]
    new_ij = ((1.0 - attn) * old_ij + attn * encoded).to(dtype=memory.dtype)
    new_ji = ((1.0 - attn) * old_ji - attn * encoded).to(dtype=memory.dtype)
    memory[batch_idx, item_i, item_j] = new_ij
    memory[batch_idx, item_j, item_i] = new_ji
    old_s = strength[batch_idx, item_i, item_j]
    new_s = (1.0 - (1.0 - old_s) * (1.0 - attn)).to(dtype=strength.dtype)
    strength[batch_idx, item_i, item_j] = new_s
    strength[batch_idx, item_j, item_i] = new_s
    state.memory = memory
    state.strength = apply_memory_decay_and_interference(strength, config)
    return state, attn


def update_scores_from_memory_pair(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    state: EpisodeState,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    item_i: torch.Tensor,
    item_j: torch.Tensor,
    train_mode: bool,
    update_scale_override: float | None = None,
) -> tuple[EpisodeState, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert state.memory is not None and state.strength is not None
    batch_idx = torch.arange(state.scores.shape[0], device=DEVICE)
    target = state.memory[batch_idx, item_i, item_j]
    weight = state.strength[batch_idx, item_i, item_j]
    scores, pred, err, gate, rnn_h, plastic, _eta = net.update_scores(
        state.scores, item_vecs, subject_z, item_i, item_j, target, state.rnn_h, state.plastic,
        train_mode=train_mode, external_weight=weight, rnn_update=False, update_scale_override=update_scale_override,
    )
    state.scores = scores
    state.rnn_h = rnn_h
    state.plastic = plastic
    return state, pred, err, gate


def schema_biased_reconsolidation(config: TrainConfig, state: EpisodeState) -> EpisodeState:
    if config.reconsolidation_strength <= 0 or state.memory is None or state.strength is None:
        return state
    schema_diff = state.scores[:, None, :] - state.scores[:, :, None]
    vulnerability = (1.0 - state.strength.clamp(0.0, 1.0)).pow(max(0.0, config.reconsolidation_power))
    mix = (config.reconsolidation_strength * vulnerability).clamp(0.0, 1.0)
    eye = torch.eye(state.memory.shape[1], device=DEVICE)[None, :, :]
    mix = mix * (1.0 - eye)
    state.memory = (1.0 - mix) * state.memory + mix * schema_diff
    if config.reconsolidation_refresh > 0:
        state.strength = (state.strength + config.reconsolidation_refresh * mix).clamp(0.0, 1.0)
        state.strength = 0.5 * (state.strength + state.strength.transpose(1, 2))
    return state


def replay_from_memory(config: TrainConfig, net: ConstructiveRankingNet, state: EpisodeState, item_vecs: torch.Tensor, subject_z: torch.Tensor) -> tuple[EpisodeState, torch.Tensor]:
    if config.replay_steps <= 0 or state.memory is None or state.strength is None:
        return state, torch.tensor(0.0, device=DEVICE)
    bsz = state.scores.shape[0]
    pair_list = config.all_rank_pairs
    pair_tensor = torch.tensor(pair_list, dtype=torch.long, device=DEVICE)
    gates = []
    for _ in range(config.replay_steps):
        with torch.no_grad():
            pair_strength = state.strength[:, pair_tensor[:, 0], pair_tensor[:, 1]].mean(dim=0)
            probs = torch.softmax(pair_strength / max(1e-3, float(config.replay_temperature)), dim=0)
            idx = torch.multinomial(probs, 1).item()
        a, b = pair_list[idx]
        item_i = torch.full((bsz,), a, dtype=torch.long, device=DEVICE)
        item_j = torch.full((bsz,), b, dtype=torch.long, device=DEVICE)
        old_strength = state.strength
        if config.replay_strength != 1.0:
            strength = state.strength.clone()
            strength[:, a, b] = (strength[:, a, b] * config.replay_strength).clamp(0.0, 1.0)
            strength[:, b, a] = strength[:, a, b]
            state.strength = strength
        state, _pred, _err, gate = update_scores_from_memory_pair(config, net, state, item_vecs, subject_z, item_i, item_j, train_mode=True)
        state.strength = old_strength
        gates.append(gate.mean())
    return state, torch.stack(gates).mean() if gates else torch.tensor(0.0, device=DEVICE)


def schema_reconstruct_from_memory(config: TrainConfig, net: ConstructiveRankingNet, state: EpisodeState, item_vecs: torch.Tensor, subject_z: torch.Tensor, train_mode: bool) -> tuple[EpisodeState, torch.Tensor]:
    if state.memory is None or state.strength is None:
        return state, torch.tensor(0.0, device=DEVICE)
    bsz = state.scores.shape[0]
    pair_list = list(config.all_rank_pairs)
    gates = []
    for _ in range(max(1, config.schema_sweeps)):
        random.shuffle(pair_list)
        for a, b in pair_list:
            item_i = torch.full((bsz,), a, dtype=torch.long, device=DEVICE)
            item_j = torch.full((bsz,), b, dtype=torch.long, device=DEVICE)
            state, _pred, _err, gate = update_scores_from_memory_pair(config, net, state, item_vecs, subject_z, item_i, item_j, train_mode=train_mode)
            gates.append(gate.mean())
    return state, torch.stack(gates).mean() if gates else torch.tensor(0.0, device=DEVICE)


def maybe_detach_state(state: EpisodeState) -> EpisodeState:
    state.scores = state.scores.detach()
    if state.memory is not None:
        state.memory = state.memory.detach()
    if state.strength is not None:
        state.strength = state.strength.detach()
    if state.rnn_h is not None:
        state.rnn_h = state.rnn_h.detach()
    if state.plastic is not None:
        state.plastic = state.plastic.detach()
    return state


def run_learning_phase(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    true_scores: torch.Tensor,
    learning_item_edges: torch.Tensor,
    train_mode: bool,
) -> tuple[EpisodeState, PhaseStats]:
    bsz, n_edges, _ = learning_item_edges.shape
    state = init_episode_state(config, net, item_vecs, subject_z)
    noise_std = config.relation_noise if train_mode else config.eval_relation_noise
    edge_dropout = config.edge_dropout if train_mode else config.eval_edge_dropout
    recon_losses, gates, etas, plastic_abs = [], [], [], []
    for _block in range(config.n_learning_blocks):
        orders = torch.argsort(torch.rand(bsz, n_edges, device=DEVICE), dim=1) if config.fast_per_subject_orders else torch.stack([torch.randperm(n_edges, device=DEVICE) for _ in range(bsz)], dim=0)
        for k in range(n_edges):
            edge_idx = orders[:, k]
            batch_idx = torch.arange(bsz, device=DEVICE)
            item_i = learning_item_edges[batch_idx, edge_idx, 0]
            item_j = learning_item_edges[batch_idx, edge_idx, 1]
            true_diff = gather_true_diff(true_scores, item_i, item_j)
            obs_diff = (true_diff + noise_std * torch.randn_like(true_diff)).clamp(-1.0, 1.0) if noise_std > 0 else true_diff
            if edge_dropout > 0:
                keep = (torch.rand_like(obs_diff) >= edge_dropout).float()
                if state.memory is not None:
                    pred_current = state.memory[batch_idx, item_i, item_j]
                else:
                    pred_current = state.scores[batch_idx, item_j] - state.scores[batch_idx, item_i]
                obs_diff = keep * obs_diff + (1.0 - keep) * pred_current.detach()

            if config.memory_mode == "direct":
                scores, _pred, err, gate, rnn_h, plastic, eta = net.update_scores(
                    state.scores, item_vecs, subject_z, item_i, item_j, obs_diff, state.rnn_h, state.plastic,
                    train_mode=train_mode, rnn_update=True, eta_scale=1.0,
                )
                state.scores, state.rnn_h, state.plastic = scores, rnn_h, plastic
                recon_losses.append(err.pow(2).mean())
                gates.append(gate.mean())
                etas.append(eta.mean())
            else:
                # One external observation drives the RNN once. Memory reconstruction/replay uses the latest hidden state without another external RNN step.
                base, _pred, _err = net.build_base_features(state.scores, item_vecs, subject_z, item_i, item_j, obs_diff)
                state.rnn_h, state.plastic, eta = net.step_rnn(base, state.rnn_h, state.plastic, train_mode=train_mode, eta_scale=1.0)
                etas.append(eta.mean())
                state, attn = encode_edge_memory(config, net, state, item_vecs, subject_z, item_i, item_j, obs_diff, train_mode=train_mode)
                recalled = state.memory[batch_idx, item_i, item_j]
                rel = state.strength[batch_idx, item_i, item_j]
                recon_losses.append((rel * (recalled - true_diff).pow(2)).mean())
                gates.append(attn.mean())
                if config.memory_mode in {"edge_online", "edge_hybrid"}:
                    state, _pred2, err2, gate2 = update_scores_from_memory_pair(config, net, state, item_vecs, subject_z, item_i, item_j, train_mode=train_mode)
                    recon_losses.append(err2.pow(2).mean())
                    gates.append(gate2.mean())
            if state.plastic is not None:
                plastic_abs.append(state.plastic.abs().mean())
        if config.memory_mode in {"edge_online", "edge_block", "edge_hybrid"}:
            if config.replay_steps > 0:
                state, rg = replay_from_memory(config, net, state, item_vecs, subject_z)
                gates.append(rg)
            if config.memory_mode in {"edge_block", "edge_hybrid"}:
                state, bg = schema_reconstruct_from_memory(config, net, state, item_vecs, subject_z, train_mode=train_mode)
                gates.append(bg)
            state = schema_biased_reconsolidation(config, state)
    zero = torch.tensor(0.0, device=DEVICE)
    stats = PhaseStats(
        recon_loss=torch.stack(recon_losses).mean() if recon_losses else zero,
        mean_gate=torch.stack(gates).mean() if gates else zero,
        mean_eta=torch.stack(etas).mean() if etas else zero,
        mean_abs_plastic=torch.stack(plastic_abs).mean() if plastic_abs else zero,
        plastic_l2=state.plastic.pow(2).mean() if state.plastic is not None else zero,
    )
    return state, stats


def pseudo_confidence_from_probs(probs: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    conf = (probs[:, 1] - probs[:, 0]).abs().clamp(0.0, 1.0)
    if config.test_pseudo_confidence_power != 1.0:
        conf = conf.pow(config.test_pseudo_confidence_power)
    return conf


def self_reconsolidate_from_choice(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    state: EpisodeState,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    item_i: torch.Tensor,
    item_j: torch.Tensor,
    pseudo_diff: torch.Tensor,
    confidence: torch.Tensor,
    train_mode: bool,
) -> tuple[EpisodeState, torch.Tensor]:
    bsz = state.scores.shape[0]
    batch_idx = torch.arange(bsz, device=DEVICE)
    gate_val = torch.tensor(0.0, device=DEVICE)
    if state.memory is not None and state.strength is not None and config.test_memory_update_strength > 0:
        g = (config.test_memory_update_strength * confidence).clamp(0.0, 1.0)
        memory = state.memory.clone()
        strength = state.strength.clone()
        old_ij = memory[batch_idx, item_i, item_j]
        old_ji = memory[batch_idx, item_j, item_i]
        new_ij = ((1.0 - g) * old_ij + g * pseudo_diff).to(dtype=memory.dtype)
        new_ji = ((1.0 - g) * old_ji - g * pseudo_diff).to(dtype=memory.dtype)
        memory[batch_idx, item_i, item_j] = new_ij
        memory[batch_idx, item_j, item_i] = new_ji
        old_s = strength[batch_idx, item_i, item_j]
        new_s = (1.0 - (1.0 - old_s) * (1.0 - g)).to(dtype=strength.dtype)
        strength[batch_idx, item_i, item_j] = new_s
        strength[batch_idx, item_j, item_i] = new_s
        state.memory = memory
        state.strength = apply_memory_decay_and_interference(strength, config)
        gate_val = g.mean()
        if config.test_score_update_scale > 0:
            state, _p, _e, score_gate = update_scores_from_memory_pair(
                config, net, state, item_vecs, subject_z, item_i, item_j, train_mode=train_mode,
                update_scale_override=config.test_score_update_scale,
            )
            gate_val = 0.5 * (gate_val + score_gate.mean())
    elif config.test_score_update_scale > 0:
        state.scores, _pred, _err, score_gate, state.rnn_h, state.plastic, _eta = net.update_scores(
            state.scores, item_vecs, subject_z, item_i, item_j, pseudo_diff, state.rnn_h, state.plastic,
            train_mode=train_mode, external_weight=confidence, rnn_update=False,
            update_scale_override=config.test_score_update_scale,
        )
        gate_val = score_gate.mean()
    return state, gate_val


def run_no_feedback_test_phase(
    config: TrainConfig,
    net: ConstructiveRankingNet,
    state: EpisodeState,
    item_vecs: torch.Tensor,
    subject_z: torch.Tensor,
    true_scores: torch.Tensor,
    pairs: torch.Tensor,
    train_mode: bool,
    repetitions: int,
    learned_sets: Optional[list[set[tuple[int, int]]]] = None,
    learned_pair_mask: Optional[torch.Tensor] = None,
    sample_choices: bool = False,
) -> dict:
    bsz, n_pairs, _ = pairs.shape
    ce_losses, entropies, corrects, learned_corrects, nonlearned_corrects = [], [], [], [], []
    eta_vals, gate_vals, plastic_vals = [], [], []
    pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)

    # DBG component counters. These are deterministic argmax diagnostics,
    # independent from optional sampled choices used for behavioral evaluation.
    score_pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    full_argmax_pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    bias_pair_correct_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    score_full_disagree_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    label_one_sum = torch.zeros(bsz, n_pairs, device=DEVICE)
    full_margin_true_aligned_vals = []
    score_margin_true_aligned_vals = []
    bias_margin_true_aligned_vals = []
    abs_score_gap_vals = []
    abs_bias_gap_vals = []

    pair_order_base = list(range(n_pairs))
    for _rep in range(repetitions):
        if config.test_order_shuffle:
            order = pair_order_base[:]
            random.shuffle(order)
        else:
            order = pair_order_base
        for pidx in order:
            item_i = pairs[:, pidx, 0]
            item_j = pairs[:, pidx, 1]
            batch_idx = torch.arange(bsz, device=DEVICE)
            pred_diff = state.scores[batch_idx, item_j] - state.scores[batch_idx, item_i]
            if state.memory is not None and state.strength is not None:
                retrieved = state.memory[batch_idx, item_i, item_j]
                mem_conf = state.strength[batch_idx, item_i, item_j]
                internal_obs = mem_conf * retrieved + (1.0 - mem_conf) * pred_diff
            else:
                internal_obs = pred_diff

            base, _pd, _err = net.build_base_features(state.scores, item_vecs, subject_z, item_i, item_j, internal_obs)
            eta = torch.zeros(bsz, device=DEVICE)
            if config.test_update_mode in {"hidden", "self_reconsolidate"} and config.use_rnn:
                state.rnn_h, state.plastic, eta = net.step_rnn(base, state.rnn_h, state.plastic, train_mode=train_mode, eta_scale=config.test_eta_scale)
            # Split full choice logits into score-only and RNN-choice-bias components for DBG.
            beta = F.softplus(net.log_beta) + 1e-3
            score_logits = beta * torch.stack([state.scores[batch_idx, item_i], state.scores[batch_idx, item_j]], dim=1)
            bias_logits = torch.zeros_like(score_logits)
            if net.choice_bias is not None and state.rnn_h is not None and config.rnn_choice_gain != 0:
                bias_logits = config.rnn_choice_gain * net.choice_bias(torch.cat([base, state.rnn_h], dim=1))
            logits = score_logits + bias_logits
            labels = (true_scores[batch_idx, item_j] > true_scores[batch_idx, item_i]).long()

            # DBG deterministic component predictions.
            score_pred = torch.argmax(score_logits, dim=1)
            full_argmax_pred = torch.argmax(logits, dim=1)
            bias_pred = torch.argmax(bias_logits, dim=1)
            score_pair_correct_sum[:, pidx] = score_pair_correct_sum[:, pidx] + (score_pred == labels).float()
            full_argmax_pair_correct_sum[:, pidx] = full_argmax_pair_correct_sum[:, pidx] + (full_argmax_pred == labels).float()
            bias_pair_correct_sum[:, pidx] = bias_pair_correct_sum[:, pidx] + (bias_pred == labels).float()
            score_full_disagree_sum[:, pidx] = score_full_disagree_sum[:, pidx] + (score_pred != full_argmax_pred).float()
            label_one_sum[:, pidx] = label_one_sum[:, pidx] + labels.float()
            true_sign = torch.where(labels == 1, torch.ones_like(pred_diff), -torch.ones_like(pred_diff))
            full_gap = logits[:, 1] - logits[:, 0]
            score_gap = score_logits[:, 1] - score_logits[:, 0]
            bias_gap = bias_logits[:, 1] - bias_logits[:, 0]
            full_margin_true_aligned_vals.append((true_sign * full_gap).mean())
            score_margin_true_aligned_vals.append((true_sign * score_gap).mean())
            bias_margin_true_aligned_vals.append((true_sign * bias_gap).mean())
            abs_score_gap_vals.append(score_gap.abs().mean())
            abs_bias_gap_vals.append(bias_gap.abs().mean())

            ce_losses.append(F.cross_entropy(logits, labels))
            probs = F.softmax(logits, dim=1)
            entropies.append((-(probs * (probs + 1e-8).log()).sum(dim=1)).mean())
            if sample_choices:
                sampled = torch.distributions.Categorical(probs).sample()
                pred = sampled
                pseudo_sign = torch.where(sampled == 1, torch.ones_like(pred_diff), -torch.ones_like(pred_diff))
            else:
                pred = torch.argmax(logits, dim=1)
                # Differentiable pseudo evidence during training.
                pseudo_sign = probs[:, 1] - probs[:, 0]
            corr = (pred == labels).float()
            corrects.append(corr)
            pair_correct_sum[:, pidx] = pair_correct_sum[:, pidx] + corr
            if learned_pair_mask is not None:
                mask = learned_pair_mask[:, pidx].bool()
                if mask.any():
                    learned_corrects.append(corr[mask])
                if (~mask).any():
                    nonlearned_corrects.append(corr[~mask])
            elif learned_sets is not None:
                # Slow compatibility path, retained only for external callers.
                for b in range(bsz):
                    key = tuple(sorted([int(item_i[b]), int(item_j[b])]))
                    if key in learned_sets[b]:
                        learned_corrects.append(corr[b:b+1])
                    else:
                        nonlearned_corrects.append(corr[b:b+1])
            if config.test_update_mode == "self_reconsolidate":
                conf = pseudo_confidence_from_probs(probs, config)
                pseudo_diff = (pseudo_sign * conf).clamp(-1.0, 1.0)
                state, sg = self_reconsolidate_from_choice(config, net, state, item_vecs, subject_z, item_i, item_j, pseudo_diff, conf, train_mode=train_mode)
                gate_vals.append(sg)
            eta_vals.append(eta.mean())
            if state.plastic is not None:
                plastic_vals.append(state.plastic.abs().mean())
            if config.detach_test_state:
                state = maybe_detach_state(state)

    zero = torch.tensor(0.0, device=DEVICE)
    correct_tensor = torch.stack(corrects, dim=1) if corrects else torch.empty(bsz, 0, device=DEVICE)
    return {
        "state": state,
        "choice_loss": torch.stack(ce_losses).mean() if ce_losses else zero,
        "entropy_loss": torch.stack(entropies).mean() if entropies else zero,
        "test_acc": correct_tensor.mean() if corrects else zero,
        "learned_acc": torch.cat(learned_corrects).mean() if learned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "nonlearned_acc": torch.cat(nonlearned_corrects).mean() if nonlearned_corrects else torch.tensor(float("nan"), device=DEVICE),
        "mean_test_eta": torch.stack(eta_vals).mean() if eta_vals else zero,
        "mean_test_gate": torch.stack(gate_vals).mean() if gate_vals else zero,
        "mean_test_abs_plastic": torch.stack(plastic_vals).mean() if plastic_vals else zero,
        "pair_correct_rate": (pair_correct_sum / max(1, repetitions)).detach(),
        "score_pair_correct_rate": (score_pair_correct_sum / max(1, repetitions)).detach(),
        "full_argmax_pair_correct_rate": (full_argmax_pair_correct_sum / max(1, repetitions)).detach(),
        "bias_pair_correct_rate": (bias_pair_correct_sum / max(1, repetitions)).detach(),
        "score_full_disagree_rate": (score_full_disagree_sum / max(1, repetitions)).detach(),
        "label_one_rate": (label_one_sum / max(1, repetitions)).detach(),
        "mean_full_margin_true_aligned": torch.stack(full_margin_true_aligned_vals).mean() if full_margin_true_aligned_vals else zero,
        "mean_score_margin_true_aligned": torch.stack(score_margin_true_aligned_vals).mean() if score_margin_true_aligned_vals else zero,
        "mean_bias_margin_true_aligned": torch.stack(bias_margin_true_aligned_vals).mean() if bias_margin_true_aligned_vals else zero,
        "mean_abs_score_logit_gap": torch.stack(abs_score_gap_vals).mean() if abs_score_gap_vals else zero,
        "mean_abs_bias_logit_gap": torch.stack(abs_bias_gap_vals).mean() if abs_bias_gap_vals else zero,
    }


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
    mean_hebb_eta: float
    mean_abs_plastic: float
    mean_abs_score: float
    mean_test_eta: float
    mean_test_gate: float


def run_training_episode(config: TrainConfig, net: ConstructiveRankingNet) -> EpisodeStats:
    bsz = config.bs
    item_vecs = sample_item_vectors(bsz, config.n_items, config.item_dim)
    subject_z = sample_subject_latents(bsz, config.subject_dim)
    r2i = rank_to_item_maps(bsz, config.n_items)
    true_scores = true_scores_by_item(r2i, config)
    rank_edges_batch = sample_rank_edges_batch(config, bsz)
    learning_item_edges = rank_edges_batch_to_item_edges(rank_edges_batch, r2i, random_orientation=True)
    state, learn_stats = run_learning_phase(config, net, item_vecs, subject_z, true_scores, learning_item_edges, train_mode=True)
    pairs = all_item_pairs(r2i, config)
    pairs = randomize_query_pair_orientation(pairs, enabled=config.train_query_random_orientation)
    learned_pair_mask = learned_pair_mask_from_edges(learning_item_edges, pairs)
    test = run_no_feedback_test_phase(
        config, net, state, item_vecs, subject_z, true_scores, pairs,
        train_mode=True, repetitions=max(1, config.train_test_repetitions), learned_pair_mask=learned_pair_mask, sample_choices=False,
    )
    final_state: EpisodeState = test["state"]
    loss = (
        test["choice_loss"]
        + config.lambda_recon * learn_stats.recon_loss
        + config.lambda_entropy * test["entropy_loss"]
        + config.lambda_score_l2 * final_state.scores.pow(2).mean()
        + config.lambda_plastic_l2 * learn_stats.plastic_l2
        + config.lambda_eta_l2 * (learn_stats.mean_eta.pow(2) + test["mean_test_eta"].pow(2))
    )
    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        choice_loss=float(test["choice_loss"].detach()),
        recon_loss=float(learn_stats.recon_loss.detach()),
        entropy_loss=float(test["entropy_loss"].detach()),
        test_acc=float(test["test_acc"].detach()),
        learned_acc=float(test["learned_acc"].detach()),
        nonlearned_acc=float(test["nonlearned_acc"].detach()),
        mean_gate=float(learn_stats.mean_gate.detach()),
        mean_hebb_eta=float(learn_stats.mean_eta.detach()),
        mean_abs_plastic=float(learn_stats.mean_abs_plastic.detach()),
        mean_abs_score=float(final_state.scores.abs().mean().detach()),
        mean_test_eta=float(test["mean_test_eta"].detach()),
        mean_test_gate=float(test["mean_test_gate"].detach()),
    )


def train(config: TrainConfig) -> ConstructiveRankingNet:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = ConstructiveRankingNet(config).to(DEVICE)
    if config.compile_model and hasattr(torch, "compile"):
        log("[setup] torch.compile enabled")
        net = torch.compile(net)  # type: ignore[assignment]
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2)
    use_amp = bool(config.amp and DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log(f"[setup] Device: {DEVICE}; amp={use_amp}")
    log(f"[setup] Batch size: {config.bs}; episodes: {config.nbiter}; output: {output_dir}")
    log(f"[setup] mode: memory_mode={config.memory_mode}; use_rnn={config.use_rnn}; use_hebbian={config.use_hebbian}; test_update_mode={config.test_update_mode}")
    log(f"[setup] hidden_size={config.hidden_size}; rnn_hidden_size={config.rnn_hidden_size}")
    log(f"[setup] Parameter count: {sum(p.numel() for p in net.parameters())}")
    if config.gpu_fast:
        log(f"[gpu-fast] enabled: graph_pool_size={config.graph_pool_size}, fast_batch_graph_sampling={config.fast_batch_graph_sampling}, fast_per_subject_orders={config.fast_per_subject_orders}")
        ensure_rank_graph_pool(config)
        _all_rank_pairs_tensor(config)
    log("[task] Inner-loop reward: none. Test phase: no-feedback; internal updates use only retrieval/self-choice pseudo evidence.")
    log_path = output_dir / "train_log.csv"
    fieldnames = [
        "episode", "loss", "choice_loss", "recon_loss", "entropy_loss", "test_acc", "learned_acc", "nonlearned_acc",
        "mean_gate", "mean_hebb_eta", "mean_abs_plastic", "mean_abs_score", "mean_test_eta", "mean_test_gate",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    start = time.time()
    last_stats: EpisodeStats | None = None
    iterator = tqdm(range(config.nbiter), desc="training episodes", unit="episode", dynamic_ncols=True, file=sys.stdout)
    fw_time_acc = 0.0
    bw_time_acc = 0.0
    opt_time_acc = 0.0
    for episode in iterator:
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=use_amp):
            stats = run_training_episode(config, net)
        if DEVICE.type == "cuda" and config.timing_breakdown:
            torch.cuda.synchronize()
        t1 = time.time()
        scaler.scale(stats.loss).backward()
        if DEVICE.type == "cuda" and config.timing_breakdown:
            torch.cuda.synchronize()
        t2 = time.time()
        if config.gc > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        scaler.step(optimizer)
        scaler.update()
        if DEVICE.type == "cuda" and config.timing_breakdown:
            torch.cuda.synchronize()
        t3 = time.time()
        fw_time_acc += t1 - t0
        bw_time_acc += t2 - t1
        opt_time_acc += t3 - t2
        last_stats = stats
        if episode % config.pe == 0 or episode == config.nbiter - 1:
            elapsed = time.time() - start
            msg = (
                f"Episode {episode} ==== {elapsed:.2f}s | loss={stats.loss_value:.4f} choice={stats.choice_loss:.4f} recon={stats.recon_loss:.4f} | "
                f"acc={stats.test_acc:.3f} learned={stats.learned_acc:.3f} nonlearned={stats.nonlearned_acc:.3f} | "
                f"gate={stats.mean_gate:.3f} eta={stats.mean_hebb_eta:.4f} test_eta={stats.mean_test_eta:.4f} |plastic|={stats.mean_abs_plastic:.4f} |score|={stats.mean_abs_score:.3f}"
            )
            if config.timing_breakdown:
                denom = config.pe if episode != 0 else 1
                msg += f" | timing/ep fw={fw_time_acc/denom:.3f}s bw={bw_time_acc/denom:.3f}s opt={opt_time_acc/denom:.3f}s"
                fw_time_acc = bw_time_acc = opt_time_acc = 0.0
            log(msg)
            start = time.time()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, stats.loss_value, stats.choice_loss, stats.recon_loss, stats.entropy_loss,
                stats.test_acc, stats.learned_acc, stats.nonlearned_acc, stats.mean_gate, stats.mean_hebb_eta,
                stats.mean_abs_plastic, stats.mean_abs_score, stats.mean_test_eta, stats.mean_test_gate,
            ])
        if config.save_every > 0 and episode > 0 and episode % config.save_every == 0:
            torch.save({"model_state": net.state_dict(), "config": asdict(config)}, output_dir / "net_unified.pt")
            log(f"[save] checkpoint: {output_dir / 'net_unified.pt'}")
    torch.save({"model_state": net.state_dict(), "config": asdict(config)}, output_dir / "net_unified.pt")
    log(f"[save] final checkpoint: {output_dir / 'net_unified.pt'}")
    if last_stats is not None:
        log(f"[done] final train acc={last_stats.test_acc:.3f}, loss={last_stats.loss_value:.4f}")
    return net


def self_consistency_from_scores(scores: np.ndarray) -> tuple[float, int]:
    n_items = scores.shape[0]
    circular = 0
    for a, b, c in combinations(range(n_items), 3):
        ab = scores[a] > scores[b]
        bc = scores[b] > scores[c]
        ac = scores[a] > scores[c]
        if (ab and bc and not ac) or ((not ab) and (not bc) and ac):
            circular += 1
    max_triads = (n_items**3 - 4 * n_items) // 24 if n_items % 2 == 0 else (n_items**3 - n_items) // 24
    return 1.0 - circular / max_triads, circular


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
    net.eval()
    old_log_beta = None
    if config.eval_beta_override and config.eval_beta_override > 0:
        old_log_beta = net.log_beta.detach().clone()
        net.log_beta.data.fill_(math.log(config.eval_beta_override))
    n_subj = config.eval_subjects
    reps = config.eval_repetitions
    one_item_set = sample_item_vectors(1, config.n_items, config.item_dim)
    item_vecs = one_item_set.expand(n_subj, -1, -1).contiguous()
    subject_z = sample_subject_latents(n_subj, config.subject_dim)
    r2i = torch.arange(config.n_items, device=DEVICE)[None, :].expand(n_subj, -1).contiguous()
    true_scores = true_scores_by_item(r2i, config)
    learning_edges, _ = rank_edges_to_item_edges(PAPER_LEARNING_PAIRS_RANK, r2i, random_orientation=False)
    state, learn_stats = run_learning_phase(config, net, item_vecs, subject_z, true_scores, learning_edges, train_mode=False)
    pairs = _all_rank_pairs_tensor(config)[None, :, :].expand(n_subj, -1, -1).contiguous()
    pairs = randomize_query_pair_orientation(pairs, enabled=config.eval_query_random_orientation)
    test = run_no_feedback_test_phase(
        config, net, state, item_vecs, subject_z, true_scores, pairs,
        train_mode=False, repetitions=reps, learned_sets=None, sample_choices=True,
    )
    pair_acc = test["pair_correct_rate"].cpu().numpy()
    score_pair_acc = test["score_pair_correct_rate"].cpu().numpy()
    full_argmax_pair_acc = test["full_argmax_pair_correct_rate"].cpu().numpy()
    bias_pair_acc = test["bias_pair_correct_rate"].cpu().numpy()
    score_full_disagree = test["score_full_disagree_rate"].cpu().numpy()
    label_one_rate = test["label_one_rate"].cpu().numpy()
    pair_list = config.all_rank_pairs
    paper_pair_set = {tuple(p) for p in PAPER_LEARNING_PAIRS_RANK}
    learned_idx = np.array([idx for idx, p in enumerate(pair_list) if p in paper_pair_set], dtype=int)
    nonlearned_idx = np.array([idx for idx, p in enumerate(pair_list) if p not in paper_pair_set], dtype=int)
    pair_error_prop = 1.0 - pair_acc
    scores_np = test["state"].scores.detach().cpu().numpy()
    subject_orders, self_consistency, circular_counts, choice_self_consistency, choice_circular_counts = [], [], [], [], []
    for b in range(n_subj):
        order = list(np.argsort(scores_np[b]))
        subject_orders.append(order)
        sc, circ = self_consistency_from_scores(scores_np[b])
        self_consistency.append(sc)
        circular_counts.append(circ)
        csc, ccirc = self_consistency_from_majority(pair_acc[b], pair_list, config.n_items)
        choice_self_consistency.append(csc)
        choice_circular_counts.append(ccirc)
    taus = [kendall_tau_order(subject_orders[i], subject_orders[j]) for i, j in combinations(range(n_subj), 2)]
    distance_acc = {}
    for dist in range(1, config.n_items):
        idx = [pidx for pidx, (a, b) in enumerate(pair_list) if b - a == dist]
        distance_acc[str(dist)] = float(pair_acc[:, idx].mean())
    summary = {
        "eval_subjects": n_subj,
        "eval_repetitions": reps,
        "memory_mode": config.memory_mode,
        "use_rnn": config.use_rnn,
        "use_hebbian": config.use_hebbian,
        "test_update_mode": config.test_update_mode,
        "train_query_random_orientation": config.train_query_random_orientation,
        "eval_query_random_orientation": config.eval_query_random_orientation,
        "overall_accuracy": float(pair_acc.mean()),
        "overall_accuracy_flipped": float(1.0 - pair_acc.mean()),
        "score_only_overall_accuracy": float(score_pair_acc.mean()),
        "score_only_overall_accuracy_flipped": float(1.0 - score_pair_acc.mean()),
        "full_argmax_overall_accuracy": float(full_argmax_pair_acc.mean()),
        "bias_only_overall_accuracy": float(bias_pair_acc.mean()),
        "bias_only_overall_accuracy_flipped": float(1.0 - bias_pair_acc.mean()),
        "score_full_prediction_disagree_rate": float(score_full_disagree.mean()),
        "mean_true_label_is_class1_j_higher": float(label_one_rate.mean()),
        "mean_full_margin_true_aligned": float(test["mean_full_margin_true_aligned"].cpu()),
        "mean_score_margin_true_aligned": float(test["mean_score_margin_true_aligned"].cpu()),
        "mean_bias_margin_true_aligned": float(test["mean_bias_margin_true_aligned"].cpu()),
        "mean_abs_score_logit_gap": float(test["mean_abs_score_logit_gap"].cpu()),
        "mean_abs_bias_logit_gap": float(test["mean_abs_bias_logit_gap"].cpu()),
        "bias_to_score_gap_ratio": float((test["mean_abs_bias_logit_gap"] / (test["mean_abs_score_logit_gap"] + 1e-8)).cpu()),
        "learned_pairs_accuracy": float(pair_acc[:, learned_idx].mean()),
        "nonlearned_pairs_accuracy": float(pair_acc[:, nonlearned_idx].mean()),
        "consistent_error_subjects_80pct": int((pair_error_prop >= 0.8).any(axis=1).sum()),
        "consistent_error_subjects_80pct_ratio": float((pair_error_prop >= 0.8).any(axis=1).mean()),
        "consistent_error_subjects_100pct": int((pair_error_prop >= 1.0).any(axis=1).sum()),
        "consistent_error_subjects_100pct_ratio": float((pair_error_prop >= 1.0).any(axis=1).mean()),
        "mean_self_consistency_from_scores": float(np.mean(self_consistency)),
        "mean_circular_triads_from_scores": float(np.mean(circular_counts)),
        "mean_self_consistency_from_majority_choices": float(np.mean(choice_self_consistency)),
        "mean_circular_triads_from_majority_choices": float(np.mean(choice_circular_counts)),
        "mean_inter_subject_kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "distance_accuracy": distance_acc,
        "reconstruction_loss_after_learning": float(learn_stats.recon_loss.cpu()),
        "mean_update_gate_learning": float(learn_stats.mean_gate.cpu()),
        "mean_hebb_eta_learning": float(learn_stats.mean_eta.cpu()),
        "mean_abs_plastic_weight_learning": float(learn_stats.mean_abs_plastic.cpu()),
        "mean_test_eta": float(test["mean_test_eta"].cpu()),
        "mean_test_gate": float(test["mean_test_gate"].cpu()),
        "mean_test_abs_plastic": float(test["mean_test_abs_plastic"].cpu()),
        "mean_abs_final_score": float(test["state"].scores.abs().mean().cpu()),
    }
    if old_log_beta is not None:
        net.log_beta.data.copy_(old_log_beta)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified stage01-04 constructive learner with optional plastic Hebbian RNN and GPU support.")
    p.add_argument("--nbiter", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=8)
    p.add_argument("--item-dim", type=int, default=15)
    p.add_argument("--subject-dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="outputs_unified")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--gpu-fast", action=argparse.BooleanOptionalAction, default=True, help="Enable GPU-fast vectorized graph sampling/order generation. Does not change learning logic.")
    p.add_argument("--graph-pool-size", type=int, default=8192, help="Number of valid sparse rank graphs precomputed once for fast batch sampling.")
    p.add_argument("--fast-batch-graph-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fast-per-subject-orders", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timing-breakdown", action="store_true", help="Print coarse episode timing every print interval; mainly for performance diagnosis.")

    p.add_argument("--eval-subjects", type=int, default=8)
    p.add_argument("--eval-repetitions", type=int, default=5)
    p.add_argument("--relation-noise", type=float, default=0.03)
    p.add_argument("--eval-relation-noise", type=float, default=0.0)
    p.add_argument("--edge-dropout", type=float, default=0.0)
    p.add_argument("--eval-edge-dropout", type=float, default=0.0)
    p.add_argument("--init-score-noise", type=float, default=0.0)
    p.add_argument("--subject-scale", type=float, default=1.0)
    p.add_argument("--update-scale", type=float, default=1.0)
    p.add_argument("--paper-graph-train-prob", type=float, default=0.0)
    p.add_argument("--eval-beta-override", type=float, default=0.0)
    p.add_argument("--rnn-choice-gain", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=2.0)

    p.add_argument("--memory-mode", choices=["direct", "edge_online", "edge_block", "edge_hybrid"], default="direct")
    p.add_argument("--forget-rate", type=float, default=0.0)
    p.add_argument("--memory-capacity", type=float, default=10.0)
    p.add_argument("--memory-encoding-noise", type=float, default=0.0)
    p.add_argument("--eval-memory-encoding-noise", type=float, default=0.0)
    p.add_argument("--memory-attention-bias", type=float, default=0.0)
    p.add_argument("--distance-salience", type=float, default=0.0)
    p.add_argument("--subject-attention-scale", type=float, default=0.0)
    p.add_argument("--item-attention-scale", type=float, default=0.0)
    p.add_argument("--pair-attention-scale", type=float, default=0.0)
    p.add_argument("--distance-attention-scale", type=float, default=0.0)
    p.add_argument("--reliability-temperature", type=float, default=1.0)
    p.add_argument("--reconsolidation-strength", type=float, default=0.0)
    p.add_argument("--reconsolidation-power", type=float, default=1.0)
    p.add_argument("--reconsolidation-refresh", type=float, default=0.0)
    p.add_argument("--schema-encoding-bias", type=float, default=0.0)
    p.add_argument("--replay-steps", type=int, default=0)
    p.add_argument("--replay-strength", type=float, default=1.0)
    p.add_argument("--replay-temperature", type=float, default=1.0)
    p.add_argument("--schema-sweeps", type=int, default=1)

    rnn = p.add_mutually_exclusive_group()
    rnn.add_argument("--use-rnn", dest="use_rnn", action="store_true")
    rnn.add_argument("--no-rnn", dest="use_rnn", action="store_false")
    # Backward-compatible aliases from the previous script.
    rnn.add_argument("--use-plastic-rnn", dest="use_rnn", action="store_true")
    rnn.add_argument("--no-plastic-rnn", dest="use_rnn", action="store_false")
    p.set_defaults(use_rnn=False)
    hebb = p.add_mutually_exclusive_group()
    hebb.add_argument("--use-hebbian", dest="use_hebbian", action="store_true")
    hebb.add_argument("--no-hebbian", dest="use_hebbian", action="store_false")
    hebb.add_argument("--disable-hebbian-plasticity", dest="use_hebbian", action="store_false")
    p.set_defaults(use_hebbian=True)
    p.add_argument("--rnn-hidden-size", type=int, default=64)
    p.add_argument("--rnn-input-gain", type=float, default=1.0)
    p.add_argument("--rnn-rec-gain", type=float, default=1.0)
    p.add_argument("--fast-weight-gain", type=float, default=0.35)
    p.add_argument("--hebb-eta", type=float, default=0.06)
    p.add_argument("--plastic-decay", type=float, default=0.96)
    p.add_argument("--plastic-clip", type=float, default=3.0)
    p.add_argument("--detach-plastic-state", action="store_true")
    p.add_argument("--rnn-dropout", type=float, default=0.0)

    p.add_argument("--test-update-mode", choices=["frozen", "hidden", "self_reconsolidate"], default="self_reconsolidate")
    p.add_argument("--train-test-repetitions", type=int, default=1)
    p.add_argument("--test-eta-scale", type=float, default=0.35)
    p.add_argument("--test-score-update-scale", type=float, default=0.10)
    p.add_argument("--test-memory-update-strength", type=float, default=0.08)
    p.add_argument("--test-pseudo-confidence-power", type=float, default=1.0)
    p.add_argument("--test-order-shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detach-test-state", action="store_true")

    p.add_argument("--lambda-recon", type=float, default=0.5)
    p.add_argument("--lambda-entropy", type=float, default=0.005)
    p.add_argument("--lambda-score-l2", type=float, default=1e-4)
    p.add_argument("--lambda-plastic-l2", type=float, default=1e-5)
    p.add_argument("--lambda-eta-l2", type=float, default=1e-4)

    # DBG helpers. These do not change the model unless explicitly enabled.
    p.add_argument("--dbg-eval-components", action="store_true", help="Print extra component metrics: score-only, RNN-bias-only, flipped accuracy, margins.")
    p.add_argument("--train-query-random-orientation", action=argparse.BooleanOptionalAction, default=True, help="Randomly flip training test-query orientation to prevent class-position shortcuts. Enabled by default; use --no-train-query-random-orientation to reproduce old behavior.")
    p.add_argument("--eval-query-random-orientation", action=argparse.BooleanOptionalAction, default=True, help="Randomly flip paper-task evaluation query orientation. Enabled by default; use --no-eval-query-random-orientation to reproduce old label=class1 behavior.")
    p.add_argument("--eval-only-checkpoint", default="", help="Load a checkpoint and run paper-task evaluation only. Architecture args must match the checkpoint.")
    return p.parse_args()


def config_from_args(a: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        rngseed=a.seed, nbiter=a.nbiter, bs=a.batch_size, hidden_size=a.hidden_size,
        item_dim=a.item_dim, subject_dim=a.subject_dim, output_dir=a.output_dir,
        save_every=a.save_every, pe=a.print_every, num_threads=a.num_threads,
        device=a.device, amp=a.amp, compile_model=a.compile_model,
        gpu_fast=a.gpu_fast, graph_pool_size=a.graph_pool_size,
        fast_batch_graph_sampling=(a.fast_batch_graph_sampling and a.gpu_fast),
        fast_per_subject_orders=(a.fast_per_subject_orders and a.gpu_fast),
        timing_breakdown=a.timing_breakdown,
        eval_subjects=a.eval_subjects, eval_repetitions=a.eval_repetitions,
        relation_noise=a.relation_noise, eval_relation_noise=a.eval_relation_noise,
        edge_dropout=a.edge_dropout, eval_edge_dropout=a.eval_edge_dropout,
        init_score_noise=a.init_score_noise, subject_scale=a.subject_scale,
        update_scale=a.update_scale, paper_graph_train_prob=a.paper_graph_train_prob,
        eval_beta_override=a.eval_beta_override, rnn_choice_gain=a.rnn_choice_gain,
        lr=a.lr, gc=a.grad_clip,
        memory_mode=a.memory_mode, forget_rate=a.forget_rate, memory_capacity=a.memory_capacity,
        memory_encoding_noise=a.memory_encoding_noise, eval_memory_encoding_noise=a.eval_memory_encoding_noise,
        memory_attention_bias=a.memory_attention_bias, distance_salience=a.distance_salience,
        subject_attention_scale=a.subject_attention_scale, item_attention_scale=a.item_attention_scale,
        pair_attention_scale=a.pair_attention_scale, distance_attention_scale=a.distance_attention_scale,
        reliability_temperature=a.reliability_temperature,
        reconsolidation_strength=a.reconsolidation_strength, reconsolidation_power=a.reconsolidation_power,
        reconsolidation_refresh=a.reconsolidation_refresh, schema_encoding_bias=a.schema_encoding_bias,
        replay_steps=a.replay_steps, replay_strength=a.replay_strength, replay_temperature=a.replay_temperature,
        schema_sweeps=a.schema_sweeps,
        use_rnn=a.use_rnn, use_hebbian=a.use_hebbian, rnn_hidden_size=a.rnn_hidden_size,
        rnn_input_gain=a.rnn_input_gain, rnn_rec_gain=a.rnn_rec_gain, fast_weight_gain=a.fast_weight_gain,
        hebb_eta=a.hebb_eta, plastic_decay=a.plastic_decay, plastic_clip=a.plastic_clip,
        detach_plastic_state=a.detach_plastic_state, rnn_dropout=a.rnn_dropout,
        test_update_mode=a.test_update_mode, train_test_repetitions=a.train_test_repetitions,
        test_eta_scale=a.test_eta_scale, test_score_update_scale=a.test_score_update_scale,
        test_memory_update_strength=a.test_memory_update_strength,
        test_pseudo_confidence_power=a.test_pseudo_confidence_power,
        test_order_shuffle=a.test_order_shuffle, detach_test_state=a.detach_test_state,
        dbg_eval_components=a.dbg_eval_components, train_query_random_orientation=a.train_query_random_orientation,
        eval_query_random_orientation=a.eval_query_random_orientation,
        eval_only_checkpoint=a.eval_only_checkpoint,
        lambda_recon=a.lambda_recon, lambda_entropy=a.lambda_entropy,
        lambda_score_l2=a.lambda_score_l2, lambda_plastic_l2=a.lambda_plastic_l2, lambda_eta_l2=a.lambda_eta_l2,
    )


def main() -> None:
    global DEVICE
    args = parse_args()
    DEVICE = configure_device(args.device)
    configure_threads(args.num_threads, DEVICE)
    config = config_from_args(args)
    set_seed(config.rngseed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_unified.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    if config.eval_only_checkpoint:
        net = ConstructiveRankingNet(config).to(DEVICE)
        ckpt = torch.load(config.eval_only_checkpoint, map_location=DEVICE)
        state_dict = ckpt.get("model_state", ckpt)
        net.load_state_dict(state_dict, strict=True)
        log(f"[load] eval-only checkpoint: {config.eval_only_checkpoint}")
    else:
        net = train(config)
    summary = evaluate_paper_task(config, net)
    with open(out / "paper_task_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("[eval] behavioral-paper task summary:")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
