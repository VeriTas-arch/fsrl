"""Generic sparse ranking episodes for held-out-graph meta-training."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..subject_encoding import (
    SubjectEncodingConfig,
    SubjectEncodingState,
    sample_subject_encoding_states,
)
from .protocol import QueryTrial, RankingProtocol, SupportTrial

GraphSignature = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RankingEpisode:
    item_codes: np.ndarray
    true_order_high_to_low: tuple[int, ...]
    graph_rank_pairs: GraphSignature
    support_trials: tuple[SupportTrial, ...]
    query_trials: tuple[QueryTrial, ...]
    subject_encoding: SubjectEncodingState


def graph_signature_from_protocol(protocol: RankingProtocol) -> GraphSignature:
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    return tuple(
        sorted(
            tuple(sorted((rank[higher], rank[lower])))
            for higher, lower in protocol.support_pairs_higher_lower
        )
    )


def graph_is_connected(n_items: int, edges: GraphSignature) -> bool:
    adjacency = [set() for _ in range(n_items)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == n_items


class GenericRankingTaskGenerator:
    def __init__(
        self,
        *,
        n_items: int = 8,
        cue_size: int = 15,
        min_edges: int = 7,
        max_edges: int = 10,
        support_blocks: int = 4,
        excluded_signatures: Collection[GraphSignature] = (),
        subject_encoding_config: SubjectEncodingConfig | None = None,
        subject_encoding_mode: str = "stable_omission",
    ) -> None:
        if n_items < 3:
            raise ValueError("n_items must be at least three")
        max_possible = n_items * (n_items - 1) // 2
        if not n_items - 1 <= min_edges <= max_edges <= max_possible:
            raise ValueError("edge range must describe connected sparse graphs")
        if cue_size < 2:
            raise ValueError("cue_size must be at least two")
        if support_blocks < 1:
            raise ValueError("support_blocks must be positive")
        self.n_items = n_items
        self.cue_size = cue_size
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.support_blocks = support_blocks
        self.subject_encoding_config = (
            subject_encoding_config or SubjectEncodingConfig()
        )
        if subject_encoding_mode not in {"stable_attenuation", "stable_omission"}:
            raise ValueError(f"unknown subject encoding mode: {subject_encoding_mode}")
        self.subject_encoding_mode = subject_encoding_mode
        self.excluded_signatures = frozenset(excluded_signatures)

    def sample(
        self, rng: np.random.Generator, *, n_edges: int | None = None
    ) -> RankingEpisode:
        if n_edges is None:
            n_edges = int(rng.integers(self.min_edges, self.max_edges + 1))
        if not self.min_edges <= n_edges <= self.max_edges:
            raise ValueError("n_edges lies outside the configured range")
        graph = self._sample_graph(rng, n_edges)
        true_order = tuple(int(item) for item in rng.permutation(self.n_items))
        codes = self._sample_item_codes(rng)
        subject_encoding = sample_subject_encoding_states(
            rng, 1, self.n_items, self.subject_encoding_config
        )[0]
        support = self._support_schedule(rng, graph, true_order, subject_encoding)
        query = self._query_schedule(rng, true_order)
        return RankingEpisode(
            codes, true_order, graph, support, query, subject_encoding
        )

    def _sample_graph(self, rng: np.random.Generator, n_edges: int) -> GraphSignature:
        all_pairs = tuple(combinations(range(self.n_items), 2))
        for _ in range(1000):
            node_order = [int(node) for node in rng.permutation(self.n_items)]
            connected = [node_order[0]]
            edges = set()
            for node in node_order[1:]:
                parent = int(rng.choice(connected))
                edges.add(tuple(sorted((parent, node))))
                connected.append(node)
            remaining = [pair for pair in all_pairs if pair not in edges]
            rng.shuffle(remaining)
            edges.update(remaining[: n_edges - len(edges)])
            signature = tuple(sorted(edges))
            distances = {abs(first - second) for first, second in signature}
            if signature not in self.excluded_signatures and len(distances) >= 2:
                return signature
        raise RuntimeError("could not sample an admissible sparse ranking graph")

    def _sample_item_codes(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(100):
            codes: list[np.ndarray] = []
            for _ in range(10000):
                candidate = (
                    rng.integers(0, 2, self.cue_size, dtype=np.int8) * 2 - 1
                ).astype(np.float32)
                if all(np.mean(previous == candidate) <= 0.66 for previous in codes):
                    codes.append(candidate)
                    if len(codes) == self.n_items:
                        return np.stack(codes)
        raise RuntimeError("could not sample a sufficiently distinct cue set")

    def _support_schedule(
        self,
        rng: np.random.Generator,
        graph: GraphSignature,
        true_order: tuple[int, ...],
        subject_encoding: SubjectEncodingState,
    ) -> tuple[SupportTrial, ...]:
        trials = []
        relation_gains = {}
        for high_rank, low_rank in graph:
            higher = true_order[high_rank]
            lower = true_order[low_rank]
            probability = subject_encoding.relation_reliability(
                higher, lower, low_rank - high_rank
            )
            relation_gains[(high_rank, low_rank)] = (
                probability
                if self.subject_encoding_mode == "stable_attenuation"
                else float(rng.random() < probability)
            )
        for block_index in range(self.support_blocks):
            for edge_index in rng.permutation(len(graph)):
                high_rank, low_rank = graph[int(edge_index)]
                higher = true_order[high_rank]
                lower = true_order[low_rank]
                symbolic_distance = low_rank - high_rank
                magnitude = symbolic_distance / float(self.n_items - 1)
                reliability = relation_gains[(high_rank, low_rank)]
                if rng.random() < 0.5:
                    left, right, signed = higher, lower, magnitude
                else:
                    left, right, signed = lower, higher, -magnitude
                trials.append(
                    SupportTrial(
                        left_item=left,
                        right_item=right,
                        higher_item=higher,
                        lower_item=lower,
                        signed_magnitude=float(signed),
                        block_index=block_index,
                        encoding_reliability=reliability,
                    )
                )
        return tuple(trials)

    def _query_schedule(
        self, rng: np.random.Generator, true_order: tuple[int, ...]
    ) -> tuple[QueryTrial, ...]:
        rank = {item: position for position, item in enumerate(true_order)}
        pairs = list(combinations(range(self.n_items), 2))
        rng.shuffle(pairs)
        trials = []
        for first, second in pairs:
            if rng.random() < 0.5:
                left, right = first, second
            else:
                left, right = second, first
            trials.append(
                QueryTrial(
                    left_item=left,
                    right_item=right,
                    correct_action=1 if rank[left] < rank[right] else 0,
                    symbolic_distance=abs(rank[left] - rank[right]),
                    block_index=0,
                )
            )
        return tuple(trials)
