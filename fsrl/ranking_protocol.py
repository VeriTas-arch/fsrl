"""Versioned task contract for the Liu constructive-ranking benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "liu_v1.json"
)


@dataclass(frozen=True)
class SupportTrial:
    left_item: int
    right_item: int
    higher_item: int
    lower_item: int
    signed_magnitude: float
    block_index: int
    encoding_reliability: float = 1.0


@dataclass(frozen=True)
class QueryTrial:
    left_item: int
    right_item: int
    correct_action: int
    symbolic_distance: int
    block_index: int


@dataclass(frozen=True)
class RankingProtocol:
    protocol_id: str
    item_labels: tuple[str, ...]
    true_order_high_to_low: tuple[int, ...]
    support_pairs_higher_lower: tuple[tuple[int, int], ...]
    support_blocks: int
    query_blocks: int
    human_targets: dict

    @property
    def n_items(self) -> int:
        return len(self.item_labels)

    @property
    def support_trials(self) -> int:
        return self.support_blocks * len(self.support_pairs_higher_lower)

    @property
    def query_trials(self) -> int:
        return self.query_blocks * (self.n_items * (self.n_items - 1) // 2)

    @property
    def learned_pairs(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            tuple(sorted(pair)) for pair in self.support_pairs_higher_lower
        )

    def support_schedule(self, rng: np.random.Generator) -> tuple[SupportTrial, ...]:
        rank = {
            item: position for position, item in enumerate(self.true_order_high_to_low)
        }
        trials: list[SupportTrial] = []
        for block_index in range(self.support_blocks):
            order = rng.permutation(len(self.support_pairs_higher_lower))
            for pair_index in order:
                higher, lower = self.support_pairs_higher_lower[int(pair_index)]
                magnitude = (rank[lower] - rank[higher]) / float(self.n_items - 1)
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
                    )
                )
        return tuple(trials)

    def query_schedule(self, rng: np.random.Generator) -> tuple[QueryTrial, ...]:
        rank = {
            item: position for position, item in enumerate(self.true_order_high_to_low)
        }
        base_pairs = tuple(combinations(range(self.n_items), 2))
        trials: list[QueryTrial] = []
        for block_index in range(self.query_blocks):
            order = rng.permutation(len(base_pairs))
            for pair_index in order:
                first, second = base_pairs[int(pair_index)]
                if rng.random() < 0.5:
                    left, right = first, second
                else:
                    left, right = second, first
                correct_action = 1 if rank[left] < rank[right] else 0
                trials.append(
                    QueryTrial(
                        left_item=left,
                        right_item=right,
                        correct_action=correct_action,
                        symbolic_distance=abs(rank[left] - rank[right]),
                        block_index=block_index,
                    )
                )
        return tuple(trials)


def load_ranking_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> RankingProtocol:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    labels = tuple(raw["item_labels"])
    if raw["n_items"] != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("Protocol item labels must be unique and match n_items")
    label_to_item = {label: index for index, label in enumerate(labels)}
    true_order = tuple(label_to_item[label] for label in raw["true_order_high_to_low"])
    support_pairs = tuple(
        (label_to_item[higher], label_to_item[lower])
        for higher, lower in raw["support"]["pairs_higher_lower"]
    )

    protocol = RankingProtocol(
        protocol_id=raw["protocol_id"],
        item_labels=labels,
        true_order_high_to_low=true_order,
        support_pairs_higher_lower=support_pairs,
        support_blocks=int(raw["support"]["blocks"]),
        query_blocks=int(raw["query"]["blocks"]),
        human_targets=raw["human_targets"],
    )
    if protocol.support_trials != int(raw["support"]["trials"]):
        raise ValueError("Protocol support trial count is inconsistent")
    if protocol.query_trials != int(raw["query"]["trials"]):
        raise ValueError("Protocol query trial count is inconsistent")
    if set(protocol.true_order_high_to_low) != set(range(protocol.n_items)):
        raise ValueError("Protocol true order must be a permutation of all items")
    for higher, lower in protocol.support_pairs_higher_lower:
        if true_order.index(higher) >= true_order.index(lower):
            raise ValueError("Support pair direction contradicts the true order")
    return protocol
