"""Graph and protocol transforms shared by relational transport evaluations."""

from __future__ import annotations

from itertools import combinations
from typing import cast

from .protocol import RankingProtocol
from .sparse_ranking import GraphSignature, graph_is_connected


def graph_descriptor(edges: GraphSignature, n_items: int = 8) -> dict:
    normalized = cast(
        GraphSignature,
        tuple(sorted(tuple(sorted(edge)) for edge in edges)),
    )
    adjacency = [set() for _ in range(n_items)]
    for first, second in normalized:
        adjacency[first].add(second)
        adjacency[second].add(first)
    distances = []
    for source in range(n_items):
        shortest = {source: 0}
        frontier = [source]
        while frontier:
            item = frontier.pop(0)
            for neighbor in adjacency[item]:
                if neighbor not in shortest:
                    shortest[neighbor] = shortest[item] + 1
                    frontier.append(neighbor)
        if len(shortest) != n_items:
            diameter = None
            break
        distances.extend(shortest.values())
    else:
        diameter = max(distances)
    triangles = sum(
        int(
            tuple(sorted((first, second))) in normalized
            and tuple(sorted((first, third))) in normalized
            and tuple(sorted((second, third))) in normalized
        )
        for first, second, third in combinations(range(n_items), 3)
    )
    return {
        "edge_count": len(normalized),
        "connected": graph_is_connected(n_items, normalized),
        "distance_multiset": sorted(
            abs(first - second) for first, second in normalized
        ),
        "sorted_degree_sequence": sorted(len(neighbors) for neighbors in adjacency),
        "triangle_count": triangles,
        "diameter": diameter,
    }


def protocol_for_graph(base: RankingProtocol, graph: dict) -> RankingProtocol:
    rank_edges = tuple(map(tuple, graph["rank_edges"]))
    support_pairs = tuple(
        (base.true_order_high_to_low[higher], base.true_order_high_to_low[lower])
        for higher, lower in rank_edges
    )
    return RankingProtocol(
        protocol_id=f"liu-support-topology-v1-{graph['graph_id']}",
        item_labels=base.item_labels,
        true_order_high_to_low=base.true_order_high_to_low,
        support_pairs_higher_lower=support_pairs,
        support_blocks=base.support_blocks,
        query_blocks=base.query_blocks,
        human_targets={},
    )
