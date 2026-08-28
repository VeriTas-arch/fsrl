"""Explicit current task-distribution holdouts selected by protocol identity."""

from __future__ import annotations

from .protocol_catalog import load_registered_protocol
from .sparse_ranking import GraphSignature, graph_signature_from_protocol


def graph_signature_for_protocol(protocol_id: str) -> GraphSignature:
    """Return one rank-graph signature from a registered semantic protocol ID."""

    return graph_signature_from_protocol(load_registered_protocol(protocol_id))


def registered_holdout_signatures(
    protocol_ids: tuple[str, ...] = ("liu_v1", "liu_v2"),
) -> frozenset[GraphSignature]:
    """Return the explicit current holdout set for generic meta-training."""

    return frozenset(graph_signature_for_protocol(value) for value in protocol_ids)
