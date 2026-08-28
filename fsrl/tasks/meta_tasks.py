"""Legacy constructor adapter for the historical training distribution."""

from __future__ import annotations

from pathlib import Path

from .holdouts import graph_signature_for_protocol, registered_holdout_signatures
from .sparse_ranking import (
    GenericRankingTaskGenerator as _GenericRankingTaskGenerator,
)
from .sparse_ranking import (
    GraphSignature,
    RankingEpisode,
    graph_is_connected,
)
from .sparse_ranking import (
    graph_signature_from_protocol as _graph_signature_from_protocol,
)


def graph_signature_from_protocol(path: Path | str | None = None) -> GraphSignature:
    if path is None:
        return graph_signature_for_protocol("liu_v1")
    from .protocol import load_ranking_protocol

    return _graph_signature_from_protocol(load_ranking_protocol(path))


def liu_graph_signature() -> GraphSignature:
    """Return the legacy v1 signature used by developmental training."""

    return graph_signature_from_protocol()


def held_out_liu_graph_signatures() -> frozenset[GraphSignature]:
    """Return the two prospectively excluded registered graph signatures."""

    return registered_holdout_signatures()


class GenericRankingTaskGenerator(_GenericRankingTaskGenerator):
    """Backward-compatible constructor with the historical exclusion switch."""

    def __init__(self, *args, exclude_liu_graph: bool = True, **kwargs) -> None:
        if "excluded_signatures" in kwargs:
            raise TypeError("use either exclude_liu_graph or excluded_signatures")
        excluded = held_out_liu_graph_signatures() if exclude_liu_graph else ()
        super().__init__(*args, excluded_signatures=excluded, **kwargs)


__all__ = [
    "GenericRankingTaskGenerator",
    "GraphSignature",
    "RankingEpisode",
    "graph_is_connected",
    "graph_signature_from_protocol",
    "held_out_liu_graph_signatures",
    "liu_graph_signature",
]
