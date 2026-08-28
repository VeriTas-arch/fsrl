"""Task contracts separated from model and execution infrastructure."""

from .evidence import broader_local_admission
from .holdouts import registered_holdout_signatures
from .protocol import QueryTrial, RankingProtocol, SupportTrial, load_ranking_protocol
from .protocol_catalog import load_registered_protocol, protocol_path
from .sparse_ranking import (
    GenericRankingTaskGenerator,
    GraphSignature,
    RankingEpisode,
    graph_is_connected,
    graph_signature_from_protocol,
)

__all__ = [
    "GenericRankingTaskGenerator",
    "GraphSignature",
    "QueryTrial",
    "RankingEpisode",
    "RankingProtocol",
    "SupportTrial",
    "broader_local_admission",
    "graph_is_connected",
    "graph_signature_from_protocol",
    "load_ranking_protocol",
    "load_registered_protocol",
    "protocol_path",
    "registered_holdout_signatures",
]
