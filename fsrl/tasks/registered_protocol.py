"""Legacy v1-default adapter; current code uses :mod:`protocol_catalog`."""

from pathlib import Path

from .protocol import (
    QueryTrial,
    RankingProtocol,
    SupportTrial,
)
from .protocol import (
    load_ranking_protocol as _load_ranking_protocol,
)
from .protocol_catalog import LIU_V1_PROTOCOL_PATH

DEFAULT_PROTOCOL_PATH = LIU_V1_PROTOCOL_PATH


def load_ranking_protocol(
    path: Path | str = DEFAULT_PROTOCOL_PATH,
) -> RankingProtocol:
    return _load_ranking_protocol(path)


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "QueryTrial",
    "RankingProtocol",
    "SupportTrial",
    "load_ranking_protocol",
]
