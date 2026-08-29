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
from .protocol_catalog import protocol_path

DEFAULT_PROTOCOL_ID = "liu_v1"


def default_protocol_path() -> Path:
    """Resolve the legacy default only when a registered protocol is requested."""

    return protocol_path(DEFAULT_PROTOCOL_ID)


def load_ranking_protocol(
    path: Path | str | None = None,
) -> RankingProtocol:
    return _load_ranking_protocol(default_protocol_path() if path is None else path)


__all__ = [
    "DEFAULT_PROTOCOL_ID",
    "QueryTrial",
    "RankingProtocol",
    "SupportTrial",
    "default_protocol_path",
    "load_ranking_protocol",
]
