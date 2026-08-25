"""Compatibility entry point for the registered v1 ranking protocol."""

from pathlib import Path

from .study_registry import resolve_record
from .tasks.protocol import (
    QueryTrial,
    RankingProtocol,
    SupportTrial,
)
from .tasks.protocol import (
    load_ranking_protocol as _load_ranking_protocol,
)

DEFAULT_PROTOCOL_PATH = resolve_record("benchmarks/liu_v1.json")


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
