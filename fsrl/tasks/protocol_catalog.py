"""Stable logical IDs for registered task protocols used by current code."""

from __future__ import annotations

from pathlib import Path

from fsrl.infra.record_catalog import resolve_record_id

from .protocol import RankingProtocol, load_ranking_protocol

PROTOCOL_RECORD_IDS = {
    "liu_v1": "study.task_fidelity.benchmarks_liu_v1_json",
    "liu_v2": "study.task_fidelity.benchmarks_liu_v2_json",
}
PROTOCOL_DOCUMENT_IDS = {
    "liu_v1": "liu-constructive-ranking-v1",
    "liu_v2": "liu-constructive-ranking-v2-source-corrected",
}


def protocol_path(protocol_id: str) -> Path:
    """Resolve a protocol by semantic ID, never by a historical locator."""

    try:
        record_id = PROTOCOL_RECORD_IDS[protocol_id]
    except KeyError as error:
        raise KeyError(f"unknown registered protocol: {protocol_id}") from error
    return resolve_record_id(record_id)


def load_registered_protocol(protocol_id: str) -> RankingProtocol:
    """Load a registered protocol through its stable logical identity."""

    protocol = load_ranking_protocol(protocol_path(protocol_id))
    if protocol.protocol_id != PROTOCOL_DOCUMENT_IDS[protocol_id]:
        raise RuntimeError(f"registered protocol identity mismatch: {protocol_id}")
    return protocol
