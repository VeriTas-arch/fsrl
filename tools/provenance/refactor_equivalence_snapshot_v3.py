"""Render deterministic protocol and exclusive-JSON bytes for audit v3."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fsrl.infra.provenance import write_json_exclusive
from fsrl.tasks.protocol_catalog import load_registered_protocol


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_snapshot() -> dict[str, object]:
    protocol = load_registered_protocol("liu_v1")
    payload = {
        "protocol": asdict(protocol),
        "support_schedule": [
            asdict(trial)
            for trial in protocol.support_schedule(np.random.default_rng(41))
        ],
        "query_schedule": [
            asdict(trial)
            for trial in protocol.query_schedule(np.random.default_rng(43))
        ],
    }
    with tempfile.TemporaryDirectory(prefix="fsrl-refactor-v3-") as directory:
        output = Path(directory) / "snapshot.json"
        write_json_exclusive(output, payload)
        serialized = output.read_bytes()
    return {
        "document_type": "fsrl.refactor_equivalence_snapshot_v3",
        "protocol_id": protocol.protocol_id,
        "serialized_bytes": len(serialized),
        "serialized_sha256": _sha256(serialized),
    }


def main() -> None:
    print(json.dumps(build_snapshot(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
