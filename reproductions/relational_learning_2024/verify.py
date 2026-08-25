"""Verify byte-locked upstream files and supplied checkpoints in this capsule."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from fsrl.infra.provenance import file_sha256

CAPSULE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = CAPSULE_ROOT / "source_manifest.toml"


def verify_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = tomllib.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("reproduction source manifest schema must be 1")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("reproduction source manifest is empty")
    seen: set[str] = set()
    checks = []
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe reproduction path: {record['path']}")
        if record["path"] in seen:
            raise ValueError(f"duplicate reproduction path: {record['path']}")
        seen.add(record["path"])
        source = path.parent / relative
        observed = file_sha256(source) if source.is_file() else None
        checks.append(
            {
                "path": record["path"],
                "role": record["role"],
                "expected": record["sha256"],
                "observed": observed,
                "passed": observed == record["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise ValueError("reproduction source hash verification failed")
    return {
        "passed": True,
        "capsule_id": manifest["id"],
        "files": len(checks),
        "checks": checks,
    }


def main() -> None:
    print(json.dumps(verify_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
