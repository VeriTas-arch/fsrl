"""Verify byte-locked upstream files and supplied checkpoints in this capsule."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from fsrl.infra.provenance import file_sha256

CAPSULE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = CAPSULE_ROOT / "source_manifest.toml"
VIEW_MANIFEST_PATH = CAPSULE_ROOT / "checkpoint_views.toml"


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe reproduction path: {value}")
    return relative


def _verify_checkpoint_views(
    path: Path, source_manifest_path: Path, source_paths: set[str]
) -> list[dict]:
    manifest = tomllib.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("checkpoint view manifest schema must be 1")
    if manifest.get("document_type") != "checkpoint_views":
        raise ValueError("checkpoint view manifest has an unknown document type")
    if manifest.get("source_manifest") != source_manifest_path.name:
        raise ValueError("checkpoint view manifest names the wrong source manifest")
    if manifest.get("source_manifest_sha256") != file_sha256(source_manifest_path):
        raise ValueError("checkpoint view manifest source binding failed")
    if manifest.get("transformation") != "byte_identity_extension_normalization":
        raise ValueError("checkpoint view manifest has an unknown transformation")

    records = manifest.get("views")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint view manifest is empty")
    seen_sources: set[str] = set()
    seen_views: set[str] = set()
    checks = []
    for record in records:
        source_name = record["source"]
        view_name = record["view"]
        source_relative = _safe_relative_path(source_name)
        view_relative = _safe_relative_path(view_name)
        if source_name not in source_paths:
            raise ValueError(
                f"checkpoint view source is not byte-locked: {source_name}"
            )
        if source_name in seen_sources or view_name in seen_views:
            raise ValueError("duplicate checkpoint view mapping")
        if (
            source_relative.suffix.lower() != ".dat"
            or view_relative.suffix.lower() != ".pth"
        ):
            raise ValueError("checkpoint views must map .dat sources to .pth views")
        seen_sources.add(source_name)
        seen_views.add(view_name)
        source = path.parent / source_relative
        view = path.parent / view_relative
        source_hash = file_sha256(source) if source.is_file() else None
        view_hash = file_sha256(view) if view.is_file() else None
        source_bytes = source.stat().st_size if source.is_file() else None
        view_bytes = view.stat().st_size if view.is_file() else None
        passed = (
            source_hash == record["sha256"]
            and view_hash == record["sha256"]
            and source_bytes == record["bytes"]
            and view_bytes == record["bytes"]
        )
        checks.append(
            {
                "source": source_name,
                "view": view_name,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "passed": passed,
            }
        )
    if not all(check["passed"] for check in checks):
        raise ValueError("checkpoint view verification failed")
    return checks


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
        relative = _safe_relative_path(record["path"])
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
    view_checks = _verify_checkpoint_views(VIEW_MANIFEST_PATH, path, seen)
    return {
        "passed": True,
        "capsule_id": manifest["id"],
        "files": len(checks),
        "views": len(view_checks),
        "checks": checks,
        "view_checks": view_checks,
    }


def main() -> None:
    print(json.dumps(verify_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
