"""Deterministic v2 catalog over byte-preserved historical research files."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from fsrl.infra.file_contracts import (
    classify_path,
    safe_relative_path,
    stable_record_id,
)
from fsrl.infra.provenance import file_sha256
from fsrl.paths import REPO_ROOT, STUDIES_ROOT, SYNTHESIS_ROOT

CATALOG_PATH = STUDIES_ROOT / "catalogs" / "record-catalog-v2.json"
REGISTRY_PATH = STUDIES_ROOT / "registry.toml"
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "manifest.toml"
RUNTIME_LOCATOR_PATH = STUDIES_ROOT / "migrations" / "runtime-locators-v1.json"
_LARGE_JSON_BYTES = 1_000_000


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalization(format_name: str, byte_count: int) -> dict[str, Any]:
    if format_name == "legacy_pytorch_state_dict":
        return {
            "status": "historical_pth_view",
            "target_format": "pytorch_state_dict",
            "target_suffix": ".pth",
            "transformation": "byte_identity_extension_normalization",
        }
    if format_name == "json" and byte_count >= _LARGE_JSON_BYTES:
        return {
            "status": "historical_gzip_view",
            "target_format": "gzip_json",
            "target_suffix": ".json.gz",
            "transformation": "deterministic_gzip_mirror",
        }
    return {"status": "already_conformant", "target_format": format_name}


def _materialized_record(
    *,
    owner_kind: str,
    owner_id: str,
    owner_root: Path,
    record: dict[str, Any],
    locator_rewrites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy_path = safe_relative_path(record["legacy_path"]).as_posix()
    relative_path = safe_relative_path(record["path"])
    path = owner_root / relative_path
    repository_path = path.relative_to(REPO_ROOT).as_posix()
    if not path.is_file():
        raise RuntimeError(f"registered record is unavailable: {repository_path}")
    observed_sha256 = file_sha256(path)
    observed_bytes = path.stat().st_size
    rewrite = locator_rewrites.get(repository_path)
    if rewrite is None:
        if observed_sha256 != record["sha256"] or observed_bytes != record["bytes"]:
            raise RuntimeError(
                f"registered record identity mismatch: {repository_path}"
            )
    else:
        before = (rewrite["before_sha256"], rewrite["before_bytes"])
        after = (rewrite["after_sha256"], rewrite["after_bytes"])
        if before != (record["sha256"], record["bytes"]):
            raise RuntimeError(f"locator source identity mismatch: {repository_path}")
        if after != (observed_sha256, observed_bytes):
            raise RuntimeError(f"locator materialization mismatch: {repository_path}")
    classification = classify_path(path)
    format_name = classification["format"]
    if not isinstance(format_name, str):
        raise TypeError(f"invalid format classification: {repository_path}")
    result = {
        "record_id": stable_record_id(owner_kind, owner_id, legacy_path),
        "owner": {"kind": owner_kind, "id": owner_id},
        "role": record["role"],
        "document_type": f"fsrl.record.{record['role']}",
        "format": classification,
        "locator": {
            "legacy_path": legacy_path,
            "repository_path": repository_path,
        },
        "registered_identity": {
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "source_ref": record["source_ref"],
        },
        "materialized_identity": {
            "sha256": observed_sha256,
            "bytes": observed_bytes,
        },
        "normalization": _normalization(format_name, observed_bytes),
        "availability": "materialized",
    }
    if rewrite is not None:
        result["transformation"] = {
            "kind": "locator_rewrite",
            "migration_id": "runtime-locators-v1",
            "replacements": rewrite["replacements"],
        }
    return result


def _retired_asset(owner_id: str, asset: dict[str, Any]) -> dict[str, Any]:
    legacy_path = safe_relative_path(asset["path"]).as_posix()
    classification = classify_path(legacy_path)
    format_name = classification["format"]
    if not isinstance(format_name, str):
        raise TypeError(f"invalid format classification: {legacy_path}")
    return {
        "record_id": stable_record_id("study", owner_id, legacy_path),
        "owner": {"kind": "study", "id": owner_id},
        "role": asset["role"],
        "document_type": f"fsrl.retired_asset.{asset['role']}",
        "format": classification,
        "locator": {"legacy_path": legacy_path, "repository_path": None},
        "registered_identity": {
            "sha256": asset["sha256"],
            "bytes": asset["bytes"],
            "source_ref": asset["source_ref"],
            "git_blob": asset["git_blob"],
            "witness_commit": asset["witness_commit"],
        },
        "materialized_identity": None,
        "normalization": _normalization(format_name, asset["bytes"]),
        "availability": "git_blob_only",
    }


def _locator_rewrite_lookup(
    registry: dict[str, Any], locator: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    direct: dict[str, str] = {}
    for relative in registry.get("migrations", [registry["migration"]]):
        migration = _load_json(STUDIES_ROOT / safe_relative_path(relative))
        direct.update(
            (record["legacy_path"], record["path"])
            for record in migration.get("records", [])
        )

    def final_path(value: str) -> str:
        seen: set[str] = set()
        while value in direct:
            if value in seen:
                raise RuntimeError(f"migration cycle detected at {value}")
            seen.add(value)
            value = direct[value]
        return value

    lookup: dict[str, dict[str, Any]] = {}
    for record in locator.get("records", []):
        lookup[record["path"]] = record
        lookup[final_path(record["path"])] = record
    return lookup


def build_record_catalog(
    *,
    registry_path: Path = REGISTRY_PATH,
    synthesis_path: Path = SYNTHESIS_MANIFEST_PATH,
    locator_path: Path = RUNTIME_LOCATOR_PATH,
) -> dict[str, Any]:
    """Build the normalized catalog without modifying registered source files."""

    registry = _load_toml(registry_path)
    synthesis = _load_toml(synthesis_path)
    locator = _load_json(locator_path)
    locator_rewrites = _locator_rewrite_lookup(registry, locator)
    entries: list[dict[str, Any]] = []
    registered_count = 0
    retired_count = 0
    for registration in registry.get("studies", []):
        study_path = REPO_ROOT / safe_relative_path(registration["path"])
        study = _load_toml(study_path)
        for record in study.get("records", []):
            entries.append(
                _materialized_record(
                    owner_kind="study",
                    owner_id=study["id"],
                    owner_root=study_path.parent,
                    record=record,
                    locator_rewrites=locator_rewrites,
                )
            )
            registered_count += 1
        for asset in study.get("retired_assets", []):
            entries.append(_retired_asset(study["id"], asset))
            retired_count += 1
    for record in synthesis.get("records", []):
        entries.append(
            _materialized_record(
                owner_kind="synthesis",
                owner_id=synthesis["id"],
                owner_root=SYNTHESIS_ROOT,
                record=record,
                locator_rewrites=locator_rewrites,
            )
        )
        registered_count += 1
    entries.sort(key=lambda entry: entry["record_id"])
    ids = [entry["record_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("generated record IDs are not unique")
    role_counts = Counter(entry["role"] for entry in entries)
    format_counts = Counter(entry["format"]["format"] for entry in entries)
    normalization_counts = Counter(
        entry["normalization"]["status"] for entry in entries
    )
    return {
        "document_type": "fsrl.registered_record_catalog",
        "schema_version": 2,
        "catalog_id": "registered-record-catalog-v2",
        "contract": {
            "source_payloads": "byte_preserved",
            "logical_ids": "derived_from_immutable_legacy_locators",
            "historical_large_json": "deterministic_gzip_mirror_preserves_decompressed_bytes_and_json_pointers",
            "prospective_bulk_arrays": "compact_json_summary_plus_numpy_npz",
        },
        "sources": {
            "registry": {
                "path": registry_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(registry_path),
            },
            "synthesis_manifest": {
                "path": synthesis_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(synthesis_path),
            },
            "runtime_locator_migration": {
                "path": locator_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(locator_path),
            },
        },
        "entry_count": len(entries),
        "registered_record_count": registered_count,
        "retired_asset_count": retired_count,
        "role_counts": dict(sorted(role_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "records": entries,
    }


def render_record_catalog() -> str:
    """Render the deterministic checked-in catalog representation."""

    return json.dumps(build_record_catalog(), indent=2, sort_keys=True) + "\n"


def check_record_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Check that the generated catalog matches every current authority."""

    expected = render_record_catalog()
    observed = path.read_text(encoding="utf-8") if path.is_file() else None
    catalog = json.loads(expected)
    return {
        "passed": observed == expected,
        "catalog": path.relative_to(REPO_ROOT).as_posix(),
        "entries": catalog["entry_count"],
        "registered_records": catalog["registered_record_count"],
        "retired_assets": catalog["retired_asset_count"],
        "stale": observed != expected,
    }
