"""Materialize reversible normalized views without changing historical sources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fsrl.infra.file_contracts import (
    classify_path,
    safe_relative_path,
    validate_run_manifest,
)
from fsrl.infra.provenance import file_sha256
from fsrl.infra.record_catalog import CATALOG_PATH
from fsrl.paths import REPO_ROOT, RUNS_ROOT

WORKFLOW_ID = "historical-file-normalization"
EXECUTION_ID = "catalog-v2"
OUTPUT_ROOT = RUNS_ROOT / WORKFLOW_ID / EXECUTION_ID


def deterministic_gzip(payload: bytes) -> bytes:
    """Return a byte-stable gzip member whose payload is exactly reversible."""

    return gzip.compress(payload, compresslevel=9, mtime=0)


def deterministic_float_npz(values: np.ndarray) -> bytes:
    """Encode one float64 vector as a deterministic, pickle-free NPZ."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("numeric text view must be a finite one-dimensional vector")
    npy_buffer = io.BytesIO()
    np.save(npy_buffer, array, allow_pickle=False)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        member = zipfile.ZipInfo("values.npy", date_time=(1980, 1, 1, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        member.create_system = 3
        member.external_attr = 0o600 << 16
        archive.writestr(member, npy_buffer.getvalue())
    return archive_buffer.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _describe_payload(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        **classify_path(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _add_payload(payloads: dict[str, bytes], path: str, payload: bytes) -> None:
    prior = payloads.setdefault(path, payload)
    if prior != payload:
        raise RuntimeError(f"normalized target collision: {path}")


def _verified_source(payload: bytes, *, sha256: str, byte_count: int) -> bytes:
    if len(payload) != byte_count or _sha256(payload) != sha256:
        raise RuntimeError(f"historical source identity mismatch: {sha256}")
    return payload


def _git_blob(blob: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", blob],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"unable to read historical Git blob {blob}: {detail}")
    return result.stdout


def _numeric_text(payload: bytes) -> tuple[bytes, dict[str, Any]]:
    lines = payload.decode("utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("text source is not a dense numeric vector")
    try:
        values = np.asarray([float(line) for line in lines], dtype=np.float64)
    except ValueError as error:
        raise ValueError("text source is not a numeric vector") from error
    return deterministic_float_npz(values), {
        "array": "values",
        "dtype": "float64",
        "shape": [len(values)],
        "source_lines": len(lines),
    }


def _normalized_payload(
    *, status: str, source_format: str, source_sha256: str, payload: bytes
) -> tuple[str, bytes, str, dict[str, Any]]:
    if status == "historical_gzip_view" and source_format == "json":
        target = f"json/{source_sha256}.json.gz"
        normalized = deterministic_gzip(payload)
        return (
            target,
            normalized,
            "deterministic_gzip_mirror",
            {
                "decompressed_bytes": len(payload),
                "decompressed_sha256": source_sha256,
                "json_pointer_semantics": "unchanged_after_decompression",
            },
        )
    if status == "historical_pth_view" and source_format == "legacy_pytorch_state_dict":
        target = f"checkpoints/{source_sha256}.pth"
        return (
            target,
            payload,
            "byte_identity_extension_normalization",
            {
                "byte_identity": True,
            },
        )
    if status == "manual_owner_review" and source_format == "text":
        normalized, array_contract = _numeric_text(payload)
        target = f"arrays/{source_sha256}.npz"
        return target, normalized, "numeric_text_to_numpy_npz", array_contract
    raise ValueError(
        f"unsupported historical normalization: status={status}, format={source_format}"
    )


def _conversion_entry(
    *,
    source_id: str,
    source_kind: str,
    source_locator: str,
    source_format: str,
    source_sha256: str,
    source_bytes: int,
    status: str,
    payload: bytes,
    payloads: dict[str, bytes],
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target, normalized, transformation, verification = _normalized_payload(
        status=status,
        source_format=source_format,
        source_sha256=source_sha256,
        payload=payload,
    )
    _add_payload(payloads, target, normalized)
    source = {
        "id": source_id,
        "kind": source_kind,
        "locator": source_locator,
        "format": source_format,
        "bytes": source_bytes,
        "sha256": source_sha256,
    }
    if source_metadata:
        source.update(source_metadata)
    return {
        "source": source,
        "transformation": transformation,
        "output": _describe_payload(target, normalized),
        "verification": verification,
    }


def _registered_conversions(payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    conversions = []
    for record in catalog.get("records", []):
        status = record["normalization"]["status"]
        if status not in {"historical_gzip_view", "historical_pth_view"}:
            continue
        materialized = record.get("materialized_identity")
        repository_path = record["locator"].get("repository_path")
        if materialized is not None and repository_path is not None:
            source_path = REPO_ROOT / safe_relative_path(repository_path)
            payload = source_path.read_bytes()
            identity = materialized
            metadata = {"availability": "materialized"}
        else:
            identity = record["registered_identity"]
            payload = _git_blob(identity["git_blob"])
            metadata = {
                "availability": "git_blob_only",
                "git_blob": identity["git_blob"],
                "witness_commit": identity["witness_commit"],
            }
        payload = _verified_source(
            payload,
            sha256=identity["sha256"],
            byte_count=identity["bytes"],
        )
        conversions.append(
            _conversion_entry(
                source_id=record["record_id"],
                source_kind="registered_record",
                source_locator=repository_path or record["locator"]["legacy_path"],
                source_format=record["format"]["format"],
                source_sha256=identity["sha256"],
                source_bytes=identity["bytes"],
                status=status,
                payload=payload,
                payloads=payloads,
                source_metadata=metadata,
            )
        )
    return conversions


def _legacy_run_manifests() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted({*RUNS_ROOT.rglob("run.json"), *RUNS_ROOT.rglob("*.run.json")})
    manifests = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("document_type") == "fsrl.legacy_run_manifest":
            manifests.append((path, manifest))
    return manifests


def _runtime_conversions(payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    conversions = []
    for manifest_path, manifest in _legacy_run_manifests():
        for entry in manifest.get("files", []):
            status = entry["normalization"]["status"]
            if status == "already_conformant":
                continue
            source_path = manifest_path.parent / safe_relative_path(entry["path"])
            payload = _verified_source(
                source_path.read_bytes(),
                sha256=entry["sha256"],
                byte_count=entry["bytes"],
            )
            source_locator = source_path.relative_to(REPO_ROOT).as_posix()
            conversions.append(
                _conversion_entry(
                    source_id=f"{manifest['execution_id']}:{entry['path']}",
                    source_kind="legacy_runtime_file",
                    source_locator=source_locator,
                    source_format=entry["format"],
                    source_sha256=entry["sha256"],
                    source_bytes=entry["bytes"],
                    status=status,
                    payload=payload,
                    payloads=payloads,
                    source_metadata={
                        "run_manifest": manifest_path.relative_to(REPO_ROOT).as_posix()
                    },
                )
            )
    return conversions


def build_materialization() -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build every expected normalized payload and its prospective run manifest."""

    payloads: dict[str, bytes] = {}
    conversions = _registered_conversions(payloads) + _runtime_conversions(payloads)
    conversions.sort(key=lambda entry: (entry["source"]["kind"], entry["source"]["id"]))
    strategy_counts = Counter(entry["transformation"] for entry in conversions)
    conversion_index = {
        "document_type": "fsrl.historical_file_conversion_index",
        "schema_version": 1,
        "conversion_id": f"{WORKFLOW_ID}.{EXECUTION_ID}",
        "contract": {
            "source_payloads": "unchanged",
            "large_json": "deterministic_gzip_with_exact_decompressed_bytes",
            "legacy_checkpoint": "byte_identical_pth_compatibility_view",
            "numeric_text": "one_dimensional_float64_npz_with_source_retained",
        },
        "source_count": len(conversions),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "conversions": conversions,
    }
    payloads["conversions.json"] = (
        json.dumps(conversion_index, indent=2, sort_keys=True) + "\n"
    ).encode()
    files = [
        _describe_payload(path, payload) for path, payload in sorted(payloads.items())
    ]
    run_manifest = {
        "document_type": "fsrl.run_manifest",
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "execution_id": EXECUTION_ID,
        "lifecycle_state": "materialized_compatibility_view",
        "producer": {
            "module": "tools.provenance.materialize_historical_file_views_v1",
            "contract_version": 1,
        },
        "source_catalog": {
            "path": CATALOG_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(CATALOG_PATH),
        },
        "resolved_config": {
            "gzip_compresslevel": 9,
            "gzip_mtime": 0,
            "numeric_text_dtype": "float64",
            "npz_allow_pickle": False,
        },
        "conversion_contract": {
            "source_payloads": "unchanged",
            "view_root": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
        },
        "source_count": len(conversions),
        "file_count": len(files),
        "bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }
    return payloads, run_manifest


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run(*, apply: bool) -> dict[str, Any]:
    payloads, run_manifest = build_materialization()
    expected = {
        OUTPUT_ROOT / safe_relative_path(path): payload
        for path, payload in payloads.items()
    }
    expected[OUTPUT_ROOT / "run.json"] = (
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    errors: list[str] = []
    written = 0
    for path, payload in sorted(expected.items()):
        if apply and not path.exists():
            _write_exclusive(path, payload)
            written += 1
        if not path.is_file() or path.read_bytes() != payload:
            errors.append(path.relative_to(REPO_ROOT).as_posix())
    if OUTPUT_ROOT.is_dir():
        unexpected = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in OUTPUT_ROOT.rglob("*")
            if path.is_file() and path not in expected
        )
        errors.extend(f"unexpected:{path}" for path in unexpected)
    manifest_path = OUTPUT_ROOT / "run.json"
    if manifest_path.is_file() and not errors:
        validation = validate_run_manifest(manifest_path)
        errors.extend(validation["errors"])
    conversion_index = json.loads(payloads["conversions.json"])
    return {
        "passed": not errors,
        "errors": errors,
        "source_files": run_manifest["source_count"],
        "normalized_files": run_manifest["file_count"],
        "strategy_counts": conversion_index["strategy_counts"],
        "output_root": OUTPUT_ROOT.relative_to(REPO_ROOT).as_posix(),
        "written": written,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(apply=arguments.apply)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
