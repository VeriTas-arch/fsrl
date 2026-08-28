"""Backfill portable manifests for legacy runtime directories without moving data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from fsrl.infra.file_contracts import describe_file
from fsrl.paths import REPO_ROOT, RUNS_ROOT

ROOT_MANIFEST_NAME = "legacy-root-files.run.json"
_ID_COMPONENT = re.compile(r"[^a-z0-9]+")


def _execution_id(relative: str) -> str:
    slug = _ID_COMPONENT.sub("-", relative.lower()).strip("-")
    return f"legacy.{slug or 'root-files'}"


def _normalization(entry: dict[str, Any]) -> dict[str, Any]:
    if entry["legacy_format"]:
        return {
            "status": "historical_pth_view",
            "target_format": "pytorch_state_dict",
            "target_suffix": ".pth",
            "transformation": "byte_identity_extension_normalization",
        }
    if entry["format"] == "json" and entry["bytes"] >= 1_000_000:
        return {
            "status": "historical_gzip_view",
            "target_format": "gzip_json",
            "target_suffix": ".json.gz",
            "transformation": "deterministic_gzip_mirror",
        }
    if entry["format"] == "text":
        return {
            "status": "manual_owner_review",
            "target_format": "typed_json_or_numpy_npz",
        }
    return {"status": "already_conformant", "target_format": entry["format"]}


def _described_files(files: list[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(files):
        entry = describe_file(path, relative_to=relative_to)
        entry["normalization"] = _normalization(entry)
        entries.append(entry)
    return entries


def build_manifest(
    run_root: Path,
    *,
    runs_root: Path = RUNS_ROOT,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build one legacy manifest from observed bytes without inferring ownership."""

    run_root = run_root.resolve()
    runs_root = runs_root.resolve()
    repository_root = repository_root.resolve()
    relative = run_root.relative_to(runs_root).as_posix()
    files = [
        path
        for path in run_root.rglob("*")
        if path.is_file()
        and path.name != "run.json"
        and not path.name.endswith(".run.json")
    ]
    entries = _described_files(files, relative_to=run_root)
    return {
        "document_type": "fsrl.legacy_run_manifest",
        "schema_version": 1,
        "execution_id": _execution_id(relative),
        "source_root": run_root.relative_to(repository_root).as_posix(),
        "layout": "legacy_workflow_root",
        "lifecycle_state": "historical_unclassified",
        "conversion_contract": {
            "source_payloads": "unchanged",
            "ownership": "not_inferred",
            "future_runs": "use a unique execution directory and a prospective run manifest",
        },
        "file_count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def build_root_file_manifest(
    *,
    runs_root: Path = RUNS_ROOT,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any] | None:
    """Describe loose files that violate the workflow-directory layout."""

    files = [
        path
        for path in runs_root.iterdir()
        if path.is_file() and not path.name.endswith(".run.json")
    ]
    if not files:
        return None
    entries = _described_files(files, relative_to=runs_root)
    return {
        "document_type": "fsrl.legacy_run_manifest",
        "schema_version": 1,
        "execution_id": "legacy.root-files",
        "source_root": runs_root.resolve()
        .relative_to(repository_root.resolve())
        .as_posix(),
        "layout": "legacy_loose_files",
        "lifecycle_state": "historical_unclassified",
        "conversion_contract": {
            "source_payloads": "unchanged",
            "ownership": "requires_manual_review",
            "relocation": "forbidden_until_owner_and_references_are_resolved",
        },
        "file_count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def planned_manifests(
    *,
    runs_root: Path = RUNS_ROOT,
    repository_root: Path = REPO_ROOT,
) -> dict[Path, dict[str, Any]]:
    """Return every deterministic legacy manifest planned for one runs root."""

    if not runs_root.is_dir():
        return {}
    manifests = {
        directory / "run.json": build_manifest(
            directory, runs_root=runs_root, repository_root=repository_root
        )
        for directory in sorted(runs_root.iterdir())
        if directory.is_dir() and _is_legacy_root(directory)
    }
    root_manifest = build_root_file_manifest(
        runs_root=runs_root, repository_root=repository_root
    )
    if root_manifest is not None:
        manifests[runs_root / ROOT_MANIFEST_NAME] = root_manifest
    return manifests


def _manifest_document_type(path: Path) -> str | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = manifest.get("document_type")
    return value if isinstance(value, str) else None


def _is_legacy_root(directory: Path) -> bool:
    manifest_path = directory / "run.json"
    if manifest_path.is_file():
        return _manifest_document_type(manifest_path) == "fsrl.legacy_run_manifest"
    prospective_children = (
        child / "run.json" for child in directory.iterdir() if child.is_dir()
    )
    return not any(
        _manifest_document_type(path) == "fsrl.run_manifest"
        for path in prospective_children
        if path.is_file()
    )


def _write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _source_identities(manifest: dict[str, Any]) -> list[tuple[str, int, str]]:
    return [
        (entry["path"], entry["bytes"], entry["sha256"])
        for entry in manifest.get("files", [])
    ]


def run(
    *,
    apply: bool,
    runs_root: Path = RUNS_ROOT,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    manifests = planned_manifests(runs_root=runs_root, repository_root=repository_root)
    errors: list[str] = []
    written = 0
    updated = 0
    files = 0
    for path, manifest in manifests.items():
        expected = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if apply and not path.exists():
            _write_exclusive(path, expected)
            written += 1
        elif apply and path.is_file() and path.read_text(encoding="utf-8") != expected:
            observed_manifest = json.loads(path.read_text(encoding="utf-8"))
            if observed_manifest.get(
                "document_type"
            ) != "fsrl.legacy_run_manifest" or _source_identities(
                observed_manifest
            ) != _source_identities(manifest):
                errors.append(path.relative_to(repository_root).as_posix())
                files += manifest["file_count"]
                continue
            path.write_text(expected, encoding="utf-8")
            updated += 1
        observed = path.read_text(encoding="utf-8") if path.is_file() else None
        if observed != expected:
            errors.append(path.relative_to(repository_root).as_posix())
        files += manifest["file_count"]
    return {
        "passed": not errors,
        "errors": errors,
        "manifests": len(manifests),
        "source_files": files,
        "updated": updated,
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
