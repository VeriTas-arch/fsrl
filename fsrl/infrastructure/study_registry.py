"""Validate and render the study-owned research record.

The registry separates three concerns:

* ``studies/`` owns experiment-level questions, outcomes, and exact records;
* ``synthesis/`` organizes the current cross-study account;
* the migration map resolves frozen pre-refactor paths without duplicating files.

Historical scientific values are immutable. Repository-wide locator-only
rewrites require their own provenance-locked migration ledger; changing a
result or interpretation still requires a new record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from fsrl.paths import REPO_ROOT, STUDIES_ROOT, SYNTHESIS_ROOT

ROOT = REPO_ROOT
REGISTRY_PATH = STUDIES_ROOT / "registry.toml"
MIGRATION_PATH = STUDIES_ROOT / "migrations" / "flat-records-v1.json"
RUNTIME_LOCATOR_MIGRATION_PATH = (
    STUDIES_ROOT / "migrations" / "runtime-locators-v1.json"
)
SYNTHESIS_MANIFEST_PATH = SYNTHESIS_ROOT / "manifest.toml"
SOURCE_PROVENANCE_PATH = SYNTHESIS_ROOT / "source-provenance.toml"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
GENERATED_PATHS = (
    Path("studies/README.md"),
    Path("synthesis/README.md"),
    Path("synthesis/figures/README.md"),
)
REQUIRED_STUDY_FIELDS = (
    "schema_version",
    "id",
    "title",
    "chapter",
    "order",
    "status",
    "review_state",
    "question",
    "finding",
    "boundary",
    "records",
)
REQUIRED_RECORD_FIELDS = (
    "path",
    "legacy_path",
    "role",
    "sha256",
    "bytes",
    "source_ref",
)
REQUIRED_RETIRED_ASSET_FIELDS = (
    "path",
    "role",
    "sha256",
    "bytes",
    "git_blob",
    "witness_commit",
    "source_ref",
    "reason",
)


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return _load_toml(path)


def load_synthesis(path: Path = SYNTHESIS_MANIFEST_PATH) -> dict[str, Any]:
    return _load_toml(path)


def load_migration(path: Path = MIGRATION_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_locator_migration(
    path: Path = RUNTIME_LOCATOR_MIGRATION_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_provenance(path: Path = SOURCE_PROVENANCE_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "sources": []}
    return _load_toml(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_file_sha256(path: Path | str) -> str:
    """Return the registered identity across a locator-only content rewrite."""

    target = Path(path)
    try:
        repository_path = target.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        repository_path = None
    rewrite = (
        runtime_locator_lookup().get(repository_path)
        if repository_path is not None
        else None
    )
    observed = file_sha256(target)
    if rewrite is None:
        return observed
    if (
        target.stat().st_size != rewrite["after_bytes"]
        or observed != rewrite["after_sha256"]
    ):
        raise RuntimeError(
            f"runtime-locator rewritten content mismatch: {repository_path}"
        )
    return rewrite["before_sha256"]


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"repository path must be a safe relative path: {value!r}")
    return Path(*pure.parts)


@lru_cache(maxsize=1)
def migration_lookup() -> dict[str, str]:
    if not MIGRATION_PATH.is_file():
        return {}
    migration = load_migration()
    return {
        entry["legacy_path"]: entry["path"] for entry in migration.get("records", [])
    }


@lru_cache(maxsize=1)
def runtime_locator_lookup() -> dict[str, dict[str, Any]]:
    migration = load_runtime_locator_migration()
    return {record["path"]: record for record in migration.get("records", [])}


@lru_cache(maxsize=1)
def source_provenance_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    provenance = load_source_provenance()
    return {
        (record["path"], record["sha256"]): record
        for record in provenance.get("sources", [])
    }


@cache
def _git_blob(git_blob: str) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", git_blob],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _verify_source_record(record: dict[str, Any]) -> dict[str, Any]:
    path = _safe_relative(record["path"]).as_posix()
    expected_sha256 = record["sha256"]
    git_blob = record["git_blob"]
    witness_commit = record["witness_commit"]
    payload = _git_blob(git_blob)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Git blob hash mismatch for {path}@{expected_sha256}")
    if len(payload) != record["bytes"]:
        raise ValueError(f"Git blob byte count mismatch for {path}@{expected_sha256}")
    completed = subprocess.run(
        ["git", "rev-parse", f"{witness_commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    witness_blob = completed.stdout.strip()
    if witness_blob != git_blob:
        raise ValueError(f"Git witness mismatch for {path}@{expected_sha256}")
    return {
        "path": path,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "bytes": len(payload),
        "git_blob": git_blob,
        "witness_commit": witness_commit,
        "passed": True,
    }


def verify_source_lock(value: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Verify a frozen Python source registration against Git object storage."""

    path = _safe_relative(Path(value).as_posix()).as_posix()
    record = source_provenance_lookup().get((path, expected_sha256))
    if record is None:
        raise FileNotFoundError(
            f"unindexed frozen source registration: {path}@{expected_sha256}"
        )
    return _verify_source_record(record)


def registered_file_sha256(
    value: str | Path,
    expected_sha256: str,
    *,
    resolved_path: Path | None = None,
) -> str:
    """Hash a registered artifact, using Git for historical Python sources."""

    candidate = Path(value)
    target = candidate if candidate.is_absolute() else resolve_record(candidate)
    if resolved_path is not None:
        target = resolved_path
    try:
        repository_path = target.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        repository_path = None
    rewrite = (
        runtime_locator_lookup().get(repository_path)
        if repository_path is not None
        else None
    )
    if rewrite is not None:
        if rewrite["before_sha256"] != expected_sha256:
            raise RuntimeError(
                f"runtime-locator source hash mismatch: {repository_path}"
            )
        return canonical_file_sha256(target)
    if candidate.is_absolute():
        return file_sha256(target)
    path = _safe_relative(candidate.as_posix()).as_posix()
    if path.endswith(".py") and path.startswith(("fsrl/", "tests/")):
        return verify_source_lock(path, expected_sha256)["observed_sha256"]
    return file_sha256(resolve_record(path) if resolved_path is None else resolved_path)


def resolve_record(value: str | Path) -> Path:
    """Resolve an active path or a frozen pre-refactor record path.

    Returning a path does not imply that the file exists.  This preserves the
    normal ``Path`` behavior for prospective outputs while making historical
    record contracts readable after the physical migration. Historical source
    locks are verified separately with :func:`verify_source_lock`.
    """

    relative = _safe_relative(Path(value).as_posix())
    direct = ROOT / relative
    if direct.exists():
        return direct
    migrated = migration_lookup().get(relative.as_posix())
    return ROOT / migrated if migrated is not None else direct


def legacy_identifier(path: str | Path) -> str:
    """Return the frozen pre-migration identifier for a current record path."""

    candidate = Path(path)
    relative = (
        candidate.resolve().relative_to(ROOT) if candidate.is_absolute() else candidate
    )
    current = _safe_relative(relative.as_posix()).as_posix()
    inverse = {value: key for key, value in migration_lookup().items()}
    return inverse.get(current, current)


def _study_manifest_path(entry: dict[str, Any]) -> Path:
    return ROOT / entry["path"]


def load_studies(
    registry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    registry = load_registry() if registry is None else registry
    studies: dict[str, dict[str, Any]] = {}
    for entry in registry.get("studies", []):
        study = _load_toml(_study_manifest_path(entry))
        studies[study["id"]] = study
    return studies


def _record_repository_path(
    owner_kind: str,
    owner: dict[str, Any],
    record: dict[str, Any],
) -> Path:
    base = STUDIES_ROOT / owner["id"] if owner_kind == "study" else SYNTHESIS_ROOT
    return (base / _safe_relative(record["path"])).relative_to(ROOT)


def _validate_record(
    *,
    owner_kind: str,
    owner: dict[str, Any],
    record: dict[str, Any],
    errors: list[str],
) -> tuple[str, str] | None:
    owner_id = owner["id"]
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        errors.append(f"{owner_id}: record missing fields {missing}")
        return None
    try:
        repository_path = _record_repository_path(owner_kind, owner, record)
        legacy_path = _safe_relative(record["legacy_path"]).as_posix()
    except (KeyError, ValueError) as error:
        errors.append(f"{owner_id}: invalid record path: {error}")
        return None
    path = ROOT / repository_path
    if not path.is_file():
        errors.append(f"{owner_id}: missing record {repository_path.as_posix()}")
    else:
        actual_bytes = path.stat().st_size
        actual_hash = file_sha256(path)
        rewrite = runtime_locator_lookup().get(repository_path.as_posix())
        if rewrite is None:
            if actual_bytes != record["bytes"]:
                errors.append(
                    f"{owner_id}: byte count changed for {repository_path.as_posix()}"
                )
            if actual_hash != record["sha256"]:
                errors.append(
                    f"{owner_id}: hash changed for {repository_path.as_posix()}"
                )
        else:
            if (
                record["bytes"] != rewrite["before_bytes"]
                or record["sha256"] != rewrite["before_sha256"]
            ):
                errors.append(
                    f"{owner_id}: runtime-locator source identity changed for "
                    f"{repository_path.as_posix()}"
                )
            if (
                actual_bytes != rewrite["after_bytes"]
                or actual_hash != rewrite["after_sha256"]
            ):
                errors.append(
                    f"{owner_id}: runtime-locator rewrite changed for "
                    f"{repository_path.as_posix()}"
                )
    return legacy_path, repository_path.as_posix()


def _validate_retired_asset(
    *, owner_id: str, asset: dict[str, Any], errors: list[str]
) -> str | None:
    missing = [field for field in REQUIRED_RETIRED_ASSET_FIELDS if field not in asset]
    if missing:
        errors.append(f"{owner_id}: retired asset missing fields {missing}")
        return None
    try:
        path = _safe_relative(asset["path"]).as_posix()
    except (TypeError, ValueError) as error:
        errors.append(f"{owner_id}: invalid retired asset path: {error}")
        return None
    if (ROOT / path).exists():
        errors.append(f"{owner_id}: retired asset still exists: {path}")
    if not isinstance(asset["reason"], str) or not asset["reason"].strip():
        errors.append(f"{owner_id}: retired asset reason must be non-empty: {path}")
    try:
        verification = _verify_source_record(asset)
        completed = subprocess.run(
            ["git", "rev-parse", f"{asset['source_ref']}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip() != verification["git_blob"]:
            errors.append(f"{owner_id}: retired asset source ref mismatch: {path}")
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        errors.append(f"{owner_id}: retired asset verification failed: {error}")
    return path


def validate_registry(
    registry: dict[str, Any] | None = None,
    synthesis: dict[str, Any] | None = None,
    migration: dict[str, Any] | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_registry() if registry is None else registry
    synthesis = load_synthesis() if synthesis is None else synthesis
    migration = load_migration() if migration is None else migration
    source_provenance = (
        load_source_provenance() if source_provenance is None else source_provenance
    )
    errors: list[str] = []

    if registry.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
    if synthesis.get("schema_version") != 1:
        errors.append("synthesis schema_version must be 1")
    if migration.get("schema_version") != 1:
        errors.append("migration schema_version must be 1")
    if source_provenance.get("schema_version") != 1:
        errors.append("source provenance schema_version must be 1")

    sources = source_provenance.get("sources", [])
    if source_provenance.get("source_version_count") != len(sources):
        errors.append("source provenance count does not match sources")
    source_pairs: set[tuple[str, str]] = set()
    required_source_fields = {
        "path",
        "sha256",
        "bytes",
        "git_blob",
        "witness_commit",
        "record_count",
        "occurrence_count",
    }
    for source in sources:
        missing = sorted(required_source_fields - set(source))
        if missing:
            errors.append(f"source provenance entry missing fields {missing}")
            continue
        path = source["path"]
        sha256 = source["sha256"]
        git_blob = source["git_blob"]
        witness_commit = source["witness_commit"]
        if not isinstance(path, str):
            errors.append("source provenance path must be a string")
            continue
        try:
            normalized_path = _safe_relative(path).as_posix()
        except ValueError as error:
            errors.append(f"invalid source provenance path: {error}")
            continue
        if not normalized_path.endswith(".py") or not normalized_path.startswith(
            ("fsrl/", "tests/")
        ):
            errors.append(f"source provenance path is not active Python: {path}")
        if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
            errors.append(f"invalid source provenance SHA-256: {path}")
            continue
        if (
            not isinstance(git_blob, str)
            or GIT_SHA1_PATTERN.fullmatch(git_blob) is None
        ):
            errors.append(f"invalid source provenance Git blob: {path}@{sha256}")
            continue
        if (
            not isinstance(witness_commit, str)
            or GIT_SHA1_PATTERN.fullmatch(witness_commit) is None
        ):
            errors.append(f"invalid source provenance witness: {path}@{sha256}")
            continue
        pair = (normalized_path, sha256)
        if pair in source_pairs:
            errors.append(f"duplicate source provenance pair: {path}@{sha256}")
            continue
        source_pairs.add(pair)
        try:
            _verify_source_record(source)
        except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
            errors.append(f"source provenance verification failed: {error}")

    chapters = registry.get("chapters", [])
    chapter_ids = [chapter.get("id") for chapter in chapters]
    if len(chapter_ids) != len(set(chapter_ids)):
        errors.append("chapter ids must be unique")
    chapter_set = set(chapter_ids)
    status_set = set(registry.get("status_legend", {}))

    registry_entries = registry.get("studies", [])
    registry_ids = [entry.get("id") for entry in registry_entries]
    if len(registry_ids) != len(set(registry_ids)):
        errors.append("study ids must be unique")

    studies: dict[str, dict[str, Any]] = {}
    for entry in registry_entries:
        study_id = entry.get("id", "<missing>")
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            errors.append(f"{study_id}: registry path must be a string")
            continue
        manifest_path = _study_manifest_path(entry)
        if not manifest_path.is_file():
            errors.append(f"{study_id}: missing manifest {path_value}")
            continue
        study = _load_toml(manifest_path)
        studies[study_id] = study
        missing = [field for field in REQUIRED_STUDY_FIELDS if field not in study]
        if missing:
            errors.append(f"{study_id}: missing fields {missing}")
            continue
        if study["id"] != study_id:
            errors.append(f"{study_id}: local manifest id mismatch")
        if study["chapter"] not in chapter_set:
            errors.append(f"{study_id}: unknown chapter {study['chapter']!r}")
        if study["status"] not in status_set:
            errors.append(f"{study_id}: unknown status {study['status']!r}")
        if study["review_state"] not in {"indexed", "reviewed"}:
            errors.append(f"{study_id}: invalid review_state")
        for field in ("title", "question", "finding", "boundary"):
            if not isinstance(study[field], str) or not study[field].strip():
                errors.append(f"{study_id}: {field} must be a non-empty string")

    for view in registry.get("views", []):
        references = view.get("studies", [])
        unknown = sorted(set(references) - set(registry_ids))
        if unknown:
            errors.append(f"view {view.get('id')}: unknown studies {unknown}")
        if len(references) != len(set(references)):
            errors.append(f"view {view.get('id')}: duplicate study references")

    record_pairs: dict[str, str] = {}
    record_provenance: dict[str, dict[str, Any]] = {}
    current_paths: set[str] = set()
    role_counts: Counter[str] = Counter()
    retired_asset_paths: set[str] = set()
    for study_id, study in studies.items():
        for record in study.get("records", []):
            pair = _validate_record(
                owner_kind="study", owner=study, record=record, errors=errors
            )
            if pair is None:
                continue
            legacy_path, current_path = pair
            if legacy_path in record_pairs:
                errors.append(f"duplicate legacy path {legacy_path}")
            if current_path in current_paths:
                errors.append(f"duplicate current path {current_path}")
            record_pairs[legacy_path] = current_path
            record_provenance[legacy_path] = {
                "path": current_path,
                "owner_id": study_id,
                "owner_kind": "study",
                **{
                    field: record[field]
                    for field in ("role", "sha256", "bytes", "source_ref")
                },
            }
            current_paths.add(current_path)
            role_counts[record["role"]] += 1
        for asset in study.get("retired_assets", []):
            path = _validate_retired_asset(
                owner_id=study_id, asset=asset, errors=errors
            )
            if path is None:
                continue
            if path in retired_asset_paths:
                errors.append(f"duplicate retired asset path: {path}")
            retired_asset_paths.add(path)

    for record in synthesis.get("records", []):
        pair = _validate_record(
            owner_kind="synthesis", owner=synthesis, record=record, errors=errors
        )
        if pair is None:
            continue
        legacy_path, current_path = pair
        if legacy_path in record_pairs:
            errors.append(f"duplicate legacy path {legacy_path}")
        if current_path in current_paths:
            errors.append(f"duplicate current path {current_path}")
        record_pairs[legacy_path] = current_path
        record_provenance[legacy_path] = {
            "path": current_path,
            "owner_id": synthesis["id"],
            "owner_kind": "synthesis",
            **{
                field: record[field]
                for field in ("role", "sha256", "bytes", "source_ref")
            },
        }
        current_paths.add(current_path)
        role_counts[record["role"]] += 1

    migration_records = migration.get("records", [])
    if migration.get("record_count") != len(migration_records):
        errors.append("migration record_count does not match records")
    migration_pairs: dict[str, str] = {}
    for record in migration_records:
        legacy = record.get("legacy_path")
        current = record.get("path")
        if not isinstance(legacy, str) or not isinstance(current, str):
            errors.append("migration record paths must be strings")
            continue
        if legacy in migration_pairs:
            errors.append(f"duplicate migration legacy path {legacy}")
        migration_pairs[legacy] = current
        if record_pairs.get(legacy) != current:
            errors.append(f"migration record disagrees for {legacy}")
        expected = record_provenance.get(legacy)
        if expected is not None:
            mismatched = [
                field for field, value in expected.items() if record.get(field) != value
            ]
            if mismatched:
                errors.append(
                    f"migration provenance disagrees for {legacy}: {mismatched}"
                )

    if record_pairs != migration_pairs:
        missing = sorted(set(record_pairs) - set(migration_pairs))
        extra = sorted(set(migration_pairs) - set(record_pairs))
        if missing:
            errors.append(f"migration is missing records {missing}")
        if extra:
            errors.append(f"migration has unowned records {extra}")

    locator_migration = load_runtime_locator_migration()
    locator_records = locator_migration.get("records", [])
    locator_paths = [record.get("path") for record in locator_records]
    if locator_migration.get("schema_version") != 1:
        errors.append("runtime-locator migration schema_version must be 1")
    if locator_migration.get("record_count") != len(locator_records):
        errors.append("runtime-locator migration record_count is stale")
    if locator_migration.get("replacement_count") != sum(
        record.get("replacements", 0) for record in locator_records
    ):
        errors.append("runtime-locator migration replacement_count is stale")
    if len(locator_paths) != len(set(locator_paths)):
        errors.append("runtime-locator migration paths must be unique")
    unknown_locator_paths = sorted(set(locator_paths) - current_paths)
    if unknown_locator_paths:
        errors.append(
            f"runtime-locator migration has unowned records {unknown_locator_paths}"
        )

    for legacy_path in migration_pairs:
        if (ROOT / legacy_path).exists():
            errors.append(f"legacy flat path still exists: {legacy_path}")

    if source_provenance.get("registered_record_files") != len(record_pairs):
        errors.append("source provenance registered-record count is stale")
    for redundant_root in (
        SYNTHESIS_ROOT / "frozen" / "source",
        SYNTHESIS_ROOT / "frozen" / "source-blobs",
    ):
        if redundant_root.exists():
            errors.append(f"redundant source tree still exists: {redundant_root}")

    expected_files = current_paths
    observed_files = {
        path.relative_to(ROOT).as_posix()
        for root in (
            STUDIES_ROOT,
            SYNTHESIS_ROOT / "records",
            SYNTHESIS_ROOT / "frozen",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"README.md", "study.toml", "registry.toml"}
        and "migrations" not in path.parts
    }
    unexpected = sorted(observed_files - expected_files)
    if unexpected:
        errors.append(f"unregistered record files: {unexpected}")

    return {
        "passed": not errors,
        "errors": errors,
        "studies": len(studies),
        "chapters": len(chapters),
        "records": len(record_pairs),
        "study_records": sum(
            len(study.get("records", [])) for study in studies.values()
        ),
        "synthesis_records": len(synthesis.get("records", [])),
        "retired_assets": len(retired_asset_paths),
        "runtime_locator_rewrites": len(locator_records),
        "source_provenance": len(sources),
        "role_counts": dict(sorted(role_counts.items())),
    }


def _ordered_chapters(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(registry["chapters"], key=lambda chapter: chapter["order"])


def _ordered_studies(
    registry: dict[str, Any],
    studies: dict[str, dict[str, Any]],
    chapter_id: str | None = None,
) -> list[dict[str, Any]]:
    values = list(studies.values())
    if chapter_id is not None:
        values = [study for study in values if study["chapter"] == chapter_id]
    chapter_order = {
        chapter["id"]: chapter["order"] for chapter in registry["chapters"]
    }
    return sorted(
        values,
        key=lambda study: (
            chapter_order[study["chapter"]],
            study["order"],
            study["id"],
        ),
    )


def _link(target: Path, output: Path, label: str) -> str:
    relative = Path(os.path.relpath(ROOT / target, start=(ROOT / output).parent))
    value = relative.as_posix()
    if " " in value:
        value = f"<{value}>"
    return f"[{label}]({value})"


def _notice(source: str) -> list[str]:
    return [
        "> [!NOTE]",
        f"> This navigation page is generated from `{source}`. The current",
        '> `review_state = "indexed"` means the records are organized and checked,',
        "> but the prose is intentionally provisional pending the second synthesis pass.",
        "",
    ]


def _study_readme(study: dict[str, Any]) -> tuple[Path, str]:
    output = Path("studies") / study["id"] / "README.md"
    lines = [f"# {study['title']}", ""] + _notice(f"studies/{study['id']}/study.toml")
    lines.extend(
        [
            _link(Path("studies/README.md"), output, "Back to the study registry"),
            "",
            f"- **Status:** `{study['status']}`",
            f"- **Review state:** `{study['review_state']}`",
            f"- **Study ID:** `{study['id']}`",
            "",
            "## Scientific role",
            "",
            f"**Question.** {study['question']}",
            "",
            f"**Finding.** {study['finding']}",
            "",
            f"**Claim boundary.** {study['boundary']}",
            "",
            "## Frozen records",
            "",
        ]
    )
    for record in study["records"]:
        target = Path("studies") / study["id"] / record["path"]
        lines.append(
            f"- `{record['role']}` — "
            f"{_link(target, output, record['legacy_path'])} "
            f"(`sha256:{record['sha256'][:12]}`)"
        )
    retired_assets = study.get("retired_assets", [])
    if retired_assets:
        lines.extend(
            [
                "",
                "## Retired historical assets",
                "",
                "These files are intentionally absent from the current worktree. Their",
                "exact bytes remain recoverable from the recorded Git source ref.",
                "",
            ]
        )
        for asset in retired_assets:
            lines.append(
                f"- `{asset['role']}` — `{asset['path']}` "
                f"(`sha256:{asset['sha256'][:12]}`, source `{asset['source_ref']}`) — "
                f"{asset['reason']}"
            )
    lines.extend(
        [
            "",
            "## Provenance rule",
            "",
            "Files under `records/` are byte-preserving relocations. Their former paths,",
            "hashes, sizes, and source ref are recorded in `study.toml` and the global",
            "migration map. New interpretation belongs in this capsule or `synthesis/`;",
            "the frozen records themselves are not rewritten.",
            "",
            "Add a `figures/` directory only when this study has a promoted, reproducible",
            "study-level figure. Cross-study paper figures belong in `synthesis/figures/`.",
            "",
        ]
    )
    return output, "\n".join(lines)


def _studies_readme(
    registry: dict[str, Any], studies: dict[str, dict[str, Any]]
) -> str:
    output = Path("studies/README.md")
    lines = ["# Study registry", ""] + _notice("studies/registry.toml")
    lines.extend(
        [
            "This directory is the experiment-level source of truth. Each capsule joins",
            "one scientific question to its registered protocol, execution locks, exact",
            "results, report, outcome boundary, and provenance hashes.",
            "",
            f"Start with {_link(Path('synthesis/README.md'), output, 'the current synthesis')}",
            "for a short reading route; return here for complete evidence and negative",
            "results.",
            "",
        ]
    )
    for chapter in _ordered_chapters(registry):
        lines.extend([f"## {chapter['title']}", "", chapter["purpose"], ""])
        for study in _ordered_studies(registry, studies, chapter["id"]):
            lines.append(
                f"- [{study['title']}]({study['id']}/README.md) — "
                f"`{study['status']}` — {study['finding']}"
            )
        lines.append("")
    lines.extend(["## Status vocabulary", ""])
    for status, meaning in registry["status_legend"].items():
        lines.append(f"- `{status}`: {meaning}")
    lines.extend(
        [
            "",
            "## Maintenance",
            "",
            "Run `direnv exec . python -m fsrl.infrastructure.study_registry check` before commit.",
            "Use `build` only to refresh generated navigation after editing TOML metadata.",
            "A path move requires a new versioned migration; it is not a prose edit.",
            "",
        ]
    )
    return "\n".join(lines)


def _synthesis_readme(
    registry: dict[str, Any],
    studies: dict[str, dict[str, Any]],
    synthesis: dict[str, Any],
) -> str:
    output = Path("synthesis/README.md")
    lines = [f"# {synthesis['title']}", ""] + _notice("synthesis/manifest.toml")
    lines.extend(
        [
            synthesis["scope"],
            "",
            f"**Current working claim.** {synthesis['working_claim']}",
            "",
            f"**Boundary.** {synthesis['boundary']}",
            "",
            "## Start here",
            "",
            f"- {_link(Path('studies/README.md'), output, 'Complete study registry')}",
            f"- {_link(Path('synthesis/frozen/README.md'), output, 'Frozen evidence overlay')}",
            f"- {_link(Path('synthesis/history.toml'), output, 'Release and migration history')}",
            f"- {_link(Path('synthesis/figures/README.md'), output, 'Figure workflow')}",
            f"- {_link(Path('workflows/relational_model/README.md'), output, 'Maintained model workflow')}",
            "",
            "The frozen overlay is a machine-verifiable historical reporting object. This",
            "page is the editable human synthesis; neither replaces the study records.",
            "",
            "## Provenance layers",
            "",
            "Byte-preserved reports, contracts, locks, results, and presentation assets",
            "live in study-owned `records/` or `synthesis/records/`. Frozen execution",
            "locks that name historical Python files are indexed by `(path, sha256)` in",
            "`synthesis/source-provenance.toml` and verified against immutable Git blobs",
            "and witness commits. Maintained model source and tests live in `fsrl/` and",
            "`tests/`; the original-paper reproduction is isolated under `reproductions/`",
            "with its own upstream byte locks. Historical replay uses a detached Git",
            "worktree rather than mixing old files into the current import tree.",
            "",
            "## Reading routes",
            "",
        ]
    )
    for view in registry["views"]:
        lines.extend([f"### {view['title']}", "", view["purpose"], ""])
        for index, study_id in enumerate(view["studies"], start=1):
            study = studies[study_id]
            target = Path("studies") / study_id / "README.md"
            lines.append(
                f"{index}. {_link(target, output, study['title'])} — "
                f"`{study['status']}` — {study['finding']}"
            )
        lines.append("")
    lines.extend(
        [
            "## What this first refactor does not claim",
            "",
            "The order above is a checked navigation layer, not yet the final manuscript",
            'argument. `review_state = "indexed"` deliberately leaves room for a second',
            "pass that compresses methods, chooses paper-level estimands, and promotes only",
            "the figures needed for the final claim structure.",
            "",
        ]
    )
    return "\n".join(lines)


def _figures_readme() -> str:
    return """# Figure workflow

This directory is reserved for cross-study figures intended for a report or
paper. The architecture distinguishes three layers:

1. exact historical outputs remain under each study's `records/` directory;
2. reproducible study-level figures may be promoted to `studies/<id>/figures/`;
3. cross-study figures and their machine-readable source tables live here.

Every promoted figure should have a source-data file, a generation command, and
study/estimand provenance. Prefer a stable figure ID whose directory contains
the rendered panel, source table, generation script or command, and a manifest
mapping every panel to study IDs and frozen estimands. Historical presentation
assets remain under `synthesis/records/` until the second curation pass decides
whether to regenerate or retire them. Do not copy an image here merely to make
it easier to find.

## Current suites

- [Published behavioral figure alignment](paper_alignment/README.md) redraws
  released human results and places the frozen two-network model on the same
  estimands. Its manifest records the exact sources, exclusions, and rendered
  outputs.
"""


def render_navigation(
    registry: dict[str, Any] | None = None,
    synthesis: dict[str, Any] | None = None,
) -> dict[Path, str]:
    registry = load_registry() if registry is None else registry
    synthesis = load_synthesis() if synthesis is None else synthesis
    studies = load_studies(registry)
    rendered = {
        GENERATED_PATHS[0]: _studies_readme(registry, studies),
        GENERATED_PATHS[1]: _synthesis_readme(registry, studies, synthesis),
        GENERATED_PATHS[2]: _figures_readme(),
    }
    for study in studies.values():
        path, content = _study_readme(study)
        rendered[path] = content
    return rendered


def build_navigation() -> dict[str, Any]:
    validation = validate_registry()
    if not validation["passed"]:
        return validation
    changed: list[str] = []
    for relative, content in render_navigation().items():
        path = ROOT / relative
        expected = content.rstrip() + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
            changed.append(relative.as_posix())
    return {**validation, "passed": True, "changed": changed}


def check_navigation() -> dict[str, Any]:
    validation = validate_registry()
    if not validation["passed"]:
        return validation
    stale: list[str] = []
    rendered = render_navigation()
    for relative, content in rendered.items():
        path = ROOT / relative
        expected = content.rstrip() + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            stale.append(relative.as_posix())
    expected_study_readmes = {
        path.as_posix()
        for path in rendered
        if path.parts[:1] == ("studies",) and len(path.parts) == 3
    }
    existing_study_readmes = {
        path.relative_to(ROOT).as_posix() for path in STUDIES_ROOT.glob("*/README.md")
    }
    unexpected = sorted(existing_study_readmes - expected_study_readmes)
    return {
        **validation,
        "passed": not stale and not unexpected,
        "stale_generated_files": stale,
        "unexpected_study_readmes": unexpected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "build", "check"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = validate_registry()
    elif args.command == "build":
        result = build_navigation()
    else:
        result = check_navigation()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
