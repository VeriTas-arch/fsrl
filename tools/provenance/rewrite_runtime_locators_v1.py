"""Normalize frozen runtime locators while preserving their original bytes in Git.

Only registered study and synthesis records are rewritten.  The content-addressed
artifact bundle remains an archival object whose member names retain the original
``output/`` prefix; active extraction translates that prefix separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fsrl.infra.study_registry import resolve_record

ROOT = Path(__file__).resolve().parents[2]
FLAT_MIGRATION_PATH = ROOT / "studies" / "migrations" / "flat-records-v1.json"
MIGRATION_PATH = ROOT / "studies" / "migrations" / "runtime-locators-v1.json"
OLD_PREFIX = b"output/"
NEW_PREFIX = b"artifacts/runs/"
DEPENDENT_SOURCE_REF = "0bbc15fc5afa3f121de93b16651936ca8c74d533"
FIGURE_SPEC_PATH = "synthesis/figures/paper_alignment/figure_spec.json"
REPLAY_MANIFEST_PATH = (
    "synthesis/figures/paper_alignment/source/model_subject_pair_accuracy.manifest.json"
)
FIGURE_MANIFEST_PATH = "synthesis/figures/paper_alignment/manifest.json"
MODEL_RESULT_PATH = (
    "studies/dual_evidence_access_confirmation/records/results/"
    "dual_evidence_access_confirmation_v2_4.json"
)
MODEL_ARTIFACT_LOCK_PATH = (
    "studies/dual_evidence_access_confirmation/records/benchmarks/"
    "dual_evidence_access_confirmation_v2_4.artifact_lock.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob(source_ref: str, source_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{source_ref}:{source_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _registered_candidates() -> list[Path]:
    paths = []
    for repository_path in _flat_records():
        is_study_record = (
            repository_path.startswith("studies/") and "/records/" in repository_path
        )
        is_synthesis_record = repository_path.startswith("synthesis/records/")
        if not (is_study_record or is_synthesis_record):
            continue
        path = resolve_record(repository_path)
        if path.is_file() and OLD_PREFIX in path.read_bytes():
            paths.append(path)
    return sorted(set(paths))


def _flat_records() -> dict[str, dict[str, Any]]:
    migration = json.loads(FLAT_MIGRATION_PATH.read_text(encoding="utf-8"))
    return {record["path"]: record for record in migration["records"]}


def _source_candidates() -> dict[str, tuple[dict[str, Any], bytes]]:
    candidates = {}
    for repository_path, registration in _flat_records().items():
        is_study_record = (
            repository_path.startswith("studies/") and "/records/" in repository_path
        )
        is_synthesis_record = repository_path.startswith("synthesis/records/")
        if not (is_study_record or is_synthesis_record):
            continue
        source = _git_blob(registration["source_ref"], registration["legacy_path"])
        if OLD_PREFIX in source:
            candidates[repository_path] = (registration, source)
    return candidates


def _replace_once(payload: bytes, before: str, after: str, *, path: str) -> bytes:
    old = before.encode()
    if payload.count(old) != 1:
        raise RuntimeError(f"dependent hash is not unique in {path}: {before}")
    return payload.replace(old, after.encode())


def _dependent_plan(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    primary = {record["path"]: record for record in records}
    figure_spec_before = _git_blob(DEPENDENT_SOURCE_REF, FIGURE_SPEC_PATH)
    figure_spec_after = figure_spec_before
    for path in (MODEL_RESULT_PATH, MODEL_ARTIFACT_LOCK_PATH):
        record = primary[path]
        figure_spec_after = _replace_once(
            figure_spec_after,
            record["before_sha256"],
            record["after_sha256"],
            path=FIGURE_SPEC_PATH,
        )

    replay_before = _git_blob(DEPENDENT_SOURCE_REF, REPLAY_MANIFEST_PATH)
    replay_after = _replace_once(
        replay_before,
        _sha256(figure_spec_before),
        _sha256(figure_spec_after),
        path=REPLAY_MANIFEST_PATH,
    )
    model_result = primary[MODEL_RESULT_PATH]
    replay_after = _replace_once(
        replay_after,
        model_result["before_sha256"],
        model_result["after_sha256"],
        path=REPLAY_MANIFEST_PATH,
    )

    manifest_before = _git_blob(DEPENDENT_SOURCE_REF, FIGURE_MANIFEST_PATH)
    manifest_after = _replace_once(
        manifest_before,
        _sha256(figure_spec_before),
        _sha256(figure_spec_after),
        path=FIGURE_MANIFEST_PATH,
    )
    manifest_after = _replace_once(
        manifest_after,
        _sha256(replay_before),
        _sha256(replay_after),
        path=FIGURE_MANIFEST_PATH,
    )

    payloads = {
        FIGURE_SPEC_PATH: figure_spec_after,
        REPLAY_MANIFEST_PATH: replay_after,
        FIGURE_MANIFEST_PATH: manifest_after,
    }
    reasons = {
        FIGURE_SPEC_PATH: "Update the two locator-rewritten model-input hashes.",
        REPLAY_MANIFEST_PATH: (
            "Update the figure-specification hash after locator normalization."
        ),
        FIGURE_MANIFEST_PATH: (
            "Update the figure-specification and replay-manifest hashes."
        ),
    }
    dependents = []
    for path, after in payloads.items():
        before = _git_blob(DEPENDENT_SOURCE_REF, path)
        dependents.append(
            {
                "path": path,
                "source_ref": DEPENDENT_SOURCE_REF,
                "before_sha256": _sha256(before),
                "before_bytes": len(before),
                "after_sha256": _sha256(after),
                "after_bytes": len(after),
                "reason": reasons[path],
            }
        )
    return dependents, payloads


def _plan() -> dict[str, Any]:
    records = []
    for repository_path, (registration, source) in _source_candidates().items():
        path = ROOT / repository_path
        before = path.read_bytes()
        if before != source:
            raise RuntimeError(
                f"runtime-locator source differs from frozen Git blob: {repository_path}"
            )
        after = before.replace(OLD_PREFIX, NEW_PREFIX)
        replacements = before.count(OLD_PREFIX)
        if replacements < 1 or after == before:
            raise RuntimeError(f"runtime-locator rewrite is empty: {repository_path}")
        records.append(
            {
                "path": repository_path,
                "source_path": registration["legacy_path"],
                "source_ref": registration["source_ref"],
                "before_sha256": _sha256(before),
                "before_bytes": len(before),
                "after_sha256": _sha256(after),
                "after_bytes": len(after),
                "replacements": replacements,
            }
        )
    dependents, _ = _dependent_plan(records)
    return {
        "schema_version": 1,
        "id": "runtime-locators-v1",
        "mode": "provenance_locked_literal_prefix_rewrite",
        "old_prefix": OLD_PREFIX.decode(),
        "new_prefix": NEW_PREFIX.decode(),
        "record_count": len(records),
        "replacement_count": sum(record["replacements"] for record in records),
        "dependent_update_count": len(dependents),
        "dependent_updates": dependents,
        "records": records,
    }


def audit() -> dict[str, Any]:
    migration = json.loads(MIGRATION_PATH.read_text(encoding="utf-8"))
    errors = []
    records = migration.get("records", [])
    source_candidates = _source_candidates()
    seen = set()
    for record in records:
        path_value = record.get("path", "")
        if path_value in seen:
            errors.append(f"duplicate runtime-locator path: {path_value}")
            continue
        seen.add(path_value)
        if path_value not in source_candidates:
            errors.append(f"unregistered runtime-locator path: {path_value}")
            continue
        try:
            source = _git_blob(record["source_ref"], record["source_path"])
        except (KeyError, subprocess.CalledProcessError):
            errors.append(f"unavailable runtime-locator source: {path_value}")
            continue
        current_path = resolve_record(path_value)
        if not current_path.is_file():
            errors.append(f"missing rewritten runtime-locator record: {path_value}")
            continue
        current = current_path.read_bytes()
        expected = source.replace(OLD_PREFIX, NEW_PREFIX)
        checks = {
            "before_sha256": _sha256(source),
            "before_bytes": len(source),
            "after_sha256": _sha256(expected),
            "after_bytes": len(expected),
            "replacements": source.count(OLD_PREFIX),
        }
        for field, observed in checks.items():
            if record.get(field) != observed:
                errors.append(f"runtime-locator {field} mismatch: {path_value}")
        if current != expected:
            errors.append(f"runtime-locator content mismatch: {path_value}")
    candidates = {
        path.relative_to(ROOT).as_posix() for path in _registered_candidates()
    }
    if candidates:
        errors.append(f"unrewritten runtime locators remain: {sorted(candidates)}")
    missing = sorted(set(source_candidates) - seen)
    extra = sorted(seen - set(source_candidates))
    if missing:
        errors.append(f"runtime-locator migration is missing records: {missing}")
    if extra:
        errors.append(f"runtime-locator migration has extra records: {extra}")
    if migration.get("schema_version") != 1:
        errors.append("runtime-locator schema_version must be 1")
    if migration.get("old_prefix") != OLD_PREFIX.decode():
        errors.append("runtime-locator old_prefix mismatch")
    if migration.get("new_prefix") != NEW_PREFIX.decode():
        errors.append("runtime-locator new_prefix mismatch")
    if migration.get("record_count") != len(records):
        errors.append("runtime-locator record_count mismatch")
    replacements = sum(record.get("replacements", 0) for record in records)
    if migration.get("replacement_count") != replacements:
        errors.append("runtime-locator replacement_count mismatch")
    dependents = migration.get("dependent_updates", [])
    expected_dependents, dependent_payloads = _dependent_plan(records)
    if migration.get("dependent_update_count") != len(dependents):
        errors.append("runtime-locator dependent_update_count mismatch")
    expected_by_path = {record["path"]: record for record in expected_dependents}
    if {record.get("path") for record in dependents} != set(expected_by_path):
        errors.append("runtime-locator dependent paths mismatch")
    for dependent in dependents:
        path_value = dependent.get("path", "")
        expected_record = expected_by_path.get(path_value)
        if expected_record is None:
            continue
        path = ROOT / path_value
        if not path.is_file():
            errors.append(f"missing runtime-locator dependent: {path_value}")
            continue
        for field, expected in expected_record.items():
            if dependent.get(field) != expected:
                errors.append(
                    f"runtime-locator dependent {field} mismatch: {path_value}"
                )
        if path.read_bytes() != dependent_payloads[path_value]:
            errors.append(f"runtime-locator dependent content mismatch: {path_value}")
    return {
        "passed": not errors,
        "records": len(records),
        "replacements": replacements,
        "dependent_updates": len(dependents),
        "errors": errors,
    }


def apply() -> dict[str, Any]:
    if MIGRATION_PATH.exists():
        return audit()
    migration = _plan()
    for record in migration["records"]:
        path = resolve_record(record["path"])
        path.write_bytes(path.read_bytes().replace(OLD_PREFIX, NEW_PREFIX))
    _, dependent_payloads = _dependent_plan(migration["records"])
    for path, payload in dependent_payloads.items():
        (ROOT / path).write_bytes(payload)
    MIGRATION_PATH.write_text(
        json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "apply"))
    parsed = parser.parse_args(argv)
    result = apply() if parsed.command == "apply" else audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
