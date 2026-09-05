"""Index every frozen Python source lock against reachable Git history."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = ROOT / "studies"
SYNTHESIS_ROOT = ROOT / "synthesis"
MANIFEST_PATH = SYNTHESIS_ROOT / "source-provenance.toml"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _git(*args: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _owner_record_paths(manifest_path: Path) -> tuple[list[Path], set[Path]]:
    """Exclude prospectively native evidence from the frozen v1 source index."""

    owner = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    historical: list[Path] = []
    native: set[Path] = set()
    for record in owner["records"]:
        path = manifest_path.parent / record["path"]
        if record.get("origin", "migrated") == "native":
            native.add(path)
        else:
            historical.append(path)
    if not historical:
        native.add(manifest_path)
    return historical, native


def _record_paths() -> tuple[list[Path], int]:
    registry = tomllib.loads(
        (STUDIES_ROOT / "registry.toml").read_text(encoding="utf-8")
    )
    paths: list[Path] = []
    native_paths: set[Path] = set()
    manifests = [ROOT / entry["path"] for entry in registry.get("studies", [])]
    for manifest in [*manifests, SYNTHESIS_ROOT / "manifest.toml"]:
        historical, native = _owner_record_paths(manifest)
        paths.extend(historical)
        native_paths.update(native)
    registered_count = len(paths)
    tracked = _git("ls-files", "-z", "--", "studies", "synthesis")
    assert isinstance(tracked, bytes)
    for value in tracked.split(b"\0"):
        if not value:
            continue
        path = ROOT / value.decode()
        if path == MANIFEST_PATH or path in native_paths or not path.is_file():
            continue
        if path.suffix in {".json", ".toml"} or path.name.endswith(".json.gz"):
            paths.append(path)
    return sorted(set(paths)), registered_count


def _load_structured(path: Path) -> Any | None:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    return None


def _source_registration(value: dict[str, Any]) -> tuple[str, str] | None:
    path = value.get("path")
    sha256 = value.get("sha256")
    if (
        not isinstance(path, str)
        or not isinstance(sha256, str)
        or not path.endswith(".py")
        or not path.startswith(("fsrl/", "tests/"))
        or SHA256_PATTERN.fullmatch(sha256) is None
    ):
        return None
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"unsafe registered source path: {path}")
    return candidate.as_posix(), sha256


def source_references() -> tuple[dict[tuple[str, str], dict[str, Any]], int, int, int]:
    references: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"records": set(), "occurrences": 0}
    )
    record_paths, registered_count = _record_paths()

    def visit(value: Any, record_path: str) -> None:
        if isinstance(value, dict):
            registration = _source_registration(value)
            if registration is not None:
                row = references[registration]
                row["records"].add(record_path)
                row["occurrences"] += 1
            for child in value.values():
                visit(child, record_path)
        elif isinstance(value, list):
            for child in value:
                visit(child, record_path)

    for path in record_paths:
        value = _load_structured(path)
        if value is not None:
            visit(value, path.relative_to(ROOT).as_posix())
    occurrences = sum(int(row["occurrences"]) for row in references.values())
    return references, registered_count, len(record_paths), occurrences


def _locate_versions(
    references: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted_by_path: dict[str, set[str]] = defaultdict(set)
    for path, sha256 in references:
        wanted_by_path[path].add(sha256)

    located: dict[tuple[str, str], dict[str, Any]] = {}
    for path, wanted in sorted(wanted_by_path.items()):
        commits = str(
            _git("log", "HEAD", "--format=%H", "--", path, text=True)
        ).splitlines()
        for commit in commits:
            try:
                payload = _git("show", f"{commit}:{path}")
            except subprocess.CalledProcessError:
                continue
            assert isinstance(payload, bytes)
            digest = _sha256(payload)
            key = (path, digest)
            if digest not in wanted or key in located:
                continue
            git_blob = str(_git("rev-parse", f"{commit}:{path}", text=True)).strip()
            reference = references[key]
            located[key] = {
                "path": path,
                "sha256": digest,
                "bytes": len(payload),
                "git_blob": git_blob,
                "witness_commit": commit,
                "record_count": len(reference["records"]),
                "occurrence_count": reference["occurrences"],
            }
            if wanted.issubset(
                {sha for candidate, sha in located if candidate == path}
            ):
                break

    missing = sorted(set(references) - set(located))
    if missing:
        raise RuntimeError(
            f"registered source versions are absent from HEAD history: {missing}"
        )
    return [located[key] for key in sorted(located)]


def source_records() -> tuple[list[dict[str, Any]], int, int, int]:
    references, registered_count, indexed_count, occurrence_count = source_references()
    return (
        _locate_versions(references),
        registered_count,
        indexed_count,
        occurrence_count,
    )


def _manifest(
    records: list[dict[str, Any]],
    registered_count: int,
    indexed_count: int,
    occurrence_count: int,
) -> str:
    lines = [
        "schema_version = 1",
        'id = "frozen-source-provenance-v1"',
        'git_object_format = "sha1"',
        f"registered_record_files = {registered_count}",
        f"indexed_structured_files = {indexed_count}",
        f"source_reference_occurrences = {occurrence_count}",
        f"source_version_count = {len(records)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                "[[sources]]",
                f"path = {_quote(record['path'])}",
                f"sha256 = {_quote(record['sha256'])}",
                f"bytes = {record['bytes']}",
                f"git_blob = {_quote(record['git_blob'])}",
                f"witness_commit = {_quote(record['witness_commit'])}",
                f"record_count = {record['record_count']}",
                f"occurrence_count = {record['occurrence_count']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run(*, apply: bool) -> dict[str, Any]:
    records, registered_count, indexed_count, occurrence_count = source_records()
    expected = _manifest(records, registered_count, indexed_count, occurrence_count)
    if apply:
        MANIFEST_PATH.write_text(expected, encoding="utf-8")

    errors: list[str] = []
    if not MANIFEST_PATH.is_file():
        errors.append("source provenance manifest is missing")
    elif MANIFEST_PATH.read_text(encoding="utf-8") != expected:
        errors.append("source provenance manifest is stale")
    for redundant_root in (
        SYNTHESIS_ROOT / "snapshots" / "reporting_v1" / "source",
        SYNTHESIS_ROOT / "snapshots" / "reporting_v1" / "source-blobs",
    ):
        if redundant_root.exists():
            errors.append(f"redundant source tree still exists: {redundant_root}")
    return {
        "passed": not errors,
        "apply": apply,
        "registered_record_files": registered_count,
        "indexed_structured_files": indexed_count,
        "source_reference_occurrences": occurrence_count,
        "source_versions": len(records),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    args = parser.parse_args(argv)
    result = run(apply=args.command == "apply")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
