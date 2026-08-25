"""Freeze historical Python bytes as non-importable content-addressed blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "fb32095df70b1265b5d14b8eda3be6cb65036c6a"
OUTPUT_ROOT = ROOT / "synthesis" / "frozen" / "source-blobs"
MANIFEST_PATH = ROOT / "synthesis" / "source-snapshots.toml"


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


def source_records() -> list[dict[str, object]]:
    changed = str(
        _git(
            "diff",
            "--name-only",
            "--diff-filter=AM",
            SOURCE_COMMIT,
            "--",
            "fsrl",
            "tests",
            text=True,
        )
    ).splitlines()
    records = []
    for value in sorted(changed):
        source_path = Path(value)
        if source_path.suffix != ".py" or not (ROOT / source_path).is_file():
            continue
        try:
            payload = _git("show", f"{SOURCE_COMMIT}:{source_path.as_posix()}")
        except subprocess.CalledProcessError:
            continue
        assert isinstance(payload, bytes)
        if payload == (ROOT / source_path).read_bytes():
            continue
        digest = _sha256(payload)
        local_path = Path("frozen") / "source-blobs" / digest
        records.append(
            {
                "source_path": source_path.as_posix(),
                "path": local_path.as_posix(),
                "sha256": digest,
                "bytes": len(payload),
                "source_commit": SOURCE_COMMIT,
                "payload": payload,
            }
        )
    return records


def _manifest(records: list[dict[str, object]]) -> str:
    lines = [
        "schema_version = 1",
        'id = "pre-refactor-source-snapshots-v1"',
        f"source_commit = {_quote(SOURCE_COMMIT)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                "[[snapshots]]",
                f"source_path = {_quote(str(record['source_path']))}",
                f"path = {_quote(str(record['path']))}",
                f"sha256 = {_quote(str(record['sha256']))}",
                f"bytes = {record['bytes']}",
                f"source_commit = {_quote(SOURCE_COMMIT)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run(*, apply: bool) -> dict[str, object]:
    records = source_records()
    expected_manifest = _manifest(records)
    if apply:
        for record in records:
            path = ROOT / "synthesis" / str(record["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(record["payload"])
        MANIFEST_PATH.write_text(expected_manifest, encoding="utf-8")
    errors: list[str] = []
    if not MANIFEST_PATH.is_file():
        errors.append("source snapshot manifest is missing")
    elif MANIFEST_PATH.read_text(encoding="utf-8") != expected_manifest:
        errors.append("source snapshot manifest is stale")
    expected_paths = {
        (ROOT / "synthesis" / str(record["path"])).resolve() for record in records
    }
    observed_paths = {
        path.resolve() for path in OUTPUT_ROOT.rglob("*") if path.is_file()
    }
    if observed_paths != expected_paths:
        errors.append("source snapshot file set is stale")
    for record in records:
        path = ROOT / "synthesis" / str(record["path"])
        payload = record["payload"]
        assert isinstance(payload, bytes)
        if not path.is_file() or path.read_bytes() != payload:
            errors.append(f"source snapshot differs: {record['source_path']}")
    return {
        "passed": not errors,
        "apply": apply,
        "source_commit": SOURCE_COMMIT,
        "snapshots": len(records),
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
