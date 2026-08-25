"""Read-only helpers for verifying historical source registrations in Git."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _registered_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"registered Git path must be repository-relative: {value}")
    return path


def git_blob_sha256(root: Path, commit: str, path: str) -> str:
    """Hash one repository blob as it existed at an exact commit."""

    if _FULL_COMMIT.fullmatch(commit) is None:
        raise RuntimeError(f"historical verification requires a full commit: {commit}")
    registered_path = _registered_path(path)
    completed = subprocess.run(
        ["git", "show", f"{commit}:{registered_path.as_posix()}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"registered Git blob is unavailable: {commit}:{registered_path} ({detail})"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def verify_git_registrations(
    root: Path,
    commit: str,
    registrations: Mapping[str, Mapping[str, str]],
) -> dict:
    """Verify named path/SHA-256 registrations against historical Git blobs."""

    if not registrations:
        raise RuntimeError("historical source registration set is empty")
    checks = []
    for name, registration in registrations.items():
        path = registration.get("path", "")
        expected = registration.get("sha256", "")
        if _SHA256.fullmatch(expected) is None:
            raise RuntimeError(f"invalid registered SHA-256 for {name}")
        observed = git_blob_sha256(root, commit, path)
        checks.append(
            {
                "name": name,
                "path": path,
                "commit": commit,
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"historical Git source lock failed: {checks}")
    return {"passed": True, "commit": commit, "checks": checks}
