"""Reject new or worsened McCabe-complexity debt while preserving legacy code."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fsrl.paths import REPO_ROOT

BUDGET_PATH = Path(__file__).with_name("complexity_budget.json")
TARGETS = ("fsrl", "tests", "tools", "reproductions")
_MESSAGE = re.compile(r"`(?P<function>[^`]+)` is too complex \((?P<value>\d+) > 10\)")


def load_budget(path: Path = BUDGET_PATH) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("document_type") != "fsrl.complexity_budget":
        raise ValueError("invalid complexity-budget document_type")
    if payload.get("schema_version") != 1 or payload.get("ruff_rule") != "C901":
        raise ValueError("unsupported complexity-budget contract")
    if payload.get("maximum_new_complexity") != 10:
        raise ValueError("complexity budget must retain the Ruff C901 threshold")
    limits = payload.get("legacy_limits")
    if not isinstance(limits, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 10
        for key, value in limits.items()
    ):
        raise ValueError("legacy complexity limits must map identities to integers")
    return limits


def collect_complexity(
    root: Path = REPO_ROOT, targets: tuple[str, ...] = TARGETS
) -> dict[str, int]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *targets,
        "--select",
        "C901",
        "--output-format",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "Ruff complexity check failed")
    diagnostics: list[dict[str, Any]] = json.loads(completed.stdout or "[]")
    observed: dict[str, int] = {}
    for diagnostic in diagnostics:
        match = _MESSAGE.fullmatch(diagnostic["message"])
        if match is None:
            raise ValueError(f"unexpected Ruff C901 message: {diagnostic['message']}")
        source = Path(diagnostic["filename"]).resolve().relative_to(root.resolve())
        identity = f"{source.as_posix()}:{match.group('function')}"
        if identity in observed:
            raise ValueError(f"duplicate complexity identity: {identity}")
        observed[identity] = int(match.group("value"))
    return observed


def audit_complexity_budget(
    budget_path: Path = BUDGET_PATH, root: Path = REPO_ROOT
) -> dict[str, Any]:
    budget = load_budget(budget_path)
    observed = collect_complexity(root)
    unexpected = {
        identity: value
        for identity, value in observed.items()
        if identity not in budget
    }
    worsened = {
        identity: {"allowed": budget[identity], "observed": value}
        for identity, value in observed.items()
        if identity in budget and value > budget[identity]
    }
    improved = {
        identity: {"baseline": allowed, "observed": observed.get(identity, 10)}
        for identity, allowed in budget.items()
        if observed.get(identity, 10) < allowed
    }
    return {
        "passed": not unexpected and not worsened,
        "threshold": 10,
        "observed_legacy_violations": len(observed),
        "unexpected": unexpected,
        "worsened": worsened,
        "improved": improved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = audit_complexity_budget()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
