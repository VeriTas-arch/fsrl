"""Audit the scoped 2026 refactors against retained reference paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT / "workflows" / "relational_model" / "refactor_equivalence_v1.json"
)
SEMANTIC_COMMANDS = {
    "study_registry": ["-m", "fsrl.infra.study_registry", "check"],
    "file_contracts": ["-m", "fsrl.infra.file_contracts", "check"],
    "workflow": [
        "-m",
        "fsrl.workflows",
        "check",
        "workflows/relational_model/workflow.toml",
    ],
    "paper_figures": ["-m", "fsrl.workflows.paper_figures", "check"],
    "frozen_evidence": ["-m", "fsrl.workflows.frozen_evidence", "verify"],
}


def _validate_cross_commit_checks(
    contract: dict[str, Any], errors: list[str]
) -> list[str]:
    checks = contract.get("cross_commit_checks", [])
    if not isinstance(checks, list):
        errors.append("cross_commit_checks must be a list")
        return []
    ids = []
    for check in checks:
        if not isinstance(check, dict):
            errors.append("cross_commit_checks entries must be objects")
            continue
        ids.append(check.get("id"))
        script = check.get("script")
        if (
            not isinstance(script, str)
            or Path(script).is_absolute()
            or ".." in Path(script).parts
            or not script.endswith(".py")
        ):
            errors.append("cross-commit scripts must be relative Python paths")
        elif not (ROOT / script).is_file():
            errors.append(f"cross-commit script does not exist: {script}")
    return ids


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if contract.get("document_type") != "fsrl.refactor_equivalence_contract":
        errors.append("invalid document_type")
    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for field in ("baseline_commit", "candidate_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", str(contract.get(field, ""))) is None:
            errors.append(f"{field} must be a full commit ID")
    for field in ("audited_paths", "baseline_reference_modules", "not_claimed"):
        values = contract.get(field)
        if not isinstance(values, list) or not values:
            errors.append(f"{field} must be a non-empty list")
        elif field == "audited_paths" and any(
            Path(value).is_absolute() or ".." in Path(value).parts for value in values
        ):
            errors.append("audited_paths must be repository-relative")
    ids = []
    for field in ("exact_checks", "bounded_checks"):
        checks = contract.get(field)
        if not isinstance(checks, list) or not checks:
            errors.append(f"{field} must be a non-empty list")
            continue
        for check in checks:
            ids.append(check.get("id"))
            selectors = check.get("selectors")
            if not isinstance(selectors, list) or not selectors:
                errors.append(f"{field} entries require selectors")
            elif any(
                not isinstance(value, str)
                or not value.startswith("tests.")
                or any(character.isspace() for character in value)
                for value in selectors
            ):
                errors.append(f"{field} contains an invalid unittest selector")
            if field == "bounded_checks" and (
                check.get("rtol", -1) < 0 or check.get("atol", -1) < 0
            ):
                errors.append("bounded checks require non-negative tolerances")
    ids.extend(_validate_cross_commit_checks(contract, errors))
    if any(not isinstance(value, str) or not value for value in ids):
        errors.append("check IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        errors.append("check IDs must be unique")
    names = contract.get("semantic_checks")
    if (
        not isinstance(names, list)
        or not names
        or any(name not in SEMANTIC_COMMANDS for name in names)
        or len(names) != len(set(names))
    ):
        errors.append("semantic_checks must select unique supported checks")
    if errors:
        raise ValueError(f"invalid refactor equivalence contract: {errors}")
    return contract


def _run(
    check_id: str,
    arguments: list[str],
    *,
    cwd: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(cwd),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    result = {
        "id": check_id,
        "passed": completed.returncode == 0,
        "argv": ["python", *arguments],
        "duration_seconds": round(time.perf_counter() - started, 3),
        "stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
        **(extra or {}),
    }
    if completed.returncode:
        result["stdout_tail"] = completed.stdout[-2000:]
        result["stderr_tail"] = completed.stderr[-2000:]
    return result


def _run_groups(groups: list[dict[str, Any]], cwd: Path) -> dict[str, Any]:
    checks = []
    for group in groups:
        tolerances = {key: group[key] for key in ("rtol", "atol") if key in group}
        checks.append(
            _run(
                group["id"],
                ["-m", "unittest", *group["selectors"]],
                cwd=cwd,
                extra={**tolerances, "selectors": group["selectors"]},
            )
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _run_cross_commit_checks(
    checks: list[dict[str, Any]], baseline_root: Path
) -> dict[str, Any]:
    results = []
    for check in checks:
        script = str(ROOT / check["script"])
        baseline = _run(f"{check['id']}-baseline", [script], cwd=baseline_root)
        candidate = _run(f"{check['id']}-candidate", [script], cwd=ROOT)
        hashes_match = baseline["stdout_sha256"] == candidate["stdout_sha256"]
        results.append(
            {
                "id": check["id"],
                "passed": baseline["passed"] and candidate["passed"] and hashes_match,
                "stdout_sha256_match": hashes_match,
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    return {"passed": all(check["passed"] for check in results), "checks": results}


def _export_commit(commit: str, destination: Path) -> None:
    archive = destination.parent / f"{commit}.tar"
    _git("archive", "--format=tar", f"--output={archive}", commit)
    destination.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")


def run_audit(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_contract(contract_path)
    baseline = contract["baseline_commit"]
    candidate = contract["candidate_commit"]
    for commit in (baseline, candidate):
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    if _git("merge-base", "--is-ancestor", baseline, candidate, check=False).returncode:
        raise RuntimeError("baseline is not an ancestor of candidate")
    if _git("merge-base", "--is-ancestor", candidate, "HEAD", check=False).returncode:
        raise RuntimeError("current HEAD does not descend from the audited candidate")
    changed = _git(
        "diff", "--name-only", candidate, "--", *contract["audited_paths"]
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(
            f"audited paths changed after the candidate commit: {changed}"
        )
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *contract["audited_paths"],
    ).stdout.splitlines()
    if dirty:
        raise RuntimeError(f"audited paths contain worktree changes: {dirty}")

    with tempfile.TemporaryDirectory(prefix="fsrl-refactor-baseline-") as directory:
        baseline_root = Path(directory) / "source"
        _export_commit(baseline, baseline_root)
        baseline_check = _run(
            "baseline-reference-suite",
            ["-m", "unittest", *contract["baseline_reference_modules"]],
            cwd=baseline_root,
            extra={"commit": baseline},
        )
        cross_commit = _run_cross_commit_checks(
            contract.get("cross_commit_checks", []), baseline_root
        )

    exact = _run_groups(contract["exact_checks"], ROOT)
    bounded = _run_groups(contract["bounded_checks"], ROOT)
    semantic_checks = [baseline_check]
    semantic_checks.extend(
        _run(name, SEMANTIC_COMMANDS[name], cwd=ROOT)
        for name in contract["semantic_checks"]
    )
    semantic = {
        "passed": all(check["passed"] for check in semantic_checks),
        "checks": semantic_checks,
    }
    passed = (
        cross_commit["passed"]
        and exact["passed"]
        and bounded["passed"]
        and semantic["passed"]
    )
    failures = [
        check["id"]
        for group in (cross_commit, exact, bounded, semantic)
        for check in group["checks"]
        if not check["passed"]
    ]
    return {
        "identity": {
            "document_type": "fsrl.refactor_equivalence_audit",
            "schema_version": 1,
            "audit_id": contract["audit_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "contract_path": contract_path.relative_to(ROOT).as_posix(),
            "contract_sha256": _sha256(contract_path.read_bytes()),
            "baseline_commit": baseline,
            "candidate_commit": candidate,
            "current_head": _git("rev-parse", "HEAD").stdout.strip(),
        },
        "coverage": {
            "method": "baseline health plus candidate retained-reference comparisons",
            "audited_paths": contract["audited_paths"],
            "not_claimed": contract["not_claimed"],
        },
        "cross_commit_exact_parity": cross_commit,
        "exact_parity": exact,
        "bounded_parity": bounded,
        "semantic_parity": semantic,
        "decision": {
            "passed": passed,
            "status": "pass" if passed else "fail",
            "claim": contract["claim"],
            "failed_checks": failures,
        },
    }


def _parser(default_contract: Path = DEFAULT_CONTRACT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=default_contract)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-contract", action="store_true")
    return parser


def main(
    argv: list[str] | None = None, *, default_contract: Path = DEFAULT_CONTRACT
) -> int:
    arguments = _parser(default_contract).parse_args(argv)
    if arguments.validate_contract:
        contract = load_contract(arguments.contract)
        result = {"passed": True, "audit_id": contract["audit_id"]}
    else:
        result = run_audit(arguments.contract)
    if arguments.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        passed = bool(result.get("passed", result.get("decision", {}).get("passed")))
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "passed": passed,
                    "output": str(arguments.output),
                }
            )
        )
    return 0 if result.get("passed", result.get("decision", {}).get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
