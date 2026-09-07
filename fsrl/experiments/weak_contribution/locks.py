"""Lock the existing cohort, six trained networks, archives and current source."""

import tomllib

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import LOCK, PROTOCOL, QUALIFICATION, parent, parent_spec, specification


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/weak_contribution").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(paths)]


def inputs():
    base = parent()
    paths = [
        base / "benchmarks/protocol.json",
        base / "benchmarks/model_lock.json",
        base / "benchmarks/source_lock.json",
        base / "artifacts/inputs/liu-8.npz",
        base / "results/result.json",
    ]
    for seed in specification()["design"]["seeds"]:
        for arm in specification()["design"]["arms"]:
            paths += [
                base / f"artifacts/training/{seed}/{arm}/{name}"
                for name in ("net.pth", "local.pth", "result.json")
            ]
            paths += [
                base / f"artifacts/liu/{seed}/{arm}/{name}"
                for name in ("raw.npz", "result.json")
            ]
    manifest = tomllib.loads((base.parent / "study.toml").read_text())
    registered = {str(base.parent / row["path"]): row for row in manifest["records"]}
    for path in paths:
        row = reference(path)
        if any(row[key] != registered[str(path)][key] for key in ("sha256", "bytes")):
            raise RuntimeError(f"parent registered evidence changed: {path}")
    paths += [
        protocol_path("liu_v2"),
        REPO_ROOT / parent_spec()["references"]["item_count_protocol"]["path"],
    ]
    return [reference(path) for path in sorted(set(paths))]


def lock():
    commit = require_clean()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("current-source qualification required")
    files = sources() + inputs() + [reference(PROTOCOL), reference(QUALIFICATION)]
    for row in files:
        verify_reference(row, commit=commit)
    result = {
        "source_commit": commit,
        "files": files,
        "runtime": qualification["runtime"],
        "status": "locked_before_new_contribution_outcomes",
    }
    write_json_exclusive(LOCK, result)
    return {"files": len(files), "source_commit": commit}


def validate():
    head = require_clean()
    verify_reference(reference(LOCK), commit=head)
    result = load_json(LOCK)
    repair_path = LOCK.with_name("source_repair_lock.json")
    replacements = {}
    repair_commit = None
    if repair_path.exists():
        verify_reference(reference(repair_path), commit=head)
        repair = load_json(repair_path)
        if repair["original_source_lock"] != reference(LOCK):
            raise RuntimeError("repair belongs to another source lock")
        originals = {row["path"]: row for row in result["files"]}
        for row in repair["replacements"]:
            path = row["original"]["path"]
            if row["original"] != originals[path] or row["replacement"]["path"] != path:
                raise RuntimeError("invalid source replacement mapping")
            if (
                git_blob_sha256(REPO_ROOT, result["source_commit"], path)
                != row["original"]["sha256"]
            ):
                raise RuntimeError("original source witness differs")
            replacements[path] = row["replacement"]
        repair_commit = repair["source_commit"]
        result["source_repair"] = reference(repair_path)
    for row in result["files"]:
        replacement = replacements.get(row["path"])
        if replacement is None:
            verify_reference(row, commit=result["source_commit"])
        else:
            verify_reference(replacement, commit=repair_commit)
    return result
