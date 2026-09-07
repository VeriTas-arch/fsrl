"""Lock the existing cohort, six trained networks, archives and current source."""

import tomllib

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.locks import reference, verify_reference
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
    for row in result["files"]:
        verify_reference(row, commit=result["source_commit"])
    return result
