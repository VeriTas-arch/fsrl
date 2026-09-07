"""Commit witnesses before trajectory exposure and calibration before intervention."""

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.locks import completed
from fsrl.experiments.write_cost.reporting import copy_artifact
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import CALIBRATION, PROTOCOL, RECORDS, RUNS, SOURCE, parents


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/modulation_schedule").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(p) for p in sorted(paths)]


def freeze():
    commit = require_clean()
    qualification = RECORDS / "benchmarks/qualification.json"
    q = load_json(qualification)
    assert q["passed"] and q["sources"] == sources()
    prior = parents()
    refs = sources() + [reference(PROTOCOL), reference(qualification)]
    refs += list(prior["result"]["artifacts"].values())
    refs += [
        v["file"]
        for p in prior["source"]["panels"].values()
        for v in p["inputs"].values()
    ]
    for ref in refs:
        verify_reference(ref, commit=commit)
    result = {"source_commit": commit, "references": refs, "runtime": q["runtime"]}
    write_json_exclusive(SOURCE, result)
    return {"source_commit": commit, "references": len(refs)}


def validate(*, calibrated=False, gpu=True):
    commit = require_clean()
    lock = load_json(verify_reference(reference(SOURCE), commit=commit))
    for ref in lock["references"]:
        verify_reference(ref, commit=lock["source_commit"])
    if gpu and json_ready(configure_execution()) != lock["runtime"]:
        raise RuntimeError("runtime differs from the frozen affine reference")
    if calibrated:
        calibration = load_json(verify_reference(reference(CALIBRATION), commit=commit))
        verify_reference(calibration["audit"])
    return parents()


def lock_calibration():
    validate(gpu=False)
    audit = RECORDS / "results/audit.json"
    result = completed(RUNS / "audit")
    assert result["passed"]
    copy_artifact(RUNS / "audit/result.json", audit)
    lock = {
        "audit": reference(audit),
        "models": result["calibration"],
        "source": reference(SOURCE),
    }
    write_json_exclusive(CALIBRATION, lock)
    return {"locked_models": len(lock["models"]), "new_interventions_exposed": False}
