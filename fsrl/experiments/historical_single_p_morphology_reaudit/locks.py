"""Fail-closed source and archived-artifact lock for the historical re-audit."""

from __future__ import annotations

import subprocess

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIR,
    REPAIR_SHA256,
    SOURCE_INPUT_LOCK,
    register,
    specification,
    unit_directory,
)


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_pushed_commit() -> str:
    if _git_text("branch", "--show-current") != "dev":
        raise RuntimeError("historical morphology re-audit requires dev")
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit qualified re-audit source before locking")
    commit = _git_text("rev-parse", "HEAD")
    remote = _git_text("ls-remote", "--exit-code", "origin", "refs/heads/dev").split()[
        0
    ]
    if commit != remote:
        raise RuntimeError("qualified re-audit source must be pushed to origin/dev")
    return commit


def require_committed(path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    commit = _git_text("rev-parse", "HEAD")
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"registered freeze point is not committed: {relative}")
    return commit


def sources() -> list[dict]:
    package = REPO_ROOT / "fsrl/experiments/historical_single_p_morphology_reaudit"
    paths = list(package.rglob("*.py"))
    paths += list(
        (
            REPO_ROOT
            / "tests/experiments/historical_single_p_morphology_reaudit"
        ).rglob("*.py")
    )
    paths += [
        REPO_ROOT / "fsrl/experiments/minimal_single_p_pair_morphology/analysis.py",
        REPO_ROOT / "fsrl/experiments/minimal_single_p_pair_morphology/methods.py",
        REPO_ROOT / "fsrl/infra/formal_runtime.py",
        PROTOCOL,
        REPAIR,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".envrc",
    ]
    return [reference(path) for path in sorted(set(paths))]


def _parent(name: str):
    row = specification()["parents"][name]
    path = REPO_ROOT / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"parent record changed: {name}")
    return path


def _units() -> dict:
    units = {}
    spec = specification()
    for family, family_spec in spec["design"]["families"].items():
        for seed in family_spec["seeds"]:
            for panel in spec["design"]["panels"]:
                for condition in spec["design"]["conditions"]:
                    directory = unit_directory(family, seed, panel, condition)
                    manifest = validate_run_manifest(directory / "run.json")
                    if not manifest["passed"]:
                        raise RuntimeError(
                            f"incomplete historical unit: {family}/{seed}/{panel}/{condition}"
                        )
                    units[f"{family}/{seed}/{panel}/{condition}"] = {
                        name: reference(directory / name)
                        for name in ("raw.npz", "behavior.json", "result.json", "run.json")
                    }
    return units


def write_source_input_lock() -> dict:
    commit = clean_pushed_commit()
    require_committed(QUALIFICATION)
    qualification = load_json(QUALIFICATION)
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("historical morphology repair contract changed")
    current = sources()
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("historical morphology qualification did not pass")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "repair": reference(REPAIR),
        "parents": {
            name: reference(_parent(name)) for name in specification()["parents"]
        },
        "units": _units(),
        "scientific_outcomes_exposed": False,
    }
    if len(payload["units"]) != specification()["design"]["mandatory_units"]:
        raise RuntimeError("historical unit count differs")
    write_json_exclusive(SOURCE_INPUT_LOCK, payload)
    register(finding="Sources and all mandatory historical artifacts locked; analysis pending.")
    return {"source_commit": commit, "units": len(payload["units"])}


def validate_source_input_lock() -> dict:
    require_committed(SOURCE_INPUT_LOCK)
    lock = load_json(SOURCE_INPUT_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("historical morphology protocol lock differs")
    if lock["sources"] != sources():
        raise RuntimeError("historical morphology source inventory differs")
    for row in lock["sources"]:
        observed = git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
        if observed != row["sha256"]:
            raise RuntimeError(f"source Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("locked qualification differs")
    if reference(REPAIR) != lock["repair"]:
        raise RuntimeError("locked repair contract differs")
    for name, row in lock["parents"].items():
        if reference(_parent(name)) != row:
            raise RuntimeError(f"locked parent differs: {name}")
    for files in lock["units"].values():
        for row in files.values():
            verify_reference(row)
    if len(lock["units"]) != specification()["design"]["mandatory_units"]:
        raise RuntimeError("locked historical unit count differs")
    return lock


__all__ = [
    "reference",
    "sources",
    "validate_source_input_lock",
    "write_source_input_lock",
]
