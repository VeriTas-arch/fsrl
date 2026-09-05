"""Worktree-aware prospective source/input witnesses; frozen parents stay untouched."""

import subprocess
import tomllib

import numpy as np

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "score_circuit/records"
PROTOCOL = RECORDS / "benchmarks/score_circuit_v1.json"
PROTOCOL_HASH = "c01cf52ee9a4720ee2d2b9067014bcee832e01bd9771f8f5f4285aa310b22e1b"
PROTOCOL_COMMIT = "db7145093501ba2051b11e0628182cd19525de72"
LOCK = RECORDS / "benchmarks/score_circuit_v1.execution_lock.json"
QUALIFICATION = RECORDS / "benchmarks/score_circuit_v1.qualification.json"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_HASH:
        raise RuntimeError("score circuit protocol changed")
    verify_reference(reference(PROTOCOL), commit=PROTOCOL_COMMIT)
    return load_json(PROTOCOL)


def require_clean_pushed() -> str:
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("circuit execution requires a clean committed worktree")
    head = git("rev-parse", "HEAD")
    if (
        git("merge-base", specification()["parent_commit"], head)
        != specification()["parent_commit"]
    ):
        raise RuntimeError("worktree is not descended from the frozen dev parent")
    if head != git("ls-remote", "origin", "refs/heads/dev").split()[0]:
        raise RuntimeError("circuit HEAD must be pushed to origin/dev")
    return head


def parent_result() -> dict:
    return load_json(REPO_ROOT / specification()["parent_result"])


def input_records() -> list[dict]:
    spec = specification()
    manifest = REPO_ROOT / spec["parent_study"]
    refs = [reference(manifest)]
    for row in tomllib.loads(manifest.read_text())["records"]:
        ref = reference(manifest.parent / row["path"])
        if ref["sha256"] != row["sha256"] or ref["bytes"] != row["bytes"]:
            raise RuntimeError("parent registered record mismatch")
        refs.append(ref)
    old_lock = load_json(
        STUDIES_ROOT
        / "minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.source_lock.json"
    )
    for ref in old_lock["inputs"]:
        if ref not in refs:
            refs.append(ref)
    for ref in refs:
        verify_reference(ref, commit=spec["parent_commit"])
    return refs


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/score_circuit").glob("*.py"))
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def parameters() -> dict:
    parent = parent_result()
    original_lock = load_json(
        STUDIES_ROOT
        / "minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.artifact_lock.json"
    )
    values = {}
    for seed in specification()["seeds"]:
        identity = f"{seed}/score_only"
        value = parent["conditions"][identity]["parameters"]
        if value != original_lock["runs"][identity]["config"]["physical_parameters"]:
            raise RuntimeError(
                "parent scalar differs from original pre-evaluation lock"
            )
        values[str(seed)] = value
    return values


def write_lock() -> dict:
    commit = require_clean_pushed()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["liu_evaluated"]:
        raise RuntimeError("non-Liu qualification is required")
    refs = input_records() + [reference(PROTOCOL), reference(QUALIFICATION)]
    implementation = sources()
    for ref in refs + implementation:
        verify_reference(ref, commit=commit)
    lock = {
        "experiment_id": "score_circuit_v1",
        "protocol_sha256": PROTOCOL_HASH,
        "protocol_commit": git("rev-parse", PROTOCOL_COMMIT),
        "source_commit": commit,
        "inputs": refs,
        "sources": implementation,
        "parameters": parameters(),
    }
    write_json_exclusive(LOCK, lock)
    return reference(LOCK)


def validate_lock(*, pushed: bool) -> dict:
    commit = require_clean_pushed() if pushed else git("rev-parse", "HEAD")
    lock = load_json(LOCK)
    verify_reference(reference(LOCK), commit=commit)
    if lock["sources"] != sources() or lock["parameters"] != parameters():
        raise RuntimeError("locked source or parameters changed")
    if lock["inputs"] != input_records() + [
        reference(PROTOCOL),
        reference(QUALIFICATION),
    ]:
        raise RuntimeError("locked input inventory changed")
    if lock["protocol_sha256"] != PROTOCOL_HASH:
        raise RuntimeError("lock protocol differs")
    for ref in lock["inputs"] + lock["sources"]:
        verify_reference(ref, commit=lock["source_commit"])
    return lock


def load_arrays(seed: int) -> dict:
    row = parent_result()["conditions"][f"{seed}/score_only"]
    path = verify_reference(
        row["registered_raw_arrays"], commit=specification()["parent_commit"]
    )
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def batches(arrays: dict) -> dict[str, dict]:
    prefixes = {"liu": "liu__inputs__"}
    for key in arrays:
        if key.startswith("generic__groups__") and "__inputs__" in key:
            length = key.split("__")[2]
            prefixes[f"generic_{length}"] = f"generic__groups__{length}__inputs__"
    return {
        name: {
            k.removeprefix(prefix): value
            for k, value in arrays.items()
            if k.startswith(prefix)
        }
        for name, prefix in sorted(prefixes.items())
    }
