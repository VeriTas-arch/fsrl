"""Native authorities and byte-verified portable parent inputs."""

from functools import lru_cache

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.training_strategy.locks import (
    git_text,
    reference,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

PARENT = STUDIES_ROOT / "structural_identification/records"
RECORDS = STUDIES_ROOT / "encoding_contribution/records"
DESIGN = RECORDS / "benchmarks/encoding_contribution_v1.json"
RUN_ROOT = RUNS_ROOT / "encoding_contribution_v1"
LOCK = RUN_ROOT / "source_lock.json"
SEEDS = (2126, 2127, 2128)
STRUCTURES = ("M10", "M11")


def specification():
    return load_json(DESIGN)


@lru_cache
def parent_result():
    return load_json(PARENT / "results/structural_identification_v1.json")


def cell(seed, structure, donor):
    return f"{seed}-{structure}-theta_{donor}"


def parameters(seed, donor):
    return parent_result()["fits"][f"{seed}-{donor}-decay"]["parameters"]


@lru_cache
def parent_members():
    mapping = load_json(PARENT / "results/archive_map.json")
    result = {}
    for archive, rows in mapping.items():
        for index, row in enumerate(rows):
            key = row["path"].split("structural_identification_v1/", 1)[1]
            result[key] = archive, f"member{index:03}__"
    return result


@lru_cache(maxsize=4)
def archived_arrays(name):
    with np.load(PARENT / "results" / name, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def parent_arrays(path):
    archive, prefix = parent_members()[path]
    return {
        key.removeprefix(prefix): value
        for key, value in archived_arrays(archive).items()
        if key.startswith(prefix)
    }


def input_batch(name):
    arrays = parent_arrays(f"inputs/{name}.npz")
    return ModelBatch(
        {
            key.removeprefix("input__"): value
            for key, value in arrays.items()
            if key.startswith("input__")
        }
    ), arrays["uniforms"]


def lock_source():
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("lock requires a clean committed tree")
    qualification = load_json(RUN_ROOT / "qualification.json")
    if not qualification["passed"]:
        raise RuntimeError("synthetic qualification failed")
    ancestor = load_json(PARENT / "benchmarks/source_lock.json")
    for row in ancestor["files"]:
        verify_reference(row)
    parent_paths = [PARENT.parent / "study.toml", *PARENT.rglob("*")]
    parent_refs = [reference(p) for p in parent_paths if p.is_file()]
    for row in parent_refs:
        verify_reference(row, commit=specification()["parent_commit"])
    paths = set((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.update((REPO_ROOT / "tests/experiments/encoding_contribution").glob("*.py"))
    paths.update(REPO_ROOT / row["path"] for row in ancestor["files"])
    paths.add(DESIGN)
    commit = git_text("rev-parse", "HEAD")
    sources = [reference(p) for p in sorted(paths)]
    for row in sources:
        verify_reference(row, commit=commit)
    result = {
        "source_commit": commit,
        "sources": sources,
        "parent_inputs": parent_refs,
        "qualification": reference(RUN_ROOT / "qualification.json"),
        "protocol": specification(),
    }
    write_json_exclusive(LOCK, result)
    return {"source_commit": commit, "parent_inputs": len(parent_refs)}


def validate_source():
    lock = load_json(LOCK)
    for row in lock["sources"] + lock["parent_inputs"] + [lock["qualification"]]:
        verify_reference(row)
    return lock


def register(status, finding=None):
    manifest = RECORDS.parent / "study.toml"
    header = manifest.read_text().split("[[records]]", 1)[0]
    lines = []
    for line in header.splitlines():
        if line.startswith("status ="):
            line = f'status = "{status}"'
        if finding is not None and line.startswith("finding ="):
            line = f'finding = "{finding}"'
        lines.append(line)
    blocks = []
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        row = reference(path)
        role = "report" if path.suffix == ".md" else "supporting_artifact"
        if path.name == "encoding_contribution_v1.json":
            role = "registered_contract" if path == DESIGN else "frozen_result"
        blocks.append(
            f'[[records]]\npath = "{path.relative_to(RECORDS.parent)}"\n'
            f'legacy_path = "{row["path"]}"\norigin = "native"\nrole = "{role}"\n'
            f'sha256 = "{row["sha256"]}"\nbytes = {row["bytes"]}\n'
            f'source_ref = "sha256:{row["sha256"]}"\n'
        )
    manifest.write_text("\n".join(lines).rstrip() + "\n\n" + "\n".join(blocks))
