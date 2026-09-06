"""Source-locked successor with portable, unchanged parent observations."""

from functools import lru_cache

from fsrl.experiments.encoding_contribution.protocol import (
    SEEDS,
    STRUCTURES,
    cell,
    input_batch,
)
from fsrl.experiments.training_strategy.locks import (
    git_text,
    reference,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

PARENT = STUDIES_ROOT / "encoding_contribution/records"
RECORDS = STUDIES_ROOT / "encoding_distribution/records"
DESIGN = RECORDS / "benchmarks/encoding_distribution_v1.json"
RUN_ROOT = RUNS_ROOT / "encoding_distribution_v1"
LOCK = RUN_ROOT / "source_lock.json"


def specification():
    return load_json(DESIGN)


@lru_cache
def parent_result():
    return load_json(PARENT / "results/encoding_contribution_v1.json")


@lru_cache
def parent_cohorts():
    paths = sorted((PARENT / "results").glob("cross-rows-*.json"))
    rows = [row for path in paths for row in load_json(path)]
    expected = {
        cell(seed, st, donor)
        for seed in SEEDS
        for st in STRUCTURES
        for donor in STRUCTURES
    }
    if len(rows) != 100 or any(set(row) != expected for row in rows):
        raise RuntimeError("parent E/Q observations incomplete")
    return rows


def lock_source():
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source lock requires clean committed source")
    qualification = RUN_ROOT / "qualification.json"
    if not load_json(qualification)["passed"]:
        raise RuntimeError("qualification failed")
    ancestor = load_json(PARENT / "benchmarks/source_lock.json")
    for row in ancestor["sources"] + ancestor["parent_inputs"]:
        verify_reference(row)
    parent_refs = [reference(p) for p in sorted(PARENT.rglob("*")) if p.is_file()]
    parent_refs.append(reference(PARENT.parent / "study.toml"))
    parent_refs += ancestor["parent_inputs"]
    for row in parent_refs:
        verify_reference(row, commit=specification()["parent_commit"])
    paths = set((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.update((REPO_ROOT / "tests/experiments/encoding_distribution").glob("*.py"))
    paths.update(REPO_ROOT / row["path"] for row in ancestor["sources"])
    paths.add(DESIGN)
    commit = git_text("rev-parse", "HEAD")
    sources = [reference(p) for p in sorted(paths)]
    for row in sources:
        verify_reference(row, commit=commit)
    # The inverse-normal transform is finite only for open-interval uniforms.
    import numpy as np

    names = [f"liu-{i:03}" for i in range(100)] + [f"generic-{i}" for i in range(8)]
    names += ["manipulation-balanced_K4", "manipulation-balanced_K8"]
    for name in names:
        _, u = input_batch(name)
        if not np.all((u > 0) & (u < 1)):
            raise RuntimeError("parent uniforms outside open interval")
    write_json_exclusive(
        LOCK,
        {
            "source_commit": commit,
            "sources": sources,
            "parent_inputs": parent_refs,
            "qualification": reference(qualification),
            "protocol": specification(),
        },
    )
    return {
        "source_commit": commit,
        "parent_inputs": len(parent_refs),
        "uniform_inputs_verified": len(names),
    }


def validate_source():
    lock = load_json(LOCK)
    for row in lock["sources"] + lock["parent_inputs"] + [lock["qualification"]]:
        verify_reference(row)
    return lock


def register(status="frozen_contract", finding=None):
    manifest = RECORDS.parent / "study.toml"
    lines = []
    for line in manifest.read_text().split("[[records]]", 1)[0].splitlines():
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
        if path.name == "encoding_distribution_v1.json":
            role = "registered_contract" if path == DESIGN else "frozen_result"
        blocks.append(
            f'[[records]]\npath = "{path.relative_to(RECORDS.parent)}"\nlegacy_path = "{row["path"]}"\norigin = "native"\nrole = "{role}"\nsha256 = "{row["sha256"]}"\nbytes = {row["bytes"]}\nsource_ref = "sha256:{row["sha256"]}"\n'
        )
    manifest.write_text("\n".join(lines).rstrip() + "\n\n" + "\n".join(blocks))
