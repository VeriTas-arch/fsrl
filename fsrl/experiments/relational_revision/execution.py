"""Prospective inputs, frozen weights and write-once development outputs."""

import itertools
import json
import shutil
import subprocess

import numpy as np
import torch

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.formal_runtime import formal_runtime_snapshot
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

from . import baselines, model
from .inputs import make_panel
from .statistics import decision, endpoints, interval

RECORDS = STUDIES_ROOT / "relational_revision/records"
RUNS = RUNS_ROOT / "relational_revision_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
LOCK = RECORDS / "benchmarks/source_lock.json"


def conditions(spec):
    design = spec["design"]
    return itertools.product(
        design["cohort_seeds"],
        design["histories"],
        design["topologies"],
        design["conditions"],
    )


def panel_name(cohort, history, topology, condition):
    return f"{cohort}-H{history}-{topology}-{condition}"


def register(
    status="unresolved", finding="Prospective history-dependent revision diagnosis."
):
    spec = load_json(PROTOCOL)
    values = {
        "schema_version": 1,
        "id": "relational_revision",
        "title": "History-dependent relational knowledge revision",
        "chapter": "structural_transport",
        "order": 1150,
        "status": status,
        "review_state": "indexed",
        "question": spec["question"],
        "finding": finding,
        "boundary": spec["claim_boundary"],
    }
    lines = [f"{k} = {json.dumps(v)}" for k, v in values.items()]
    for path in sorted(RECORDS.rglob("*")):
        if path.is_file():
            ref = reference(path)
            row = {
                "path": str(path.relative_to(RECORDS.parent)),
                "legacy_path": ref["path"],
                "origin": "native",
                "role": "registered_contract"
                if path == PROTOCOL
                else record_role(path),
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
                "source_ref": "sha256:" + ref["sha256"],
            }
            lines += [
                "",
                "[[records]]",
                *[f"{k} = {json.dumps(v)}" for k, v in row.items()],
            ]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")


def freeze():
    spec = load_json(PROTOCOL)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError(
            "commit source/protocol and qualification before freezing inputs"
        )
    if not load_json(RECORDS / "results/qualification.json")["passed"]:
        raise RuntimeError("numerical qualification required")
    inputs = {}
    directory = RUNS / "inputs"
    directory.mkdir(parents=True, exist_ok=False)
    for args in conditions(spec):
        name = panel_name(*args)
        panel = make_panel(args[0], spec["design"]["episodes_per_cohort"], *args[1:])
        path = directory / (name + ".npz")
        np.savez_compressed(path, allow_pickle=False, **panel)
        inputs[name] = reference(path)
    parent = STUDIES_ROOT / "linear_modulation/records/artifacts/training"
    models = {
        f"{seed}-{arm}": reference(parent / str(seed) / arm / "net.pth")
        for seed, arm in itertools.product(
            spec["design"]["model_seeds"], spec["design"]["training_arms"]
        )
    }
    sources = [reference(p) for p in sorted((REPO_ROOT / "fsrl").rglob("*.py"))]
    sources += [
        reference(PROTOCOL),
        reference(REPO_ROOT / "pyproject.toml"),
        reference(REPO_ROOT / "tests/experiments/test_relational_revision.py"),
    ]
    write_json_exclusive(
        LOCK,
        {
            "source_commit": commit,
            "sources": sources,
            "inputs": inputs,
            "models": models,
            "runtime": formal_runtime_snapshot(),
        },
    )
    register()
    return {"inputs": len(inputs), "models": len(models), "source_commit": commit}


def verified_lock():
    lock = load_json(LOCK)
    for ref in lock["sources"]:
        verify_reference(ref, commit=lock["source_commit"])
    for ref in lock["inputs"].values():
        verify_reference(ref)
    for ref in lock["models"].values():
        verify_reference(ref, commit=lock["source_commit"])
    return lock


def evaluate_learner(identity, net, lock):
    directory = RUNS / "evaluation" / identity
    directory.mkdir(parents=True, exist_ok=True)
    before = None if net is None else tensor_hashes(net)
    for name, ref in lock["inputs"].items():
        path = directory / (name + ".npz")
        marker = directory / (name + ".json")
        if marker.exists():
            verify_reference(load_json(marker)["raw"])
            continue
        if path.exists():
            raise RuntimeError(f"uncommitted partial unit requires inspection: {path}")
        with np.load(verify_reference(ref), allow_pickle=False) as data:
            panel = dict(data)
        with torch.inference_mode():
            raw = (
                baselines.trajectory(panel, identity)
                if net is None
                else model.trajectory(net, panel)
            )
        if not all(np.isfinite(array).all() for array in raw.values()):
            raise RuntimeError("nonfinite rollout")
        if net is not None and before != tensor_hashes(net):
            raise RuntimeError("slow parameters changed")
        np.savez_compressed(path, allow_pickle=False, **raw)
        write_json_exclusive(
            marker,
            {
                "input": ref,
                "raw": reference(path),
                "model": lock["models"].get(identity),
                "passed_integrity": True,
            },
        )
        print(
            json.dumps({"learner": identity, "unit": name, "complete": True}),
            flush=True,
        )


def evaluate():
    lock = verified_lock()
    for identity, ref in lock["models"].items():
        net = model.network(verify_reference(ref)).requires_grad_(False).eval()
        evaluate_learner(identity, net, lock)
        del net
        torch.cuda.empty_cache()
    for rule in load_json(PROTOCOL)["design"]["baseline_rules"]:
        evaluate_learner(rule, None, lock)
    return {"network_units": 144, "mathematical_control_units": 48}


def load_margins(identity, cohort, history, topology, condition):
    name = panel_name(cohort, history, topology, condition)
    directory = RUNS / "evaluation" / identity
    marker = load_json(directory / (name + ".json"))
    with np.load(verify_reference(marker["raw"]), allow_pickle=False) as data:
        return data["margins"]


def summarize_learner(identity, spec, lock, draws):
    result = {}
    for topology in spec["design"]["topologies"]:
        paired = []
        for history in spec["design"]["histories"]:
            rows = []
            for cohort in spec["design"]["cohort_seeds"]:
                raw = {
                    c: load_margins(identity, cohort, history, topology, c)
                    for c in spec["design"]["conditions"]
                }
                np.testing.assert_array_equal(
                    raw["revision"][:, :2], raw["outlier"][:, :2]
                )
                with np.load(
                    verify_reference(
                        lock["inputs"][
                            panel_name(cohort, history, topology, "revision")
                        ]
                    ),
                    allow_pickle=False,
                ) as data:
                    rows.append(
                        endpoints(
                            raw["stable"], raw["revision"], raw["outlier"], dict(data)
                        )
                    )
            joined = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
            summary = {key: interval(value, draws) for key, value in joined.items()}
            result[f"{topology}-H{history}"] = {
                "endpoints": summary,
                "gates": decision(summary, history),
            }
            paired.append(joined)
        result[f"{topology}-interaction"] = interval(
            paired[0]["AD_change"] - paired[1]["AD_change"], draws
        )
    return result


def report():
    spec, lock = load_json(PROTOCOL), verified_lock()
    rng = np.random.default_rng(spec["statistics"]["bootstrap_seed"])
    count = spec["design"]["episodes_per_cohort"]
    draws = rng.integers(
        count, size=(2, spec["statistics"]["bootstrap_samples"], count)
    )
    identities = [*lock["models"], *spec["design"]["baseline_rules"]]
    results = {key: summarize_learner(key, spec, lock, draws) for key in identities}
    qualified = {
        str(seed): all(
            all(results[f"{seed}-noisy"][f"chain-H{h}"]["gates"].values())
            for h in (0, 1)
        )
        for seed in spec["design"]["model_seeds"]
    }
    continuation = (
        "retain frozen candidate; train persistent-hidden comparator and qualify causal intervention"
        if all(qualified.values())
        else "freeze joint-training and persistent-hidden comparator successor; do not alter phase1"
    )
    output = {
        "protocol": reference(PROTOCOL),
        "lock": reference(LOCK),
        "results": results,
        "qualified_noisy_seeds": qualified,
        "continuation": continuation,
    }
    archive = RECORDS / "artifacts"
    archive.mkdir(exist_ok=False)
    shutil.copytree(RUNS / "inputs", archive / "inputs")
    for identity in identities:
        shutil.copytree(RUNS / "evaluation" / identity, archive / identity)
    output["archived_inputs"] = {
        name: reference(archive / "inputs" / (name + ".npz")) for name in lock["inputs"]
    }
    output["archived_evaluation"] = {
        str(path.relative_to(archive)): reference(path)
        for path in sorted(archive.rglob("*.npz"))
        if path.parent.name != "inputs"
    }
    write_json_exclusive(RECORDS / "results/result.json", output)
    register(
        "mixed",
        f"Frozen synthetic revision diagnosis complete; candidate qualification {qualified}. {continuation}.",
    )
    return {"qualified_noisy_seeds": qualified, "continuation": continuation}
