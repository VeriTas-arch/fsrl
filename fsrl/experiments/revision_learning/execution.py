"""Separate prospective successor; no changes to the frozen diagnosis."""

import hashlib
import itertools
import json
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

from fsrl.experiments.relational_revision.execution import panel_name
from fsrl.experiments.relational_revision.inputs import make_panel
from fsrl.experiments.relational_revision.statistics import (
    decision,
    endpoints,
    interval,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.formal_runtime import formal_runtime_snapshot
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

from .inputs import fingerprint, generic, training_batch
from .model import forward, make_model, trajectory

RECORDS = STUDIES_ROOT / "revision_learning/records"
RUNS = RUNS_ROOT / "revision_learning_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
SOURCE = RECORDS / "benchmarks/source_lock.json"
MODELS = RECORDS / "benchmarks/model_lock.json"


def register(
    status="unresolved", finding="Prospective fixed-budget joint learning comparison."
):
    spec = load_json(PROTOCOL)
    values = {
        "schema_version": 1,
        "id": "revision_learning",
        "title": "Joint learning of relational revision",
        "chapter": "structural_transport",
        "order": 1160,
        "status": status,
        "review_state": "indexed",
        "question": spec["question"],
        "finding": finding,
        "boundary": spec["boundary"],
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


def clean_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError(
            "commit qualified source or all trained artifacts before lock"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def freeze():
    commit = clean_commit()
    spec = load_json(PROTOCOL)
    if not load_json(RECORDS / "results/qualification.json")["passed"]:
        raise RuntimeError("qualification missing")
    directory = RECORDS / "artifacts/inputs"
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for cohort, history, topology, condition in itertools.product(
        spec["evaluation"]["cohorts"],
        (0, 1),
        ("chain", "star"),
        ("stable", "outlier", "revision"),
    ):
        name = panel_name(cohort, history, topology, condition)
        panel = make_panel(
            cohort, spec["evaluation"]["count"], history, topology, condition
        )
        path = directory / (name + ".npz")
        np.savez_compressed(path, allow_pickle=False, **panel)
        inputs[name] = reference(path)
    for seed in spec["evaluation"]["generic_seeds"]:
        panel = generic(
            np.random.default_rng(seed),
            128,
            spec["ranking_recipe"],
            seed + 100000000,
            True,
        )
        path = directory / f"generic-{seed}.npz"
        np.savez_compressed(path, allow_pickle=False, **panel)
        inputs[f"generic-{seed}"] = reference(path)
    sources = [reference(p) for p in sorted((REPO_ROOT / "fsrl").rglob("*.py"))]
    sources += [
        reference(PROTOCOL),
        reference(REPO_ROOT / "pyproject.toml"),
        reference(REPO_ROOT / "tests/experiments/test_revision_learning.py"),
    ]
    write_json_exclusive(
        SOURCE,
        {
            "source_commit": commit,
            "sources": sources,
            "inputs": inputs,
            "runtime": formal_runtime_snapshot(),
        },
    )
    register()
    return {"source_commit": commit, "inputs": len(inputs)}


def verified_source():
    source = load_json(SOURCE)
    for ref in source["sources"]:
        verify_reference(ref, commit=source["source_commit"])
    for ref in source["inputs"].values():
        verify_reference(ref)
    return source


def loss_for(net, panel, penalty):
    device = next(net.parameters()).device
    support, query, signs = [
        torch.as_tensor(panel[k], device=device) for k in ("support", "query", "signs")
    ]
    margins, state = forward(net, support, query, panel["prefixes"])
    ce = F.softplus(-margins * signs).mean()
    loss = ce + penalty * state.square().mean()
    return loss, ce


def train():
    source, spec = verified_source(), load_json(PROTOCOL)
    opt = spec["training"]
    for seed in opt["seeds"]:
        for kind in opt["models"]:
            directory = RUNS / "training" / f"{seed}-{kind}"
            marker = directory / "result.json"
            if marker.exists():
                for ref in load_json(marker)["artifacts"].values():
                    verify_reference(ref)
                continue
            directory.mkdir(parents=True, exist_ok=False)
            net = make_model(kind, seed)
            initial = tensor_hashes(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=opt["learning_rate"])
            digest = hashlib.sha256()
            started = time.perf_counter()
            with (directory / "train.jsonl").open("x") as log:
                for step in range(opt["steps"]):
                    panel = training_batch(seed, step, spec)
                    fp = fingerprint(panel)
                    digest.update(bytes.fromhex(fp))
                    optimizer.zero_grad(set_to_none=True)
                    loss, ce = loss_for(
                        net, panel, opt["plastic_penalty"] if kind == "plastic" else 0.0
                    )
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), 2.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    row = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "ce": float(ce.detach()),
                        "gradient": float(gradient),
                        "batch": fp,
                        "stream": digest.hexdigest(),
                    }
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if step % 100 == 0 or step == opt["steps"] - 1:
                        log.flush()
                        print(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    model=kind,
                                    elapsed=time.perf_counter() - started,
                                    **row,
                                )
                            ),
                            flush=True,
                        )
            with (directory / "net.pth").open("xb") as handle:
                torch.save(net.state_dict(), handle)
            write_json_exclusive(
                marker,
                {
                    "seed": seed,
                    "model": kind,
                    "source": reference(SOURCE),
                    "source_commit": source["source_commit"],
                    "initial": initial,
                    "final": tensor_hashes(net),
                    "parameters": sum(p.numel() for p in net.parameters()),
                    "stream": digest.hexdigest(),
                    "steps": opt["steps"],
                    "seconds": time.perf_counter() - started,
                    "artifacts": {
                        name: reference(directory / name)
                        for name in ("net.pth", "train.jsonl")
                    },
                },
            )
            shutil.copytree(directory, RECORDS / "artifacts/training" / directory.name)
            del net, optimizer, loss, ce
            torch.cuda.empty_cache()
    register()
    return {"trained": len(opt["seeds"]) * len(opt["models"]), "evaluated": False}


def lock_models():
    commit = clean_commit()
    verified_source()
    spec = load_json(PROTOCOL)
    models = {}
    for seed in spec["training"]["seeds"]:
        streams = []
        for kind in spec["training"]["models"]:
            identity = f"{seed}-{kind}"
            directory = RECORDS / "artifacts/training" / identity
            metadata = load_json(directory / "result.json")
            streams.append(metadata["stream"])
            models[identity] = {
                "seed": seed,
                "kind": kind,
                "weights": reference(directory / "net.pth"),
                "result": reference(directory / "result.json"),
            }
        if len(set(streams)) != 1:
            raise RuntimeError("training streams differ between learners")
    write_json_exclusive(MODELS, {"source_commit": commit, "models": models})
    register()
    return {"locked_models": len(models)}


def evaluate():
    source, lock = verified_source(), load_json(MODELS)
    for identity, record in lock["models"].items():
        weights = verify_reference(record["weights"], commit=lock["source_commit"])
        net = make_model(record["kind"], record["seed"])
        net.load_state_dict(torch.load(weights, weights_only=True, map_location="cuda"))
        net.requires_grad_(False).eval()
        before = tensor_hashes(net)
        directory = RECORDS / "artifacts/evaluation" / identity
        directory.mkdir(parents=True, exist_ok=True)
        for name, ref in source["inputs"].items():
            path, marker = directory / (name + ".npz"), directory / (name + ".json")
            if marker.exists():
                verify_reference(load_json(marker)["raw"])
                continue
            if path.exists():
                raise RuntimeError("partial unit requires inspection")
            with np.load(verify_reference(ref), allow_pickle=False) as data:
                panel = dict(data)
            with torch.inference_mode():
                if name.startswith("generic"):
                    chunks = []
                    for start in range(0, 128, 16):
                        margins, _ = forward(
                            net,
                            torch.as_tensor(
                                panel["support"][:, :, start : start + 16],
                                device="cuda",
                            ),
                            torch.as_tensor(
                                panel["query"][:, :, start : start + 16], device="cuda"
                            ),
                            panel["prefixes"],
                        )
                        chunks.append(margins.cpu().numpy())
                    raw = {"margins": np.concatenate(chunks)}
                else:
                    raw = trajectory(net, panel)
            if before != tensor_hashes(net) or not all(
                np.isfinite(v).all() for v in raw.values()
            ):
                raise RuntimeError("evaluation integrity failure")
            np.savez_compressed(path, allow_pickle=False, **raw)
            write_json_exclusive(
                marker,
                {"raw": reference(path), "input": ref, "model": record["weights"]},
            )
            print(
                json.dumps({"model": identity, "unit": name, "complete": True}),
                flush=True,
            )
    return {"completed_models": len(lock["models"])}


def read_raw(identity, name):
    marker = load_json(RECORDS / "artifacts/evaluation" / identity / (name + ".json"))
    with np.load(verify_reference(marker["raw"]), allow_pickle=False) as data:
        return data["margins"]


def report():
    source, spec, lock = verified_source(), load_json(PROTOCOL), load_json(MODELS)
    count = spec["evaluation"]["count"]
    draws = np.random.default_rng(992001).integers(count, size=(2, 2000, count))
    results, qualified = {}, {}
    for identity in lock["models"]:
        result = {}
        for topology in ("chain", "star"):
            for history in (0, 1):
                rows = []
                for cohort in spec["evaluation"]["cohorts"]:
                    raw = {
                        c: read_raw(identity, panel_name(cohort, history, topology, c))
                        for c in ("stable", "revision", "outlier")
                    }
                    np.testing.assert_array_equal(
                        raw["revision"][:, :2], raw["outlier"][:, :2]
                    )
                    with np.load(
                        verify_reference(
                            source["inputs"][
                                panel_name(cohort, history, topology, "revision")
                            ]
                        ),
                        allow_pickle=False,
                    ) as data:
                        rows.append(
                            endpoints(
                                raw["stable"],
                                raw["revision"],
                                raw["outlier"],
                                dict(data),
                            )
                        )
                summary = {
                    k: interval(np.stack([r[k] for r in rows]), draws) for k in rows[0]
                }
                result[f"{topology}-H{history}"] = {
                    "endpoints": summary,
                    "gates": decision(summary, history),
                }
        generic_values = []
        for seed in spec["evaluation"]["generic_seeds"]:
            name = f"generic-{seed}"
            with np.load(
                verify_reference(source["inputs"][name]), allow_pickle=False
            ) as data:
                signs = data["signs"]
            margins = read_raw(identity, name)
            generic_values.append(expit(margins * signs).mean((1, 2)))
        result["generic_probability"] = interval(np.stack(generic_values), draws)
        results[identity] = result
        qualified[identity] = (
            all(all(result[f"chain-H{h}"]["gates"].values()) for h in (0, 1))
            and result["generic_probability"]["ci95"][0] > 0.5
        )
    eligible = all(qualified[f"{s}-plastic"] for s in spec["training"]["seeds"])
    output = {
        "protocol": reference(PROTOCOL),
        "source": reference(SOURCE),
        "models": reference(MODELS),
        "results": results,
        "qualified": qualified,
        "intervention_eligible": eligible,
        "continuation": "freeze directional intervention and external topology prediction"
        if eligible
        else "close fixed-budget joint candidate; no qualified directional mechanism claim",
    }
    write_json_exclusive(RECORDS / "results/result.json", output)
    register("mixed", output["continuation"])
    return {
        k: output[k] for k in ("qualified", "intervention_eligible", "continuation")
    }
