"""Source/input freeze, unchanged training, and the six-model evaluation barrier."""

import gc
import hashlib
import json

import numpy as np
import torch

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.memory_structure.inputs import liu_inputs
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.observation_uncertainty.inputs import attach_noise
from fsrl.experiments.observation_uncertainty.training import train_steps
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import (
    reference,
    scientific_inputs,
    verify_reference,
)
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import with_learned
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT

from .protocol import (
    MODELS,
    PROTOCOL,
    PROTOCOL_SHA256,
    RECORDS,
    RUNS,
    SOURCE,
    recipe,
    specification,
)


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list(
        (REPO_ROOT / "tests/experiments/observation_replication").glob("*.py")
    )
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(p) for p in sorted(paths)]


def save_input(panel, name, cpu):
    path = RUNS / "inputs" / str(panel) / (name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, cpu.arrays)
    return {"file": reference(path), "fingerprint": cpu.fingerprint()}


def freeze_inputs():
    panels = {}
    for panel in specification()["design"]["panels"]:
        spec, inputs = recipe(panel), {}
        episodes = validation_episodes(spec)
        for length, indices in validation_groups(episodes).items():
            cpu = with_learned(tuple(episodes[i] for i in indices))
            cpu.arrays["episode_indices"] = np.asarray(indices)
            inputs[f"test-{length}"] = save_input(
                panel, f"test-{length}", attach_noise(cpu, 100 + panel, length)
            )
        _, cpu = liu_inputs(spec, 8)
        inputs["liu-8"] = save_input(panel, "liu-8", attach_noise(cpu, 200 + panel))
        panels[str(panel)] = {"inputs": inputs, "recipe": spec}
    return panels


def freeze():
    commit = require_clean()
    qualification = RECORDS / "benchmarks/qualification.json"
    q = load_json(qualification)
    if not q["passed"] or q["sources"] != sources():
        raise RuntimeError("qualification does not cover current implementation")
    files = (
        sources()
        + scientific_inputs()
        + [reference(PROTOCOL), specification()["parent_protocol"]]
    )
    files += [
        reference(REPO_ROOT / row["path"]) for row in recipe()["references"].values()
    ]
    for row in files:
        verify_reference(row, commit=commit)
    result = {
        "source_commit": commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": files,
        "qualification": reference(qualification),
        "runtime": q["runtime"],
        "panels": freeze_inputs(),
    }
    write_json_exclusive(SOURCE, result)
    return {"panels": len(result["panels"]), "source_commit": commit}


def validate_source():
    verify_reference(reference(SOURCE), commit=require_clean())
    lock = load_json(SOURCE)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("protocol mismatch")
    for ref in lock["sources"]:
        verify_reference(ref, commit=lock["source_commit"])
    verify_reference(lock["qualification"])
    for panel in lock["panels"].values():
        for row in panel["inputs"].values():
            verify_reference(row["file"])
    if json_ready(configure_execution()) != lock["runtime"]:
        raise RuntimeError("runtime mismatch")
    return lock


def train_one(seed, arm, lock):
    directory = RUNS / "training" / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = recipe()
    backbone, local = make_model(spec, seed, "dual", "cuda")
    assert local is not None
    initial = {
        "initial_backbone": tensor_hashes(backbone),
        "initial_local": tensor_hashes(local),
    }
    identity = {
        "seed": seed,
        "arm": arm,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE),
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="observation_replication_v1",
        execution_id=f"train-{seed}-{arm}",
        producer=identity,
        resolved_config={"recipe": spec, "runtime": lock["runtime"]},
    ):
        stats = train_steps(
            backbone,
            local,
            sequences(backbone, None, compiled=True),
            spec,
            seed,
            arm,
            None,
            0.0,
            directory,
        )
        for name, module in (("net", backbone), ("local", local)):
            with (directory / (name + ".pth")).open("xb") as handle:
                torch.save(module.state_dict(), handle)
        result = {
            **identity,
            **initial,
            **stats,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "runtime": lock["runtime"],
        }
        write_json_exclusive(directory / "result.json", result)
    return result


def train():
    lock, spec = validate_source(), specification()
    for index, seed in enumerate(spec["design"]["seeds"]):
        for arm in spec["design"]["training_conditions"][
            :: 1 if index % 2 == 0 else -1
        ]:
            train_one(seed, arm, lock)
            gc.collect()
            torch.cuda.empty_cache()
    return {"trained_models": 6, "new_evaluation_exposed": False}


def training_record(seed, arm):
    directory = RUNS / "training" / str(seed) / arm
    metadata = completed(directory)
    if (metadata["seed"], metadata["arm"], metadata["protocol_sha256"]) != (
        seed,
        arm,
        PROTOCOL_SHA256,
    ):
        raise RuntimeError("training identity mismatch")
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    steps = recipe()["optimization"]["total_steps"]
    if [r["step"] for r in logs] != list(range(steps)):
        raise RuntimeError("incomplete training trajectory")
    digest = hashlib.sha256()
    for row in logs:
        digest.update(bytes.fromhex(row["batch_fingerprint"]))
        if (
            row["stream_fingerprint"] != digest.hexdigest()
            or row["observation_seed"] != 400000000 + seed * 10000 + row["step"]
        ):
            raise RuntimeError("training RNG chain mismatch")
    if digest.hexdigest() != metadata["stream_fingerprint"]:
        raise RuntimeError("final training digest mismatch")
    for key in ("backbone.i2h.weight", "backbone.h2DA.weight", "local.raw_gain"):
        if metadata["optimizer_steps"][key] != steps:
            raise RuntimeError("joint optimization incomplete")
    return {
        "metadata": metadata,
        "files": {
            p.name: reference(p) for p in sorted(directory.iterdir()) if p.is_file()
        },
    }


def lock_models():
    validate_source()
    runs = {}
    for seed in specification()["design"]["seeds"]:
        pair = {arm: training_record(seed, arm) for arm in ("clean", "noisy")}
        for key in ("initial_backbone", "initial_local", "stream_fingerprint"):
            if pair["clean"]["metadata"][key] != pair["noisy"]["metadata"][key]:
                raise RuntimeError("paired initialization or task stream mismatch")
        runs.update({f"{seed}/{arm}": row for arm, row in pair.items()})
    write_json_exclusive(
        MODELS,
        {
            "source_lock": reference(SOURCE),
            "protocol_sha256": PROTOCOL_SHA256,
            "runs": runs,
            "status": "all_six_locked_before_any_new_evaluation",
        },
    )
    return {"models": len(runs), "new_evaluation_exposed": False}


def validate_models():
    source = validate_source()
    verify_reference(reference(MODELS), commit=require_clean())
    models = load_json(MODELS)
    if (
        models["source_lock"] != reference(SOURCE)
        or models["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise RuntimeError("model source mismatch")
    expected = {
        f"{s}/{a}"
        for s in specification()["design"]["seeds"]
        for a in ("clean", "noisy")
    }
    if set(models["runs"]) != expected:
        raise RuntimeError("incomplete six-model barrier")
    for row in models["runs"].values():
        for ref in row["files"].values():
            verify_reference(ref)
    return source, models
