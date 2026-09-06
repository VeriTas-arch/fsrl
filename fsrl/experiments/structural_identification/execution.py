"""Qualified source locks and final-only matched structural training."""

import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.adaptive_plasticity.data import model_tensors
from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.locks import (
    git_text,
    verify_reference,
)
from fsrl.experiments.training_strategy.locks import (
    reference as file_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .model import encode, make_model, reference
from .protocol import (
    DESIGN,
    DESIGN_COMMIT,
    DESIGN_HASH,
    RUN_ROOT,
    SCHEDULES,
    STRUCTURES,
    fit_order,
    fit_path,
    model_specification,
    specification,
)

SOURCE_LOCK = RUN_ROOT / "source_lock.json"
FIT_LOCK = RUN_ROOT / "fit_lock.json"


def qualify() -> dict:
    snapshot = runtime()
    rng = np.random.default_rng(1370001)
    batch = generic_batch(sample_episodes(task_generator(), rng, 4))
    uniforms = rng.random(batch.arrays["signed"].shape)
    results = {}
    for structure in STRUCTURES:
        encoded = encode(batch, structure, uniforms)
        for schedule in SCHEDULES:
            model = make_model(schedule, "cuda")
            tensors = model_tensors(encoded, "cuda")
            expected, _ = reference(encoded, 0.5, 1, schedule)
            eager = model(*tensors)[0]
            np.testing.assert_allclose(
                eager.detach().cpu().numpy(), expected, atol=1e-5, rtol=1e-4
            )
            runner = compiled(model)
            output = runner(*tensors)[0]
            torch.testing.assert_close(output, eager, atol=1e-5, rtol=1e-4)
            signs = torch.as_tensor(2 * encoded.arrays["targets"] - 1, device="cuda")
            gradients = []
            updated = []
            original = {
                name: p.detach().clone() for name, p in model.named_parameters()
            }
            for forward in (model, runner):
                model.load_state_dict(original)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                optimizer.zero_grad()
                F.softplus(-signs * forward(*tensors)[0]).mean().backward()
                current_gradients = []
                for p in model.parameters():
                    assert p.grad is not None
                    current_gradients.append(p.grad.detach().clone())
                gradients.append(current_gradients)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2, error_if_nonfinite=True
                )
                optimizer.step()
                updated.append([p.detach().clone() for p in model.parameters()])
            for a, b in zip(
                gradients[0] + updated[0], gradients[1] + updated[1], strict=True
            ):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
            if not all(
                torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients[0]
            ):
                raise RuntimeError("inactive or nonfinite slow-parameter gradient")
            results[f"{structure}-{schedule}"] = True
            del model, runner
            gc.collect()
    result = {
        "passed": all(results.values()),
        "checks": results,
        "runtime": snapshot,
        "liu_evaluated": False,
    }
    write_json_exclusive(RUN_ROOT / "qualification.json", result)
    return result


def lock_source() -> dict:
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source lock requires a clean committed tree")
    qualification = load_json(RUN_ROOT / "qualification.json")
    if not qualification["passed"] or qualification["liu_evaluated"]:
        raise RuntimeError("source lock requires non-Liu CUDA qualification")
    commit = git_text("rev-parse", "HEAD")
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list(
        (REPO_ROOT / "tests/experiments/structural_identification").glob("*.py")
    )
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc", DESIGN]
    paths.append(protocol_path("liu_v2"))
    paths += [
        REPO_ROOT / path
        for key, path in specification()["inherit"].items()
        if key != "meaning"
    ]
    paths += list((REPO_ROOT / "data/external/liu2026").glob("*"))
    records = [file_reference(path) for path in sorted(paths) if path.is_file()]
    for row in records:
        verify_reference(row, commit=commit)
    verify_reference(file_reference(DESIGN), commit=DESIGN_COMMIT)
    result = {
        "source_commit": commit,
        "protocol_sha256": DESIGN_HASH,
        "files": records,
        "qualification": file_reference(RUN_ROOT / "qualification.json"),
        "source_inventory": file_reference(
            RUN_ROOT / "source_audit/osf_inventory.json"
        ),
        "resolved_model": model_specification(),
        "design": specification(),
    }
    write_json_exclusive(SOURCE_LOCK, result)
    return {"source_commit": commit, "files": len(records)}


def validate_source() -> dict:
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != DESIGN_HASH:
        raise RuntimeError("source lock protocol differs")
    for row in lock["files"] + [lock["qualification"], lock["source_inventory"]]:
        verify_reference(row)
    return lock


def train_fit(seed, structure, schedule, source) -> dict:
    path = fit_path(seed, structure, schedule)
    if path.exists():
        validate_complete(path)
        result = load_json(path / "parameters.json")
        if result["optimizer_steps"] != {"raw_eta": 1500, "raw_global_gain": 1500}:
            raise RuntimeError("fit has wrong optimizer counters")
        return result
    model = make_model(schedule, "cuda")
    runner = compiled(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    settings = specification()["design"]["rng"]
    task_rng = np.random.default_rng(settings["training_offset"] + seed)
    code_rng = np.random.default_rng(settings["encoding_offset"] + seed)
    generator = task_generator()
    hashes = {key: hashlib.sha256() for key in ("physical", "uniform", "encoded")}
    shapes = set()
    start = time.perf_counter()
    warmup = 0.0
    with ProspectiveRun.start(
        path,
        workflow_id="structural_identification_v1",
        execution_id=f"train-{seed}-{structure}-{schedule}",
        producer={"module": __name__, "source_commit": source["source_commit"]},
        resolved_config={
            "design": specification(),
            "structure": structure,
            "schedule": schedule,
        },
    ):
        with (path / "training.jsonl").open("x") as handle:
            for step in range(1500):
                base = generic_batch(sample_episodes(generator, task_rng, 32))
                uniforms = code_rng.random(base.arrays["signed"].shape)
                batch = encode(base, structure, uniforms)
                for key, value in (
                    ("physical", bytes.fromhex(base.fingerprint())),
                    ("uniform", uniforms.tobytes()),
                    ("encoded", bytes.fromhex(batch.fingerprint())),
                ):
                    hashes[key].update(value)
                tensors = model_tensors(batch, "cuda")
                signs = torch.as_tensor(2 * batch.arrays["targets"] - 1, device="cuda")
                shape = batch.arrays["signed"].shape
                if shape not in shapes:
                    before = tensor_hashes(model)
                    stamp = time.perf_counter()
                    F.softplus(-signs * runner(*tensors)[0]).mean().backward()
                    torch.cuda.synchronize()
                    warmup += time.perf_counter() - stamp
                    if tensor_hashes(model) != before:
                        raise RuntimeError("compiler warmup changed parameters")
                    shapes.add(shape)
                optimizer.zero_grad(set_to_none=True)
                loss = F.softplus(-signs * runner(*tensors)[0]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2, error_if_nonfinite=True
                )
                optimizer.step()
                row = {
                    "step": step + 1,
                    "loss": loss.item(),
                    "eta0": model.eta.item(),
                    "gamma_G": model.global_gain.item(),
                }
                if not np.isfinite(list(row.values())).all() or not 0 < row["eta0"] < 1:
                    raise RuntimeError("nonfinite or boundary optimization")
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                if (step + 1) % 250 == 0:
                    handle.flush()
                    print(seed, structure, schedule, row, flush=True)
        result = {
            "seed": seed,
            "structure": structure,
            "schedule": schedule,
            "eta0": model.eta.item(),
            "gamma_G": model.global_gain.item(),
            "raw_parameters": {
                name: p.detach().cpu().tolist()
                for name, p in model.state_dict().items()
            },
            "streams": {key: h.hexdigest() for key, h in hashes.items()},
            "optimizer_steps": {
                name: int(optimizer.state[p]["step"].item())
                for name, p in model.named_parameters()
            },
            "source_commit": source["source_commit"],
            "protocol_sha256": DESIGN_HASH,
            "seconds": time.perf_counter() - start,
            "warmup_seconds": warmup,
            "score_entries": 15,
            "trainable_scalars": 2,
            "relation_efficacy_entries": 0,
        }
        write_json_exclusive(path / "parameters.json", result)
    del model, runner, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_all() -> dict:
    runtime()
    source = validate_source()
    records = [train_fit(*identity, source) for identity in fit_order()]
    for seed in specification()["design"]["seeds"]:
        for key in ("physical", "uniform"):
            if (
                len({row["streams"][key] for row in records if row["seed"] == seed})
                != 1
            ):
                raise RuntimeError("paired training streams differ")
    from .measurement import REFERENCE

    lock = {
        "source_lock": file_reference(SOURCE_LOCK),
        "human_reference": file_reference(REFERENCE),
        "fits": [
            file_reference(fit_path(*identity) / "parameters.json")
            for identity in fit_order()
        ],
    }
    if FIT_LOCK.exists():
        if load_json(FIT_LOCK) != lock:
            raise RuntimeError("existing fit lock differs")
    else:
        write_json_exclusive(FIT_LOCK, lock)
    return {"fits": len(records), "evaluation_performed": False}


def fitted_parameters() -> list[dict]:
    validate_source()
    lock = load_json(FIT_LOCK)
    verify_reference(lock["source_lock"])
    verify_reference(lock["human_reference"])
    if len(lock["fits"]) != 24:
        raise RuntimeError("all 24 fits must be locked before evaluation")
    return [load_json(verify_reference(row)) for row in lock["fits"]]
