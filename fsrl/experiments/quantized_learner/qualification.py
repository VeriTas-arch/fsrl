"""Non-Liu code invariants and actual-shape CUDA output/gradient/Adam parity."""

import copy
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.locks import reference, require_pushed_clean
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT

from .encoding import encode_batch
from .protocol import (
    PROTOCOL_HASH,
    RUN_ROOT,
    make_model,
    resolved_specification,
    specification,
)
from .reference import rollout

CPU_TESTS = (
    "tests.experiments.quantized_learner.test_encoding",
    "tests.experiments.quantized_learner.test_model",
    "tests.experiments.quantized_learner.test_recovery",
    "tests.experiments.quantized_learner.test_qualification",
)


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/quantized_learner").glob("*.py"))
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def comparison(first, second, *, atol, rtol) -> dict:
    a, b = (torch.as_tensor(value).detach().cpu() for value in (first, second))
    same_shape = a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "passed": same_shape and finite and torch.allclose(a, b, atol=atol, rtol=rtol),
        "max_abs_error": float((a - b).abs().max()) if a.numel() and same_shape else 0,
    }


def parity_checks(eager, model, runner, batch, spec, tolerance) -> dict:
    """Three scratch Adam steps; no fitted state or evaluation data is used."""
    device = str(next(model.parameters()).device)
    args = batch.tensors(device)
    signs = torch.as_tensor(2 * batch.arrays["targets"] - 1, device=device)
    settings = spec["optimization"]
    optimizers = [
        torch.optim.Adam(current.parameters(), lr=settings["learning_rate"])
        for current in (eager, model)
    ]
    checks = {}
    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        outputs = eager(*args), runner(*args)
        for index, (first, second) in enumerate(zip(*outputs, strict=True)):
            checks[f"step-{step}/output-{index}"] = comparison(
                first, second, **tolerance
            )
        losses = [F.softplus(-signs * output[0]).mean() for output in outputs]
        checks[f"step-{step}/loss"] = comparison(*losses, **tolerance)
        for loss in losses:
            loss.backward()
        before = {}
        for (name, p), q in zip(
            eager.named_parameters(), model.parameters(), strict=True
        ):
            assert p.grad is not None and q.grad is not None
            before[name] = (p.detach().clone(), q.detach().clone())
            checks[f"step-{step}/gradient-{name}"] = comparison(
                p.grad, q.grad, **tolerance
            )
            checks[f"step-{step}/nonzero-gradient-{name}"] = {
                "passed": bool(
                    p.grad.abs().max() > 1e-12 and q.grad.abs().max() > 1e-12
                )
            }
        for current, optimizer in zip((eager, model), optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(
                current.parameters(), settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
        for (name, p), q in zip(
            eager.named_parameters(), model.parameters(), strict=True
        ):
            checks[f"step-{step}/updated-{name}"] = comparison(p, q, **tolerance)
            checks[f"step-{step}/actual-update-{name}"] = {
                "passed": not torch.equal(p, before[name][0])
                and not torch.equal(q, before[name][1])
            }
            for key in ("step", "exp_avg", "exp_avg_sq"):
                checks[f"step-{step}/adam-{name}-{key}"] = comparison(
                    optimizers[0].state[p][key],
                    optimizers[1].state[q][key],
                    **tolerance,
                )
            checks[f"step-{step}/counter-{name}"] = {
                "passed": all(
                    opt.state[param]["step"].item() == step + 1
                    for opt, param in zip(optimizers, (p, q), strict=True)
                )
            }
    return checks


def reference_checks(batch, spec, tolerance) -> dict:
    model = make_model(spec).double()
    with torch.no_grad():
        observed = model(*batch.tensors("cpu", torch.float64))
    expected = rollout(
        batch.arrays,
        eta=model.eta.item(),
        gain=model.global_gain.item(),
        epsilon=model.epsilon,
    )
    return {
        name: comparison(observed[index], expected[name], **tolerance)
        for name, index in (("margins", 0), ("w", 3))
    }


def qualify(attempt: int) -> dict:
    if attempt < 1:
        raise ValueError("qualification attempt must be positive")
    commit = require_pushed_clean()
    candidate, spec = specification(), resolved_specification()
    execution = runtime()
    settings = candidate["integrity"]
    rng = np.random.default_rng(settings["qualification_seed"])
    generator = task_generator()
    expected_lengths = {
        edges * spec["task"]["support_blocks"]
        for edges in range(spec["task"]["min_edges"], spec["task"]["max_edges"] + 1)
    }
    fixtures = {}
    while set(fixtures) != expected_lengths:
        batch = generic_batch(
            sample_episodes(generator, rng, spec["optimization"]["batch_size"])
        )
        fixtures.setdefault(len(batch.arrays["signed"]), batch)
    directory = RUN_ROOT / "qualification" / f"attempt-{attempt}"
    checks, batches = {}, {}
    tolerance = settings["tolerances"]
    with ProspectiveRun.start(
        directory,
        workflow_id=candidate["experiment_id"],
        execution_id=f"qualification-{attempt}",
        producer={"module": __name__, "source_commit": commit},
        resolved_config={"runtime": execution, "integrity": settings},
    ):
        command = [
            sys.executable,
            "-m",
            "fsrl.infra.test_runtime",
            "--timeout",
            "120",
            "--framework",
            "unittest",
            "--",
            "-v",
            *CPU_TESTS,
        ]
        with (directory / "cpu-tests.log").open("x") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        checks["cpu-invariants"] = {
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
        }
        for length, batch in sorted(fixtures.items()):
            uniforms = rng.random(batch.arrays["signed"].shape)
            batches[str(length)] = batch.fingerprint()
            for condition in candidate["seeds"]["conditions"]:
                encoded, _ = encode_batch(
                    batch, condition, uniforms, candidate["encoding"]["codebook"]
                )
                eager = make_model(spec, "cuda")
                model = copy.deepcopy(eager)
                stamp = time.perf_counter()
                result = parity_checks(
                    eager,
                    model,
                    compiled(model),
                    encoded,
                    spec,
                    {"atol": tolerance["cuda_atol"], "rtol": tolerance["cuda_rtol"]},
                )
                result.update(
                    reference_checks(
                        encoded,
                        spec,
                        {
                            "atol": tolerance["float64_atol"],
                            "rtol": tolerance["float64_rtol"],
                        },
                    )
                )
                torch.cuda.synchronize()
                checks.update(
                    {
                        f"{length}/{condition}/{key}": value
                        for key, value in result.items()
                    }
                )
                print(
                    length,
                    condition,
                    "passed",
                    all(row["passed"] for row in result.values()),
                    "seconds",
                    time.perf_counter() - stamp,
                    flush=True,
                )
        result = {
            "passed": all(row["passed"] for row in checks.values()),
            "source_commit": commit,
            "sources": sources(),
            "protocol_sha256": PROTOCOL_HASH,
            "seed": settings["qualification_seed"],
            "liu_evaluated": False,
            "runtime": execution,
            "fixture_batch_hashes": batches,
            "cpu_test_modules": list(CPU_TESTS),
            "checks": checks,
        }
        write_json_exclusive(directory / "qualification.json", result)
    return {
        "passed": result["passed"],
        "directory": str(directory),
        "check_count": len(checks),
    }
