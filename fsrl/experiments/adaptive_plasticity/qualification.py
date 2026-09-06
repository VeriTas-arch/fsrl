"""Non-Liu analytical and fullgraph CUDA qualification."""

from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.training_strategy.generic_validation import validation_episodes
from fsrl.experiments.training_strategy.locks import reference, require_pushed_clean
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT

from .data import model_tensors
from .model import make_model
from .protocol import (
    CODEBOOK,
    CONDITIONS,
    DESIGN_HASH,
    RUN_ROOT,
    resolved_specification,
)
from .provenance import implementation_sources
from .reference import rollout

CPU_TESTS = (
    "tests.experiments.adaptive_plasticity.test_model",
    "tests.experiments.adaptive_plasticity.test_pipeline",
)


def comparison(first, second, *, atol=1e-5, rtol=1e-4) -> dict:
    a, b = (torch.as_tensor(value).detach().cpu() for value in (first, second))
    b = b.to(dtype=a.dtype)
    same = a.shape == b.shape and bool(
        torch.isfinite(a).all() and torch.isfinite(b).all()
    )
    error = float((a - b).abs().max()) if same and a.numel() else 0.0
    return {
        "passed": same and torch.allclose(a, b, atol=atol, rtol=rtol),
        "max_abs_error": error,
    }


def fixture() -> ModelBatch:
    spec = resolved_specification()
    episodes = [
        row for row in validation_episodes(spec) if len(row.support_trials) == 32
    ]
    base = generic_batch(tuple(episodes[:16]))
    uniforms = np.random.default_rng(1_090_001).random(base.arrays["signed"].shape)
    return encode_batch(base, "resampled", uniforms, CODEBOOK)[0]


def numerical_checks(condition: str, batch: ModelBatch, spec: dict) -> dict:
    eager = make_model(condition, spec, "cuda")
    model = copy.deepcopy(eager)
    runner = compiled(model)
    arguments = model_tensors(batch, "cuda")
    eager_output = eager(*arguments)
    compiled_output = runner(*arguments)
    checks = {
        f"output-{index}": comparison(first, second)
        for index, (first, second) in enumerate(
            zip(eager_output, compiled_output, strict=True)
        )
    }
    expected = rollout(
        batch,
        eta=eager.eta.item(),
        gain=eager.global_gain.item(),
        epsilon=eager.epsilon,
        adaptive=eager.adaptive,
    )
    checks["reference-margins"] = comparison(eager_output[0], expected["margins"])
    checks["reference-w"] = comparison(eager_output[1], expected["w"])
    signs = torch.as_tensor(2 * batch.arrays["targets"] - 1, device="cuda")
    optimizers = [
        torch.optim.Adam(current.parameters(), lr=spec["optimization"]["learning_rate"])
        for current in (eager, model)
    ]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    outputs = eager(*arguments), runner(*arguments)
    losses = [F.softplus(-signs * output[0]).mean() for output in outputs]
    checks["loss"] = comparison(*losses)
    for loss in losses:
        loss.backward()
    for (name, first), second in zip(
        eager.named_parameters(), model.parameters(), strict=True
    ):
        if first.grad is None or second.grad is None:
            raise RuntimeError("qualification gradient missing")
        checks[f"gradient-{name}"] = comparison(first.grad, second.grad)
        checks[f"nonzero-gradient-{name}"] = {
            "passed": bool(
                first.grad.abs().max() > 1e-12 and second.grad.abs().max() > 1e-12
            )
        }
    before = [tensor_hashes(current) for current in (eager, model)]
    for current, optimizer in zip((eager, model), optimizers, strict=True):
        torch.nn.utils.clip_grad_norm_(
            current.parameters(),
            spec["optimization"]["gradient_clip"],
            error_if_nonfinite=True,
        )
        optimizer.step()
    checks["adam-update"] = {
        "passed": all(
            previous != tensor_hashes(current)
            for previous, current in zip(before, (eager, model), strict=True)
        )
        and tensor_hashes(eager) == tensor_hashes(model)
    }
    if condition == "adaptive_eta_resampled":
        global_model = make_model(condition, spec, "cuda", scheduler="global")
        global_model.load_state_dict(eager.state_dict())
        with torch.no_grad():
            relation = eager(*arguments)[:2]
            global_output = global_model(*arguments)[:2]
        for index, (first, second) in enumerate(
            zip(relation, global_output, strict=True)
        ):
            checks[f"balanced-global-identity-{index}"] = comparison(first, second)
    return checks


def qualify(attempt: int) -> dict:
    if attempt < 1:
        raise ValueError("qualification attempt must be positive")
    commit = require_pushed_clean()
    execution = runtime()
    spec = resolved_specification()
    directory = RUN_ROOT / "qualification" / f"attempt-{attempt}"
    with ProspectiveRun.start(
        directory,
        workflow_id="experience_dependent_plasticity_v1",
        execution_id=f"qualification-{attempt}",
        producer={"module": __name__, "source_commit": commit},
        resolved_config={"runtime": execution},
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", *CPU_TESTS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        transcript = completed.stdout + completed.stderr
        (directory / "tests.txt").open("x").write(transcript)
        if completed.returncode:
            raise RuntimeError("adaptive-plasticity CPU qualification failed")
        batch = fixture()
        checks = {
            f"{condition}/{name}": value
            for condition in CONDITIONS
            for name, value in numerical_checks(condition, batch, spec).items()
        }
        result = {
            "passed": all(row["passed"] for row in checks.values()),
            "source_commit": commit,
            "sources": implementation_sources(),
            "protocol_sha256": DESIGN_HASH,
            "seed": 1_090_001,
            "liu_evaluated": False,
            "parameters_trained": False,
            "runtime": execution,
            "cpu_test_modules": list(CPU_TESTS),
            "checks": checks,
        }
        write_json_exclusive(directory / "qualification.json", result)
        if not result["passed"]:
            raise RuntimeError("adaptive-plasticity numerical qualification failed")
    return {"passed": True, "record": reference(directory / "qualification.json")}
