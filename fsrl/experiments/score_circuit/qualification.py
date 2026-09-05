"""Non-Liu fixtures and numerical qualification, never candidate selection."""

import time

import numpy as np
import torch
import torch._inductor.config as inductor_config

from fsrl.infra.runtime import ExecutionProfile, configure_runtime

from .circuit import (
    coefficients,
    initial_state,
    integrate_support,
    integration_chunk,
    query_read,
)
from .reference import affine_support, differences


def fixture(trials: int = 3, subjects: int = 4) -> dict:
    rng = np.random.default_rng(930101)
    return {
        "support_cues": rng.choice([-1.0, 1.0], (trials, subjects, 30)),
        "signed": rng.uniform(-1, 1, (trials, subjects)),
        "retention": np.ones((trials, subjects)),
        "query_cues": rng.choice([-1.0, 1.0], (subjects, 6, 30)),
    }


def runtime() -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inductor_config.compile_threads = 1
    torch._dynamo.config.recompile_limit = 16
    result = configure_runtime(ExecutionProfile())
    result["compiler_threads"] = 1
    result["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return result


def compiled_runner():
    return torch.compile(integration_chunk, fullgraph=True, mode="default")


def compiler_check(runner) -> dict:
    inputs = fixture()
    rates = torch.as_tensor(2 + differences(inputs["support_cues"])[0], device="cuda")
    y = initial_state(4, 15, "cuda")
    diagnostics = y.new_full((4, 5), float("inf"))
    diagnostics[:, 4] = 0
    args = (
        y,
        diagnostics,
        rates,
        y.new_tensor(inputs["signed"][0]),
        y.new_ones(4),
        y.new_ones(4),
        y.new_tensor(1),
        coefficients(0.989, 1, 4096, "cuda"),
    )
    before = time.perf_counter()
    actual = runner(*args)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - before
    expected = integration_chunk(*args)
    errors = []
    for a, b in zip(actual, expected, strict=True):
        error = float((a - b).abs().max().item())
        if error > 1e-9:
            raise RuntimeError("compiled/eager circuit mismatch")
        errors.append(error)
    return {"max_abs_errors": errors, "compile_warmup_seconds": seconds}


def qualify() -> dict:
    snapshot = runtime()
    runner = compiled_runner()
    compile_result = compiler_check(runner)
    print("CUDA compile/eager fixture passed", flush=True)
    inputs = fixture()
    checks = {}
    for scale in (0.5, 1.0, 2.0):
        expected = affine_support(inputs, 0.989, scale)
        for steps in (4096, 8192):
            output = integrate_support(inputs, 0.989, scale, steps, runner)
            error = float(np.max(np.abs(output["trajectory"] - expected)))
            checks[f"affine/{scale}/{steps}"] = error
            if error > 1e-5:
                raise RuntimeError(f"non-Liu affine reference mismatch: {error}")
        print(f"Non-Liu scale {scale} reference checks passed", flush=True)
    for control in ("teacher_off", "mismatch_clamp"):
        output = integrate_support(inputs, 0.989, 1, 4096, runner, control=control)
        np.testing.assert_array_equal(output["trajectory"][..., :30], 1)
        checks[control] = True
    single = fixture(trials=1)
    durations = {}
    for duration in (0.25, 0.5, 1.0, 2.0):
        output = integrate_support(single, 0.989, 1, 4096, runner, duration=duration)
        expected = affine_support(single, 0.989, 1, duration=duration)
        np.testing.assert_allclose(output["trajectory"], expected, atol=1e-5, rtol=0)
        x = differences(single["support_cues"])[0]
        y = output["trajectory"][:, -1]
        prediction = np.sum((y[:, :15] - y[:, 15:30]) * x, axis=-1)
        durations[str(duration)] = {
            "realized_fraction": (prediction / single["signed"][0]).tolist(),
            "fast_limit_fraction": float(1 - (1 - 0.989) ** duration),
        }
    stress = fixture(trials=1, subjects=1)
    stress["support_cues"][..., :15] = 1
    stress["support_cues"][..., 15:] = -1
    stress["signed"][:] = 100
    bounded = integrate_support(stress, 0.989, 1, 4096, runner)
    if bounded["diagnostics"][0, 4] <= 0:
        raise RuntimeError("deliberate non-Liu bound fixture did not engage bounds")
    unbounded = affine_support(stress, 0.989, 1)
    state = output["trajectory"][:, -1].copy()
    before = state.copy()
    query_read(state, single["query_cues"], 7.2, 0.002)
    np.testing.assert_array_equal(before, state)
    return {
        "experiment_id": "score_circuit_v1",
        "seed": 930101,
        "liu_evaluated": False,
        "passed": True,
        "runtime": snapshot,
        "compiler": compile_result,
        "checks": checks,
        "duration": durations,
        "deliberate_bound": {
            "engagement_count": float(bounded["diagnostics"][0, 4]),
            "unbounded_max_efficacy": float(unbounded[..., :30].max()),
            "bounded_max_efficacy": float(bounded["trajectory"][..., :30].max()),
        },
    }
