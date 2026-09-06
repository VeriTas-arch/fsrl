"""Non-Liu CUDA parity of the changed input and optional-memory computation."""

import copy

import numpy as np

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .inputs import generator, prepare_shared
from .locks import sources
from .model import forward_batch, make_model, objective, optimizer_for, update
from .protocol import RUN_ROOT, specification


def compare(first, second) -> dict:
    np.testing.assert_allclose(
        first.detach().cpu().numpy(),
        second.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-4,
    )
    return {
        "passed": True,
        "max_absolute_error": float((first.detach() - second.detach()).abs().max()),
    }


def qualify_condition(spec: dict, condition: str) -> dict:
    seed = spec["seeds"]["smoke"]
    backbone, local = make_model(spec, seed, condition, "cuda")
    compiled_backbone, compiled_local = copy.deepcopy(backbone), copy.deepcopy(local)
    eager = RecurrentSequence(backbone)
    compiled = compile_module(RecurrentSequence(compiled_backbone), PROFILE)
    batch = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(seed), 2)
    ).to("cuda")
    first = forward_batch(backbone, local, eager, batch)
    second = forward_batch(compiled_backbone, compiled_local, compiled, batch)
    checks = {
        "P_T": compare(first.weights, second.weights),
        "logits": compare(first.logits, second.logits),
    }
    first_loss, _ = objective(first, batch, 0.0)
    second_loss, _ = objective(second, batch, 0.0)
    first_loss.backward()
    second_loss.backward()
    checks["loss"] = compare(first_loss, second_loss)
    for name in ("i2h.weight", "h2DA.weight", "alpha", "h2o.weight"):
        a = dict(backbone.named_parameters())[name].grad
        b = dict(compiled_backbone.named_parameters())[name].grad
        if a is None or b is None or float(a.norm()) == 0:
            raise RuntimeError(f"missing query-task gradient: {name}")
        checks[f"gradient_{name}"] = compare(a, b)
    assert backbone.i2h.weight.grad is not None
    if float(backbone.i2h.weight.grad[:, -1].abs().sum()) == 0:
        raise RuntimeError("weak-evidence channel receives no learning signal")
    if local is not None:
        assert compiled_local is not None
        checks["gradient_local"] = compare(
            local.raw_gain.grad, compiled_local.raw_gain.grad
        )
    before = tensor_hashes(backbone)
    update(backbone, local, eager, batch, optimizer_for(backbone, local, spec), spec)
    update(
        compiled_backbone,
        compiled_local,
        compiled,
        batch,
        optimizer_for(compiled_backbone, compiled_local, spec),
        spec,
    )
    for name, value in backbone.state_dict().items():
        checks[f"updated_{name}"] = compare(value, compiled_backbone.state_dict()[name])
    if before == tensor_hashes(backbone):
        raise RuntimeError("optimizer made no update")
    if local is not None:
        assert compiled_local is not None
        checks["updated_local"] = compare(local.raw_gain, compiled_local.raw_gain)
    return checks


def run_qualification() -> dict:
    spec = specification()
    runtime = configure_execution()
    directory = RUN_ROOT / "qualification"
    with ProspectiveRun.start(
        directory,
        workflow_id="matched_memory_structure_v1",
        execution_id="qualification",
        producer={"module": __name__},
        resolved_config={"runtime": runtime},
    ):
        checks = {
            condition: qualify_condition(spec, condition)
            for condition in spec["seeds"]["conditions"]
        }
        result = {
            "passed": True,
            "seed": spec["seeds"]["smoke"],
            "liu_evaluated": False,
            "runtime": runtime,
            "checks": checks,
            "sources": sources(),
        }
        write_json_exclusive(directory / "result.json", result)
    return {"passed": True, "conditions": list(checks)}
