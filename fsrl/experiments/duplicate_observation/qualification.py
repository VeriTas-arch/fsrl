"""Duplicate-observation numerical qualification and archived single-memory collector parity."""

import copy
import hashlib
import json

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.model import rollout, update
from fsrl.experiments.local_memory_removal.statistics import qualification_checks
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    make_model,
    objective,
    optimizer_for,
)
from fsrl.experiments.memory_structure.model import (
    update as reference_update,
)
from fsrl.experiments.observation_replication.reporting import read_raw
from fsrl.experiments.observation_uncertainty.inputs import encode
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .compact import compact_batch, compact_model, compare_rollouts
from .execution import sources
from .inputs import duplicate_observation
from .protocol import RECORDS, RUNS, parents, recipe


def numerical_checks(device="cpu", compiled=False):
    spec = recipe()
    if device == "cpu":
        spec["architecture"]["hidden_size"] = 8
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(925001), 2)
    )
    cpu = encode(cpu, "noisy", 1 / 7, seed=925002)
    cpu.arrays["support_inputs"] = cpu.arrays["support_inputs"][:4]
    cpu.arrays["local_evidence"] = cpu.arrays["local_evidence"][:4]
    cpu = duplicate_observation(cpu)
    batch = cpu.to(device)
    net, local = make_model(spec, 925003, "single", device)
    assert local is None
    other = copy.deepcopy(net)
    seqs = sequences(net, None, compiled=compiled)
    seq = RecurrentSequence(other)
    actual, cost, writes = rollout(net, None, seqs, batch, None, 0)
    expected = forward_batch(other, None, seq, batch)
    smaller = compact_model(net)
    small_seqs = sequences(smaller, None, compiled=compiled)
    mapped = rollout(smaller, None, small_seqs, compact_batch(cpu).to(device), None, 0)
    compact_checks = compare_rollouts((actual, cost, writes), mapped, cpu)
    assert actual.local_state is None
    assert torch.equal(actual.logits, actual.global_logits)
    checks = {
        "compact_mapping": {"passed": True, "checks": compact_checks},
        "P_T": compare(actual.weights, expected.weights),
        "logits": compare(actual.logits, expected.logits),
    }
    loss, _ = objective(actual, batch, spec["optimization"]["fast_weight_penalty"])
    ref_loss, _ = objective(
        expected, batch, spec["optimization"]["fast_weight_penalty"]
    )
    loss.backward()
    ref_loss.backward()
    assert net.i2h.weight.grad is not None
    assert torch.equal(net.i2h.weight.grad[:, 34], net.i2h.weight.grad[:, 37])
    checks["loss"] = compare(loss, ref_loss)
    for (name, parameter), (other_name, reference) in zip(
        net.named_parameters(), other.named_parameters(), strict=True
    ):
        assert name == other_name
        if reference.grad is None:
            assert name in ("h2v.weight", "h2v.bias")
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            checks["unused_value_gradient_" + name] = {"passed": True}
        else:
            assert parameter.grad is not None
            checks["gradient_" + name] = compare(parameter.grad, reference.grad)
    optimizer, ref_optimizer = (
        optimizer_for(net, None, spec),
        optimizer_for(other, None, spec),
    )
    update(net, None, seqs, batch, optimizer, spec, None, 0.0, 0)
    reference_update(other, None, seq, batch, ref_optimizer, spec)
    for (name, parameter), (_, reference) in zip(
        net.named_parameters(), other.named_parameters(), strict=True
    ):
        checks["updated_" + name] = compare(parameter, reference)
    assert len(optimizer.param_groups) == 1
    checks["no_local_state_or_parameters"] = {"passed": True}
    return checks


def qualify():
    from .evaluation import collect

    runtime = json_ready(configure_execution())
    prior = parents()
    if runtime != prior["source"]["runtime"]:
        raise RuntimeError("runtime differs from reference")
    files = sources()
    suffix = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    directory = RUNS / "qualification" / suffix
    with ProspectiveRun.start(
        directory,
        workflow_id="duplicate_observation_v1",
        execution_id="duplicate-observation-qualification",
        producer={"sources": files},
        resolved_config={"runtime": runtime},
    ):
        # Perform archived replay before a new compiled fixture changes specialization.
        panel = copy.deepcopy(prior["source"]["panels"]["1"])
        artifacts = {
            **prior["source"]["parent_artifacts"],
            **prior["result"]["artifacts"],
        }
        for name, row in panel["inputs"].items():
            row["file"] = artifacts[f"inputs/1/{name}.npz"]
        model = {
            name: artifacts[f"training/2531/clean/{name}"]
            for name in ("net.pth", "result.json")
        }
        _, raw, _ = collect(
            2531, "clean", panel, model, recipe(1), 1002531, keep_hint=True
        )
        old = read_raw(artifacts["evaluation/2531/1/A0/raw.npz"])
        exact = []
        for phase in raw:
            for key, value in raw[phase].items():
                if not np.array_equal(value, old[phase][key], equal_nan=True):
                    raise RuntimeError(f"hint-present replay differs: {phase}/{key}")
                exact.append(phase + "/" + key)
        checks = numerical_checks("cuda", True)
        result = {
            "passed": True,
            "sources": files,
            "runtime": runtime,
            "exact_hint_present_arrays": exact,
            "single_numeric_checks": checks,
            "statistics": qualification_checks(),
            "new_duplicate_outcomes": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", json_ready(result))
    return {
        "passed": True,
        "hint_present_arrays": len(exact),
        "single_numeric_checks": len(checks),
    }
