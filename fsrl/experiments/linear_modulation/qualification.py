"""Independent affine equations and unchanged reference collector qualification."""

import copy
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.admission_hint_removal.compact import compare_rollouts
from fsrl.experiments.duplicate_observation.compact import compact_batch, compact_model
from fsrl.experiments.duplicate_observation.inputs import duplicate_observation
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.model import rollout, update
from fsrl.experiments.local_memory_removal.statistics import qualification_checks
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    objective,
    optimizer_for,
)
from fsrl.experiments.memory_structure.model import make_model as original_model
from fsrl.experiments.memory_structure.model import update as reference_update
from fsrl.experiments.observation_replication.reporting import read_raw
from fsrl.experiments.observation_uncertainty.inputs import encode
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import sources
from .model import LinearModulationRNN, common_hashes, make_model
from .protocol import RECORDS, RUNS, parents, recipe


class IndependentAffine(LinearModulationRNN):
    """Independent tensor expressions; never calls the candidate forward method."""

    def forward(self, inputs, hidden, et, pw):
        h = torch.tanh(
            F.linear(inputs, self.i2h.weight, self.i2h.bias)
            + F.linear(hidden, self.w)
            + torch.einsum("ij,bij,bj->bi", self.alpha, pw, hidden)
        )
        m = (h * self.h2DA.weight[0]).sum(-1, keepdim=True) + self.h2DA.bias
        updated = torch.clamp(pw + m[:, :, None] * et, -50, 50)
        trace = et + self.etaet * (torch.tanh(h[:, :, None] * hidden[:, None, :]) - et)
        return (
            F.linear(h, self.h2o.weight, self.h2o.bias),
            F.linear(h, self.h2v.weight, self.h2v.bias),
            m,
            h,
            trace,
            updated,
        )


def initialization_checks(spec, seed, device):
    original, _ = original_model(spec, seed, "single", device)
    old_rng = torch.get_rng_state().clone()
    net, _ = make_model(spec, seed, "single", device)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert common_hashes(tensor_hashes(net)) == common_hashes(tensor_hashes(original))
    assert not hasattr(net, "DAmult") and net.h2DA.out_features == 1
    torch.testing.assert_close(
        net.h2DA.weight[0],
        original.DAmult * (original.h2DA.weight[0] - original.h2DA.weight[1]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        net.h2DA.bias,
        original.DAmult * (original.h2DA.bias[:1] - original.h2DA.bias[1:]),
        rtol=0,
        atol=0,
    )
    return {"passed": True, "common_initialization_and_rng_exact": True}


def numerical_checks(device="cpu", compiled=False):
    spec = recipe()
    if device == "cpu":
        spec["architecture"]["hidden_size"] = 8
    init = initialization_checks(spec, 925003, device)
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(925001), 2)
    )
    cpu = duplicate_observation(encode(cpu, "noisy", 1 / 7, seed=925002))
    cpu.arrays["support_inputs"] = cpu.arrays["support_inputs"][:4]
    cpu.arrays["local_evidence"] = cpu.arrays["local_evidence"][:4]
    batch = cpu.to(device)
    net, _ = make_model(spec, 925003, "single", device)
    other = IndependentAffine(net.model_config, device=device)
    other.load_state_dict(net.state_dict())
    seqs, seq = sequences(net, None, compiled=compiled), RecurrentSequence(other)
    actual, cost, writes = rollout(net, None, seqs, batch, None, 0)
    expected = forward_batch(other, None, seq, batch)
    smaller = compact_model(net)
    mapped = rollout(
        smaller,
        None,
        sequences(smaller, None, compiled=compiled),
        compact_batch(cpu).to(device),
        None,
        0,
    )
    checks = {
        "initialization": init,
        "compact_mapping": {
            "passed": True,
            "checks": compare_rollouts((actual, cost, writes), mapped, cpu),
        },
        "P_T": compare(actual.weights, expected.weights),
        "logits": compare(actual.logits, expected.logits),
    }
    loss, _ = objective(actual, batch, spec["optimization"]["fast_weight_penalty"])
    ref_loss, _ = objective(
        expected, batch, spec["optimization"]["fast_weight_penalty"]
    )
    loss.backward()
    ref_loss.backward()
    checks["loss"] = compare(loss, ref_loss)
    checks.update(gradient_checks(net, other))
    update(net, None, seqs, batch, optimizer_for(net, None, spec), spec, None, 0.0, 0)
    reference_update(other, None, seq, batch, optimizer_for(other, None, spec), spec)
    for (name, p), (_, q) in zip(
        net.named_parameters(), other.named_parameters(), strict=True
    ):
        checks["updated/" + name] = compare(p, q)
    return checks


def gradient_checks(net, other):
    checks = {}
    for (name, p), (other_name, q) in zip(
        net.named_parameters(), other.named_parameters(), strict=True
    ):
        assert name == other_name
        if q.grad is None:
            assert name in ("h2v.weight", "h2v.bias")
            assert p.grad is None or torch.count_nonzero(p.grad) == 0
            checks["unused/" + name] = {"passed": True}
        else:
            assert p.grad is not None
            checks["gradient/" + name] = compare(p.grad, q.grad)
    return checks


def qualify():
    from fsrl.experiments.duplicate_observation.evaluation import collect

    runtime, prior = json_ready(configure_execution()), parents()
    if runtime != prior["source"]["runtime"]:
        raise RuntimeError("runtime differs from frozen reference")
    files = sources()
    suffix = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    directory = RUNS / "qualification" / suffix
    with ProspectiveRun.start(
        directory,
        workflow_id="linear_modulation_v1",
        execution_id="affine-qualification",
        producer={"sources": files},
        resolved_config={"runtime": runtime},
    ):
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
        _, raw, _ = collect(2531, "clean", panel, model, recipe(1), 1002531)
        old = read_raw(artifacts["evaluation/2531/1/A0/raw.npz"])
        exact = []
        for phase in raw:
            for key, value in raw[phase].items():
                if not np.array_equal(value, old[phase][key], equal_nan=True):
                    raise RuntimeError(
                        f"opponent reference replay differs: {phase}/{key}"
                    )
                exact.append(phase + "/" + key)
        checks = numerical_checks("cuda", True)
        result = {
            "passed": True,
            "sources": files,
            "runtime": runtime,
            "exact_opponent_arrays": exact,
            "affine_numeric_checks": checks,
            "statistics": qualification_checks(),
            "new_affine_outcomes": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", json_ready(result))
    return {
        "passed": True,
        "opponent_arrays": len(exact),
        "affine_numeric_checks": len(checks),
    }
