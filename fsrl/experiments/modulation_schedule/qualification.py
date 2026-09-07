"""Independent support equations, unchanged baseline replay and audit parity."""

import hashlib
import json

import numpy as np
import torch

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.linear_modulation.model import load_model, make_model
from fsrl.experiments.linear_modulation.qualification import IndependentAffine
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.observation_replication.reporting import read_raw
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .audit import trace
from .evaluation import collect
from .execution import sources
from .model import ScheduledSupport, scheduled_sequences
from .protocol import RECORDS, RUNS, parents, recipe


def numerical_checks(device="cpu", compiled=False):
    spec = recipe()
    spec["architecture"]["hidden_size"] = 8
    net, _ = make_model(spec, 936003, device=device)
    other = IndependentAffine(net.model_config, device=device)
    other.load_state_dict(net.state_dict())
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(936001), 2)
    )
    inputs = cpu.to(device).support_inputs[0]
    h, e, p = (
        net.initial_hidden(2),
        net.initial_eligibility(2),
        net.initial_fast_weights(2),
    )
    values = (0.0, 0.0, -0.25, 0.35)
    seqs = scheduled_sequences(net, sequences(net, None), values, compiled=compiled)
    actual, actual_cost = seqs[0](inputs, h, e, p, p.new_zeros(()))
    cost = p.new_zeros(2)
    for phase, current in enumerate(inputs):
        _, _, _, h, next_e, _ = other(current, h, e, p)
        updated = (p + e * values[phase]).clamp(-50, 50)
        cost += ((updated - p) * other.alpha).abs().mean((1, 2))
        if phase < 2:
            assert torch.count_nonzero(e) == 0 and torch.count_nonzero(updated) == 0
        e, p = next_e, updated
    checks = {
        "scheduled_P": compare(actual, p),
        "scheduled_cost": compare(actual_cost, cost),
    }
    original = sequences(net, None)
    preserved = scheduled_sequences(net, original, None, compiled=compiled)
    batch = cpu.to(device)
    with torch.no_grad():
        a, ca, wa = rollout(net, None, original, batch, None, 0)
        b, cb, wb = rollout(net, None, preserved, batch, None, 0)
    checks.update(
        {
            name: compare(x, y)
            for name, x, y in (
                ("intact_P", a.weights, b.weights),
                ("intact_logits", a.logits, b.logits),
                ("intact_cost", ca, cb),
                ("intact_writes", wa, wb),
            )
        }
    )
    # Independent clipping fixture with nonzero old eligibility at sequence entry.
    h, e, p = net.initial_hidden(2), torch.ones_like(p), torch.full_like(p, 49.9)
    clipped, _ = ScheduledSupport(net, (2.0,))(inputs[:1], h, e, p, p.new_zeros(()))
    assert torch.all(clipped == 50)
    return checks


def qualify():
    prior = parents()
    runtime = json_ready(configure_execution())
    assert runtime == prior["source"]["runtime"]
    suffix = hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[
        :12
    ]
    directory = RUNS / ("qualification-" + suffix)
    with ProspectiveRun.start(
        directory,
        workflow_id="modulation_schedule_v1",
        execution_id="qualification",
        producer={"sources": sources()},
        resolved_config={"runtime": runtime},
    ):
        panel = {**prior["source"]["panels"]["1"], "id": 1}
        model = prior["models"]["runs"]["2531/clean"]["files"]
        _, raw, _ = collect(2531, "clean", panel, model, None)
        old = read_raw(prior["result"]["artifacts"]["evaluation/2531/1/A0/raw.npz"])
        exact = []
        for task in raw:
            for name, value in raw[task].items():
                np.testing.assert_array_equal(value, old[task][name])
                exact.append(task + "/" + name)
        checks = numerical_checks("cuda", True)
        net, _, seqs = load_model(2531, model, recipe())
        cpu = observed(load_input(panel["inputs"]["test-28"]), "clean", recipe(1))
        with torch.no_grad():
            trace_raw, weights = trace(net, cpu)
            original, _, _ = rollout(net, None, seqs, cpu.to("cuda"), None, 0)
        checks["trace_P"] = compare(weights, original.weights)
        for phase in (0, 1):
            assert np.all(trace_raw["potential"][trace_raw["phase"] == phase] == 0)
        result = {
            "passed": True,
            "runtime": runtime,
            "sources": sources(),
            "checks": checks,
            "intact_arrays_exact": exact,
            "new_interventions_exposed": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", json_ready(result))
    return {"passed": True, "exact_arrays": len(exact), "numeric_checks": len(checks)}
