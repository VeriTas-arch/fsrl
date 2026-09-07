"""Independent L1 phase accounting, clipping, and unchanged rollout endpoints."""

import hashlib
import json

import numpy as np
import torch

from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.linear_modulation.model import load_model, make_model
from fsrl.experiments.linear_modulation.qualification import IndependentAffine
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.modulation_schedule.model import scheduled_sequences
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .budgets import BudgetSupport, inputs_for, measure, probes, totals
from .execution import sources
from .protocol import RECORDS, RUNS, prior_records, recipe


def numerical_checks(device="cpu", compiled=False):
    spec = recipe()
    spec["architecture"]["hidden_size"] = 8
    net, _ = make_model(spec, 946003, device=device)
    other = IndependentAffine(net.model_config, device=device)
    other.load_state_dict(net.state_dict())
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(946001), 2)
    )
    inputs = cpu.to(device).support_inputs[0]
    h, e, p = (
        net.initial_hidden(2),
        net.initial_eligibility(2),
        net.initial_fast_weights(2),
    )
    values = p.new_tensor([0, 0, -0.3, 0.5])
    checks = {}
    for native in (False, True):
        probe = BudgetSupport(net, native)
        if compiled:
            probe = compile_module(probe, PROFILE)
        actual, amounts, clips = probe(inputs, h, e, p, values)
        hh, ee, pp = h, e, p
        expected = []
        for phase, x in enumerate(inputs):
            _, _, _, hh, new_e, new_p = other(x, hh, ee, pp)
            updated = new_p if native else (pp + values[phase] * ee).clamp(-50, 50)
            expected.append(((updated - pp) * other.alpha).abs().sum((1, 2)) / 64)
            ee, pp = new_e, updated
        checks[f"{native}/P"] = compare(actual, pp)
        checks[f"{native}/phase_L1"] = compare(amounts, torch.stack(expected))
        assert torch.count_nonzero(amounts[:2]) == 0 and torch.count_nonzero(clips) == 0
    h, e, p = net.initial_hidden(2), torch.ones_like(p), torch.full_like(p, 49.9)
    updated, amounts, clips = BudgetSupport(net, False)(
        inputs[:1], h, e, p, p.new_tensor([2])
    )
    assert torch.all(updated == 50) and torch.all(clips == 1)
    checks["clipped_L1"] = compare(
        amounts[0], ((updated - p) * net.alpha).abs().mean((1, 2))
    )
    return checks


def qualify():
    prior = prior_records()
    runtime = json_ready(configure_execution())
    suffix = hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[
        :12
    ]
    directory = RUNS / ("qualification-" + suffix)
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="write_budget_match_v1",
            execution_id="qualification",
            producer={"sources": sources()},
            resolved_config={"runtime": runtime},
        ),
        torch.no_grad(),
    ):
        checks = numerical_checks("cuda", True)
        panel = {**prior["source"]["panels"]["1"], "id": 1}
        files = prior["models"]["runs"]["2531/clean"]["files"]
        net, _, seqs = load_model(2531, files, recipe())
        cpu = inputs_for(panel, generic_only=True)["clean/test-28"]
        kappa = prior["schedule_calibration"]["models"]["2531/clean"]["phase"]
        blocks = probes(net)
        for mode in ("B", "S"):
            selected = seqs if mode == "B" else scheduled_sequences(net, seqs, kappa)
            result, _, w = rollout(net, None, selected, cpu.to("cuda"), None, 0)
            raw, p = measure(
                net, seqs[1], blocks["B" if mode == "B" else "M"], cpu, kappa
            )
            checks[mode + "/P"] = compare(p, result.weights)
            checks[mode + "/total_L1"] = compare(
                torch.tensor(totals(raw).sum(1), device="cuda"), w.sum(0)
            )
        result = {
            "passed": True,
            "sources": sources(),
            "runtime": runtime,
            "checks": checks,
            "new_behavior_exposed": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", json_ready(result))
    return {"passed": True, "checks": len(checks)}
