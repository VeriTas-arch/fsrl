"""Locate the first compact mismatch without changing its acceptance tolerance."""

import copy

import numpy as np
import torch

from fsrl.experiments.admission_hint_removal.compact import complete_order
from fsrl.experiments.duplicate_observation.compact import compact_batch, compact_model
from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.linear_modulation.model import load_model
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate
from .protocol import RUNS, SOURCE, recipe


def difference(a, b):
    aa, bb = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    bad = ~np.isclose(aa, bb, atol=1e-5, rtol=1e-4)
    result = {
        "passed": not bool(bad.any()),
        "elements": aa.size,
        "violations": int(bad.sum()),
        "max_absolute_error": float(np.max(np.abs(aa - bb))),
    }
    if bad.any():
        index = tuple(np.argwhere(bad)[0])
        result["first_violation"] = {
            "index": list(index),
            "original": float(aa[index]),
            "merged": float(bb[index]),
            "absolute_error": float(abs(aa[index] - bb[index])),
        }
    return result


def compare_case(net, smaller, padded, compact, cpu):
    a, ca, wa = rollout(net, None, padded, cpu.to("cuda"), None, 0)
    b, cb, wb = rollout(smaller, None, compact, compact_batch(cpu).to("cuda"), None, 0)
    checks: dict = {
        key: difference(x, y)
        for key, x, y in (
            ("logits", a.logits, b.logits),
            ("P_T", a.weights, b.weights),
            ("first_P", a.first_write, b.first_write),
            ("cost", ca, cb),
            ("writes", wa, wb),
        )
    }
    checks["argmax_equal"] = bool(torch.equal(a.logits.argmax(-1), b.logits.argmax(-1)))
    checks["order_equal"] = bool(
        np.array_equal(complete_order(a.logits, cpu), complete_order(b.logits, cpu))
    )
    passed = all(v["passed"] for v in checks.values() if isinstance(v, dict))
    return {
        "passed": passed and checks["argmax_equal"] and checks["order_equal"],
        "checks": checks,
    }


def locate(prior, net, smaller, padded, compact, directory):
    records = {}
    task = size_protocol(recipe(), 8)
    for panel in (1, 2, 3):
        for arm in ("clean", "noisy"):
            for name, ref in sorted(
                prior["source"]["panels"][str(panel)]["inputs"].items()
            ):
                cpu = observed(load_input(ref), arm, recipe(panel))
                cases = {"intact": cpu}
                if name == "liu-8":
                    shuffled, _ = shuffle_evidence(
                        cpu,
                        task.support_blocks,
                        recipe(panel)["evaluation"]["liu"]["evidence_shuffle_seed"],
                    )
                    cases["evidence_shuffle"] = shuffled
                    cases.update(
                        {
                            f"removed-{i}": remove_relation(cpu, pair)
                            for i, pair in enumerate(task.support_pairs_higher_lower)
                        }
                    )
                for label, batch in cases.items():
                    key = f"{panel}/{arm}/{name}/{label}"
                    checks = compare_case(net, smaller, padded, compact, batch)
                    records[key] = checks
                    write_json_exclusive(
                        directory / (key.replace("/", "-") + ".json"),
                        json_ready(checks),
                    )
                    if not checks["passed"]:
                        return records, key, batch
    return records, None, None


def trajectory_pair(a, b, cpu):
    dtype = a.w.dtype
    x = cpu.to("cuda").support_inputs.to(dtype)
    y = compact_batch(cpu).to("cuda").support_inputs.to(dtype)
    subjects = x.shape[2]
    pa, pb = a.initial_fast_weights(subjects), b.initial_fast_weights(subjects)
    xa = [x.new_zeros(2, subjects, x.shape[-1]), *x.unbind(0)]
    xb = [y.new_zeros(2, subjects, y.shape[-1]), *y.unbind(0)]
    rows = []
    for trial, (sa, sb) in enumerate(zip(xa, xb, strict=True)):
        ha, hb = a.initial_hidden(subjects), b.initial_hidden(subjects)
        ea, eb = a.initial_eligibility(subjects), b.initial_eligibility(subjects)
        for phase, (ia, ib) in enumerate(zip(sa, sb, strict=True)):
            projection_a, projection_b = a.i2h(ia), b.i2h(ib)
            _, _, ma, ha, ea, pa = a(ia, ha, ea, pa)
            _, _, mb, hb, eb, pb = b(ib, hb, eb, pb)
            values = {
                "input_projection": (projection_a, projection_b),
                "h": (ha, hb),
                "m": (ma, mb),
                "E": (ea, eb),
                "P": (pa, pb),
            }
            errors = {}
            for name, (first, second) in values.items():
                delta = (first - second).abs()
                errors[name] = {
                    "max_absolute_error": float(delta.max()),
                    "max_tolerance_ratio": float(
                        (delta / (1e-5 + 1e-4 * second.abs())).max()
                    ),
                }
            rows.append({"trial": trial - 1, "phase": phase + 1, "errors": errors})
    return {"rows": rows, "final_P": difference(pa, pb)}


def run():
    prior = validate()
    model = prior["models"]["runs"]["2533/clean"]["files"]
    net, _, padded = load_model(2533, model, recipe())
    smaller = compact_model(net).requires_grad_(False).eval()
    compact = sequences(smaller, None, compiled=True)
    directory = RUNS / "compact_diagnostic"
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="modulation_schedule_v1",
            execution_id="compact-first-mismatch",
            producer={"source": reference(SOURCE)},
            resolved_config={"model": "2533/clean", "atol": 1e-5, "rtol": 1e-4},
        ),
        torch.no_grad(),
    ):
        records, key, cpu = locate(prior, net, smaller, padded, compact, directory)
        traces = {}
        if cpu is not None:
            traces["float32_eager"] = trajectory_pair(net, smaller, cpu)
            double = copy.deepcopy(net).double()
            traces["float64_after_float32_merge"] = trajectory_pair(
                double, copy.deepcopy(smaller).double(), cpu
            )
            traces["float64_before_merge"] = trajectory_pair(
                double, compact_model(double), cpu
            )
        result = {
            "diagnostic_complete": True,
            "failed_case": key,
            "cases": records,
            "propagation": traces,
            "new_export_qualification": False,
            "boundary": "First failure localization on frozen2533/clean; standalone compilation may not reproduce prior warmup. Float64 comparisons diagnose arithmetic only, never replace original qualification.",
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return {"diagnostic_complete": True, "failed_case": key, "qualified_export": False}
