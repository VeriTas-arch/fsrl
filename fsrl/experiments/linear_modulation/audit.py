"""Read-only modulation and potential-write audit of frozen opponent networks."""

import numpy as np
import torch

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.local_memory_removal.evaluation import load_model
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.model import read_queries
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate_source
from .protocol import RECORDS, RUNS, SOURCE, recipe, specification


def trace(net, cpu):
    batch = cpu.to("cuda")
    subjects = batch.support_inputs.shape[2]
    weights = net.initial_fast_weights(subjects)
    blank = weights.new_zeros(2, subjects, net.model_config.input_size)
    rows, trials, phases = [], [], []
    for trial, sequence in enumerate([blank, *batch.support_inputs.unbind(0)]):
        hidden, eligibility = (
            net.initial_hidden(subjects),
            net.initial_eligibility(subjects),
        )
        for phase, inputs in enumerate(sequence.unbind(0)):
            _, _, m, hidden, updated_e, updated_p = net(
                inputs, hidden, eligibility, weights
            )
            drives = net.h2DA(hidden)
            potential = torch.linalg.vector_norm(net.alpha * eligibility, dim=(1, 2))
            actual = torch.linalg.vector_norm(
                net.alpha * (updated_p - weights), dim=(1, 2)
            )
            proposed = weights + m[:, :, None] * eligibility
            clipped = ((proposed < -50) | (proposed > 50)).float().mean((1, 2))
            rows.append(
                torch.stack(
                    (drives[:, 0], drives[:, 1], m[:, 0], potential, actual, clipped),
                    -1,
                )
                .cpu()
                .numpy()
            )
            trials.append(trial - 1)
            phases.append(phase)
            eligibility, weights = updated_e, updated_p
    values = np.stack(rows)
    return (
        {
            **{
                name: values[..., i]
                for i, name in enumerate(
                    (
                        "a",
                        "b",
                        "m",
                        "eligibility_norm",
                        "actual_write_norm",
                        "clipped_fraction",
                    )
                )
            },
            "trial": np.asarray(trials),
            "phase": np.asarray(phases),
        },
        weights,
        batch,
    )


def stats(a, b, m, e, write, clipped, beta):
    a, b, m, e, write, clipped = (
        v.astype(float).ravel() for v in (a, b, m, e, write, clipped)
    )
    assert all(np.isfinite(v).all() for v in (a, b, m, e, write, clipped))
    assert beta != 0
    c, d = (a + b) / 2, (a - b) / 2
    indicators = {
        "both_small": (np.abs(a) <= 0.25) & (np.abs(b) <= 0.25),
        "both_same_sign_saturated": (np.abs(np.tanh(a)) >= 0.95)
        & (np.abs(np.tanh(b)) >= 0.95)
        & (a * b > 0),
        "near_zero_m": np.abs(m) <= 0.05 * 2 * abs(beta),
    }
    weight = e**2
    active = e > 1e-12
    normalized = lambda value: (
        float(np.sum(weight * value) / weight.sum()) if weight.sum() else None
    )
    actual_power = np.sum(weight * m**2)
    return {
        "observations": len(a),
        "eligibility_active": int(active.sum()),
        "quantiles_01_05_50_95_99": {
            k: np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
            for k, v in {
                "a": a,
                "b": b,
                "common": c,
                "difference": d,
                "m": m,
                "eligibility_norm": e,
                "actual_write_norm": write,
            }.items()
        },
        "correlation_a_b": float(np.corrcoef(a, b)[0, 1])
        if a.std() and b.std()
        else None,
        "rms_common_over_difference": float(np.linalg.norm(c) / np.linalg.norm(d))
        if np.linalg.norm(d)
        else None,
        "fraction": {k: float(v.mean()) for k, v in indicators.items()},
        "eligibility_active_fraction": {
            k: float(v[active].mean()) if active.any() else None
            for k, v in indicators.items()
        },
        "potential_write_weighted_fraction": {
            k: normalized(v) for k, v in indicators.items()
        },
        "near_zero_given_saturated_and_active": float(
            indicators["near_zero_m"][
                indicators["both_same_sign_saturated"] & active
            ].mean()
        )
        if np.any(indicators["both_same_sign_saturated"] & active)
        else None,
        "linear_open_loop_relative_write_mse": float(
            np.sum(weight * (beta * (a - b) - m) ** 2) / actual_power
        )
        if actual_power
        else None,
        "mean_clipped_fraction": float(clipped.mean()),
    }


def summarize_trace(raw, beta):
    arrays = [
        raw[k]
        for k in (
            "a",
            "b",
            "m",
            "eligibility_norm",
            "actual_write_norm",
            "clipped_fraction",
        )
    ]
    np.testing.assert_allclose(
        raw["m"], beta * (np.tanh(raw["a"]) - np.tanh(raw["b"])), atol=1e-6, rtol=1e-5
    )
    a, b, m, e, write, clipped = arrays
    result = {"all": stats(a, b, m, e, write, clipped, beta)}
    for initial in (True, False):
        for phase in np.unique(raw["phase"]):
            mask = (raw["trial"] < 0) == initial
            mask &= raw["phase"] == phase
            if mask.any():
                result[f"{'initial' if initial else 'support'}/{phase}"] = stats(
                    a[mask], b[mask], m[mask], e[mask], write[mask], clipped[mask], beta
                )
    return result


def audit_reference():
    source = validate_source()
    directory = RUNS / "reference_audit"
    results = {}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="linear_modulation_v1",
            execution_id="frozen-reference-write-audit",
            producer={"source_lock": reference(SOURCE)},
            resolved_config=specification()["reference_audit"],
        ),
        torch.no_grad(),
    ):
        for seed in specification()["design"]["seeds"]:
            for training in ("clean", "noisy"):
                files = {
                    name: source["parent_artifacts"][
                        f"training/{seed}/{training}/{name}"
                    ]
                    for name in ("net.pth", "result.json")
                }
                net, _, seqs = load_model(seed, files, recipe())
                before = tensor_hashes(net)
                for observation in ("clean", "noisy"):
                    key = f"{seed}/{training}/{observation}"
                    raw, summary = {}, {}
                    for name, ref in sorted(source["panels"]["1"]["inputs"].items()):
                        cpu = observed(load_input(ref), observation, recipe(1))
                        values, weights, batch = trace(net, cpu)
                        original, _, _ = rollout(net, None, seqs, batch, None, 0)
                        logits, _ = read_queries(
                            net, None, seqs[1], batch, weights, None
                        )
                        summary[name] = {
                            "statistics": summarize_trace(values, float(net.DAmult)),
                            "P_T": compare(weights, original.weights),
                            "logits": compare(logits, original.logits),
                        }
                        raw.update({name + "__" + k: v for k, v in values.items()})
                    assert tensor_hashes(net) == before
                    path = directory / (key.replace("/", "-") + ".npz")
                    write_arrays(path, raw)
                    results[key] = {
                        "beta": float(net.DAmult),
                        "raw": reference(path),
                        "batches": summary,
                    }
                    print({"audited": key}, flush=True)
        result = {
            "passed": True,
            "source_lock": reference(SOURCE),
            "models": results,
            "panel": 1,
            "new_affine_outcomes": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    write_json_exclusive(RECORDS / "results/reference_audit.json", json_ready(result))
    return {
        "passed": True,
        "model_observation_cells": len(results),
        "parameter_updates": 0,
    }
