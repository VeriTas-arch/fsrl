"""Active-phase scalar trajectories and generic-only schedule calibration."""

import numpy as np
import torch

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.linear_modulation.model import load_model
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.model import read_queries
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate
from .protocol import RUNS, SOURCE, recipe


def experience_features(cpu):
    q = cpu.arrays["local_evidence"]
    pairs = cpu.arrays["support_pairs"]
    repeated, discrepancy = np.zeros_like(q), np.zeros_like(q)
    for subject in range(q.shape[1]):
        history = {}
        for trial in range(q.shape[0]):
            left, right = map(int, pairs[trial, subject])
            key = tuple(sorted((left, right)))
            value = float(q[trial, subject]) * (1 if left < right else -1)
            count, total = history.get(key, (0, 0.0))
            repeated[trial, subject] = count
            if count:
                discrepancy[trial, subject] = abs(value - total / count)
            history[key] = count + 1, total + value
    return q, repeated, discrepancy


def trace(net, cpu):
    batch = cpu.to(net.w.device)
    subjects = batch.support_inputs.shape[2]
    weights = net.initial_fast_weights(subjects)
    blank = weights.new_zeros(2, subjects, net.model_config.input_size)
    rows, phase_ids, trial_ids = [], [], []
    for trial, sequence in enumerate([blank, *batch.support_inputs.unbind(0)]):
        hidden = net.initial_hidden(subjects)
        eligibility = net.initial_eligibility(subjects)
        for phase, inputs in enumerate(sequence.unbind(0)):
            _, _, m, hidden, next_e, next_p = net(inputs, hidden, eligibility, weights)
            potential = torch.linalg.vector_norm(net.alpha * eligibility, dim=(1, 2))
            actual = torch.linalg.vector_norm(
                net.alpha * (next_p - weights), dim=(1, 2)
            )
            proposal = weights + m[:, :, None] * eligibility
            clipping = ((proposal < -50) | (proposal > 50)).float().mean((1, 2))
            rows.append(
                torch.stack(
                    (m[:, 0], m[:, 0] - net.h2DA.bias[0], potential, actual, clipping),
                    -1,
                )
                .cpu()
                .numpy()
            )
            phase_ids.append(phase)
            trial_ids.append(trial - 1)
            eligibility, weights = next_e, next_p
    values = np.stack(rows)
    raw = {
        name: values[..., i]
        for i, name in enumerate(
            ("m", "state_readout", "potential", "actual", "clipping")
        )
    }
    raw.update(phase=np.asarray(phase_ids), trial=np.asarray(trial_ids))
    q, repetitions, discrepancy = experience_features(cpu)
    for key, value in (
        ("q", q),
        ("repetitions", repetitions),
        ("discrepancy", discrepancy),
    ):
        raw[key] = np.concatenate(
            (np.zeros((2, subjects)), np.repeat(value, 4, axis=0))
        )
    raw["time"] = np.concatenate(
        (
            np.zeros((2, subjects)),
            np.repeat(cpu.arrays["support_inputs"][:, 0, :, 32], 4, axis=0),
        )
    )
    return raw, weights


def phase_statistics(raw, phase):
    mask = (raw["trial"] >= 0) & (raw["phase"] == phase)
    m, g = raw["m"][mask].astype(float), raw["potential"][mask].astype(float)
    weight = g**2
    denominator = weight.sum()
    result = {
        "m_quantiles": np.quantile(m, [0.01, 0.05, 0.5, 0.95, 0.99]),
        "potential_rms": np.sqrt(np.mean(weight)),
        "actual_write_mean": raw["actual"][mask].mean(),
        "mean_clipped_fraction": raw["clipping"][mask].mean(),
        "all_eligibility_zero": bool(np.all(g == 0)),
    }
    if denominator > 0:
        result.update(
            weighted_m=float(np.sum(weight * m) / denominator),
            weighted_m_rms=float(np.sqrt(np.sum(weight * m**2) / denominator)),
            weighted_negative_fraction=float(np.sum(weight * (m < 0)) / denominator),
            weighted_positive_fraction=float(np.sum(weight * (m > 0)) / denominator),
        )
    return result


def design_rows(raw, phase):
    mask = (raw["trial"] >= 0) & (raw["phase"] == phase)
    q, time, repeated, discrepancy = (
        raw[k][mask].astype(float).ravel()
        for k in ("q", "time", "repetitions", "discrepancy")
    )
    x = np.column_stack(
        (np.ones(len(q)), time, q, np.abs(q), repeated, repeated > 0, discrepancy)
    )
    return (
        x,
        raw["m"][mask].astype(float).ravel(),
        raw["potential"][mask].astype(float).ravel() ** 2,
    )


def fit_schedule(batches):
    # Each generic archive/observation has equal episode count; normalize by rows
    # before weighting to avoid longer support sequences dominating calibration.
    phase_values, cv = [], {}
    global_num, global_den = 0.0, 0.0
    for phase in (2, 3):
        rows = {key: design_rows(raw, phase) for key, raw in batches.items()}
        num = sum(np.mean(w * m) for _, m, w in rows.values())
        den = sum(np.mean(w) for _, _, w in rows.values())
        assert den > 0
        phase_values.append(num / den)
        global_num, global_den = global_num + num, global_den + den
        errors = {name: [] for name in ("phase", "current", "history")}
        for held in sorted({key.split("/")[1] for key in rows}):
            training = [v for k, v in rows.items() if k.split("/")[1] != held]
            x = np.concatenate([v[0] for v in training])
            y = np.concatenate([v[1] for v in training])
            w = np.concatenate([v[2] / len(v[2]) for v in training])
            for name, width in (("phase", 1), ("current", 4), ("history", 7)):
                beta = np.linalg.lstsq(
                    x[:, :width] * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None
                )[0]
                for key, (xx, yy, ww) in rows.items():
                    if key.split("/")[1] == held:
                        errors[name].append(
                            float(np.mean(ww * (yy - xx[:, :width] @ beta) ** 2))
                        )
        cv[str(phase + 1)] = {k: float(np.mean(v)) for k, v in errors.items()}
    return {
        "phase": [0.0, 0.0, *phase_values],
        "constant": [global_num / global_den] * 4,
        "generic_leave_archive_out_weighted_prediction_error": cv,
        "calibration_observations": ["clean", "noisy"],
        "liu_used_for_calibration": False,
    }


def run():
    prior = validate()
    directory = RUNS / "audit"
    calibration, records = {}, {}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="modulation_schedule_v1",
            execution_id="affine-trajectories",
            producer={"source": reference(SOURCE)},
            resolved_config={"panel": 1},
        ),
        torch.no_grad(),
    ):
        for key, model in prior["models"]["runs"].items():
            net, _, seqs = load_model(int(key.split("/")[0]), model["files"], recipe())
            before = tensor_hashes(net)
            generic = {}
            for arm in ("clean", "noisy"):
                for name, ref in sorted(
                    prior["source"]["panels"]["1"]["inputs"].items()
                ):
                    cpu = observed(load_input(ref), arm, recipe(1))
                    raw, weights = trace(net, cpu)
                    logits, _ = read_queries(
                        net, None, seqs[1], cpu.to("cuda"), weights, None
                    )
                    original, _, _ = rollout(net, None, seqs, cpu.to("cuda"), None, 0)
                    checks = {
                        "P_T": compare(weights, original.weights),
                        "logits": compare(logits, original.logits),
                    }
                    stats = {str(p + 1): phase_statistics(raw, p) for p in range(4)}
                    assert (
                        stats["1"]["all_eligibility_zero"]
                        and stats["2"]["all_eligibility_zero"]
                    )
                    path = directory / key / arm / (name + ".npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_arrays(path, raw)
                    records[f"{key}/{arm}/{name}"] = {
                        "raw": reference(path),
                        "checks": checks,
                        "phases": stats,
                        "bias": float(net.h2DA.bias[0]),
                    }
                    if name.startswith("test-"):
                        generic[f"{arm}/{name}"] = raw
            assert before == tensor_hashes(net)
            calibration[key] = fit_schedule(generic)
            print({"audited": key, "batches": 10}, flush=True)
        result = {
            "passed": True,
            "records": records,
            "calibration": calibration,
            "scope": "Panel1; six frozen models; 60 batches. Descriptive prediction is not causal evidence.",
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return {"passed": True, "batches": len(records)}
