"""Independent observation equations, history identities and update parity."""

import copy

import numpy as np
import torch

from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import rollout, sequences, update
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import (
    make_model,
    optimizer_for,
    read_queries,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import with_learned
from fsrl.experiments.write_cost.model import rollout as old_rollout
from fsrl.experiments.write_cost.model import sequences as old_sequences
from fsrl.experiments.write_cost.model import update as old_update
from fsrl.infra.provenance import write_json_exclusive

from .comparator import solve
from .inputs import attach_noise, encode, history_pair
from .protocol import RECORDS, specification


def reference_encoding(cpu, arm, sigma):
    a = copy.deepcopy(cpu.arrays)
    noise = a["encoding_noise_0"]
    if arm == "clean" or sigma == 0:
        return EpisodeBatch(a)
    z = a.get("trial_retention", a["retention"])
    for t in range(len(a["local_evidence"])):
        for s in range(a["local_evidence"].shape[1]):
            value = float(a["local_evidence"][t, s]) + sigma * float(noise[t, s])
            if arm == "folded":
                value = abs(value) if a["signed_magnitudes"][t, s] > 0 else -abs(value)
            q = np.float32(value)
            a["local_evidence"][t, s] = q
            a["support_inputs"][t, 0, s, 34] = z[t, s] * q
            a["support_inputs"][t, 0, s, 37] = q
    return EpisodeBatch(a)


def input_checks(spec):
    task = generator(spec)
    rng = np.random.default_rng(913002)
    cpu = attach_noise(with_learned(sample_episodes(task, rng, 8)), 1, 913)
    checks = {}
    sigma = spec["observation"]["sigma"]
    fingerprint = cpu.fingerprint()
    for arm in spec["seeds"]["conditions"]:
        actual = encode(cpu, arm, sigma)
        expected = reference_encoding(cpu, arm, sigma)
        for key in cpu.arrays:
            np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])
        assert encode(cpu, arm, 0).fingerprint() == fingerprint
        checks[f"encoding_reference_and_zero_{arm}"] = {"passed": True}
    folded = encode(cpu, "folded", sigma).arrays
    noisy = encode(cpu, "noisy", sigma).arrays
    np.testing.assert_array_equal(
        abs(folded["local_evidence"]), abs(noisy["local_evidence"])
    )
    np.testing.assert_array_equal(
        np.sign(folded["local_evidence"]), np.sign(cpu.arrays["signed_magnitudes"])
    )
    assert cpu.fingerprint() == fingerprint
    checks["paired_magnitudes_and_immutable_base"] = {"passed": True}
    histories = []
    while len(histories) < 8:
        pair = history_pair(
            sample_episodes(task, rng, 1, validation=True)[0],
            rng,
            sigma,
            len(histories),
        )
        if pair is not None:
            histories.append(pair)
    for supported, conflicting in histories:
        a, b = supported.arrays, conflicting.arrays
        last = int(a["target_index"])
        relation = set(a["support_pairs"][last, 0])
        for key in ("support_inputs", "local_evidence"):
            np.testing.assert_array_equal(a[key][last], b[key][last])
        np.testing.assert_array_equal(a["local_evidence"], b["local_evidence"])
        np.testing.assert_array_equal(a["query_inputs"], b["query_inputs"])
        assert all(set(pair) != relation for pair in a["support_pairs"][:last, 0])
        assert all(set(pair) != relation for pair in b["support_pairs"][:last, 0])
        direct = a["direct"]
        assert np.all(a["targets"][direct] != b["targets"][direct])
        for c in (a, b):
            codes = c["item_codes"][0]
            cs = codes.shape[1]
            np.testing.assert_array_equal(
                c["support_inputs"][:, 0, 0, :cs], codes[c["support_pairs"][:, 0, 0]]
            )
            np.testing.assert_array_equal(
                c["support_inputs"][:, 0, 0, cs : 2 * cs],
                codes[c["support_pairs"][:, 0, 1]],
            )
            rank = np.argsort(c["orders"][0])
            qp = c["query_pairs"][:, 0]
            np.testing.assert_array_equal(
                c["targets"], (rank[qp[:, 0]] < rank[qp[:, 1]]).astype(int)
            )
    checks["eight_controlled_history_pairs"] = {"passed": True}
    margin, _ = solve(cpu, sigma)
    altered = copy.deepcopy(cpu.arrays)
    for key in ("targets", "signed_magnitudes", "probabilities", "orders", "retention"):
        altered[key] = np.zeros_like(altered[key])
    np.testing.assert_array_equal(margin, solve(EpisodeBatch(altered), sigma)[0])
    checks["comparator_no_latent_or_target_access"] = {"passed": True}
    return checks


def observation_preflight(spec):
    task = generator(spec)
    rng = np.random.default_rng(914002)
    sigma = spec["observation"]["sigma"]
    counts = {
        "presentations": 0,
        "sign_errors": 0,
        "relations": 0,
        "mean_sign_errors": 0,
        "weak_presentations": 0,
        "weak_sign_errors": 0,
    }
    for batch_id in range(8):
        cpu = attach_noise(
            with_learned(sample_episodes(task, rng, 32, validation=True)),
            1,
            920 + batch_id,
        )
        a = encode(cpu, "noisy", sigma).arrays
        signs = np.sign(a["signed_magnitudes"])
        wrong = signs * a["local_evidence"] < 0
        weak = a["retention"] == 0
        counts["presentations"] += wrong.size
        counts["sign_errors"] += int(wrong.sum())
        counts["weak_presentations"] += int(weak.sum())
        counts["weak_sign_errors"] += int(wrong[weak].sum())
        for s in range(32):
            pairs = np.sort(a["support_pairs"][:, s], axis=1)
            for pair in np.unique(pairs, axis=0):
                mask = np.all(pairs == pair, axis=1)
                assert mask.sum() == 4
                counts["relations"] += 1
                counts["mean_sign_errors"] += int(
                    np.mean((signs * a["local_evidence"])[mask, s]) < 0
                )
    return {
        **counts,
        "sigma": sigma,
        "single_sign_error_fraction": counts["sign_errors"] / counts["presentations"],
        "four_presentation_mean_sign_error_fraction": counts["mean_sign_errors"]
        / counts["relations"],
        "role": "Descriptive fixed-scale non-Liu check; not scale selection or a human estimate.",
    }


def qualify(spec, device="cpu", compiled=False):
    checks = input_checks(spec)
    config = copy.deepcopy(spec)
    if device == "cpu":
        config["architecture"]["hidden_size"] = 8
    cpu = attach_noise(
        prepare_shared(
            sample_episodes(generator(config), np.random.default_rng(915002), 2)
        ),
        1,
        915,
    )
    cpu.arrays["support_inputs"] = cpu.arrays["support_inputs"][:4]
    cpu.arrays["local_evidence"] = cpu.arrays["local_evidence"][:4]
    cpu.arrays["signed_magnitudes"] = cpu.arrays["signed_magnitudes"][:4]
    cpu.arrays["retention"] = cpu.arrays["retention"][:4]
    cpu.arrays["encoding_noise_0"] = cpu.arrays["encoding_noise_0"][:4]
    for arm in spec["seeds"]["conditions"]:
        first, a = make_model(config, 915003, "dual", device)
        assert a is not None
        second, b = copy.deepcopy(first), copy.deepcopy(a)
        prior_net, prior_local = copy.deepcopy(first), copy.deepcopy(a)
        actual = encode(cpu, arm, spec["observation"]["sigma"]).to(device)
        expected = reference_encoding(cpu, arm, spec["observation"]["sigma"]).to(device)
        eager = sequences(first, None)
        other = sequences(second, None, compiled=compiled)
        x, _, _ = rollout(first, a, eager, actual, None, 0)
        y, _, _ = rollout(second, b, other, expected, None, 0)
        checks[f"{arm}_forward"] = compare(x.logits, y.logits)
        checks[f"{arm}_state"] = compare(x.weights, y.weights)
        saved = x.weights.detach().clone()
        q1, _ = read_queries(first, a, eager[1], actual, x.weights, x.local_state)
        q2, _ = read_queries(first, a, eager[1], actual, x.weights, x.local_state)
        checks[f"{arm}_fixed_query"] = compare(q1, q2)
        assert torch.equal(saved, x.weights)
        if arm == "clean":
            prior = old_rollout(first, a, *old_sequences(first), actual)
            checks["original_continuous_logits"] = compare(x.logits, prior[0].logits)
            checks["original_continuous_weights"] = compare(x.weights, prior[0].weights)
        for net, local, seq, batch in [
            (first, a, eager, actual),
            (second, b, other, expected),
        ]:
            update(
                net,
                local,
                seq,
                batch,
                optimizer_for(net, local, config),
                config,
                None,
                0.0,
                0,
            )
        if arm == "clean":
            old_update(
                prior_net,
                prior_local,
                *old_sequences(prior_net),
                actual,
                optimizer_for(prior_net, prior_local, config),
                config,
                0.0,
            )
            for name, value in first.named_parameters():
                peer = dict(prior_net.named_parameters())[name]
                checks[f"original_update_{name}"] = compare(value, peer)
                if value.grad is None or peer.grad is None:
                    assert value.grad is None and peer.grad is None
                else:
                    checks[f"original_gradient_{name}"] = compare(value.grad, peer.grad)
            checks["original_local_update"] = compare(a.raw_gain, prior_local.raw_gain)
            checks["original_local_gradient"] = compare(
                a.raw_gain.grad, prior_local.raw_gain.grad
            )
        for name, value in first.state_dict().items():
            checks[f"{arm}_update_{name}"] = compare(value, second.state_dict()[name])
        checks[f"{arm}_local_update"] = compare(a.raw_gain, b.raw_gain)
    if not all(row["passed"] for row in checks.values()):
        raise RuntimeError("observation qualification failed")
    return checks


def run_qualification():
    from .locks import sources

    runtime = configure_execution()
    spec = specification()
    checks = qualify(spec, "cuda", True)
    result = {
        "passed": True,
        "runtime": runtime,
        "checks": checks,
        "observation_preflight": observation_preflight(spec),
        "sources": sources(),
        "liu_evaluated": False,
    }
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", result)
    return {
        "passed": True,
        "checks": len(checks),
        "observation_preflight": result["observation_preflight"],
    }
