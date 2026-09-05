"""Locked-model evaluation with unchanged task inputs and descriptive behavior."""

from __future__ import annotations

import time

import numpy as np
import torch

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.relational_transport import constructive_metrics
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.evaluation.local_access import apply_blockwise_route
from fsrl.experiments.local_fidelity.evidence_access_pilot import blockwise_derangements
from fsrl.experiments.local_fidelity.trace_pilot import shuffled_pair_indices
from fsrl.experiments.training_strategy.behavior import evaluate_behavior
from fsrl.experiments.training_strategy.estimands import estimate, query_endpoints
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    summarize_endpoints,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .data import ModelBatch, generic_batch, liu_batch
from .history import score_history
from .locks import (
    ARTIFACT_LOCK,
    reference,
    require_pushed_clean,
    validate_artifacts,
    validate_complete,
)
from .model import make_model
from .protocol import PROTOCOL_SHA256, RUN_ROOT, run_directory, specification
from .training import compiled, physical_parameters, runtime


def evaluate_generic(runner, spec: dict) -> dict:
    device = next(runner.parameters()).device.type
    episodes = validation_episodes(spec)
    margins = {
        name: np.empty((len(episodes), 28))
        for name in ("intact", "local_off", "global_off")
    }
    signs = np.empty((len(episodes), 28))
    learned = np.empty_like(signs, dtype=bool)
    groups = {}
    for length, indices in validation_groups(episodes).items():
        batch = generic_batch(tuple(episodes[index] for index in indices))
        with torch.no_grad():
            values = runner(*batch.tensors(device))
        for name, value in zip(margins, values[:3], strict=True):
            margins[name][indices] = value.cpu().numpy()
        signs[indices] = 2 * batch.arrays["targets"] - 1
        learned[indices] = batch.arrays["learned"]
        groups[str(length)] = {
            "episode_indices": np.asarray(indices),
            "inputs": batch.arrays,
            "fingerprint": batch.fingerprint(),
        }
    endpoints = {
        name: query_endpoints(
            values[..., None],
            signs[..., None],
            {"learned": learned, "nonlearned": ~learned},
            temperature=1.0,
        )
        for name, values in margins.items()
    }
    return {
        "endpoints": endpoints,
        "margins": margins,
        "signs": signs,
        "learned": learned,
        "groups": groups,
    }


def read_liu_controls(model, runner, batch, settings, protocol) -> dict:
    device = next(model.parameters()).device.type
    arguments = batch.tensors(device)
    with torch.no_grad():
        runner(*arguments)  # Compilation/warmup, not extra memory presentations.
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        values = runner(*arguments)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        intact, global_margin, local_margin, w, trace = (
            value.cpu().numpy().astype(np.float64) for value in values
        )
    subjects = intact.shape[0]
    query_map = shuffled_pair_indices(
        subjects, protocol.n_items, settings["query_shuffle_seed"]
    )
    routing = blockwise_derangements(
        subjects,
        protocol.support_blocks,
        len(protocol.support_pairs_higher_lower),
        settings["evidence_shuffle_seed"],
    )
    local_values = apply_blockwise_route(batch.arrays["local_evidence"].T, routing).T
    shuffled_batch = ModelBatch({**batch.arrays, "local_evidence": local_values})
    with torch.no_grad():
        shuffled = runner(*shuffled_batch.tensors(device))
    np.testing.assert_array_equal(shuffled[1].cpu().numpy(), global_margin)
    bundles = {
        "intact": {"logits": intact},
        "local_off": {"logits": global_margin},
        "global_off": {"logits": local_margin},
        "query_shuffle": {
            "logits": global_margin
            + np.take_along_axis(local_margin, query_map, axis=1)
        },
        "evidence_shuffle": {"logits": shuffled[0].cpu().numpy().astype(np.float64)},
    }
    return {
        "bundles": bundles,
        "w": w,
        "local_state": trace,
        "query_routing": query_map,
        "evidence_routing": routing,
        "shuffled_local_evidence": local_values,
        "inference_seconds": seconds,
        "inference_seconds_per_episode": seconds / subjects,
    }


def retention_by_relation(batch: ModelBatch, protocol) -> np.ndarray:
    values = []
    for subject, pairs in enumerate(batch.arrays["support_pairs"]):
        row = []
        for relation in protocol.support_pairs_higher_lower:
            selected = np.all(np.sort(pairs, axis=1) == sorted(relation), axis=1)
            retained = batch.arrays["retention"][:, subject][selected]
            if len(retained) != protocol.support_blocks or not np.all(
                retained == retained[0]
            ):
                raise RuntimeError(
                    "stable relation retention changed across repetitions"
                )
            row.append(retained[0])
        values.append(row)
    return np.asarray(values, dtype=bool)


def history_audit(model, batch, bundles, seed, spec) -> dict:
    cues = batch.arrays["support_cues"].transpose(1, 0, 2)
    query = batch.arrays["query_cues"][:, ::2]
    d = model.cue_size
    parameters = physical_parameters(model)
    history = score_history(
        cues[..., :d] - cues[..., d:],
        batch.arrays["signed"].T,
        batch.arrays["retention"].T,
        query[..., :d] - query[..., d:],
        eta=parameters["eta"],
        gain=parameters["gamma_G"],
        epsilon=model.epsilon,
    )
    observed = bundles["local_off"]["logits"][:, ::2]
    np.testing.assert_allclose(history["global_margin"], observed, atol=1e-5, rtol=1e-4)
    support_pairs = batch.arrays["support_pairs"]
    query_pairs = batch.arrays["query_pairs"][:, ::2]
    remote = ~(
        support_pairs[:, :, None, :, None] == query_pairs[:, None, :, None, :]
    ).any(axis=(-2, -1))
    summaries = {}
    for key in ("sensitivity", "direct_sensitivity", "history_sensitivity"):
        effect = np.abs(history[key] * batch.arrays["signed"].T[:, :, None])
        for scope, mask in (("all", np.ones_like(remote)), ("remote", remote)):
            values = np.where(mask, effect, 0.0).sum(axis=(1, 2)) / mask.sum(
                axis=(1, 2)
            )
            summaries[f"{scope}_{key}"] = estimate(
                values, seed=85000 + seed, statistics=spec["statistics"]
            )
    return {
        "arrays": {**history, "remote_mask": remote},
        "summary": summaries,
        "float64_to_float32_max_abs_error": float(
            np.max(np.abs(history["global_margin"] - observed))
        ),
    }


def condition_analysis(model, runner, seed, spec) -> tuple[dict, dict]:
    generic = evaluate_generic(runner, spec)
    protocol, batch = liu_batch(spec)
    liu = read_liu_controls(model, runner, batch, spec["evaluation"]["liu"], protocol)
    retention = retention_by_relation(batch, protocol)
    endpoints = liu_endpoints(
        liu["bundles"], retention, protocol, spec["evaluation"]["liu"]["temperature"]
    )
    behavior = evaluate_behavior(liu["bundles"]["intact"], protocol, seed, spec)
    history = history_audit(model, batch, liu["bundles"], seed, spec)
    geometry = build_complete_graph_geometry(protocol)
    counts = bootstrap_counts(
        np.random.default_rng(85000 + seed),
        spec["statistics"]["samples"],
        retention.shape[0],
    )
    fields = {
        name: (row["logits"][:, ::2] - row["logits"][:, 1::2]) / 2
        for name, row in liu["bundles"].items()
    }
    geometric = constructive_metrics(
        fields["intact"],
        fields["local_off"],
        geometry,
        counts,
        spec["statistics"]["interval"],
    )
    raw_endpoints = {"generic": generic["endpoints"], "liu": endpoints}
    summary = {
        domain: {
            name: summarize_endpoints(
                value, (85000 if domain == "liu" else 86000) + seed, spec["statistics"]
            )
            for name, value in values.items()
        }
        for domain, values in raw_endpoints.items()
    }
    result = {
        "raw_endpoints": raw_endpoints,
        "endpoints": summary,
        "behavior": behavior["record"],
        "history": {key: value for key, value in history.items() if key != "arrays"},
        "geometry": geometric,
        "input_fingerprint": batch.fingerprint(),
        "generic_fingerprints": {
            key: value["fingerprint"] for key, value in generic["groups"].items()
        },
        "inference_seconds": liu["inference_seconds"],
        "inference_seconds_per_episode": liu["inference_seconds_per_episode"],
    }
    raw = {
        "generic": generic,
        "liu": {**liu, "inputs": batch.arrays, "retention": retention},
        "history": history["arrays"],
    }
    return result, {
        "arrays": flatten_arrays(raw),
        "sampled_behavior": behavior["sampled_behavior"],
    }


def evaluation_directory(seed, condition):
    directory = run_directory(seed, condition)
    return RUN_ROOT / "evaluation" / directory.parent.name / directory.name


def validate_evaluation(seed, condition, lock) -> dict:
    directory = evaluation_directory(seed, condition)
    validate_complete(directory)
    result = load_json(directory / "result.json")
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "source_commit": lock["source_commit"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("evaluation identity differs from locked training")
    return result


def evaluate_all() -> dict:
    lock = validate_artifacts()  # Must precede runtime, task generation and rollout.
    admission_commit = require_pushed_clean()
    execution = runtime()
    spec = specification()
    completed = []
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["conditions"]:
            directory = evaluation_directory(seed, condition)
            if directory.exists():
                validate_evaluation(seed, condition, lock)
                completed.append(f"{seed}/{condition}")
                continue
            metadata = lock["runs"][f"{seed}/{condition}"]["config"]
            model = make_model(condition, spec, "cuda")
            model.load_state_dict(
                torch.load(
                    run_directory(seed, condition) / "model.pth",
                    map_location="cuda",
                    weights_only=True,
                )
            )
            if tensor_hashes(model) != metadata["final_parameters"]:
                raise RuntimeError(
                    "checkpoint tensors differ from final training metadata"
                )
            model.requires_grad_(False).eval()
            with ProspectiveRun.start(
                directory,
                workflow_id="minimal_relational_learner_v1",
                execution_id=f"evaluate-{seed}-{condition}",
                producer={
                    "module": __name__,
                    "admission_commit": admission_commit,
                    "artifact_lock": reference(ARTIFACT_LOCK),
                },
                resolved_config={
                    "evaluation": spec["evaluation"],
                    "statistics": spec["statistics"],
                    "runtime": execution,
                },
            ):
                result, raw = condition_analysis(model, compiled(model), seed, spec)
                write_arrays(directory / "raw.npz", raw["arrays"])
                write_json_exclusive(
                    directory / "behavior.json", json_ready(raw["sampled_behavior"])
                )
                result.update(
                    {
                        "seed": seed,
                        "condition": condition,
                        "source_commit": lock["source_commit"],
                        "protocol_sha256": PROTOCOL_SHA256,
                        "artifact_lock": reference(ARTIFACT_LOCK),
                        "runtime": execution,
                        "cost": metadata["cost"],
                        "parameters": metadata["physical_parameters"],
                        "raw_arrays": reference(directory / "raw.npz"),
                        "sampled_behavior": reference(directory / "behavior.json"),
                    }
                )
                write_json_exclusive(directory / "result.json", json_ready(result))
            validate_evaluation(seed, condition, lock)
            completed.append(f"{seed}/{condition}")
            print("Evaluation complete:", completed[-1], flush=True)
    return {"complete_evaluations": completed}
