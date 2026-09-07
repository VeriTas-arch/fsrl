"""Locked generic competence and selection, including paired acute controls."""

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .inputs import rounding_seed
from .locks import DEVELOPMENT, SELECTION, training_dir, validate_phase
from .model import rollout, sequences
from .protocol import PROTOCOL_SHA256, RUNS, specification


def loaded_model(seed, arm, source):
    runtime = configure_execution()
    if json_ready(runtime) != source["runtime"]:
        raise RuntimeError("evaluation runtime mismatch")
    base_arm = arm.removeprefix("acute_")
    metadata = completed(training_dir(seed, base_arm))
    backbone, local = make_model(specification(), seed, "dual", "cuda")
    assert local is not None
    for name, module, key in [
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ]:
        module.load_state_dict(
            torch.load(
                training_dir(seed, base_arm) / f"{name}.pth",
                weights_only=True,
                map_location="cuda",
            )
        )
        if tensor_hashes(module) != metadata[key]:
            raise RuntimeError("loaded parameters differ")
        module.requires_grad_(False).eval()
    states = (
        load_json(SELECTION)["selected_K"]
        if arm.startswith("acute_")
        else metadata["states"]
    )
    return backbone, local, sequences(backbone, states, compiled=True), states


def storage_arrays(weights, alpha, states):
    p = weights.detach().cpu().numpy()
    effective = (weights * alpha).detach().cpu().numpy()
    result = {
        "P_abs_mean": np.abs(p).mean((1, 2)),
        "P_abs_max": np.abs(p).max((1, 2)),
        "A_abs_mean": np.abs(effective).mean((1, 2)),
        "A_abs_max": np.abs(effective).max((1, 2)),
        "zero_fraction": (p == 0).mean((1, 2)),
        "boundary_fraction": (np.abs(p) == 50).mean((1, 2)),
    }
    if states is not None:
        index = p / (100.0 / (states - 1))
        if not np.array_equal(index, np.rint(index)) or np.abs(p).max() > 50:
            raise RuntimeError("stored fast state escaped finite grid")
        result["used_states"] = np.asarray([np.unique(row).size for row in index])
        result["effective_spacing_mean"] = np.full(
            len(p), float(alpha.detach().abs().mean()) * 100.0 / (states - 1)
        )
    return result


def generic_arrays(backbone, local, seqs, states, source, split):
    collected = {}
    for batch_id, (name, record) in enumerate(sorted(source["inputs"].items())):
        if not name.startswith(split + "-"):
            continue
        cpu = load_input(record)
        draw = rounding_seed(1 if split == "development" else 2, batch_id=batch_id)
        result, cost, writes = rollout(
            backbone, local, seqs, cpu.to("cuda"), states, draw
        )
        subjects = len(cost)
        row = {
            "margins": margin_bundle(result.logits, subjects)["logits"],
            "global_margins": margin_bundle(result.global_logits, subjects)["logits"],
            "signs": (2 * cpu.arrays["targets"] - 1).reshape(-1, subjects).T,
            "learned": cpu.arrays["learned"],
            "cost": cost.cpu().numpy().astype(float),
            "total_write": writes.sum(0).cpu().numpy().astype(float),
            "episode_indices": cpu.arrays["episode_indices"],
            **storage_arrays(result.weights, backbone.alpha, states),
        }
        for key, value in row.items():
            collected.setdefault(key, []).append(value)
    return {key: np.concatenate(values) for key, values in collected.items()}


def generic_run(seed, arm, split, source):
    directory = RUNS / "generic" / split / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs, states = loaded_model(seed, arm, source)
    identity = {
        "seed": seed,
        "arm": arm,
        "states": states,
        "split": split,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="finite_state_memory_v1",
            execution_id=f"{split}-{seed}-{arm}",
            producer=identity,
            resolved_config={
                "evaluation": spec["evaluation"],
                "rounding": spec["rounding"],
            },
        ),
        torch.no_grad(),
    ):
        arrays = generic_arrays(backbone, local, seqs, states, source, split)
        summary = summarize_generic(arrays, spec, seed)
        global_arrays = {**arrays, "margins": arrays["global_margins"]}
        global_summary = summarize_generic(global_arrays, spec, seed)
        arrays["global_ce"] = global_arrays["ce"]
        summary["global"] = global_summary
        summary["storage"] = {
            key: float(arrays[key].mean())
            for key in storage_arrays(
                backbone.initial_fast_weights(1), backbone.alpha, states
            )
        }
        summary["alpha_abs_mean"] = float(backbone.alpha.abs().mean())
        summary["local_gain"] = float(local.gain)
        result = {**identity, **summary}
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def arrays_at(directory):
    with np.load(directory / "raw.npz", allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def select():
    barrier = validate_phase(DEVELOPMENT)
    from .locks import SOURCE

    source, spec = load_json(SOURCE), specification()
    seed = spec["seeds"]["development"]
    baseline = generic_run(seed, "shared", "development", source)
    baseline_arrays = arrays_at(RUNS / "generic/development" / str(seed) / "shared")
    curve = []
    for states in spec["storage"]["state_candidates"]:
        arm = f"K-{states}"
        row = generic_run(seed, arm, "development", source)
        raw = arrays_at(RUNS / "generic/development" / str(seed) / arm)
        delta = estimate(
            raw["ce"] - baseline_arrays["ce"],
            seed=spec["statistics"]["seed_offset"] + seed,
            statistics=spec["statistics"],
        )
        upper = delta["bootstrap"]["upper"]
        eligible = (
            baseline["competence"]
            and row["competence"]
            and upper is not None
            and upper <= spec["selection"]["generic_ce_allowance_nats"]
        )
        curve.append(
            {"K": states, "eligible": eligible, "paired_ce_increase": delta, **row}
        )
    candidates = [row["K"] for row in curve if row["eligible"]]
    selected = min(candidates) if candidates else None
    result = {
        "selected_K": selected,
        "baseline": baseline,
        "curve": curve,
        "runs": barrier["runs"],
        "files": [reference(DEVELOPMENT)],
        "liu_evaluated": False,
        "outcome": "selected_on_generic_only" if selected else "no_feasible_candidate",
    }
    write_json_exclusive(SELECTION, result)
    return {"selected_K": selected, "outcome": result["outcome"], "curve": curve}
