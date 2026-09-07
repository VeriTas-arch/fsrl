"""Matched generic evaluation and a competence-only development barrier."""

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.finite_state.evaluation import storage_arrays
from fsrl.experiments.finite_state.model import rollout, sequences
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .inputs import encode
from .locks import DEVELOPMENT, SCREEN, SOURCE, training_dir, validate_phase
from .protocol import PROTOCOL_SHA256, RUNS, specification


def loaded_model(seed, arm, source):
    runtime = configure_execution()
    if json_ready(runtime) != source["runtime"]:
        raise RuntimeError("evaluation runtime mismatch")
    base = "clean" if arm == "acute_noisy" else arm
    metadata = completed(training_dir(seed, base))
    backbone, local = make_model(specification(), seed, "dual", "cuda")
    assert local is not None
    for name, module, key in [
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ]:
        module.load_state_dict(
            torch.load(
                training_dir(seed, base) / f"{name}.pth",
                weights_only=True,
                map_location="cuda",
            )
        )
        if tensor_hashes(module) != metadata[key]:
            raise RuntimeError("loaded parameter mismatch")
        module.requires_grad_(False).eval()
    return backbone, local, sequences(backbone, None, compiled=True)


def observed(cpu, arm, spec, replay=0):
    return encode(
        cpu,
        "noisy" if arm == "acute_noisy" else arm,
        spec["observation"]["sigma"],
        replay=replay,
    )


def generic_arrays(backbone, local, seqs, source, split, arm, spec):
    collected = {}
    for name, ref in sorted(source["inputs"].items()):
        if not name.startswith(split + "-"):
            continue
        cpu = observed(load_input(ref), arm, spec)
        result, cost, writes = rollout(backbone, local, seqs, cpu.to("cuda"), None, 0)
        n = len(cost)
        values = {
            "margins": margin_bundle(result.logits, n)["logits"],
            "global_margins": margin_bundle(result.global_logits, n)["logits"],
            "signs": (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T,
            "learned": cpu.arrays["learned"],
            "cost": cost.cpu().numpy().astype(float),
            "total_write": writes.sum(0).cpu().numpy().astype(float),
            "episode_indices": cpu.arrays["episode_indices"],
            **storage_arrays(result.weights, backbone.alpha, None),
        }
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in collected.items()}


def generic_run(seed, arm, split, source):
    directory = RUNS / "generic" / split / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs = loaded_model(seed, arm, source)
    identity = {
        "seed": seed,
        "arm": arm,
        "split": split,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="observation_uncertainty_v1",
            execution_id=f"{split}-{seed}-{arm}",
            producer=identity,
            resolved_config={
                "evaluation": spec["evaluation"],
                "observation": spec["observation"],
            },
        ),
        torch.no_grad(),
    ):
        arrays = generic_arrays(backbone, local, seqs, source, split, arm, spec)
        summary = summarize_generic(arrays, spec, seed)
        global_arrays = {**arrays, "margins": arrays["global_margins"]}
        summary["global"] = summarize_generic(global_arrays, spec, seed)
        arrays["global_ce"] = global_arrays["ce"]
        summary["alpha_abs_mean"] = float(backbone.alpha.abs().mean())
        summary["local_gain"] = float(local.gain)
        result = {**identity, **summary}
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def screen():
    barrier = validate_phase(DEVELOPMENT)
    source = load_json(SOURCE)
    spec = specification()
    rows = {
        arm: generic_run(spec["seeds"]["development"], arm, "development", source)
        for arm in spec["seeds"]["conditions"]
    }
    eligible = all(
        row["competence"] and row["global"]["competence"] for row in rows.values()
    )
    result = {
        "eligible": eligible,
        "models": rows,
        "runs": barrier["runs"],
        "files": [reference(DEVELOPMENT)],
        "liu_evaluated": False,
        "sigma": spec["observation"]["sigma"],
    }
    write_json_exclusive(SCREEN, json_ready(result))
    return {
        "eligible": eligible,
        "generic_ce": {arm: row["ce"] for arm, row in rows.items()},
    }
