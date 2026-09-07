"""Read archived weights independently of the evaluation observation operator."""

import numpy as np
import torch

from fsrl.experiments.finite_state.liu import primary_analysis, primary_rollouts
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.observation_uncertainty.evaluation import generic_arrays, observed
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .protocol import (
    LOCK,
    RUNS,
    execution_identity,
    inherited,
    specification,
    validate_lock,
)


def arrays(ref):
    with np.load(verify_reference(ref), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_model(seed, training, lock, spec):
    if json_ready(configure_execution()) != lock["runtime"]:
        raise RuntimeError("cross-evaluation runtime differs")
    backbone, local = make_model(spec, seed, "dual", "cuda")
    assert local is not None
    files = lock["models"][f"{seed}/{training}"]
    metadata = load_json(verify_reference(files["result.json"]))
    parent_metadata = inherited()["parent_model_lock"]["runs"][f"{seed}/{training}"][
        "metadata"
    ]
    if metadata != parent_metadata:
        raise RuntimeError("archived model metadata differs from model lock")
    for name, module, key in (
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ):
        module.load_state_dict(
            torch.load(
                verify_reference(files[f"{name}.pth"]),
                weights_only=True,
                map_location="cuda",
            )
        )
        if tensor_hashes(module) != metadata[key]:
            raise RuntimeError("loaded weights differ")
        module.requires_grad_(False).eval()
    return backbone, local, sequences(backbone, None, compiled=True)


def collect(seed, training, observation, lock):
    spec = inherited()["parent_protocol"]
    backbone, local, seqs = load_model(seed, training, lock, spec)
    before = tensor_hashes(backbone), tensor_hashes(local)
    with torch.no_grad():
        # Preserve original generic-before-Liu compiled shape specialization.
        generic_raw = generic_arrays(
            backbone, local, seqs, lock, "test", observation, spec
        )
        generic = summarize_generic(generic_raw, spec, seed)
        global_raw = {**generic_raw, "margins": generic_raw["global_margins"]}
        generic["global"] = summarize_generic(global_raw, spec, seed)
        generic_raw["global_ce"] = global_raw["ce"]
        cpu = observed(load_input(lock["inputs"]["liu-8"]), observation, spec)
        task = size_protocol(spec, 8)
        raw = primary_rollouts(backbone, local, seqs, None, cpu, task, spec)
        liu, analysis, sampled = primary_analysis(raw, cpu, task, spec, seed, generic)
    if before != (tensor_hashes(backbone), tensor_hashes(local)):
        raise RuntimeError("evaluation changed slow parameters")
    return (
        {"generic": generic, "liu": liu},
        {"generic": generic_raw, "liu": flatten_arrays({**raw, **analysis})},
        sampled,
    )


def freeze():
    lock = execution_identity()
    directory = RUNS / "qualification"
    with ProspectiveRun.start(
        directory,
        workflow_id="observation_crossover_v1",
        execution_id="known-A0-replay",
        producer=lock,
        resolved_config=specification()["execution"],
    ):
        _, raw, _ = collect(2432, "clean", "clean", lock)
        checks = []
        for phase in ("generic", "liu"):
            old = arrays(lock["previous_cells"]["2432"]["A0"][phase])
            for key, value in raw[phase].items():
                if not np.array_equal(value, old[key], equal_nan=True):
                    raise RuntimeError(f"known A0 replay differs: {phase}/{key}")
                checks.append(f"{phase}/{key}")
        result = {
            "passed": True,
            "exact_array_checks": checks,
            "seed": 2432,
            "new_C0_evaluated": False,
        }
        write_json_exclusive(directory / "result.json", result)
    lock["qualification"] = reference(directory / "result.json")
    write_json_exclusive(LOCK, lock)
    return {
        "passed": True,
        "exact_array_checks": len(checks),
        "new_C0_evaluated": False,
    }


def evaluate():
    lock, spec = validate_lock(), specification()
    for seed in spec["design"]["seeds"]:
        directory = RUNS / str(seed) / "C0"
        if directory.exists():
            completed(directory)
            continue
        producer = {
            "execution_lock": reference(LOCK),
            "seed": seed,
            **spec["design"]["cells"]["C0"],
        }
        with ProspectiveRun.start(
            directory,
            workflow_id="observation_crossover_v1",
            execution_id=f"{seed}-C0",
            producer=producer,
            resolved_config=spec["design"],
        ):
            result, raw, sampled = collect(seed, "noisy", "clean", lock)
            write_arrays(directory / "raw.npz", flatten_arrays(raw))
            write_json_exclusive(directory / "behavior.json", json_ready(sampled))
            write_json_exclusive(directory / "result.json", json_ready(result))
        print(
            {
                "seed": seed,
                "core": {
                    k: v["core_passed"] for k, v in result["liu"]["routes"].items()
                },
            },
            flush=True,
        )
    return {"completed": spec["design"]["seeds"], "trained_models": 0}
