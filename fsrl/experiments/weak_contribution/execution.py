"""One new joint support rollout per frozen model, with archived baseline parity."""

import gc

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.model import forward_batch, make_model
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.evaluation import flatten_arrays, write_arrays
from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from . import computation, locks, measurement
from .protocol import LOCK, PROTOCOL_SHA256, RUNS, parent, parent_spec, specification


def arrays_at(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_model(seed, arm):
    directory = parent() / f"artifacts/training/{seed}/{arm}"
    metadata = load_json(directory / "result.json")
    backbone, local = make_model(parent_spec(), seed, "dual", "cuda")
    assert local is not None
    for name, model, key in (
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ):
        model.load_state_dict(
            torch.load(
                directory / f"{name}.pth", map_location="cuda", weights_only=True
            )
        )
        if tensor_hashes(model) != metadata[key]:
            raise RuntimeError("archived parameters differ from training record")
        model.requires_grad_(False).eval()
    return backbone, local


def completed(directory):
    if (
        not validate_run_manifest(directory / "run.json")["passed"]
        or load_json(directory / "run.json")["lifecycle_state"] != "complete"
    ):
        raise RuntimeError(f"incomplete or invalid run: {directory}")
    return load_json(directory / "result.json")


def calculate(backbone, local, sequence, cpu, protocol, archive):
    subjects = specification()["design"]["subjects"]
    joint_cpu = computation.remove_all_weak(cpu)
    checks = {
        "inputs": computation.input_checks(
            cpu, joint_cpu, protocol.support_pairs_higher_lower
        )
    }
    intact = forward_batch(backbone, local, sequence, cpu.to("cuda"))
    raw = {
        "intact_global": archive["N8__bundles__local_off__logits"],
        "intact_complete": archive["N8__bundles__intact__logits"],
    }
    raw["replay_global"] = margin_bundle(intact.global_logits, subjects)["logits"]
    raw["replay_complete"] = margin_bundle(intact.logits, subjects)["logits"]
    checks["archived_global"] = computation.compare(
        raw["replay_global"], raw["intact_global"]
    )
    checks["archived_complete"] = computation.compare(
        raw["replay_complete"], raw["intact_complete"]
    )
    raw["local_intact"] = computation.local_margin(local, cpu)
    checks["archived_local"] = computation.compare(
        raw["local_intact"], raw["intact_complete"] - raw["intact_global"]
    )
    raw["single_global"], raw["single_local"] = computation.reconstruct(
        local, cpu, protocol.support_pairs_higher_lower, archive["N8__removed"]
    )
    joint = forward_batch(backbone, local, sequence, joint_cpu.to("cuda"))
    assert intact.local_state is not None and joint.local_state is not None
    checks["unchanged_local_state"] = computation.compare(
        intact.local_state.cpu().numpy(), joint.local_state.cpu().numpy()
    )
    raw["joint_global"] = margin_bundle(joint.global_logits, subjects)["logits"]
    raw["joint_complete"] = margin_bundle(joint.logits, subjects)["logits"]
    checks["joint_local_correction"] = computation.compare(
        raw["joint_complete"] - raw["joint_global"], raw["local_intact"]
    )
    no_weak = np.asarray(checks["inputs"]["zero_weak_subjects"], dtype=int)
    # For these subjects the intervention is exactly the identity, including arithmetic.
    for route in ("global", "complete"):
        np.testing.assert_array_equal(
            raw[f"joint_{route}"][no_weak], raw[f"replay_{route}"][no_weak]
        )
    return raw, checks


def run_cell(seed, arm, cpu, protocol, lock):
    directory = RUNS / str(seed) / arm
    if directory.exists():
        return completed(directory)
    backbone, local = load_model(seed, arm)
    before = {"backbone": tensor_hashes(backbone), "local": tensor_hashes(local)}
    sequence = compile_module(RecurrentSequence(backbone), PROFILE)
    identity = {
        "seed": seed,
        "arm": arm,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": lock["source_commit"],
        "source_repair": lock.get("source_repair"),
    }
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="weak_evidence_contribution_v1",
            execution_id=f"{seed}-{arm}",
            producer=identity,
            resolved_config={
                "protocol": specification(),
                "runtime": lock["runtime"],
                "source_lock": str(LOCK),
            },
        ),
        torch.no_grad(),
    ):
        archive = arrays_at(parent() / f"artifacts/liu/{seed}/{arm}/raw.npz")
        raw, checks = calculate(backbone, local, sequence, cpu, protocol, archive)
        values, structures = measurement.endpoints(raw, cpu, protocol)
        summary = measurement.summarize_cell(values, structures, seed)
        if before != {
            "backbone": tensor_hashes(backbone),
            "local": tensor_hashes(local),
        }:
            raise RuntimeError("frozen parameters changed")
        result = {
            **identity,
            "integrity": checks,
            "parameters_unchanged": True,
            **summary,
        }
        write_arrays(
            directory / "raw.npz",
            flatten_arrays(
                {"margins": raw, "endpoints": values, "structure": structures}
            ),
        )
        write_json_exclusive(directory / "result.json", result)
    del backbone, local, sequence
    gc.collect()
    torch.cuda.empty_cache()
    return {"seed": seed, "arm": arm, "passed": True}


def check_runtime(runtime, expected):
    # Version metadata is descriptive; actual numerical compatibility is checked per cell.
    for key in (
        "profile",
        "torch_intraop_threads",
        "torch_interop_threads",
        "blas_thread_limit",
        "cuda_available",
        "compiler_threads",
        "matmul_allow_tf32",
        "triton_mix_order_reduction",
    ):
        if runtime[key] != expected[key]:
            raise RuntimeError(f"runtime contract changed: {key}")


def evaluate():
    lock = locks.validate()
    runtime = configure_execution()
    check_runtime(runtime, lock["runtime"])
    cpu = EpisodeBatch(arrays_at(parent() / "artifacts/inputs/liu-8.npz"))
    protocol = size_protocol(parent_spec(), 8)
    for seed in specification()["design"]["seeds"]:
        for arm in specification()["design"]["arms"]:
            print(
                run_cell(seed, arm, cpu, protocol, {**lock, "runtime": runtime}),
                flush=True,
            )
    return {"completed_cells": 6}
