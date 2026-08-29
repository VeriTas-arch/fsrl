"""Engineering-only benchmark for the prospective meta-training hot path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import (
    ExecutionProfile,
    begin_compiled_iteration,
    configure_runtime,
    default_device,
)
from fsrl.training.backbone import (
    OPTIMIZED_TRAINING_PROFILE,
    MetaTrainConfig,
    apply_meta_batch_update,
    compile_meta_sequence,
    make_model_and_tasks,
)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _cuda_memory(device: str, baseline_allocated: int | None) -> dict:
    if device != "cuda":
        return {
            "available": False,
            "baseline_allocated_bytes": None,
            "peak_allocated_bytes": None,
            "incremental_peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    assert baseline_allocated is not None
    peak_allocated = torch.cuda.max_memory_allocated()
    return {
        "available": True,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def benchmark_training_hot_path(
    *,
    batch_size: int = 32,
    hidden_size: int = 200,
    cue_size: int = 15,
    min_edges: int = 7,
    max_edges: int = 10,
    support_blocks: int = 4,
    warmups: int = 2,
    repeats: int = 5,
    seed: int = 1701,
    device: str | None = None,
    compile_model: bool | None = None,
) -> dict:
    """Measure one optimizer-step path without producing scientific evidence."""

    if warmups < 1 or repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    if min(batch_size, hidden_size, cue_size, support_blocks) < 1:
        raise ValueError("benchmark dimensions must be positive")
    if min_edges < 1 or max_edges < min_edges:
        raise ValueError("edge bounds are invalid")
    resolved_device = device or default_device()
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {resolved_device}")
    resolved_compile = (
        resolved_device == "cuda" if compile_model is None else compile_model
    )
    profile = ExecutionProfile(
        device=resolved_device,
        compile=resolved_compile,
        compile_mode=OPTIMIZED_TRAINING_PROFILE.compile_mode,
        require_cuda=resolved_device == "cuda",
    )
    runtime = configure_runtime(profile)
    training_config = MetaTrainConfig(
        seed=seed,
        outer_steps=warmups + repeats,
        batch_size=batch_size,
        hidden_size=hidden_size,
        cue_size=cue_size,
        min_edges=min_edges,
        max_edges=max_edges,
        support_blocks=support_blocks,
        save_every=0,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_config, net, task_generator = make_model_and_tasks(
        training_config, device=resolved_device
    )
    sequence_runner = (
        compile_meta_sequence(net, profile)
        if profile.compile
        else RecurrentSequence(net)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=training_config.learning_rate)

    for _ in range(warmups):
        begin_compiled_iteration(profile)
        stats = apply_meta_batch_update(
            training_config,
            model_config,
            net,
            net,
            sequence_runner,
            task_generator,
            rng,
            optimizer,
        )
        stats.materialize_metrics()

    _synchronize(resolved_device)
    if resolved_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated: int | None = torch.cuda.memory_allocated()
    else:
        baseline_allocated = None

    durations = []
    edge_counts = []
    query_counts = []
    for _ in range(repeats):
        _synchronize(resolved_device)
        started = time.perf_counter()
        begin_compiled_iteration(profile)
        stats = apply_meta_batch_update(
            training_config,
            model_config,
            net,
            net,
            sequence_runner,
            task_generator,
            rng,
            optimizer,
        )
        metrics = stats.materialize_metrics()
        _synchronize(resolved_device)
        durations.append(time.perf_counter() - started)
        edge_counts.append(metrics.n_edges)
        query_counts.append(stats.query_count)

    total_seconds = sum(durations)
    median_seconds = statistics.median(durations)
    return {
        "benchmark_schema_version": 1,
        "scope": "engineering_only_not_scientific_evidence",
        "benchmark": "prospective_meta_training_optimizer_step",
        "runtime": runtime,
        "training_config": asdict(training_config),
        "measurement": {
            "warmups": warmups,
            "repeats": repeats,
            "seconds": {
                "samples": durations,
                "mean": statistics.mean(durations),
                "median": median_seconds,
            },
            "sampled_edge_counts": edge_counts,
            "query_decisions": sum(query_counts),
            "shape_policy": (
                "variable edge counts; timed samples can include compilation of "
                "previously unseen sequence shapes"
                if profile.compile and min_edges != max_edges
                else "fixed edge count"
            ),
        },
        "throughput": {
            "optimizer_steps_per_second": repeats / total_seconds,
            "episodes_per_second": repeats * batch_size / total_seconds,
            "query_decisions_per_second": sum(query_counts) / total_seconds,
            "median_sample_optimizer_steps_per_second": 1.0 / median_seconds,
            "median_sample_episodes_per_second": batch_size / median_seconds,
        },
        "host_synchronization": {
            "pre_backward_metric_materializations_per_step": 0,
            "post_optimizer_metric_materializations_per_step": 1,
            "batched_metric_values_per_materialization": 4,
            "explicit_timing_synchronizations_per_sample": (
                2 if resolved_device == "cuda" else 0
            ),
        },
        "memory": _cuda_memory(resolved_device, baseline_allocated),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--cue-size", type=int, default=15)
    parser.add_argument("--min-edges", type=int, default=7)
    parser.add_argument("--max-edges", type=int, default=10)
    parser.add_argument("--support-blocks", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device())
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: enabled on CUDA and disabled on CPU",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args=None) -> None:
    parsed = parse_args(args)
    result = benchmark_training_hot_path(
        batch_size=parsed.batch_size,
        hidden_size=parsed.hidden_size,
        cue_size=parsed.cue_size,
        min_edges=parsed.min_edges,
        max_edges=parsed.max_edges,
        support_blocks=parsed.support_blocks,
        warmups=parsed.warmups,
        repeats=parsed.repeats,
        seed=parsed.seed,
        device=parsed.device,
        compile_model=parsed.compile_model,
    )
    if parsed.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        write_json_exclusive(parsed.output, result)


if __name__ == "__main__":
    main()
