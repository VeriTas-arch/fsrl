"""Engineering benchmark for legacy and batched frozen evaluation backends."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenEvaluationBackend,
    FrozenFastWeightEvaluator,
)
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import ExecutionProfile, configure_runtime
from fsrl.tasks.protocol_catalog import load_registered_protocol


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_once(
    evaluator: FrozenFastWeightEvaluator,
    schedules: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[torch.Tensor, tuple[dict[tuple[int, int], float], ...]]:
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    logits = evaluator.readout_logits(fast_weights, schedules)
    return fast_weights, logits


def _measure(
    evaluator: FrozenFastWeightEvaluator,
    schedules: tuple[tuple[tuple[int, int], ...], ...],
    *,
    warmups: int,
    repeats: int,
) -> tuple[list[float], torch.Tensor, tuple[dict[tuple[int, int], float], ...]]:
    result = None
    for _ in range(warmups):
        result = _run_once(evaluator, schedules)
    _synchronize()
    durations = []
    for _ in range(repeats):
        _synchronize()
        started = time.perf_counter()
        result = _run_once(evaluator, schedules)
        _synchronize()
        durations.append(time.perf_counter() - started)
    assert result is not None
    return durations, result[0], result[1]


def _logit_array(
    logits: tuple[dict[tuple[int, int], float], ...],
    schedules: tuple[tuple[tuple[int, int], ...], ...],
) -> np.ndarray:
    return np.asarray(
        [
            [row[pair] for pair in schedule]
            for row, schedule in zip(logits, schedules, strict=True)
        ],
        dtype=np.float64,
    )


def benchmark_frozen_evaluation(
    *,
    batch_size: int = 32,
    hidden_size: int = 200,
    warmups: int = 1,
    repeats: int = 3,
) -> dict:
    """Benchmark production-sized frozen evaluation without writing evidence."""

    if warmups < 1 or repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    profile = ExecutionProfile(
        device="cuda",
        cpu_threads=1,
        blas_threads=1,
        compile=True,
        compile_fullgraph=True,
        compile_mode="default",
        require_cuda=True,
    )
    runtime = configure_runtime(profile)
    torch.manual_seed(1701)
    protocol = load_registered_protocol("liu_v1")
    config = TrainConfig(
        bs=batch_size,
        hs=hidden_size,
        cs=8,
        nbcues_min=protocol.n_items,
        nbcues_max=protocol.n_items,
    )
    net = RetroModulRNN(config.to_model_dict(), device="cuda")
    net.eval()
    common = {
        "cue_seed": 5,
        "support_seed": 7,
        "cue_mode": "shared",
        "subject_encoding_mode": "none",
    }
    legacy = FrozenFastWeightEvaluator(net, config, protocol, **common)
    batched = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
        execution_profile=profile,
        **common,
    )
    ordered = tuple(
        oriented
        for first, second in combinations(range(protocol.n_items), 2)
        for oriented in ((first, second), (second, first))
    )
    schedules = tuple(ordered for _ in range(batch_size))
    legacy_times, legacy_weights, legacy_logits = _measure(
        legacy, schedules, warmups=warmups, repeats=repeats
    )
    batched_times, batched_weights, batched_logits = _measure(
        batched, schedules, warmups=warmups, repeats=repeats
    )
    weight_error = float(torch.max(torch.abs(legacy_weights - batched_weights)))
    logit_error = float(
        np.max(
            np.abs(
                _logit_array(legacy_logits, schedules)
                - _logit_array(batched_logits, schedules)
            )
        )
    )
    tolerance = 1e-6
    if max(weight_error, logit_error) > tolerance:
        raise RuntimeError(
            "batched evaluator exceeded the registered engineering parity tolerance"
        )
    legacy_mean = statistics.mean(legacy_times)
    batched_mean = statistics.mean(batched_times)
    return {
        "benchmark_schema_version": 1,
        "scope": "engineering_only_not_scientific_evidence",
        "runtime": runtime,
        "dimensions": {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "support_trials": protocol.support_trials,
            "oriented_queries": len(ordered),
        },
        "parity": {
            "tolerance": tolerance,
            "max_abs_fast_weight_error": weight_error,
            "max_abs_logit_error": logit_error,
            "passed": True,
        },
        "legacy_stepwise_seconds": {
            "samples": legacy_times,
            "mean": legacy_mean,
            "median": statistics.median(legacy_times),
        },
        "batched_sequence_seconds": {
            "samples": batched_times,
            "mean": batched_mean,
            "median": statistics.median(batched_times),
        },
        "mean_speedup": legacy_mean / batched_mean,
        "batched_execution": batched.evaluation_execution_record(),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args=None) -> None:
    parsed = parse_args(args)
    result = benchmark_frozen_evaluation(
        batch_size=parsed.batch_size,
        hidden_size=parsed.hidden_size,
        warmups=parsed.warmups,
        repeats=parsed.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if parsed.output is None:
        print(rendered, end="")
    else:
        write_json_exclusive(parsed.output, result)


if __name__ == "__main__":
    main()
