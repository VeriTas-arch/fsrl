"""Exact reconstruction and identity checks for the confirmed local ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from itertools import combinations
from typing import cast

import numpy as np

from fsrl.tasks.protocol import SupportTrial

from .frozen_fast_weight import FrozenFastWeightEvaluator


def edge_key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    flat = (
        (np.outer(left, right) - np.outer(right, left)).reshape(-1).astype(np.float64)
    )
    return flat / max(float(np.linalg.norm(flat)), 1e-8)


def reconstruct_local_ledger(
    item_codes: np.ndarray,
    schedules: Sequence[Sequence[SupportTrial]],
    natural_scalars: np.ndarray,
    actual_state: np.ndarray,
    actual_canonical_reads: np.ndarray,
) -> dict:
    pairs = tuple(combinations(range(item_codes.shape[1]), 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    state_errors = []
    read_errors = []
    gpu_state_errors = []
    gpu_read_errors = []
    for subject, schedule in enumerate(schedules):
        codes = np.asarray(item_codes[subject], dtype=np.float64)
        keys = np.stack(
            [edge_key(codes[first], codes[second]) for first, second in pairs]
        )
        reconstructed = np.zeros(keys.shape[1], dtype=np.float64)
        ledger = np.zeros(len(pairs), dtype=np.float64)
        for trial_index, trial in enumerate(schedule):
            scalar = float(natural_scalars[subject, trial_index])
            reconstructed += scalar * edge_key(
                codes[trial.left_item], codes[trial.right_item]
            )
            canonical = cast(
                tuple[int, int], tuple(sorted((trial.left_item, trial.right_item)))
            )
            orientation = 1.0 if trial.left_item < trial.right_item else -1.0
            ledger[pair_index[canonical]] += orientation * scalar
        ledger_state = ledger @ keys
        direct_reads = reconstructed @ keys.T
        compressed_reads = (keys @ keys.T) @ ledger
        state_errors.append(float(np.max(np.abs(reconstructed - ledger_state))))
        read_errors.append(float(np.max(np.abs(direct_reads - compressed_reads))))
        gpu_state_errors.append(
            float(np.max(np.abs(reconstructed - actual_state[subject])))
        )
        gpu_read_errors.append(
            float(np.max(np.abs(compressed_reads - actual_canonical_reads[subject])))
        )
    return {
        "tensor_state_max_abs_error": max(state_errors, default=0.0),
        "ledger_tensor_state_max_abs_error": max(state_errors, default=0.0),
        "all_query_raw_read_max_abs_error": max(read_errors, default=0.0),
        "raw_subject_tensor_state_max_abs_error": state_errors,
        "raw_subject_ledger_tensor_state_max_abs_error": state_errors,
        "raw_subject_query_read_max_abs_error": read_errors,
        "gpu_tensor_state_max_abs_error_diagnostic": max(gpu_state_errors, default=0.0),
        "gpu_query_read_max_abs_error_diagnostic": max(gpu_read_errors, default=0.0),
        "raw_subject_gpu_tensor_state_max_abs_error_diagnostic": gpu_state_errors,
        "raw_subject_gpu_query_read_max_abs_error_diagnostic": gpu_read_errors,
    }


def support_schedule_hash(evaluator: FrozenFastWeightEvaluator) -> str:
    payload = json.dumps(
        [
            [asdict(trial) for trial in schedule]
            for schedule in evaluator.support_schedules
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
