"""Locally balanced training chronology for presentation horizons."""

from __future__ import annotations

import hashlib

import numpy as np

HORIZONS = (2, 3, 4, 5, 6)
EDGE_COUNTS = (7, 8, 9, 10)
CYCLES = 75
CELLS = np.asarray(
    [(horizon, edges) for horizon in HORIZONS for edges in EDGE_COUNTS],
    dtype=np.int16,
)


def schedule_seed(network_seed: int) -> int:
    return 620000 + network_seed


def build_schedule(network_seed: int) -> np.ndarray:
    rng = np.random.default_rng(schedule_seed(network_seed))
    cycles = [CELLS[rng.permutation(len(CELLS))] for _ in range(CYCLES)]
    return np.ascontiguousarray(np.concatenate(cycles, axis=0))


def schedule_sha256(schedule: np.ndarray) -> str:
    array = np.ascontiguousarray(schedule, dtype=np.int16)
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_schedule(schedule: np.ndarray) -> dict:
    expected = {tuple(map(int, row)) for row in CELLS}
    if schedule.shape != (CYCLES * len(CELLS), 2):
        raise RuntimeError("anytime schedule has the wrong shape")
    for cycle in range(CYCLES):
        rows = schedule[cycle * len(CELLS) : (cycle + 1) * len(CELLS)]
        if {tuple(map(int, row)) for row in rows} != expected:
            raise RuntimeError(f"anytime macro-cycle {cycle} is not balanced")
    counts = {
        f"B{horizon}_E{edges}": int(
            np.count_nonzero((schedule[:, 0] == horizon) & (schedule[:, 1] == edges))
        )
        for horizon, edges in expected
    }
    if set(counts.values()) != {CYCLES}:
        raise RuntimeError("anytime schedule cells do not occur 75 times")
    variable_exposure = int(np.sum(schedule[:, 0] * schedule[:, 1]))
    fixed_exposure = int(np.sum(4 * schedule[:, 1]))
    if variable_exposure != fixed_exposure:
        raise RuntimeError("FH and VH aggregate presentations differ")
    return {
        "shape": list(schedule.shape),
        "counts": dict(sorted(counts.items())),
        "variable_exposure": variable_exposure,
        "fixed_exposure": fixed_exposure,
        "sha256": schedule_sha256(schedule),
    }


__all__ = [
    "CELLS",
    "CYCLES",
    "EDGE_COUNTS",
    "HORIZONS",
    "build_schedule",
    "schedule_seed",
    "schedule_sha256",
    "validate_schedule",
]
