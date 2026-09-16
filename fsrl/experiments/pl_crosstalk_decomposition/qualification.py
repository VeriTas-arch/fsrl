"""Synthetic pre-input qualification for cross-talk decomposition."""

from __future__ import annotations

import io

import numpy as np

from fsrl.infra.formal_runtime import formal_runtime_snapshot
from fsrl.infra.provenance import write_json_exclusive

from .estimands import (
    packed_keys,
    probability_components,
    relation_source_contributions,
    retained_subject_mean,
    source_concentration,
)
from .locks import QUALIFICATION_PATH, implementation_sources
from .protocol import PROTOCOL_SHA256
from .storage import deterministic_npz_bytes


def _analysis_runtime() -> dict:
    runtime = formal_runtime_snapshot()
    if not (
        runtime["active"]
        and runtime["torch_intraop_threads"] == 1
        and runtime["torch_interop_threads"] == 1
        and runtime["blas_thread_limit"] == 1
    ):
        raise RuntimeError("use the registered bounded formal runtime")
    return runtime


def _check(error: float, tolerance: float = 1e-12) -> dict:
    return {
        "passed": bool(np.isfinite(error) and error <= tolerance),
        "max_abs_error": float(error),
        "tolerance": float(tolerance),
    }


def run_qualification() -> dict:
    runtime = _analysis_runtime()
    left = np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], dtype=np.float32)
    right = np.asarray([[-1.0, 1.0, 1.0], [1.0, 1.0, -1.0]], dtype=np.float32)
    keys = packed_keys(left, right)
    reversed_keys = packed_keys(right, left)
    key_error = max(
        float(np.max(np.abs(np.linalg.norm(keys, axis=1) - 1.0))),
        float(np.max(np.abs(keys + reversed_keys))),
    )

    support = np.stack((keys, reversed_keys, keys, reversed_keys), axis=1)
    query = np.stack((keys, reversed_keys), axis=1)
    evidence = np.asarray([[0.5, -0.25, 0.1, 0.3], [0.2, 0.4, -0.1, 0.6]])
    source_indices = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]])
    contributions = relation_source_contributions(
        evidence,
        support,
        source_indices,
        query,
        relation_count=2,
    )
    overlaps = np.einsum("itk,iek->ite", support, query, dtype=np.float32)
    expected_total = np.sum(evidence[:, :, None] * overlaps, axis=1)
    source_error = float(np.max(np.abs(np.sum(contributions, axis=1) - expected_total)))

    oriented = np.sum(contributions, axis=1).reshape(2, 1, 2)
    baseline = np.asarray([[[0.2, 0.4]], [[-0.1, 0.7]]], dtype=np.float64)
    signs = np.asarray([[1.0, -1.0]], dtype=np.float64)
    gain = 0.3
    temperature = 0.25
    components = probability_components(
        baseline,
        oriented,
        signs,
        gain=gain,
        temperature=temperature,
    )
    perturbation = gain * oriented * signs[None, :, :]
    gain_error = float(
        np.max(np.abs(components["correct_perturbation"] - perturbation))
    )
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    expected_effect = sigmoid((baseline + perturbation) / temperature) - sigmoid(
        baseline / temperature
    )
    probability_error = float(
        np.max(np.abs(components["exact_effect"] - expected_effect))
    )
    expected_first_order = (
        sigmoid(baseline / temperature)
        * (1.0 - sigmoid(baseline / temperature))
        / temperature
        * perturbation
    )
    first_order_error = float(
        np.max(np.abs(components["first_order_effect"] - expected_first_order))
    )

    retained = np.asarray([[True], [True]])
    subject_values = retained_subject_mean(components["exact_effect"], retained)
    weighting_error = float(
        np.max(
            np.abs(subject_values - np.mean(components["exact_effect"], axis=(1, 2)))
        )
    )

    shaped_contributions = contributions.reshape(2, 1, 2, 2)
    concentration = source_concentration(shaped_contributions, signs)
    concentration_error = max(
        float(np.nanmax(np.maximum(-concentration["top_one_share"], 0.0))),
        float(np.nanmax(np.maximum(concentration["top_one_share"] - 1.0, 0.0))),
        float(np.nanmax(np.maximum(-concentration["cancellation_ratio"], 0.0))),
        float(np.nanmax(np.maximum(concentration["cancellation_ratio"] - 1.0, 0.0))),
    )

    fixture = {
        "baseline": baseline,
        "effect": components["exact_effect"],
        "retention": retained,
    }
    first_bytes = deterministic_npz_bytes(fixture)
    second_bytes = deterministic_npz_bytes(dict(reversed(list(fixture.items()))))
    with np.load(io.BytesIO(first_bytes), allow_pickle=False) as restored:
        roundtrip_error = max(
            float(np.max(np.abs(restored[name] - fixture[name])))
            if fixture[name].dtype != np.bool_
            else float(not np.array_equal(restored[name], fixture[name]))
            for name in fixture
        )
    storage_error = max(float(first_bytes != second_bytes), roundtrip_error)

    checks = {
        "packed_key_norm_and_antisymmetry": _check(key_error, 1e-7),
        "source_relation_sum": _check(source_error, 1e-7),
        "gain_scaled_perturbation": _check(gain_error),
        "exact_probability_effect": _check(probability_error),
        "first_order_definition": _check(first_order_error),
        "retained_subject_weighting": _check(weighting_error),
        "source_concentration": _check(concentration_error),
        "deterministic_npz_roundtrip": _check(storage_error),
    }
    result = {
        "schema_version": 1,
        "study_id": "pl_crosstalk_decomposition",
        "seed": 941001,
        "protocol_sha256": PROTOCOL_SHA256,
        "frozen_inputs_loaded": False,
        "runtime": runtime,
        "sources": implementation_sources(),
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError("synthetic cross-talk qualification failed")
    QUALIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(QUALIFICATION_PATH, result)
    return result
