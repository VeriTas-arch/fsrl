"""Independent post-execution checks from saved arrays, not analysis reruns."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json

from .evidence import load_arrays, validate_parent


@lru_cache(maxsize=32)
def verification_counts(seed, subjects, samples):
    return np.random.default_rng(seed).multinomial(
        subjects, np.ones(subjects) / subjects, samples
    )


def verify_estimate(values, summary, seed, samples) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    excluded = np.flatnonzero(~np.isfinite(values)).tolist()
    if (
        summary["subjects"] != len(finite)
        or summary["excluded_subject_indices"] != excluded
    ):
        raise RuntimeError("participant denominator/exclusion mismatch")
    if not len(finite):
        if summary["mean"] is not None or summary["bootstrap"]["lower"] is not None:
            raise RuntimeError("empty cohort must be undefined")
        return 0.0
    weights = verification_counts(seed, len(finite), samples)
    draws = weights @ finite / len(finite)
    expected = [finite.mean(), *np.quantile(draws, [0.025, 0.975])]
    observed = [
        summary["mean"],
        summary["bootstrap"]["lower"],
        summary["bootstrap"]["upper"],
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
    return float(np.max(np.abs(np.asarray(observed) - expected)))


def verify_summaries(
    arrays, raw_prefix: str, summaries: dict, seed: int, samples: int
) -> tuple:
    count, error = 0, 0.0
    for name, row in summaries.items():
        key = f"{raw_prefix}__{name}"
        if "bootstrap" in row:
            error = max(error, verify_estimate(arrays[key], row, seed, samples))
            count += 1
        else:
            c, e = verify_summaries(arrays, key, row, seed, samples)
            count, error = count + c, max(error, e)
    return count, error


def verify_numeric(seed, arrays, original, spec) -> dict:
    prefix = str(seed)
    source, inputs = load_arrays(original["conditions"][f"{seed}/score_only"])
    params = original["conditions"][f"{seed}/score_only"]["parameters"]
    left, right = np.split(inputs["support_cues"].transpose(1, 0, 2), 2, -1)
    x = left - right
    ql, qr = np.split(inputs["query_cues"], 2, -1)
    eps = spec["integrity"]["float32_bridge_atol"]
    for cell in ("RF", "AF", "RL", "AL"):
        base = f"{prefix}__global__cells__{cell}"
        w = arrays[f"{base}__state"]
        margins = params["gamma_G"] * np.sum(w[:, None] * (ql - qr), axis=-1)
        np.testing.assert_allclose(
            margins, arrays[f"{base}__margin"], atol=1e-10, rtol=0
        )
        verify_global_endpoints(seed, cell, margins, arrays)
        residual = np.sum(w[:, None] * x, axis=-1) - inputs["signed"].T
        np.testing.assert_allclose(
            residual, arrays[f"{base}__support_residual"], atol=1e-10, rtol=0
        )
        if cell.endswith("L"):
            normal = np.sum(
                x * (residual * arrays[f"{base}__admitted"])[..., None], axis=1
            )
            np.testing.assert_allclose(normal, 0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        arrays[f"{prefix}__global__cells__RF__margin"],
        source["liu__bundles__intact__logits"],
        atol=eps,
        rtol=0,
    )
    base = f"{prefix}__local"
    contribution = arrays[f"{base}__trial_contribution"]
    _, trace_inputs = load_arrays(original["conditions"][f"{seed}/score_trace"])
    independent = gram_contributions(
        trace_inputs,
        original["conditions"][f"{seed}/score_trace"]["parameters"]["gamma_L"],
    )
    np.testing.assert_allclose(contribution, independent, atol=1e-10, rtol=0)
    same = arrays[f"{base}__same_relation"]
    np.testing.assert_allclose(
        (contribution * same).sum(1), arrays[f"{base}__self_margin"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        (contribution * ~same).sum(1),
        arrays[f"{base}__cross_margin"],
        atol=1e-10,
        rtol=0,
    )
    signs = arrays[f"{prefix}__scoring__signs"]
    p = {}
    for cell in ("G", "GS", "GC", "GSC"):
        a = arrays[f"{base}__margins__{cell}"] * signs / 0.25
        p[cell] = np.exp(-np.logaddexp(0.0, -a))
    np.testing.assert_allclose(
        arrays[f"{base}__oriented_effects__self"],
        (p["GS"] - p["G"] + p["GSC"] - p["GC"]) / 2,
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        arrays[f"{base}__oriented_effects__cross"],
        (p["GC"] - p["G"] + p["GSC"] - p["GS"]) / 2,
        atol=1e-10,
        rtol=0,
    )
    bridge = f"{base}__oriented_bridge"
    np.testing.assert_allclose(
        arrays[f"{bridge}__total_recipe_difference"],
        arrays[f"{bridge}__acute_local"] + arrays[f"{bridge}__global_fit_difference"],
        atol=1e-10,
        rtol=0,
    )
    return {"passed": True, "cells": 4, "local_probability_identities": True}


def gram_contributions(inputs, gain):
    """Compute wedge inner products from cue Gram products, without tensor keys."""
    a, b = np.split(inputs["support_cues"].transpose(1, 0, 2).astype(float), 2, -1)
    c, d = np.split(inputs["query_cues"].astype(float), 2, -1)
    numerator = 2 * (
        (a @ c.swapaxes(1, 2)) * (b @ d.swapaxes(1, 2))
        - (a @ d.swapaxes(1, 2)) * (b @ c.swapaxes(1, 2))
    )
    ab_norm = np.sqrt(2 * ((a * a).sum(-1) * (b * b).sum(-1) - (a * b).sum(-1) ** 2))
    cd_norm = np.sqrt(2 * ((c * c).sum(-1) * (d * d).sum(-1) - (c * d).sum(-1) ** 2))
    return (
        gain
        * inputs["local_evidence"].T[..., None]
        * numerator
        / np.maximum(ab_norm[..., None] * cd_norm[:, None], 1e-16)
    )


def verify_global_endpoints(seed, cell, margins, arrays):
    correct = margins * arrays[f"{seed}__scoring__signs"]
    measures = {
        "probability": np.exp(-np.logaddexp(0.0, -correct / 0.25)),
        "exact": (np.sign(correct) + 1) / 2,
    }
    prefix = f"{seed}__global__endpoints__{cell}"
    for measure, oriented in measures.items():
        pairs = (oriented[:, ::2] + oriented[:, 1::2]) / 2
        for group in ("overall", "learned", "nonlearned", "retained", "omitted"):
            mask = np.broadcast_to(
                arrays[f"{seed}__scoring__groups__{group}"], pairs.shape
            )
            means = np.asarray(
                [
                    v[m].mean() if m.any() else np.nan
                    for v, m in zip(pairs, mask, strict=True)
                ]
            )
            np.testing.assert_allclose(
                means, arrays[f"{prefix}__{measure}/{group}"], atol=1e-10, rtol=0
            )
        for endpoint, weight in (
            ("distance_slope", "slope_weight"),
            ("serial_contrast", "serial_weight"),
        ):
            value = (pairs * arrays[f"{seed}__scoring__{weight}"]).sum(1)
            np.testing.assert_allclose(
                value, arrays[f"{prefix}__{measure}/{endpoint}"], atol=1e-10, rtol=0
            )


def verify_run(directory, spec) -> dict:
    if not validate_run_manifest(directory / "run.json")["passed"]:
        raise RuntimeError("diagnostic run manifest invalid")
    manifest = load_json(directory / "run.json")
    if manifest["lifecycle_state"] != "complete":
        raise RuntimeError("diagnostic run incomplete")
    result = load_json(directory / "result.json")
    raw_path = verify_reference(result["arrays"])
    original = validate_parent()
    verified = {}
    with np.load(raw_path, allow_pickle=False) as arrays:
        for seed in spec["seeds"]:
            row = result["fits"][str(seed)]
            check = verify_numeric(seed, arrays, original, spec)
            count, error = 0, 0.0
            mappings = {
                "global__endpoints": row["global"]["cells"],
                "global__contrasts": row["global"]["contrasts"],
                "global__coverage__endpoints": row["global"]["coverage"],
                "global__readout": row["global"]["readout"],
                "global__readout_difference": row["global"]["readout_difference"],
                "local__effects": row["local"]["effects"],
                "local__cells": row["local"]["cells"],
                "local__between_recipe": row["local"]["between_recipe"],
            }
            for prefix, summaries in mappings.items():
                c, e = verify_summaries(
                    arrays,
                    f"{seed}__{prefix}",
                    summaries,
                    seed + spec["statistics"]["seed_offset"],
                    spec["statistics"]["samples"],
                )
                count, error = count + c, max(error, e)
            verified[str(seed)] = {
                **check,
                "estimates_checked": count,
                "bootstrap_max_error": error,
            }
    return {"passed": True, "fits": verified, "manifest": str(directory / "run.json")}
