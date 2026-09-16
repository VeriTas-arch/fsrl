"""Synthetic pre-outcome qualification for time-role estimands."""

from __future__ import annotations

import numpy as np

from fsrl.infra.provenance import write_json_exclusive

from .estimands import (
    canonical_field,
    derangement,
    effective_distance,
    fixed_effect_slope,
    hodge,
    positive_scale,
)
from .locks import sources
from .protocol import PROTOCOL_SHA256, QUALIFICATION, specification


def run_qualification() -> dict:
    specification()
    rng = np.random.default_rng(941001)
    forward = rng.normal(size=(5, 28))
    pairs = np.asarray(
        [(i, j) for i in range(8) for j in range(8) if i != j], dtype=np.int64
    )
    # Reorder values to match the lexicographic ordered-pair list.
    lookup = {
        pair: forward[:, index]
        for index, pair in enumerate((i, j) for i in range(8) for j in range(i + 1, 8))
    }
    values = np.stack(
        [lookup[(i, j)] if i < j else -lookup[(j, i)] for i, j in pairs], axis=-1
    )
    expanded_pairs = np.broadcast_to(pairs, (5, *pairs.shape))
    canonical = canonical_field(values, expanded_pairs)
    geometry = hodge(canonical)
    scale = positive_scale(0.4 * canonical, canonical)
    permutation = derangement(32, 941001)
    baseline = rng.normal(size=(4, 3, 3))
    changed = baseline + 0.1
    distance = effective_distance(baseline, changed, np.ones((3, 3)))
    prefix = np.tile(np.arange(4), 6)
    maturity = rng.normal(size=len(prefix)) + prefix
    susceptibility = (
        -2.0 * maturity + 3.0 * prefix + 0.01 * rng.normal(size=len(prefix))
    )
    slope = fixed_effect_slope(susceptibility, maturity, prefix)
    checks = {
        "canonical_max_abs_error": float(np.max(np.abs(canonical - forward))),
        "hodge_reconstruction_max_abs_error": float(
            np.max(np.abs(geometry["gradient"] + geometry["residual"] - canonical))
        ),
        "positive_scale": scale,
        "derangement_fixed_points": int(np.sum(permutation == np.arange(32))),
        "effective_distance_finite": bool(np.all(np.isfinite(distance))),
        "fixed_effect_slope": slope,
        "label_free_estimands": True,
        "no_model_or_parent_outcome_loaded": True,
    }
    passed = (
        checks["canonical_max_abs_error"] < 1e-12
        and checks["hodge_reconstruction_max_abs_error"] < 1e-12
        and abs(scale - 2.5) < 1e-12
        and checks["derangement_fixed_points"] == 0
        and checks["effective_distance_finite"]
        and abs(slope + 2.0) < 0.05
    )
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "checks": checks,
        "passed": passed,
        "scientific_seeds_or_outcomes_exposed": False,
    }
    write_json_exclusive(QUALIFICATION, payload)
    return {"passed": passed, "checks": checks}


__all__ = ["run_qualification"]
