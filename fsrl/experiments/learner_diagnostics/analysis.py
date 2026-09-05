"""The four preregistered diagnostic axes on frozen saved arrays."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.statistics import stable_sigmoid
from fsrl.experiments.training_strategy.estimands import subject_means
from fsrl.tasks.protocol import ordered_pairs

from .algebra import (
    component_counts,
    differences,
    global_references,
    local_decomposition,
    online_state,
    sigmoid_attribution,
)
from .estimands import (
    endpoints,
    global_contrasts,
    grouped,
    readout_accounting,
    scoring,
    summarize,
)
from .evidence import load_arrays


def assert_bridge(observed, expected, tolerance: float) -> float:
    observed, expected = (
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
    )
    np.testing.assert_allclose(observed, expected, atol=tolerance, rtol=0)
    error = np.abs(observed - expected)
    return float(np.nanmax(error)) if np.any(np.isfinite(error)) else 0.0


def input_checks(arrays, inputs, protocol) -> None:
    expected = np.broadcast_to(
        ordered_pairs(protocol.n_items), inputs["query_pairs"].shape
    )
    np.testing.assert_array_equal(inputs["query_pairs"], expected)
    for subject, pairs in enumerate(inputs["support_pairs"]):
        for index, relation in enumerate(protocol.support_pairs_higher_lower):
            selected = np.all(np.sort(pairs, axis=-1) == sorted(relation), axis=-1)
            values = inputs["retention"][selected, subject]
            if len(values) != protocol.support_blocks:
                raise RuntimeError("support presentation count differs")
            np.testing.assert_array_equal(
                values, np.full(len(values), arrays["liu__retention"][subject, index])
            )


def cell_endpoints(cells, context, parameters, spec, inputs) -> tuple:
    values, pairs = {}, {}
    retained = inputs["retention"].T.astype(bool)
    for name, cell in cells.items():
        values[name], pairs[name] = endpoints(
            cell["margin"],
            parameters["gamma_G"],
            context,
            0.25,
            spec["integrity"]["rank_tie_tolerance"],
        )
        squared = cell["support_residual"] ** 2
        values[name]["support/retained_rmse"] = np.sqrt(
            subject_means(squared, retained)
        )
        values[name]["support/all_rmse"] = np.sqrt(squared.mean(1))
    return values, pairs


def coverage_summary(values, inputs, protocol, seed, statistics) -> tuple:
    counts = component_counts(
        inputs["support_pairs"], inputs["retention"].T, protocol.n_items
    )
    raw = {}
    for cohort, selected in (("connected", counts == 1), ("disconnected", counts > 1)):
        for cell in ("RF", "RL"):
            for name, row in values[cell].items():
                if (
                    name.startswith("probability/")
                    or name == "latent/strict_correct_order"
                ):
                    raw[f"{cohort}/{cell}/{name}"] = np.where(selected, row, np.nan)
    return {"component_counts": counts, "endpoints": raw}, summarize(
        raw, seed, statistics
    )


def global_analysis(arrays, inputs, row, protocol, spec, seed) -> tuple:
    integrity = spec["integrity"]
    cells = global_references(inputs, row["parameters"], integrity)
    checks = {
        "global_margin": assert_bridge(
            cells["RF"]["margin"],
            arrays["liu__bundles__local_off__logits"],
            integrity["float32_bridge_atol"],
        ),
        "global_state": assert_bridge(
            cells["RF"]["state"], arrays["liu__w"], integrity["float32_bridge_atol"]
        ),
    }
    context = scoring(protocol, arrays["liu__retention"])
    values, pairs = cell_endpoints(cells, context, row["parameters"], spec, inputs)
    for name, expected in row["raw_endpoints"]["liu"]["intact"]["probability"].items():
        checks[f"parent_probability/{name}"] = assert_bridge(
            values["RF"][f"probability/{name}"],
            expected,
            integrity["float32_bridge_atol"],
        )
    contrasts = global_contrasts(values)
    for name in values["RF"]:
        np.testing.assert_allclose(
            contrasts["total"][name],
            contrasts["admission_at_finite"][name]
            + contrasts["integration_at_all"][name],
            atol=integrity["algebra_atol"],
            rtol=0,
            equal_nan=True,
        )
    coverage, coverage_stats = coverage_summary(
        values, inputs, protocol, seed, spec["statistics"]
    )
    readout = readout_accounting(cells["RF"]["margin"], context, 0.25)
    readout_difference = {
        name.removeprefix("probability/"): value
        - values["RF"][name.replace("probability/", "exact/")]
        for name, value in values["RF"].items()
        if name.startswith("probability/")
    }
    raw = {
        "cells": cells,
        "endpoints": values,
        "pairs": pairs,
        "contrasts": contrasts,
        "coverage": coverage,
        "readout": readout,
        "readout_difference": readout_difference,
    }
    summary = {
        "cells": {
            name: summarize(v, seed, spec["statistics"]) for name, v in values.items()
        },
        "contrasts": {
            name: summarize(v, seed, spec["statistics"])
            for name, v in contrasts.items()
        },
        "coverage": coverage_stats,
        "readout": {
            name: summarize(v, seed, spec["statistics"]) for name, v in readout.items()
        },
        "readout_difference": summarize(readout_difference, seed, spec["statistics"]),
        "gamma_over_temperature": row["parameters"]["gamma_G"] / 0.25,
        "checks": checks,
    }
    return raw, summary, context


def local_analysis(arrays, inputs, row, score_margin, context, spec, seed) -> tuple:
    tolerance = spec["integrity"]["float32_bridge_atol"]
    pieces = local_decomposition(inputs, row["parameters"]["gamma_L"])
    x = differences(inputs["support_cues"]).transpose(1, 0, 2)
    w = online_state(
        x, inputs["signed"].T, inputs["retention"].T, row["parameters"]["eta"], 1e-8
    )
    g = row["parameters"]["gamma_G"] * np.einsum(
        "sd,sqd->sq", w, differences(inputs["query_cues"])
    )
    effects, margins = sigmoid_attribution(
        g, pieces["self_margin"], pieces["cross_margin"], context["signs"], 0.25
    )
    checks = {
        "global_margin": assert_bridge(
            g, arrays["liu__bundles__local_off__logits"], tolerance
        ),
        "global_state": assert_bridge(w, arrays["liu__w"], tolerance),
        "local_margin": assert_bridge(
            pieces["self_margin"] + pieces["cross_margin"],
            arrays["liu__bundles__global_off__logits"],
            tolerance,
        ),
        "intact_margin": assert_bridge(
            margins["GSC"], arrays["liu__bundles__intact__logits"], tolerance
        ),
        "shapley": assert_bridge(
            effects["self"] + effects["cross"],
            effects["total"],
            spec["integrity"]["algebra_atol"],
        ),
    }
    nonlearned = np.repeat(context["groups"]["nonlearned"], 2)
    checks["nonlearned_self"] = assert_bridge(
        pieces["self_margin"][:, nonlearned], 0, 0
    )
    probabilities = {
        name: stable_sigmoid(m * context["signs"] / 0.25) for name, m in margins.items()
    }
    score_p = stable_sigmoid(score_margin * context["signs"] / 0.25)
    bridge = {
        "total_recipe_difference": probabilities["GSC"] - score_p,
        "acute_local": probabilities["GSC"] - probabilities["G"],
        "global_fit_difference": probabilities["G"] - score_p,
    }
    checks["between_recipe"] = assert_bridge(
        bridge["total_recipe_difference"],
        bridge["acute_local"] + bridge["global_fit_difference"],
        spec["integrity"]["algebra_atol"],
    )
    vectors = {
        "effects": {name: grouped(value, context) for name, value in effects.items()},
        "cells": {
            name: grouped(value, context) for name, value in probabilities.items()
        },
        "between_recipe": {
            name: grouped(value, context) for name, value in bridge.items()
        },
    }
    raw = {
        **pieces,
        "margins": margins,
        "oriented_effects": effects,
        "oriented_bridge": bridge,
        **vectors,
    }
    summary = {
        domain: {
            name: summarize(v, seed, spec["statistics"]) for name, v in rows.items()
        }
        for domain, rows in vectors.items()
    }
    return raw, {**summary, "checks": checks}


def analyze_pair(seed, original, protocol, spec) -> tuple:
    score, trace = (
        original["conditions"][f"{seed}/{name}"] for name in spec["conditions"]
    )
    sa, si = load_arrays(score)
    ta, ti = load_arrays(trace)
    for key in si:
        np.testing.assert_array_equal(si[key], ti[key])
    input_checks(sa, si, protocol)
    input_checks(ta, ti, protocol)
    bootstrap_seed = seed + spec["statistics"]["seed_offset"]
    gr, gs, context = global_analysis(sa, si, score, protocol, spec, bootstrap_seed)
    lr, ls = local_analysis(
        ta, ti, trace, gr["cells"]["RF"]["margin"], context, spec, bootstrap_seed
    )
    result = {
        "global": gs,
        "local": ls,
        "parameters": {
            "score_only": score["parameters"],
            "score_trace": trace["parameters"],
        },
        "parent_behavior": {
            "score_only": score["behavior"],
            "score_trace": trace["behavior"],
        },
        "parent_endpoints": {
            "score_only": score["endpoints"],
            "score_trace": trace["endpoints"],
        },
    }
    return {"global": gr, "local": lr, "scoring": context}, result
