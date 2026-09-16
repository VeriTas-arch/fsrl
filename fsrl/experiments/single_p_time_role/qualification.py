"""Synthetic pre-outcome qualification for time-role estimands."""

from __future__ import annotations

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.clean_single_p.adapter import evaluation_adapter
from fsrl.experiments.clean_single_p.model import (
    AffineSinglePSequence,
    expand_shadow_inputs,
    map_shadow,
)
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.infra.provenance import write_json_exclusive

from .analysis import generic_endpoints, paired_interval, validated_replay_error
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


def _synthetic_rollout_checks() -> dict:
    torch.manual_seed(941002)
    shadow = LinearModulationRNN(RetroModelConfig(38, 7, 2, 3), device="cpu")
    model = map_shadow(shadow, "time_retained_control").requires_grad_(False).eval()
    adapter = evaluation_adapter(model)
    parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    generator = torch.Generator().manual_seed(941003)
    inputs = torch.randn(4, 3, 32, generator=generator)
    times = torch.rand(4, 3, 1, generator=generator)
    sequence = AffineSinglePSequence(model)
    with torch.no_grad():
        direct_margin, _, _, _, direct_weights = sequence(
            inputs,
            model.initial_hidden(3),
            model.initial_eligibility(3),
            model.initial_fast_weights(3),
            True,
            time_values=times,
        )
        legacy_sequence = RecurrentSequence(adapter)
        blank = torch.zeros(2, 3, 38)
        _, _, _, _, _, blank_weights = legacy_sequence(
            blank,
            adapter.initial_hidden(3),
            adapter.initial_eligibility(3),
            adapter.initial_fast_weights(3),
            True,
        )
        logits, _, _, _, _, adapter_weights = legacy_sequence(
            expand_shadow_inputs(inputs, times, 15),
            adapter.initial_hidden(3),
            adapter.initial_eligibility(3),
            blank_weights,
            True,
        )
        adapter_margin = logits[:, 1] - logits[:, 0]
        prefix = direct_weights.clone()
        untouched = prefix.clone()
        probe_weights = prefix.clone()
        _, _, _, _, probe_weights = sequence(
            inputs,
            model.initial_hidden(3),
            model.initial_eligibility(3),
            probe_weights,
            True,
            time_values=times,
        )
        query_results = []
        for query_time in (0.0, 2.0 / 3.0):
            query_times = torch.full((4, 3, 1), query_time)
            _, _, _, _, query_weights = sequence(
                inputs,
                model.initial_hidden(3),
                model.initial_eligibility(3),
                prefix,
                False,
                time_values=query_times,
            )
            query_results.append(query_weights)
    return {
        "legacy_blank_P_max_abs": float(blank_weights.abs().max()),
        "direct_adapter_margin_max_abs_error": float(
            (direct_margin[:, 0] - adapter_margin).abs().max()
        ),
        "direct_adapter_P_max_abs_error": float(
            (direct_weights - adapter_weights).abs().max()
        ),
        "prefix_clone_has_independent_storage": prefix.data_ptr()
        != probe_weights.data_ptr(),
        "probe_preserves_natural_prefix_bitwise": torch.equal(prefix, untouched),
        "query_P_bitwise_identical": all(
            torch.equal(prefix, value) for value in query_results
        ),
        "parameters_bitwise_unchanged": all(
            torch.equal(parameters[name], value)
            for name, value in model.named_parameters()
        ),
        "all_parameters_frozen": not any(
            value.requires_grad for value in model.parameters()
        ),
    }


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
    endpoint_margins = np.asarray([[2.0, -1.0], [-2.0, 1.0]])
    endpoint_targets = np.asarray([[1, 0], [0, 1]])
    endpoint_learned = np.asarray([[True, False], [True, False]])
    reconstructed = generic_endpoints(
        endpoint_margins, endpoint_targets, endpoint_learned
    )
    expected_probability = {
        "generic_learned": 1.0 / (1.0 + np.exp(-2.0)),
        "generic_nonlearned": 1.0 / (1.0 + np.exp(-1.0)),
    }
    endpoint_error = float(
        max(
            np.max(np.abs(values - expected_probability[name]))
            for name, values in reconstructed.items()
        )
    )
    complete_case = paired_interval(
        [np.asarray([np.nan, 1.0]), np.asarray([3.0, np.nan])],
        seed=941004,
        draws=20,
    )
    relative_tolerance_error = validated_replay_error(
        np.asarray([2.000025]), np.asarray([2.0])
    )
    rollout = _synthetic_rollout_checks()
    checks = {
        "canonical_max_abs_error": float(np.max(np.abs(canonical - forward))),
        "hodge_reconstruction_max_abs_error": float(
            np.max(np.abs(geometry["gradient"] + geometry["residual"] - canonical))
        ),
        "positive_scale": scale,
        "derangement_fixed_points": int(np.sum(permutation == np.arange(32))),
        "effective_distance_finite": bool(np.all(np.isfinite(distance))),
        "fixed_effect_slope": slope,
        "endpoint_reconstruction_max_abs_error": endpoint_error,
        "paired_complete_case_point": complete_case["point"],
        "paired_complete_case_interval_finite": bool(
            np.all(np.isfinite(tuple(complete_case["interval"].values())))
        ),
        "relative_tolerance_absolute_error": relative_tolerance_error,
        "synthetic_rollout": rollout,
        "label_free_estimands": True,
        "no_scientific_model_or_parent_outcome_loaded": True,
    }
    passed = bool(
        checks["canonical_max_abs_error"] < 1e-12
        and checks["hodge_reconstruction_max_abs_error"] < 1e-12
        and abs(scale - 2.5) < 1e-12
        and checks["derangement_fixed_points"] == 0
        and checks["effective_distance_finite"]
        and abs(slope + 2.0) < 0.05
        and endpoint_error < 1e-12
        and complete_case["point"] == 2.0
        and checks["paired_complete_case_interval_finite"]
        and relative_tolerance_error > 1e-5
        and rollout["legacy_blank_P_max_abs"] == 0.0
        and rollout["direct_adapter_margin_max_abs_error"] < 1e-6
        and rollout["direct_adapter_P_max_abs_error"] < 1e-6
        and rollout["prefix_clone_has_independent_storage"]
        and rollout["probe_preserves_natural_prefix_bitwise"]
        and rollout["query_P_bitwise_identical"]
        and rollout["parameters_bitwise_unchanged"]
        and rollout["all_parameters_frozen"]
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
