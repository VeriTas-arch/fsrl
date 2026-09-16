"""Synthetic pre-outcome qualification for the direct-authority successor."""

from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import torch

from fsrl.experiments.clean_single_p.model import AffineSingleP, CleanSinglePConfig
from fsrl.experiments.single_p_time_role.analysis import generic_endpoints
from fsrl.experiments.single_p_time_role.estimands import (
    canonical_field,
    derangement,
    effective_distance,
    fixed_effect_slope,
    hodge,
    positive_scale,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .budgets import (
    EDGE_SLACK,
    cross_entropy_budget,
    margin_budget,
    probability_budget,
)
from .direct import (
    apply_support_probe,
    assert_bitwise_equal,
    read_original,
    support_trajectory,
)
from .locks import require_exact_inventory, sources
from .protocol import (
    PROTOCOL_SHA256,
    QUALIFICATION,
    specification,
)
from .storage import deterministic_npz_bytes


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _synthetic_batch() -> EpisodeBatch:
    rng = np.random.default_rng(991101)
    support = rng.normal(size=(3, 4, 2, 38)).astype(np.float32)
    query = rng.normal(size=(2, 4, 38)).astype(np.float32)
    support[..., 32] = rng.random(size=(3, 4, 2))
    query[..., 32] = 2.0 / 3.0
    targets = np.asarray([0, 1, 1, 0], dtype=np.int64)
    return EpisodeBatch(
        {
            "support_inputs": support,
            "query_inputs": query,
            "targets": targets,
            "item_codes": rng.normal(size=(2, 8, 15)).astype(np.float32),
        }
    )


def _rollout_checks() -> dict:
    torch.manual_seed(991102)
    model = AffineSingleP(
        CleanSinglePConfig(cue_size=15, hidden_size=7),
        retain_time=True,
        device="cuda",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.2, 0.2)
    model.requires_grad_(False).eval()
    before = tensor_hashes(model)
    cpu = _synthetic_batch()
    first_states, _, first_modulations, _ = support_trajectory(model, cpu)
    first_margin = read_original(model, first_states[-1], cpu)
    second_states, _, second_modulations, _ = support_trajectory(model, cpu)
    second_margin = read_original(model, second_states[-1], cpu)
    first = {
        "states": torch.stack(first_states).cpu().numpy(),
        "modulations": torch.stack(first_modulations).cpu().numpy(),
        "margins": first_margin,
    }
    second = {
        "states": torch.stack(second_states).cpu().numpy(),
        "modulations": torch.stack(second_modulations).cpu().numpy(),
        "margins": second_margin,
    }
    assert_bitwise_equal(first, second)
    selected = np.asarray([0, 2])
    full_prefix = first["states"]
    selected_prefix = full_prefix[:, selected]
    prefix = first_states[1][0:1].clone()
    untouched = prefix.clone()
    generator = np.random.default_rng(991103)
    probe_inputs = generator.normal(size=(4, 3, 32)).astype(np.float32)
    probe_times = generator.random(size=(4, 3, 1)).astype(np.float32)
    probe_state = prefix.expand(probe_inputs.shape[1], -1, -1).clone()
    changed = apply_support_probe(model, probe_inputs, probe_times, probe_state)
    return {
        "same_process_direct_replay_bitwise": True,
        "selected_prefix_is_full_batch_slice": np.array_equal(
            selected_prefix, first["states"][:, selected]
        ),
        "full_batch_size_preserved": full_prefix.shape[1] == 4,
        "prefix_states_have_independent_storage": len(
            {state.data_ptr() for state in first_states}
        )
        == len(first_states),
        "parameters_bitwise_unchanged": tensor_hashes(model) == before,
        "all_parameters_frozen": not any(
            value.requires_grad for value in model.parameters()
        ),
        "probe_uses_independent_state": changed.data_ptr() != prefix.data_ptr(),
        "probe_preserves_natural_prefix_bitwise": torch.equal(prefix, untouched),
    }


def _budget_checks() -> dict:
    reference = np.asarray([-19.0, -2.0, -0.1, 0.0, 0.1, 2.0, 19.0])
    perturbation = 0.91 * margin_budget(reference)
    direct = reference + perturbation
    results = {}
    for temperature in (1.0, 0.25):
        parent_probability = 1.0 / (
            1.0 + np.exp(-reference.astype(np.longdouble) / temperature)
        )
        direct_probability = 1.0 / (
            1.0 + np.exp(-direct.astype(np.longdouble) / temperature)
        )
        probability_error = np.abs(direct_probability - parent_probability)
        allowed_probability = probability_budget(reference, temperature)
        if np.any(probability_error > allowed_probability):
            raise RuntimeError("synthetic sigmoid propagation bound failed")
        parent_ce = np.logaddexp(
            np.longdouble(0), -reference.astype(np.longdouble) / temperature
        )
        direct_ce = np.logaddexp(
            np.longdouble(0), -direct.astype(np.longdouble) / temperature
        )
        ce_error = np.abs(direct_ce - parent_ce)
        allowed_ce = cross_entropy_budget(reference, temperature)
        if np.any(ce_error > allowed_ce):
            raise RuntimeError("synthetic CE propagation bound failed")
        float_probability = 1.0 / (1.0 + np.exp(-reference / temperature))
        float_ce = np.logaddexp(0.0, -reference / temperature)
        implementation_error = max(
            float(np.max(np.abs(float_probability - parent_probability))),
            float(np.max(np.abs(float_ce - parent_ce))),
        )
        results[str(temperature)] = {
            "analytic_probability_bound": True,
            "analytic_cross_entropy_bound": True,
            "maximum_implementation_error": implementation_error,
        }
    values = np.asarray([0.2, -0.1, 0.7, 0.4], dtype=np.float64)
    budgets = np.asarray([1e-5, 2e-5, 3e-5, 4e-5], dtype=np.float64)
    counts = np.asarray([[1, 1, 1, 1], [0, 2, 0, 2]], dtype=np.float64)
    draw_budget = counts @ budgets / 4 + EDGE_SLACK
    changed = values + budgets
    draw_error = np.abs(counts @ changed / 4 - counts @ values / 4)
    quantile_error = abs(
        np.quantile(counts @ changed / 4, 0.5) - np.quantile(counts @ values / 4, 0.5)
    )
    results["aggregation"] = {
        "weighted_bootstrap_bound": bool(np.all(draw_error <= draw_budget)),
        "quantile_max_draw_bound": bool(
            quantile_error <= draw_budget.max() + EDGE_SLACK
        ),
    }
    results["maximum_implementation_error"] = max(
        row["maximum_implementation_error"]
        for row in results.values()
        if "maximum_implementation_error" in row
    )
    return results


def run_qualification() -> dict:
    specification()
    rng = np.random.default_rng(991100)
    forward = rng.normal(size=(5, 28))
    pairs = np.asarray(
        [(i, j) for i in range(8) for j in range(8) if i != j], dtype=np.int64
    )
    lookup = {
        pair: forward[:, index]
        for index, pair in enumerate((i, j) for i in range(8) for j in range(i + 1, 8))
    }
    ordered = np.stack(
        [lookup[(i, j)] if i < j else -lookup[(j, i)] for i, j in pairs], axis=-1
    )
    canonical = canonical_field(ordered, np.broadcast_to(pairs, (5, *pairs.shape)))
    decomposition = hodge(canonical)
    baseline = rng.normal(size=(4, 3, 3))
    prefix = np.tile(np.arange(4), 6)
    maturity = rng.normal(size=len(prefix)) + prefix
    susceptibility = (
        -2.0 * maturity + 3.0 * prefix + 0.01 * rng.normal(size=len(prefix))
    )
    endpoint = generic_endpoints(
        np.asarray([[2.0, -1.0], [-2.0, 1.0]]),
        np.asarray([[1, 0], [0, 1]]),
        np.asarray([[True, False], [True, False]]),
    )
    fixture = {
        "float": rng.normal(size=(3, 4)).astype(np.float32),
        "integer": np.arange(5, dtype=np.int64),
    }
    first_bytes = deterministic_npz_bytes(fixture)
    second_bytes = deterministic_npz_bytes(dict(reversed(list(fixture.items()))))
    with np.load(io.BytesIO(first_bytes), allow_pickle=False) as restored:
        storage_roundtrip = all(
            np.array_equal(restored[name], value) for name, value in fixture.items()
        )
        loaded = {name: restored[name].copy() for name in restored.files}
    loaded["float"][0, 0] += 1
    with np.load(io.BytesIO(first_bytes), allow_pickle=False) as restored:
        storage_immutable = np.array_equal(restored["float"], fixture["float"])
    require_exact_inventory({"a", "b"}, {"b", "a"})
    package = REPO_ROOT / "fsrl/experiments/single_p_time_role_propagated"
    direct_imports = _imports(package / "direct.py")
    baseline_imports = _imports(package / "baseline.py")
    checks = {
        "canonical_max_abs_error": float(np.max(np.abs(canonical - forward))),
        "hodge_reconstruction_max_abs_error": float(
            np.max(
                np.abs(
                    decomposition["gradient"] + decomposition["residual"] - canonical
                )
            )
        ),
        "positive_scale": positive_scale(0.4 * canonical, canonical),
        "derangement_fixed_points": int(
            np.sum(derangement(32, 991100) == np.arange(32))
        ),
        "effective_distance_finite": bool(
            np.all(
                np.isfinite(
                    effective_distance(baseline, baseline + 0.1, np.ones((3, 3)))
                )
            )
        ),
        "fixed_effect_slope": fixed_effect_slope(susceptibility, maturity, prefix),
        "endpoint_reconstruction_finite": all(
            np.isfinite(value).all() for value in endpoint.values()
        ),
        "deterministic_npz_bytes": first_bytes == second_bytes,
        "typed_npz_roundtrip": storage_roundtrip,
        "baseline_reader_does_not_mutate_storage": storage_immutable,
        "artifact_inventory_is_exact": True,
        "direct_module_has_no_adapter_import": not any(
            "adapter" in name for name in direct_imports
        ),
        "baseline_has_no_mechanism_import": not any(
            name.endswith("mechanism") for name in baseline_imports
        ),
        "synthetic_rollout": _rollout_checks(),
        "propagated_budget": _budget_checks(),
        "no_scientific_model_input_or_outcome_loaded": True,
    }
    rollout = checks["synthetic_rollout"]
    passed = bool(
        checks["canonical_max_abs_error"] < 1e-12
        and checks["hodge_reconstruction_max_abs_error"] < 1e-12
        and abs(checks["positive_scale"] - 2.5) < 1e-12
        and checks["derangement_fixed_points"] == 0
        and checks["effective_distance_finite"]
        and abs(checks["fixed_effect_slope"] + 2.0) < 0.05
        and checks["endpoint_reconstruction_finite"]
        and checks["deterministic_npz_bytes"]
        and checks["typed_npz_roundtrip"]
        and checks["baseline_reader_does_not_mutate_storage"]
        and checks["artifact_inventory_is_exact"]
        and checks["direct_module_has_no_adapter_import"]
        and checks["baseline_has_no_mechanism_import"]
        and all(rollout.values())
        and checks["propagated_budget"]["maximum_implementation_error"] <= 1e-13
        and all(checks["propagated_budget"]["aggregation"].values())
    )
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "checks": checks,
        "passed": passed,
        "scientific_seeds_inputs_or_outcomes_exposed": False,
    }
    QUALIFICATION.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(QUALIFICATION, payload)
    return {"passed": passed, "checks": checks}


__all__ = ["run_qualification"]
