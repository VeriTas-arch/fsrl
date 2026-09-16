from __future__ import annotations

import io

import numpy as np

from fsrl.experiments.single_p_time_role_direct.baseline import _decision
from fsrl.experiments.single_p_time_role_direct.direct import (
    assert_bitwise_equal,
    clean_inputs,
)
from fsrl.experiments.single_p_time_role_direct.locks import (
    reference,
    verify_reference,
)
from fsrl.experiments.single_p_time_role_direct.protocol import PROTOCOL
from fsrl.experiments.single_p_time_role_direct.storage import deterministic_npz_bytes
from fsrl.experiments.single_p_time_role_direct.trajectory import regression_summary
from fsrl.infra.provenance import load_json
from fsrl.paths import STUDIES_ROOT


def test_clean_input_projection_is_exact_32d_contract():
    legacy = np.arange(2 * 3 * 38, dtype=np.float32).reshape(2, 3, 38)
    clean, times = clean_inputs(legacy)
    assert clean.shape == (2, 3, 32)
    np.testing.assert_array_equal(clean[..., :31], legacy[..., :31])
    np.testing.assert_array_equal(clean[..., 31], legacy[..., 37])
    np.testing.assert_array_equal(times, legacy[..., 32:33])


def test_bitwise_comparison_rejects_one_bit_difference():
    first = {"x": np.asarray([1.0], dtype=np.float32)}
    assert_bitwise_equal(first, {"x": first["x"].copy()})
    changed = first["x"].view(np.uint32).copy()
    changed[0] += 1
    try:
        assert_bitwise_equal(first, {"x": changed.view(np.float32)})
    except RuntimeError as error:
        assert "not bitwise identical" in str(error)
    else:
        raise AssertionError("one-bit change was accepted")


def test_deterministic_typed_npz_round_trip():
    arrays = {
        "b": np.arange(5, dtype=np.int64),
        "a": np.linspace(0, 1, 7, dtype=np.float32),
    }
    first = deterministic_npz_bytes(arrays)
    second = deterministic_npz_bytes(dict(reversed(list(arrays.items()))))
    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as payload:
        assert payload.files == ["a", "b"]
        for name, value in arrays.items():
            np.testing.assert_array_equal(payload[name], value)


def test_materialized_verifier_allows_separately_locked_tensor_metadata():
    row = {**reference(PROTOCOL), "tensor_hashes": {"weight": "frozen"}}
    assert verify_reference(row) == PROTOCOL


def test_copied_parent_decision_rule_reconstructs_frozen_outcomes():
    parent = load_json(
        STUDIES_ROOT / "clean_single_p/records/results/clean_single_p_v1.json"
    )
    for row in parent["pairs"].values():
        observed = _decision(
            row["panels"],
            row["equal_panel_mean"]["clean_no_time"],
            row["noninferiority_equal_panel_mean"],
        )
        assert observed == row["decision"]


def test_cluster_regression_retains_episode_trajectories():
    panels = []
    for panel in range(3):
        episode = np.repeat(np.arange(8), 4)
        prefix = np.tile(np.arange(4), 8)
        maturity = episode * 0.1 + prefix + panel * 0.01
        functional = -0.4 * maturity + 2.0 * prefix
        panels.append(
            {
                "episode": episode,
                "prefix": prefix,
                "maturity": maturity,
                "functional_susceptibility": functional,
            }
        )
    result = regression_summary(panels, seed=17, draws=50)
    assert abs(result["point"] + 0.4) < 1e-10
    assert result["interval"]["upper"] < 0
