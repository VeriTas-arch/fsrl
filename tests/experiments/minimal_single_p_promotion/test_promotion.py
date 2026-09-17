from __future__ import annotations

import copy

import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence, make_model
from fsrl.experiments.minimal_single_p_promotion.adapter import evaluation_adapter
from fsrl.experiments.minimal_single_p_promotion.decisions import (
    generic_category,
    study_outcome,
    wilson,
)
from fsrl.experiments.minimal_single_p_promotion.protocol import (
    inherited_recipe,
    specification,
)


def _panel(lower: float = 1.0) -> dict:
    return {
        "competence": {
            group: {"bootstrap": {"lower": lower}}
            for group in ("learned", "nonlearned")
        },
        "P_dependence": {
            group: {"bootstrap": {"lower": lower}}
            for group in ("learned", "nonlearned")
        },
        "coherence": {"bootstrap": {"lower": lower}},
    }


def test_frozen_design_identity() -> None:
    spec = specification()
    assert spec["architecture"]["name"] == "M2"
    assert spec["architecture"]["parameter_count"] == 47003
    assert spec["design"]["network_seeds"] == list(range(3041, 3061))
    assert spec["design"]["evaluation_panels"] == [1, 2, 3]
    assert inherited_recipe(2)["evaluation"]["generic"]["rng_seed"] == 961200


def test_adapter_matches_native_m2() -> None:
    native = make_model("M2", 77, hidden_size=9)
    adapter = evaluation_adapter(copy.deepcopy(native))
    inputs = torch.randn(4, 3, 32)
    legacy = torch.zeros(4, 3, 38)
    legacy[..., :31] = inputs[..., :31]
    legacy[..., 37] = inputs[..., 31]
    direct = MinimalSinglePSequence(native)(
        inputs,
        native.initial_hidden(3),
        native.initial_eligibility(3),
        native.initial_fast_weights(3),
        True,
    )
    wrapped = RecurrentSequence(adapter)(
        legacy,
        adapter.initial_hidden(3),
        adapter.initial_eligibility(3),
        adapter.initial_fast_weights(3),
        True,
    )
    torch.testing.assert_close(direct[0], wrapped[0][:, 1:2] - wrapped[0][:, 0:1])
    torch.testing.assert_close(direct[1], wrapped[2])
    torch.testing.assert_close(direct[2], wrapped[3])
    torch.testing.assert_close(direct[3], wrapped[4])
    torch.testing.assert_close(direct[4], wrapped[5])


def test_distribution_decisions_have_registered_precedence() -> None:
    passing, failing = _panel(), _panel(-1.0)
    assert generic_category({"1": passing, "2": passing, "3": passing}) == (
        "stable_constructive"
    )
    assert generic_category({"1": passing, "2": failing, "3": failing}) == (
        "panel_variable"
    )
    assert generic_category({"1": failing, "2": failing, "3": failing}) == (
        "nonconstructive"
    )
    assert study_outcome({"a": "nonconstructive"}, {"a": True}) == (
        "generic_recipe_failure"
    )
    assert study_outcome({"a": "stable_constructive"}, {"a": True}) == (
        "uniform_complete_pilot_compatibility"
    )
    assert study_outcome({"a": "stable_constructive"}, {"a": False}) == (
        "heterogeneous_solution_distribution"
    )


def test_wilson_interval() -> None:
    row = wilson(10, 20)
    assert row["proportion"] == 0.5
    assert row["wilson95"]["lower"] < 0.5 < row["wilson95"]["upper"]
