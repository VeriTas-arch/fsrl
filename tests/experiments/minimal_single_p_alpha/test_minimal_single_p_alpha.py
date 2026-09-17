import torch

from fsrl.experiments.minimal_single_p.model import (
    MinimalSinglePSequence,
)
from fsrl.experiments.minimal_single_p.model import (
    make_model as make_m2,
)
from fsrl.experiments.minimal_single_p_alpha.decisions import (
    all_nine,
    evidence_binding,
    outcome,
)
from fsrl.experiments.minimal_single_p_alpha.model import make_model


def test_alpha_is_the_only_added_tensor_and_initial_computation_is_exact():
    alpha = make_model(17)
    parent = make_m2("M2", 17)

    assert set(alpha.state_dict()) == set(parent.state_dict()) | {"alpha"}
    assert all(
        torch.equal(alpha.state_dict()[name], value)
        for name, value in parent.state_dict().items()
    )
    assert torch.all(alpha.alpha == 1.0)
    assert torch.count_nonzero(alpha.h2modulation.weight) == 0
    assert torch.count_nonzero(alpha.h2modulation.bias) == 0
    assert sum(value.numel() for value in alpha.parameters()) == 87003

    inputs = torch.randn(4, 3, 32)
    alpha_values = MinimalSinglePSequence(alpha)(
        inputs,
        alpha.initial_hidden(3),
        alpha.initial_eligibility(3),
        alpha.initial_fast_weights(3),
        True,
    )
    parent_values = MinimalSinglePSequence(parent)(
        inputs,
        parent.initial_hidden(3),
        parent.initial_eligibility(3),
        parent.initial_fast_weights(3),
        True,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(alpha_values, parent_values, strict=True)
    )


def test_registered_outcome_precedence():
    assert outcome(39, 60, 60) == "generic_inadequate"
    assert outcome(60, 40, 60) == "robust_constrained_rescue"
    assert outcome(60, 0, 40) == "error_inflation"
    assert outcome(60, 1, 0) == "partial_or_mixed"
    assert outcome(60, 0, 0) == "no_constrained_rescue"


def test_liu_decisions_read_primary_analysis_schema():
    result = {
        "liu": {
            "effects": {
                "intact_minus_evidence_shuffle_learned": {"bootstrap": {"lower": 0.1}}
            },
            "routes": {
                "full": {
                    "behavior": {
                        "historical_nine_rows": {
                            "flags": {
                                str(index): {"qualitative": True} for index in range(9)
                            }
                        }
                    }
                }
            },
        }
    }
    assert evidence_binding(result)
    assert all_nine(result)
