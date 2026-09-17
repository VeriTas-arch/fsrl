import numpy as np
import torch

from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence
from fsrl.experiments.minimal_single_p.model import make_model as make_m2
from fsrl.experiments.minimal_single_p_vector_modulation.decisions import outcome
from fsrl.experiments.minimal_single_p_vector_modulation.diagnostics import (
    geometry_records,
    summarize_geometry,
)
from fsrl.experiments.minimal_single_p_vector_modulation.model import make_model


def test_only_head_shape_changes_and_shared_initialization_is_exact():
    vector = make_model(17)
    parent = make_m2("M2", 17)

    assert sum(value.numel() for value in vector.parameters()) == 87002
    assert vector.h2modulation.weight.shape == (200, 200)
    assert vector.h2modulation.bias.shape == (200,)
    assert torch.count_nonzero(vector.h2modulation.weight) == 0
    assert torch.count_nonzero(vector.h2modulation.bias) == 0
    for name, value in parent.state_dict().items():
        if not name.startswith("h2modulation."):
            assert torch.equal(vector.state_dict()[name], value)


def test_scalar_embedding_reproduces_parent_forward_exactly():
    vector = make_model(18, hidden_size=9)
    parent = make_m2("M2", 18, hidden_size=9)
    with torch.no_grad():
        parent.h2modulation.weight.normal_(0.0, 0.1)
        parent.h2modulation.bias.normal_(0.0, 0.1)
        vector.h2modulation.weight.copy_(
            parent.h2modulation.weight.expand_as(vector.h2modulation.weight)
        )
        vector.h2modulation.bias.copy_(
            parent.h2modulation.bias.expand_as(vector.h2modulation.bias)
        )
    inputs = torch.randn(4, 3, 32)
    vector_values = MinimalSinglePSequence(vector)(
        inputs,
        vector.initial_hidden(3),
        vector.initial_eligibility(3),
        vector.initial_fast_weights(3),
        True,
    )
    parent_values = MinimalSinglePSequence(parent)(
        inputs,
        parent.initial_hidden(3),
        parent.initial_eligibility(3),
        parent.initial_fast_weights(3),
        True,
    )
    assert torch.equal(vector_values[1], parent_values[1].expand_as(vector_values[1]))
    for index in (0, 2, 3, 4):
        assert torch.equal(vector_values[index], parent_values[index])


def test_row_modulation_and_geometry_contract():
    model = make_model(19, hidden_size=3)
    with torch.no_grad():
        model.h2modulation.weight.zero_()
        model.h2modulation.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    inputs = torch.randn(2, 4, 5, 32)
    records = geometry_records(model, inputs)
    summary = summarize_geometry(records)
    assert records["R"].shape == records["active"].shape
    assert np.isnan(records["R"][~records["active"].astype(bool)]).all()
    assert summary["vector_used"]
    assert summary["R"]["maximum"] > 0

    scalar = make_model(19, hidden_size=3)
    with torch.no_grad():
        scalar.h2modulation.bias.fill_(1.0)
    scalar_summary = summarize_geometry(geometry_records(scalar, inputs))
    assert not scalar_summary["vector_used"]
    assert scalar_summary["R"]["maximum"] <= 1e-10


def test_registered_outcome_precedence():
    assert outcome(1, 0, 0, 3) == "generic_inadequate"
    assert outcome(3, 2, 1, 3) == "mixed_or_damaging"
    assert outcome(3, 2, 0, 3) == "candidate_rescue"
    assert outcome(3, 1, 0, 3) == "used_without_rescue"
    assert outcome(3, 0, 0, 0) == "vector_not_used"
