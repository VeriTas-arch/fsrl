"""Pre-outcome qualification of M2, its adapter, and registered decisions."""

from __future__ import annotations

import copy

import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence, make_model
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import compile_module

from .adapter import evaluation_adapter
from .decisions import generic_category, study_outcome, wilson
from .locks import sources
from .protocol import PROTOCOL_SHA256, QUALIFICATION, specification


def _max_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)).detach())


def _structure() -> dict:
    model = make_model("M2", 9901)
    count = sum(value.numel() for value in model.parameters())
    passed = (
        count == specification()["architecture"]["parameter_count"]
        and model.alpha is None
        and model.etaet is not None
        and float(model.etaet.detach()) == torch.tensor(0.7).item()
        and bool(torch.count_nonzero(model.h2modulation.weight) == 0)
        and bool(torch.count_nonzero(model.h2modulation.bias) == 0)
    )
    return {
        "parameter_count": count,
        "alpha_absent": model.alpha is None,
        "eta_learned": model.etaet is not None,
        "eta_initial": float(model.etaet.detach()),
        "zero_modulation": bool(torch.count_nonzero(model.h2modulation.weight) == 0)
        and bool(torch.count_nonzero(model.h2modulation.bias) == 0),
        "passed": passed,
    }


def _adapter_parity(device: str, *, compiled: bool) -> dict:
    torch.manual_seed(9902)
    native = make_model("M2", 9903, device=device)
    adapter = evaluation_adapter(copy.deepcopy(native))
    native_sequence = MinimalSinglePSequence(native)
    adapter_sequence = RecurrentSequence(adapter)
    if compiled:
        native_sequence = compile_module(native_sequence, PROFILE)
        adapter_sequence = compile_module(adapter_sequence, PROFILE)
    batch = 5
    inputs = torch.randn(4, batch, 32, device=device)
    legacy = torch.zeros(4, batch, 38, device=device)
    legacy[..., :31] = inputs[..., :31]
    legacy[..., 37] = inputs[..., 31]
    native_values = native_sequence(
        inputs,
        native.initial_hidden(batch),
        native.initial_eligibility(batch),
        native.initial_fast_weights(batch),
        True,
    )
    adapter_values = adapter_sequence(
        legacy,
        adapter.initial_hidden(batch),
        adapter.initial_eligibility(batch),
        adapter.initial_fast_weights(batch),
        True,
    )
    adapter_margin = adapter_values[0][:, 1:2] - adapter_values[0][:, 0:1]
    errors = {
        "margin": _max_error(native_values[0], adapter_margin),
        "modulation": _max_error(native_values[1], adapter_values[2]),
        "hidden": _max_error(native_values[2], adapter_values[3]),
        "eligibility": _max_error(native_values[3], adapter_values[4]),
        "P": _max_error(native_values[4], adapter_values[5]),
    }
    tolerance = 1e-5 if device == "cuda" else 1e-6
    return {
        "errors": errors,
        "tolerance": tolerance,
        "passed": max(errors.values()) <= tolerance,
    }


def _decisions() -> dict:
    passing = {
        "competence": {
            group: {"bootstrap": {"lower": 0.6}} for group in ("learned", "nonlearned")
        },
        "P_dependence": {
            group: {"bootstrap": {"lower": 0.1}} for group in ("learned", "nonlearned")
        },
        "coherence": {"bootstrap": {"lower": 0.99}},
    }
    failing = copy.deepcopy(passing)
    failing["coherence"]["bootstrap"]["lower"] = 0.9
    categories = {
        "stable": generic_category({"1": passing, "2": passing, "3": passing}),
        "variable": generic_category({"1": passing, "2": failing, "3": failing}),
        "none": generic_category({"1": failing, "2": failing, "3": failing}),
    }
    interval = wilson(10, 20)
    outcomes = {
        "failure": study_outcome({"a": "nonconstructive"}, {"a": False}),
        "uniform": study_outcome({"a": "stable_constructive"}, {"a": True}),
        "mixed": study_outcome({"a": "stable_constructive"}, {"a": False}),
    }
    passed = categories == {
        "stable": "stable_constructive",
        "variable": "panel_variable",
        "none": "nonconstructive",
    } and outcomes == {
        "failure": "generic_recipe_failure",
        "uniform": "uniform_complete_pilot_compatibility",
        "mixed": "heterogeneous_solution_distribution",
    }
    return {
        "categories": categories,
        "outcomes": outcomes,
        "wilson": interval,
        "passed": passed,
    }


def run_qualification() -> dict:
    runtime = configure_execution()
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": runtime,
        "structure": _structure(),
        "adapter_cpu": _adapter_parity("cpu", compiled=False),
        "adapter_cuda_eager": _adapter_parity("cuda", compiled=False),
        "adapter_cuda_compiled": _adapter_parity("cuda", compiled=True),
        "decisions": _decisions(),
        "human_outcomes_exposed": False,
    }
    result["passed"] = all(
        result[name]["passed"]
        for name in (
            "structure",
            "adapter_cpu",
            "adapter_cuda_eager",
            "adapter_cuda_compiled",
            "decisions",
        )
    )
    write_json_exclusive(QUALIFICATION, result)
    return result


__all__ = ["run_qualification"]
