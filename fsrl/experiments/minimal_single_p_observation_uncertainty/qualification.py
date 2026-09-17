"""Pre-outcome qualification for the acute M2 observation intervention."""

from __future__ import annotations

import copy

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence, make_model
from fsrl.experiments.minimal_single_p_promotion.adapter import evaluation_adapter
from fsrl.experiments.minimal_single_p_promotion.decisions import ROWS
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import compile_module

from .decisions import (
    compact_panel,
    expected_direction_supported,
    paired_contrast,
    study_outcome,
)
from .locks import sources, validate_parent
from .observations import encode, epsilon_for, observation_checks
from .protocol import PROTOCOL_SHA256, QUALIFICATION, specification


def _fixture() -> EpisodeBatch:
    rng = np.random.default_rng(99101)
    trials, subjects = 8, 5
    signed = rng.choice(
        (-2.0 / 7.0, -1.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0), size=(trials, subjects)
    )
    retention = rng.integers(0, 2, size=(trials, subjects)).astype(float)
    local = signed * (0.5 + 0.5 * retention)
    support = rng.standard_normal((trials, 4, subjects, 38)).astype(np.float32)
    support[:, 0, :, 37] = local
    support[:, 0, :, 34] = retention * local
    return EpisodeBatch(
        {
            "support_inputs": support,
            "local_evidence": local.astype(np.float32),
            "query_inputs": rng.standard_normal((2, 56 * subjects, 38)).astype(
                np.float32
            ),
            "targets": rng.integers(0, 2, size=56 * subjects, dtype=np.int64),
            "support_pairs": rng.integers(
                0, 8, size=(trials, subjects, 2), dtype=np.int64
            ),
            "query_pairs": np.asarray([(0, 1)] * 56, dtype=np.int64),
            "retention": retention.astype(bool),
            "signed_magnitudes": signed,
            "trial_retention": retention,
            "probabilities": np.full((trials, subjects), 0.5),
            "item_codes": rng.standard_normal((subjects, 8, 15)).astype(np.float32),
        }
    )


def _observation_operator() -> dict:
    cpu = _fixture()
    epsilon = epsilon_for(cpu, 99102)
    clean = encode(cpu, "clean", 1 / 7, epsilon)
    folded = encode(cpu, "folded", 1 / 7, epsilon)
    noisy = encode(cpu, "noisy", 1 / 7, epsilon)
    zero = encode(cpu, "noisy", 0, epsilon)
    checks = observation_checks(clean, folded, noisy)
    unchanged = tuple(
        name for name in cpu.arrays if name not in {"support_inputs", "local_evidence"}
    )
    deterministic = (
        encode(cpu, "noisy", 1 / 7, epsilon).fingerprint() == noisy.fingerprint()
    )
    passed = (
        clean.fingerprint() == cpu.fingerprint()
        and zero.fingerprint() == cpu.fingerprint()
        and checks["absolute_amplitudes_equal"]
        and checks["folded_signs_restored"]
        and all(
            np.array_equal(cpu.arrays[name], noisy.arrays[name]) for name in unchanged
        )
        and np.array_equal(
            cpu.arrays["support_inputs"][:, 1:], noisy.arrays["support_inputs"][:, 1:]
        )
        and deterministic
    )
    return {**checks, "deterministic": deterministic, "passed": passed}


def _max_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)).detach())


def _adapter(device: str, *, compiled: bool) -> dict:
    torch.manual_seed(99103)
    native = make_model("M2", 99104, device=device)
    adapter = evaluation_adapter(copy.deepcopy(native))
    native_sequence = MinimalSinglePSequence(native)
    adapter_sequence = RecurrentSequence(adapter)
    if compiled:
        native_sequence = compile_module(native_sequence, PROFILE)
        adapter_sequence = compile_module(adapter_sequence, PROFILE)
    inputs = torch.randn(4, 5, 32, device=device)
    legacy = torch.zeros(4, 5, 38, device=device)
    legacy[..., :31] = inputs[..., :31]
    legacy[..., 37] = inputs[..., 31]
    native_values = native_sequence(
        inputs,
        native.initial_hidden(5),
        native.initial_eligibility(5),
        native.initial_fast_weights(5),
        True,
    )
    adapter_values = adapter_sequence(
        legacy,
        adapter.initial_hidden(5),
        adapter.initial_eligibility(5),
        adapter.initial_fast_weights(5),
        True,
    )
    changed = legacy.clone()
    changed[..., 34] = torch.randn_like(changed[..., 34])
    bookkeeping_values = adapter_sequence(
        changed,
        adapter.initial_hidden(5),
        adapter.initial_eligibility(5),
        adapter.initial_fast_weights(5),
        True,
    )
    errors = {
        "margin": _max_error(
            native_values[0], adapter_values[0][:, 1:2] - adapter_values[0][:, 0:1]
        ),
        "modulation": _max_error(native_values[1], adapter_values[2]),
        "hidden": _max_error(native_values[2], adapter_values[3]),
        "eligibility": _max_error(native_values[3], adapter_values[4]),
        "P": _max_error(native_values[4], adapter_values[5]),
        "bookkeeping_invariance": _max_error(adapter_values[0], bookkeeping_values[0]),
    }
    tolerance = 1e-5 if device == "cuda" else 1e-6
    return {
        "errors": errors,
        "tolerance": tolerance,
        "passed": max(errors.values()) <= tolerance,
    }


def _decisions() -> dict:
    first = np.arange(20, dtype=float) + 1
    second = np.arange(20, dtype=float)
    positive = paired_contrast(first, second, seed=99105, samples=2000)
    negative = paired_contrast(second, first, seed=99106, samples=2000)
    undefined = paired_contrast(
        np.asarray([1.0, np.nan]), np.asarray([0.0, 0.0]), seed=99107, samples=20
    )
    outcomes = {
        "none": study_outcome(0, True, True),
        "specific": study_outcome(1, True, True),
        "nonspecific": study_outcome(1, True, False),
    }
    flags = {name: {"qualitative": True, "calibration": False} for name in ROWS}
    liu = {
        "routes": {
            "full": {
                "behavior": {
                    "historical_nine_rows": {
                        "eligible_subjects": 77,
                        "analysis_subjects_excluding_correct_rankers": 70,
                        "flags": flags,
                        "metrics": {
                            "difficult_pair_bimodality": {"bimodal": 15},
                            "stable_within_subject_errors": {"point": 0.8},
                            "inter_subject_ranking_diversity": {"point": 0.7},
                        },
                    }
                }
            }
        },
        "effects": {
            "intact_minus_evidence_shuffle_learned": {
                "mean": 0.2,
                "bootstrap": {"lower": 0.1, "upper": 0.3},
            }
        },
    }
    complete = compact_panel(liu)
    failed_liu = copy.deepcopy(liu)
    failed_liu["routes"]["full"]["behavior"]["historical_nine_rows"]["flags"][
        "learned_accuracy"
    ]["qualitative"] = False
    preservation_failure = compact_panel(failed_liu)
    design = specification()["design"]
    identities_exact = (
        len(design["network_seeds"]) == 20
        and design["evaluation_panels"] == [1, 2, 3]
        and design["conditions"] == ["clean", "folded", "noisy"]
    )
    passed = (
        expected_direction_supported("stable_within_subject_errors", positive)
        and expected_direction_supported("inter_subject_ranking_diversity", negative)
        and not undefined["defined"]
        and outcomes
        == {
            "none": "no_acute_completion",
            "specific": "direction_specific_acute_completion",
            "nonspecific": "acute_completion_without_direction_specificity",
        }
        and complete["structured_complete"]
        and not preservation_failure["structured_complete"]
        and identities_exact
    )
    return {
        "positive": positive,
        "negative": negative,
        "undefined": undefined,
        "outcomes": outcomes,
        "structured_completion": complete["structured_complete"],
        "preservation_failure_rejected": not preservation_failure[
            "structured_complete"
        ],
        "identities_exact": identities_exact,
        "passed": passed,
    }


def run_qualification() -> dict:
    runtime = configure_execution()
    parent_models, generic, parent_source = validate_parent()
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": runtime,
        "parent_identity": {
            "models": len(parent_models["runs"]),
            "generic_networks": len(generic["network_results"]),
            "panels": len(parent_source["panels"]),
            "passed": (
                len(parent_models["runs"]) == 20
                and len(generic["network_results"]) == 20
                and len(parent_source["panels"]) == 3
            ),
        },
        "observation_operator": _observation_operator(),
        "adapter_cpu": _adapter("cpu", compiled=False),
        "adapter_cpu_compiled": _adapter("cpu", compiled=True),
        "adapter_cuda_eager": _adapter("cuda", compiled=False),
        "adapter_cuda_compiled": _adapter("cuda", compiled=True),
        "decisions": _decisions(),
        "scientific_outcomes_exposed": False,
        "qualification_fixture_only": True,
        "sigma": specification()["observation"]["sigma"],
    }
    result["passed"] = all(
        result[name]["passed"]
        for name in (
            "parent_identity",
            "observation_operator",
            "adapter_cpu",
            "adapter_cpu_compiled",
            "adapter_cuda_eager",
            "adapter_cuda_compiled",
            "decisions",
        )
    )
    write_json_exclusive(QUALIFICATION, result)
    return result


__all__ = ["run_qualification"]
