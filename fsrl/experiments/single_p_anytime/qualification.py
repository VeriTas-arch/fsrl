"""Pre-outcome qualification for single-P anytime V1."""

from __future__ import annotations

import copy

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.clean_single_p.model import AffineSinglePSequence, map_shadow
from fsrl.experiments.clean_single_p.optimization import (
    forward_batch,
    make_optimizer,
    training_step,
)
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import compile_module

from .decisions import classify, interval_gate
from .locks import sources
from .protocol import (
    PROTOCOL_SHA256,
    QUALIFICATION,
    historical_recipe,
    optimizer_specification,
    specification,
)
from .schedule import build_schedule, schedule_sha256, validate_schedule
from .streams import generate_max_stream, make_generator, prefix_batch
from .trajectory import nested_forward


def _stream_checks(task: dict) -> dict:
    generator = make_generator(task, support_blocks=6)
    stream = generate_max_stream(
        generator,
        network_seed=9301,
        update=0,
        edge_count=7,
        batch_size=3,
    )
    fingerprints = {}
    prefix_exact = True
    for blocks in range(2, 7):
        clean = prefix_batch(stream.clean, edge_count=7, blocks=blocks)
        noisy = prefix_batch(stream.noisy, edge_count=7, blocks=blocks)
        fingerprints[str(blocks)] = {
            "clean": clean.fingerprint(),
            "noisy": noisy.fingerprint(),
        }
        trials = blocks * 7
        for name in (
            "support_inputs",
            "local_evidence",
            "signed_magnitudes",
            "retention",
            "probabilities",
            "support_pairs",
            "realized_q",
        ):
            prefix_exact &= np.array_equal(
                noisy.arrays[name], stream.noisy.arrays[name][:trials]
            )
    regenerated = generate_max_stream(
        generator,
        network_seed=9301,
        update=0,
        edge_count=7,
        batch_size=3,
    )
    next_first = generate_max_stream(
        generator,
        network_seed=9301,
        update=1,
        edge_count=8,
        batch_size=3,
    )
    # Prefixing short and long views cannot consume or alter the next update RNG.
    prefix_batch(stream.noisy, edge_count=7, blocks=2)
    prefix_batch(stream.noisy, edge_count=7, blocks=6)
    next_second = generate_max_stream(
        generator,
        network_seed=9301,
        update=1,
        edge_count=8,
        batch_size=3,
    )
    result = {
        "task_fingerprint": stream.task_fingerprint,
        "clean_fingerprint": stream.clean.fingerprint(),
        "noisy_fingerprint": stream.noisy.fingerprint(),
        "prefix_fingerprints": fingerprints,
        "prefix_arrays_are_exact_slices": bool(prefix_exact),
        "unused_noisy_blocks_5_6_exist": stream.noisy.arrays["realized_q"].shape[0]
        == 42,
        "deterministic_regeneration": (
            stream.task_fingerprint == regenerated.task_fingerprint
            and stream.clean.fingerprint() == regenerated.clean.fingerprint()
            and stream.noisy.fingerprint() == regenerated.noisy.fingerprint()
        ),
        "next_update_independent_of_prefix": (
            next_first.task_fingerprint == next_second.task_fingerprint
            and next_first.clean.fingerprint() == next_second.clean.fingerprint()
            and next_first.noisy.fingerprint() == next_second.noisy.fingerprint()
        ),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "prefix_arrays_are_exact_slices",
            "unused_noisy_blocks_5_6_exist",
            "deterministic_regeneration",
            "next_update_independent_of_prefix",
        )
    )
    return result


def _nested_checks(task: dict) -> dict:
    generator = make_generator(task, support_blocks=7)
    stream = generate_max_stream(
        generator,
        network_seed=9302,
        update=0,
        edge_count=7,
        batch_size=3,
        validation=True,
    )
    full, full_times = stream.noisy.to("cpu")
    torch.manual_seed(9302)
    shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cpu")
    model = map_shadow(shadow, "clean_no_time")
    sequence = AffineSinglePSequence(model)
    nested = nested_forward(model, sequence, full, full_times, edge_count=7)
    p_error = margin_error = 0.0
    for blocks in range(1, 8):
        cpu = prefix_batch(stream.noisy, edge_count=7, blocks=blocks)
        batch, times = cpu.to("cpu")
        independent = forward_batch(
            "clean_no_time", model, sequence, batch, times, penalty=1e-4
        )
        p_error = max(
            p_error,
            float(
                torch.max(
                    torch.abs(
                        nested.prefix_weights[blocks - 1] - independent.fast_weights
                    )
                ).detach()
            ),
        )
        margin_error = max(
            margin_error,
            float(
                torch.max(
                    torch.abs(nested.margins[blocks - 1] - independent.margins[:, 0])
                ).detach()
            ),
        )
    return {
        "prefix_P_max_abs_error": p_error,
        "prefix_margin_max_abs_error": margin_error,
        "saved_prefixes": len(nested.prefix_weights),
        "passed": p_error == 0.0 and margin_error <= 1e-6,
    }


def _cuda_parity(task: dict, parent: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("single-P anytime qualification requires CUDA")
    generator = make_generator(task, support_blocks=6)
    errors = {}
    passed = True
    for blocks in (2, 6):
        stream = generate_max_stream(
            generator,
            network_seed=9303,
            update=blocks,
            edge_count=7,
            batch_size=3,
        )
        cpu = prefix_batch(stream.noisy, edge_count=7, blocks=blocks)
        batch, times = cpu.to("cuda")
        torch.manual_seed(9303)
        shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cuda")
        eager = map_shadow(shadow, "clean_no_time")
        compiled = copy.deepcopy(eager)
        first = training_step(
            "clean_no_time",
            eager,
            AffineSinglePSequence(eager),
            batch,
            times,
            make_optimizer(eager, parent),
            parent,
        )
        second = training_step(
            "clean_no_time",
            compiled,
            compile_module(AffineSinglePSequence(compiled), PROFILE),
            batch,
            times,
            make_optimizer(compiled, parent),
            parent,
        )
        output_error = max(
            float(torch.max(torch.abs(first.margins - second.margins)).detach()),
            float(
                torch.max(torch.abs(first.fast_weights - second.fast_weights)).detach()
            ),
            float(torch.abs(first.loss - second.loss).detach()),
        )
        parameter_error = max(
            float(torch.max(torch.abs(left - right)).detach())
            for left, right in zip(
                eager.parameters(), compiled.parameters(), strict=True
            )
        )
        errors[str(blocks)] = {
            "output_max_abs_error": output_error,
            "parameter_max_abs_error": parameter_error,
        }
        passed &= output_error <= 1e-5 and parameter_error <= 1e-5
    return {"horizons": errors, "passed": passed}


def _decision_checks() -> dict:
    expected = {
        "noninterpretable": {
            "integrity": False,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": True,
        },
        "training_recipe_failure": {
            "integrity": True,
            "fresh_fixed_valid": False,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": True,
        },
        "historical_task_degradation": {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": False,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": True,
        },
        "anytime_competence_failure": {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": False,
            "short_improved": True,
            "long_improved": True,
        },
        "no_horizon_benefit": {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": False,
            "long_improved": False,
        },
        "horizon_tradeoff": {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": False,
        },
        "anytime_admitted": {
            "integrity": True,
            "fresh_fixed_valid": True,
            "historical_preserved": True,
            "anytime_competent": True,
            "short_improved": True,
            "long_improved": True,
        },
    }
    observed = {name: classify(**values) for name, values in expected.items()}
    strict = (
        interval_gate(ce_upper=-1e-9, probability_lower=-0.02)
        and not interval_gate(ce_upper=0.0, probability_lower=-0.02)
        and not interval_gate(ce_upper=-1e-9, probability_lower=-0.0200001)
    )
    return {
        "observed": observed,
        "strict_endpoint_boundaries": strict,
        "passed": observed == {name: name for name in expected} and strict,
    }


def run_qualification() -> dict:
    spec = specification()
    parent = optimizer_specification()
    task = historical_recipe(1)["task"]
    schedules = {}
    for seed in spec["design"]["network_seeds"]:
        schedule = build_schedule(seed)
        summary = validate_schedule(schedule)
        regenerated = build_schedule(seed)
        summary["deterministic_regeneration"] = np.array_equal(schedule, regenerated)
        modified = schedule.copy()
        modified[0] = modified[1]
        summary["hash_sensitive"] = schedule_sha256(modified) != schedule_sha256(
            schedule
        )
        summary["passed"] = (
            summary["deterministic_regeneration"] and summary["hash_sensitive"]
        )
        schedules[str(seed)] = summary
    torch.manual_seed(9304)
    shadow = LinearModulationRNN(RetroModelConfig(38, 200, 2, 3), device="cpu")
    model = map_shadow(shadow, "clean_no_time")
    counts = {
        "parameters": sum(value.numel() for value in model.parameters()),
        "input_size": model.model_config.input_size,
        "hidden_size": model.model_config.hidden_size,
        "has_time_parameter": "time_weight" in dict(model.named_parameters()),
        "buffers": len(list(model.named_buffers())),
    }
    stream = _stream_checks(task)
    nested = _nested_checks(task)
    cuda = _cuda_parity(task, parent)
    decisions = _decision_checks()
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": configure_execution(),
        "schedules": schedules,
        "streams": stream,
        "architecture": counts,
        "nested_evaluation": nested,
        "cuda_parity": cuda,
        "decisions": decisions,
    }
    result["passed"] = (
        all(row["passed"] for row in schedules.values())
        and stream["passed"]
        and nested["passed"]
        and cuda["passed"]
        and decisions["passed"]
        and counts
        == {
            "parameters": 87003,
            "input_size": 32,
            "hidden_size": 200,
            "has_time_parameter": False,
            "buffers": 0,
        }
    )
    write_json_exclusive(QUALIFICATION, result)
    return result


__all__ = ["run_qualification"]
