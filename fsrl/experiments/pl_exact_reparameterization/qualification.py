"""Numerical qualification of the exact P/L reparameterization."""

from __future__ import annotations

import math

import numpy as np
import torch

from fsrl.core.factorized_plastic_rnn import (
    FactorizedPlasticRNN,
    FactorizedRecurrentSequence,
    factorize_legacy_inputs,
)
from fsrl.core.local_trace import ConjunctiveLocalTrace, PackedConjunctiveLocalTrace
from fsrl.core.sequence import RecurrentSequence
from fsrl.evaluation.sampling import deterministic_cue_codes
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import write_json
from fsrl.paths import REPO_ROOT, RUNS_ROOT
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator
from fsrl.training.legacy_checkpoints import load_frozen_retro_checkpoint

from .locks import SOURCE_LOCK_PATH, validate_source_lock
from .protocol import PROTOCOL_SHA256, load_specification

RUN_ROOT = RUNS_ROOT / "pl_exact_reparameterization_v1"
RESULT_PATH = RUN_ROOT / "qualification-attempt2" / "qualification.json"


def active_input_sequence(
    codes: np.ndarray,
    pairs: np.ndarray,
    evidence: np.ndarray,
    *,
    steps: int,
    time_value: float,
) -> torch.Tensor:
    """Build one legacy active-task trial without passive legacy signals."""

    batch_size = pairs.shape[0]
    inputs = np.zeros((steps, batch_size, 37), dtype=np.float32)
    inputs[:, :, 31] = 1.0
    inputs[:, :, 32] = time_value
    rows = np.arange(batch_size)
    inputs[0, :, :15] = codes[rows, pairs[:, 0]]
    inputs[0, :, 15:30] = codes[rows, pairs[:, 1]]
    inputs[0, :, 34] = evidence
    if steps > 1:
        inputs[1, :, 30] = 1.0
    return torch.from_numpy(inputs)


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _record_error(errors: dict[str, float], name: str, value: float) -> None:
    errors[name] = max(errors.get(name, 0.0), value)


def _support_panel(
    protocol: RankingProtocol,
    subjects: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple], np.ndarray]:
    codes = deterministic_cue_codes(
        subjects, protocol.n_items, 15, seed, mode="permuted_shared"
    )
    schedules = [
        protocol.support_schedule(np.random.default_rng(seed * 100 + subject))
        for subject in range(subjects)
    ]
    evidence = np.asarray(
        [[trial.signed_magnitude for trial in schedule] for schedule in schedules],
        dtype=np.float32,
    )
    return codes, schedules, evidence


def _generic_support_panel(
    subjects: int, seed: int
) -> tuple[np.ndarray, list[tuple], np.ndarray]:
    generator = GenericRankingTaskGenerator(
        n_items=8,
        cue_size=15,
        min_edges=10,
        max_edges=10,
        support_blocks=4,
    )
    episodes = [
        generator.sample(np.random.default_rng(seed * 100 + subject), n_edges=10)
        for subject in range(subjects)
    ]
    codes = np.stack([episode.item_codes for episode in episodes])
    schedules = [episode.support_trials for episode in episodes]
    evidence = np.asarray(
        [
            [
                trial.signed_magnitude * trial.encoding_reliability
                for trial in episode.support_trials
            ]
            for episode in episodes
        ],
        dtype=np.float32,
    )
    return codes, schedules, evidence


def _run_panel(
    legacy,
    factorized: FactorizedPlasticRNN,
    local_gain: float,
    protocol: RankingProtocol,
    *,
    device: torch.device,
    seed: int,
    panel_kind: str,
) -> dict:
    subjects = 4
    if panel_kind == "liu":
        codes, schedules, scheduled_evidence = _support_panel(protocol, subjects, seed)
    elif panel_kind == "generic":
        codes, schedules, scheduled_evidence = _generic_support_panel(subjects, seed)
    else:
        raise ValueError(f"unknown panel kind: {panel_kind}")
    support_trials = len(schedules[0])
    legacy_sequence = RecurrentSequence(legacy)
    factorized_sequence = FactorizedRecurrentSequence(factorized)
    full_local = ConjunctiveLocalTrace(15, device=device)
    packed_local = PackedConjunctiveLocalTrace(15, device=device)
    with torch.no_grad():
        full_local.raw_gain.fill_(local_gain)
        packed_local.raw_gain.fill_(local_gain)
    legacy_p = legacy.initial_fast_weights(subjects)
    factorized_p = factorized.initial_fast_weights(subjects)
    full_l = full_local.initial_state(subjects)
    packed_l = packed_local.initial_state(subjects)
    errors: dict[str, float] = {}

    relation_count = len(schedules[0])
    for trial_index in range(support_trials):
        source_index = trial_index % relation_count
        pairs = np.asarray(
            [
                (
                    schedules[subject][source_index].left_item,
                    schedules[subject][source_index].right_item,
                )
                for subject in range(subjects)
            ],
            dtype=np.int64,
        )
        evidence = scheduled_evidence[:, source_index]
        time_value = trial_index / max(1, support_trials - 1) * (2.0 / 3.0)
        legacy_inputs = active_input_sequence(
            codes, pairs, evidence, steps=4, time_value=time_value
        ).to(device)
        external, times = factorize_legacy_inputs(legacy_inputs, 15)
        with torch.no_grad():
            legacy_outputs = legacy_sequence(
                legacy_inputs,
                legacy.initial_hidden(subjects),
                legacy.initial_eligibility(subjects),
                legacy_p,
                True,
            )
            factorized_outputs = factorized_sequence(
                external,
                times,
                factorized.initial_hidden(subjects),
                factorized.initial_eligibility(subjects),
                factorized_p,
                True,
            )
            legacy_p = legacy_outputs[-1]
            factorized_p = factorized_outputs[-1]
            pair_cues = legacy_inputs[0, :, :30]
            values = torch.as_tensor(evidence, device=device)
            full_l = full_local.write(full_l, pair_cues, values)
            packed_l = packed_local.write(packed_l, pair_cues, values)
        legacy_margin = legacy_outputs[0][:, 1:2] - legacy_outputs[0][:, 0:1]
        for name, observed, target in (
            ("support_margin", factorized_outputs[0], legacy_margin),
            ("support_modulation", factorized_outputs[1], legacy_outputs[2]),
            ("support_hidden", factorized_outputs[2], legacy_outputs[3]),
            ("support_eligibility", factorized_outputs[3], legacy_outputs[4]),
            ("support_fast_weights", factorized_p, legacy_p),
        ):
            _record_error(errors, name, _max_error(observed, target))

    upper = torch.triu_indices(15, 15, offset=1, device=device)
    full_matrix = full_l.reshape(subjects, 15, 15)
    expected_packed = math.sqrt(2.0) * full_matrix[:, upper[0], upper[1]]
    _record_error(errors, "packed_local_state", _max_error(packed_l, expected_packed))

    pairs = np.asarray(ordered_pairs(protocol.n_items), dtype=np.int64)
    query_count = len(pairs)
    tiled_pairs = np.tile(pairs[:, None, :], (1, subjects, 1)).reshape(-1, 2)
    tiled_codes = np.tile(codes[None, :, :, :], (query_count, 1, 1, 1)).reshape(
        query_count * subjects, protocol.n_items, 15
    )
    legacy_inputs = active_input_sequence(
        tiled_codes,
        tiled_pairs,
        np.zeros(query_count * subjects, dtype=np.float32),
        steps=2,
        time_value=2.0 / 3.0,
    ).to(device)
    external, times = factorize_legacy_inputs(legacy_inputs, 15)
    repeated_legacy_p = legacy_p.repeat(query_count, 1, 1)
    repeated_factorized_p = factorized_p.repeat(query_count, 1, 1)
    pair_cues = legacy_inputs[0, :, :30]
    repeated_full_l = full_l.repeat(query_count, 1)
    repeated_packed_l = packed_l.repeat(query_count, 1)

    conditions = {
        "intact": (repeated_legacy_p, repeated_factorized_p, True),
        "P_off": (
            torch.zeros_like(repeated_legacy_p),
            torch.zeros_like(repeated_factorized_p),
            True,
        ),
        "L_off": (repeated_legacy_p, repeated_factorized_p, False),
    }
    mismatches: dict[str, int] = {}
    with torch.no_grad():
        for name, (old_p, new_p, local_enabled) in conditions.items():
            old = legacy_sequence(
                legacy_inputs,
                legacy.initial_hidden(len(tiled_pairs)),
                legacy.initial_eligibility(len(tiled_pairs)),
                old_p,
                False,
            )
            new = factorized_sequence(
                external,
                times,
                factorized.initial_hidden(len(tiled_pairs)),
                factorized.initial_eligibility(len(tiled_pairs)),
                new_p,
                False,
            )
            old_margin = old[0][:, 1:2] - old[0][:, 0:1]
            new_margin = new[0]
            if local_enabled:
                old_combined, old_raw, _, old_correction = full_local(
                    old[0], repeated_full_l, pair_cues
                )
                old_final = old_combined[:, 1:2] - old_combined[:, 0:1]
                new_final, new_raw, _, new_correction = packed_local(
                    new_margin, repeated_packed_l, pair_cues
                )
                _record_error(errors, "local_raw_margin", _max_error(new_raw, old_raw))
                _record_error(
                    errors,
                    "local_correction",
                    _max_error(new_correction, old_correction),
                )
            else:
                old_final = old_margin
                new_final = new_margin
            _record_error(
                errors, f"{name}_global_margin", _max_error(new_margin, old_margin)
            )
            _record_error(
                errors, f"{name}_final_margin", _max_error(new_final, old_final)
            )
            mismatches[name] = int(
                torch.count_nonzero((new_final > 0) != (old_final > 0))
            )

    return {
        "panel": panel_kind,
        "support_trials": support_trials,
        "support_microsteps": support_trials * 4,
        "effective_P_writes": support_trials * 2,
        "query_unordered_pairs": query_count // 2,
        "query_orientations": query_count,
        "query_microsteps_per_pair": 2,
        "max_absolute_errors": errors,
        "categorical_mismatches": mismatches,
    }


def _single_step_and_blank_checks(legacy, factorized, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(1701)
    batch_size = 3
    inputs = torch.randn(batch_size, 37, generator=generator)
    inputs[:, 31] = 1.0
    inputs[:, 33] = 0.0
    inputs[:, 35:] = 0.0
    hidden_size = factorized.model_config.hidden_size
    hidden = torch.randn(batch_size, hidden_size, generator=generator).to(device)
    eligibility = (
        0.01 * torch.randn(batch_size, hidden_size, hidden_size, generator=generator)
    ).to(device)
    fast_weights = (
        0.01 * torch.randn(batch_size, hidden_size, hidden_size, generator=generator)
    ).to(device)
    inputs = inputs.to(device)
    external, times = factorize_legacy_inputs(inputs, 15)
    with torch.no_grad():
        old = legacy(inputs, hidden, eligibility, fast_weights)
        new = factorized(external, times, hidden, eligibility, fast_weights)
    expected = (old[0][:, 1:2] - old[0][:, 0:1], *old[2:])
    errors = {
        name: _max_error(observed, target)
        for name, observed, target in zip(
            ("margin", "modulation", "hidden", "eligibility", "fast_weights"),
            new,
            expected,
            strict=True,
        )
    }
    blank = torch.zeros(2, batch_size, 37, device=device)
    with torch.no_grad():
        blank_result = RecurrentSequence(legacy)(
            blank,
            legacy.initial_hidden(batch_size),
            legacy.initial_eligibility(batch_size),
            fast_weights,
            True,
        )
    return {
        "max_absolute_errors": errors,
        "blank_fast_weight_max_absolute_error": _max_error(
            blank_result[-1], fast_weights
        ),
    }


def _parameter_mapping_errors(legacy, factorized) -> dict[str, float]:
    if not factorized.uses_legacy_numerics:
        raise RuntimeError("converted checkpoints require numerical compatibility")
    assert factorized.legacy_constant_weight is not None
    assert factorized.legacy_input_bias is not None
    assert factorized.legacy_output_weight is not None
    assert factorized.legacy_output_bias is not None
    return {
        "external_cues_and_response": _max_error(
            factorized.input_projection.weight[:, :31], legacy.i2h.weight[:, :31]
        ),
        "external_evidence": _max_error(
            factorized.input_projection.weight[:, 31], legacy.i2h.weight[:, 34]
        ),
        "folded_active_bias": _max_error(
            factorized.input_projection.bias,
            legacy.i2h.bias + legacy.i2h.weight[:, 31],
        ),
        "time_weight": _max_error(factorized.time_weight, legacy.i2h.weight[:, 32]),
        "recurrent_weight": _max_error(factorized.w, legacy.w),
        "plasticity_weight": _max_error(factorized.alpha, legacy.alpha),
        "eligibility_rate": _max_error(factorized.etaet, legacy.etaet),
        "modulation_scale": _max_error(factorized.modulation_scale, legacy.DAmult),
        "modulation_weight": _max_error(
            factorized.h2modulation.weight, legacy.h2DA.weight
        ),
        "modulation_bias": _max_error(factorized.h2modulation.bias, legacy.h2DA.bias),
        "margin_weight": _max_error(
            factorized.h2margin.weight,
            legacy.h2o.weight[1:2] - legacy.h2o.weight[0:1],
        ),
        "margin_bias": _max_error(
            factorized.h2margin.bias,
            legacy.h2o.bias[1:2] - legacy.h2o.bias[0:1],
        ),
        "legacy_constant_weight": _max_error(
            factorized.legacy_constant_weight, legacy.i2h.weight[:, 31]
        ),
        "legacy_input_bias": _max_error(factorized.legacy_input_bias, legacy.i2h.bias),
        "legacy_output_weight": _max_error(
            factorized.legacy_output_weight, legacy.h2o.weight
        ),
        "legacy_output_bias": _max_error(
            factorized.legacy_output_bias, legacy.h2o.bias
        ),
    }


def qualify_device(device_name: str, specification: dict) -> dict:
    device = torch.device(device_name)
    protocol = load_registered_protocol("liu_v2")
    rows = []
    tolerance = specification["qualification"]["tolerances"][
        "float32_max_absolute_error"
    ]
    for seed, source in specification["frozen_sources"].items():
        legacy, _, _ = load_frozen_retro_checkpoint(
            REPO_ROOT / source["checkpoint"], batch_size=4, device=device
        )
        legacy.eval()
        factorized = FactorizedPlasticRNN.from_legacy(legacy).eval()
        parameter_errors = _parameter_mapping_errors(legacy, factorized)
        elementary = _single_step_and_blank_checks(legacy, factorized, device)
        panels = [
            _run_panel(
                legacy,
                factorized,
                source["raw_local_gain"],
                protocol,
                device=device,
                seed=int(seed),
                panel_kind=panel_kind,
            )
            for panel_kind in ("liu", "generic")
        ]
        all_errors = [
            *parameter_errors.values(),
            *elementary["max_absolute_errors"].values(),
        ]
        all_errors.append(elementary["blank_fast_weight_max_absolute_error"])
        for panel in panels:
            all_errors.extend(panel["max_absolute_errors"].values())
        mismatch_count = sum(
            sum(panel["categorical_mismatches"].values()) for panel in panels
        )
        nonfinite_count = sum(not math.isfinite(value) for value in all_errors)
        rows.append(
            {
                "seed": int(seed),
                "parameter_mapping_max_absolute_errors": parameter_errors,
                "elementary": elementary,
                "panels": panels,
                "maximum_absolute_error": max(all_errors),
                "categorical_mismatch_count": mismatch_count,
                "nonfinite_count": nonfinite_count,
                "passed": max(all_errors) <= tolerance
                and mismatch_count == 0
                and nonfinite_count == 0,
            }
        )
    return {
        "device": device_name,
        "seeds": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run_qualification() -> dict:
    runtime = require_formal_runtime()
    lock = validate_source_lock()
    specification = load_specification()
    devices = ["cpu"]
    if not torch.cuda.is_available():
        raise RuntimeError("qualification requires the registered CUDA eager replay")
    devices.append("cuda")
    results = [qualify_device(device, specification) for device in devices]
    result = {
        "schema_version": 1,
        "attempt": 2,
        "study_id": specification["study_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": {
            "path": SOURCE_LOCK_PATH.relative_to(REPO_ROOT).as_posix(),
            "source_commit": lock["source_commit"],
        },
        "timestep_accounting": specification["complete_timestep_accounting"],
        "runtime": runtime,
        "devices": results,
        "passed": all(row["passed"] for row in results),
    }
    write_json(RESULT_PATH, result)
    return result
