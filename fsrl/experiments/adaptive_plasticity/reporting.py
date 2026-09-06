"""Independent reconstruction and publication of the complete fixed pilot."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.quantized_learner.verification import reconstruct_codes
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .cohorts import (
    ALL_ENDPOINTS,
    INPUT_LOCK,
    _output_directory,
    cohort_point,
    summarize_points,
    validate_input_lock,
    validate_output,
)
from .evidence import validate_artifacts
from .generic_selection import validate_selection
from .inputs import cohort_indices, load_cohorts, read_arrays
from .protocol import (
    CODEBOOK,
    COHORT_SHARD_SIZE,
    COHORTS,
    CONDITIONS,
    DESIGN_HASH,
    RECORDS,
    SEEDS,
    resolved_specification,
)
from .reference import rollout

RESULT = RECORDS / "results/experience_dependent_plasticity_v1.json"
REPORT = RECORDS / "reports/experience_dependent_plasticity_v1.md"


def verify_shard(
    record: dict, input_ref: dict, start: int, artifacts: dict, scheduler: str
) -> dict:
    validate_output(record, input_ref, start)
    arrays = read_arrays(record["arrays"])
    identities = [
        f"{int(seed)}/{condition}"
        for seed, condition in zip(
            arrays["fit_seeds"], arrays["fit_conditions"], strict=True
        )
    ]
    expected_identities = [
        f"{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS
    ]
    if identities != expected_identities:
        raise RuntimeError("published fit order differs")
    np.testing.assert_array_equal(arrays["cohort_indices"], cohort_indices(start))
    maximum = 0.0
    for position, (index, base, uniforms) in enumerate(load_cohorts(input_ref, start)):
        signed, _, _ = reconstruct_codes(base, uniforms, "resampled", CODEBOOK)
        batch = ModelBatch(
            {**base.arrays, "signed": signed, "local_evidence": np.zeros_like(signed)}
        )
        for column, identity in enumerate(identities):
            config = artifacts["archives"][identity]["config"]
            condition = config["condition"]
            parameters = config["physical_parameters"]
            selected = (
                scheduler if condition == "adaptive_eta_resampled" else "relation"
            )
            expected = rollout(
                batch,
                eta=parameters["eta0"],
                gain=parameters["gamma_G"],
                epsilon=resolved_specification()["model"]["epsilon"],
                adaptive=condition == "adaptive_eta_resampled",
                scheduler=selected,
            )
            for name in ("w", "margins"):
                actual = arrays[name][position, column]
                np.testing.assert_allclose(actual, expected[name], atol=1e-5, rtol=1e-4)
                maximum = max(maximum, float(np.max(np.abs(actual - expected[name]))))
            rebuilt = json_ready(
                cohort_point(
                    arrays["margins"][position, column],
                    arrays["w"][position, column],
                    batch,
                    index,
                )
            )
            if rebuilt != record["points"][position]["fits"][identity]:
                raise RuntimeError("Liu cohort point does not reconstruct")
    return {
        "passed": True,
        "cohorts": COHORT_SHARD_SIZE,
        "max_recurrence_error": maximum,
    }


def _collect_runtime() -> tuple[list[dict], list[dict]]:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    shards, verification = [], []
    for position, input_ref in enumerate(lock["cohort_shards"]):
        start = position * COHORT_SHARD_SIZE
        directory = _output_directory(start)
        validate_complete(directory)
        shard = load_json(directory / "result.json")
        verification.append(
            verify_shard(
                shard,
                input_ref,
                start,
                artifacts,
                lock["selected_scheduler"],
            )
        )
        shards.append(shard)
        print(
            f"Reconstructed paired cohorts {start}..{start + COHORT_SHARD_SIZE - 1}",
            flush=True,
        )
    return shards, verification


def decision(fits: dict, generic: dict) -> dict:
    candidate = [fits[f"{seed}/adaptive_eta_resampled"] for seed in SEEDS]
    paired = [fits[str(seed)] for seed in SEEDS]
    targeted = generic["generic_gate_passed"] and all(
        row["gates"]["internal_strict_correct"] and row["gates"]["mean_inversion_count"]
        for row in paired
    )
    composition = all(
        row["gates"]["ranking_composition_total_variation"] for row in paired
    )
    behavior = all(
        row["core_behavior_passed"] and row["preservation_profile_passed"]
        for row in candidate
    )
    supported = targeted and composition and behavior
    if supported:
        outcome = (
            "relation_specific_supported"
            if generic["selected_scheduler"] == "relation"
            else "global_schedule_preferred"
        )
    elif targeted:
        outcome = "internal_only_support"
    elif composition:
        outcome = "behavior_without_targeted_mechanism"
    else:
        outcome = "valid_negative"
    full_every_cohort = all(
        row["all_nine_qualitative"]["successes"] == COHORTS
        and row["all_nine_quantitative"]["successes"] == COHORTS
        for row in candidate
    )
    return {
        "outcome": outcome,
        "targeted_internal_mechanism_passed": targeted,
        "paired_ranking_composition_passed": composition,
        "core_and_preservation_passed": behavior,
        "adaptive_mechanism_supported": supported,
        "relation_specific_supported": supported
        and generic["selected_scheduler"] == "relation",
        "legacy_full_nine_every_cohort": full_every_cohort,
        "main_model_promoted": False,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Experience-dependent score plasticity pilot",
        "",
        f"Registered outcome: `{result['decision']['outcome']}`. Selected scheduler: `{result['generic_selection']['selected_scheduler']}`.",
        "",
        "All six paired 2117--2119 fits and all 400 prospective 77-subject cohorts per fit are included. Participants are analyzed within each fit; no network or participant pooling is used.",
        "",
        "| Seed | eta fixed | eta0 adaptive | gamma fixed | gamma adaptive | Internal correct delta | Inversion delta | Rank-TV delta | Adaptive core | Preservation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed in SEEDS:
        fixed = result["fits"][f"{seed}/fixed_eta_resampled"]
        adaptive = result["fits"][f"{seed}/adaptive_eta_resampled"]
        paired = result["fits"][str(seed)]
        parameters = result["parameters"]
        lines.append(
            f"| {seed} | {parameters[f'{seed}/fixed_eta_resampled']['eta0']} | {parameters[f'{seed}/adaptive_eta_resampled']['eta0']} | {parameters[f'{seed}/fixed_eta_resampled']['gamma_G']} | {parameters[f'{seed}/adaptive_eta_resampled']['gamma_G']} | {paired['internal_strict_correct']} | {paired['mean_inversion_count']} | {paired['ranking_composition_total_variation']} | {adaptive['core_behavior_passed']} | {adaptive['preservation_profile_passed']} |"
        )
        lines.extend(
            [
                "",
                f"## Seed {seed}: continuous profile",
                "",
                "| Endpoint | Fixed mean (95% CI) | Adaptive mean (95% CI) | Human reference | Adaptive classification |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {name} | {fixed['continuous'][name]['mean']} {fixed['continuous'][name]['interval']} | {adaptive['continuous'][name]['mean']} {adaptive['continuous'][name]['interval']} | {adaptive['continuous'][name]['reference']} | {adaptive['continuous'][name]['classification']} |"
            for name in ALL_ENDPOINTS
        )
        lines.extend(
            [
                "",
                f"Adaptive all-nine qualitative stability: {adaptive['all_nine_qualitative']}; quantitative stability: {adaptive['all_nine_quantitative']}; joint: {adaptive['joint_nine']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Mechanism and claim boundary",
            "",
            f"Generic late-sensitivity and scheduler decision: `{result['generic_selection']['generic_gate_passed']}`; relation-specific support: `{result['generic_selection']['relation_specific_supported']}`. Exact occurrence sensitivities, conditional mean margins and independently propagated quantization variances are retained in the generic-selection arrays.",
            "",
            "The ranking-composition gate is a paired relative improvement in total variation, not equality of the complete human distribution. The unchanged nine-row profile remains fully reported. Internal strict order has no observed human latent-state counterpart.",
            "",
            "This fixed pilot cannot promote a main model. Fresh unchanged training replication and a finite-time circuit check of the selected efficacy rule remain outside this authorization.",
            "",
            result["stop_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def publish() -> dict:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    selection = validate_selection()["summary"]
    shards, verification = _collect_runtime()
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=True)
    published = []
    points = []
    for position, shard in enumerate(shards):
        start = position * COHORT_SHARD_SIZE
        arrays_path = destination / f"liu-outputs-{start:03d}.npz"
        write_arrays(arrays_path, read_arrays(shard["arrays"]))
        record = {**shard, "arrays": reference(arrays_path)}
        path = destination / f"liu-cohorts-{start:03d}.json"
        write_json_exclusive(path, record)
        published.append(reference(path))
        points.extend(shard["points"])
    fits = summarize_points(points)
    parameters = {
        identity: row["config"]["physical_parameters"]
        for identity, row in artifacts["archives"].items()
    }
    result = {
        "schema_version": 1,
        "experiment_id": "experience_dependent_plasticity_v1",
        "contract_sha256": DESIGN_HASH,
        "input_lock": reference(INPUT_LOCK),
        "selected_scheduler": lock["selected_scheduler"],
        "generic_selection": selection,
        "parameters": parameters,
        "fits": fits,
        "decision": decision(fits, selection),
        "verification": verification,
        "shards": published,
        "training_seeds": list(SEEDS),
        "cohorts_per_fit": COHORTS,
        "participants_pooled": False,
        "main_model_promoted": False,
        "stop_rule": "After this complete fixed pilot, do not tune the decay curve, codebook, temperature, loss, local branch, training length, checkpoints or seeds. A successor requires a separately frozen question and explicit authorization.",
    }
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.open("x").write(render_report(result))
    return {
        "outcome": result["decision"]["outcome"],
        "result": reference(RESULT),
        "report": reference(REPORT),
    }


def verify_record() -> dict:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    result = load_json(RESULT)
    if (
        result["contract_sha256"] != DESIGN_HASH
        or result["input_lock"] != reference(INPUT_LOCK)
        or result["selected_scheduler"] != lock["selected_scheduler"]
        or result["participants_pooled"]
        or result["main_model_promoted"]
    ):
        raise RuntimeError("published adaptive-plasticity result identity differs")
    shards, checks, points = [], [], []
    for position, (record_ref, input_ref) in enumerate(
        zip(result["shards"], lock["cohort_shards"], strict=True)
    ):
        start = position * COHORT_SHARD_SIZE
        shard = load_json(verify_reference(record_ref))
        checks.append(
            verify_shard(
                shard,
                input_ref,
                start,
                artifacts,
                lock["selected_scheduler"],
            )
        )
        shards.append(shard)
        points.extend(shard["points"])
    if checks != result["verification"]:
        raise RuntimeError("saved recurrence verification differs")
    fits = summarize_points(points)
    selection = validate_selection()["summary"]
    if (
        json_ready(fits) != result["fits"]
        or json_ready(decision(fits, selection)) != result["decision"]
    ):
        raise RuntimeError("published adaptive-plasticity summary differs")
    if REPORT.read_text() != render_report(result):
        raise RuntimeError("adaptive-plasticity report differs")
    return {
        "passed": True,
        "outcome": result["decision"]["outcome"],
        "fits": len(SEEDS) * len(CONDITIONS),
        "cohorts_per_fit": COHORTS,
        "max_recurrence_error": max(row["max_recurrence_error"] for row in checks),
    }
