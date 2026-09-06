"""Independent reconstruction and publication of the fresh replication."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.adaptive_plasticity.reference import rollout
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
from .generic import validate_generic
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

RESULT = RECORDS / "results/global_decay_replication_v1.json"
REPORT = RECORDS / "reports/global_decay_replication_v1.md"


def verify_shard(record: dict, input_ref: dict, start: int, artifacts: dict) -> dict:
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
        raise RuntimeError("published replication fit order differs")
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
            scheduler = (
                "global" if condition == "adaptive_eta_resampled" else "relation"
            )
            expected = rollout(
                batch,
                eta=parameters["eta0"],
                gain=parameters["gamma_G"],
                epsilon=resolved_specification()["model"]["epsilon"],
                adaptive=condition == "adaptive_eta_resampled",
                scheduler=scheduler,
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
                raise RuntimeError("replication Liu cohort point does not reconstruct")
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
        verification.append(verify_shard(shard, input_ref, start, artifacts))
        shards.append(shard)
        print(
            f"Reconstructed paired cohorts {start}..{start + COHORT_SHARD_SIZE - 1}",
            flush=True,
        )
    return shards, verification


def decision(fits: dict, generic: dict) -> dict:
    candidate = [fits[f"{seed}/adaptive_eta_resampled"] for seed in SEEDS]
    paired = [fits[str(seed)] for seed in SEEDS]
    internal = all(row["gates"]["internal_strict_correct"] for row in paired)
    inversion = all(row["gates"]["mean_inversion_count"] for row in paired)
    composition = all(
        row["gates"]["ranking_composition_total_variation"] for row in paired
    )
    preservation = all(
        row["core_behavior_passed"] and row["preservation_profile_passed"]
        for row in candidate
    )
    replicated = (
        generic["generic_gate_passed"]
        and internal
        and inversion
        and composition
        and preservation
    )
    return {
        "outcome": "replicated_global_decay"
        if replicated
        else "heterogeneous_replication",
        "generic_prerequisite_passed": generic["generic_gate_passed"],
        "internal_strict_correct_replicated": internal,
        "mean_inversion_count_replicated": inversion,
        "ranking_composition_replicated": composition,
        "core_and_preservation_replicated": preservation,
        "replicated_global_decay": replicated,
        "main_model_promoted": False,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Fresh global-decay replication",
        "",
        f"Registered outcome: `{result['decision']['outcome']}`.",
        "",
        "All six paired 2120--2122 fits and all 400 prospective 77-subject cohorts per fit are included. Participants are analyzed within each fit; no network or participant pooling is used.",
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
            f"Generic competence and late-sensitivity confirmation: `{result['generic_confirmation']['generic_gate_passed']}`. Exact occurrence sensitivities and mean/variance decomposition are retained in the locked generic arrays.",
            "",
            "The primary claim is unchanged replication of a global blockwise diminishing-plasticity effect. Complete nine-row quantitative equality remains a mandatory report but is not a primary replication gate.",
            "",
            "This result neither reopens relation familiarity nor promotes a main model. Common encoding state and finite-time circuit realization remain separate prospective questions.",
            "",
            result["stop_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def publish() -> dict:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    generic = validate_generic()["summary"]
    shards, verification = _collect_runtime()
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=True)
    published, points = [], []
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
        "experiment_id": "global_decay_replication_v1",
        "contract_sha256": DESIGN_HASH,
        "input_lock": reference(INPUT_LOCK),
        "selected_scheduler": lock["selected_scheduler"],
        "generic_confirmation": generic,
        "parameters": parameters,
        "fits": fits,
        "decision": decision(fits, generic),
        "published_shards": published,
        "verification": verification,
        "stop_rule": "After the fixed replication, do not tune or rerun the candidate and do not use these fresh results to choose a common-state encoder.",
    }
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.open("x").write(render_report(result))
    return result["decision"]


def verify_record() -> dict:
    result = load_json(verify_reference(reference(RESULT)))
    if REPORT.read_text() != render_report(result):
        raise RuntimeError("replication report does not reconstruct")
    if result["decision"] != decision(result["fits"], result["generic_confirmation"]):
        raise RuntimeError("replication decision does not reconstruct")
    if len(result["published_shards"]) * COHORT_SHARD_SIZE != COHORTS:
        raise RuntimeError("replication result omits cohort shards")
    return {"passed": True, "outcome": result["decision"]["outcome"]}
