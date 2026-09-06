"""Independent reconstruction and publication of the common-state pilot."""

from __future__ import annotations

import json

import numpy as np

from fsrl.experiments.adaptive_plasticity.reference import rollout
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .cohorts import (
    ALL_ENDPOINTS,
    INPUT_LOCK,
    PARENT_ARTIFACT_LOCK,
    _output_directory,
    cohort_point,
    summarize_points,
    validate_input_lock,
    validate_output,
)
from .encoding import encode_common
from .evidence import validate_artifacts, validate_recovery
from .generic import validate_generic
from .inputs import cohort_indices, load_cohorts, read_arrays
from .protocol import (
    COHORT_SHARD_SIZE,
    COHORTS,
    CONDITIONS,
    DESIGN_HASH,
    PARENT_SEEDS,
    RECORDS,
    SEEDS,
    resolved_specification,
)

RESULT = RECORDS / "results/common_encoding_state_v1.json"
REPORT = RECORDS / "reports/common_encoding_state_v1.md"


def _cell(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parent_archives(commit: str) -> dict:
    lock = load_json(verify_reference(reference(PARENT_ARTIFACT_LOCK), commit=commit))
    return {
        seed: load_json(
            verify_reference(
                lock["runs"][f"{seed}/adaptive_eta_resampled"], commit=commit
            )
        )
        for seed in PARENT_SEEDS
    }


def verify_shard(
    record: dict,
    input_ref: dict,
    start: int,
    rho: float,
    artifacts: dict,
    parent: dict,
) -> dict:
    validate_output(record, input_ref, start)
    arrays = read_arrays(record["arrays"])
    identities = [str(value) for value in arrays["fit_identities"]]
    expected = [
        *(
            f"fixed/{seed}/{condition}"
            for seed in PARENT_SEEDS
            for condition in CONDITIONS
        ),
        *(f"trained/{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS),
    ]
    if identities != expected:
        raise RuntimeError("published common-encoding fit order differs")
    np.testing.assert_array_equal(arrays["cohort_indices"], cohort_indices(start))
    maximum = 0.0
    epsilon = resolved_specification()["model"]["epsilon"]
    for position, (index, base, streams) in enumerate(load_cohorts(input_ref, start)):
        encoded = {
            condition: encode_common(base, condition, rho, streams)[0]
            for condition in CONDITIONS
        }
        for column, identity in enumerate(identities):
            source, seed_text, condition = identity.split("/")
            archive = (
                parent[int(seed_text)]
                if source == "fixed"
                else artifacts["archives"][f"{seed_text}/{condition}"]
            )
            parameters = archive["config"]["physical_parameters"]
            expected_output = rollout(
                encoded[condition],
                eta=parameters["eta0"],
                gain=parameters["gamma_G"],
                epsilon=epsilon,
                adaptive=True,
                scheduler="global",
            )
            for name in ("w", "margins"):
                actual = arrays[name][position, column]
                np.testing.assert_allclose(
                    actual, expected_output[name], atol=1e-5, rtol=1e-4
                )
                maximum = max(
                    maximum, float(np.max(np.abs(actual - expected_output[name])))
                )
            rebuilt = json_ready(
                cohort_point(
                    arrays["margins"][position, column],
                    arrays["w"][position, column],
                    encoded[condition],
                    index,
                )
            )
            if rebuilt != record["points"][position]["fits"][identity]:
                raise RuntimeError("common-encoding cohort point does not reconstruct")
    return {
        "passed": True,
        "cohorts": COHORT_SHARD_SIZE,
        "max_recurrence_error": maximum,
    }


def _collect_runtime() -> tuple[list[dict], list[dict]]:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    from fsrl.experiments.training_strategy.locks import require_pushed_clean

    parent = _parent_archives(require_pushed_clean())
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
                lock["selected_rho"],
                artifacts,
                parent,
            )
        )
        shards.append(shard)
        print(
            f"Reconstructed common-state cohorts {start}..{start + COHORT_SHARD_SIZE - 1}",
            flush=True,
        )
    return shards, verification


def render_report(result: dict) -> str:
    lines = [
        "# Marginally matched common encoding state",
        "",
        f"Registered outcome: `{result['decision']['outcome']}`.",
        "",
        f"The non-Liu recovery-selected perturbation was rho=`{result['selected_rho']}`. All 400 prospectively locked 77-subject cohorts were analyzed within each fit; networks and participants were not pooled.",
        "",
        "| Source | Seed | Correct delta | Self-consistent-error delta | Self-inconsistent delta | Composition-TV delta | Internal-correct delta | Inversion delta | Specificity PASS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source, seeds in (("fixed", PARENT_SEEDS), ("trained", SEEDS)):
        for seed in seeds:
            row = result["contrasts"][source][str(seed)]
            lines.append(
                f"| {source} | {seed} | {_cell(row['correct_ranker'])} | {_cell(row['self_consistent_incorrect'])} | {_cell(row['self_inconsistent'])} | {_cell(row['ranking_composition_total_variation'])} | {_cell(row['internal_strict_correct'])} | {_cell(row['mean_inversion_count'])} | {row['passed']} |"
            )
    for seed in SEEDS:
        candidate = result["fits"][f"trained/{seed}/episode_common"]
        control = result["fits"][f"trained/{seed}/relation_common"]
        lines.extend(
            [
                "",
                f"## Trained seed {seed}: episode-common versus relation-common",
                "",
                "| Endpoint | Relation mean (95% CI) | Episode mean (95% CI) | Human reference | Episode classification |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {name} | {control['continuous'][name]['mean']} {_cell(control['continuous'][name]['interval'])} | {candidate['continuous'][name]['mean']} {_cell(candidate['continuous'][name]['interval'])} | {_cell(candidate['continuous'][name]['reference'])} | {candidate['continuous'][name]['classification']} |"
            for name in ALL_ENDPOINTS
        )
        lines.extend(
            [
                "",
                f"Episode-common core: `{candidate['core_behavior_passed']}`; preservation: `{candidate['preservation_profile_passed']}`; all-nine qualitative: {_cell(candidate['all_nine_qualitative'])}; quantitative: {_cell(candidate['all_nine_quantitative'])}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The primary contrast is episode-common minus relation-common at matched rho. This isolates cross-relation dependence from within-relation repetition dependence while preserving every presentation's marginal quantization distribution.",
            "",
            "Complete nine-endpoint human-distribution equality is reported but is not the registered primary gate. A positive pilot would still require unchanged fresh replication and a separate biological-boundary test before main-model promotion.",
            "",
            result["stop_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def publish() -> dict:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    recovery = validate_recovery()
    generic = validate_generic()["result_record"]["summary"]
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
    summary = summarize_points(points)
    parameters = {
        identity: row["config"]["physical_parameters"]
        for identity, row in artifacts["archives"].items()
    }
    result = {
        "schema_version": 1,
        "experiment_id": "common_encoding_state_v1",
        "contract_sha256": DESIGN_HASH,
        "input_lock": reference(INPUT_LOCK),
        "rho_lock": recovery["lock_reference"],
        "selected_rho": lock["selected_rho"],
        "generic_confirmation": generic,
        "parameters": parameters,
        **summary,
        "published_shards": published,
        "verification": verification,
        "stop_rule": "Do not tune rho or repair the codebook, temperature, decay, loss, classifier, threshold, seeds or checkpoints after this result.",
    }
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.open("x").write(render_report(result))
    return result["decision"]


def verify_record() -> dict:
    result = load_json(verify_reference(reference(RESULT)))
    if REPORT.read_text() != render_report(result):
        raise RuntimeError("common-encoding report does not reconstruct")
    if len(result["published_shards"]) * COHORT_SHARD_SIZE != COHORTS:
        raise RuntimeError("common-encoding result omits cohort shards")
    return {"passed": True, "outcome": result["decision"]["outcome"]}
