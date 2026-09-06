"""Prospective paired Liu cohorts for the common-state pilot."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.experiments.adaptive_plasticity.data import model_tensors
from fsrl.experiments.adaptive_plasticity.model import load_model as load_parent_model
from fsrl.experiments.cohort_diagnostic.statistics import (
    interval_classification,
    reference_intervals,
    wilson,
)
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .encoding import encode_common
from .evidence import ARTIFACT_LOCK, validate_artifacts, validate_recovery
from .generic import LOCK as GENERIC_LOCK
from .generic import load_fit, validate_generic
from .inputs import LIU_MANIFEST, cohort_indices, load_cohorts, save_liu_inputs
from .protocol import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED_BASE,
    COHORT_SHARD_SIZE,
    COHORT_SIZE,
    COHORTS,
    CONDITIONS,
    PARENT_SEEDS,
    RECORDS,
    RUN_ROOT,
    SEEDS,
    cohort_specification,
    resolved_specification,
)

INPUT_LOCK = RECORDS / "benchmarks/liu_input_lock.json"
PARENT_ARTIFACT_LOCK = (
    REPO_ROOT / "studies/global_decay_replication/records/benchmarks/artifact_lock.json"
)
CLASS_NAMES = ("correct", "self_consistent_incorrect", "self_inconsistent")
PRESERVED_ENDPOINTS = (
    "serial_position_effect",
    "stable_within_subject_errors",
    "self_inconsistent",
    "inter_subject_ranking_diversity",
)
ALL_ENDPOINTS = (
    "learned_accuracy",
    "nonlearned_accuracy",
    "symbolic_distance_effect",
    "serial_position_effect",
    "stable_within_subject_errors",
    "self_consistent_incorrect",
    "self_inconsistent",
    "correct_ranker",
    "inter_subject_ranking_diversity",
)
HUMAN_COMPOSITION = np.asarray([8, 64, 5], dtype=np.float64) / 77


def lock_liu_inputs() -> dict:
    commit = require_pushed_clean()
    generic = validate_generic()
    if not generic["generic_gate_passed"]:
        raise RuntimeError(
            "generic confirmation failed; Liu evaluation is not admitted"
        )
    recovery = validate_recovery()
    manifest = (
        save_liu_inputs() if not LIU_MANIFEST.exists() else load_json(LIU_MANIFEST)
    )
    lock = {
        "source_commit": generic["source_commit"],
        "generic_commit": commit,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "parent_artifact_lock": reference(PARENT_ARTIFACT_LOCK),
        "rho_lock": recovery["lock_reference"],
        "selected_rho": recovery["selected_rho"],
        "generic_lock": reference(GENERIC_LOCK),
        "generic_result": generic["result"],
        "liu_manifest": reference(LIU_MANIFEST),
        "cohort_shards": manifest["shards"],
        "cohorts": COHORTS,
        "subjects": COHORT_SIZE,
        "model_rollout_performed": False,
    }
    write_json_exclusive(INPUT_LOCK, lock)
    return lock


def validate_input_lock() -> dict:
    validate_artifacts()
    generic = validate_generic()
    recovery = validate_recovery()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(INPUT_LOCK), commit=commit))
    if (
        lock["artifact_lock"] != reference(ARTIFACT_LOCK)
        or lock["parent_artifact_lock"] != reference(PARENT_ARTIFACT_LOCK)
        or lock["rho_lock"] != recovery["lock_reference"]
        or lock["selected_rho"] != recovery["selected_rho"]
        or lock["generic_lock"] != reference(GENERIC_LOCK)
        or lock["generic_result"] != generic["result"]
        or lock["cohorts"] != COHORTS
        or lock["subjects"] != COHORT_SIZE
        or lock["model_rollout_performed"]
    ):
        raise RuntimeError("common-encoding Liu input lock differs")
    verify_reference(lock["parent_artifact_lock"], commit=commit)
    manifest = load_json(verify_reference(lock["liu_manifest"], commit=commit))
    if (
        manifest["shards"] != lock["cohort_shards"]
        or manifest["model_rollout_performed"]
    ):
        raise RuntimeError("common-encoding Liu input manifest differs")
    if len(lock["cohort_shards"]) * COHORT_SHARD_SIZE != COHORTS:
        raise RuntimeError("common-encoding input lock omits mandatory cohorts")
    for position, row in enumerate(lock["cohort_shards"]):
        verify_reference(row, commit=commit)
        load_cohorts(row, position * COHORT_SHARD_SIZE)
    return lock


def true_positions(protocol: RankingProtocol) -> np.ndarray:
    result = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        result[item] = position
    return result


def inversion_counts(weights, codes, protocol: RankingProtocol) -> np.ndarray:
    scores = np.einsum("sd,sid->si", weights, codes, dtype=np.float64)
    positions = true_positions(protocol)
    result = np.zeros(len(weights), dtype=np.int16)
    for first, second in combinations(range(protocol.n_items), 2):
        higher, lower = (
            (first, second) if positions[first] < positions[second] else (second, first)
        )
        result += scores[:, higher] <= scores[:, lower]
    return result


def inversion_bin_fractions(values: np.ndarray) -> list[float]:
    return [
        float(np.mean(values == 0)),
        float(np.mean(values == 1)),
        float(np.mean((values >= 2) & (values <= 3))),
        float(np.mean(values >= 4)),
    ]


def sampled_behavior(margins: np.ndarray, index: int) -> dict:
    spec = cohort_specification(index)
    settings = spec["evaluation"]["liu"]
    protocol = load_registered_protocol(settings["protocol_id"])
    schedules = (ordered_pairs(protocol.n_items),) * len(margins)
    return analyze_sampled_query_policy(
        protocol,
        bundle_logits({"logits": margins}, schedules),
        seed=settings["choice_seed"],
        temperature=settings["temperature"],
    )


def cohort_point(margins, weights, batch: ModelBatch, index: int) -> dict:
    from fsrl.experiments.cohort_diagnostic.statistics import cohort_record

    protocol = load_registered_protocol("liu_v2")
    sampled = sampled_behavior(margins, index)
    behavior = cohort_record(sampled, human_references(cohort_specification(index)))
    inversions = inversion_counts(weights, batch.arrays["codes"], protocol)
    classes = np.asarray(
        [CLASS_NAMES.index(row["ranking_class"]) for row in sampled["subjects"]]
    )
    internal = inversions == 0
    transition = np.zeros((2, 3), dtype=np.float64)
    for subject in range(COHORT_SIZE):
        transition[0 if internal[subject] else 1, classes[subject]] += 1 / COHORT_SIZE
    composition = np.asarray(
        [
            behavior["values"][name]
            for name in (
                "correct_ranker",
                "self_consistent_incorrect",
                "self_inconsistent",
            )
        ]
    )
    return {
        **behavior,
        "internal_strict_correct": float(np.mean(internal)),
        "mean_inversion_count": float(inversions.mean()),
        "inversion_bin_fractions": inversion_bin_fractions(inversions),
        "internal_to_sampled_transition": transition.tolist(),
        "sampled_correct_all_subjects": float(np.mean(classes == 0)),
        "loss_flow": float(np.mean(internal & (classes != 0))),
        "rescue_flow": float(np.mean(~internal & (classes == 0))),
        "net_sampling_shift": float(np.mean(classes == 0) - np.mean(internal)),
        "ranking_composition_total_variation": float(
            0.5 * np.abs(composition - HUMAN_COMPOSITION).sum()
        ),
    }


def _parent_models(commit: str) -> tuple[dict, dict]:
    lock = load_json(verify_reference(reference(PARENT_ARTIFACT_LOCK), commit=commit))
    from fsrl.experiments.global_decay_replication.protocol import (
        resolved_specification as parent_specification,
    )

    spec = parent_specification()
    models = {}
    for seed in PARENT_SEEDS:
        archive = load_json(
            verify_reference(
                lock["runs"][f"{seed}/adaptive_eta_resampled"], commit=commit
            )
        )
        models[seed] = load_parent_model(archive["config"], spec, scheduler="global")
    return models, {seed: compiled(model) for seed, model in models.items()}


def _trained_models(artifacts: dict) -> tuple[dict, dict]:
    spec = resolved_specification()
    models = {
        f"{seed}/{condition}": load_fit(
            artifacts["archives"][f"{seed}/{condition}"]["config"], spec
        )
        for seed in SEEDS
        for condition in CONDITIONS
    }
    return models, {key: compiled(value) for key, value in models.items()}


def _output_directory(start: int):
    return RUN_ROOT / "liu-evaluation" / f"cohorts-{start:03d}"


def validate_output(record: dict, input_ref: dict, start: int) -> dict:
    if (
        record["input_lock"] != reference(INPUT_LOCK)
        or record["input"] != input_ref
        or record["cohort_indices"] != cohort_indices(start)
    ):
        raise RuntimeError("common-encoding Liu output shard identity differs")
    verify_reference(record["arrays"])
    return record


def evaluate_shard(
    input_ref: dict,
    start: int,
    rho: float,
    parent_runners: dict,
    trained_runners: dict,
    execution: dict,
) -> None:
    directory = _output_directory(start)
    if directory.exists():
        validate_complete(directory)
        validate_output(load_json(directory / "result.json"), input_ref, start)
        return
    identities = [
        *(
            f"fixed/{seed}/{condition}"
            for seed in PARENT_SEEDS
            for condition in CONDITIONS
        ),
        *(f"trained/{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS),
    ]
    with ProspectiveRun.start(
        directory,
        workflow_id="common_encoding_state_v1",
        execution_id=f"liu-cohorts-{start:03d}",
        producer={"module": __name__, "input_lock": reference(INPUT_LOCK)},
        resolved_config={"runtime": execution},
    ):
        points, output_rows = [], []
        for index, base, streams in load_cohorts(input_ref, start):
            outputs = {}
            encoded = {
                condition: encode_common(base, condition, rho, streams)[0]
                for condition in CONDITIONS
            }
            for condition, batch in encoded.items():
                arguments = model_tensors(batch, "cuda")
                for seed in PARENT_SEEDS:
                    with torch.no_grad():
                        margin, weights, _ = parent_runners[seed](*arguments)
                    outputs[f"fixed/{seed}/{condition}"] = {
                        "margins": margin.cpu().numpy().astype(np.float64),
                        "w": weights.cpu().numpy().astype(np.float64),
                    }
                for seed in SEEDS:
                    with torch.no_grad():
                        margin, weights, _ = trained_runners[f"{seed}/{condition}"](
                            *arguments
                        )
                    outputs[f"trained/{seed}/{condition}"] = {
                        "margins": margin.cpu().numpy().astype(np.float64),
                        "w": weights.cpu().numpy().astype(np.float64),
                    }
            points.append(
                {
                    "cohort": index,
                    "fits": {
                        identity: cohort_point(
                            outputs[identity]["margins"],
                            outputs[identity]["w"],
                            encoded[identity.split("/")[-1]],
                            index,
                        )
                        for identity in identities
                    },
                }
            )
            output_rows.append(outputs)
        arrays = {
            name: np.stack(
                [
                    np.stack([row[identity][name] for identity in identities])
                    for row in output_rows
                ]
            )
            for name in ("w", "margins")
        }
        arrays["fit_identities"] = np.asarray(identities)
        arrays["cohort_indices"] = np.asarray(cohort_indices(start), dtype=np.int64)
        path = directory / "outputs.npz"
        write_arrays(path, arrays)
        write_json_exclusive(
            directory / "result.json",
            json_ready(
                {
                    "input_lock": reference(INPUT_LOCK),
                    "input": input_ref,
                    "cohort_indices": cohort_indices(start),
                    "arrays": reference(path),
                    "points": points,
                }
            ),
        )


def evaluate_cohorts() -> dict:
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    parent_models, parent_runners = _parent_models(require_pushed_clean())
    trained_models, trained_runners = _trained_models(artifacts)
    execution = runtime()
    for position, input_ref in enumerate(lock["cohort_shards"]):
        start = position * COHORT_SHARD_SIZE
        evaluate_shard(
            input_ref,
            start,
            lock["selected_rho"],
            parent_runners,
            trained_runners,
            execution,
        )
        validate_complete(_output_directory(start))
        print(
            f"Completed paired common-state cohorts {start}..{start + COHORT_SHARD_SIZE - 1}",
            flush=True,
        )
    del parent_models, trained_models
    return {"cohorts": COHORTS, "selected_rho": lock["selected_rho"]}


def _statistic(values: np.ndarray, counts: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    means = counts @ values / len(values)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "interval": {"lower": float(low), "upper": float(high)},
    }


def _fit_summary(rows: list[dict], counts: np.ndarray, refs: dict) -> dict:
    continuous = {}
    for endpoint in ALL_ENDPOINTS:
        values = np.asarray([row["values"][endpoint] for row in rows], dtype=np.float64)
        statistic = _statistic(values, counts)
        continuous[endpoint] = {
            **statistic,
            "reference": refs[endpoint],
            "classification": interval_classification(
                statistic["interval"], refs[endpoint]
            ),
        }
    qualitative = [
        all(value["qualitative"] for value in row["flags"].values()) for row in rows
    ]
    quantitative = [
        all(value["calibration"] for value in row["flags"].values()) for row in rows
    ]
    core = wilson(qualitative)["lower"] > 0.90 and all(
        continuous[name]["classification"] == "mean_within_reference"
        for name in ("learned_accuracy", "nonlearned_accuracy")
    )
    preservation = all(
        continuous[name]["classification"] == "mean_within_reference"
        for name in PRESERVED_ENDPOINTS
    )
    return {
        "continuous": continuous,
        "all_nine_qualitative": wilson(qualitative),
        "all_nine_quantitative": wilson(quantitative),
        "joint_nine": wilson(np.asarray(qualitative) & np.asarray(quantitative)),
        "core_behavior_passed": core,
        "preservation_profile_passed": preservation,
        "internal_strict_correct": _statistic(
            np.asarray([row["internal_strict_correct"] for row in rows]), counts
        ),
        "mean_inversion_count": _statistic(
            np.asarray([row["mean_inversion_count"] for row in rows]), counts
        ),
        "inversion_bins": [
            _statistic(
                np.asarray([row["inversion_bin_fractions"][index] for row in rows]),
                counts,
            )
            for index in range(4)
        ],
        "internal_to_sampled": {
            name: _statistic(np.asarray([row[name] for row in rows]), counts)
            for name in (
                "sampled_correct_all_subjects",
                "loss_flow",
                "rescue_flow",
                "net_sampling_shift",
            )
        },
        "internal_to_sampled_transition": [
            [
                _statistic(
                    np.asarray(
                        [
                            row["internal_to_sampled_transition"][internal][category]
                            for row in rows
                        ]
                    ),
                    counts,
                )
                for category in range(3)
            ]
            for internal in range(2)
        ],
        "ranking_composition_total_variation": _statistic(
            np.asarray([row["ranking_composition_total_variation"] for row in rows]),
            counts,
        ),
    }


def _contrast(points: list[dict], source: str, seed: int, counts: np.ndarray) -> dict:
    relation = [row["fits"][f"{source}/{seed}/relation_common"] for row in points]
    episode = [row["fits"][f"{source}/{seed}/episode_common"] for row in points]
    endpoints = {
        "correct_ranker": lambda row: row["values"]["correct_ranker"],
        "self_consistent_incorrect": lambda row: row["values"][
            "self_consistent_incorrect"
        ],
        "self_inconsistent": lambda row: row["values"]["self_inconsistent"],
        "ranking_composition_total_variation": lambda row: row[
            "ranking_composition_total_variation"
        ],
        "internal_strict_correct": lambda row: row["internal_strict_correct"],
        "mean_inversion_count": lambda row: row["mean_inversion_count"],
    }
    result: dict = {
        name: _statistic(
            np.asarray(
                [
                    getter(candidate) - getter(control)
                    for candidate, control in zip(episode, relation, strict=True)
                ]
            ),
            counts,
        )
        for name, getter in endpoints.items()
    }
    result["gates"] = {
        "correct_ranker": result["correct_ranker"]["interval"]["lower"] > 0,
        "self_consistent_incorrect": result["self_consistent_incorrect"]["interval"][
            "upper"
        ]
        < 0,
        "ranking_composition_total_variation": result[
            "ranking_composition_total_variation"
        ]["interval"]["upper"]
        < 0,
    }
    result["passed"] = all(result["gates"].values())
    return result


def summarize_points(points: list[dict]) -> dict:
    if [row["cohort"] for row in points] != list(range(COHORTS)):
        raise RuntimeError("common-encoding summary requires every prospective cohort")
    refs = reference_intervals(human_references(resolved_specification()))
    fits, contrasts = {}, {"fixed": {}, "trained": {}}
    for source, seeds in (("fixed", PARENT_SEEDS), ("trained", SEEDS)):
        for seed in seeds:
            rng = np.random.default_rng(
                BOOTSTRAP_SEED_BASE + seed + (0 if source == "fixed" else 100_000)
            )
            counts = rng.multinomial(
                COHORTS, np.full(COHORTS, 1 / COHORTS), size=BOOTSTRAP_SAMPLES
            )
            for condition in CONDITIONS:
                identity = f"{source}/{seed}/{condition}"
                fits[identity] = _fit_summary(
                    [row["fits"][identity] for row in points], counts, refs
                )
            contrasts[source][str(seed)] = _contrast(points, source, seed, counts)
    trained_candidates = [fits[f"trained/{seed}/episode_common"] for seed in SEEDS]
    joint_allocation = all(
        row["continuous"]["self_inconsistent"]["classification"]
        == "mean_within_reference"
        for row in trained_candidates
    )
    preservation = all(
        row["core_behavior_passed"] and row["preservation_profile_passed"]
        for row in trained_candidates
    )
    fixed_passed = all(row["passed"] for row in contrasts["fixed"].values())
    trained_passed = all(row["passed"] for row in contrasts["trained"].values())
    if fixed_passed and trained_passed and joint_allocation and preservation:
        outcome = "common_encoding_supported"
    elif fixed_passed and not trained_passed:
        outcome = "fixed_effect_only"
    elif trained_passed and not fixed_passed:
        outcome = "coadaptation_only"
    else:
        outcome = "joint_structure_unresolved"
    return {
        "fits": fits,
        "contrasts": contrasts,
        "decision": {
            "fixed_encoder_specificity": fixed_passed,
            "trained_encoder_specificity": trained_passed,
            "joint_allocation": joint_allocation,
            "preservation": preservation,
            "outcome": outcome,
            "main_model_promoted": False,
        },
    }
