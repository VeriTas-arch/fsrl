"""Prospective paired Liu cohorts, rollout and claim-relative statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.experiments.cohort_diagnostic.statistics import (
    interval_classification,
    reference_intervals,
    wilson,
)
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .data import model_tensors
from .evidence import ARTIFACT_LOCK, validate_artifacts
from .generic_selection import SELECTION_LOCK, validate_selection
from .inputs import (
    LIU_MANIFEST,
    cohort_indices,
    load_cohorts,
    save_liu_inputs,
)
from .model import load_model
from .protocol import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED_BASE,
    CODEBOOK,
    COHORT_SHARD_SIZE,
    COHORT_SIZE,
    COHORTS,
    CONDITIONS,
    RECORDS,
    RUN_ROOT,
    SEEDS,
    cohort_specification,
    resolved_specification,
)

INPUT_LOCK = RECORDS / "benchmarks/liu_input_lock.json"
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
    selection = validate_selection()
    if not selection["summary"]["generic_gate_passed"]:
        raise RuntimeError("generic gate failed; Liu evaluation is not admitted")
    manifest = (
        save_liu_inputs() if not LIU_MANIFEST.exists() else load_json(LIU_MANIFEST)
    )
    lock = {
        "source_commit": selection["source_commit"],
        "selection_commit": commit,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "selection_lock": reference(SELECTION_LOCK),
        "selection_result": selection["result"],
        "selected_scheduler": selection["selected_scheduler"],
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
    selection = validate_selection()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(INPUT_LOCK), commit=commit))
    if (
        lock["artifact_lock"] != reference(ARTIFACT_LOCK)
        or lock["selection_lock"] != reference(SELECTION_LOCK)
        or lock["selection_result"] != selection["result"]
        or lock["selected_scheduler"] != selection["selected_scheduler"]
        or lock["cohorts"] != COHORTS
        or lock["subjects"] != COHORT_SIZE
        or lock["model_rollout_performed"]
    ):
        raise RuntimeError("adaptive-plasticity Liu input lock differs")
    manifest = load_json(verify_reference(lock["liu_manifest"], commit=commit))
    if (
        manifest["shards"] != lock["cohort_shards"]
        or manifest["model_rollout_performed"]
    ):
        raise RuntimeError("Liu input manifest differs")
    if len(lock["cohort_shards"]) * COHORT_SHARD_SIZE != COHORTS:
        raise RuntimeError("Liu input lock omits mandatory cohorts")
    for position, row in enumerate(lock["cohort_shards"]):
        verify_reference(row, commit=commit)
        load_cohorts(row, position * COHORT_SHARD_SIZE)
    return lock


def true_positions(protocol: RankingProtocol) -> np.ndarray:
    result = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        result[item] = position
    return result


def inversion_counts(
    weights: np.ndarray, codes: np.ndarray, protocol: RankingProtocol
) -> np.ndarray:
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


def cohort_point(
    margins: np.ndarray, weights: np.ndarray, batch: ModelBatch, index: int
) -> dict:
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
        "internal_strict_correct": float(np.mean(inversions == 0)),
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


def _models(artifacts: dict, scheduler: str) -> tuple[dict, dict]:
    spec = resolved_specification()
    models = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            identity = f"{seed}/{condition}"
            selected = (
                scheduler if condition == "adaptive_eta_resampled" else "relation"
            )
            models[identity] = load_model(
                artifacts["archives"][identity]["config"], spec, scheduler=selected
            )
    return models, {key: compiled(value) for key, value in models.items()}


def _output_directory(start: int):
    return RUN_ROOT / "liu-evaluation" / f"cohorts-{start:03d}"


def validate_output(record: dict, input_ref: dict, start: int) -> dict:
    if (
        record["input_lock"] != reference(INPUT_LOCK)
        or record["input"] != input_ref
        or record["cohort_indices"] != cohort_indices(start)
    ):
        raise RuntimeError("Liu output shard identity differs")
    verify_reference(record["arrays"])
    return record


def evaluate_shard(
    input_ref: dict, start: int, models: dict, runners: dict, execution: dict
) -> None:
    directory = _output_directory(start)
    if directory.exists():
        validate_complete(directory)
        validate_output(load_json(directory / "result.json"), input_ref, start)
        return
    identities = list(models)
    with ProspectiveRun.start(
        directory,
        workflow_id="experience_dependent_plasticity_v1",
        execution_id=f"liu-cohorts-{start:03d}",
        producer={"module": __name__, "input_lock": reference(INPUT_LOCK)},
        resolved_config={"runtime": execution},
    ):
        points, output_rows = [], []
        for index, base, uniforms in load_cohorts(input_ref, start):
            batch = encode_batch(base, "resampled", uniforms, CODEBOOK)[0]
            outputs = {}
            for identity in identities:
                with torch.no_grad():
                    margin, weights, _ = runners[identity](
                        *model_tensors(batch, "cuda")
                    )
                outputs[identity] = {
                    "margins": margin.cpu().numpy().astype(np.float64),
                    "w": weights.cpu().numpy().astype(np.float64),
                }
            points.append(
                {
                    "cohort": index,
                    "fits": {
                        identity: cohort_point(
                            values["margins"], values["w"], batch, index
                        )
                        for identity, values in outputs.items()
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
        arrays["fit_seeds"] = np.asarray([int(key.split("/")[0]) for key in identities])
        arrays["fit_conditions"] = np.asarray([key.split("/")[1] for key in identities])
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
    models, runners = _models(artifacts, lock["selected_scheduler"])
    execution = runtime()
    for position, input_ref in enumerate(lock["cohort_shards"]):
        start = position * COHORT_SHARD_SIZE
        evaluate_shard(input_ref, start, models, runners, execution)
        validate_complete(_output_directory(start))
        print(
            f"Completed paired cohorts {start}..{start + COHORT_SHARD_SIZE - 1}",
            flush=True,
        )
    return {
        "cohorts": COHORTS,
        "fits": list(models),
        "selected_scheduler": lock["selected_scheduler"],
    }


def _statistic(values: np.ndarray, counts: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    means = counts @ values / len(values)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "interval": {"lower": float(low), "upper": float(high)},
    }


def summarize_points(points: list[dict]) -> dict:
    if [row["cohort"] for row in points] != list(range(COHORTS)):
        raise RuntimeError("summary requires every prospective cohort")
    refs = reference_intervals(human_references(resolved_specification()))
    fits = {}
    for seed in SEEDS:
        rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + seed)
        counts = rng.multinomial(
            COHORTS, np.full(COHORTS, 1 / COHORTS), size=BOOTSTRAP_SAMPLES
        )
        for condition in CONDITIONS:
            identity = f"{seed}/{condition}"
            rows = [row["fits"][identity] for row in points]
            continuous = {}
            for endpoint in ALL_ENDPOINTS:
                values = np.asarray(
                    [row["values"][endpoint] for row in rows], dtype=np.float64
                )
                statistic = _statistic(values, counts)
                continuous[endpoint] = {
                    **statistic,
                    "reference": refs[endpoint],
                    "classification": interval_classification(
                        statistic["interval"], refs[endpoint]
                    ),
                }
            qualitative = [
                all(value["qualitative"] for value in row["flags"].values())
                for row in rows
            ]
            quantitative = [
                all(value["calibration"] for value in row["flags"].values())
                for row in rows
            ]
            core = wilson(qualitative)["lower"] > 0.90 and all(
                continuous[name]["classification"] == "mean_within_reference"
                for name in ("learned_accuracy", "nonlearned_accuracy")
            )
            preservation = all(
                continuous[name]["classification"] == "mean_within_reference"
                for name in PRESERVED_ENDPOINTS
            )
            fits[identity] = {
                "continuous": continuous,
                "all_nine_qualitative": wilson(qualitative),
                "all_nine_quantitative": wilson(quantitative),
                "joint_nine": wilson(
                    np.asarray(qualitative) & np.asarray(quantitative)
                ),
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
                        np.asarray(
                            [row["inversion_bin_fractions"][index] for row in rows]
                        ),
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
                                    row["internal_to_sampled_transition"][
                                        internal_index
                                    ][class_index]
                                    for row in rows
                                ]
                            ),
                            counts,
                        )
                        for class_index in range(3)
                    ]
                    for internal_index in range(2)
                ],
                "ranking_composition_total_variation": _statistic(
                    np.asarray(
                        [row["ranking_composition_total_variation"] for row in rows]
                    ),
                    counts,
                ),
            }
        fixed_rows = [row["fits"][f"{seed}/fixed_eta_resampled"] for row in points]
        adaptive_rows = [
            row["fits"][f"{seed}/adaptive_eta_resampled"] for row in points
        ]
        differences = {
            "internal_strict_correct": np.asarray(
                [
                    a["internal_strict_correct"] - f["internal_strict_correct"]
                    for a, f in zip(adaptive_rows, fixed_rows, strict=True)
                ]
            ),
            "mean_inversion_count": np.asarray(
                [
                    a["mean_inversion_count"] - f["mean_inversion_count"]
                    for a, f in zip(adaptive_rows, fixed_rows, strict=True)
                ]
            ),
            "ranking_composition_total_variation": np.asarray(
                [
                    a["ranking_composition_total_variation"]
                    - f["ranking_composition_total_variation"]
                    for a, f in zip(adaptive_rows, fixed_rows, strict=True)
                ]
            ),
        }
        paired = {
            name: _statistic(values, counts) for name, values in differences.items()
        }
        paired["gates"] = {
            "internal_strict_correct": paired["internal_strict_correct"]["interval"][
                "lower"
            ]
            > 0,
            "mean_inversion_count": paired["mean_inversion_count"]["interval"]["upper"]
            < 0,
            "ranking_composition_total_variation": paired[
                "ranking_composition_total_variation"
            ]["interval"]["upper"]
            < 0,
        }
        fits[str(seed)] = paired
    return fits
