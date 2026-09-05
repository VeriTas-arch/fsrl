"""Claim-relative audit of frozen Resampled cohorts; no rollout or training."""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np

from fsrl.evaluation.metrics import count_circular_triads
from fsrl.experiments.cohort_diagnostic.statistics import wilson
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol, protocol_path

STUDY = REPO_ROOT / "studies/main_model_evaluation_v2"
RECORDS = STUDY / "records"
CONTRACT = RECORDS / "benchmarks/main_model_evaluation_v2.json"
CONTRACT_HASH = "bbdedc41384987f3c38588c29f14b6ee88203991543edb2980e8073f3f9ce41b"
CONTRACT_COMMIT = "076ef760c59b1a06cfcb1438afecd56bee197f0d"
QUALIFICATION = RECORDS / "benchmarks/main_model_evaluation_v2.qualification.json"
LOCK = RECORDS / "benchmarks/main_model_evaluation_v2.execution_lock.json"
RESULT = RECORDS / "results/main_model_evaluation_v2.json"
ARRAYS = RECORDS / "results/main_model_evaluation_v2.npz"
REPORT = RECORDS / "reports/main_model_evaluation_v2.md"

COHORT_RECORDS = REPO_ROOT / "studies/resampled_cohort_diagnostic/records"
COHORT_CONTRACT = COHORT_RECORDS / "benchmarks/resampled_cohort_diagnostic_v1.json"
COHORT_RESULT = COHORT_RECORDS / "results/resampled_cohort_diagnostic_v1.json"
PILOT_RESULT = (
    REPO_ROOT
    / "studies/quantized_relational_learner/records/results/quantized_relational_learner_v1.json"
)
LEGACY_BEHAVIOR = (
    REPO_ROOT
    / "studies/behavior_reproduction_map/records/benchmarks/model_behavior_reproduction_map_v1.json"
)
LEGACY_ADMISSION = (
    REPO_ROOT
    / "studies/main_model_admission/records/benchmarks/main_model_admission_v1.json"
)

FIT_SEEDS = np.asarray([2114, 2115, 2116], dtype=np.int64)
COHORT_COUNT = 400
SUBJECTS = 77
SHARD_SIZE = 20
TEMPERATURE = 0.25
HUMAN_COMPOSITION = np.asarray([8, 64, 5], dtype=np.float64) / 77
CLASS_NAMES = ("correct", "self_consistent_incorrect", "self_inconsistent")
INTERNAL_NAMES = ("strict_correct", "incorrect")
INVERSION_NAMES = ("0", "1", "2-3", "4+")

SOURCE_PATHS = (
    REPO_ROOT / "tools/provenance/main_model_evaluation_v2.py",
    REPO_ROOT / "tests/infra/test_main_model_evaluation_v2.py",
    REPO_ROOT / "fsrl/analysis/behavioral.py",
    REPO_ROOT / "fsrl/analysis/policy.py",
    REPO_ROOT / "fsrl/evaluation/metrics.py",
    REPO_ROOT / "fsrl/experiments/cohort_diagnostic/statistics.py",
    REPO_ROOT / "fsrl/experiments/training_strategy/evaluation.py",
    REPO_ROOT / "fsrl/experiments/training_strategy/locks.py",
    REPO_ROOT / "fsrl/infra/file_contracts.py",
    REPO_ROOT / "fsrl/infra/git_provenance.py",
    REPO_ROOT / "fsrl/infra/provenance.py",
    REPO_ROOT / "fsrl/infra/record_catalog.py",
    REPO_ROOT / "fsrl/paths.py",
    REPO_ROOT / "fsrl/tasks/protocol.py",
    REPO_ROOT / "fsrl/tasks/protocol_catalog.py",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / ".envrc",
)


def specification() -> dict:
    """Load the immutable evaluation contract through its Git witness."""
    if file_sha256(CONTRACT) != CONTRACT_HASH:
        raise RuntimeError("main-model evaluation contract changed")
    if git_blob_sha256(REPO_ROOT, CONTRACT_COMMIT, reference(CONTRACT)["path"]) != (
        CONTRACT_HASH
    ):
        raise RuntimeError("main-model evaluation contract Git witness differs")
    return load_json(CONTRACT)


def implementation_sources() -> list[dict]:
    return [reference(path) for path in sorted(SOURCE_PATHS)]


def _load_arrays(record: dict) -> dict[str, np.ndarray]:
    path = verify_reference(record)
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def cohort_shards() -> list[tuple[dict, dict]]:
    parent = load_json(COHORT_RESULT)
    if parent["experiment_id"] != "resampled_cohort_diagnostic_v1":
        raise RuntimeError("unexpected parent cohort result")
    pairs = []
    for shard_ref in parent["shards"]:
        shard = load_json(verify_reference(shard_ref))
        pairs.append((shard_ref, shard))
    if len(pairs) != COHORT_COUNT // SHARD_SIZE:
        raise RuntimeError("parent result omits a registered shard")
    return pairs


def scientific_inputs() -> list[dict]:
    paths = {
        CONTRACT,
        COHORT_CONTRACT,
        COHORT_RESULT,
        PILOT_RESULT,
        LEGACY_BEHAVIOR,
        LEGACY_ADMISSION,
        protocol_path("liu_v2"),
    }
    records = {reference(path)["path"]: reference(path) for path in paths}
    for shard_ref, shard in cohort_shards():
        for row in (shard_ref, shard["input"], shard["arrays"]):
            records[row["path"]] = row
    return [records[key] for key in sorted(records)]


def _validate_shard_inventory(shard_ref: dict, shard: dict, start: int) -> None:
    expected = np.arange(start, start + SHARD_SIZE, dtype=np.int64)
    input_arrays = _load_arrays(shard["input"])
    output_arrays = _load_arrays(shard["arrays"])
    expected_input = {
        "input__support_cues",
        "input__signed",
        "input__retention",
        "input__probabilities",
        "input__local_evidence",
        "input__query_cues",
        "input__codes",
        "input__support_pairs",
        "input__query_pairs",
        "encoding_uniforms",
        "cohort_indices",
    }
    if set(input_arrays) != expected_input:
        raise RuntimeError(f"input array inventory differs: {shard_ref['path']}")
    if set(output_arrays) != {"w", "margins", "cohort_indices", "fit_seeds"}:
        raise RuntimeError(f"output array inventory differs: {shard_ref['path']}")
    np.testing.assert_array_equal(input_arrays["cohort_indices"], expected)
    np.testing.assert_array_equal(output_arrays["cohort_indices"], expected)
    np.testing.assert_array_equal(output_arrays["fit_seeds"], FIT_SEEDS)
    if input_arrays["input__codes"].shape != (SHARD_SIZE, SUBJECTS, 8, 15):
        raise RuntimeError("registered code array shape differs")
    if input_arrays["input__query_pairs"].shape != (SHARD_SIZE, SUBJECTS, 56, 2):
        raise RuntimeError("registered query-pair shape differs")
    if output_arrays["w"].shape != (SHARD_SIZE, len(FIT_SEEDS), SUBJECTS, 15):
        raise RuntimeError("registered terminal-score shape differs")
    if output_arrays["margins"].shape != (
        SHARD_SIZE,
        len(FIT_SEEDS),
        SUBJECTS,
        56,
    ):
        raise RuntimeError("registered margin shape differs")
    points = [row["cohort"] for row in shard["points"]]
    if points != expected.tolist():
        raise RuntimeError("parent point inventory differs")


def qualify() -> dict:
    """Check identities and schemas without computing the new estimands."""
    specification()
    shards = cohort_shards()
    for index, (shard_ref, shard) in enumerate(shards):
        _validate_shard_inventory(shard_ref, shard, index * SHARD_SIZE)
    return {
        "schema_version": 1,
        "experiment_id": "main_model_evaluation_v2",
        "passed": True,
        "new_training_performed": False,
        "new_simulation_performed": False,
        "new_estimands_computed": False,
        "fits": FIT_SEEDS.tolist(),
        "cohorts_per_fit": COHORT_COUNT,
        "input_shards": len(shards),
        "output_shards": len(shards),
        "registered_subjects_per_cohort": SUBJECTS,
        "sources": implementation_sources(),
        "inputs": scientific_inputs(),
    }


def write_lock() -> dict:
    source_commit = require_pushed_clean()
    qualification = {**qualify(), "source_commit": source_commit}
    for row in qualification["sources"] + qualification["inputs"]:
        verify_reference(row, commit=source_commit)
    write_json_exclusive(QUALIFICATION, qualification)
    lock = {
        "schema_version": 1,
        "experiment_id": "main_model_evaluation_v2",
        "source_commit": source_commit,
        "contract": reference(CONTRACT),
        "sources": qualification["sources"],
        "inputs": qualification["inputs"],
        "qualification": reference(QUALIFICATION),
        "fits": FIT_SEEDS.tolist(),
        "cohorts_per_fit": COHORT_COUNT,
        "new_training_performed": False,
        "new_simulation_performed": False,
        "retrospective_metrics_computed": False,
    }
    write_json_exclusive(LOCK, lock)
    return lock


def validate_lock() -> dict:
    require_pushed_clean()
    lock = load_json(LOCK)
    if (
        lock["contract"] != reference(CONTRACT)
        or lock["sources"] != implementation_sources()
        or lock["inputs"] != scientific_inputs()
        or lock["fits"] != FIT_SEEDS.tolist()
        or lock["cohorts_per_fit"] != COHORT_COUNT
        or lock["new_training_performed"]
        or lock["new_simulation_performed"]
        or lock["retrospective_metrics_computed"]
    ):
        raise RuntimeError("main-model evaluation lock differs")
    for row in lock["sources"] + lock["inputs"]:
        verify_reference(row, commit=lock["source_commit"])
    qualification = load_json(verify_reference(lock["qualification"]))
    if qualification != {**qualify(), "source_commit": lock["source_commit"]}:
        raise RuntimeError("main-model evaluation qualification differs")
    return lock


def true_positions(protocol: RankingProtocol) -> np.ndarray:
    positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        positions[item] = position
    return positions


def internal_inversion_counts(
    weights: np.ndarray, codes: np.ndarray, protocol: RankingProtocol
) -> np.ndarray:
    """Count nonpositive correct-signed score differences for every fit/subject."""
    scores = np.einsum("fsd,sid->fsi", weights, codes, dtype=np.float64)
    positions = true_positions(protocol)
    inversions = np.zeros(scores.shape[:2], dtype=np.int16)
    for first, second in combinations(range(protocol.n_items), 2):
        higher, lower = (
            (first, second) if positions[first] < positions[second] else (second, first)
        )
        inversions += scores[..., higher] <= scores[..., lower]
    return inversions


def inversion_bins(inversions: np.ndarray) -> np.ndarray:
    result = np.empty_like(inversions, dtype=np.int8)
    result[inversions == 0] = 0
    result[inversions == 1] = 1
    result[(inversions >= 2) & (inversions <= 3)] = 2
    result[inversions >= 4] = 3
    return result


def _ranking_class(
    first_counts: np.ndarray,
    totals: np.ndarray,
    canonical_margins: np.ndarray,
    protocol: RankingProtocol,
) -> int:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    winners = {}
    for index, pair in enumerate(pairs):
        second_counts = totals[index] - first_counts[index]
        if first_counts[index] > second_counts:
            winners[pair] = pair[0]
        elif second_counts > first_counts[index]:
            winners[pair] = pair[1]
        else:
            winners[pair] = pair[0] if canonical_margins[index] > 0 else pair[1]
    positions = true_positions(protocol)
    majority_correct = all(
        positions[winner] < positions[pair[1] if winner == pair[0] else pair[0]]
        for pair, winner in winners.items()
    )
    if majority_correct:
        return 0
    return 1 if count_circular_triads(winners, protocol.n_items) == 0 else 2


def sampled_classes(
    margins: np.ndarray,
    protocol: RankingProtocol,
    *,
    choice_seed: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct only the frozen subject ranking class and eligibility."""
    if margins.ndim != 3 or margins.shape[1:] != (SUBJECTS, 56):
        raise ValueError("sampled classification margin shape differs")
    oriented = ordered_pairs(protocol.n_items)
    oriented_index = {pair: index for index, pair in enumerate(oriented)}
    canonical = tuple(combinations(range(protocol.n_items), 2))
    canonical_index = {pair: index for index, pair in enumerate(canonical)}
    canonical_margin_index = np.asarray(
        [
            [oriented_index[pair], oriented_index[(pair[1], pair[0])]]
            for pair in canonical
        ]
    )
    classes = np.empty(margins.shape[:2], dtype=np.int8)
    eligible = np.empty(margins.shape[:2], dtype=bool)
    for subject in range(SUBJECTS):
        schedule_rng = np.random.default_rng(choice_seed + 2 * subject)
        choice_rng = np.random.default_rng(choice_seed + 2 * subject + 1)
        schedule = protocol.query_schedule(schedule_rng)
        left = np.asarray([trial.left_item for trial in schedule], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in schedule], dtype=np.int64)
        correct_action = np.asarray(
            [trial.correct_action for trial in schedule], dtype=bool
        )
        query_index = np.asarray(
            [oriented_index[(trial.left_item, trial.right_item)] for trial in schedule]
        )
        pair_index = np.asarray(
            [
                canonical_index[tuple(sorted((a, b)))]
                for a, b in zip(left, right, strict=True)
            ]
        )
        random = choice_rng.random(len(schedule))
        logits = margins[:, subject, query_index] / temperature
        probability_left = 1.0 / (1.0 + np.exp(-logits))
        choose_left = random[None, :] < probability_left
        first = np.minimum(left, right)
        chosen = np.where(choose_left, left[None, :], right[None, :])
        totals = np.bincount(pair_index, minlength=len(canonical))
        for fit in range(len(margins)):
            correct_counts = np.bincount(
                pair_index,
                weights=choose_left[fit] == correct_action,
                minlength=len(canonical),
            )
            eligible[fit, subject] = float(np.mean(correct_counts / totals)) >= 0.5
            first_counts = np.bincount(
                pair_index,
                weights=chosen[fit] == first,
                minlength=len(canonical),
            )
            canonical_margins = 0.5 * (
                margins[fit, subject, canonical_margin_index[:, 0]]
                - margins[fit, subject, canonical_margin_index[:, 1]]
            )
            classes[fit, subject] = _ranking_class(
                first_counts, totals, canonical_margins, protocol
            )
    return classes, eligible


def ranking_total_variation(composition: np.ndarray) -> np.ndarray:
    values = np.asarray(composition, dtype=np.float64)
    if values.shape[-1] != len(HUMAN_COMPOSITION):
        raise ValueError("ranking composition must contain three classes")
    return 0.5 * np.abs(values - HUMAN_COMPOSITION).sum(axis=-1)


def _assert_parent_point(
    point: dict, classes: np.ndarray, eligible: np.ndarray
) -> None:
    count = int(eligible.sum())
    if count != point["eligible_subjects"] or count == 0:
        raise RuntimeError("sampled eligibility does not reconstruct")
    counts = np.bincount(classes[eligible], minlength=len(CLASS_NAMES))
    expected = np.asarray(
        [
            point["values"]["correct_ranker"],
            point["values"]["self_consistent_incorrect"],
            point["values"]["self_inconsistent"],
        ]
    )
    np.testing.assert_allclose(counts / count, expected, atol=1e-12, rtol=0)


def reconstruct() -> dict[str, np.ndarray]:
    protocol = load_registered_protocol("liu_v2")
    expected_pairs = np.asarray(ordered_pairs(protocol.n_items), dtype=np.int64)
    transitions = np.zeros((len(FIT_SEEDS), COHORT_COUNT, 2, 3), dtype=np.int16)
    inversion_counts = np.zeros((len(FIT_SEEDS), COHORT_COUNT, 4), dtype=np.int16)
    eligible_composition = np.zeros((len(FIT_SEEDS), COHORT_COUNT, 3), dtype=np.int16)
    eligible_subjects = np.zeros((len(FIT_SEEDS), COHORT_COUNT), dtype=np.int16)
    qualitative = np.zeros((len(FIT_SEEDS), COHORT_COUNT), dtype=bool)
    for shard_index, (shard_ref, shard) in enumerate(cohort_shards()):
        start = shard_index * SHARD_SIZE
        _validate_shard_inventory(shard_ref, shard, start)
        inputs, outputs = _load_arrays(shard["input"]), _load_arrays(shard["arrays"])
        for position, cohort in enumerate(outputs["cohort_indices"]):
            pairs = inputs["input__query_pairs"][position]
            np.testing.assert_array_equal(
                pairs, np.broadcast_to(expected_pairs, pairs.shape)
            )
            inversions = internal_inversion_counts(
                outputs["w"][position], inputs["input__codes"][position], protocol
            )
            margins = outputs["margins"][position]
            signs = np.asarray(
                [
                    1
                    if true_positions(protocol)[left] < true_positions(protocol)[right]
                    else -1
                    for left, right in expected_pairs
                ]
            )
            np.testing.assert_array_equal(
                (signs[None, None, :] * margins > 0).all(axis=-1), inversions == 0
            )
            classes, eligible = sampled_classes(
                margins,
                protocol,
                choice_seed=2_000_000 + 1000 * int(cohort) + 800,
                temperature=TEMPERATURE,
            )
            point = shard["points"][position]
            if point["cohort"] != int(cohort):
                raise RuntimeError("cohort point identity differs")
            bins = inversion_bins(inversions)
            for fit, seed in enumerate(FIT_SEEDS):
                fit_point = point["fits"][str(seed)]
                _assert_parent_point(fit_point, classes[fit], eligible[fit])
                for subject in range(SUBJECTS):
                    internal = 0 if inversions[fit, subject] == 0 else 1
                    transitions[fit, cohort, internal, classes[fit, subject]] += 1
                    inversion_counts[fit, cohort, bins[fit, subject]] += 1
                eligible_composition[fit, cohort] = np.bincount(
                    classes[fit, eligible[fit]], minlength=len(CLASS_NAMES)
                )
                eligible_subjects[fit, cohort] = eligible[fit].sum()
                qualitative[fit, cohort] = all(
                    row["qualitative"] for row in fit_point["flags"].values()
                )
    if not np.all(transitions.sum(axis=(2, 3)) == SUBJECTS):
        raise RuntimeError("transition matrix does not contain every subject")
    if not np.all(inversion_counts.sum(axis=2) == SUBJECTS):
        raise RuntimeError("inversion bins do not contain every subject")
    return {
        "fit_seeds": FIT_SEEDS,
        "cohort_indices": np.arange(COHORT_COUNT, dtype=np.int64),
        "transition_counts": transitions,
        "inversion_bin_counts": inversion_counts,
        "eligible_composition_counts": eligible_composition,
        "eligible_subject_counts": eligible_subjects,
        "legacy_joint_qualitative": qualitative,
    }


def _summaries(values: np.ndarray, counts: np.ndarray) -> list[dict]:
    values = np.asarray(values, dtype=np.float64)
    means = counts @ values.reshape(len(values), -1) / len(values)
    point = values.mean(axis=0).reshape(-1)
    lower = np.quantile(means, 0.025, axis=0)
    upper = np.quantile(means, 0.975, axis=0)
    return [
        {"mean": float(mean), "interval": {"lower": float(low), "upper": float(high)}}
        for mean, low, high in zip(point, lower, upper, strict=True)
    ]


def _named(values: np.ndarray, names: tuple[str, ...], counts: np.ndarray) -> dict:
    rows = _summaries(values, counts)
    if len(rows) != len(names):
        raise RuntimeError("summary label count differs")
    return dict(zip(names, rows, strict=True))


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    parent, pilot = load_json(COHORT_RESULT), load_json(PILOT_RESULT)
    fits = {}
    for fit, seed_value in enumerate(FIT_SEEDS):
        seed = int(seed_value)
        transitions = arrays["transition_counts"][fit] / SUBJECTS
        inversions = arrays["inversion_bin_counts"][fit] / SUBJECTS
        eligible = arrays["eligible_subject_counts"][fit]
        composition = arrays["eligible_composition_counts"][fit] / eligible[:, None]
        rank_tv = ranking_total_variation(composition)
        internal_correct = transitions[:, 0].sum(axis=1)
        sampled_correct = transitions[:, :, 0].sum(axis=1)
        loss = transitions[:, 0, 1:].sum(axis=1)
        rescue = transitions[:, 1, 0]
        net = sampled_correct - internal_correct
        rng = np.random.default_rng(3_200_000 + seed)
        bootstrap = rng.multinomial(
            COHORT_COUNT,
            np.full(COHORT_COUNT, 1 / COHORT_COUNT),
            size=10_000,
        )
        transition_rows = _summaries(transitions, bootstrap)
        transition = {
            internal_name: {
                class_name: transition_rows[internal_index * 3 + class_index]
                for class_index, class_name in enumerate(CLASS_NAMES)
            }
            for internal_index, internal_name in enumerate(INTERNAL_NAMES)
        }
        scalar = _named(
            np.column_stack(
                (internal_correct, sampled_correct, loss, rescue, net, rank_tv)
            ),
            (
                "internal_strict_correct",
                "sampled_correct_all_subjects",
                "loss_flow",
                "rescue_flow",
                "net_sampling_shift",
                "ranking_composition_total_variation",
            ),
            bootstrap,
        )
        net_interval = scalar["net_sampling_shift"]["interval"]
        sampling = (
            "sampling_loss"
            if net_interval["upper"] < 0
            else "sampling_rescue"
            if net_interval["lower"] > 0
            else "direction_unresolved"
        )
        parent_fit = parent["fits"][str(seed)]
        reconstructed_qualitative = wilson(arrays["legacy_joint_qualitative"][fit])
        if (
            json_ready(reconstructed_qualitative)
            != parent_fit["all_nine"]["qualitative"]
        ):
            raise RuntimeError(
                "legacy joint qualitative stability does not reconstruct"
            )
        core = (
            parent_fit["all_nine"]["qualitative"]["lower"] > 0.90
            and parent_fit["continuous"]["learned_accuracy"]["classification"]
            == "mean_within_reference"
            and parent_fit["continuous"]["nonlearned_accuracy"]["classification"]
            == "mean_within_reference"
        )
        pilot_fit = pilot["fits"][f"{seed}/resampled"]
        pilot_flags = pilot_fit["behavior"]["flags"]
        fits[str(seed)] = {
            "retrospective_core_behavior_supported": bool(core),
            "core_stability_threshold_prospective_for_this_fit": False,
            "legacy_full_quantitative_fidelity": False,
            "legacy_pilot_rows": {
                "qualitative": sum(row["qualitative"] for row in pilot_flags.values()),
                "quantitative": sum(row["calibration"] for row in pilot_flags.values()),
                "total": len(pilot_flags),
            },
            "legacy_joint_qualitative": parent_fit["all_nine"]["qualitative"],
            "legacy_joint_quantitative": parent_fit["all_nine"]["calibration"],
            "difficulty_guardrails": {
                name: parent_fit["continuous"][name]
                for name in ("learned_accuracy", "nonlearned_accuracy")
            },
            "ranking_composition": _named(composition, CLASS_NAMES, bootstrap),
            "ranking_composition_total_variation": scalar.pop(
                "ranking_composition_total_variation"
            ),
            "internal_to_sampled": {
                "classification": sampling,
                "transition": transition,
                "inversion_bins": _named(inversions, INVERSION_NAMES, bootstrap),
                **scalar,
            },
        }
    common_sampling = {
        row["internal_to_sampled"]["classification"] for row in fits.values()
    }
    return {
        "outcome": "retrospective_core_behavior_supported"
        if all(row["retrospective_core_behavior_supported"] for row in fits.values())
        else "retrospective_core_behavior_not_supported",
        "fits": fits,
        "sampling_localization": next(iter(common_sampling))
        if len(common_sampling) == 1
        else "fit_heterogeneity",
        "labels": {
            "retrospective_core_behavior_supported": all(
                row["retrospective_core_behavior_supported"] for row in fits.values()
            ),
            "core_working_model_eligible": False,
            "full_quantitative_fidelity": False,
        },
        "next_model": {
            "primary_candidate": "relation_specific_experience_dependent_plasticity",
            "reason": "Test whether reducing late-code domination improves deterministic internal organization under the frozen paired-development design.",
            "sampling_boundary": "The internal-to-sampled result is a separate policy-expression contribution and must not be attributed to the update rule.",
            "training_authorized": False,
        },
    }


def render_report(result: dict) -> str:
    lines = [
        "# Claim-relative main-model evaluation",
        "",
        f"Registered outcome: `{result['outcome']}`. This is a retrospective audit of exposed frozen simulations, not confirmation under the new stability threshold.",
        "",
        "The unchanged legacy 9/9 metric remains `full_quantitative_fidelity`. The added `core_mechanism_adequacy` label separates central qualitative phenomena and task-difficulty guardrails from strict calibration of every summary statistic.",
        "",
        "| Fit | Core (retrospective) | Legacy pilot | Joint qualitative stability | Mean learned | Mean nonlearned |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for seed, row in result["fits"].items():
        learned = row["difficulty_guardrails"]["learned_accuracy"]
        nonlearned = row["difficulty_guardrails"]["nonlearned_accuracy"]
        legacy = row["legacy_pilot_rows"]
        lines.append(
            f"| {seed} | {row['retrospective_core_behavior_supported']} | {legacy['qualitative']}/{legacy['total']} qualitative; {legacy['quantitative']}/{legacy['total']} quantitative | {row['legacy_joint_qualitative']} | {learned['mean']} {learned['interval']} | {nonlearned['mean']} {nonlearned['interval']} |"
        )
    lines.extend(
        [
            "",
            "## Internal-to-sampled localization",
            "",
            f"Cross-fit classification: `{result['sampling_localization']}`.",
            "",
            "| Fit | Internal strict correct | Sampled correct (all 77) | Loss flow | Rescue flow | Net sampling shift | Rank-composition TV |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for seed, row in result["fits"].items():
        internal = row["internal_to_sampled"]
        lines.append(
            f"| {seed} | {internal['internal_strict_correct']} | {internal['sampled_correct_all_subjects']} | {internal['loss_flow']} | {internal['rescue_flow']} | {internal['net_sampling_shift']} | {row['ranking_composition_total_variation']} |"
        )
    lines.extend(
        [
            "",
            "Transition cells and inversion bins are stored in the frozen JSON/NPZ result. Internal strict correctness is an unobserved model state and is not compared with a human latent-state target. Ranking composition uses the unchanged eligible-subject denominator; transition flows use all 77 simulated subjects.",
            "",
            "## Consequence for the next single-stage model",
            "",
            "The next primary candidate remains the prospectively specified relation-specific experience-dependent plasticity rule. Its purpose is narrow: reduce late-code domination and improve deterministic internal ordering while keeping the Resampled codebook, stable admission, 15-dimensional state, query form and single-stage objective fixed. Because the rule changes both mean shrinkage and variance propagation, those effects must be reported separately.",
            "",
            "A paired fixed-eta baseline, a nonbalanced schedule control, one to three fresh development seeds and internal-order improvement are required before any replication. The observed sampling contribution remains a separate policy-expression boundary; the plasticity candidate is not allowed to claim it away. No training is authorized by this audit.",
            "",
            "The current exposed recipe is not promoted: it lacks prospective validation under the new threshold, unchanged fresh training replication and selected-parameter biological-boundary verification, and it does not satisfy the preserved full 9/9 quantitative label.",
            "",
        ]
    )
    return "\n".join(lines)


def publish() -> dict:
    lock = validate_lock()
    arrays = reconstruct()
    result = {
        "schema_version": 1,
        "experiment_id": "main_model_evaluation_v2",
        "contract_sha256": CONTRACT_HASH,
        "execution_lock": reference(LOCK),
        "source_commit": lock["source_commit"],
        "arrays": None,
        "training_performed": False,
        "new_simulation_performed": False,
        **summarize(arrays),
        "claim_boundary": specification()["execution"]["claim_boundary"],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=False)
    write_arrays(ARRAYS, arrays)
    result["arrays"] = reference(ARRAYS)
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=False)
    REPORT.write_text(render_report(result), encoding="utf-8")
    return {
        "outcome": result["outcome"],
        "sampling_localization": result["sampling_localization"],
        "result": reference(RESULT),
        "arrays": reference(ARRAYS),
        "report": reference(REPORT),
    }


def verify_record() -> dict:
    lock = validate_lock()
    result = load_json(RESULT)
    if (
        result["execution_lock"] != reference(LOCK)
        or result["contract_sha256"] != CONTRACT_HASH
        or result["source_commit"] != lock["source_commit"]
        or result["training_performed"]
        or result["new_simulation_performed"]
    ):
        raise RuntimeError("saved main-model evaluation provenance differs")
    rebuilt = reconstruct()
    saved = _load_arrays(result["arrays"])
    if set(saved) != set(rebuilt):
        raise RuntimeError("saved audit array inventory differs")
    for name, value in rebuilt.items():
        np.testing.assert_array_equal(saved[name], value)
    summary = summarize(rebuilt)
    for name, value in summary.items():
        if json_ready(value) != result[name]:
            raise RuntimeError(f"saved audit summary differs: {name}")
    if REPORT.read_text(encoding="utf-8") != render_report(result):
        raise RuntimeError("saved main-model evaluation report differs")
    return {
        "passed": True,
        "outcome": result["outcome"],
        "sampling_localization": result["sampling_localization"],
        "fits": len(FIT_SEEDS),
        "cohorts_per_fit": COHORT_COUNT,
        "training_performed": False,
        "new_simulation_performed": False,
    }


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "lock", "publish", "verify-record"))
    parsed = parser.parse_args(args)
    result = {
        "audit": qualify,
        "lock": write_lock,
        "publish": publish,
        "verify-record": verify_record,
    }[parsed.stage]()
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
