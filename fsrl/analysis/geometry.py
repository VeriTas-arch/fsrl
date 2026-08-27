"""Neural-geometry test against subjective and true ranking RDMs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats

from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol

DEFAULT_GEOMETRY_GATE_PATH = resolve_record("benchmarks/geometry_gate_v2.json")


def context_averaged_item_representations(
    hidden_states: dict[tuple[int, int], np.ndarray], n_items: int
) -> np.ndarray:
    """Average response hidden states across both roles and all query partners."""

    representations = []
    for item in range(n_items):
        states = [state for pair, state in hidden_states.items() if item in pair]
        if len(states) != 2 * (n_items - 1):
            raise ValueError(
                "hidden states must contain both orientations of all pairs"
            )
        representations.append(np.mean(np.stack(states), axis=0))
    return np.stack(representations)


def antisymmetric_hodge_item_representations(
    hidden_states: dict[tuple[int, int], np.ndarray], n_items: int
) -> np.ndarray:
    """Reconstruct item vectors from signed hidden-state orientation contrasts."""

    rows = []
    contrasts = []
    for first, second in combinations(range(n_items), 2):
        if (first, second) not in hidden_states or (second, first) not in hidden_states:
            raise ValueError(
                "hidden states must contain both orientations of all pairs"
            )
        row = np.zeros(n_items, dtype=np.float64)
        row[first] = 1.0
        row[second] = -1.0
        rows.append(row)
        contrasts.append(
            0.5 * (hidden_states[(first, second)] - hidden_states[(second, first)])
        )
    rows.append(np.ones(n_items, dtype=np.float64))
    contrasts.append(np.zeros_like(contrasts[0]))
    representations, *_ = np.linalg.lstsq(
        np.vstack(rows), np.stack(contrasts), rcond=None
    )
    return representations


def representation_rdm(representations: np.ndarray) -> np.ndarray:
    differences = representations[:, None, :] - representations[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def rank_positions(order_high_to_low: list[int] | tuple[int, ...]) -> np.ndarray:
    positions = np.empty(len(order_high_to_low), dtype=np.int64)
    for position, item in enumerate(order_high_to_low):
        positions[item] = position
    return positions


def ranking_rdm(positions: np.ndarray) -> np.ndarray:
    return np.abs(positions[:, None] - positions[None, :]).astype(np.float64)


def rdm_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[0] != first.shape[1]
    ):
        raise ValueError("RDMs must be equally sized square matrices")
    upper = np.triu_indices(first.shape[0], k=1)
    result = cast(Any, stats.spearmanr(first[upper], second[upper]))
    return float(result.statistic)


def _ranked_spearman(
    first: np.ndarray,
    first_centered_ranks: np.ndarray,
    second: np.ndarray,
    second_centered_ranks: np.ndarray,
) -> float:
    """Reuse precomputed ranks while preserving SciPy's exceptional semantics."""

    if (
        first.size == 0
        or second.size == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or np.all(first == first[0])
        or np.all(second == second[0])
    ):
        result = cast(Any, stats.spearmanr(first, second))
        return float(result.statistic)
    centered = np.stack((first_centered_ranks, second_centered_ranks))
    covariance = np.dot(centered, centered.T)
    covariance *= np.true_divide(1, centered.shape[1] - 1)
    scale = np.sqrt(np.real(np.diag(covariance)))
    covariance /= scale[:, None]
    covariance /= scale[None, :]
    np.clip(covariance.real, -1, 1, out=covariance.real)
    return float(covariance[1, 0])


def _centered_rdm_ranks(rdm: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(rdm)
    return ranks - np.mean(ranks)


def analyze_item_geometry(
    protocol: RankingProtocol,
    representations: tuple[np.ndarray, ...],
    behavior: dict,
    *,
    representation_description: str = "context-averaged response hidden state",
) -> dict:
    subjects = behavior["subjects"]
    if len(subjects) != len(representations):
        raise ValueError("behavior and neural representations have different cohorts")
    upper = np.triu_indices(protocol.n_items, k=1)
    true_rdm = ranking_rdm(rank_positions(protocol.true_order_high_to_low))[upper]
    true_ranks = _centered_rdm_ranks(true_rdm)
    subjective_rdms = tuple(
        ranking_rdm(rank_positions(subject["subjective_order_high_to_low"]))[upper]
        for subject in subjects
    )
    subjective_ranks = tuple(_centered_rdm_ranks(rdm) for rdm in subjective_rdms)
    primary_indices = [
        index
        for index, subject in enumerate(subjects)
        if subject["ranking_class"] == "self_consistent_incorrect"
        and subject["overall_accuracy"] >= 0.5
    ]
    rows = []
    for subject_index in primary_indices:
        neural_rdm = representation_rdm(representations[subject_index])[upper]
        neural_ranks = _centered_rdm_ranks(neural_rdm)
        subjective_correlation = _ranked_spearman(
            neural_rdm,
            neural_ranks,
            subjective_rdms[subject_index],
            subjective_ranks[subject_index],
        )
        true_correlation = _ranked_spearman(
            neural_rdm,
            neural_ranks,
            true_rdm,
            true_ranks,
        )
        other_correlations = [
            _ranked_spearman(
                neural_rdm,
                neural_ranks,
                subjective_rdms[other_index],
                subjective_ranks[other_index],
            )
            for other_index in primary_indices
            if other_index != subject_index
        ]
        other_mean = (
            None if not other_correlations else float(np.mean(other_correlations))
        )
        rows.append(
            {
                "subject": subject_index,
                "rho_neural_subjective": subjective_correlation,
                "rho_neural_true": true_correlation,
                "rho_neural_other_subjective_mean": other_mean,
                "subjective_minus_true": subjective_correlation - true_correlation,
                "subjective_minus_other": (
                    None if other_mean is None else subjective_correlation - other_mean
                ),
            }
        )

    subjective_true_deltas = np.asarray(
        [row["subjective_minus_true"] for row in rows], dtype=np.float64
    )
    subjective_other_deltas = np.asarray(
        [
            row["subjective_minus_other"]
            for row in rows
            if row["subjective_minus_other"] is not None
        ],
        dtype=np.float64,
    )
    positive = int(np.sum(subjective_true_deltas > 0.0))
    sign_test_p = (
        None
        if len(subjective_true_deltas) == 0
        else float(
            stats.binomtest(
                positive, len(subjective_true_deltas), p=0.5, alternative="greater"
            ).pvalue
        )
    )
    return {
        "estimand": {
            "neural_representation": (representation_description),
            "neural_rdm": "euclidean_distance",
            "ranking_rdm": "absolute_rank_distance",
            "rdm_similarity": "spearman_upper_triangle",
            "primary_subject_class": "self_consistent_incorrect",
        },
        "group": {
            "subjects": len(rows),
            "mean_subjective_minus_true": (
                None
                if not len(subjective_true_deltas)
                else float(np.mean(subjective_true_deltas))
            ),
            "proportion_subjective_greater_than_true": (
                None
                if not len(subjective_true_deltas)
                else positive / len(subjective_true_deltas)
            ),
            "one_sided_sign_test_p": sign_test_p,
            "mean_subjective_minus_other": (
                None
                if not len(subjective_other_deltas)
                else float(np.mean(subjective_other_deltas))
            ),
        },
        "subjects": rows,
    }


def evaluate_geometry_gate(result: dict, specification: dict) -> dict:
    group = result["group"]
    checks = []

    def add(name: str, observed, passed: bool, criterion: str) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "passed": bool(passed),
                "criterion": criterion,
            }
        )

    minimum_subjects = specification["minimum_subjects"]
    add(
        "subjects",
        group["subjects"],
        group["subjects"] >= minimum_subjects,
        f">= {minimum_subjects}",
    )
    for result_name, specification_name in (
        (
            "mean_subjective_minus_true",
            "minimum_mean_subjective_minus_true_spearman",
        ),
        (
            "mean_subjective_minus_other",
            "minimum_mean_subjective_minus_other_spearman",
        ),
    ):
        observed = group[result_name]
        threshold = specification[specification_name]
        add(
            result_name,
            observed,
            observed is not None and observed > threshold,
            f"> {threshold}",
        )
    observed_p = group["one_sided_sign_test_p"]
    maximum_p = specification["maximum_one_sided_sign_test_p"]
    add(
        "one_sided_sign_test_p",
        observed_p,
        observed_p is not None and observed_p <= maximum_p,
        f"<= {maximum_p}",
    )
    return {
        "geometry_gate_id": specification["geometry_gate_id"],
        "registration_status": specification.get("registration_status"),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def run_geometry_analysis(
    checkpoint: Path,
    behavior_path: Path,
    gate_path: Path = DEFAULT_GEOMETRY_GATE_PATH,
) -> dict:
    behavior = load_json(behavior_path)
    batch_size = len(behavior["subjects"])
    net, config, checkpoint_info = load_retro_checkpoint(checkpoint, batch_size)
    if behavior.get("checkpoint", {}).get("sha256") != checkpoint_info.sha256:
        raise ValueError("behavior result and checkpoint SHA-256 do not match")
    protocol_path_value = behavior.get("protocol_path")
    protocol = (
        load_ranking_protocol()
        if protocol_path_value is None
        else load_ranking_protocol(protocol_path_value)
    )
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=int(behavior["cue_seed"]),
        support_seed=int(behavior["support_seed"]),
        cue_mode="permuted_shared",
        subject_encoding_mode=behavior["subject_encoding_mode"],
        subject_encoding_seed=int(behavior["subject_encoding_seed"]),
    )
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    ordered_pairs = tuple(
        oriented
        for first, second in combinations(range(protocol.n_items), 2)
        for oriented in ((first, second), (second, first))
    )
    hidden = evaluator.readout_hidden_states(
        fast_weights, tuple(ordered_pairs for _ in range(batch_size))
    )
    primary_representations = tuple(
        antisymmetric_hodge_item_representations(states, protocol.n_items)
        for states in hidden
    )
    context_representations = tuple(
        context_averaged_item_representations(states, protocol.n_items)
        for states in hidden
    )
    result = analyze_item_geometry(
        protocol,
        primary_representations,
        behavior,
        representation_description=(
            "Hodge item vectors reconstructed from half the response-hidden orientation contrast"
        ),
    )
    context_control = analyze_item_geometry(
        protocol,
        context_representations,
        behavior,
        representation_description="context-averaged response hidden state",
    )
    result["context_average_control"] = {
        "estimand": context_control["estimand"],
        "group": context_control["group"],
    }
    result["protocol_id"] = protocol.protocol_id
    result["protocol_path"] = (
        None
        if protocol_path_value is None
        else str(Path(protocol_path_value).resolve())
    )
    result["checkpoint"] = asdict(checkpoint_info)
    result["behavior_path"] = str(behavior_path.resolve())
    result["gate"] = evaluate_geometry_gate(result, load_json(gate_path))
    return result


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Test whether query-hidden geometry follows subjective rankings."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GEOMETRY_GATE_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_geometry_analysis(parsed.checkpoint, parsed.behavior, parsed.gate)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
