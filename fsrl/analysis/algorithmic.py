"""Compare frozen neural rankings with exact inference from realized evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats

from fsrl.analysis.behavioral import hodge_rank_positions, kendall_tau_positions
from fsrl.analysis.posterior import ExactRankingPosterior, RelationEvidence
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from fsrl.tasks.registered_protocol import load_ranking_protocol


def _order_positions(order: tuple[int, ...] | list[int]) -> np.ndarray:
    positions = np.empty(len(order), dtype=np.int64)
    positions[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
    return positions


def compare_neural_policy_to_exact_posterior(
    protocol,
    realized_evidence: tuple[tuple[dict, ...], ...],
    subject_logits: tuple[dict[tuple[int, int], float], ...],
    behavior: dict,
    *,
    posterior_temperature: float,
    readout_temperature: float,
) -> dict:
    if len(realized_evidence) != len(subject_logits):
        raise ValueError("evidence and neural logits have different cohorts")
    if len(behavior["subjects"]) != len(subject_logits):
        raise ValueError("behavior and neural logits have different cohorts")

    exact = ExactRankingPosterior(protocol.n_items, temperature=posterior_temperature)
    order_to_index = {
        tuple(int(item) for item in order): index
        for index, order in enumerate(exact.orders)
    }
    pairs = tuple(combinations(range(protocol.n_items), 2))
    rows = []
    for subject_index, (evidence_rows, logits) in enumerate(
        zip(realized_evidence, subject_logits, strict=True)
    ):
        evidence = tuple(
            RelationEvidence(
                higher_item=row["higher_item"],
                lower_item=row["lower_item"],
                magnitude=row["magnitude"],
                reliability=row["reliability"],
            )
            for row in evidence_rows
        )
        posterior = exact.fit(evidence)
        minimum_energy = float(np.min(posterior.energy))
        map_indices = np.flatnonzero(
            np.isclose(posterior.energy, minimum_energy, rtol=0.0, atol=1e-12)
        )

        preference = np.zeros((protocol.n_items, protocol.n_items))
        neural_pair_probability = []
        posterior_pair_probability = []
        for first, second in pairs:
            margin = 0.5 * (logits[(first, second)] - logits[(second, first)])
            preference[first, second] = margin
            preference[second, first] = -margin
            neural_pair_probability.append(
                float(1.0 / (1.0 + np.exp(-margin / readout_temperature)))
            )
            posterior_pair_probability.append(
                exact.pair_probability(posterior, first, second)
            )
        neural_positions = hodge_rank_positions(preference)
        neural_order = tuple(int(item) for item in np.argsort(neural_positions))
        neural_index = order_to_index[neural_order]
        closest_map_tau = max(
            kendall_tau_positions(
                neural_positions,
                exact.positions[int(map_index)],
            )
            for map_index in map_indices
        )
        correlation_result = cast(
            Any,
            stats.spearmanr(neural_pair_probability, posterior_pair_probability),
        )
        pair_probability_correlation = correlation_result.statistic
        pair_probability_correlation = (
            None
            if not np.isfinite(pair_probability_correlation)
            else float(pair_probability_correlation)
        )
        posterior_choice_agreement = np.mean(
            [
                probability if neural_probability > 0.5 else 1.0 - probability
                for neural_probability, probability in zip(
                    neural_pair_probability,
                    posterior_pair_probability,
                    strict=True,
                )
            ]
        )
        subject_behavior = behavior["subjects"][subject_index]
        rows.append(
            {
                "subject": subject_index,
                "behavior_class": subject_behavior["ranking_class"],
                "neural_order_high_to_low": list(neural_order),
                "behavior_order_high_to_low": subject_behavior[
                    "subjective_order_high_to_low"
                ],
                "map_order_high_to_low": [
                    int(item) for item in exact.orders[int(map_indices[0])]
                ],
                "map_order_count": len(map_indices),
                "neural_is_map": bool(neural_index in set(map_indices.tolist())),
                "neural_posterior_probability": float(
                    posterior.probabilities[neural_index]
                ),
                "closest_map_kendall_tau": float(closest_map_tau),
                "neural_to_behavior_kendall_tau": float(
                    kendall_tau_positions(
                        neural_positions,
                        _order_positions(
                            subject_behavior["subjective_order_high_to_low"]
                        ),
                    )
                ),
                "pair_probability_spearman": pair_probability_correlation,
                "posterior_expected_agreement_with_neural_choices": float(
                    posterior_choice_agreement
                ),
                "posterior_entropy_nats": exact.posterior_entropy(posterior),
                "retained_presentations": int(
                    sum(row["reliability"] > 0.0 for row in evidence_rows)
                ),
                "mean_evidence_reliability": float(
                    np.mean([row["reliability"] for row in evidence_rows])
                ),
            }
        )

    def mean(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        return None if not values else float(np.mean(values))

    return {
        "estimand": {
            "evidence": "realized support magnitude and encoding reliability",
            "hypothesis_space": f"all {exact.n_hypotheses} global orders",
            "posterior_energy": "reliability-weighted squared magnitude residual",
            "neural_order": "Hodge ranking of frozen orientation-antisymmetric logits",
            "claim_status": "algorithmic comparison; not a qualification gate",
        },
        "posterior_temperature": posterior_temperature,
        "readout_temperature": readout_temperature,
        "group": {
            "subjects": len(rows),
            "neural_map_proportion": float(
                np.mean([row["neural_is_map"] for row in rows])
            ),
            "mean_closest_map_kendall_tau": mean("closest_map_kendall_tau"),
            "mean_pair_probability_spearman": mean("pair_probability_spearman"),
            "mean_posterior_expected_agreement_with_neural_choices": mean(
                "posterior_expected_agreement_with_neural_choices"
            ),
            "mean_neural_posterior_probability": mean("neural_posterior_probability"),
        },
        "subjects": rows,
    }


def run_algorithmic_comparison(
    checkpoint: Path,
    behavior_path: Path,
    *,
    posterior_temperature: float = 0.05,
) -> dict:
    with behavior_path.open(encoding="utf-8") as handle:
        behavior = json.load(handle)
    protocol = load_ranking_protocol(behavior["protocol_path"])
    batch_size = len(behavior["subjects"])
    net, config, checkpoint_info = load_retro_checkpoint(checkpoint, batch_size)
    if behavior.get("checkpoint", {}).get("sha256") != checkpoint_info.sha256:
        raise ValueError("behavior result and checkpoint SHA-256 do not match")
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
    logits = evaluator.readout_logits(
        fast_weights, tuple(ordered_pairs for _ in range(batch_size))
    )
    result = compare_neural_policy_to_exact_posterior(
        protocol,
        evaluator.realized_support_evidence(),
        logits,
        behavior,
        posterior_temperature=posterior_temperature,
        readout_temperature=float(behavior["sampling"]["temperature"]),
    )
    result["protocol_id"] = protocol.protocol_id
    result["protocol_path"] = behavior["protocol_path"]
    result["checkpoint"] = asdict(checkpoint_info)
    result["behavior_path"] = str(behavior_path.resolve())
    return result


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Compare frozen neural rankings with exact ranking posteriors."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posterior-temperature", type=float, default=0.05)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_algorithmic_comparison(
        parsed.checkpoint,
        parsed.behavior,
        posterior_temperature=parsed.posterior_temperature,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
