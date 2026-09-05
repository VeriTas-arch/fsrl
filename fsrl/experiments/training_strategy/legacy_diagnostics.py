"""Secondary own-global qualification and frozen posterior projection audits."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    hodge_potentials,
    potential_alignment,
)
from fsrl.analysis.posterior import ExactRankingPosterior, RelationEvidence
from fsrl.evaluation.contracts import FastWeightIntervention
from fsrl.evaluation.qualification import (
    DEFAULT_QUALIFICATION_PATH,
    evaluate_qualification,
)
from fsrl.infra.provenance import load_json

from .estimands import estimate


def terminal_posterior(evaluator) -> dict[str, np.ndarray]:
    exact = ExactRankingPosterior(evaluator.protocol.n_items, temperature=0.05)
    expected, maps = [], []
    for evidence in evaluator.realized_support_evidence():
        observations = [
            RelationEvidence(
                row["higher_item"],
                row["lower_item"],
                row["magnitude"],
                row["reliability"],
            )
            for row in evidence
        ]
        posterior = exact.fit(observations)
        expected.append(-(posterior.probabilities @ exact.positions.astype(np.float64)))
        maps.append(-exact.positions[posterior.map_index].astype(np.float64))
    return {"expected_rank": np.asarray(expected), "MAP": np.asarray(maps)}


def projection_audit(evaluator, bundles: dict, seed: int, statistics: dict) -> dict:
    geometry = build_complete_graph_geometry(evaluator.protocol)
    posterior = terminal_posterior(evaluator)
    raw, summary = {}, {}
    for name in ("intact", "local_off", "P_off"):
        margins = bundles[name]["logits"]
        potential = hodge_potentials((margins[:, ::2] - margins[:, 1::2]) / 2, geometry)
        alignments = {
            key: potential_alignment(potential, value)["cosine"]
            for key, value in posterior.items()
        }
        raw[name] = {
            **alignments,
            "expected_minus_MAP": alignments["expected_rank"] - alignments["MAP"],
        }
        summary[name] = {
            key: estimate(values, seed=seed, statistics=statistics)
            for key, values in raw[name].items()
        }
    return {"posterior_temperature": 0.05, "summary": summary, "raw_subject": raw}


def own_global_qualification(
    evaluator, fast_weights, metadata: dict, specification: dict
) -> dict:
    conditions, winners = {}, {}
    for intervention in FastWeightIntervention:
        metrics, decisions = evaluator.condition_evaluation(intervention)
        conditions[intervention.value], winners[intervention.value] = (
            asdict(metrics),
            decisions,
        )
    intact = winners["intact"]
    for name, rows in winners.items():
        agreements = [
            winner == intact[subject][pair]
            for subject, row in enumerate(rows)
            for pair, winner in row.items()
        ]
        conditions[name]["mean_pair_decision_agreement_to_intact"] = float(
            np.mean(agreements)
        )
    invariance = evaluator.order_invariance(
        fast_weights, schedules=8, seed=specification["evaluation"]["liu"]["order_seed"]
    )
    observed = {
        "cue_mode": evaluator.cue_mode,
        "subject_encoding": {"mode": evaluator.subject_encoding_mode},
        "training_provenance": {
            "present": True,
            "checkpoint_sha_matches": True,
            "task_distribution": metadata["task_distribution"],
        },
        "conditions": conditions,
        "order_invariance": asdict(invariance),
    }
    return {
        "observed": observed,
        "qualification": evaluate_qualification(
            observed, load_json(DEFAULT_QUALIFICATION_PATH)
        ),
        "role": "Secondary own-global diagnostic, not the joint-model primary competence gate.",
    }
