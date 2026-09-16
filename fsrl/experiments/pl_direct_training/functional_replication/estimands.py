"""Frozen functional-replication estimands and decision composition."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.policy import exact_probability, margin_fields
from fsrl.analysis.statistics import (
    bootstrap_counts,
    json_values,
    summarize_difference,
    summarize_subjects,
)
from fsrl.experiments.local_fidelity.evidence_access_pilot import field_metrics
from fsrl.tasks.protocol import ordered_pairs


def resampling_counts(seed: int, subjects: int, statistics: dict) -> np.ndarray:
    return bootstrap_counts(
        np.random.default_rng(89000 + seed),
        int(statistics["samples"]),
        subjects,
    )


def learned_probabilities(protocol, bundle: dict, temperature: float) -> np.ndarray:
    relations = tuple(protocol.support_pairs_higher_lower)
    pair_index = {
        pair: index for index, pair in enumerate(ordered_pairs(protocol.n_items))
    }
    subjects = int(bundle["logits"].shape[0])
    values = np.empty((subjects, len(relations), 2), dtype=np.float64)
    for relation_index, relation in enumerate(relations):
        for orientation, pair in enumerate((relation, relation[::-1])):
            sign = 1.0 if orientation == 0 else -1.0
            margin = sign * bundle["logits"][:, pair_index[pair]]
            values[:, relation_index, orientation] = exact_probability(
                margin, temperature
            )
    return values


def probability_metrics(
    probabilities: np.ndarray,
    retention: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    retained = np.broadcast_to(retention[:, :, None], probabilities.shape)

    def subject_mean(mask: np.ndarray) -> np.ndarray:
        rows = np.where(mask, probabilities, np.nan).reshape(probabilities.shape[0], -1)
        finite = np.sum(np.isfinite(rows), axis=1)
        return np.divide(
            np.nansum(rows, axis=1),
            finite,
            out=np.full(rows.shape[0], np.nan, dtype=np.float64),
            where=finite > 0,
        )

    raw = {
        "retained": subject_mean(retained),
        "omitted": subject_mean(~retained),
        "all": np.mean(probabilities, axis=(1, 2)),
    }
    return {
        "summary": {
            group: summarize_subjects(values, counts, interval=interval)
            for group, values in raw.items()
        },
        "raw_subject_level": {
            group: json_values(values) for group, values in raw.items()
        },
        "raw_relation_orientation": json_values(probabilities),
    }


def paired_summary(first, second, counts: np.ndarray, interval: float) -> dict:
    return summarize_difference(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        counts,
        interval=interval,
    )


def causal_fields(
    protocol,
    bundles: dict,
    loo: dict,
    retention: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> dict:
    geometry = build_complete_graph_geometry(protocol)
    relations = tuple(protocol.support_pairs_higher_lower)
    return {
        condition: field_metrics(
            margin_fields(bundle, protocol.n_items),
            np.asarray(
                [
                    margin_fields({"logits": row}, protocol.n_items)
                    for row in loo[condition]
                ]
            ),
            relations,
            retention.T,
            geometry,
            counts,
            interval,
        )
        for condition, bundle in bundles.items()
    }


def functional_contrasts(
    probabilities: dict,
    fields: dict,
    counts: np.ndarray,
    interval: float,
) -> dict:
    def probability_raw(condition: str, group: str):
        return probabilities[condition]["raw_subject_level"][group]

    def field_raw(condition: str, group: str, metric: str):
        return fields[condition]["raw_subject_level"][group][metric]

    return {
        "dual_minus_shared_omitted_exact_probability": paired_summary(
            probability_raw("dual_access", "omitted"),
            probability_raw("shared_access", "omitted"),
            counts,
            interval,
        ),
        "dual_minus_shared_retained_exact_probability": paired_summary(
            probability_raw("dual_access", "retained"),
            probability_raw("shared_access", "retained"),
            counts,
            interval,
        ),
        "dual_minus_shared_omitted_direct_correctness": paired_summary(
            field_raw("dual_access", "omitted", "direct_correctness"),
            field_raw("shared_access", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_shared_retained_direct_correctness": paired_summary(
            field_raw("dual_access", "retained", "direct_correctness"),
            field_raw("shared_access", "retained", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_evidence_shuffle_omitted_exact_probability": paired_summary(
            probability_raw("dual_access", "omitted"),
            probability_raw("dual_evidence_shuffle", "omitted"),
            counts,
            interval,
        ),
        "dual_minus_evidence_shuffle_omitted_direct_correctness": paired_summary(
            field_raw("dual_access", "omitted", "direct_correctness"),
            field_raw("dual_evidence_shuffle", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "dual_minus_query_shuffle_omitted_direct_correctness": paired_summary(
            field_raw("dual_access", "omitted", "direct_correctness"),
            field_raw("dual_query_shuffle", "omitted", "direct_correctness"),
            counts,
            interval,
        ),
        "P_off_dual_minus_shared_omitted_exact_probability": paired_summary(
            probability_raw("P_off_dual", "omitted"),
            probability_raw("P_off_shared", "omitted"),
            counts,
            interval,
        ),
        "P_off_all_remote_minus_quarter_shared_all_remote": summarize_subjects(
            np.asarray(
                field_raw("P_off_dual", "all", "remote_absolute"),
                dtype=np.float64,
            )
            - 0.25
            * np.asarray(
                field_raw("shared_access", "all", "remote_absolute"),
                dtype=np.float64,
            ),
            counts,
            interval=interval,
        ),
        "dual_minus_local_off_omitted_exact_probability": paired_summary(
            probability_raw("dual_access", "omitted"),
            probability_raw("local_off", "omitted"),
            counts,
            interval,
        ),
        "dual_minus_local_off_retained_exact_probability": paired_summary(
            probability_raw("dual_access", "retained"),
            probability_raw("local_off", "retained"),
            counts,
            interval,
        ),
    }


def four_link_flags(
    contrasts: dict,
    probabilities: dict,
    p_off_nonlearned: dict,
    retained_identity_error: float,
    global_path_passed: bool,
) -> dict[str, bool]:
    bootstrap = lambda row, edge: row["bootstrap"][edge]
    return {
        "omitted_direct_fidelity_rescue": bool(
            bootstrap(contrasts["dual_minus_shared_omitted_exact_probability"], "lower")
            > 0.0
            and bootstrap(
                contrasts["dual_minus_shared_omitted_direct_correctness"], "lower"
            )
            > 0.0
        ),
        "retained_fidelity_preservation": bool(
            bootstrap(
                contrasts["dual_minus_shared_retained_exact_probability"], "lower"
            )
            >= -0.005
            and retained_identity_error <= 1e-7
        ),
        "evidence_and_query_specificity": bool(
            bootstrap(
                contrasts["dual_minus_evidence_shuffle_omitted_exact_probability"],
                "lower",
            )
            > 0.0
            and bootstrap(
                contrasts["dual_minus_evidence_shuffle_omitted_direct_correctness"],
                "lower",
            )
            > 0.0
            and bootstrap(
                contrasts["dual_minus_query_shuffle_omitted_direct_correctness"],
                "lower",
            )
            > 0.0
        ),
        "local_only_nontransitive_access": bool(
            bootstrap(probabilities["P_off_dual"]["summary"]["omitted"], "lower") > 0.50
            and bootstrap(
                contrasts["P_off_dual_minus_shared_omitted_exact_probability"],
                "lower",
            )
            > 0.0
            and bootstrap(p_off_nonlearned, "upper") <= 0.55
            and bootstrap(
                contrasts["P_off_all_remote_minus_quarter_shared_all_remote"],
                "upper",
            )
            < 0.0
            and global_path_passed
        ),
    }


OUTCOME_PRECEDENCE = (
    "noninterpretable",
    "competence_failure",
    "competent_alternative_organization",
    "mechanism_replication_behavior_incomplete",
    "clean_no_time_pl_functional_replication",
)


def seed_outcome(
    *,
    integrity: bool,
    competence: bool,
    global_path: bool,
    four_links: dict[str, bool],
    omitted_materiality: bool,
    qualitative_behavior: bool,
) -> str:
    if not integrity:
        return "noninterpretable"
    if not competence:
        return "competence_failure"
    if not global_path or not all(four_links.values()) or not omitted_materiality:
        return "competent_alternative_organization"
    if not qualitative_behavior:
        return "mechanism_replication_behavior_incomplete"
    return "clean_no_time_pl_functional_replication"


def cohort_outcome(rows: dict[str, dict]) -> str:
    observed = {row["outcome"] for row in rows.values()}
    return next(label for label in OUTCOME_PRECEDENCE if label in observed)
