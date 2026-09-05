"""Compose the registered endpoints without substituting canonical probabilities."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.relational_transport import (
    constructive_metrics,
    relation_loo_metrics,
)
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.tasks.protocol import RankingProtocol

from .estimands import estimate, paired_estimate, query_endpoints


def summarize_endpoints(endpoints: dict, seed: int, statistics: dict) -> dict:
    return {
        metric: {
            group: estimate(values, seed=seed, statistics=statistics)
            for group, values in groups.items()
        }
        for metric, groups in endpoints.items()
    }


def liu_endpoints(
    bundles: dict, retention: np.ndarray, protocol: RankingProtocol, temperature: float
) -> dict:
    geometry = build_complete_graph_geometry(protocol)
    learned = np.asarray([pair in protocol.learned_pairs for pair in geometry.pairs])
    retained = np.zeros((retention.shape[0], len(geometry.pairs)), dtype=bool)
    for index, relation in enumerate(protocol.support_pairs_higher_lower):
        retained[:, geometry.pairs.index(tuple(sorted(relation)))] = retention[:, index]
    groups = {
        "overall": np.ones_like(learned),
        "learned": learned,
        "nonlearned": ~learned,
        "retained": retained,
        "omitted": learned & ~retained,
    }
    signs = geometry.true_sign[:, None] * np.asarray([1.0, -1.0])
    return {
        name: query_endpoints(
            bundle["logits"].reshape(retention.shape[0], -1, 2),
            signs,
            groups,
            temperature=temperature,
        )
        for name, bundle in bundles.items()
    }


def summarize_geometry(
    bundles: dict, loo: dict, protocol: RankingProtocol, seed: int, statistics: dict
) -> dict:
    geometry = build_complete_graph_geometry(protocol)
    fields = {
        name: (bundle["logits"][:, ::2] - bundle["logits"][:, 1::2]) / 2
        for name, bundle in bundles.items()
    }
    counts = bootstrap_counts(
        np.random.default_rng(seed), statistics["samples"], fields["intact"].shape[0]
    )
    constructive = constructive_metrics(
        fields["intact"], fields["local_off"], geometry, counts, statistics["interval"]
    )
    constructive_summary = {
        name: estimate(
            np.asarray(values, dtype=np.float64), seed=seed, statistics=statistics
        )
        for name, values in constructive["raw_subject"].items()
    }
    relations = protocol.support_pairs_higher_lower
    raw = {}
    relation_subject = {}
    for branch, intact in (
        ("global", "local_off"),
        ("local", "P_off"),
        ("combined", "intact"),
    ):
        margins = loo[branch]
        field_loo = (margins[:, :, ::2] - margins[:, :, 1::2]) / 2
        measured = relation_loo_metrics(
            fields[intact],
            field_loo,
            relations,
            geometry,
            counts,
            statistics["interval"],
        )
        raw[branch] = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in measured["raw_subject"].items()
        }
        relation_subject[branch] = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in measured["raw_relation_subject"].items()
        }
    return {
        "constructive": constructive_summary,
        "loo_subject": raw,
        "loo_relation_subject": relation_subject,
    }


def mechanism_effects(
    endpoints: dict, geometry: dict, seed: int, statistics: dict
) -> dict:
    probabilities = {name: row["probability"] for name, row in endpoints.items()}
    differences = {
        "intact_minus_P_off_nonlearned": ("P_off", "nonlearned"),
        "intact_minus_local_off_retained": ("local_off", "retained"),
        "intact_minus_local_off_omitted": ("local_off", "omitted"),
        "intact_minus_query_shuffle_learned": ("query_shuffle", "learned"),
        "intact_minus_evidence_shuffle_learned": ("evidence_shuffle", "learned"),
    }
    result = {
        name: paired_estimate(
            probabilities["intact"][group],
            probabilities[condition][group],
            seed=seed,
            statistics=statistics,
        )
        for name, (condition, group) in differences.items()
    }
    loo = geometry["loo_subject"]
    direct = {
        "P_off_nonlearned": probabilities["P_off"]["nonlearned"],
        "P_off_learned": probabilities["P_off"]["learned"],
        "global_remote_absolute": loo["global"]["remote_absolute"],
        "global_third_party_relational": loo["global"]["third_party_relational"],
        "local_remote_minus_quarter_combined": loo["local"]["remote_absolute"]
        - 0.25 * loo["combined"]["remote_absolute"],
    }
    result.update(
        {
            name: estimate(values, seed=seed, statistics=statistics)
            for name, values in direct.items()
        }
    )
    return result


def paired_endpoints(
    joint: dict, staged: dict, network_seed: int, specification: dict
) -> dict:
    contract = specification["decision_contract"]["paired_noninferiority"]
    return {
        domain: {
            group: paired_estimate(
                joint[domain]["intact"]["probability"][group],
                staged[domain]["intact"]["probability"][group],
                seed=(85000 if domain == "liu" else 86000) + network_seed,
                statistics=specification["statistics"],
            )
            for group in contract[f"{domain}_groups"]
        }
        for domain in ("generic", "liu")
    }
