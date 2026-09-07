"""Participant-level contribution and fixed-panel structural contrasts."""

import numpy as np

from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
)
from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.evidence_routing.measurement import tau_matrix, weighted_tau
from fsrl.experiments.training_strategy.estimands import estimate, subject_means

from .protocol import specification


def query_groups(cpu, relations):
    queries = cpu.arrays["query_pairs"]
    support = {tuple(sorted(r)) for r in relations}
    learned = np.asarray([tuple(sorted(q)) in support for q in queries])
    return {
        "all": np.ones(len(queries), bool),
        "learned": learned,
        "nonlearned": ~learned,
    }


def effects(intact, changed, signs):
    temperature = specification()["metrics"]["temperature"]
    original, altered = intact * signs, changed * signs
    return {
        "ce_benefit": np.logaddexp(0, -altered / temperature)
        - np.logaddexp(0, -original / temperature),
        "probability_benefit": exact_probability(original, temperature)
        - exact_probability(altered, temperature),
        "absolute_margin": np.abs(intact - changed),
    }


def single_endpoints(raw, cpu, relations):
    weak = ~cpu.arrays["retention"].T
    groups = query_groups(cpu, relations)
    signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, weak.shape[1]).T
    collected = {}
    for index, relation in enumerate(relations):
        queries = cpu.arrays["query_pairs"]
        masks = {
            "direct": np.asarray([set(q) == set(relation) for q in queries]),
            "remote": np.asarray([not set(q).intersection(relation) for q in queries]),
            "nonlearned": groups["nonlearned"],
        }
        for name, values in effects(
            raw["intact_global"], raw["single_global"][index], signs
        ).items():
            for group, mask in masks.items():
                collected.setdefault(f"single_{group}_{name}", []).append(
                    subject_means(values, mask)
                )
    endpoints = {}
    for name, rows in collected.items():
        values = np.asarray(rows)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("undefined per-relation endpoint")
        count = weak.sum(0)
        endpoints[name] = np.divide(
            np.where(weak, values, 0).sum(0),
            count,
            out=np.full(len(count), np.nan),
            where=count > 0,
        )
    return endpoints


def structure(margins, protocol):
    geometry = build_complete_graph_geometry(protocol)
    field = (margins[:, ::2] - margins[:, 1::2]) / 2
    potentials = hodge_potentials(field, geometry)
    return {
        "field": field,
        "coherence": gradient_energy_fraction(field, geometry),
        "orders": np.argsort(-potentials, axis=1, kind="stable"),
    }


def endpoints(raw, cpu, protocol):
    relations = protocol.support_pairs_higher_lower
    result = single_endpoints(raw, cpu, relations)
    subjects = len(raw["intact_global"])
    signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, subjects).T
    groups = query_groups(cpu, relations)
    for name, values in effects(
        raw["replay_global"], raw["joint_global"], signs
    ).items():
        for group, mask in groups.items():
            result[f"joint_{group}_{name}"] = subject_means(values, mask)
    structural = {}
    for condition in ("intact", "joint"):
        baseline = "replay" if condition == "intact" else condition
        structural[condition] = structure(raw[f"{baseline}_global"], protocol)
        result[f"{condition}_coherence"] = structural[condition]["coherence"]
        for route in ("global", "complete"):
            probability = exact_probability(
                raw[f"{baseline}_{route}"] * signs,
                specification()["metrics"]["temperature"],
            )
            for group in ("learned", "nonlearned"):
                result[f"{condition}_{route}_{group}_probability"] = subject_means(
                    probability, groups[group]
                )
    result["joint_minus_intact_coherence"] = (
        result["joint_coherence"] - result["intact_coherence"]
    )
    for name, values in result.items():
        expected_missing = (
            (~cpu.arrays["retention"]).sum(1) == 0
            if name.startswith("single_")
            else np.zeros(subjects, bool)
        )
        if not np.array_equal(~np.isfinite(values), expected_missing):
            raise RuntimeError(f"unexpected missing endpoint: {name}")
    return result, structural


def summarize(values, seed):
    spec = specification()["statistics"]
    return estimate(values, seed=spec["seed_offset"] + seed, statistics=spec)


def tau_contrasts(structures, seed):
    subjects = len(next(iter(structures.values()))["orders"])
    counts = bootstrap_counts(
        np.random.default_rng(specification()["statistics"]["seed_offset"] + seed),
        specification()["statistics"]["samples"],
        subjects,
    )
    estimates, samples = {}, {}
    for name, row in structures.items():
        matrix = tau_matrix(row["orders"])
        estimates[name] = weighted_tau(matrix, np.ones((1, subjects)))[0]
        samples[name] = weighted_tau(matrix, counts)
    return estimates, samples


def interval(point, draws):
    invalid = int((~np.isfinite(draws)).sum())
    bounds = [None, None] if invalid else np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "point": float(point),
        "lower": bounds[0],
        "upper": bounds[1],
        "undefined_draws": invalid,
    }


def summarize_cell(values, structures, seed):
    scalar = {name: summarize(value, seed) for name, value in values.items()}
    points, draws = tau_contrasts(structures, seed)
    tau = {name: interval(points[name], draws[name]) for name in points}
    tau["joint_minus_intact"] = interval(
        points["joint"] - points["intact"], draws["joint"] - draws["intact"]
    )
    competence = {
        f"{condition}_{route}": all(
            scalar[f"{condition}_{route}_{group}_probability"]["bootstrap"]["lower"]
            > 0.5
            for group in ("learned", "nonlearned")
        )
        for condition in ("intact", "joint")
        for route in ("global", "complete")
    }
    coherence = {
        condition: scalar[f"{condition}_coherence"]["bootstrap"]["lower"] > 0.95
        for condition in ("intact", "joint")
    }
    return {
        "endpoints": scalar,
        "tau": tau,
        "competence": competence,
        "coherence": coherence,
    }


def summarize_pair(first, second, first_structure, second_structure, seed):
    scalar = {name: summarize(second[name] - first[name], seed) for name in first}
    structures = {
        f"{arm}_{name}": row
        for arm, rows in (("shared", first_structure), ("cost", second_structure))
        for name, row in rows.items()
    }
    points, draws = tau_contrasts(structures, seed)
    point = (points["cost_joint"] - points["cost_intact"]) - (
        points["shared_joint"] - points["shared_intact"]
    )
    sample = (draws["cost_joint"] - draws["cost_intact"]) - (
        draws["shared_joint"] - draws["shared_intact"]
    )
    return {"cost_minus_shared": scalar, "tau_interaction": interval(point, sample)}
