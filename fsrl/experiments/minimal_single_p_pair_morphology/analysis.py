"""Locked read-only analysis of all frozen M2 pair-morphology artifacts."""

from __future__ import annotations

from collections import Counter

import numpy as np

from fsrl.analysis.behavioral import (
    analyze_sampled_query_policy,
    fit_beta_distribution,
)
from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.analysis.policy import bundle_logits, exact_probability
from fsrl.experiments.pl_crosstalk_decomposition.storage import (
    load_npz,
    write_npz_exclusive,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .locks import reference, validate_source_input_lock
from .methods import (
    hodge_components,
    interface_checks,
    panel_stage,
    sample_pair_accuracies,
    study_outcome,
)
from .protocol import PROTOCOL_SHA256, RUNS, SOURCE_INPUT_LOCK, specification

BETA_CLASSES = (
    "ordinary_unimodal",
    "high_accuracy",
    "low_accuracy",
    "bimodal",
    "boundary",
    "not_fit",
)
CONDITIONS = ("clean", "folded", "noisy")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _subject_logits(logits: np.ndarray, protocol):
    schedules = (ordered_pairs(protocol.n_items),) * len(logits)
    return bundle_logits({"logits": logits}, schedules)


def _analysis_mask(behavior: dict) -> np.ndarray:
    return np.asarray(
        [
            row["overall_accuracy"] >= 0.5 and row["ranking_class"] != "correct"
            for row in behavior["subjects"]
        ],
        dtype=bool,
    )


def _replay_route(logits, stored, protocol, choice_seed, temperature):
    keyed = _subject_logits(logits, protocol)
    replayed = analyze_sampled_query_policy(
        protocol, keyed, seed=choice_seed, temperature=temperature
    )
    if replayed != stored:
        raise RuntimeError("frozen sampled behavior does not replay exactly")
    accuracy = sample_pair_accuracies(
        protocol, keyed, seed=choice_seed, temperature=temperature
    )
    means = np.asarray([row["mean_accuracy_all"] for row in stored["pairs"]])
    error = float(np.max(np.abs(accuracy.mean(0) - means)))
    if error > 1e-12:
        raise RuntimeError(f"replayed pair means differ: {error}")
    return replayed, accuracy, error


def _latent_probability(logits, geometry, temperature):
    oriented = logits.reshape(len(logits), len(geometry.pairs), 2)
    signs = geometry.true_sign[None, :, None] * np.asarray([1.0, -1.0])
    return exact_probability(oriented * signs, temperature).mean(2)


def _pair_rows(
    exact_p,
    accuracy,
    behavior,
    mask,
    field,
    gradient,
    residual,
    removed,
    geometry,
    protocol,
):
    if not np.any(mask):
        raise RuntimeError("registered analysis cohort is empty")
    removed_field = 0.5 * (removed[:, :, 0::2] - removed[:, :, 1::2])
    delta = removed_field - field[None]
    selected_delta = delta[:, mask]
    sign_flips = (removed_field[:, mask] * field[None, mask]) < 0.0
    classes = {name: index for index, name in enumerate(BETA_CLASSES)}
    pairs = []
    latent_names = []
    sampled_names = []
    for index, pair in enumerate(geometry.pairs):
        values = exact_p[mask, index]
        latent = fit_beta_distribution(values)["class"]
        sampled = behavior["pairs"][index]["beta_fit_analysis"]["class"]
        effects = np.mean(np.abs(selected_delta[:, :, index]), axis=1)
        pairs.append(
            {
                "exact_mean": float(np.mean(values)),
                "fraction_wrong_direction": float(np.mean(values < 0.5)),
                "fraction_strong_error": float(np.mean(values <= 0.2)),
                "fraction_strong_correct": float(np.mean(values >= 0.8)),
                "sampled_mean": float(np.mean(accuracy[mask, index])),
                "latent_class": classes[latent],
                "sampled_class": classes[sampled],
                "gradient_correct_signed_mean": float(
                    np.mean(gradient[mask, index] * geometry.true_sign[index])
                ),
                "residual_correct_signed_mean": float(
                    np.mean(residual[mask, index] * geometry.true_sign[index])
                ),
                "removal_absolute_mean": float(
                    np.mean(np.abs(selected_delta[:, :, index]))
                ),
                "removal_sign_flip_fraction": float(np.mean(sign_flips[:, :, index])),
                "strongest_support_relation": int(np.argmax(effects)),
                "direct_support": any(
                    set(pair) == set(relation)
                    for relation in protocol.support_pairs_higher_lower
                ),
            }
        )
        latent_names.append(latent)
        sampled_names.append(sampled)
    direction = sum(
        np.any(exact_p[mask, index] < 0.5) and np.any(exact_p[mask, index] > 0.5)
        for index in range(len(geometry.pairs))
    )
    strong = sum(
        np.any(exact_p[mask, index] <= 0.2) and np.any(exact_p[mask, index] >= 0.8)
        for index in range(len(geometry.pairs))
    )
    latent_bimodal = latent_names.count("bimodal")
    sampled_bimodal = sampled_names.count("bimodal")
    strong_matrix = exact_p[mask] <= 0.2
    pair_totals = strong_matrix.sum(0)
    total = int(pair_totals.sum())
    return pairs, {
        "direction_present_pairs": int(direction),
        "strong_two_sided_pairs": int(strong),
        "latent_bimodal_pairs": int(latent_bimodal),
        "sampled_bimodal_pairs": int(sampled_bimodal),
        "strong_error_subject_prevalence": float(np.mean(strong_matrix.any(1))),
        "strong_error_cases": total,
        "strong_error_pair_prevalence_mean": float(np.mean(strong_matrix.mean(0))),
        "strong_error_pair_prevalence_max": float(np.max(strong_matrix.mean(0))),
        "strong_error_top5_share": (
            None if total == 0 else float(np.sort(pair_totals)[-5:].sum() / total)
        ),
    }


def _unit(seed, panel, condition, lock, protocol, geometry):
    identity = f"{seed}/{panel}/{condition}"
    files = lock["m2_units"][identity]
    raw = load_npz(reference_path(files["raw.npz"]))
    stored = load_json(reference_path(files["behavior.json"]))
    run = load_json(reference_path(files["run.json"]))
    settings = run["resolved_config"]["liu"]
    full, accuracy, replay_error = _replay_route(
        raw["bundles__intact__logits"],
        stored["full"],
        protocol,
        settings["choice_seed"],
        settings["temperature"],
    )
    _replay_route(
        raw["bundles__local_off__logits"],
        stored["global"],
        protocol,
        settings["choice_seed"],
        settings["temperature"],
    )
    logits = raw["bundles__intact__logits"]
    field = 0.5 * (logits[:, 0::2] - logits[:, 1::2])
    probability = _latent_probability(logits, geometry, settings["temperature"])
    potentials = hodge_potentials(field, geometry)
    field_error = float(np.max(np.abs(field - raw["routes__full__internal__field"])))
    probability_error = float(
        np.max(np.abs(probability - raw["routes__full__internal__correct_probability"]))
    )
    potential_error = float(
        np.max(np.abs(potentials - raw["routes__full__internal__potentials"]))
    )
    if max(field_error, probability_error, potential_error) > 1e-12:
        raise RuntimeError(f"archived latent arrays fail parity: {identity}")
    gradient, residual = hodge_components(field, geometry)
    reconstruction = float(np.max(np.abs(field - gradient - residual)))
    inner = np.sum(gradient * residual, axis=1)
    energy = np.sum(field * field, axis=1)
    orthogonality = float(np.max(np.abs(inner) / np.maximum(energy, 1e-30)))
    if reconstruction > 1e-12 or orthogonality > 1e-10:
        raise RuntimeError(f"Hodge identity failed: {identity}")
    mask = _analysis_mask(full)
    pairs, counts = _pair_rows(
        probability,
        accuracy,
        full,
        mask,
        field,
        gradient,
        residual,
        raw["removed"],
        geometry,
        protocol,
    )
    target = specification()["decision"]["target_pairs"]
    stage = panel_stage(
        counts["direction_present_pairs"],
        counts["strong_two_sided_pairs"],
        counts["latent_bimodal_pairs"],
        counts["sampled_bimodal_pairs"],
        target=target,
    )
    return {
        "seed": seed,
        "panel": panel,
        "condition": condition,
        "analysis_subjects": int(mask.sum()),
        "stage": stage,
        "counts": counts,
        "geometry": {
            "gradient_rms": float(np.sqrt(np.mean(gradient[mask] ** 2))),
            "residual_rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "gradient_energy_fraction": float(
                np.mean(
                    np.sum(gradient[mask] ** 2, axis=1)
                    / np.sum(field[mask] ** 2, axis=1)
                )
            ),
        },
        "parity": {
            "sampled_behavior_exact": True,
            "pair_mean_max_abs_error": replay_error,
            "field_max_abs_error": field_error,
            "probability_max_abs_error": probability_error,
            "potential_max_abs_error": potential_error,
            "hodge_reconstruction_max_abs_error": reconstruction,
            "hodge_relative_orthogonality_max": orthogonality,
        },
        "pairs": pairs,
        "arrays": {
            "probability": probability,
            "field": field,
            "gradient": gradient,
            "residual": residual,
        },
    }


def reference_path(row: dict):
    from fsrl.paths import REPO_ROOT

    return REPO_ROOT / row["path"]


def _transition(current, clean, evidence, clean_evidence):
    current_p = current["arrays"]["probability"]
    clean_p = clean["arrays"]["probability"]
    delta_field = current["arrays"]["field"] - clean["arrays"]["field"]
    geometry = build_complete_graph_geometry(load_registered_protocol("liu_v2"))
    gradient, residual = hodge_components(delta_field, geometry)
    categories = lambda values: np.where(
        values <= 0.2, 0, np.where(values >= 0.8, 2, 1)
    )
    return {
        "direction_flip_fraction": float(np.mean((current_p < 0.5) != (clean_p < 0.5))),
        "stable_category_crossing_fraction": float(
            np.mean(categories(current_p) != categories(clean_p))
        ),
        "gradient_delta_rms": float(np.sqrt(np.mean(gradient**2))),
        "residual_delta_rms": float(np.sqrt(np.mean(residual**2))),
        "mean_absolute_evidence_delta": float(
            np.mean(np.abs(evidence - clean_evidence))
        ),
        "mean_absolute_amplitude_delta": float(
            np.mean(np.abs(np.abs(evidence) - np.abs(clean_evidence)))
        ),
    }


def _score_interface(lock):
    score_rows = {}
    for seed, row in lock["score_only"].items():
        raw = load_npz(reference_path(row))
        score_rows[seed] = {
            "support_cues": raw["liu__inputs__support_cues"],
            "signed": raw["liu__inputs__signed"],
            "retention": raw["liu__inputs__retention"],
            "probabilities": raw["liu__inputs__probabilities"],
            "local_evidence": raw["liu__inputs__local_evidence"],
            "query_cues": raw["liu__inputs__query_cues"],
            "codes": raw["liu__inputs__codes"],
            "support_pairs": raw["liu__inputs__support_pairs"],
            "query_pairs": raw["liu__inputs__query_pairs"],
        }
    first = score_rows[next(iter(score_rows))]
    score_seed_identity = all(
        all(np.array_equal(first[key], row[key]) for key in first)
        for row in score_rows.values()
    )
    panels = {}
    for panel in specification()["design"]["panels"]:
        m2 = load_npz(reference_path(lock["m2_inputs"][f"{panel}/clean"]))
        panels[str(panel)] = interface_checks(m2, first)
    semantic = {
        "task_timing_equal": True,
        "temperature_equal": True,
        "admission_semantics_equal": False,
        "admission_difference": (
            "score-only z=0 suppresses the entire normalized score update; M2 receives "
            "the realized q-only evidence value and does not read the zq bookkeeping channel"
        ),
    }
    exact = (
        score_seed_identity
        and all(all(checks.values()) for checks in panels.values())
        and all(
            semantic[key]
            for key in (
                "task_timing_equal",
                "temperature_equal",
                "admission_semantics_equal",
            )
        )
    )
    return {
        "outcome": "operator_comparison_qualified" if exact else "complete_recipe_only",
        "score_seed_input_identity": score_seed_identity,
        "panels": panels,
        "semantic_checks": semantic,
        "pair_geometry_comparison_performed": exact,
    }


def _table(units, geometry):
    rows = sorted(
        units,
        key=lambda row: (
            row["seed"],
            row["panel"],
            CONDITIONS.index(row["condition"]),
        ),
    )
    names = tuple(rows[0]["pairs"][0])
    arrays = {
        "unit_seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "unit_panel": np.asarray([row["panel"] for row in rows], dtype=np.int64),
        "unit_condition": np.asarray(
            [CONDITIONS.index(row["condition"]) for row in rows], dtype=np.int8
        ),
        "unit_analysis_subjects": np.asarray(
            [row["analysis_subjects"] for row in rows], dtype=np.int64
        ),
        "pair_items": np.asarray(geometry.pairs, dtype=np.int64),
        "pair_true_sign": geometry.true_sign.astype(np.float64),
    }
    for name in names:
        dtype = (
            np.int64
            if name in {"latent_class", "sampled_class", "strongest_support_relation"}
            else None
        )
        arrays[name] = np.asarray(
            [[pair[name] for pair in row["pairs"]] for row in rows], dtype=dtype
        )
    return arrays


def _aggregate(units):
    noisy = [row for row in units if row["condition"] == "noisy"]
    stage_counts = dict(sorted(Counter(row["stage"] for row in noisy).items()))
    outcome = study_outcome(stage_counts, required=40)
    count_names = (
        "direction_present_pairs",
        "strong_two_sided_pairs",
        "latent_bimodal_pairs",
        "sampled_bimodal_pairs",
    )
    counts = {
        name: {
            "mean": float(np.mean([row["counts"][name] for row in noisy])),
            "median": float(np.median([row["counts"][name] for row in noisy])),
            "minimum": int(min(row["counts"][name] for row in noisy)),
            "maximum": int(max(row["counts"][name] for row in noisy)),
        }
        for name in count_names
    }
    wrong = sum(row["counts"]["strong_error_cases"] for row in noisy)
    return {
        "outcome": outcome,
        "noisy_stage_counts": stage_counts,
        "noisy_pair_stage_counts": counts,
        "noisy_strong_error_subject_prevalence_mean": float(
            np.mean([row["counts"]["strong_error_subject_prevalence"] for row in noisy])
        ),
        "noisy_strong_error_top5_share_mean": float(
            np.mean(
                [
                    row["counts"]["strong_error_top5_share"]
                    for row in noisy
                    if row["counts"]["strong_error_top5_share"] is not None
                ]
            )
        ),
        "noisy_total_strong_error_cases": int(wrong),
    }


def analyze_all() -> dict:
    lock = validate_source_input_lock()
    spec = specification()
    protocol = load_registered_protocol("liu_v2")
    geometry = build_complete_graph_geometry(protocol)
    output = RUNS / "analysis"
    with ProspectiveRun.start(
        output,
        workflow_id="minimal_single_p_pair_morphology_attribution_v1",
        execution_id="all-frozen-units",
        producer={"source_input_lock": reference(SOURCE_INPUT_LOCK)},
        resolved_config=spec["estimands"],
    ):
        units = []
        by_identity = {}
        evidence = {}
        for panel in spec["design"]["panels"]:
            for condition in CONDITIONS:
                evidence[(panel, condition)] = load_npz(
                    reference_path(lock["m2_inputs"][f"{panel}/{condition}"])
                )["local_evidence"]
        for seed in spec["design"]["M2_network_seeds"]:
            for panel in spec["design"]["panels"]:
                for condition in CONDITIONS:
                    row = _unit(seed, panel, condition, lock, protocol, geometry)
                    units.append(row)
                    by_identity[(seed, panel, condition)] = row
        transitions = {}
        for seed in spec["design"]["M2_network_seeds"]:
            for panel in spec["design"]["panels"]:
                clean = by_identity[(seed, panel, "clean")]
                transitions[f"{seed}/{panel}"] = {
                    condition: _transition(
                        by_identity[(seed, panel, condition)],
                        clean,
                        evidence[(panel, condition)],
                        evidence[(panel, "clean")],
                    )
                    for condition in ("folded", "noisy")
                }
        aggregate = _aggregate(units)
        comparator = _score_interface(lock)
        table = _table(units, geometry)
        write_npz_exclusive(output / "pair_table.npz", table)
        public_units = [
            {key: value for key, value in row.items() if key not in {"arrays", "pairs"}}
            for row in units
        ]
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_input_lock": reference(SOURCE_INPUT_LOCK),
            "units": public_units,
            "transitions": transitions,
            "aggregate": aggregate,
            "score_interface": comparator,
            "pair_table": reference(output / "pair_table.npz"),
            "selection_performed": False,
            "model_execution_performed": False,
            "main_model_promoted": False,
            "claim_boundary": spec["claim_boundary"],
        }
        write_json_exclusive(output / "result.json", _json_ready(result))
    return {
        "outcome": aggregate["outcome"],
        "noisy_stage_counts": aggregate["noisy_stage_counts"],
        "score_interface": comparator["outcome"],
    }


__all__ = ["analyze_all"]
