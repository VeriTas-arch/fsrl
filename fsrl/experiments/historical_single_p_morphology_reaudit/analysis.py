"""Locked read-only analysis of historical alpha single-P artifacts."""

from __future__ import annotations

from collections import Counter

import numpy as np

from fsrl.analysis.hodge import (
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
)
from fsrl.experiments.minimal_single_p_pair_morphology.analysis import (
    _analysis_mask,
    _latent_probability,
    _pair_rows,
    _replay_route,
)
from fsrl.experiments.minimal_single_p_pair_morphology.methods import (
    hodge_components,
    panel_stage,
)
from fsrl.experiments.pl_crosstalk_decomposition.storage import (
    load_npz,
    write_npz_exclusive,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .decisions import (
    all_nine,
    constrained_unit,
    error_inflation,
    evidence_binding,
    family_replicated,
    inherited_competence,
    outcome,
)
from .locks import reference, validate_source_input_lock
from .protocol import PROTOCOL_SHA256, RUNS, SOURCE_INPUT_LOCK, specification

CONDITIONS = ("A0", "Ae")


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


def _path(row: dict):
    from fsrl.paths import REPO_ROOT

    return REPO_ROOT / row["path"]


def _morphology(logits, removed, behavior, task, settings):
    geometry = build_complete_graph_geometry(task)
    full, accuracy, replay_error = _replay_route(
        logits, behavior["full"], task, settings["choice_seed"], settings["temperature"]
    )
    exact_p = _latent_probability(logits, geometry, settings["temperature"])
    field = 0.5 * (logits[:, 0::2] - logits[:, 1::2])
    gradient, residual = hodge_components(field, geometry)
    mask = _analysis_mask(full)
    pairs, counts = _pair_rows(
        exact_p,
        accuracy,
        full,
        mask,
        field,
        gradient,
        residual,
        removed,
        geometry,
        task,
    )
    positions = np.empty(task.n_items, dtype=int)
    for position, item in enumerate(task.true_order_high_to_low):
        positions[item] = position
    for row, pair in zip(pairs, geometry.pairs, strict=True):
        row["pair"] = list(pair)
        row["symbolic_distance"] = abs(int(positions[pair[0]] - positions[pair[1]]))
    by_distance = {}
    for label, selected in (
        ("1", [row for row in pairs if row["symbolic_distance"] == 1]),
        ("2", [row for row in pairs if row["symbolic_distance"] == 2]),
        ("3", [row for row in pairs if row["symbolic_distance"] == 3]),
        ("4+", [row for row in pairs if row["symbolic_distance"] >= 4]),
    ):
        by_distance[label] = {
            "pairs": len(selected),
            "latent_bimodal_fraction": float(
                np.mean([row["latent_class"] == 3 for row in selected])
            ),
            "sampled_bimodal_fraction": float(
                np.mean([row["sampled_class"] == 3 for row in selected])
            ),
            "fraction_strong_error": float(
                np.mean([row["fraction_strong_error"] for row in selected])
            ),
            "fraction_strong_correct": float(
                np.mean([row["fraction_strong_correct"] for row in selected])
            ),
        }
    counts["stage"] = panel_stage(
        counts["direction_present_pairs"],
        counts["strong_two_sided_pairs"],
        counts["latent_bimodal_pairs"],
        counts["sampled_bimodal_pairs"],
        target=specification()["decision"]["target_pairs"],
    )
    counts["analysis_subjects"] = int(mask.sum())
    counts["choice_replay_max_abs_error"] = replay_error
    counts["by_distance"] = by_distance
    counts["gradient_energy_fraction"] = float(
        np.mean(gradient_energy_fraction(field[mask], geometry))
    )
    return counts, pairs, field, exact_p, gradient, residual


def _unit(family, seed, panel, condition, lock, task, geometry):
    identity = f"{family}/{seed}/{panel}/{condition}"
    files = lock["units"][identity]
    raw = load_npz(_path(files["raw.npz"]))
    behavior = load_json(_path(files["behavior.json"]))
    archived = load_json(_path(files["result.json"]))
    run = load_json(_path(files["run.json"]))
    settings = run["resolved_config"]["liu"]
    if (
        settings["protocol_id"] != "liu_v2"
        or settings["subjects"] != 77
        or settings["temperature"] != 0.25
    ):
        raise RuntimeError(f"historical interface differs: {identity}")
    logits = raw["liu__bundles__intact__logits"]
    local_off = raw["liu__bundles__local_off__logits"]
    removed = raw["liu__removed"]
    if logits.shape != (77, 56) or removed.shape != (8, 77, 56):
        raise RuntimeError(f"historical array shape differs: {identity}")
    morphology, pairs, field, probability, gradient, residual = _morphology(
        logits, removed, behavior, task, settings
    )
    _, _, global_replay_error = _replay_route(
        local_off,
        behavior["global"],
        task,
        settings["choice_seed"],
        settings["temperature"],
    )
    potentials = hodge_potentials(field, geometry)
    parity = {
        "field_max_abs_error": float(
            np.max(np.abs(field - raw["liu__routes__full__internal__field"]))
        ),
        "probability_max_abs_error": float(
            np.max(
                np.abs(
                    probability
                    - raw["liu__routes__full__internal__correct_probability"]
                )
            )
        ),
        "potential_max_abs_error": float(
            np.max(np.abs(potentials - raw["liu__routes__full__internal__potentials"]))
        ),
        "full_pair_mean_max_abs_error": morphology["choice_replay_max_abs_error"],
        "global_pair_mean_max_abs_error": global_replay_error,
        "hodge_reconstruction_max_abs_error": float(
            np.max(np.abs(field - gradient - residual))
        ),
        "hodge_relative_orthogonality_max": float(
            np.max(
                np.abs(np.sum(gradient * residual, axis=1))
                / np.maximum(np.sum(field * field, axis=1), 1e-30)
            )
        ),
    }
    if (
        max(
            parity["field_max_abs_error"],
            parity["probability_max_abs_error"],
            parity["potential_max_abs_error"],
            parity["full_pair_mean_max_abs_error"],
            parity["global_pair_mean_max_abs_error"],
            parity["hodge_reconstruction_max_abs_error"],
        )
        > 1e-12
        or parity["hodge_relative_orthogonality_max"] > 1e-10
    ):
        raise RuntimeError(f"historical numerical parity failed: {identity}")
    nine = all_nine(archived)
    qualitative_rows = {
        name: bool(row["qualitative"])
        for name, row in archived["liu"]["routes"]["full"]["behavior"][
            "historical_nine_rows"
        ]["flags"].items()
    }
    binding = evidence_binding(archived)
    competent = inherited_competence(archived)
    bad_pair = any(row["sampled_class"] in {0, 2} for row in pairs)
    constrained = constrained_unit(
        competent=competent,
        binding=binding,
        nine=nine,
        latent_bimodal=morphology["latent_bimodal_pairs"],
        sampled_bimodal=morphology["sampled_bimodal_pairs"],
        bad_pair=bad_pair,
    )
    return {
        "family": family,
        "seed": seed,
        "panel": panel,
        "condition": condition,
        "inherited_competence": competent,
        "evidence_binding": binding,
        "all_nine_qualitative": nine,
        "qualitative_rows": qualitative_rows,
        "bad_sampled_pair": bad_pair,
        "constrained_morphology": constrained,
        "error_inflation": False,
        "stage": morphology["stage"],
        "counts": {
            key: morphology[key]
            for key in (
                "direction_present_pairs",
                "strong_two_sided_pairs",
                "latent_bimodal_pairs",
                "sampled_bimodal_pairs",
                "strong_error_subject_prevalence",
                "strong_error_cases",
                "strong_error_top5_share",
                "gradient_energy_fraction",
            )
        },
        "by_distance": morphology["by_distance"],
        "parity": parity,
        "pairs": pairs,
    }


def _pair_table(units, geometry):
    rows = sorted(
        units,
        key=lambda row: (
            row["family"],
            row["seed"],
            row["panel"],
            CONDITIONS.index(row["condition"]),
        ),
    )
    pair_names = (
        "exact_mean",
        "fraction_wrong_direction",
        "fraction_strong_error",
        "fraction_strong_correct",
        "sampled_mean",
        "latent_class",
        "sampled_class",
        "gradient_correct_signed_mean",
        "residual_correct_signed_mean",
        "removal_absolute_mean",
        "removal_sign_flip_fraction",
        "strongest_support_relation",
        "direct_support",
        "symbolic_distance",
    )
    arrays = {
        "unit_family": np.asarray(
            [0 if row["family"] == "local_memory_removal" else 1 for row in rows],
            dtype=np.int8,
        ),
        "unit_seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "unit_panel": np.asarray([row["panel"] for row in rows], dtype=np.int8),
        "unit_condition": np.asarray(
            [CONDITIONS.index(row["condition"]) for row in rows], dtype=np.int8
        ),
        "pair_items": np.asarray(geometry.pairs, dtype=np.int64),
        "pair_true_sign": geometry.true_sign.astype(np.float64),
    }
    integer = {
        "latent_class",
        "sampled_class",
        "strongest_support_relation",
        "symbolic_distance",
    }
    for name in pair_names:
        dtype = np.int64 if name in integer else None
        arrays[name] = np.asarray(
            [[pair[name] for pair in row["pairs"]] for row in rows], dtype=dtype
        )
    return arrays


def _aggregate(units, spec):
    primary = [row for row in units if row["condition"] == "Ae"]
    controls = {
        (row["family"], row["seed"], row["panel"]): row
        for row in units
        if row["condition"] == "A0"
    }
    for row in primary:
        control = controls[(row["family"], row["seed"], row["panel"])]
        row["control_sampled_bimodal_pairs"] = control["counts"][
            "sampled_bimodal_pairs"
        ]
        row["sampled_bimodal_pair_difference"] = (
            row["counts"]["sampled_bimodal_pairs"]
            - row["control_sampled_bimodal_pairs"]
        )
        row["error_inflation"] = error_inflation(
            sampled_bimodal=row["counts"]["sampled_bimodal_pairs"],
            control_sampled_bimodal=control["counts"]["sampled_bimodal_pairs"],
            nine=row["all_nine_qualitative"],
            bad_pair=row["bad_sampled_pair"],
        )
    families = {}
    replicated = 0
    for family, family_spec in spec["design"]["families"].items():
        selected = [row for row in primary if row["family"] == family]
        family_pass = family_replicated(selected, family_spec["seeds"])
        replicated += int(family_pass)
        families[family] = {
            "primary_units": len(selected),
            "inherited_competent_units": sum(
                row["inherited_competence"] for row in selected
            ),
            "evidence_binding_units": sum(row["evidence_binding"] for row in selected),
            "all_nine_units": sum(row["all_nine_qualitative"] for row in selected),
            "constrained_units": sum(row["constrained_morphology"] for row in selected),
            "error_inflation_units": sum(row["error_inflation"] for row in selected),
            "replicated_precedent": family_pass,
            "seeds_with_constrained_panel": sum(
                any(
                    row["seed"] == seed and row["constrained_morphology"]
                    for row in selected
                )
                for seed in family_spec["seeds"]
            ),
            "stage_counts": dict(
                sorted(Counter(row["stage"] for row in selected).items())
            ),
            "qualitative_failure_counts": dict(
                sorted(
                    Counter(
                        name
                        for row in selected
                        for name, passed in row["qualitative_rows"].items()
                        if not passed
                    ).items()
                )
            ),
            "control_sampled_bimodal_pairs_mean": float(
                np.mean([row["control_sampled_bimodal_pairs"] for row in selected])
            ),
            "sampled_bimodal_pair_difference_mean": float(
                np.mean([row["sampled_bimodal_pair_difference"] for row in selected])
            ),
            "latent_bimodal_pairs_mean": float(
                np.mean([row["counts"]["latent_bimodal_pairs"] for row in selected])
            ),
            "sampled_bimodal_pairs_mean": float(
                np.mean([row["counts"]["sampled_bimodal_pairs"] for row in selected])
            ),
            "strong_error_top5_share_mean": float(
                np.mean(
                    [
                        row["counts"]["strong_error_top5_share"]
                        for row in selected
                        if row["counts"]["strong_error_top5_share"] is not None
                    ]
                )
            ),
        }
    constrained = sum(row["constrained_morphology"] for row in primary)
    return {
        "outcome": outcome(constrained, replicated),
        "primary_units": len(primary),
        "constrained_units": constrained,
        "all_nine_units": sum(row["all_nine_qualitative"] for row in primary),
        "error_inflation_units": sum(row["error_inflation"] for row in primary),
        "replicated_families": replicated,
        "families": families,
    }


def analyze_all() -> dict:
    lock = validate_source_input_lock()
    spec = specification()
    task = load_registered_protocol("liu_v2")
    geometry = build_complete_graph_geometry(task)
    output = RUNS / "analysis"
    with ProspectiveRun.start(
        output,
        workflow_id="historical_single_p_morphology_reaudit_v1",
        execution_id="all-historical-units",
        producer={"source_input_lock": reference(SOURCE_INPUT_LOCK)},
        resolved_config=spec["estimands"],
    ):
        units = []
        for family, family_spec in spec["design"]["families"].items():
            for seed in family_spec["seeds"]:
                for panel in spec["design"]["panels"]:
                    for condition in CONDITIONS:
                        units.append(
                            _unit(
                                family,
                                seed,
                                panel,
                                condition,
                                lock,
                                task,
                                geometry,
                            )
                        )
        aggregate = _aggregate(units, spec)
        known = {
            "outcome": "no_current_standard_precedent",
            "constrained_units": 0,
            "all_nine_units": 0,
            "error_inflation_units": 18,
        }
        if any(aggregate[key] != value for key, value in known.items()):
            raise RuntimeError("reporting repair changed the known registered outcome")
        write_npz_exclusive(output / "pair_table.npz", _pair_table(units, geometry))
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_input_lock": reference(SOURCE_INPUT_LOCK),
            "units": [
                {key: value for key, value in row.items() if key != "pairs"}
                for row in units
            ],
            "aggregate": aggregate,
            "pair_table": reference(output / "pair_table.npz"),
            "selection_performed": False,
            "model_execution_performed": False,
            "training_performed": False,
            "claim_boundary": spec["claim_boundary"],
        }
        write_json_exclusive(output / "result.json", _json_ready(result))
    return {
        "outcome": aggregate["outcome"],
        "constrained_units": aggregate["constrained_units"],
        "all_nine_units": aggregate["all_nine_units"],
        "error_inflation_units": aggregate["error_inflation_units"],
    }


__all__ = ["analyze_all"]
