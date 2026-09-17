"""Registered historical-preservation and anytime comparison report."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.local_memory_removal.statistics import (
    noninferiority,
    paired_probability,
    probabilities,
)
from fsrl.experiments.observation_replication.protocol import (
    specification as analysis_specification,
)
from fsrl.experiments.observation_replication.reporting import read_raw, summarize
from fsrl.experiments.observation_replication.statistics import panel_draws, panel_mean
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .decisions import classify, interval_gate
from .evaluation import CELLS, legacy_input_record
from .locks import reference, validate_model_lock
from .protocol import (
    EVALUATION_RUNS,
    MODEL_LOCK,
    PROTOCOL,
    REPORT,
    RESULT,
    RUNS,
    historical_recipe,
    specification,
)

BOOTSTRAP_DRAWS = 2000


def _historical_cell_data(seed: int, panel: int, recipe_name: str):
    rows, raw = {}, {}
    for cell in CELLS:
        directory = (
            EVALUATION_RUNS / "historical" / str(seed) / str(panel) / recipe_name / cell
        )
        rows[cell] = completed(directory)
        raw[cell] = read_raw(reference(directory / "raw.npz"))
    return rows, raw


def _historical_panel(seed, panel, recipe_name, source):
    rows, raw = _historical_cell_data(seed, panel, recipe_name)
    analysis_spec = analysis_specification()
    sample_seed = seed + panel * 1000000
    result, endpoints = summarize(
        raw,
        rows,
        load_input(
            legacy_input_record(
                source["historical_panels"][str(panel)]["inputs"]["liu-8"]
            )
        ),
        sample_seed,
        historical_recipe(panel),
        analysis_spec,
    )
    points, draws = panel_draws(
        raw,
        endpoints["CE"],
        endpoints["order_shift"],
        analysis_spec["statistics"]["seed_offset"] + sample_seed,
        analysis_spec,
    )
    return result, raw, (points, draws)


def _compensation(mean: dict) -> dict:
    return {
        "positive_I_tau": mean["interaction/global_all77_tau"]["interval"]["lower"] > 0,
        "negative_I_CE": mean["interaction/generic_global"]["interval"]["upper"] < 0,
        "negative_shift_difference": mean["order_shift/C_minus_A"]["interval"]["upper"]
        < 0,
        "both_error_tau_points_negative": all(
            mean[f"{name}/global_all77_tau"]["point"] < 0
            for name in ("error_A", "error_C")
        ),
    }


def _historical_status(panels: dict, mean: dict, recipe_name: str) -> dict:
    competent_bound = all(
        panel[recipe_name]["decision"]["all_four_competent"]
        and all(panel[recipe_name]["evidence_binding"][cell] for cell in ("Ae", "Ce"))
        for panel in panels.values()
    )
    core = all(
        any(
            panel[recipe_name]["cells"][cell]["liu"]["routes"]["full"]["core_passed"]
            for panel in panels.values()
        )
        for cell in ("Ae", "Ce")
    )
    compensation = _compensation(mean)
    return {
        "competent_and_bound": competent_bound,
        "Ae_and_Ce_core_present": core,
        "compensation": compensation,
        "valid": competent_bound and core and all(compensation.values()),
    }


def _historical_pair(seed: int, source: dict):
    recipes = specification()["design"]["training_recipes"]
    panels, recipe_draws, raw = {}, {name: [] for name in recipes}, {}
    ni_panel = []
    analysis_spec = analysis_specification()
    for panel in specification()["design"]["evaluation_panels"]:
        panels[str(panel)], raw[str(panel)] = {}, {}
        recipe_raw = {}
        for recipe_name in recipes:
            result, cell_raw, draws = _historical_panel(
                seed, panel, recipe_name, source
            )
            panels[str(panel)][recipe_name] = result
            recipe_raw[recipe_name] = cell_raw
            recipe_draws[recipe_name].append(draws)
            raw[str(panel)][recipe_name] = draws[1]
        boot = analysis_spec["statistics"]["seed_offset"] + seed + panel * 1000000
        values = {}
        for cell in CELLS:
            values[cell], _ = paired_probability(
                probabilities(recipe_raw["variable_horizon"][cell]),
                probabilities(recipe_raw["fixed_horizon"][cell]),
                boot,
            )
        ni_panel.append(values)
    means, mean_draws = {}, {}
    for recipe_name, draws in recipe_draws.items():
        means[recipe_name], mean_draws[recipe_name] = panel_mean(draws)
    ni, ni_draws = {}, {}
    for cell in CELLS:
        ni[cell], ni_draws[cell] = panel_mean(
            [panel_values[cell] for panel_values in ni_panel]
        )
    fixed = _historical_status(panels, means["fixed_horizon"], "fixed_horizon")
    variable = _historical_status(panels, means["variable_horizon"], "variable_horizon")
    continuity = all(row["noninferior"] for row in noninferiority(ni["Ce"]).values())
    decision = {
        "fresh_fixed_valid": fixed["valid"],
        "fixed_horizon": fixed,
        "variable_horizon": variable,
        "Ce_five_endpoints_noninferior": continuity,
        "historical_preserved": variable["valid"] and continuity,
    }
    return {
        "panels": panels,
        "equal_panel_mean": means,
        "noninferiority_equal_panel_mean": ni,
        "decision": decision,
    }, {"recipe": mean_draws, "noninferiority": ni_draws, "panel": raw}


def _load_anytime(seed: int, panel: int, recipe_name: str, cell: str) -> dict:
    path = (
        EVALUATION_RUNS
        / "anytime"
        / str(seed)
        / str(panel)
        / recipe_name
        / cell
        / "raw.npz"
    )
    with np.load(verify_reference(reference(path)), allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def _masked_episode_mean(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values.mean(axis=1)
    numerator = np.where(mask[None, :, :], values, 0.0).sum(axis=1)
    denominator = mask.sum(axis=0)
    if np.any(denominator == 0):
        raise RuntimeError("anytime learned/nonlearned episode mask is empty")
    return numerator / denominator[None, :]


def _curve(point: np.ndarray, draws: np.ndarray) -> list[dict]:
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=1)
    return [
        {
            "B": index + 1,
            "point": float(point[index]),
            "interval": {
                "lower": float(lower[index]),
                "upper": float(upper[index]),
            },
        }
        for index in range(len(point))
    ]


def _episode_metrics(raw: dict, learned: np.ndarray) -> dict:
    return {
        "probability": _masked_episode_mean(raw["correct_probability"], None),
        "CE": _masked_episode_mean(raw["query_ce"], None),
        "learned_probability": _masked_episode_mean(
            raw["correct_probability"], learned
        ),
        "nonlearned_probability": _masked_episode_mean(
            raw["correct_probability"], ~learned
        ),
    }


def _anytime_panel(seed: int, panel: int, cell: str, cell_index: int) -> dict:
    fixed = _load_anytime(seed, panel, "fixed_horizon", cell)
    variable = _load_anytime(seed, panel, "variable_horizon", cell)
    for key in ("targets", "learned", "edge_count"):
        if not np.array_equal(fixed[key], variable[key]):
            raise RuntimeError(f"paired anytime {key} differs")
    learned = np.asarray(variable["learned"], dtype=bool)
    episode = {
        "fixed_horizon": _episode_metrics(fixed, learned),
        "variable_horizon": _episode_metrics(variable, learned),
    }
    rng = np.random.default_rng(880000000 + seed * 10000 + panel * 100 + cell_index)
    count = variable["targets"].shape[1]
    indices = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    result: dict = {
        recipe_name: {
            metric: (values.mean(axis=1), values[:, indices].mean(axis=2))
            for metric, values in episode[recipe_name].items()
        }
        for recipe_name in ("fixed_horizon", "variable_horizon")
    }
    for metric, source_metric in (
        ("delta_CE", "CE"),
        ("delta_probability", "probability"),
    ):
        difference = (
            episode["variable_horizon"][source_metric]
            - episode["fixed_horizon"][source_metric]
        )
        result[metric] = (
            difference.mean(axis=1),
            difference[:, indices].mean(axis=2),
        )
    return result


def _aggregate_anytime_panels(panel_summaries: list[dict]) -> tuple[dict, dict]:
    summary, raw_output = {}, {}
    metrics = ("probability", "CE", "learned_probability", "nonlearned_probability")
    for recipe_name in ("fixed_horizon", "variable_horizon"):
        summary[recipe_name], raw_output[recipe_name] = {}, {}
        for metric in metrics:
            point = np.mean(
                [row[recipe_name][metric][0] for row in panel_summaries], axis=0
            )
            draws = np.mean(
                [row[recipe_name][metric][1] for row in panel_summaries], axis=0
            )
            summary[recipe_name][metric] = _curve(point, draws)
            raw_output[recipe_name][metric] = draws
    for metric in ("delta_CE", "delta_probability"):
        point = np.mean([row[metric][0] for row in panel_summaries], axis=0)
        draws = np.mean([row[metric][1] for row in panel_summaries], axis=0)
        summary[metric] = _curve(point, draws)
        raw_output[metric] = draws
    return summary, raw_output


def _anytime_decision(cells: dict) -> dict:
    absolute_by_cell = {
        cell: all(
            row["interval"]["lower"] > 0.5
            for metric in ("learned_probability", "nonlearned_probability")
            for row in cells[cell]["variable_horizon"][metric]
        )
        for cell in ("A0", "Ce")
    }
    heldout = {}
    for horizon, index in (("B1", 0), ("B7", 6)):
        heldout[horizon] = {}
        for cell in ("A0", "Ce"):
            ce = cells[cell]["delta_CE"][index]
            probability = cells[cell]["delta_probability"][index]
            heldout[horizon][cell] = {
                "CE_superiority": ce["interval"]["upper"] < 0.0,
                "probability_nonharm": probability["interval"]["lower"] >= -0.02,
                "passed": interval_gate(
                    ce_upper=ce["interval"]["upper"],
                    probability_lower=probability["interval"]["lower"],
                ),
            }
    decision = {
        "absolute_by_primary_cell": absolute_by_cell,
        "anytime_competent": all(absolute_by_cell.values()),
        "heldout": heldout,
    }
    return decision


def _anytime_seed(seed: int):
    cells, raw_output = {}, {}
    for cell_index, cell in enumerate(CELLS):
        panel_summaries = [
            _anytime_panel(seed, panel, cell, cell_index)
            for panel in specification()["design"]["evaluation_panels"]
        ]
        cells[cell], raw_output[cell] = _aggregate_anytime_panels(panel_summaries)
    return {"cells": cells, "decision": _anytime_decision(cells)}, raw_output


def render_report(result: dict) -> str:
    lines = [
        "# Single-P anytime V1 result",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        "| seed | fresh FH valid | VH B4 preserved | B1 A0/Ce | B7 A0/Ce | anytime competent |",
        "|---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for seed in specification()["design"]["network_seeds"]:
        historical = result["historical"][str(seed)]["decision"]
        anytime = result["anytime"][str(seed)]["decision"]
        lines.append(
            "| {seed} | {fh} | {vh} | {b1} | {b7} | {competent} |".format(
                seed=seed,
                fh="PASS" if historical["fresh_fixed_valid"] else "FAIL",
                vh="PASS" if historical["historical_preserved"] else "FAIL",
                b1="/".join(
                    "PASS" if anytime["heldout"]["B1"][cell]["passed"] else "FAIL"
                    for cell in ("A0", "Ce")
                ),
                b7="/".join(
                    "PASS" if anytime["heldout"]["B7"][cell]["passed"] else "FAIL"
                    for cell in ("A0", "Ce")
                ),
                competent="PASS" if anytime["anytime_competent"] else "FAIL",
            )
        )
    lines += [
        "",
        "The registered contrast changes the presentation-horizon training schedule under the inherited terminal-P regularizer. It does not add new relation information, an explicit time/count input, or an uncertainty signal.",
        "",
        "This is a three-seed development result. Only a 3/3 `anytime_admitted` result permits a separately registered fresh confirmation; no result here alone promotes a new main model.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    source, _ = validate_model_lock()
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="single_p_anytime_v1",
        execution_id="registered-comparison",
        producer={"model_lock": reference(MODEL_LOCK)},
        resolved_config=specification()["decision"],
    ):
        historical, anytime, raw = {}, {}, {"historical": {}, "anytime": {}}
        for seed in specification()["design"]["network_seeds"]:
            historical[str(seed)], raw["historical"][str(seed)] = _historical_pair(
                seed, source
            )
            anytime[str(seed)], raw["anytime"][str(seed)] = _anytime_seed(seed)
        fresh_fixed_valid = all(
            row["decision"]["fresh_fixed_valid"] for row in historical.values()
        )
        historical_preserved = all(
            row["decision"]["historical_preserved"] for row in historical.values()
        )
        anytime_competent = all(
            row["decision"]["anytime_competent"] for row in anytime.values()
        )
        horizon_pass = {
            horizon: all(
                row["decision"]["heldout"][horizon][cell]["passed"]
                for row in anytime.values()
                for cell in ("A0", "Ce")
            )
            for horizon in ("B1", "B7")
        }
        arm_heterogeneity = any(
            row["decision"]["absolute_by_primary_cell"]["A0"]
            != row["decision"]["absolute_by_primary_cell"]["Ce"]
            or any(
                row["decision"]["heldout"][horizon]["A0"]["passed"]
                != row["decision"]["heldout"][horizon]["Ce"]["passed"]
                for horizon in ("B1", "B7")
            )
            for row in anytime.values()
        )
        outcome = classify(
            integrity=True,
            fresh_fixed_valid=fresh_fixed_valid,
            historical_preserved=historical_preserved,
            anytime_competent=anytime_competent,
            short_improved=horizon_pass["B1"],
            long_improved=horizon_pass["B7"],
        )
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "model_lock": reference(MODEL_LOCK),
            "historical": historical,
            "anytime": anytime,
            "decision": {
                "integrity": True,
                "fresh_fixed_valid": fresh_fixed_valid,
                "historical_preserved": historical_preserved,
                "anytime_competent": anytime_competent,
                "heldout_horizon_improvement": horizon_pass,
                "arm_heterogeneity": arm_heterogeneity,
            },
            "outcome": outcome,
            "admitted": outcome == "anytime_admitted",
            "trained_models": 12,
            "historical_evaluation_units": 72,
            "anytime_evaluation_units": 72,
            "claim_boundary": specification()["claim_boundary"],
        }
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(directory / "result.json", json_ready(result))
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(
        RESULT, json_ready({**result, "raw": reference(directory / "raw.npz")})
    )
    REPORT.write_text(render_report(result), encoding="utf-8")
    return {
        "outcome": outcome,
        "admitted": result["admitted"],
        "arm_heterogeneity": arm_heterogeneity,
    }


__all__ = ["render_report", "write_report"]
