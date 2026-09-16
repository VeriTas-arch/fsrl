"""Registered clean single-P comparison and concise report."""

from __future__ import annotations

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
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import CELLS, legacy_input_record
from .locks import reference, validate_model_lock
from .protocol import (
    EVALUATION_RUNS,
    MODEL_LOCK,
    PROTOCOL,
    REPORT,
    RESULT,
    RUNS,
    inherited_recipe,
    specification,
)


def _cell_data(seed, panel, condition):
    rows, raw = {}, {}
    for cell in CELLS:
        directory = EVALUATION_RUNS / str(seed) / str(panel) / condition / cell
        rows[cell] = completed(directory)
        raw[cell] = read_raw(reference(directory / "raw.npz"))
    return rows, raw


def _panel(seed, panel, condition, source):
    rows, raw = _cell_data(seed, panel, condition)
    analysis_spec = analysis_specification()
    sample_seed = seed + panel * 1000000
    result, endpoints = summarize(
        raw,
        rows,
        load_input(
            legacy_input_record(source["panels"][str(panel)]["inputs"]["liu-8"])
        ),
        sample_seed,
        inherited_recipe(panel),
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


def _decision(panels, candidate_mean, ni):
    control_valid = all(
        panel["time_retained_control"]["decision"]["all_four_competent"]
        and all(
            panel["time_retained_control"]["evidence_binding"][cell]
            for cell in ("Ae", "Ce")
        )
        for panel in panels.values()
    )
    candidate_competent = all(
        panel["clean_no_time"]["decision"]["all_four_competent"]
        and all(
            panel["clean_no_time"]["evidence_binding"][cell] for cell in ("Ae", "Ce")
        )
        for panel in panels.values()
    )
    candidate_core = all(
        any(
            panel["clean_no_time"]["cells"][cell]["liu"]["routes"]["full"][
                "core_passed"
            ]
            for panel in panels.values()
        )
        for cell in ("Ae", "Ce")
    )
    ce_noninferior = all(
        row["noninferior"] for row in noninferiority(ni["Ce"]).values()
    )
    compensation = {
        "positive_I_tau": candidate_mean["interaction/global_all77_tau"]["interval"][
            "lower"
        ]
        > 0,
        "negative_I_CE": candidate_mean["interaction/generic_global"]["interval"][
            "upper"
        ]
        < 0,
        "negative_shift_difference": candidate_mean["order_shift/C_minus_A"][
            "interval"
        ]["upper"]
        < 0,
        "both_error_tau_points_negative": all(
            candidate_mean[f"{name}/global_all77_tau"]["point"] < 0
            for name in ("error_A", "error_C")
        ),
    }
    if not control_valid:
        outcome = "training_recipe_failure"
    elif not candidate_competent or not ce_noninferior:
        outcome = "time_removal_failure"
    elif not candidate_core or not all(compensation.values()):
        outcome = "phenotype_incomplete"
    else:
        outcome = "clean_single_p_admitted"
    return {
        "control_valid": control_valid,
        "candidate_competent_and_bound": candidate_competent,
        "Ce_five_endpoints_noninferior": ce_noninferior,
        "Ae_and_Ce_core_present": candidate_core,
        "compensation": compensation,
        "outcome": outcome,
        "admitted": outcome == "clean_single_p_admitted",
    }


def _pair(seed, source):
    panels, condition_draws, raw = (
        {},
        {name: [] for name in ("time_retained_control", "clean_no_time")},
        {},
    )
    ni_panel = []
    analysis_spec = analysis_specification()
    for panel in specification()["design"]["evaluation_panels"]:
        panels[str(panel)] = {}
        raw[str(panel)] = {}
        condition_raw = {}
        for condition in specification()["design"]["architecture_conditions"]:
            result, cell_raw, draws = _panel(seed, panel, condition, source)
            panels[str(panel)][condition] = result
            condition_raw[condition] = cell_raw
            condition_draws[condition].append(draws)
            raw[str(panel)][condition] = draws[1]
        boot = analysis_spec["statistics"]["seed_offset"] + seed + panel * 1000000
        cell_values = {}
        for cell in CELLS:
            values, _ = paired_probability(
                probabilities(condition_raw["clean_no_time"][cell]),
                probabilities(condition_raw["time_retained_control"][cell]),
                boot,
            )
            cell_values[cell] = values
        ni_panel.append(cell_values)
    means, mean_draws = {}, {}
    for condition, draws in condition_draws.items():
        means[condition], mean_draws[condition] = panel_mean(draws)
    ni, ni_draws = {}, {}
    for cell in CELLS:
        ni[cell], ni_draws[cell] = panel_mean(
            [panel_values[cell] for panel_values in ni_panel]
        )
    decision = _decision(panels, means["clean_no_time"], ni)
    return {
        "panels": panels,
        "equal_panel_mean": means,
        "noninferiority_equal_panel_mean": ni,
        "decision": decision,
    }, {
        "condition": mean_draws,
        "noninferiority": ni_draws,
        "panel": raw,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Clean single-P direct-training result",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        "| seed | control valid | candidate competent/bound | Ce noninferior | Ae/Ce core | compensation | outcome |",
        "|---:|:---:|:---:|:---:|:---:|:---:|---|",
    ]
    for seed, row in result["pairs"].items():
        decision = row["decision"]
        lines.append(
            "| {seed} | {control} | {competent} | {ni} | {core} | {comp} | `{outcome}` |".format(
                seed=seed,
                control="PASS" if decision["control_valid"] else "FAIL",
                competent="PASS"
                if decision["candidate_competent_and_bound"]
                else "FAIL",
                ni="PASS" if decision["Ce_five_endpoints_noninferior"] else "FAIL",
                core="PASS" if decision["Ae_and_Ce_core_present"] else "FAIL",
                comp="PASS" if all(decision["compensation"].values()) else "FAIL",
                outcome=decision["outcome"],
            )
        )
    lines += [
        "",
        "The candidate has one 32-channel task input, one affine modulation scalar, one binary margin, one 40,000-scalar P state, no local store, no normalized-time parameter, no value head, and no initial blank rollout. Both conditions retained H=200, four support microsteps, two query microsteps, 1,500 updates and the registered P penalty.",
        "",
        "This development result does not establish universal minimality or biological implementation. It does not authorize width/timestep compression, post-result repair, main-model promotion, or checkpoint deletion.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    source, _ = validate_model_lock()
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="clean_single_p_v1",
        execution_id="registered-comparison",
        producer={"model_lock": reference(MODEL_LOCK)},
        resolved_config=specification()["decision"],
    ):
        pairs, raw = {}, {}
        for seed in specification()["design"]["network_seeds"]:
            pairs[str(seed)], raw[str(seed)] = _pair(seed, source)
        outcomes = [row["decision"]["outcome"] for row in pairs.values()]
        if all(value == "clean_single_p_admitted" for value in outcomes):
            outcome = "clean_single_p_admitted"
        elif "training_recipe_failure" in outcomes:
            outcome = "training_recipe_failure"
        elif "time_removal_failure" in outcomes:
            outcome = "time_removal_failure"
        else:
            outcome = "phenotype_incomplete"
        result = {
            "schema_version": 1,
            "protocol": reference(PROTOCOL),
            "model_lock": reference(MODEL_LOCK),
            "pairs": pairs,
            "outcome": outcome,
            "admitted": outcome == "clean_single_p_admitted",
            "trained_models": 12,
            "evaluation_units": 72,
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
    return {"outcome": outcome, "admitted": result["admitted"]}


__all__ = ["render_report", "write_report"]
