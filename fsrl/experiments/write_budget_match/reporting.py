"""Paired budget equivalence and prediction recovery; retain failed controls."""

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.admission_hint_removal.reporting import decision
from fsrl.experiments.local_memory_removal.statistics import (
    paired_probability,
    probabilities,
)
from fsrl.experiments.modulation_schedule.protocol import analysis_seed
from fsrl.experiments.observation_replication.reporting import read_raw, summarize
from fsrl.experiments.observation_replication.statistics import panel_draws, panel_mean
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.experiments.write_cost.reporting import copy_artifact
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate
from .protocol import CALIBRATION, PROTOCOL, RECORDS, RUNS, recipe, specification


def paired_effect(a, b, seed):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    counts = bootstrap_counts(np.random.default_rng(seed), 2000, len(a))
    return float((a - b).mean()), counts @ (a - b) / len(a)


def budget_effect(new, old, weights, seed):
    assert new.shape == old.shape and new.shape[1] == 4
    counts = bootstrap_counts(np.random.default_rng(seed), 2000, len(new))
    a, b = new * weights[:, None], old * weights[:, None]
    assert np.all(b[:, 2:].sum(0) > 0) and np.all(counts @ b[:, 2:] > 0)
    return a[:, 2:].sum(0) / b[:, 2:].sum(0), (counts @ a[:, 2:]) / (counts @ b[:, 2:])


def budgets_for(seed, training, panel, arm):
    root = RUNS / "evaluation_budgets" / str(seed) / training / str(panel) / arm
    tasks, weights, clipping = {}, {}, {}
    for task, pattern in (("generic", "test-*.npz"), ("liu", "liu-8.npz")):
        modes = {key: [] for key in ("B", "S", "M")}
        weight, clips = [], {key: [] for key in modes}
        for path in sorted(root.glob(pattern)):
            with np.load(path, allow_pickle=False) as raw:
                for mode, rows in modes.items():
                    w = raw[mode + "__writes"].astype(float)
                    assert np.count_nonzero(w[:, :2]) == 0
                    rows.append(w.sum(0).T)
                    clips[mode].append(raw[mode + "__clipping"].mean((0, 2)))
                    np.testing.assert_array_equal(
                        raw[mode + "__episode_indices"], raw["B__episode_indices"]
                    )
                weight.append(np.full(w.shape[2], 1 / w.shape[2]))
        tasks[task] = {mode: np.concatenate(rows) for mode, rows in modes.items()}
        weights[task] = np.concatenate(weight)
        clipping[task] = {
            mode: np.mean(rows, axis=0).tolist() for mode, rows in clips.items()
        }
    return tasks, weights, clipping


def panel_result(seed, number, prior):
    spec, sample = prior["protocol"], analysis_seed(seed, number)
    boot = spec["statistics"]["seed_offset"] + sample
    old = prior["schedule_result"]["phase"]["pairs"][str(seed)]["panels"][str(number)]
    rows, raw, points, draws, ni, ni_raw, budgets, budget_draws, clips = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for cell, settings in spec["design"]["cells"].items():
        directory = RUNS / "evaluation" / str(seed) / str(number) / cell
        rows[cell] = completed(directory)
        m = read_raw(reference(directory / "raw.npz"))
        b = read_raw(
            prior["result"]["artifacts"][f"evaluation/{seed}/{number}/{cell}/raw.npz"]
        )
        s = read_raw(
            prior["schedule_result"]["artifacts"][
                f"phase/{seed}/{number}/{cell}/raw.npz"
            ]
        )
        for other in (b, s):
            for key in ("episode_indices", "signs", "learned"):
                np.testing.assert_array_equal(m["generic"][key], other["generic"][key])
        raw[cell] = m
        ce = {
            key: np.logaddexp(
                0, -v["generic"]["global_margins"].astype(float) * v["generic"]["signs"]
            ).mean(1)
            for key, v in (("M", m), ("B", b), ("S", s))
        }
        for name, first, second in (("residual", "M", "B"), ("recovery", "S", "M")):
            key = cell + "/" + name
            points[key], draws[key] = paired_effect(ce[first], ce[second], boot)
        probability, included = paired_probability(
            probabilities(m), probabilities(b), boot
        )
        ni[cell], _ = panel_mean([probability])
        ni_raw[cell] = probability
        data, weights, clips[cell] = budgets_for(
            seed, settings["training"], number, settings["observation"]
        )
        for task, modes in data.items():
            for mode, behavioral in (("M", m), ("B", b), ("S", s)):
                np.testing.assert_allclose(
                    modes[mode].sum(1),
                    behavioral[task]["total_write"],
                    rtol=1e-4,
                    atol=1e-5,
                )
            for mode in ("S", "M"):
                p, d = budget_effect(modes[mode], modes["B"], weights[task], boot)
                for phase in (3, 4):
                    key = f"{cell}/{task}/{mode}/phase{phase}"
                    budgets[key], budget_draws[key] = p[phase - 3], d[:, phase - 3]
        rows[cell] = {**rows[cell], "paired_probability_subjects": included}
    cpu = load_input(prior["source"]["panels"][str(number)]["inputs"]["liu-8"])
    candidate, endpoints = summarize(raw, rows, cpu, sample, recipe(number), spec)
    comp = panel_draws(raw, endpoints["CE"], endpoints["order_shift"], boot, spec)
    return {
        "candidate": candidate,
        "baseline": old["baseline"],
        "stage": old["candidate"],
        "noninferiority": ni,
        "clipping": clips,
    }, (comp, (points, draws), ni_raw, (budgets, budget_draws))


def summarize_pair(seed, prior):
    panels, values = {}, []
    for number in specification()["evaluation_panels"]:
        panels[str(number)], data = panel_result(seed, number, prior)
        values.append(data)
    comp, comp_raw = panel_mean([v[0] for v in values])
    primary, primary_raw = panel_mean([v[1] for v in values])
    budgets, budget_raw = panel_mean([v[3] for v in values])
    ni, ni_raw = {}, {}
    for cell in prior["protocol"]["design"]["cells"]:
        ni[cell], ni_raw[cell] = panel_mean([v[2][cell] for v in values])
    delta = specification()["solver"]["budget_tolerance"]
    matching = {}
    for task in ("generic", "liu"):
        matching[task] = {
            cell: all(
                budgets[f"{cell}/{task}/M/phase{k}"]["interval"]["lower"] >= 1 - delta
                and budgets[f"{cell}/{task}/M/phase{k}"]["interval"]["upper"]
                <= 1 + delta
                for k in (3, 4)
            )
            for cell in ("A0", "Ae", "C0", "Ce")
        }
    return {
        "panels": panels,
        "candidate_equal_panel_mean": comp,
        "prediction": primary,
        "budget_ratios": budgets,
        "budget_equivalence": matching,
        "noninferiority_equal_panel_mean": ni,
        "preservation": decision(panels, comp, ni),
        "primary_budget_identified": matching["generic"]["Ce"],
    }, {
        "compensation": comp_raw,
        "prediction": primary_raw,
        "budget_ratios": budget_raw,
        "noninferiority": ni_raw,
    }


def report():
    prior = validate(gpu=False, calibrated=True)
    calibration = load_json(CALIBRATION)
    directory = RUNS / "comparison"
    with ProspectiveRun.start(
        directory,
        workflow_id="write_budget_match_v1",
        execution_id="budget-and-prediction-comparison",
        producer={"calibration": reference(CALIBRATION)},
        resolved_config=specification()["decision"],
    ):
        pairs, raw = {}, {}
        if calibration["all_six_matched"]:
            completed(RUNS / "evaluation_budgets")
            for seed in specification()["seeds"]:
                pairs[str(seed)], raw[str(seed)] = summarize_pair(seed, prior)
            write_arrays(directory / "raw.npz", flatten_arrays(raw))
        result = {
            "protocol": reference(PROTOCOL),
            "calibration": reference(CALIBRATION),
            "evaluation_executed": calibration["all_six_matched"],
            "evaluation_units": 24 if calibration["all_six_matched"] else 0,
            "pairs": pairs,
            "stop_reason": None
            if calibration["all_six_matched"]
            else "Budget-only calibration did not match all four targets in all six models; no behavior was evaluated.",
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    archived = {
        str(p.relative_to(RUNS)): copy_artifact(
            p, RECORDS / "artifacts" / p.relative_to(RUNS)
        )
        for p in sorted(RUNS.rglob("*"))
        if p.is_file() and p.suffix in (".json", ".npz")
    }
    write_json_exclusive(
        RECORDS / "results/result.json", json_ready({**result, "artifacts": archived})
    )
    return {"evaluation_units": result["evaluation_units"], "archived": len(archived)}
