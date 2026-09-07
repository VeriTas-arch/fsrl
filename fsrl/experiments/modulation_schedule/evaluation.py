"""Matched support-only interventions on the six immutable affine models."""

import gc

import torch

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.duplicate_observation.rollouts import generic_arrays
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.linear_modulation.model import load_model
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate
from .model import scheduled_sequences
from .protocol import CALIBRATION, RUNS, analysis_seed, recipe, specification


def collect(seed, arm, panel, model, values):
    spec = recipe(panel["id"])
    bootstrap_seed = analysis_seed(seed, panel["id"])
    net, _, original = load_model(seed, model, spec)
    before = tensor_hashes(net)
    seqs = original if values is None else scheduled_sequences(net, original, values)
    with torch.no_grad():
        raw_generic = generic_arrays(net, None, seqs, panel, "test", arm, spec)
        generic = summarize_generic(raw_generic, spec, bootstrap_seed)
        global_raw = {**raw_generic, "margins": raw_generic["global_margins"]}
        generic["global"] = summarize_generic(global_raw, spec, bootstrap_seed)
        raw_generic["global_ce"] = global_raw["ce"]
        cpu = observed(load_input(panel["inputs"]["liu-8"]), arm, spec)
        task = size_protocol(spec, 8)
        raw = primary_rollouts(net, None, seqs, None, cpu, task, spec)
        liu, analysis, sampled = primary_analysis(
            raw, cpu, task, spec, bootstrap_seed, generic
        )
    assert before == tensor_hashes(net)
    return (
        {"generic": generic, "liu": liu},
        {"generic": raw_generic, "liu": flatten_arrays({**raw, **analysis})},
        sampled,
    )


def evaluate(mode):
    prior = validate(calibrated=True)
    if mode == "constant":
        assert load_json(RUNS / "phase_comparison/result.json")[
            "preserved_in_all_pairs"
        ]
    calibration = load_json(CALIBRATION)["models"]
    spec = specification()
    for seed in spec["seeds"]:
        for number in spec["evaluation_panels"]:
            panel = {**prior["source"]["panels"][str(number)], "id": number}
            for cell, settings in prior["protocol"]["design"]["cells"].items():
                key = f"{seed}/{settings['training']}"
                directory = RUNS / mode / str(seed) / str(number) / cell
                with ProspectiveRun.start(
                    directory,
                    workflow_id="modulation_schedule_v1",
                    execution_id=f"{mode}-{seed}-{number}-{cell}",
                    producer={"calibration": reference(CALIBRATION)},
                    resolved_config={"mode": mode, "values": calibration[key][mode]},
                ):
                    result, raw, sampled = collect(
                        seed,
                        settings["observation"],
                        panel,
                        prior["models"]["runs"][key]["files"],
                        calibration[key][mode],
                    )
                    write_arrays(directory / "raw.npz", flatten_arrays(raw))
                    write_json_exclusive(
                        directory / "behavior.json", json_ready(sampled)
                    )
                    write_json_exclusive(directory / "result.json", json_ready(result))
                print(
                    {
                        "mode": mode,
                        "seed": seed,
                        "panel": number,
                        "cell": cell,
                        "core": result["liu"]["routes"]["full"]["core_passed"],
                    },
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
    return {"mode": mode, "completed_evaluation_units": 24}
