"""Complete four-cell evaluation after all six final weights are locked."""

import gc

import numpy as np
import torch

from fsrl.experiments.finite_state.liu import primary_analysis, primary_rollouts
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.observation_crossover.evaluation import arrays
from fsrl.experiments.observation_uncertainty.evaluation import generic_arrays, observed
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import STUDIES_ROOT

from .execution import sources, validate_models
from .protocol import MODELS, RECORDS, RUNS, analysis_seed, recipe, specification


def load_model(seed, model, spec):
    backbone, local = make_model(spec, seed, "dual", "cuda")
    assert local is not None
    metadata = load_json(verify_reference(model["result.json"]))
    for name, module, key in (
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ):
        module.load_state_dict(
            torch.load(
                verify_reference(model[name + ".pth"]),
                weights_only=True,
                map_location="cuda",
            )
        )
        if tensor_hashes(module) != metadata[key]:
            raise RuntimeError("loaded parameter mismatch")
        module.requires_grad_(False).eval()
    return backbone, local, sequences(backbone, None, compiled=True)


def collect(seed, observation, inputs, model, spec, bootstrap_seed):
    backbone, local, seqs = load_model(seed, model, spec)
    before = tensor_hashes(backbone), tensor_hashes(local)
    with torch.no_grad():
        # Preserve original generic-before-Liu compiled shape specialization.
        generic_raw = generic_arrays(
            backbone, local, seqs, inputs, "test", observation, spec
        )
        generic = summarize_generic(generic_raw, spec, bootstrap_seed)
        global_raw = {**generic_raw, "margins": generic_raw["global_margins"]}
        generic["global"] = summarize_generic(global_raw, spec, bootstrap_seed)
        generic_raw["global_ce"] = global_raw["ce"]
        cpu = observed(load_input(inputs["inputs"]["liu-8"]), observation, spec)
        task = size_protocol(spec, 8)
        raw = primary_rollouts(backbone, local, seqs, None, cpu, task, spec)
        liu, analysis, sampled = primary_analysis(
            raw, cpu, task, spec, bootstrap_seed, generic
        )
    if before != (tensor_hashes(backbone), tensor_hashes(local)):
        raise RuntimeError("evaluation changed slow parameters")
    return (
        {"generic": generic, "liu": liu},
        {"generic": generic_raw, "liu": flatten_arrays({**raw, **analysis})},
        sampled,
    )


def qualify():
    old = load_json(
        STUDIES_ROOT / "observation_crossover/records/benchmarks/execution_lock.json"
    )
    runtime = json_ready(configure_execution())
    if runtime != old["runtime"]:
        raise RuntimeError("qualification runtime differs from parent")
    directory = RUNS / "qualification_final"
    with ProspectiveRun.start(
        directory,
        workflow_id="observation_replication_v1",
        execution_id="old-A0-parity",
        producer={"sources": sources()},
        resolved_config={"runtime": runtime},
    ):
        _, raw, _ = collect(
            2432, "clean", old, old["models"]["2432/clean"], recipe(), 2432
        )
        checks = []
        for phase in ("generic", "liu"):
            previous = arrays(old["previous_cells"]["2432"]["A0"][phase])
            for key, value in raw[phase].items():
                if not np.array_equal(value, previous[key], equal_nan=True):
                    raise RuntimeError(f"old replay mismatch: {phase}/{key}")
                checks.append(phase + "/" + key)
        from .reporting import qualify_estimators

        estimator_checks = qualify_estimators()
        result = {
            "passed": True,
            "sources": sources(),
            "runtime": runtime,
            "exact_array_checks": checks,
            "estimator_checks": estimator_checks,
            "new_models_or_outcomes": False,
        }
        write_json_exclusive(directory / "result.json", result)
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", result)
    return {
        "passed": True,
        "exact_arrays": len(checks),
        "estimator_checks": estimator_checks,
    }


def evaluate_one(seed, panel, cell, settings, source, models):
    directory = RUNS / "evaluation" / str(seed) / str(panel) / cell
    if directory.exists():
        return completed(directory)
    spec = recipe(panel)
    model = models["runs"][f"{seed}/{settings['training']}"]["files"]
    with ProspectiveRun.start(
        directory,
        workflow_id="observation_replication_v1",
        execution_id=f"{seed}-{panel}-{cell}",
        producer={
            "model_lock": reference(MODELS),
            "seed": seed,
            "panel": panel,
            "cell": cell,
        },
        resolved_config=spec["evaluation"],
    ):
        result, raw, sampled = collect(
            seed,
            settings["observation"],
            source["panels"][str(panel)],
            model,
            spec,
            analysis_seed(seed, panel),
        )
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    print(
        {
            "seed": seed,
            "panel": panel,
            "cell": cell,
            "core": {k: v["core_passed"] for k, v in result["liu"]["routes"].items()},
        },
        flush=True,
    )
    return result


def evaluate():
    source, models = validate_models()
    spec = specification()
    for seed in spec["design"]["seeds"]:
        for panel in spec["design"]["panels"]:
            for cell, settings in spec["design"]["cells"].items():
                evaluate_one(seed, panel, cell, settings, source, models)
                gc.collect()
                torch.cuda.empty_cache()
    return {"completed_evaluation_units": 36}
