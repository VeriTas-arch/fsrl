"""Complete four-cell evaluation after all six final weights are locked."""

import gc

import torch

from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.observation_uncertainty.evaluation import observed
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate_models
from .protocol import MODELS, RUNS, analysis_seed, recipe, specification
from .rollouts import generic_arrays, primary_rollouts


def load_model(seed, model, spec, condition="single"):
    backbone, local = make_model(spec, seed, condition, "cuda")
    metadata = load_json(verify_reference(model["result.json"]))
    for name, module, key in (
        ("net", backbone, "final_backbone"),
        ("local", local, "final_local"),
    ):
        if module is None:
            continue
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


def collect(seed, observation, inputs, model, spec, bootstrap_seed, condition="single"):
    backbone, local, seqs = load_model(seed, model, spec, condition)
    before = tensor_hashes(backbone), (None if local is None else tensor_hashes(local))
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
    if before != (
        tensor_hashes(backbone),
        (None if local is None else tensor_hashes(local)),
    ):
        raise RuntimeError("evaluation changed slow parameters")
    return (
        {"generic": generic, "liu": liu},
        {"generic": generic_raw, "liu": flatten_arrays({**raw, **analysis})},
        sampled,
    )


def qualify():
    from .qualification import qualify as run_checks

    return run_checks()


def evaluate_one(seed, panel, cell, settings, source, models):
    directory = RUNS / "evaluation" / str(seed) / str(panel) / cell
    if directory.exists():
        return completed(directory)
    spec = recipe(panel)
    model = models["runs"][f"{seed}/{settings['training']}"]["files"]
    with ProspectiveRun.start(
        directory,
        workflow_id="local_memory_removal_v1",
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
