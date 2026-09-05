"""All-artifact-locked evaluation; no training, tuning or checkpoint selection."""

import torch

from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .analysis import analyze_batch, conditional_codes, query_signs
from .assessment import assess_groups
from .evidence import ARTIFACT_LOCK, validate_artifacts
from .inputs import load_group
from .protocol import (
    PROTOCOL_HASH,
    RUN_ROOT,
    make_model,
    resolved_specification,
    run_directory,
    specification,
)


def load_model(config, spec, device="cuda"):
    model = make_model(spec, device)
    model.load_state_dict(
        {
            key: torch.tensor(value, device=device, dtype=torch.float32)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("archived final parameters changed")
    return model.requires_grad_(False).eval()


def evaluation_directory(seed, condition):
    training = run_directory(seed, condition)
    return RUN_ROOT / "evaluation" / training.parent.name / training.name


def validate_evaluation(seed, condition, lock) -> dict:
    directory = evaluation_directory(seed, condition)
    validate_complete(directory)
    result = load_json(directory / "result.json")
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_HASH,
        "source_commit": lock["source_commit"],
        "artifact_lock": reference(ARTIFACT_LOCK),
        "cohorts": lock["cohorts"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("evaluation does not belong to these locked fits/inputs")
    return result


def evaluate_fit(seed, condition, lock, execution, spec) -> None:
    directory = evaluation_directory(seed, condition)
    if directory.exists():
        validate_evaluation(seed, condition, lock)
        return
    config = lock["archives"][f"{seed}/{condition}"]["config"]
    model = load_model(config, spec)
    fixed_config = lock["archives"][f"{seed}/exact"]["config"]
    fixed = load_model(fixed_config, spec)
    runner, fixed_runner = compiled(model), compiled(fixed)
    candidate = specification()
    codebook = candidate["encoding"]["codebook"]
    protocol = load_registered_protocol(spec["evaluation"]["liu"]["protocol_id"])
    files = {}
    with ProspectiveRun.start(
        directory,
        workflow_id=spec["experiment_id"],
        execution_id=f"evaluate-{seed}-{condition}",
        producer={"module": __name__, "artifact_lock": reference(ARTIFACT_LOCK)},
        resolved_config={"evaluation": spec["evaluation"], "runtime": execution},
    ):
        for domain, groups in lock["cohorts"].items():
            for name, input_record in groups.items():
                batch, auxiliary = load_group(input_record)
                temperature = spec["evaluation"][domain]["temperature"]
                analysis = analyze_batch(
                    model,
                    runner,
                    fixed,
                    fixed_runner,
                    batch,
                    auxiliary,
                    condition,
                    codebook,
                    temperature,
                    protocol if domain == "liu" else None,
                )
                path = directory / f"{domain}-{name}.npz"
                write_arrays(path, flatten_arrays(analysis))
                files[f"{domain}-{name}"] = reference(path)
                if domain == "liu" and condition == "persistent":
                    mixture = conditional_codes(
                        batch,
                        codebook,
                        model,
                        temperature,
                        query_signs(batch, protocol),
                    )
                    path = directory / "conditional-codes.npz"
                    write_arrays(path, mixture)
                    files["conditional_codes"] = reference(path)
        result, sampled = assess_groups(files, lock["cohorts"], seed, spec)
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        result.update(
            {
                "seed": seed,
                "condition": condition,
                "protocol_sha256": PROTOCOL_HASH,
                "source_commit": lock["source_commit"],
                "artifact_lock": reference(ARTIFACT_LOCK),
                "cohorts": lock["cohorts"],
                "files": files,
                "sampled_behavior": reference(directory / "behavior.json"),
                "parameters": config["physical_parameters"],
                "raw_parameters": config["raw_parameters"],
                "fixed_parameters": fixed_config["physical_parameters"],
                "runtime": execution,
                "cost": config["cost"],
            }
        )
        write_json_exclusive(directory / "result.json", json_ready(result))


def evaluate_all() -> dict:
    lock = validate_artifacts()  # All nine fits, public lock, before ANY evaluation.
    execution = runtime()
    spec = resolved_specification()
    completed = []
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["conditions"]:
            evaluate_fit(seed, condition, lock, execution, spec)
            validate_evaluation(seed, condition, lock)
            completed.append(f"{seed}/{condition}")
            print("Evaluation complete", completed[-1], flush=True)
    return {"complete_evaluations": completed}
