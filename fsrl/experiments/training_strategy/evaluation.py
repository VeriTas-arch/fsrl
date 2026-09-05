"""Post-lock execution of every registered generic and Liu comparison."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.evaluation.contracts import FrozenEvaluationBackend
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol_catalog import load_registered_protocol
from fsrl.training.checkpoints import load_retro_checkpoint

from . import decisions
from .behavior import evaluate_behavior
from .execution import PROFILE, configure_execution
from .generic_validation import evaluate_generic
from .legacy_diagnostics import own_global_qualification, projection_audit
from .liu_rollout import rollout_liu
from .locks import (
    ARTIFACT_LOCK_PATH,
    RUN_ROOT,
    reference,
    run_directory,
    validate_artifact_lock,
)
from .protocol import PROTOCOL_SHA256, load_specification
from .summaries import (
    liu_endpoints,
    mechanism_effects,
    summarize_endpoints,
    summarize_geometry,
)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def flatten_arrays(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    arrays = {}
    for key, value in tree.items():
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            arrays.update(flatten_arrays(value, name))
        elif isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise ValueError(f"non-numeric scientific array: {name}")
            arrays[name] = value
    return arrays


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("saved scientific array inventory differs")
        for key, expected in arrays.items():
            np.testing.assert_array_equal(saved[key], expected)


def evaluation_directory(seed: int, condition: str) -> Path:
    # Reuse the training identity guard without admitting any new seed/condition.
    training = run_directory(seed, condition)
    return RUN_ROOT / "evaluation" / training.parent.name / training.name


def load_condition(seed: int, condition: str, metadata: dict, specification: dict):
    settings = specification["evaluation"]["liu"]
    directory = run_directory(seed, condition)
    backbone, config, _ = load_retro_checkpoint(
        directory / "net.pth", settings["subjects"], device="cuda"
    )
    local = ConjunctiveLocalTrace(config.cs, device="cuda")
    local.load_state_dict(
        torch.load(directory / "local.pth", map_location="cuda", weights_only=True),
        strict=True,
    )
    if (
        tensor_hashes(backbone) != metadata["final_backbone"]
        or tensor_hashes(local) != metadata["final_local"]
    ):
        raise RuntimeError("loaded final tensors differ from the joint artifact lock")
    backbone.requires_grad_(False).eval()
    local.requires_grad_(False).eval()
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        config,
        load_registered_protocol(settings["protocol_id"]),
        cue_seed=settings["cue_seed"],
        support_seed=settings["support_seed"],
        cue_mode=settings["cue_mode"],
        subject_encoding_mode=settings["subject_encoding_mode"],
        subject_encoding_seed=settings["subject_encoding_seed"],
        test_time_value=specification["architecture"]["support_query_time"],
        backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
        execution_profile=PROFILE,
    )
    return evaluator, local


def condition_analysis(
    evaluator, local, metadata: dict, seed: int, specification: dict
) -> tuple[dict, dict]:
    statistics = specification["statistics"]
    generic = evaluate_generic(evaluator.net, local, specification)
    liu = rollout_liu(evaluator, local, specification)
    endpoints = liu_endpoints(
        liu["bundles"],
        liu["retention"],
        evaluator.protocol,
        specification["evaluation"]["liu"]["temperature"],
    )
    geometry = summarize_geometry(
        liu["bundles"], liu["loo"], evaluator.protocol, 85000 + seed, statistics
    )
    effects = mechanism_effects(endpoints, geometry, 85000 + seed, statistics)
    behavior = evaluate_behavior(
        liu["bundles"]["intact"], evaluator.protocol, seed, specification
    )
    raw_endpoints = {"generic": generic["endpoints"], "liu": endpoints}
    summaries = {
        domain: {
            name: summarize_endpoints(
                row, (85000 if domain == "liu" else 86000) + seed, statistics
            )
            for name, row in conditions.items()
        }
        for domain, conditions in raw_endpoints.items()
    }
    summaries["constructive"] = geometry["constructive"]
    legacy = own_global_qualification(
        evaluator, liu["fast_weights"], metadata, specification
    )
    projection = projection_audit(evaluator, liu["bundles"], 85000 + seed, statistics)
    result = {
        "summaries": summaries,
        "effects": effects,
        "raw_endpoints": raw_endpoints,
        "behavior": behavior["record"],
        "legacy_qualification": legacy,
        "posterior_projection": {
            key: value for key, value in projection.items() if key != "raw_subject"
        },
        "cost": metadata["cost"],
        "generic_stream_fingerprints": {
            key: row["fingerprint"] for key, row in generic["groups"].items()
        },
        "decisions": {
            "competence": decisions.competence(summaries, specification),
            "mechanism": decisions.mechanism(effects),
            "behavior": decisions.behavior_preservation(
                behavior["record"], specification
            ),
        },
    }
    raw = {
        "generic": generic,
        "liu": {key: value for key, value in liu.items() if key != "fast_weights"},
        "geometry": geometry,
        "projection": projection["raw_subject"],
    }
    return result, {
        "arrays": flatten_arrays(raw),
        "sampled_behavior": behavior["sampled_behavior"],
    }


def validate_evaluation(seed: int, condition: str, artifact_lock: dict) -> dict:
    directory = evaluation_directory(seed, condition)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("evaluation run is incomplete or modified")
    result = load_json(directory / "result.json")
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("evaluation provenance differs from the joint artifact lock")
    return result


def evaluate_one(
    seed: int, condition: str, artifact_lock: dict, specification: dict, runtime: dict
) -> dict:
    directory = evaluation_directory(seed, condition)
    if directory.exists():
        return validate_evaluation(seed, condition, artifact_lock)
    metadata = artifact_lock["runs"][f"{seed}/{condition}"]["metadata"]
    identity = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="joint_training_strategy_v1",
        execution_id=f"evaluation-{seed}-{condition}",
        producer={"module": __name__, **identity},
        resolved_config={
            "evaluation": specification["evaluation"],
            "statistics": specification["statistics"],
            "runtime": runtime,
        },
    ):
        evaluator, local = load_condition(seed, condition, metadata, specification)
        result, raw = condition_analysis(
            evaluator, local, metadata, seed, specification
        )
        write_arrays(directory / "raw.npz", raw["arrays"])
        write_json_exclusive(
            directory / "behavior.json", json_ready(raw["sampled_behavior"])
        )
        result.update(
            {
                **identity,
                "runtime": runtime,
                "raw_arrays": reference(directory / "raw.npz"),
                "sampled_behavior": reference(directory / "behavior.json"),
            }
        )
        write_json_exclusive(directory / "result.json", json_ready(result))
    return validate_evaluation(seed, condition, artifact_lock)


def evaluate_all() -> dict:
    artifact_lock = validate_artifact_lock()
    specification = load_specification()
    runtime = configure_execution()
    completed = []
    for seed in specification["seeds"]["mandatory"]:
        for condition in specification["seeds"]["conditions"]:
            print(f"Evaluating {seed}/{condition}", flush=True)
            result = evaluate_one(
                seed, condition, artifact_lock, specification, runtime
            )
            completed.append(f"{seed}/{condition}")
            print(
                {"completed": completed[-1], "decisions": result["decisions"]},
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    return {"completed": completed, "artifact_lock": reference(ARTIFACT_LOCK_PATH)}
