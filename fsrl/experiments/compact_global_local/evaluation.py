"""Post-lock generic and Liu evaluation of one compact-model cohort."""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fsrl.experiments.training_strategy import decisions
from fsrl.experiments.training_strategy.behavior import evaluate_behavior
from fsrl.experiments.training_strategy.estimands import query_endpoints
from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    mechanism_effects,
    summarize_endpoints,
    summarize_geometry,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .batches import prepare_batch, sample_episodes
from .liu import CompactLiuEvaluator, rollout_liu
from .locks import (
    RUN_ROOT,
    artifact_lock_path,
    reference,
    run_directory,
    validate_artifact_lock,
)
from .model import (
    CompactModelConfig,
    CompactPlasticRNN,
    CompactRecurrentSequence,
    PackedLocalTrace,
)
from .optimization import forward_batch, query_from_state
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    registered_seeds,
)
from .task import make_task_generator


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
                raise ValueError(f"non-numeric array: {name}")
            arrays[name] = value
    return arrays


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("saved array inventory differs")
        for key, expected in arrays.items():
            np.testing.assert_array_equal(saved[key], expected)


def analysis_specification(specification: dict) -> dict:
    return {
        "evaluation": {
            "liu": {
                "choice_seed": 31301,
                "temperature": specification["liu_evaluation"]["temperature"],
            }
        },
        "statistics": {"samples": 10000, "interval": 0.95},
        "decision_contract": {
            "behavior": {
                "reference_contract": specification["parent_evidence"][
                    "behavior_contract"
                ],
                "reference_result": specification["parent_evidence"]["behavior_result"],
                "historically_quantitative_rows": [
                    "learned_accuracy",
                    "nonlearned_accuracy",
                    "difficult_pair_bimodality",
                    "stable_within_subject_errors",
                    "hodge_reconstructed_subjective_ranking",
                    "inter_subject_ranking_diversity",
                ],
            }
        },
    }


def load_model(seed: int, cohort: str, artifact_lock: dict):
    directory = run_directory(seed, cohort)
    payload = torch.load(
        directory / "model.pth", map_location="cuda", weights_only=True
    )
    config = CompactModelConfig(**payload["model_config"])
    backbone = CompactPlasticRNN(config, device="cuda")
    local = PackedLocalTrace(config.cue_size, device="cuda")
    backbone.load_state_dict(payload["backbone"], strict=True)
    local.load_state_dict(payload["local"], strict=True)
    metadata = artifact_lock["runs"][str(seed)]["metadata"]
    if (
        tensor_hashes(backbone) != metadata["final_backbone"]
        or tensor_hashes(local) != metadata["final_local"]
    ):
        raise RuntimeError("loaded tensors differ from the artifact lock")
    backbone.requires_grad_(False).eval()
    local.requires_grad_(False).eval()
    return backbone, local, metadata


def validation_episodes(specification: dict) -> tuple:
    task = make_task_generator(specification)
    rng = np.random.default_rng(specification["generic_evaluation"]["rng_seed"])
    return tuple(
        sample_episodes(task, rng, 1, validation=True)[0]
        for _ in range(specification["generic_evaluation"]["episodes"])
    )


def evaluate_generic(backbone, local, specification: dict) -> dict:
    episodes = validation_episodes(specification)
    groups = defaultdict(list)
    for index, episode in enumerate(episodes):
        groups[len(episode.support_trials)].append(index)
    sequence = compile_module(
        CompactRecurrentSequence(backbone),
        PROFILE,
    )
    margins = {
        name: np.empty((len(episodes), 28), dtype=np.float64)
        for name in specification["generic_evaluation"]["conditions"]
    }
    signs = np.empty((len(episodes), 28), dtype=np.float64)
    learned = np.empty((len(episodes), 28), dtype=bool)
    fingerprints = {}
    for length, indices in groups.items():
        selected = tuple(episodes[index] for index in indices)
        cpu = prepare_batch(selected)
        batch = cpu.to("cuda")
        with torch.no_grad():
            intact = forward_batch(
                backbone,
                local,
                sequence,
                batch,
                local_active=True,
                fast_weight_penalty=0.0,
            )
            p_off, _ = query_from_state(
                backbone,
                local,
                sequence,
                batch,
                torch.zeros_like(intact.fast_weights),
                intact.local_state,
                local_active=True,
            )
        rows = {
            "intact": intact.margins,
            "local_off": intact.global_margins,
            "P_off": p_off,
        }
        for name, values in rows.items():
            margins[name][indices] = (
                values[:, 0]
                .reshape(28, len(selected))
                .T.cpu()
                .numpy()
                .astype(np.float64)
            )
        signs[indices] = (2 * cpu.arrays["targets"] - 1).reshape(28, len(selected)).T
        learned[indices] = np.asarray(
            [
                [
                    tuple(sorted((query.left_item, query.right_item)))
                    in {
                        tuple(sorted((trial.left_item, trial.right_item)))
                        for trial in episode.support_trials
                    }
                    for query in episode.query_trials
                ]
                for episode in selected
            ],
            dtype=bool,
        )
        fingerprints[str(length)] = cpu.fingerprint()
    endpoints = {
        name: query_endpoints(
            values[:, :, None],
            signs[:, :, None],
            {"learned": learned, "nonlearned": ~learned},
            temperature=specification["generic_evaluation"]["temperature"],
        )
        for name, values in margins.items()
    }
    return {
        "endpoints": endpoints,
        "margins": margins,
        "correct_signs": signs,
        "learned": learned,
        "fingerprints": fingerprints,
    }


def competence_decision(summaries: dict) -> dict:
    rules = {
        "generic_learned": decisions.criterion(
            summaries["generic"]["intact"]["exact_decision"]["learned"],
            0.75,
            statistic="mean",
            operator=">=",
        ),
        "generic_nonlearned": decisions.criterion(
            summaries["generic"]["intact"]["exact_decision"]["nonlearned"],
            0.70,
            statistic="mean",
            operator=">=",
        ),
        "liu_overall": decisions.criterion(
            summaries["liu"]["intact"]["exact_decision"]["overall"],
            0.75,
            statistic="mean",
            operator=">=",
        ),
        "liu_nonlearned": decisions.criterion(
            summaries["liu"]["intact"]["exact_decision"]["nonlearned"],
            0.70,
            statistic="mean",
            operator=">=",
        ),
        "liu_transitivity": decisions.criterion(
            summaries["constructive"]["intact_transitive_triplet_fraction"],
            0.95,
            statistic="mean",
            operator=">=",
        ),
    }
    return {"checks": rules, "passed": all(row["passed"] for row in rules.values())}


def evaluate_one(seed: int, cohort: str, artifact_lock: dict, specification: dict):
    directory = RUN_ROOT / cohort / "evaluation" / f"seed-{seed}"
    if directory.exists():
        result = load_json(directory / "result.json")
        return result
    backbone, local, metadata = load_model(seed, cohort, artifact_lock)
    identity = {
        "seed": seed,
        "cohort": cohort,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(artifact_lock_path(cohort)),
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="compact_global_local_model_v1",
        execution_id=f"evaluation-{cohort}-{seed}",
        producer={"module": __name__, **identity},
        resolved_config={
            "generic": specification["generic_evaluation"],
            "liu": specification["liu_evaluation"],
        },
    ):
        generic = evaluate_generic(backbone, local, specification)
        protocol = load_registered_protocol("liu_v2")
        evaluator = CompactLiuEvaluator(
            backbone,
            local,
            protocol,
            subjects=specification["liu_evaluation"]["subjects"],
            cue_seed=31001,
            support_seed=31101,
            subject_encoding_seed=31201,
            cue_mode=specification["liu_evaluation"]["cue_mode"],
            subject_encoding_mode=specification["liu_evaluation"][
                "subject_encoding_mode"
            ],
        )
        liu = rollout_liu(evaluator)
        analysis = analysis_specification(specification)
        endpoints = liu_endpoints(
            liu["bundles"],
            liu["retention"],
            protocol,
            specification["liu_evaluation"]["temperature"],
        )
        geometry = summarize_geometry(
            liu["bundles"],
            liu["loo"],
            protocol,
            85000 + seed,
            analysis["statistics"],
        )
        effects = mechanism_effects(
            endpoints, geometry, 85000 + seed, analysis["statistics"]
        )
        raw_endpoints = {"generic": generic["endpoints"], "liu": endpoints}
        summaries = {
            domain: {
                name: summarize_endpoints(
                    row,
                    (85000 if domain == "liu" else 86000) + seed,
                    analysis["statistics"],
                )
                for name, row in conditions.items()
            }
            for domain, conditions in raw_endpoints.items()
        }
        summaries["constructive"] = geometry["constructive"]
        behavior = evaluate_behavior(liu["bundles"]["intact"], protocol, seed, analysis)
        competence = competence_decision(summaries)
        mechanism = decisions.mechanism(effects)
        behavior_decision = decisions.behavior_preservation(
            behavior["record"], analysis
        )
        seed_decisions = {
            "competence": competence,
            "mechanism": mechanism,
            "behavior": behavior_decision,
            "admitted": all(
                row["passed"] for row in (competence, mechanism, behavior_decision)
            ),
        }
        raw = {
            "generic": generic,
            "liu": liu,
            "geometry": geometry,
        }
        write_arrays(directory / "raw.npz", flatten_arrays(raw))
        write_json_exclusive(
            directory / "behavior.json", json_ready(behavior["sampled_behavior"])
        )
        result = {
            **identity,
            "metadata": metadata,
            "summaries": summaries,
            "effects": effects,
            "behavior": behavior["record"],
            "decisions": seed_decisions,
            "generic_stream_fingerprints": generic["fingerprints"],
            "raw_arrays": reference(directory / "raw.npz"),
            "sampled_behavior": reference(directory / "behavior.json"),
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return load_json(directory / "result.json")


def evaluate_cohort(cohort: str) -> dict:
    artifact_lock = validate_artifact_lock(cohort)
    specification = load_specification()
    runtime = configure_execution()
    completed = {}
    for seed in registered_seeds(specification, cohort):
        result = evaluate_one(seed, cohort, artifact_lock, specification)
        completed[str(seed)] = result["decisions"]
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "cohort": cohort,
        "runtime": runtime,
        "completed": completed,
        "all_admitted": all(row["admitted"] for row in completed.values()),
    }
