"""Post-lock paired generic and Liu evaluation for direct P/L training."""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fsrl.core.factorized_plastic_rnn import (
    FactorizedPlasticRNN,
    FactorizedPlasticRNNConfig,
    FactorizedRecurrentSequence,
)
from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.experiments.training_strategy import decisions
from fsrl.experiments.training_strategy.behavior import evaluate_behavior
from fsrl.experiments.training_strategy.estimands import (
    paired_estimate,
    query_endpoints,
)
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    mechanism_effects,
    summarize_endpoints,
    summarize_geometry,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .batches import prepare_batch, sample_episodes
from .execution import PROFILE, configure_execution
from .liu import DirectLiuEvaluator, rollout_liu
from .locks import (
    RUN_ROOT,
    artifact_lock_path,
    reference,
    run_directory,
    validate_artifact_lock,
)
from .model import NoTimePlasticRNN, NoTimeRecurrentSequence
from .optimization import forward_batch, query_from_state
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    registered_conditions,
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
                raise ValueError(f"non-numeric scientific array: {name}")
            arrays[name] = value
    return arrays


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("saved direct-training array inventory differs")
        for key, expected in arrays.items():
            np.testing.assert_array_equal(saved[key], expected)


def analysis_specification(specification: dict) -> dict:
    return {
        "evaluation": specification["evaluation"],
        "statistics": {
            "samples": specification["statistics"]["bootstrap_samples"],
            "interval": specification["statistics"]["interval"],
        },
        "decision_contract": {
            "behavior": {
                "reference_contract": (
                    "studies/behavior_reproduction_map/records/benchmarks/"
                    "model_behavior_reproduction_map_v1.json"
                ),
                "reference_result": (
                    "studies/behavior_reproduction_map/records/results/"
                    "model_behavior_reproduction_map_v1.json"
                ),
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


def load_model(seed: int, condition: str, cohort: str, artifact_lock: dict):
    directory = run_directory(seed, condition, cohort)
    payload = torch.load(
        directory / "model.pth", map_location="cuda", weights_only=True
    )
    config = FactorizedPlasticRNNConfig(**payload["model_config"])
    backbone = (
        NoTimePlasticRNN(config, device="cuda")
        if condition == "no_time_candidate"
        else FactorizedPlasticRNN(
            config, device="cuda", legacy_numerical_compatibility=False
        )
    )
    local = PackedConjunctiveLocalTrace(config.cue_size, device="cuda")
    backbone.load_state_dict(payload["backbone"], strict=True)
    local.load_state_dict(payload["local"], strict=True)
    metadata = artifact_lock["runs"][f"{seed}/{condition}"]["metadata"]
    if (
        payload["condition"] != condition
        or tensor_hashes(backbone) != metadata["final_backbone"]
        or tensor_hashes(local) != metadata["final_local"]
    ):
        raise RuntimeError("loaded direct-training tensors differ from artifact lock")
    backbone.requires_grad_(False).eval()
    local.requires_grad_(False).eval()
    return backbone, local, metadata


def validation_episodes(specification: dict) -> tuple:
    task = make_task_generator(specification)
    rng = np.random.default_rng(specification["evaluation"]["generic"]["rng_seed"])
    return tuple(
        sample_episodes(task, rng, 1, validation=True)[0]
        for _ in range(specification["evaluation"]["generic"]["episodes"])
    )


def _sequence(backbone):
    module = (
        NoTimeRecurrentSequence(backbone)
        if isinstance(backbone, NoTimePlasticRNN)
        else FactorizedRecurrentSequence(backbone)
    )
    return compile_module(module, PROFILE)


def evaluate_generic(condition, backbone, local, specification: dict) -> dict:
    episodes = validation_episodes(specification)
    groups = defaultdict(list)
    for index, episode in enumerate(episodes):
        groups[len(episode.support_trials)].append(index)
    sequence = _sequence(backbone)
    settings = specification["evaluation"]["generic"]
    margins = {
        name: np.empty((len(episodes), 28), dtype=np.float64)
        for name in settings["conditions"]
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
                condition,
                backbone,
                local,
                sequence,
                batch,
                fast_weight_penalty=0.0,
            )
            p_off, _ = query_from_state(
                condition,
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
            temperature=settings["temperature"],
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


def condition_analysis(
    seed: int,
    condition: str,
    cohort: str,
    artifact_lock: dict,
    specification: dict,
) -> tuple[dict, dict, dict]:
    backbone, local, metadata = load_model(seed, condition, cohort, artifact_lock)
    generic = evaluate_generic(condition, backbone, local, specification)
    settings = specification["evaluation"]["liu"]
    protocol = load_registered_protocol(settings["protocol_id"])
    evaluator = DirectLiuEvaluator(
        condition,
        backbone,
        local,
        protocol,
        subjects=settings["subjects"],
        cue_seed=settings["cue_seed"],
        support_seed=settings["support_seed"],
        subject_encoding_seed=settings["subject_encoding_seed"],
        cue_mode=settings["cue_mode"],
        subject_encoding_mode=settings["subject_encoding_mode"],
        test_time_value=2.0 / 3.0,
    )
    liu = rollout_liu(evaluator, settings)
    analysis = analysis_specification(specification)
    endpoints = liu_endpoints(
        liu["bundles"], liu["retention"], protocol, settings["temperature"]
    )
    bootstrap_seed = 87000 + seed
    geometry = summarize_geometry(
        liu["bundles"],
        liu["loo"],
        protocol,
        bootstrap_seed,
        analysis["statistics"],
    )
    effects = mechanism_effects(
        endpoints, geometry, bootstrap_seed, analysis["statistics"]
    )
    behavior = evaluate_behavior(liu["bundles"]["intact"], protocol, seed, analysis)
    raw_endpoints = {"generic": generic["endpoints"], "liu": endpoints}
    summaries = {
        domain: {
            name: summarize_endpoints(
                row,
                (87000 if domain == "liu" else 88000) + seed,
                analysis["statistics"],
            )
            for name, row in conditions.items()
        }
        for domain, conditions in raw_endpoints.items()
    }
    summaries["constructive"] = geometry["constructive"]
    competence = competence_decision(summaries)
    mechanism = decisions.mechanism(effects)
    behavior_decision = decisions.behavior_preservation(behavior["record"], analysis)
    admitted = all(row["passed"] for row in (competence, mechanism, behavior_decision))
    condition_decisions = {
        "competence": competence,
        "mechanism": mechanism,
        "behavior": behavior_decision,
        "admitted": admitted,
    }
    result = {
        "metadata": metadata,
        "summaries": summaries,
        "effects": effects,
        "behavior": behavior["record"],
        "decisions": condition_decisions,
        "generic_stream_fingerprints": generic["fingerprints"],
    }
    raw = {"generic": generic, "liu": liu, "geometry": geometry}
    return (
        result,
        raw_endpoints,
        {
            "arrays": flatten_arrays(raw),
            "sampled_behavior": behavior["sampled_behavior"],
        },
    )


def paired_analysis(raw: dict, seed: int, specification: dict) -> tuple[dict, dict]:
    statistics = {
        "samples": specification["statistics"]["bootstrap_samples"],
        "interval": specification["statistics"]["interval"],
    }
    groups = {
        "generic": ("learned", "nonlearned"),
        "liu": ("learned", "nonlearned", "retained", "omitted"),
    }
    estimates = {
        domain: {
            group: paired_estimate(
                raw["no_time_candidate"][domain]["intact"]["probability"][group],
                raw["time_retained_control"][domain]["intact"]["probability"][group],
                seed=(87000 if domain == "liu" else 88000) + seed,
                statistics=statistics,
            )
            for group in domain_groups
        }
        for domain, domain_groups in groups.items()
    }
    threshold = -0.02
    checks = {
        f"{domain}_{group}": decisions.criterion(
            estimate, threshold, statistic="lower", operator=">="
        )
        for domain, rows in estimates.items()
        for group, estimate in rows.items()
    }
    decision = {
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }
    return estimates, decision


def seed_outcome(condition_rows: dict, paired_decision: dict) -> str:
    control = condition_rows["time_retained_control"]["decisions"]
    candidate = condition_rows["no_time_candidate"]["decisions"]
    if not control["competence"]["passed"] or not control["mechanism"]["passed"]:
        return "training_parameterization_failure"
    if not candidate["competence"]["passed"]:
        return "no_time_recipe_failure"
    if not paired_decision["passed"]:
        return "competent_but_time_noninferior_failure"
    if not candidate["mechanism"]["passed"]:
        return "alternative_no_time_organization"
    if not control["behavior"]["passed"] or not candidate["behavior"]["passed"]:
        return "behavior_incomplete"
    return "development_admitted"


def evaluation_directory(seed: int, cohort: str) -> Path:
    return RUN_ROOT / cohort / "evaluation" / f"seed-{seed}"


def validate_evaluation(seed: int, cohort: str, artifact_lock: dict) -> dict:
    directory = evaluation_directory(seed, cohort)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("direct-training evaluation is incomplete or modified")
    result = load_json(directory / "result.json")
    expected = {
        "seed": seed,
        "cohort": cohort,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(artifact_lock_path(cohort)),
        "source_commit": artifact_lock["source_commit"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("direct-training evaluation provenance differs")
    return result


def evaluate_seed(
    seed: int,
    cohort: str,
    artifact_lock: dict,
    specification: dict,
    runtime: dict,
) -> dict:
    directory = evaluation_directory(seed, cohort)
    if directory.exists():
        return validate_evaluation(seed, cohort, artifact_lock)
    identity = {
        "seed": seed,
        "cohort": cohort,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(artifact_lock_path(cohort)),
        "source_commit": artifact_lock["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_direct_training_v1",
        execution_id=f"evaluation-{cohort}-{seed}",
        producer={"module": __name__, **identity},
        resolved_config={
            "evaluation": specification["evaluation"],
            "statistics": specification["statistics"],
            "runtime": runtime,
        },
    ):
        condition_rows = {}
        raw_endpoints = {}
        raw = {}
        sampled_behavior = {}
        for condition in registered_conditions(specification):
            row, endpoints, payload = condition_analysis(
                seed, condition, cohort, artifact_lock, specification
            )
            condition_rows[condition] = row
            raw_endpoints[condition] = endpoints
            raw[condition] = payload["arrays"]
            sampled_behavior[condition] = payload["sampled_behavior"]
            gc.collect()
            torch.cuda.empty_cache()
        if (
            condition_rows["time_retained_control"]["generic_stream_fingerprints"]
            != condition_rows["no_time_candidate"]["generic_stream_fingerprints"]
        ):
            raise RuntimeError("paired generic validation streams differ")
        paired, paired_decision = paired_analysis(raw_endpoints, seed, specification)
        outcome = seed_outcome(condition_rows, paired_decision)
        arrays = {
            f"{condition}__{name}": value
            for condition, rows in raw.items()
            for name, value in rows.items()
        }
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "behavior.json", json_ready(sampled_behavior))
        result = {
            **identity,
            "runtime": runtime,
            "conditions": condition_rows,
            "paired": paired,
            "paired_decision": paired_decision,
            "outcome": outcome,
            "admitted": outcome == "development_admitted",
            "raw_arrays": reference(directory / "raw.npz"),
            "sampled_behavior": reference(directory / "behavior.json"),
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return validate_evaluation(seed, cohort, artifact_lock)


def evaluate_cohort(cohort: str) -> dict:
    artifact_lock = validate_artifact_lock(cohort)
    specification = load_specification()
    runtime = configure_execution()
    completed = {}
    for seed in registered_seeds(specification, cohort):
        result = evaluate_seed(seed, cohort, artifact_lock, specification, runtime)
        completed[str(seed)] = {
            "outcome": result["outcome"],
            "admitted": result["admitted"],
        }
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "cohort": cohort,
        "completed": completed,
        "all_admitted": all(row["admitted"] for row in completed.values()),
    }
