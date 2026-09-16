"""Post-lock functional evaluation of the clean no-time P/L candidate."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.analysis.statistics import summarize_subjects
from fsrl.core.factorized_plastic_rnn import FactorizedPlasticRNNConfig
from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.experiments.training_strategy import decisions
from fsrl.experiments.training_strategy.behavior import (
    behavior_metrics,
    classify_rows,
    human_references,
)
from fsrl.experiments.training_strategy.estimands import paired_estimate
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    summarize_endpoints,
    summarize_geometry,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from ..evaluation import evaluate_generic, flatten_arrays, json_ready, write_arrays
from ..model import NoTimePlasticRNN
from .estimands import (
    causal_fields,
    cohort_outcome,
    four_link_flags,
    functional_contrasts,
    learned_probabilities,
    probability_metrics,
    resampling_counts,
    seed_outcome,
)
from .liu import FunctionalLiuEvaluator, rollout_functional
from .locks import (
    ARTIFACT_LOCK_PATH,
    RUN_ROOT,
    reference,
    run_directory,
    validate_artifact_lock,
)
from .protocol import (
    CONDITION,
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    candidate_specification,
    load_specification,
    registered_seeds,
)


def analysis_specification(specification: dict) -> dict:
    return {
        "evaluation": {"liu": specification["evaluation_conditions"]["liu"]},
        "statistics": {
            "samples": specification["statistics"]["bootstrap_samples"],
            "interval": specification["statistics"]["interval"],
        },
        "decision_contract": {
            "behavior": {
                "reference_contract": specification["behavior_reproduction_contract"][
                    "reference_contract"
                ]["path"],
                "reference_result": specification["behavior_reproduction_contract"][
                    "reference_result"
                ]["path"],
            }
        },
    }


def load_model(seed: int, artifact_lock: dict):
    directory = run_directory(seed)
    payload = torch.load(
        directory / "model.pth", map_location="cuda", weights_only=True
    )
    config = FactorizedPlasticRNNConfig(**payload["model_config"])
    backbone = NoTimePlasticRNN(config, device="cuda")
    local = PackedConjunctiveLocalTrace(config.cue_size, device="cuda")
    backbone.load_state_dict(payload["backbone"], strict=True)
    local.load_state_dict(payload["local"], strict=True)
    metadata = artifact_lock["runs"][str(seed)]["metadata"]
    if (
        payload["condition"] != CONDITION
        or tensor_hashes(backbone) != metadata["final_backbone"]
        or tensor_hashes(local) != metadata["final_local"]
    ):
        raise RuntimeError("loaded functional tensors differ from artifact lock")
    backbone.requires_grad_(False).eval()
    local.requires_grad_(False).eval()
    return backbone, local, metadata


def _behavior_analysis(bundle: dict, protocol, seed: int, analysis: dict) -> dict:
    settings = analysis["evaluation"]["liu"]
    schedules = (ordered_pairs(protocol.n_items),) * bundle["logits"].shape[0]
    sampled = analyze_sampled_query_policy(
        protocol,
        bundle_logits(bundle, schedules),
        seed=int(settings["choice_seed"]),
        temperature=float(settings["temperature"]),
    )
    record = behavior_metrics(sampled, 89000 + seed, analysis["statistics"])
    record["flags"] = classify_rows(record["metrics"], human_references(analysis))
    record["seed"] = seed
    record["bootstrap_scope"] = (
        "Frozen cohort and point definitions; no human interval refit."
    )
    return {"sampled": sampled, "record": record}


def _sampled_nonlearned(
    bundle: dict,
    protocol,
    settings: dict,
    counts: np.ndarray,
    interval: float,
) -> tuple[dict, dict]:
    schedules = (ordered_pairs(protocol.n_items),) * bundle["logits"].shape[0]
    sampled = analyze_sampled_query_policy(
        protocol,
        bundle_logits(bundle, schedules),
        seed=int(settings["choice_seed"]),
        temperature=float(settings["temperature"]),
    )
    values = np.asarray(
        [row["nonlearned_accuracy"] for row in sampled["subjects"]],
        dtype=np.float64,
    )
    return sampled, summarize_subjects(values, counts, interval=interval)


def _access_integrity(evaluator: FunctionalLiuEvaluator, rollout: dict) -> dict:
    retained_errors = []
    omitted_errors = []
    shared_errors = []
    for subject, schedule in enumerate(evaluator.support_schedules):
        for trial_index, trial in enumerate(schedule):
            retained = evaluator.trial_gain(subject, trial_index)
            probability = evaluator.relation_probability(
                subject, trial.higher_item, trial.lower_item
            )
            shared = rollout["shared_local_evidence"][subject, trial_index]
            dual = rollout["dual_local_evidence"][subject, trial_index]
            shared_errors.append(abs(shared - trial.signed_magnitude * retained))
            if retained > 0.0:
                retained_errors.append(abs(dual - shared))
            else:
                omitted_errors.append(abs(dual - trial.signed_magnitude * probability))
    relations = len(evaluator.protocol.support_pairs_higher_lower)
    multiset_error = 0.0
    for subject in range(evaluator.subjects):
        for block in range(evaluator.protocol.support_blocks):
            start = block * relations
            stop = start + relations
            multiset_error = max(
                multiset_error,
                float(
                    np.max(
                        np.abs(
                            np.sort(rollout["dual_local_evidence"][subject, start:stop])
                            - np.sort(
                                rollout["shuffled_local_evidence"][subject, start:stop]
                            )
                        )
                    )
                ),
            )
    evidence_maps = rollout["evidence_routing"]
    query_maps = rollout["query_routing"]
    return {
        "shared_scalar_max_abs_error": float(max(shared_errors, default=0.0)),
        "retained_own_write_max_abs_error": float(max(retained_errors, default=0.0)),
        "omitted_weak_scalar_max_abs_error": float(max(omitted_errors, default=0.0)),
        "evidence_shuffle_multiset_max_abs_error": multiset_error,
        "all_evidence_maps_are_derangements": bool(
            np.all(evidence_maps != np.arange(relations)[None, None, :])
        ),
        "all_query_maps_are_derangements": bool(
            np.all(
                query_maps[:, 0::2] // 2 != np.arange(query_maps.shape[1] // 2)[None, :]
            )
        ),
    }


def _criterion(estimate: dict, threshold: float, statistic: str, operator: str) -> dict:
    return decisions.criterion(
        estimate, threshold, statistic=statistic, operator=operator
    )


def evaluate_seed(
    seed: int,
    artifact_lock: dict,
    specification: dict,
    runtime: dict,
) -> dict:
    directory = evaluation_directory(seed)
    if directory.exists():
        return validate_evaluation(seed, artifact_lock)
    identity = {
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
    }
    analysis = analysis_specification(specification)
    statistics = analysis["statistics"]
    settings = specification["evaluation_conditions"]["liu"]
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_functional_replication_v1",
        execution_id=f"evaluation-{seed}",
        producer={"module": __name__, **identity},
        resolved_config={
            "evaluation": specification["evaluation_conditions"],
            "statistics": specification["statistics"],
            "runtime": runtime,
        },
    ):
        backbone, local, metadata = load_model(seed, artifact_lock)
        before = (tensor_hashes(backbone), tensor_hashes(local))
        generic = evaluate_generic(
            CONDITION, backbone, local, candidate_specification()
        )
        protocol = load_registered_protocol(settings["protocol_id"])
        evaluator = FunctionalLiuEvaluator(
            CONDITION,
            backbone,
            local,
            protocol,
            subjects=int(settings["subjects"]),
            cue_seed=int(settings["cue_seed"]),
            support_seed=int(settings["support_seed"]),
            subject_encoding_seed=int(settings["subject_encoding_seed"]),
            cue_mode=str(settings["cue_mode"]),
            subject_encoding_mode=str(settings["subject_encoding_mode"]),
            test_time_value=2.0 / 3.0,
        )
        rollout = rollout_functional(evaluator, settings)
        counts = resampling_counts(seed, evaluator.subjects, statistics)
        interval = float(statistics["interval"])
        retention = rollout["retention"]
        probabilities = {
            condition: probability_metrics(
                learned_probabilities(protocol, bundle, float(settings["temperature"])),
                retention,
                counts,
                interval,
            )
            for condition, bundle in rollout["bundles"].items()
        }
        fields = causal_fields(
            protocol,
            rollout["bundles"],
            rollout["loo"],
            retention,
            counts,
            interval,
        )
        contrasts = functional_contrasts(probabilities, fields, counts, interval)
        endpoints = liu_endpoints(
            rollout["bundles"],
            retention,
            protocol,
            float(settings["temperature"]),
        )
        generic_summary = {
            condition: summarize_endpoints(row, 90000 + seed, statistics)
            for condition, row in generic["endpoints"].items()
        }
        liu_summary = {
            condition: summarize_endpoints(row, 89000 + seed, statistics)
            for condition, row in endpoints.items()
        }
        geometry = summarize_geometry(
            {
                "intact": rollout["bundles"]["dual_access"],
                "local_off": rollout["bundles"]["local_off"],
                "P_off": rollout["bundles"]["P_off_dual"],
            },
            {
                "global": rollout["loo"]["local_off"],
                "local": rollout["loo"]["P_off_dual"],
                "combined": rollout["loo"]["dual_access"],
            },
            protocol,
            89000 + seed,
            statistics,
        )
        behavior = _behavior_analysis(
            rollout["bundles"]["dual_access"], protocol, seed, analysis
        )
        p_off_sampled, p_off_nonlearned = _sampled_nonlearned(
            rollout["bundles"]["P_off_dual"],
            protocol,
            settings,
            counts,
            interval,
        )
        competence_checks = {
            "generic_learned": _criterion(
                generic_summary["intact"]["exact_decision"]["learned"],
                0.75,
                "mean",
                ">=",
            ),
            "generic_nonlearned": _criterion(
                generic_summary["intact"]["exact_decision"]["nonlearned"],
                0.70,
                "mean",
                ">=",
            ),
            "liu_overall": _criterion(
                liu_summary["dual_access"]["exact_decision"]["overall"],
                0.75,
                "mean",
                ">=",
            ),
            "liu_nonlearned": _criterion(
                liu_summary["dual_access"]["exact_decision"]["nonlearned"],
                0.70,
                "mean",
                ">=",
            ),
            "liu_transitivity": _criterion(
                geometry["constructive"]["intact_transitive_triplet_fraction"],
                0.95,
                "mean",
                ">=",
            ),
        }
        competence_passed = all(row["passed"] for row in competence_checks.values())
        global_necessity = paired_estimate(
            endpoints["dual_access"]["probability"]["nonlearned"],
            endpoints["P_off_dual"]["probability"]["nonlearned"],
            seed=89000 + seed,
            statistics=statistics,
        )
        global_checks = {
            "P_global_necessity_effect": _criterion(
                global_necessity, 0.10, "lower", ">="
            ),
            "P_off_nonlearned_ceiling": _criterion(
                liu_summary["P_off_dual"]["probability"]["nonlearned"],
                0.55,
                "upper",
                "<=",
            ),
            "global_remote_absolute": _criterion(
                fields["local_off"]["summary"]["all"]["remote_absolute"],
                0.01,
                "lower",
                ">",
            ),
            "global_third_party_relational": _criterion(
                fields["local_off"]["summary"]["all"]["third_party_relational"],
                0.05,
                "lower",
                ">",
            ),
        }
        global_path_passed = all(row["passed"] for row in global_checks.values())
        access_integrity = _access_integrity(evaluator, rollout)
        endpoint_parity_error = max(
            float(
                np.nanmax(
                    np.abs(
                        np.asarray(
                            probabilities["dual_access"]["raw_subject_level"][group]
                        )
                        - endpoints["dual_access"]["probability"][group]
                    )
                )
            )
            for group in ("retained", "omitted")
        )
        integrity = {
            **rollout["integrity"],
            **access_integrity,
            "historical_probability_endpoint_max_abs_error": endpoint_parity_error,
            "backbone_and_gain_hashes_unchanged": before
            == (tensor_hashes(backbone), tensor_hashes(local)),
            "artifact_validation_passed": True,
        }
        integrity["all_passed"] = bool(
            integrity["global_condition_logit_max_abs_error"] <= 1e-7
            and integrity["local_margin_identity_max_abs_error"] <= 1e-6
            and integrity["shared_scalar_max_abs_error"] <= 1e-7
            and integrity["retained_own_write_max_abs_error"] <= 1e-7
            and integrity["omitted_weak_scalar_max_abs_error"] <= 1e-7
            and integrity["evidence_shuffle_multiset_max_abs_error"] <= 1e-7
            and integrity["all_evidence_maps_are_derangements"]
            and integrity["all_query_maps_are_derangements"]
            and integrity["historical_probability_endpoint_max_abs_error"] <= 1e-12
            and integrity["backbone_and_gain_hashes_unchanged"]
        )
        link_flags = four_link_flags(
            contrasts,
            probabilities,
            p_off_nonlearned,
            integrity["retained_own_write_max_abs_error"],
            global_path_passed,
        )
        omitted_materiality = _criterion(
            contrasts["dual_minus_local_off_omitted_exact_probability"],
            0.01,
            "lower",
            ">=",
        )
        qualitative_behavior = all(
            row["qualitative"] for row in behavior["record"]["flags"].values()
        )
        quantitative_count = sum(
            row["calibration"] for row in behavior["record"]["flags"].values()
        )
        outcome = seed_outcome(
            integrity=integrity["all_passed"],
            competence=competence_passed,
            global_path=global_path_passed,
            four_links=link_flags,
            omitted_materiality=omitted_materiality["passed"],
            qualitative_behavior=qualitative_behavior,
        )
        arrays = {
            **{
                f"liu__{name}": value for name, value in flatten_arrays(rollout).items()
            },
            **{
                f"generic__{name}": value
                for name, value in flatten_arrays(generic).items()
            },
        }
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(
            directory / "behavior.json",
            json_ready(
                {
                    "dual_access": behavior["sampled"],
                    "P_off_dual": p_off_sampled,
                }
            ),
        )
        result = {
            **identity,
            "runtime": runtime,
            "metadata": metadata,
            "integrity": integrity,
            "summaries": {
                "generic": generic_summary,
                "liu": liu_summary,
                "constructive": geometry["constructive"],
            },
            "global_path": {
                "effect": global_necessity,
                "checks": global_checks,
                "passed": global_path_passed,
            },
            "probability": probabilities,
            "causal_fields": fields,
            "contrasts": contrasts,
            "P_off_dual_nonlearned_sampled": p_off_nonlearned,
            "four_v2_4_links": {
                "flags": link_flags,
                "passed": all(link_flags.values()),
            },
            "omitted_local_materiality": omitted_materiality,
            "retained_local_materiality_diagnostic": contrasts[
                "dual_minus_local_off_retained_exact_probability"
            ],
            "competence": {
                "checks": competence_checks,
                "passed": competence_passed,
            },
            "behavior": {
                "record": behavior["record"],
                "all_nine_qualitative_pass": qualitative_behavior,
                "quantitative_calibration_pass_count_report_only": quantitative_count,
                "quantitative_calibration_total": len(behavior["record"]["flags"]),
            },
            "outcome": outcome,
            "primary_functional_replication": outcome
            in {
                "clean_no_time_pl_functional_replication",
                "mechanism_replication_behavior_incomplete",
            },
            "raw_arrays": reference(directory / "raw.npz"),
            "sampled_behavior": reference(directory / "behavior.json"),
            "generic_stream_fingerprints": generic["fingerprints"],
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return validate_evaluation(seed, artifact_lock)


def evaluation_directory(seed: int) -> Path:
    return RUN_ROOT / "evaluation" / f"seed-{seed}"


def validate_evaluation(seed: int, artifact_lock: dict) -> dict:
    directory = evaluation_directory(seed)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("functional evaluation is incomplete or modified")
    result = load_json(directory / "result.json")
    expected = {
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("functional evaluation provenance differs")
    return result


def evaluate_cohort() -> dict:
    artifact_lock = validate_artifact_lock()
    specification = load_specification()
    from ..execution import configure_execution

    runtime = configure_execution()
    completed = {}
    for seed in registered_seeds(specification):
        result = evaluate_seed(seed, artifact_lock, specification, runtime)
        completed[str(seed)] = {
            "outcome": result["outcome"],
            "primary_functional_replication": result["primary_functional_replication"],
        }
        gc.collect()
        torch.cuda.empty_cache()
    transport_triggered = all(
        row["primary_functional_replication"] for row in completed.values()
    )
    if transport_triggered:
        from .transport import evaluate_transport

        transport = evaluate_transport(artifact_lock, specification, runtime)[
            "decision"
        ]
    else:
        transport = {
            "outcome": "not_triggered",
            "reason": "At least one fresh seed failed primary functional replication.",
        }
    return {
        "completed": completed,
        "outcome": cohort_outcome(completed),
        "transport_triggered": transport_triggered,
        "transport": transport,
    }
