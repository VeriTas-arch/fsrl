"""Diagnose slope and additive-versus-conjunctive assembly in frozen pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .config import DEVICE
from .confirmation import _validate_checkpoint
from .constructive import ExactRankingPosterior, RelationEvidence
from .human_benchmark import (
    DEFAULT_PREREGISTERED_PATH,
    DEFAULT_REPLICATION_PATH,
    SOURCE_FILES,
    load_human_cohort,
)
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from .ranking_protocol import RankingProtocol, load_ranking_protocol
from .study_registry import registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/assembly_diagnostics_v1.json")
DEFAULT_OUTPUT_PATH = resolve_record("results/assembly_diagnostics_v1.json")


@dataclass(frozen=True)
class FieldDesign:
    pairs: tuple[tuple[int, int], ...]
    projection: np.ndarray
    true_sign: np.ndarray
    symbolic_distance: np.ndarray
    learned_mask: np.ndarray
    slope_weights: np.ndarray
    learned_effect_weights: np.ndarray


def load_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def build_field_design(protocol: RankingProtocol) -> FieldDesign:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    incidence = np.zeros((len(pairs), protocol.n_items), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        incidence[index, first] = 1.0
        incidence[index, second] = -1.0
    projection = incidence @ np.linalg.pinv(incidence)

    true_positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        true_positions[item] = position
    true_sign = np.asarray(
        [
            1.0 if true_positions[first] < true_positions[second] else -1.0
            for first, second in pairs
        ]
    )
    symbolic_distance = np.asarray(
        [abs(true_positions[first] - true_positions[second]) for first, second in pairs]
    )
    learned_mask = np.asarray([pair in protocol.learned_pairs for pair in pairs])

    distances = np.arange(1, protocol.n_items, dtype=np.float64)
    centered = distances - np.mean(distances)
    distance_slope_weights = centered / np.sum(centered * centered)
    slope_weights = np.zeros(len(pairs), dtype=np.float64)
    for distance, weight in zip(distances.astype(int), distance_slope_weights):
        mask = symbolic_distance == distance
        slope_weights[mask] = weight / np.sum(mask)

    distance_dummies = np.column_stack(
        [symbolic_distance == distance for distance in range(2, protocol.n_items)]
    )
    learned_design = np.column_stack(
        [np.ones(len(pairs)), distance_dummies, learned_mask.astype(np.float64)]
    )
    learned_effect_weights = np.linalg.pinv(learned_design)[-1]
    return FieldDesign(
        pairs=pairs,
        projection=projection,
        true_sign=true_sign,
        symbolic_distance=symbolic_distance,
        learned_mask=learned_mask,
        slope_weights=slope_weights,
        learned_effect_weights=learned_effect_weights,
    )


def choice_fields_from_pair_accuracy(
    pair_accuracy: np.ndarray, design: FieldDesign
) -> np.ndarray:
    values = np.asarray(pair_accuracy, dtype=np.float64)
    if values.shape[-1] != len(design.pairs):
        raise ValueError("pair-accuracy array does not match the registered graph")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pair accuracies must lie in [0, 1]")
    return (2.0 * values - 1.0) * design.true_sign


def metric_arrays(fields: np.ndarray, design: FieldDesign) -> dict[str, np.ndarray]:
    values = np.atleast_2d(np.asarray(fields, dtype=np.float64))
    if values.shape[1] != len(design.pairs):
        raise ValueError("field array does not match the registered edge order")
    gradient = values @ design.projection.T
    residual = values - gradient
    gradient_energy = np.sum(gradient * gradient, axis=1)
    residual_energy = np.sum(residual * residual, axis=1)
    total_energy = gradient_energy + residual_energy
    gradient_fraction = np.divide(
        gradient_energy,
        total_energy,
        out=np.full_like(total_energy, np.nan),
        where=total_energy > 0.0,
    )

    accuracy = 0.5 * (1.0 + values * design.true_sign)
    gradient_accuracy = 0.5 * (1.0 + gradient * design.true_sign)
    total_slope = accuracy @ design.slope_weights
    gradient_slope = gradient_accuracy @ design.slope_weights
    residual_slope = total_slope - gradient_slope
    aligned_residual_accuracy = 0.5 * residual * design.true_sign

    return {
        "total_accuracy_slope": total_slope,
        "gradient_accuracy_slope": gradient_slope,
        "residual_accuracy_slope": residual_slope,
        "gradient_energy_fraction": gradient_fraction,
        "mean_absolute_residual": np.mean(np.abs(residual), axis=1),
        "mean_absolute_residual_learned": np.mean(
            np.abs(residual[:, design.learned_mask]), axis=1
        ),
        "mean_absolute_residual_nonlearned": np.mean(
            np.abs(residual[:, ~design.learned_mask]), axis=1
        ),
        "learned_absolute_residual_effect_adjusted_for_symbolic_distance": (
            np.abs(residual) @ design.learned_effect_weights
        ),
        "mean_correctness_aligned_residual_accuracy_learned": np.mean(
            aligned_residual_accuracy[:, design.learned_mask], axis=1
        ),
        "mean_correctness_aligned_residual_accuracy_nonlearned": np.mean(
            aligned_residual_accuracy[:, ~design.learned_mask], axis=1
        ),
        "learned_correctness_aligned_residual_accuracy_effect_adjusted_for_symbolic_distance": (
            aligned_residual_accuracy @ design.learned_effect_weights
        ),
    }


def _point_metrics(fields: np.ndarray, design: FieldDesign) -> dict[str, float]:
    arrays = metric_arrays(np.mean(fields, axis=0), design)
    return {name: float(values[0]) for name, values in arrays.items()}


def _distance_profiles(fields: np.ndarray, design: FieldDesign) -> dict:
    mean_field = np.mean(fields, axis=0, keepdims=True)
    gradient = mean_field @ design.projection.T
    total_accuracy = 0.5 * (1.0 + mean_field * design.true_sign)
    gradient_accuracy = 0.5 * (1.0 + gradient * design.true_sign)
    total = {}
    gradient_only = {}
    residual_contribution = {}
    for distance in range(1, len(np.unique(design.symbolic_distance)) + 1):
        mask = design.symbolic_distance == distance
        total[str(distance)] = float(np.mean(total_accuracy[:, mask]))
        gradient_only[str(distance)] = float(np.mean(gradient_accuracy[:, mask]))
        residual_contribution[str(distance)] = (
            total[str(distance)] - gradient_only[str(distance)]
        )
    return {
        "total_accuracy": total,
        "gradient_linear_accuracy_component": gradient_only,
        "residual_linear_accuracy_contribution": residual_contribution,
    }


def _subject_level_descriptive(fields: np.ndarray, design: FieldDesign) -> dict:
    arrays = metric_arrays(fields, design)
    return {
        name: {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "lower_quartile": float(np.nanquantile(values, 0.25)),
            "upper_quartile": float(np.nanquantile(values, 0.75)),
        }
        for name, values in arrays.items()
    }


def _bootstrap_counts(
    rng: np.random.Generator, samples: int, subjects: int
) -> np.ndarray:
    return rng.multinomial(
        n=subjects, pvals=np.full(subjects, 1.0 / subjects), size=samples
    )


def _bootstrap_metric_arrays(
    fields: np.ndarray, counts: np.ndarray, design: FieldDesign
) -> dict[str, np.ndarray]:
    means = counts @ fields / fields.shape[0]
    return metric_arrays(means, design)


def _interval(values: np.ndarray, interval: float) -> dict[str, float]:
    tail = (1.0 - interval) / 2.0
    return {
        "bootstrap_mean": float(np.mean(values)),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
    }


def summarize_cohort(
    fields: np.ndarray,
    counts: np.ndarray,
    design: FieldDesign,
    *,
    interval: float,
    metrics: tuple[str, ...] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    point = _point_metrics(fields, design)
    samples = _bootstrap_metric_arrays(fields, counts, design)
    selected = tuple(point) if metrics is None else metrics
    summary = {
        "subjects": int(fields.shape[0]),
        "point": {name: point[name] for name in selected},
        "subject_level_descriptive": {
            name: value
            for name, value in _subject_level_descriptive(fields, design).items()
            if name in selected
        },
        "bootstrap": {name: _interval(samples[name], interval) for name in selected},
    }
    if "total_accuracy_slope" in selected:
        summary["distance_profiles"] = _distance_profiles(fields, design)
    return summary, {name: samples[name] for name in selected}


def compare_cohorts(
    first_summary: dict,
    first_samples: dict[str, np.ndarray],
    second_summary: dict,
    second_samples: dict[str, np.ndarray],
    *,
    interval: float,
) -> dict:
    shared = tuple(name for name in first_summary["point"] if name in second_samples)
    return {
        "estimand": "first minus second",
        "point": {
            name: first_summary["point"][name] - second_summary["point"][name]
            for name in shared
        },
        "bootstrap": {
            name: _interval(first_samples[name] - second_samples[name], interval)
            for name in shared
        },
    }


def directional_diagnosis(
    exact_minus_human: dict,
    neural_minus_exact: dict,
    neural_minus_human: dict,
    human_minus_neural: dict,
) -> dict:
    def lower_positive(comparison: dict, metric: str) -> bool:
        return comparison["bootstrap"][metric]["lower"] > 0.0

    evidence_model = lower_positive(exact_minus_human, "total_accuracy_slope")
    neural_sharpening = lower_positive(neural_minus_exact, "total_accuracy_slope")
    mixed_code = lower_positive(
        neural_minus_human, "gradient_energy_fraction"
    ) and lower_positive(
        human_minus_neural,
        "learned_correctness_aligned_residual_accuracy_effect_adjusted_for_symbolic_distance",
    )
    return {
        "evidence_model_contribution": evidence_model,
        "neural_over_sharpening": neural_sharpening,
        "human_mixed_code_signal": mixed_code,
        "non_significant_means": "unresolved, not equivalent",
    }


def _validate_registered_file(registration: dict) -> dict:
    path = resolve_path(registration["path"])
    observed = registered_file_sha256(
        registration["path"], registration["sha256"], resolved_path=path
    )
    if observed != registration["sha256"]:
        raise RuntimeError(f"registered SHA-256 mismatch: {path}")
    return {"path": registration["path"], "sha256": observed}


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    validated = {
        name: _validate_registered_file(sources[name])
        for name in ("pilot_specification", "protocol", "human_benchmark")
    }
    artifacts = []
    for registration in sources["pilot_artifacts"]:
        row = {"seed": registration["seed"]}
        for prefix in ("checkpoint", "config", "behavior"):
            path = resolve_path(registration[f"{prefix}_path"])
            observed = file_sha256(path)
            if observed != registration[f"{prefix}_sha256"]:
                raise RuntimeError(f"registered SHA-256 mismatch: {path}")
            row[prefix] = {
                "path": registration[f"{prefix}_path"],
                "sha256": observed,
            }
        artifacts.append(row)
    validated["pilot_artifacts"] = artifacts
    return validated


def load_human_choice_fields(
    protocol: RankingProtocol, design: FieldDesign
) -> np.ndarray:
    preregistered = load_human_cohort(
        DEFAULT_PREREGISTERED_PATH,
        "preregistered",
        protocol,
        expected_sha256=SOURCE_FILES["preregistered"]["sha256"],
    )
    replication = load_human_cohort(
        DEFAULT_REPLICATION_PATH,
        "replication",
        protocol,
        expected_sha256=SOURCE_FILES["replication"]["sha256"],
    )
    eligible = [
        subject
        for subject in preregistered + replication
        if subject["overall_accuracy"] >= 0.5
    ]
    return choice_fields_from_pair_accuracy(
        np.asarray([subject["pair_accuracy"] for subject in eligible]), design
    )


def _read_frozen_pilot(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
) -> tuple[np.ndarray, tuple[tuple[dict, ...], ...], float]:
    seed = int(registration["seed"])
    checkpoint = resolve_path(registration["checkpoint_path"])
    behavior_path = resolve_path(registration["behavior_path"])
    _validate_checkpoint(checkpoint, pilot_specification, seed)
    behavior = load_json(behavior_path)
    net, config, checkpoint_info = load_retro_checkpoint(
        checkpoint, len(behavior["subjects"])
    )
    if behavior["checkpoint"]["sha256"] != checkpoint_info.sha256:
        raise RuntimeError(f"seed {seed} behavior and checkpoint do not match")
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=int(behavior["cue_seed"]),
        support_seed=int(behavior["support_seed"]),
        cue_mode="permuted_shared",
        subject_encoding_mode=behavior["subject_encoding_mode"],
        subject_encoding_seed=int(behavior["subject_encoding_seed"]),
    )
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    ordered_pairs = tuple(
        oriented
        for first, second in combinations(range(protocol.n_items), 2)
        for oriented in ((first, second), (second, first))
    )
    logits = evaluator.readout_logits(
        fast_weights, tuple(ordered_pairs for _ in behavior["subjects"])
    )
    margins = np.asarray(
        [
            [
                0.5 * (row[pair] - row[(pair[1], pair[0])])
                for pair in combinations(range(protocol.n_items), 2)
            ]
            for row in logits
        ],
        dtype=np.float64,
    )
    return (
        margins,
        evaluator.realized_support_evidence(),
        float(behavior["summary"]["symbolic_distance_slope"]["mean"]),
    )


def exact_posterior_choice_fields(
    protocol: RankingProtocol,
    evidence_by_subject: tuple[tuple[dict, ...], ...],
    *,
    temperature: float,
) -> np.ndarray:
    exact = ExactRankingPosterior(protocol.n_items, temperature=temperature)
    pairs = tuple(combinations(range(protocol.n_items), 2))
    fields = []
    for rows in evidence_by_subject:
        evidence = tuple(
            RelationEvidence(
                higher_item=row["higher_item"],
                lower_item=row["lower_item"],
                magnitude=row["magnitude"],
                reliability=row["reliability"],
            )
            for row in rows
        )
        posterior = exact.fit(evidence)
        fields.append(
            [2.0 * exact.pair_probability(posterior, *pair) - 1.0 for pair in pairs]
        )
    return np.asarray(fields, dtype=np.float64)


def _overall_diagnosis(per_seed: dict[str, dict], evidence_model: bool) -> dict:
    neural_sharpening = all(
        row["directional_diagnosis"]["neural_over_sharpening"]
        for row in per_seed.values()
    )
    mixed_code = all(
        row["directional_diagnosis"]["human_mixed_code_signal"]
        for row in per_seed.values()
    )
    if mixed_code:
        next_test = "test local/conjunctive traces alongside the preserved global assembly baseline"
    elif evidence_model and not neural_sharpening:
        next_test = "test magnitude compression and evidence uncertainty"
    elif neural_sharpening and not evidence_model:
        next_test = "identify learned-prior or neural over-sharpening dynamics"
    else:
        next_test = "run intermediate-fast-weight trajectory and leave-one-relation-out diagnostics"
    return {
        "evidence_model_contribution": evidence_model,
        "neural_over_sharpening_replicated_across_pilot_seeds": neural_sharpening,
        "human_mixed_code_signal_replicated_across_pilot_seeds": mixed_code,
        "next_mechanism_test": next_test,
        "formal_confirmation_status": "deferred; frozen contract remains unchanged",
    }


def run_assembly_diagnostics(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    human_benchmark = load_json(resolve_path(sources["human_benchmark"]["path"]))
    design = build_field_design(protocol)

    human_fields = load_human_choice_fields(protocol, design)
    expected_human_slope = human_benchmark["combined"]["symbolic_distance_slope"][
        "mean"
    ]
    observed_human_slope = _point_metrics(human_fields, design)["total_accuracy_slope"]
    if not np.isclose(observed_human_slope, expected_human_slope, atol=1e-12):
        raise RuntimeError("human pair field does not reproduce the registered slope")

    pilot_rows = []
    reference_evidence = None
    for registration in sources["pilot_artifacts"]:
        margins, evidence, registered_slope = _read_frozen_pilot(
            registration, pilot_specification, protocol
        )
        if reference_evidence is None:
            reference_evidence = evidence
        elif evidence != reference_evidence:
            raise RuntimeError("pilot seeds used different realized support evidence")
        temperature = float(pilot_specification["evaluation"]["temperature"])
        pilot_rows.append(
            {
                "seed": int(registration["seed"]),
                "logit_fields": margins,
                "choice_fields": np.tanh(margins / (2.0 * temperature)),
                "registered_sampled_neural_slope": registered_slope,
            }
        )
    assert reference_evidence is not None
    exact_fields = exact_posterior_choice_fields(
        protocol,
        reference_evidence,
        temperature=float(pilot_specification["evaluation"]["posterior_temperature"]),
    )

    bootstrap = specification["bootstrap"]
    samples = int(bootstrap["samples"])
    interval = float(bootstrap["interval"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    human_counts = _bootstrap_counts(rng, samples, len(human_fields))
    model_counts = _bootstrap_counts(rng, samples, len(exact_fields))
    human_summary, human_samples = summarize_cohort(
        human_fields, human_counts, design, interval=interval
    )
    exact_summary, exact_samples = summarize_cohort(
        exact_fields, model_counts, design, interval=interval
    )
    exact_minus_human = compare_cohorts(
        exact_summary,
        exact_samples,
        human_summary,
        human_samples,
        interval=interval,
    )
    evidence_model = (
        exact_minus_human["bootstrap"]["total_accuracy_slope"]["lower"] > 0.0
    )

    logit_metrics = (
        "gradient_energy_fraction",
        "mean_absolute_residual",
        "mean_absolute_residual_learned",
        "mean_absolute_residual_nonlearned",
        "learned_absolute_residual_effect_adjusted_for_symbolic_distance",
    )
    per_seed = {}
    for row in pilot_rows:
        neural_summary, neural_samples = summarize_cohort(
            row["choice_fields"], model_counts, design, interval=interval
        )
        logit_summary, _logit_samples = summarize_cohort(
            row["logit_fields"],
            model_counts,
            design,
            interval=interval,
            metrics=logit_metrics,
        )
        neural_minus_exact = compare_cohorts(
            neural_summary,
            neural_samples,
            exact_summary,
            exact_samples,
            interval=interval,
        )
        neural_minus_human = compare_cohorts(
            neural_summary,
            neural_samples,
            human_summary,
            human_samples,
            interval=interval,
        )
        human_minus_neural = compare_cohorts(
            human_summary,
            human_samples,
            neural_summary,
            neural_samples,
            interval=interval,
        )
        diagnosis = directional_diagnosis(
            exact_minus_human,
            neural_minus_exact,
            neural_minus_human,
            human_minus_neural,
        )
        per_seed[str(row["seed"])] = {
            "seed": row["seed"],
            "registered_sampled_neural_slope": row["registered_sampled_neural_slope"],
            "neural_choice_field": neural_summary,
            "neural_logit_field": logit_summary,
            "comparisons": {
                "neural_minus_exact_posterior": neural_minus_exact,
                "neural_minus_human": neural_minus_human,
                "human_minus_neural": human_minus_neural,
            },
            "directional_diagnosis": diagnosis,
        }

    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "device": {
            "neural_readout": DEVICE,
            "posterior_hodge_bootstrap": "cpu_numpy",
        },
        "artifact_validation": validation,
        "estimand": specification["field_contract"],
        "projection_boundary": (
            "Hodge gradient fields are not clipped to [-1, 1]. Their linear "
            "accuracy components may fall outside [0, 1]; only the recombined "
            "total is a choice probability, while component slopes are additive "
            "decomposition estimands."
        ),
        "bootstrap": bootstrap,
        "human_choice_field": human_summary,
        "exact_posterior_choice_field": exact_summary,
        "exact_posterior_minus_human": exact_minus_human,
        "pilot_seeds": per_seed,
        "overall_diagnosis": _overall_diagnosis(per_seed, evidence_model),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run registered slope-source and Hodge diagnostics."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_assembly_diagnostics(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
