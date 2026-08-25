"""Frozen multi-seed confirmation workflow for the Liu ranking candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsrl.analysis.algorithmic import run_algorithmic_comparison
from fsrl.analysis.behavioral import run_behavioral_analysis
from fsrl.analysis.geometry import run_geometry_analysis
from fsrl.core.config import DEVICE
from fsrl.evaluation.frozen_fast_weight import run_causal_suite
from fsrl.evaluation.qualification import evaluate_qualification
from fsrl.infrastructure.formal_runtime import require_formal_runtime
from fsrl.infrastructure.study_registry import canonical_file_sha256, resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.meta_tasks import held_out_liu_graph_signatures
from fsrl.training.backbone import (
    COMPILED_TRAINING_EXECUTION,
    MetaTrainConfig,
    train_meta_model,
)

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/confirmation_v1.json")
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "runs" / "confirmation-v1"
FORMAL_CONFIRMATION_ID = "liu-neural-constructive-ranking-confirmation-v1"
FORMAL_RUNTIME_SOURCE = ROOT / "fsrl" / "infrastructure" / "formal_runtime.py"
FORMAL_TRAINING_SOURCE = ROOT / "fsrl" / "training" / "backbone.py"


def file_sha256(path: Path) -> str:
    return canonical_file_sha256(path)


def load_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resolve_registered_path(specification_path: Path, registered: str) -> Path:
    candidate = Path(registered)
    if candidate.is_absolute():
        return candidate
    return resolve_record(candidate)


def validate_confirmation_contract(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = specification_path.resolve()
    specification = load_json(specification_path)
    checks = []
    for section in ("protocol", "human_benchmark", "qualification", "geometry"):
        registration = specification[section]
        path = resolve_registered_path(specification_path, registration["path"])
        observed_hash = file_sha256(path)
        checks.append(
            {
                "name": f"{section}.sha256",
                "passed": observed_hash == registration["sha256"],
                "observed": observed_hash,
                "expected": registration["sha256"],
                "path": str(path),
            }
        )
    human = load_json(
        resolve_registered_path(
            specification_path, specification["human_benchmark"]["path"]
        )
    )
    human_registration = specification["human_benchmark"]
    checks.extend(
        [
            {
                "name": "human_benchmark.status",
                "passed": human["status"] == human_registration["required_status"],
                "observed": human["status"],
                "expected": human_registration["required_status"],
            },
            {
                "name": "human_benchmark.bootstrap_samples",
                "passed": human["bootstrap"]["samples"]
                == human_registration["bootstrap_samples"],
                "observed": human["bootstrap"]["samples"],
                "expected": human_registration["bootstrap_samples"],
            },
        ]
    )
    return {
        "confirmation_id": specification["confirmation_id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _specification_and_paths(specification_path: Path) -> tuple[dict, dict[str, Path]]:
    specification_path = specification_path.resolve()
    specification = load_json(specification_path)
    paths = {
        section: resolve_registered_path(
            specification_path, specification[section]["path"]
        )
        for section in ("protocol", "human_benchmark", "qualification", "geometry")
    }
    return specification, paths


def _training_config(specification: dict, seed: int) -> MetaTrainConfig:
    registered = dict(specification["training"])
    registered.pop("seeds")
    registered.pop("checkpoint_selection")
    return MetaTrainConfig(seed=seed, **registered)


def _require_registered_runtime(specification: dict) -> dict | None:
    if specification["confirmation_id"] == FORMAL_CONFIRMATION_ID:
        return require_formal_runtime()
    return None


def _formal_runtime_source_registration() -> dict:
    return {
        "path": str(FORMAL_RUNTIME_SOURCE.relative_to(ROOT)),
        "sha256": file_sha256(FORMAL_RUNTIME_SOURCE),
    }


def _formal_training_source_registration() -> dict:
    return {
        "path": str(FORMAL_TRAINING_SOURCE.relative_to(ROOT)),
        "sha256": file_sha256(FORMAL_TRAINING_SOURCE),
    }


def _validate_formal_runtime_record(result: dict) -> None:
    runtime = result.get("execution_runtime", {})
    if not (
        runtime.get("active")
        and runtime.get("cpu_thread_limit") == 1
        and runtime.get("torch_intraop_threads") == 1
        and runtime.get("torch_interop_threads") == 1
        and runtime.get("cuda_available")
        and runtime.get("device") == "cuda"
        and runtime.get("torch_version")
        and runtime.get("cuda_version")
        and runtime.get("device_name")
    ):
        raise RuntimeError("formal seed result lacks the bounded GPU runtime record")
    registration = result.get("execution_runtime_source", {})
    expected_registration = _formal_runtime_source_registration()
    if registration != expected_registration:
        raise RuntimeError("formal seed runtime source hash does not match")


def _validate_formal_training_execution(result: dict) -> None:
    if result.get("training_execution") != COMPILED_TRAINING_EXECUTION:
        raise RuntimeError("formal seed result lacks the registered compiled training")
    if (
        result.get("training_execution_source")
        != _formal_training_source_registration()
    ):
        raise RuntimeError("formal seed compiled-training source hash does not match")


def _validate_seed(specification: dict, seed: int) -> None:
    if seed not in specification["training"]["seeds"]:
        raise ValueError(f"seed {seed} is not registered for confirmation")


def _validate_checkpoint(checkpoint: Path, specification: dict, seed: int) -> dict:
    metadata = load_json(checkpoint.parent / "config.json")
    expected_training = dict(specification["training"])
    expected_training.pop("seeds")
    expected_training.pop("checkpoint_selection")
    expected_training["seed"] = seed
    observed_training = metadata["training"]
    if observed_training != expected_training:
        raise RuntimeError(
            f"seed {seed} checkpoint training configuration is not registered"
        )
    if (
        specification["confirmation_id"] == FORMAL_CONFIRMATION_ID
        and metadata.get("execution") != COMPILED_TRAINING_EXECUTION
    ):
        raise RuntimeError(
            f"seed {seed} checkpoint was not trained with the registered compiler"
        )
    if metadata["completed_outer_steps"] != specification["training"]["outer_steps"]:
        raise RuntimeError(f"seed {seed} checkpoint is not the fixed final step")
    observed_signatures = {
        tuple(tuple(pair) for pair in signature)
        for signature in metadata["task_distribution"].get(
            "held_out_rank_graph_signatures", []
        )
    }
    if observed_signatures != set(held_out_liu_graph_signatures()):
        raise RuntimeError(
            f"seed {seed} training did not exclude both Liu graph signatures"
        )
    observed_hash = file_sha256(checkpoint)
    if metadata["checkpoint"]["sha256"] != observed_hash:
        raise RuntimeError(f"seed {seed} checkpoint hash does not match metadata")
    return metadata


def train_confirmation_seed(
    seed: int,
    *,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    contract = validate_confirmation_contract(specification_path)
    if not contract["passed"]:
        raise RuntimeError("confirmation contract validation failed before training")
    specification = load_json(specification_path)
    _require_registered_runtime(specification)
    _validate_seed(specification, seed)
    seed_dir = output_root / f"seed-{seed}"
    checkpoint = seed_dir / "net.dat"
    if checkpoint.exists():
        _validate_checkpoint(checkpoint, specification, seed)
        return checkpoint
    train_meta_model(
        _training_config(specification, seed),
        seed_dir,
        compile_model=specification["confirmation_id"] == FORMAL_CONFIRMATION_ID,
    )
    _validate_checkpoint(checkpoint, specification, seed)
    return checkpoint


def compare_behavior_to_human(
    behavior: dict, human: dict, registered_metrics: list[str]
) -> dict:
    summary = behavior["summary"]
    eligible = summary["eligible_subjects"]
    model_values = {
        "overall_accuracy": summary["overall_accuracy"],
        "learned_accuracy": summary["learned_accuracy"],
        "nonlearned_accuracy": summary["nonlearned_accuracy"],
        "symbolic_distance_slope": summary["symbolic_distance_slope"]["mean"],
        "correct_ranker_proportion": summary["ranking_class_counts"]["correct"]
        / eligible,
        "self_consistent_incorrect_proportion": summary["ranking_class_counts"][
            "self_consistent_incorrect"
        ]
        / eligible,
        "self_inconsistent_proportion": summary["ranking_class_counts"][
            "self_inconsistent"
        ]
        / eligible,
        "stable_error_80_analysis_proportion": summary[
            "stable_error_subject_prevalence"
        ]["80"]["analysis"],
        "stable_error_100_analysis_proportion": summary[
            "stable_error_subject_prevalence"
        ]["100"]["analysis"],
    }
    bootstrap = human["bootstrap"]["metrics"]
    checks = []
    for metric in registered_metrics:
        model_value = model_values[metric]
        target = bootstrap[metric]
        standard_deviation = target["standard_deviation"]
        checks.append(
            {
                "metric": metric,
                "model": model_value,
                "human_bootstrap_mean": target["mean"],
                "human_bootstrap_95_interval": [target["lower"], target["upper"]],
                "standardized_discrepancy": (
                    None
                    if standard_deviation == 0.0
                    else (model_value - target["mean"]) / standard_deviation
                ),
                "inside_human_bootstrap_95_interval": (
                    target["lower"] <= model_value <= target["upper"]
                ),
            }
        )
    return {
        "passed": all(check["inside_human_bootstrap_95_interval"] for check in checks),
        "rule": "all registered scalar metrics inside human bootstrap 95% intervals",
        "checks": checks,
        "model_beta_pair_class_counts": summary["beta_pair_class_counts_analysis"],
        "human_published_beta_pair_class_counts": human["combined"][
            "published_figure_checks"
        ]["beta_pair_class_counts"],
    }


def _control_summary(behavior: dict) -> dict:
    summary = behavior["summary"]
    eligible = summary["eligible_subjects"]
    return {
        "overall_accuracy": summary["overall_accuracy"],
        "nonlearned_accuracy": summary["nonlearned_accuracy"],
        "self_consistent_incorrect_proportion": summary["ranking_class_counts"][
            "self_consistent_incorrect"
        ]
        / eligible,
        "self_inconsistent_proportion": summary["ranking_class_counts"][
            "self_inconsistent"
        ]
        / eligible,
        "mean_inter_subject_kendall_tau": summary["mean_inter_subject_kendall_tau"],
    }


def evaluate_confirmation_seed(
    seed: int,
    *,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    contract = validate_confirmation_contract(specification_path)
    if not contract["passed"]:
        raise RuntimeError("confirmation contract validation failed before evaluation")
    specification, paths = _specification_and_paths(specification_path)
    runtime = _require_registered_runtime(specification)
    _validate_seed(specification, seed)
    seed_dir = output_root / f"seed-{seed}"
    checkpoint = seed_dir / "net.dat"
    checkpoint_metadata = _validate_checkpoint(checkpoint, specification, seed)
    evaluation = specification["evaluation"]

    causal = run_causal_suite(
        checkpoint,
        batch_size=evaluation["batch_size"],
        cue_seed=evaluation["cue_seed"],
        support_seed=evaluation["support_seed"],
        order_seed=evaluation["order_seed"],
        order_schedules=evaluation["order_schedules"],
        cue_mode=evaluation["cue_mode"],
        subject_encoding_mode="stable_omission",
        subject_encoding_seed=evaluation["subject_encoding_seed"],
        protocol_path=paths["protocol"],
    )
    qualification = evaluate_qualification(causal, load_json(paths["qualification"]))
    write_json(seed_dir / "causal.json", causal)
    write_json(seed_dir / "qualification.json", qualification)

    controls = {}
    stable_behavior_path = seed_dir / "behavior-stable_omission.json"
    for mode in specification["matched_controls"]["modes"]:
        behavior = run_behavioral_analysis(
            checkpoint,
            batch_size=evaluation["batch_size"],
            cue_seed=evaluation["cue_seed"],
            support_seed=evaluation["support_seed"],
            subject_encoding_seed=evaluation["subject_encoding_seed"],
            choice_seed=evaluation["choice_seed"],
            temperature=evaluation["temperature"],
            subject_encoding_mode=mode,
            protocol_path=paths["protocol"],
        )
        behavior_path = seed_dir / f"behavior-{mode}.json"
        write_json(behavior_path, behavior)
        controls[mode] = _control_summary(behavior)

    stable_behavior = load_json(stable_behavior_path)
    human = load_json(paths["human_benchmark"])
    human_comparison = compare_behavior_to_human(
        stable_behavior,
        human,
        specification["behavioral_confirmation"]["metrics"],
    )
    geometry = run_geometry_analysis(
        checkpoint, stable_behavior_path, paths["geometry"]
    )
    algorithmic = run_algorithmic_comparison(
        checkpoint,
        stable_behavior_path,
        posterior_temperature=evaluation["posterior_temperature"],
    )
    write_json(seed_dir / "human-comparison.json", human_comparison)
    write_json(seed_dir / "geometry.json", geometry)
    write_json(seed_dir / "algorithmic.json", algorithmic)

    summary = {
        "confirmation_id": specification["confirmation_id"],
        "seed": seed,
        "device": DEVICE,
        "checkpoint": checkpoint_metadata["checkpoint"],
        "fixed_outer_steps": checkpoint_metadata["completed_outer_steps"],
        "fixed_temperature": evaluation["temperature"],
        "passes": {
            "causal_qualification": qualification["passed"],
            "behavioral_confirmation": human_comparison["passed"],
            "geometry": geometry["gate"]["passed"],
            "joint": (
                qualification["passed"]
                and human_comparison["passed"]
                and geometry["gate"]["passed"]
            ),
        },
        "stable_behavior": _control_summary(stable_behavior),
        "human_comparison": human_comparison,
        "matched_controls": {
            "results": controls,
            "identifiability_note": specification["matched_controls"][
                "identifiability_note"
            ],
        },
        "algorithmic": algorithmic["group"],
    }
    if runtime is not None:
        summary.update(
            {
                "execution_runtime": runtime,
                "execution_runtime_source": _formal_runtime_source_registration(),
                "training_execution": checkpoint_metadata.get("execution"),
                "training_execution_source": _formal_training_source_registration(),
            }
        )
    write_json(seed_dir / "confirmation.json", summary)
    return summary


def aggregate_confirmation(
    *,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    specification = load_json(specification_path)
    formal = specification["confirmation_id"] == FORMAL_CONFIRMATION_ID
    seeds = specification["training"]["seeds"]
    summaries = []
    missing = []
    for seed in seeds:
        path = output_root / f"seed-{seed}" / "confirmation.json"
        if not path.is_file():
            missing.append(seed)
        else:
            summary = load_json(path)
            if summary.get("seed") != seed:
                raise RuntimeError(f"confirmation result has wrong seed: {path}")
            if formal:
                _validate_formal_runtime_record(summary)
                _validate_formal_training_execution(summary)
            summaries.append(summary)
    if missing:
        raise RuntimeError(
            f"all registered seeds are mandatory; missing results for {missing}"
        )
    if (
        formal
        and len(
            {
                json.dumps(summary["execution_runtime"], sort_keys=True)
                for summary in summaries
            }
        )
        != 1
    ):
        raise RuntimeError("formal seeds used different runtime environments")

    pass_names = (
        "causal_qualification",
        "behavioral_confirmation",
        "geometry",
        "joint",
    )
    rates = {
        f"{name}_pass_proportion": float(
            np.mean([summary["passes"][name] for summary in summaries])
        )
        for name in pass_names
    }
    metric_paths = {
        "overall_accuracy": ("stable_behavior", "overall_accuracy"),
        "nonlearned_accuracy": ("stable_behavior", "nonlearned_accuracy"),
        "self_consistent_incorrect_proportion": (
            "stable_behavior",
            "self_consistent_incorrect_proportion",
        ),
        "geometry_pass": ("passes", "geometry"),
        "neural_map_proportion": ("algorithmic", "neural_map_proportion"),
        "closest_map_kendall_tau": (
            "algorithmic",
            "mean_closest_map_kendall_tau",
        ),
    }
    distributions = {}
    for name, path in metric_paths.items():
        values = [summary[path[0]][path[1]] for summary in summaries]
        distributions[name] = {
            "values_by_seed": dict(zip(map(str, seeds), values, strict=True)),
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "confirmation_id": specification["confirmation_id"],
        "status": "complete_all_declared_seeds_reported",
        "declared_seeds": seeds,
        "reported_seeds": [summary["seed"] for summary in summaries],
        "seed_filtering": False,
        "pass_rates": rates,
        "seed_distributions": distributions,
        "seed_summaries": summaries,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen confirmation workflow."
    )
    parser.add_argument(
        "action", choices=["validate", "train", "evaluate", "run-seed", "aggregate"]
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    if parsed.action == "validate":
        result = validate_confirmation_contract(parsed.specification)
    elif parsed.action == "aggregate":
        result = aggregate_confirmation(
            specification_path=parsed.specification,
            output_root=parsed.output_root,
        )
    else:
        if parsed.seed is None:
            raise ValueError(f"--seed is required for {parsed.action}")
        if parsed.action in {"train", "run-seed"}:
            checkpoint = train_confirmation_seed(
                parsed.seed,
                specification_path=parsed.specification,
                output_root=parsed.output_root,
            )
            result = {"seed": parsed.seed, "checkpoint": str(checkpoint)}
        if parsed.action in {"evaluate", "run-seed"}:
            result = evaluate_confirmation_seed(
                parsed.seed,
                specification_path=parsed.specification,
                output_root=parsed.output_root,
            )
    if parsed.output is not None:
        write_json(parsed.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
