"""Fresh-backbone replication of the frozen global-policy field fingerprint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    checkpoint_sha256,
    load_retro_checkpoint,
    load_training_provenance,
)
from fsrl.evaluation.qualification import evaluate_qualification
from fsrl.experiments.assembly.trajectory import readout_margin_fields
from fsrl.experiments.global_policy.amplitude_provenance import (
    NonInterpretableEstimate,
    posterior_descriptors,
)
from fsrl.experiments.global_policy.field_reassembly import (
    classify_status,
    field_reassembly_estimands,
    summarize_estimand,
)
from fsrl.experiments.global_policy.slope_localization import subject_slopes
from fsrl.experiments.local_fidelity.behavior_attribution import exact_probability
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import load_json, tensor_hashes, write_json
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import (
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import symbolic_distances
from fsrl.tasks.registered_protocol import load_ranking_protocol
from fsrl.training.backbone import (
    COMPILED_TRAINING_EXECUTION,
    MetaTrainConfig,
    train_meta_model,
)

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.lock.json"
)
DEFAULT_ARTIFACT_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.artifact_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "runs" / "global-policy-field-fingerprint-replication-v1"
)
DEFAULT_RESULT_PATH = resolve_record(
    "results/global_policy_field_fingerprint_replication_v1.json"
)

REPORTED_ESTIMANDS = (
    "S_NN",
    "S_PN",
    "S_NP",
    "S_PP",
    "D",
    "A",
    "R",
    "I",
    "Delta_A",
    "C_A",
    "Delta_R",
    "C_R",
    "S_tildePN",
    "Q_shape",
    "C_shape",
    "Q_amp",
)
DECISION_CONTRASTS = (
    "D",
    "A",
    "R",
    "I",
    "Delta_A",
    "C_A",
    "Delta_R",
    "C_R",
    "Q_shape",
    "C_shape",
    "Q_amp",
)
DIAGNOSTIC_ARRAYS = (
    "norm_g_N",
    "norm_g_P",
    "posterior_to_neural_scale_k",
    "norm_c_N",
    "norm_c_P",
    "norm_g_P_tilde",
    "a_N_bridge",
    "a_post_bridge",
    "Y_margin",
    "Y_bridge",
    "neural_field_reconstruction_error",
    "posterior_field_reconstruction_error",
    "neural_zero_sum_gauge_error",
    "posterior_zero_sum_gauge_error",
    "neural_residual_orthogonality_error",
    "posterior_residual_orthogonality_error",
    "norm_match_norm_error",
    "norm_match_scale_reconstruction_error",
    "norm_match_gradient_reconstruction_error",
    "norm_match_zero_sum_gauge_error",
    "norm_match_gradient_residual_error",
    "norm_match_field_reconstruction_error",
    "norm_match_residual_orthogonality_error",
    "norm_match_energy_error",
    "NN_probability_reconstruction_error",
    "PP_probability_reconstruction_error",
    "margin_field_interaction_error",
    "margin_I",
)
ALGEBRA_ERROR_ARRAYS = (
    "neural_field_reconstruction_error",
    "posterior_field_reconstruction_error",
    "neural_zero_sum_gauge_error",
    "posterior_zero_sum_gauge_error",
    "neural_residual_orthogonality_error",
    "posterior_residual_orthogonality_error",
    "norm_match_norm_error",
    "norm_match_scale_reconstruction_error",
    "norm_match_gradient_reconstruction_error",
    "norm_match_zero_sum_gauge_error",
    "norm_match_gradient_residual_error",
    "norm_match_field_reconstruction_error",
    "norm_match_residual_orthogonality_error",
    "norm_match_energy_error",
    "NN_probability_reconstruction_error",
    "PP_probability_reconstruction_error",
    "margin_field_interaction_error",
    "margin_I",
)
DOCUMENTATION_TRAINING_FIELDS = (
    "held_out_graph",
    "architecture",
    "checkpoint_selection",
    "local_gain_adaptation",
)
NORM_TOLERANCE = 1e-12
REQUIRED_IMPLEMENTATION_SOURCE_PATHS = {
    "replication_runner": "fsrl/global_policy_field_fingerprint_replication.py",
    "replication_tests": "tests/test_global_policy_field_fingerprint_replication.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}
REQUIRED_REUSED_SOURCE_PATHS = {
    "hash_validation": "fsrl/confirmation.py",
    "serialization_and_tensor_hashes": "fsrl/curvature_gate_pilot.py",
    "posterior_descriptor": "fsrl/global_policy_amplitude_provenance.py",
    "slope_estimator": "fsrl/global_policy_slope_localization.py",
    "exact_probability": "fsrl/local_behavior_attribution.py",
    "transitive_artifact_validation_import": "fsrl/dual_evidence_access_confirmation.py",
}


def _canonical_paths(parsed: argparse.Namespace) -> None:
    expected = {
        "specification": DEFAULT_SPECIFICATION_PATH,
        "implementation_lock": DEFAULT_IMPLEMENTATION_LOCK_PATH,
        "artifact_lock": DEFAULT_ARTIFACT_LOCK_PATH,
        "output_root": DEFAULT_OUTPUT_ROOT,
    }
    for name, canonical in expected.items():
        if getattr(parsed, name).resolve() != canonical.resolve():
            raise RuntimeError(f"formal workflow requires canonical {name}")
    result = parsed.result.resolve()
    if result != DEFAULT_RESULT_PATH.resolve():
        if parsed.stage != "evaluate":
            raise RuntimeError("result override is permitted only for evaluate replay")
        try:
            relative_result = result.relative_to(Path("/tmp").resolve())
        except ValueError as error:
            raise RuntimeError(
                "formal result override is permitted only under /tmp for replay"
            ) from error
        if not relative_result.parts:
            raise RuntimeError("formal replay result must be a file below /tmp")


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git freeze verification failed: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def require_pushed_freeze(paths: tuple[Path, ...]) -> dict:
    """Require canonical freeze files at clean shared-dev HEAD and origin/dev."""

    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_dev = _git("rev-parse", "origin/dev")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if branch != "dev" or head != origin_dev or status:
        raise RuntimeError(
            "formal execution requires clean dev with HEAD equal to origin/dev"
        )
    tracked = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError("freeze file is outside the repository") from error
        _git("ls-files", "--error-unmatch", str(relative))
        _git("diff", "--quiet", "HEAD", "--", str(relative))
        tracked.append(str(relative))
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "worktree_clean": True,
        "tracked_freeze_files": tracked,
    }


def mandatory_seeds(specification: dict) -> tuple[int, ...]:
    """Return the exact two registered fresh seeds, rejecting every variant."""

    seeds = tuple(
        int(seed) for seed in specification["network_seed_contract"]["mandatory_seeds"]
    )
    training_seeds = tuple(
        int(seed) for seed in specification["v1_backbone_training"]["seeds"]
    )
    if seeds != (2106, 2107) or training_seeds != seeds:
        raise RuntimeError(
            "the frozen replication requires exactly seeds 2106 and 2107"
        )
    if int(specification["network_seed_contract"]["mandatory_seed_count"]) != 2:
        raise RuntimeError("the frozen replication requires exactly two backbones")
    return seeds


def backbone_training_config(specification: dict, seed: int) -> MetaTrainConfig:
    """Construct the exact v1 training config after validating document-only fields."""

    registered = dict(specification["v1_backbone_training"])
    declared = tuple(int(value) for value in registered.pop("seeds"))
    if declared != mandatory_seeds(specification) or seed not in declared:
        raise ValueError(f"seed {seed} is not registered")
    expected_documentation = {
        "held_out_graph": "source-correct Liu graph and its rank-axis reflection",
        "architecture": "unaltered v1 RetroModulRNN",
        "checkpoint_selection": (
            "final registered step only; intermediate checkpoints are never evaluated"
        ),
        "local_gain_adaptation": (
            "not_applicable_and_forbidden_because_the_primary_condition_is_pure_L_off"
        ),
    }
    for name in DOCUMENTATION_TRAINING_FIELDS:
        observed = registered.pop(name)
        if observed != expected_documentation[name]:
            raise RuntimeError(f"frozen training documentation mismatch: {name}")
    training = MetaTrainConfig(seed=seed, **registered)
    if training.save_every != training.outer_steps:
        raise RuntimeError("only the final registered checkpoint may be saved")
    return training


def seed_paths(output_root: Path, seed: int) -> dict[str, Path]:
    backbone = output_root / f"seed-{seed}" / "backbone"
    return {
        "backbone_dir": backbone,
        "checkpoint": backbone / "net.dat",
        "backbone_config": backbone / "config.json",
        "backbone_log": backbone / "train_log.jsonl",
        "backbone_manifest": backbone / "replication_manifest.json",
    }


def _source_freeze_record(
    specification_path: Path, implementation_lock_path: Path
) -> dict:
    return {
        "replication_specification": {
            "path": str(specification_path.resolve().relative_to(ROOT)),
            "sha256": file_sha256(specification_path),
        },
        "implementation_lock": {
            "path": str(implementation_lock_path.resolve().relative_to(ROOT)),
            "sha256": file_sha256(implementation_lock_path),
        },
    }


def _validate_output_members(
    output_root: Path, seeds: tuple[int, ...], *, allow_partial: bool
) -> None:
    if output_root.is_symlink():
        raise RuntimeError("registered output root may not be a symlink")
    if not output_root.exists():
        if allow_partial:
            return
        raise RuntimeError("registered output root does not exist")
    observed = {path.name for path in output_root.iterdir()}
    expected = {f"seed-{seed}" for seed in seeds}
    if (allow_partial and not observed.issubset(expected)) or (
        not allow_partial and observed != expected
    ):
        raise RuntimeError("registered output root has extra or missing seed paths")
    if any(path.is_symlink() for path in output_root.iterdir()):
        raise RuntimeError("registered seed paths may not be symlinks")


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    """Validate registered files without loading discovery result contents."""

    specification = load_json(specification_path)
    implementation_lock = load_json(implementation_lock_path)
    locked_implementation = implementation_lock.get("implementation_sources", {})
    locked_reused = implementation_lock.get("reused_frozen_sources", {})
    required_groups = (
        (locked_implementation, REQUIRED_IMPLEMENTATION_SOURCE_PATHS),
        (locked_reused, REQUIRED_REUSED_SOURCE_PATHS),
    )
    for registrations, required_paths in required_groups:
        if set(registrations) != set(required_paths):
            raise RuntimeError("implementation source lock has missing or extra keys")
        for name, required_path in required_paths.items():
            if Path(registrations[name].get("path", "")) != Path(required_path):
                raise RuntimeError(f"implementation source path mismatch: {name}")
    registrations = {
        **specification["registered_sources"],
        "replication_specification": {
            "path": str(specification_path.resolve()),
            "sha256": implementation_lock["replication_specification_sha256"],
        },
        **locked_implementation,
        **locked_reused,
    }
    checks = []
    for name, registration in registrations.items():
        path = resolve_registered_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": registration["sha256"],
                "passed": observed == registration["sha256"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(
            f"field-fingerprint replication source lock failed: {checks}"
        )
    return {"passed": True, "checks": checks, "lock": implementation_lock}


def _read_training_log(path: Path, outer_steps: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            record = json.loads(line)
            if int(record.get("outer_step", -1)) != line_number:
                raise RuntimeError("backbone training log is not complete and ordered")
            numeric = (
                record.get("loss"),
                record.get("query_cross_entropy"),
                record.get("query_accuracy"),
                record.get("mean_abs_fast_weight"),
                record.get("n_edges"),
            )
            if not all(
                value is not None and math.isfinite(float(value)) for value in numeric
            ):
                raise RuntimeError("backbone training log contains a nonfinite value")
            records.append(record)
    if len(records) != outer_steps:
        raise RuntimeError("backbone training log has the wrong number of records")
    return records


def validate_complete_backbone(
    specification: dict,
    output_root: Path,
    seed: int,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> Path:
    """Fail closed unless one seed has the complete registered final artifact set."""

    paths = seed_paths(output_root, seed)
    seed_dir = paths["backbone_dir"].parent
    if (
        seed_dir.is_symlink()
        or paths["backbone_dir"].is_symlink()
        or not seed_dir.is_dir()
        or {path.name for path in seed_dir.iterdir()} != {"backbone"}
    ):
        raise RuntimeError(f"seed {seed} contains non-backbone artifacts")
    expected_backbone_members = {
        "net.dat",
        "config.json",
        "train_log.jsonl",
        "replication_manifest.json",
    }
    if {path.name for path in paths["backbone_dir"].iterdir()} != (
        expected_backbone_members
    ):
        raise RuntimeError(f"seed {seed} backbone artifact set is not exact")
    if any(path.is_symlink() for path in paths["backbone_dir"].iterdir()):
        raise RuntimeError(f"seed {seed} backbone artifacts may not be symlinks")
    required = tuple(
        paths[name]
        for name in (
            "checkpoint",
            "backbone_config",
            "backbone_log",
            "backbone_manifest",
        )
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"seed {seed} backbone artifact set is incomplete")
    training = backbone_training_config(specification, seed)
    expected_training = asdict(training)
    metadata = load_json(paths["backbone_config"])
    if metadata.get("training") != expected_training:
        raise RuntimeError(f"seed {seed} backbone training configuration mismatch")
    if int(metadata.get("completed_outer_steps", -1)) != training.outer_steps:
        raise RuntimeError(f"seed {seed} backbone is not the final registered step")
    if metadata.get("execution") != COMPILED_TRAINING_EXECUTION:
        raise RuntimeError(f"seed {seed} compiler execution contract mismatch")
    observed_checkpoint_hash = checkpoint_sha256(paths["checkpoint"])
    if metadata.get("checkpoint", {}).get("sha256") != observed_checkpoint_hash:
        raise RuntimeError(f"seed {seed} checkpoint hash mismatch in config")
    _read_training_log(paths["backbone_log"], training.outer_steps)
    manifest = load_json(paths["backbone_manifest"])
    expected_source_freeze = _source_freeze_record(
        specification_path, implementation_lock_path
    )
    if (
        manifest.get("replication_id") != specification["replication_id"]
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("training") != expected_training
        or manifest.get("checkpoint", {}).get("sha256") != observed_checkpoint_hash
        or Path(manifest.get("checkpoint", {}).get("path", "")).resolve()
        != paths["checkpoint"].resolve()
        or manifest.get("source_freeze") != expected_source_freeze
    ):
        raise RuntimeError(f"seed {seed} backbone manifest mismatch")
    runtime = manifest.get("runtime", {})
    if not (
        runtime.get("active") is True
        and runtime.get("cuda_available") is True
        and runtime.get("device") == "cuda"
        and int(runtime.get("torch_intraop_threads", -1)) == 1
        and int(runtime.get("torch_interop_threads", -1)) == 1
    ):
        raise RuntimeError(f"seed {seed} backbone runtime provenance mismatch")
    return paths["checkpoint"]


def train_backbone(
    specification: dict,
    output_root: Path,
    seed: int,
    runtime: dict,
    specification_path: Path,
    implementation_lock_path: Path,
) -> Path:
    """Train one registered final backbone, or validate an already complete one."""

    paths = seed_paths(output_root, seed)
    seed_dir = paths["backbone_dir"].parent
    if seed_dir.exists() and (
        seed_dir.is_symlink()
        or any(
            path.name != "backbone" or path.is_symlink() for path in seed_dir.iterdir()
        )
    ):
        raise RuntimeError(f"seed {seed} has invalid pre-existing members")
    if paths["backbone_dir"].exists():
        return validate_complete_backbone(
            specification,
            output_root,
            seed,
            specification_path,
            implementation_lock_path,
        )
    training = backbone_training_config(specification, seed)
    train_meta_model(training, paths["backbone_dir"], compile_model=True)
    manifest = {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "seed": seed,
        "runtime": runtime,
        "training": asdict(training),
        "source_freeze": _source_freeze_record(
            specification_path, implementation_lock_path
        ),
        "checkpoint": {
            "path": str(paths["checkpoint"].resolve()),
            "sha256": checkpoint_sha256(paths["checkpoint"]),
        },
    }
    write_json(paths["backbone_manifest"], manifest)
    return validate_complete_backbone(
        specification,
        output_root,
        seed,
        specification_path,
        implementation_lock_path,
    )


def train_artifacts(
    specification: dict,
    output_root: Path,
    runtime: dict,
    specification_path: Path,
    implementation_lock_path: Path,
) -> None:
    """Train all mandatory backbones before any Liu evaluator can be constructed."""

    seeds = mandatory_seeds(specification)
    _validate_output_members(output_root, seeds, allow_partial=True)
    for seed in seeds:
        train_backbone(
            specification,
            output_root,
            seed,
            runtime,
            specification_path,
            implementation_lock_path,
        )
    _validate_output_members(output_root, seeds, allow_partial=False)


def artifact_lock_document(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    output_root: Path,
) -> dict:
    """Create one backbone-only lock after both mandatory artifact sets exist."""

    artifacts = {}
    seeds = mandatory_seeds(specification)
    _validate_output_members(output_root, seeds, allow_partial=False)
    for seed in seeds:
        validate_complete_backbone(
            specification,
            output_root,
            seed,
            specification_path,
            implementation_lock_path,
        )
        paths = seed_paths(output_root, seed)
        artifacts[str(seed)] = {
            name: {
                "path": str(paths[name].resolve().relative_to(ROOT)),
                "sha256": file_sha256(paths[name]),
            }
            for name in (
                "checkpoint",
                "backbone_config",
                "backbone_log",
                "backbone_manifest",
            )
        }
        artifacts[str(seed)]["backbone_log"]["records"] = int(
            specification["v1_backbone_training"]["outer_steps"]
        )
    return {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "freeze_status": (
            "both_fresh_final_backbones_frozen_before_either_liu_evaluation"
        ),
        "replication_specification_sha256": file_sha256(specification_path),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "artifacts": artifacts,
        "mandatory_joint_freeze": (
            "Both complete backbone artifact sets existed before this lock was "
            "written; neither seed had been evaluated on Liu."
        ),
        "local_gain": "not_constructed_or_used",
        "next_step": (
            "Commit and push this joint lock before the one-command two-seed Liu "
            "fingerprint evaluation."
        ),
    }


def validate_artifacts(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    artifact_lock_path: Path,
    output_root: Path,
) -> dict:
    """Validate exact seed and artifact sets plus all registered hashes."""

    lock = load_json(artifact_lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("replication_id") != specification["replication_id"]
        or lock.get("freeze_status")
        != "both_fresh_final_backbones_frozen_before_either_liu_evaluation"
    ):
        raise RuntimeError("artifact lock identity or freeze status mismatch")
    expected_seed_keys = {str(seed) for seed in mandatory_seeds(specification)}
    if set(lock.get("artifacts", {})) != expected_seed_keys:
        raise RuntimeError("artifact lock must contain exactly seeds 2106 and 2107")
    if lock.get("local_gain") != "not_constructed_or_used":
        raise RuntimeError("local-gain artifacts are forbidden in this replication")
    checks = []
    top_level = {
        "replication_specification": (
            specification_path,
            lock["replication_specification_sha256"],
        ),
        "implementation_lock": (
            implementation_lock_path,
            lock["implementation_lock_sha256"],
        ),
    }
    for name, (path, expected) in top_level.items():
        observed = file_sha256(path)
        checks.append(
            {
                "name": name,
                "path": str(path.resolve().relative_to(ROOT)),
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )
    exact_artifact_names = {
        "checkpoint",
        "backbone_config",
        "backbone_log",
        "backbone_manifest",
    }
    seeds = mandatory_seeds(specification)
    _validate_output_members(output_root, seeds, allow_partial=False)
    for seed in seeds:
        validate_complete_backbone(
            specification,
            output_root,
            seed,
            specification_path,
            implementation_lock_path,
        )
        registrations = lock["artifacts"][str(seed)]
        if set(registrations) != exact_artifact_names:
            raise RuntimeError(f"seed {seed} artifact lock has the wrong keys")
        expected_paths = seed_paths(output_root, seed)
        for name, registration in registrations.items():
            path = resolve_registered_path(registration["path"])
            if path.resolve() != expected_paths[name].resolve():
                raise RuntimeError(f"seed {seed} {name} escaped the registered root")
            observed = file_sha256(path)
            checks.append(
                {
                    "name": f"seed_{seed}_{name}",
                    "path": str(path.relative_to(ROOT)),
                    "observed": observed,
                    "expected": registration["sha256"],
                    "passed": observed == registration["sha256"],
                }
            )
        if int(registrations["backbone_log"].get("records", -1)) != int(
            specification["v1_backbone_training"]["outer_steps"]
        ):
            raise RuntimeError(f"seed {seed} locked training-log length mismatch")
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"field-fingerprint artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _factorial_identity_errors(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "D_equals_A_plus_R": np.abs(values["D"] - values["A"] - values["R"]),
        "D_equals_Delta_A_plus_C_A": np.abs(
            values["D"] - values["Delta_A"] - values["C_A"]
        ),
        "D_equals_Delta_R_plus_C_R": np.abs(
            values["D"] - values["Delta_R"] - values["C_R"]
        ),
        "D_equals_Q_shape_plus_C_shape": np.abs(
            values["D"] - values["Q_shape"] - values["C_shape"]
        ),
        "I_equals_Delta_A_minus_C_R": np.abs(
            values["I"] - values["Delta_A"] + values["C_R"]
        ),
        "I_equals_Delta_R_minus_C_A": np.abs(
            values["I"] - values["Delta_R"] + values["C_A"]
        ),
        "Delta_A_equals_A_plus_half_I": np.abs(
            values["Delta_A"] - values["A"] - 0.5 * values["I"]
        ),
        "C_A_equals_R_minus_half_I": np.abs(
            values["C_A"] - values["R"] + 0.5 * values["I"]
        ),
        "Delta_R_equals_R_plus_half_I": np.abs(
            values["Delta_R"] - values["R"] - 0.5 * values["I"]
        ),
        "C_R_equals_A_minus_half_I": np.abs(
            values["C_R"] - values["A"] + 0.5 * values["I"]
        ),
        "Delta_A_equals_Q_shape_plus_Q_amp": np.abs(
            values["Delta_A"] - values["Q_shape"] - values["Q_amp"]
        ),
        "C_shape_equals_C_A_plus_Q_amp": np.abs(
            values["C_shape"] - values["C_A"] - values["Q_amp"]
        ),
    }


def _bootstrap_counts(specification: dict, seed: int, subjects: int) -> np.ndarray:
    bootstrap = specification["statistical_estimands"]["bootstrap"]
    return (
        np.random.default_rng(int(bootstrap["seeds"][str(seed)]))
        .multinomial(
            subjects,
            np.full(subjects, 1.0 / subjects),
            size=int(bootstrap["samples"]),
        )
        .astype(np.float64)
    )


def _statistics(
    specification: dict,
    seed: int,
    estimands: dict[str, np.ndarray],
    direct_endpoints: dict[str, np.ndarray],
) -> dict:
    subjects = len(np.asarray(estimands["D"]))
    counts = _bootstrap_counts(specification, seed, subjects)
    summaries = {
        name: summarize_estimand(np.asarray(estimands[name]), counts)
        for name in REPORTED_ESTIMANDS
    }
    direct_summaries = {
        name: summarize_estimand(np.asarray(values), counts)
        for name, values in direct_endpoints.items()
    }
    margin = float(specification["statistical_estimands"]["equivalence_margin"])
    statuses = {
        name: classify_status(summaries[name], margin) for name in DECISION_CONTRASTS
    }
    denominator = np.sum(counts, axis=1)
    bootstrap_values = {
        name: counts @ np.asarray(estimands[name], dtype=np.float64) / denominator
        for name in REPORTED_ESTIMANDS
    }
    participant_identity = {
        name: float(np.max(error))
        for name, error in estimands["factorial_identity_errors"].items()
    }
    bootstrap_identity = {
        name: float(np.max(error))
        for name, error in _factorial_identity_errors(bootstrap_values).items()
    }
    raw = {
        name: np.asarray(estimands[name], dtype=np.float64).tolist()
        for name in (*REPORTED_ESTIMANDS, *DIAGNOSTIC_ARRAYS)
    }
    raw.update(
        {
            name: np.asarray(values, dtype=np.float64).tolist()
            for name, values in direct_endpoints.items()
        }
    )
    raw.update(
        {
            f"factorial_{name}": np.asarray(error, dtype=np.float64).tolist()
            for name, error in estimands["factorial_identity_errors"].items()
        }
    )
    return {
        "summaries": summaries,
        "direct_endpoint_summaries": direct_summaries,
        "statuses": statuses,
        "raw_subject_level": raw,
        "integrity": {
            "bootstrap_samples": int(counts.shape[0]),
            "bootstrap_subjects": int(counts.shape[1]),
            "all_bootstrap_estimates_finite": all(
                summary["bootstrap"]["finite_samples"] == counts.shape[0]
                for summary in (*summaries.values(), *direct_summaries.values())
            ),
            "participant_factorial_identity_max_abs_errors": participant_identity,
            "bootstrap_factorial_identity_max_abs_errors": bootstrap_identity,
        },
    }


def _qualification(
    specification: dict,
    evaluator: FrozenFastWeightEvaluator,
    checkpoint_path: Path,
    checkpoint_hash: str,
    intact_fast_weights,
) -> dict:
    evaluation = specification["evaluation"]
    conditions = {}
    condition_winners = {}
    for intervention in FastWeightIntervention:
        metrics, winners = evaluator.condition_evaluation(intervention)
        conditions[intervention.value] = asdict(metrics)
        condition_winners[intervention.value] = winners
    intact_winners = condition_winners[FastWeightIntervention.INTACT.value]
    for intervention, winners_by_subject in condition_winners.items():
        agreements = []
        for subject, winners in enumerate(winners_by_subject):
            agreements.extend(
                int(winner == intact_winners[subject][pair])
                for pair, winner in winners.items()
            )
        conditions[intervention]["mean_pair_decision_agreement_to_intact"] = float(
            np.mean(agreements)
        )
    invariance = evaluator.order_invariance(
        intact_fast_weights,
        schedules=int(evaluation["order_schedules"]),
        seed=int(evaluation["order_seed"]),
    )
    causal = {
        "protocol_id": evaluator.protocol.protocol_id,
        "protocol_path": str(
            resolve_registered_path(
                specification["registered_sources"]["liu_protocol"]["path"]
            )
        ),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_hash,
        },
        "batch_size": int(evaluation["subjects"]),
        "cue_seed": int(evaluation["cue_seed"]),
        "cue_mode": str(evaluation["cue_mode"]),
        "subject_encoding": {
            "mode": str(evaluation["subject_encoding_mode"]),
            "seed": int(evaluation["subject_encoding_seed"]),
        },
        "training_provenance": load_training_provenance(
            checkpoint_path, checkpoint_hash
        ),
        "support_seed": int(evaluation["support_seed"]),
        "conditions": conditions,
        "order_invariance": asdict(invariance),
    }
    report = evaluate_qualification(
        causal,
        load_json(
            resolve_registered_path(
                specification["registered_sources"]["qualification"]["path"]
            )
        ),
    )
    report["causal_result"] = causal
    return report


def analyze_seed(
    specification: dict,
    seed: int,
    artifact_validation: dict,
) -> dict:
    """Evaluate one locked backbone without any old-seed result dependency."""

    evaluation = specification["evaluation"]
    artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
    checkpoint_path = resolve_registered_path(artifact["path"])
    backbone, model_config, checkpoint = load_retro_checkpoint(
        checkpoint_path, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    before = tensor_hashes(backbone)
    protocol = load_ranking_protocol(
        resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        )
    )
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocol,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
    )
    geometry = build_complete_graph_geometry(protocol)
    distances = symbolic_distances(protocol, geometry.pairs)
    nonlearned = np.asarray(
        [pair not in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    neural_margin = readout_margin_fields(evaluator, fast_weights, geometry)
    posterior, posterior_integrity = posterior_descriptors(
        evaluator, geometry, specification
    )
    posterior_margin = posterior["fields"]["same_unit_margin"]
    temperature = float(specification["posterior_comparator"]["choice_temperature"])
    estimands = field_reassembly_estimands(
        neural_margin,
        posterior_margin,
        geometry,
        distances,
        nonlearned,
        temperature,
    )

    correct_sign = np.asarray(geometry.true_sign, dtype=np.float64)[None, :]
    neural_direct_probability = exact_probability(
        correct_sign * neural_margin, temperature
    )
    posterior_direct_probability = 0.5 * (
        1.0 + correct_sign * posterior["fields"]["pair_probability_field"]
    )
    direct_endpoints = {
        "S_N_direct": subject_slopes(neural_direct_probability, distances, nonlearned),
        "S_P_direct": subject_slopes(
            posterior_direct_probability, distances, nonlearned
        ),
    }
    direct_endpoints["D_direct"] = (
        direct_endpoints["S_N_direct"] - direct_endpoints["S_P_direct"]
    )
    statistics = _statistics(specification, seed, estimands, direct_endpoints)
    qualification = _qualification(
        specification,
        evaluator,
        checkpoint_path,
        checkpoint.sha256,
        fast_weights,
    )
    after = tensor_hashes(backbone)

    tolerance = float(
        specification["competence_and_integrity_gates"][
            "field_probability_and_posterior_tolerance"
        ]
    )
    direct_identity_errors = {
        "NN_direct_probability_max_abs_error": float(
            np.max(np.abs(estimands["probabilities"]["NN"] - neural_direct_probability))
        ),
        "PP_direct_probability_max_abs_error": float(
            np.max(
                np.abs(estimands["probabilities"]["PP"] - posterior_direct_probability)
            )
        ),
        "S_NN_equals_S_N_direct": float(
            np.max(np.abs(estimands["S_NN"] - direct_endpoints["S_N_direct"]))
        ),
        "S_PP_equals_S_P_direct": float(
            np.max(np.abs(estimands["S_PP"] - direct_endpoints["S_P_direct"]))
        ),
        "D_equals_D_direct": float(
            np.max(np.abs(estimands["D"] - direct_endpoints["D_direct"]))
        ),
    }
    algebra_errors = {
        f"{name}_max_abs_error": float(np.max(np.abs(estimands[name])))
        for name in ALGEBRA_ERROR_ARRAYS
    }
    identity_errors = {
        **statistics["integrity"]["participant_factorial_identity_max_abs_errors"],
        **{
            f"bootstrap_{name}": value
            for name, value in statistics["integrity"][
                "bootstrap_factorial_identity_max_abs_errors"
            ].items()
        },
    }
    integrity = {
        **posterior_integrity,
        **direct_identity_errors,
        **algebra_errors,
        "factorial_identity_max_abs_errors": identity_errors,
        "minimum_neural_additive_norm": float(np.min(estimands["norm_g_N"])),
        "minimum_posterior_additive_norm": float(np.min(estimands["norm_g_P"])),
        "subjects": int(model_config.bs),
        "edges": len(geometry.pairs),
        "orientations": 2 * len(geometry.pairs),
        "nonlearned_pairs": int(np.sum(nonlearned)),
        "bootstrap_samples": statistics["integrity"]["bootstrap_samples"],
        "bootstrap_subjects": statistics["integrity"]["bootstrap_subjects"],
        "all_bootstrap_estimates_finite": statistics["integrity"][
            "all_bootstrap_estimates_finite"
        ],
        "backbone_tensor_hashes_unchanged": before == after,
    }
    posterior_error_names = (
        "posterior_inverse_link_max_abs_error",
        "posterior_orientation_reversal_max_abs_error",
        "posterior_expected_rank_Hodge_max_abs_error",
        "coverage_binary_max_abs_error",
        "coverage_relation_reuse_max_abs_error",
        "coverage_unique_fraction_max_abs_error",
    )
    integrity["passed"] = bool(
        qualification["passed"]
        and all(value <= tolerance for value in direct_identity_errors.values())
        and all(value <= tolerance for value in algebra_errors.values())
        and all(value <= tolerance for value in identity_errors.values())
        and all(integrity[name] <= tolerance for name in posterior_error_names)
        and integrity["minimum_neural_additive_norm"] > NORM_TOLERANCE
        and integrity["minimum_posterior_additive_norm"] > NORM_TOLERANCE
        and integrity["subjects"] == int(evaluation["subjects"]) == 77
        and integrity["edges"] == 28
        and integrity["orientations"] == 56
        and integrity["nonlearned_pairs"] == 20
        and integrity["bootstrap_samples"]
        == int(specification["statistical_estimands"]["bootstrap"]["samples"])
        and integrity["bootstrap_subjects"] == 77
        and integrity["all_bootstrap_estimates_finite"]
        and integrity["backbone_tensor_hashes_unchanged"]
    )
    return {
        "seed": seed,
        "checkpoint": {"path": artifact["path"], "sha256": checkpoint.sha256},
        "condition": "pure_L_off_intact_P_T",
        "qualification": qualification,
        "statistics": statistics,
        "integrity": integrity,
    }


def _gated_decision(outcome: str) -> dict:
    return {
        "outcome": outcome,
        "links": "not_evaluated",
        "secondary_boundaries": "not_evaluated",
        "network_population_inference": "not_performed",
    }


def _link_status(by_seed: dict[str, str], expected: str) -> str:
    values = tuple(by_seed.values())
    if all(value == expected for value in values):
        return "replicated"
    if (
        "unresolved" in values
        or sum(value == expected for value in values) == 1
        or len(set(values)) != 1
    ):
        return "heterogeneous_or_unresolved"
    return "not_replicated"


def _secondary_status(by_seed: dict[str, str]) -> str:
    values = tuple(by_seed.values())
    if all(value == "material_positive" for value in values):
        return "replicated_nonclosure"
    if all(value == "equivalent" for value in values):
        return "replicated_closure_boundary_shift"
    if "unresolved" in values or len(set(values)) != 1:
        return "heterogeneous_or_unresolved"
    return "same_resolved_other_status"


def cross_seed_decision(specification: dict, seeds: dict[str, dict]) -> dict:
    """Apply the frozen prerequisite, primary links, and secondary boundaries."""

    expected_seed_keys = {str(seed) for seed in mandatory_seeds(specification)}
    if set(seeds) != expected_seed_keys or not all(
        row["qualification"]["passed"] and row["integrity"]["passed"]
        for row in seeds.values()
    ):
        return _gated_decision("noninterpretable_competence_or_integrity_failure")
    if not all(
        float(row["statistics"]["summaries"]["D"]["bootstrap"]["lower95"]) > 0.0
        for row in seeds.values()
    ):
        return _gated_decision("premise_not_confirmed")

    expected = {
        "A": "material_positive",
        "Q_shape": "material_positive",
        "I": "material_negative",
    }
    primary_links = {}
    for name, expected_status in expected.items():
        by_seed = {
            seed: row["statistics"]["statuses"][name] for seed, row in seeds.items()
        }
        primary_links[name] = {
            "status": _link_status(by_seed, expected_status),
            "expected_status": expected_status,
            "by_seed": by_seed,
        }
    link_outcomes = tuple(row["status"] for row in primary_links.values())
    if all(outcome == "replicated" for outcome in link_outcomes):
        outcome = "replicated_field_fingerprint"
    elif "heterogeneous_or_unresolved" in link_outcomes:
        outcome = "heterogeneous_or_unresolved_fingerprint"
    else:
        outcome = "field_fingerprint_not_replicated"

    secondary_boundaries = {}
    for name in ("C_A", "C_shape"):
        by_seed = {
            seed: row["statistics"]["statuses"][name] for seed, row in seeds.items()
        }
        secondary_boundaries[name] = {
            "outcome": _secondary_status(by_seed),
            "by_seed": by_seed,
        }
    r_by_seed = {
        seed: row["statistics"]["statuses"]["R"] for seed, row in seeds.items()
    }
    secondary_boundaries["R"] = {
        "status": (
            next(iter(set(r_by_seed.values())))
            if len(set(r_by_seed.values())) == 1
            else "heterogeneous_or_unresolved"
        ),
        "primary_gate": False,
        "by_seed": r_by_seed,
    }
    return {
        "outcome": outcome,
        "anchor_D": {
            "outcome": "confirmed",
            "lower95_by_seed": {
                seed: row["statistics"]["summaries"]["D"]["bootstrap"]["lower95"]
                for seed, row in seeds.items()
            },
        },
        "links": primary_links,
        "secondary_boundaries": secondary_boundaries,
        "network_population_inference": "not_performed",
    }


def evaluate_replication(
    specification: dict,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    seeds = {}
    for seed in mandatory_seeds(specification):
        try:
            seeds[str(seed)] = analyze_seed(specification, seed, artifact_validation)
        except NonInterpretableEstimate as error:
            artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
            seeds[str(seed)] = {
                "seed": seed,
                "checkpoint": {
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                },
                "condition": "pure_L_off_intact_P_T",
                "qualification": {
                    "passed": False,
                    "not_evaluated_due_to_integrity_failure": True,
                },
                "statistics": {"summaries": {}, "statuses": {}},
                "integrity": {
                    "passed": False,
                    "failure_type": "registered_noninterpretable_estimate",
                    "error": str(error),
                },
            }
    return {
        "schema_version": 1,
        "replication_id": specification["replication_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "primary_condition": "pure_L_off_intact_P_T",
        "seeds": seeds,
        "decision": cross_seed_decision(specification, seeds),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen global-policy field fingerprint replication."
    )
    parser.add_argument(
        "stage", choices=("train-artifacts", "write-artifact-lock", "evaluate")
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument(
        "--artifact-lock", type=Path, default=DEFAULT_ARTIFACT_LOCK_PATH
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    _canonical_paths(parsed)
    runtime = require_formal_runtime()
    specification = load_json(parsed.specification)
    mandatory_seeds(specification)
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    if parsed.stage == "train-artifacts":
        require_pushed_freeze((parsed.specification, parsed.implementation_lock))
        if parsed.artifact_lock.exists() or parsed.result.exists():
            raise RuntimeError(
                "training cannot run after an artifact lock or result exists"
            )
        train_artifacts(
            specification,
            parsed.output_root,
            runtime,
            parsed.specification,
            parsed.implementation_lock,
        )
        return 0
    if parsed.stage == "write-artifact-lock":
        require_pushed_freeze((parsed.specification, parsed.implementation_lock))
        if parsed.artifact_lock.exists() or parsed.result.exists():
            raise RuntimeError("artifact lock/result already exists")
        write_json(
            parsed.artifact_lock,
            artifact_lock_document(
                specification,
                parsed.specification,
                parsed.implementation_lock,
                parsed.output_root,
            ),
        )
        return 0
    git_freeze = require_pushed_freeze(
        (parsed.specification, parsed.implementation_lock, parsed.artifact_lock)
    )
    artifact_validation = validate_artifacts(
        specification,
        parsed.specification,
        parsed.implementation_lock,
        parsed.artifact_lock,
        parsed.output_root,
    )
    if parsed.result.exists():
        raise RuntimeError("replication result already exists")
    result = evaluate_replication(
        specification, source_validation, artifact_validation, runtime
    )
    result["git_freeze_validation"] = git_freeze
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
