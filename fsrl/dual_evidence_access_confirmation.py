"""Fresh-backbone confirmation for the frozen dual-evidence-access v2.4 model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import fsrl.dual_evidence_access_pilot as dual_access

from .confirmation import file_sha256
from .conjunctive_local_trace_pilot import evaluate_pilot as evaluate_v2_3_pilot
from .conjunctive_local_trace_replication import (
    _attribution_for_seed,
    _seed_paths,
    _validate_complete_backbone,
    _validate_complete_gain,
    adapt_gain,
    seed_specification,
    train_backbone,
)
from .curvature_gate_pilot import configure_runtime, load_json, write_json
from .study_registry import registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/dual_evidence_access_confirmation_v2_4.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/dual_evidence_access_confirmation_v2_4.lock.json")
)
DEFAULT_ARTIFACT_LOCK_PATH = (
    resolve_record("benchmarks/dual_evidence_access_confirmation_v2_4.artifact_lock.json")
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "dual-evidence-access-confirmation-v2-4"
DEFAULT_RESULT_PATH = resolve_record("results/dual_evidence_access_confirmation_v2_4.json")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def fresh_seeds(specification: dict) -> tuple[int, ...]:
    seeds = tuple(
        int(seed) for seed in specification["network_seed_contract"]["mandatory_seeds"]
    )
    if seeds != (2104, 2105):
        raise RuntimeError("the frozen confirmation requires seeds 2104 and 2105")
    if (
        tuple(specification["development_seed_contract"]["mandatory_frozen_seeds"])
        != seeds
    ):
        raise RuntimeError("training and evaluation seed contracts disagree")
    return seeds


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    registrations = {
        **specification["registered_sources"],
        "confirmation_specification": {
            "path": str(specification_path.resolve()),
            "sha256": lock["confirmation_specification_sha256"],
        },
        **lock["implementation_sources"],
        **lock["reused_frozen_sources"],
    }
    checks = []
    for name, registration in registrations.items():
        path = _resolve(registration["path"])
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
        raise RuntimeError(f"dual-evidence confirmation source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def train_artifacts(
    specification: dict,
    output_root: Path,
    source_validation: dict,
    runtime: dict,
) -> None:
    for seed in fresh_seeds(specification):
        checkpoint = train_backbone(specification, output_root, seed, runtime)
        adapt_gain(
            specification,
            checkpoint,
            output_root,
            seed,
            source_validation,
            runtime,
        )


def artifact_lock_document(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    output_root: Path,
) -> dict:
    artifacts = {}
    for seed in fresh_seeds(specification):
        _validate_complete_backbone(specification, output_root, seed)
        gain_path = _validate_complete_gain(specification, output_root, seed)
        paths = _seed_paths(output_root, seed)
        gain = load_json(gain_path)
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
                "gain",
                "local_log",
            )
        }
        artifacts[str(seed)]["gain"]["lambda_L"] = gain["lambda_L"]
        artifacts[str(seed)]["backbone_log"]["records"] = specification[
            "v1_backbone_training"
        ]["outer_steps"]
        artifacts[str(seed)]["local_log"]["records"] = specification[
            "local_only_adaptation"
        ]["outer_steps"]
    return {
        "schema_version": 1,
        "confirmation_id": specification["confirmation_id"],
        "freeze_status": "both_fresh_backbones_and_gains_frozen_before_either_liu_evaluation",
        "confirmation_specification_sha256": file_sha256(specification_path),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "artifacts": artifacts,
        "mandatory_joint_freeze": "Both fresh artifact sets were complete before this lock was written; neither seed had been evaluated on Liu.",
        "next_step": "Commit and push this joint lock before the one-command two-seed Liu evaluation.",
    }


def validate_artifacts(
    specification: dict,
    specification_path: Path,
    implementation_lock_path: Path,
    artifact_lock_path: Path,
    output_root: Path,
) -> dict:
    lock = load_json(artifact_lock_path)
    checks = []
    top_level = {
        "confirmation_specification": (
            specification_path,
            lock["confirmation_specification_sha256"],
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
    for seed in fresh_seeds(specification):
        _validate_complete_backbone(specification, output_root, seed)
        _validate_complete_gain(specification, output_root, seed)
        for name, registration in lock["artifacts"][str(seed)].items():
            path = _resolve(registration["path"])
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
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"dual-evidence confirmation artifact lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def _v2_3_reference_seed(
    specification: dict,
    output_root: Path,
    seed: int,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    paths = _seed_paths(output_root, seed)
    pilot = evaluate_v2_3_pilot(
        seed_specification(specification, seed),
        paths["checkpoint"],
        paths["gain"],
        source_validation,
        artifact_validation,
        runtime,
    )
    attribution = _attribution_for_seed(
        specification, seed, paths["checkpoint"], paths["gain"]
    )
    return {
        "seed": seed,
        "checkpoint": pilot["checkpoint"],
        "gain_artifact": pilot["gain_artifact"],
        "original_v1_qualification": pilot["original_v1_qualification"],
        "local_fidelity": pilot["local_fidelity"],
        "behavior": pilot["behavior"],
        "query_binding": pilot["query_binding"],
        "terminal_projection": pilot["terminal_projection"],
        "legacy_v2_3_decision": pilot["decision"],
        "attribution": attribution,
    }


@contextmanager
def bind_fresh_artifacts(specification_path: Path, output_root: Path):
    original_specification = dual_access.V2_3_SPECIFICATION_PATH
    original_output = dual_access.V2_3_OUTPUT_ROOT
    dual_access.V2_3_SPECIFICATION_PATH = specification_path
    dual_access.V2_3_OUTPUT_ROOT = output_root
    try:
        yield
    finally:
        dual_access.V2_3_SPECIFICATION_PATH = original_specification
        dual_access.V2_3_OUTPUT_ROOT = original_output


def confirmation_decision(specification: dict, seeds: dict[str, dict]) -> dict:
    decision = dual_access.cross_seed_decision(specification, seeds)
    v2_4_outcome = decision["outcome"]
    return {
        **decision,
        "outcome": (
            "fresh_backbone_confirmation_pass"
            if decision["all_four_links_pass"]
            else v2_4_outcome
        ),
        "v2_4_rule_outcome": v2_4_outcome,
    }


def evaluate_confirmation(
    specification: dict,
    specification_path: Path,
    output_root: Path,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    reference_seeds = {
        str(seed): _v2_3_reference_seed(
            specification,
            output_root,
            seed,
            source_validation,
            artifact_validation,
            runtime,
        )
        for seed in fresh_seeds(specification)
    }
    frozen_reference = {"seeds": reference_seeds}
    with bind_fresh_artifacts(specification_path, output_root):
        seeds = {
            str(seed): dual_access.evaluate_seed(
                specification,
                seed,
                frozen_reference,
                artifact_validation,
                runtime,
            )
            for seed in fresh_seeds(specification)
        }
    return {
        "schema_version": 1,
        "confirmation_id": specification["confirmation_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "v2_3_references": reference_seeds,
        "seeds": seeds,
        "decision": confirmation_decision(specification, seeds),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the fresh-backbone dual-evidence-access v2.4 confirmation."
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
    runtime = configure_runtime()
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    specification = load_json(parsed.specification)
    if parsed.stage == "train-artifacts":
        train_artifacts(specification, parsed.output_root, source_validation, runtime)
        return 0
    if parsed.stage == "write-artifact-lock":
        if parsed.artifact_lock.exists():
            raise RuntimeError("confirmation artifact lock already exists")
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
    artifact_validation = validate_artifacts(
        specification,
        parsed.specification,
        parsed.implementation_lock,
        parsed.artifact_lock,
        parsed.output_root,
    )
    result = evaluate_confirmation(
        specification,
        parsed.specification,
        parsed.output_root,
        source_validation,
        artifact_validation,
        runtime,
    )
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
