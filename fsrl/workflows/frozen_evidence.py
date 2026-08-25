"""Verify, summarize, and selectively replay the Liu v1 evidence overlay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath
from typing import Any

from fsrl.infra.git_provenance import git_blob_sha256, verify_git_registrations
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import (
    SYNTHESIS_ROOT,
    registered_file_sha256,
    resolve_record,
)
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
MAINLINE_ROOT = SYNTHESIS_ROOT / "snapshots" / "reporting_v1"
MANIFEST_PATH = MAINLINE_ROOT / "manifest.json"
ARTIFACTS_PATH = MAINLINE_ROOT / "artifacts.json"
ENVIRONMENT_PATH = MAINLINE_ROOT / "environment.json"
REPORT_VIEW_PATH = MAINLINE_ROOT / "report_view.json"

EXPECTED_DAG = {
    "task_fidelity": [],
    "behavioral_competence": ["task_fidelity"],
    "global_reassembly": ["task_fidelity"],
    "local_direct_fidelity": ["task_fidelity"],
    "algorithmic_asymmetry": ["global_reassembly", "local_direct_fidelity"],
    "structural_transport": ["global_reassembly", "local_direct_fidelity"],
    "claim_freeze": [
        "behavioral_competence",
        "global_reassembly",
        "local_direct_fidelity",
        "algorithmic_asymmetry",
        "structural_transport",
    ],
}
REQUIRED_PRINCIPLES = {
    "historical_experiment_is_not_mainline_orchestration_or_presentation",
    "motivated_by_is_not_depends_on",
    "historical_replay_is_not_live_head_validation",
}


def canonical_manifest_payload_sha256(manifest: dict) -> str:
    """Hash the manifest payload without its one-time freeze attestations."""

    payload = json.loads(json.dumps(manifest))
    payload["status"] = "draft"
    payload["freeze"]["validated_candidate_commit"] = None
    payload["freeze"]["canonical_payload_sha256"] = None
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_repo_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise RuntimeError(f"mainline path must be repository-relative: {value}")
    return resolve_record(Path(*pure.parts))


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise RuntimeError(f"invalid JSON pointer: {pointer}")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise TypeError(f"JSON pointer crosses a scalar: {pointer}")
    return value


def validate_manifest_structure(manifest: dict) -> dict:
    if manifest.get("schema_version") != 1 or manifest.get("mainline_id") != "liu_v1":
        raise RuntimeError("Liu mainline manifest identity mismatch")
    if manifest.get("status") not in {"draft", "frozen"}:
        raise RuntimeError("Liu mainline lifecycle status must be draft or frozen")
    if set(manifest.get("design_principles", [])) != REQUIRED_PRINCIPLES:
        raise RuntimeError("Liu mainline design principles changed")
    nodes = manifest.get("claim_nodes", {})
    if set(nodes) != set(EXPECTED_DAG):
        raise RuntimeError("Liu mainline claim-node set changed")
    observed_dag = {name: node.get("depends_on") for name, node in nodes.items()}
    if observed_dag != EXPECTED_DAG:
        raise RuntimeError("Liu mainline claim DAG changed")
    for name, node in nodes.items():
        motivated = node.get("motivated_by", [])
        if any(value in node["depends_on"] for value in motivated):
            raise RuntimeError(f"motivated_by overlaps depends_on for {name}")
        if not node.get("claim") or not node.get("evidence"):
            raise RuntimeError(f"claim node is incomplete: {name}")
    replay_stages = manifest.get("replay_stages", {})
    executions = manifest.get("execution_records", {})
    if not replay_stages or not executions:
        raise RuntimeError("mainline execution registry is empty")
    if any(record not in executions for record in replay_stages.values()):
        raise RuntimeError("replay stage references an unknown execution record")
    if "all" in replay_stages:
        raise RuntimeError("aggregate replay is forbidden")
    freeze = manifest.get("freeze", {})
    if freeze.get("freeze_ref") != "refs/tags/liu-mainline-v1":
        raise RuntimeError("Liu mainline freeze ref changed")
    candidate = freeze.get("validated_candidate_commit")
    payload_hash = freeze.get("canonical_payload_sha256")
    if manifest["status"] == "draft":
        if candidate is not None or payload_hash is not None:
            raise RuntimeError("draft mainline contains freeze attestations")
    else:
        if (
            not isinstance(candidate, str)
            or re.fullmatch(r"[0-9a-f]{40}", candidate) is None
        ):
            raise RuntimeError("frozen mainline has an invalid candidate commit")
        if payload_hash != canonical_manifest_payload_sha256(manifest):
            raise RuntimeError("frozen mainline canonical payload hash mismatch")
    return {
        "passed": True,
        "claim_nodes": len(nodes),
        "dag_edges": sum(len(parents) for parents in EXPECTED_DAG.values()),
        "execution_records": len(executions),
        "replay_stages": len(replay_stages),
    }


def verify_evidence_files(manifest: dict) -> dict:
    registrations = manifest.get("evidence_files", {})
    if not registrations:
        raise RuntimeError("mainline evidence-file registry is empty")
    checks = []
    for name, registration in registrations.items():
        path = _safe_repo_path(registration.get("path", ""))
        registered_path = PurePosixPath(registration["path"])
        historical_python = registered_path.suffix == ".py" and registered_path.parts[
            :1
        ] in {("fsrl",), ("tests",)}
        if (not historical_python and not path.is_file()) or path.is_symlink():
            raise RuntimeError(f"mainline evidence file is unavailable: {name}")
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        check = {
            "name": name,
            "path": str(path.relative_to(ROOT)),
            "expected": registration.get("sha256"),
            "observed": observed,
            "passed": observed == registration.get("sha256"),
        }
        checks.append(check)
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"mainline evidence hash verification failed: {checks}")
    return {"passed": True, "checks": checks}


def verify_freeze_attestation(manifest: dict) -> dict:
    if manifest["status"] == "draft":
        return {"passed": True, "status": "draft", "candidate_commit": None}
    candidate = manifest["freeze"]["validated_candidate_commit"]
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{candidate}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"validated candidate commit is unavailable: {candidate}")
    return {"passed": True, "status": "frozen", "candidate_commit": candidate}


def verify_historical_executions(manifest: dict) -> dict:
    files = manifest["evidence_files"]
    validations = []
    for record_id, record in manifest["execution_records"].items():
        commit = record["execution_commit"]
        historical_ids = record.get("historical_files", [])
        historical_checks = []
        for file_id in historical_ids:
            registration = files[file_id]
            observed = git_blob_sha256(ROOT, commit, registration["path"])
            historical_checks.append(
                {
                    "file_id": file_id,
                    "path": registration["path"],
                    "expected": registration["sha256"],
                    "observed": observed,
                    "passed": observed == registration["sha256"],
                }
            )
        if not all(check["passed"] for check in historical_checks):
            raise RuntimeError(
                f"historical execution file mismatch for {record_id}: "
                f"{historical_checks}"
            )
        lock = load_json(_safe_repo_path(files[record["execution_lock"]]["path"]))
        registrations = {}
        for group in record.get("historical_source_groups", []):
            values = lock.get(group)
            if not isinstance(values, dict) or not values:
                raise RuntimeError(
                    f"historical source group is absent for {record_id}: {group}"
                )
            registrations.update(
                {f"{group}:{name}": value for name, value in values.items()}
            )
        source_validation = verify_git_registrations(ROOT, commit, registrations)
        validations.append(
            {
                "record_id": record_id,
                "execution_commit": commit,
                "historical_files": historical_checks,
                "source_checks": source_validation["checks"],
                "passed": True,
            }
        )
    return {"passed": True, "executions": validations}


def _tar(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["tar", "--zstd", *args],
        cwd=ROOT,
        check=False,
        capture_output=capture,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"artifact bundle operation failed: {detail}")
    return completed


def verify_artifact_bundle(artifacts: dict | None = None) -> dict:
    artifacts = load_json(ARTIFACTS_PATH) if artifacts is None else artifacts
    if artifacts.get("schema_version") != 1:
        raise RuntimeError("artifact registry schema mismatch")
    bundle = artifacts.get("bundle", {})
    bundle_path = _safe_repo_path(bundle.get("path", ""))
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise RuntimeError("artifact bundle is unavailable")
    if bundle_path.stat().st_size != bundle.get("size"):
        raise RuntimeError("artifact bundle size mismatch")
    if file_sha256(bundle_path) != bundle.get("sha256"):
        raise RuntimeError("artifact bundle hash mismatch")
    listing = _tar(["-tf", str(bundle_path)]).stdout.decode("utf-8").splitlines()
    members = artifacts.get("members", [])
    registered_names = [member["bundle_member"] for member in members]
    expected_count = bundle.get("member_count")
    if (
        not isinstance(expected_count, int)
        or expected_count <= 0
        or len(members) != expected_count
        or len(set(registered_names)) != len(registered_names)
    ):
        raise RuntimeError("artifact member registry count or uniqueness mismatch")
    if sorted(listing) != sorted(registered_names):
        raise RuntimeError("artifact bundle member set mismatch")
    checks = []
    for member in members:
        name = member["bundle_member"]
        payload = _tar(["-xOf", str(bundle_path), name]).stdout
        observed = hashlib.sha256(payload).hexdigest()
        checks.append(
            {
                "logical_name": member["logical_name"],
                "bundle_member": name,
                "expected": member["sha256"],
                "observed": observed,
                "size": len(payload),
                "passed": observed == member["sha256"]
                and len(payload) == member["size"],
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"artifact member verification failed: {checks}")
    if sum(check["size"] for check in checks) != bundle["uncompressed_member_bytes"]:
        raise RuntimeError("artifact uncompressed byte ledger mismatch")
    return {
        "passed": True,
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundle_sha256": bundle["sha256"],
        "members": len(checks),
        "uncompressed_member_bytes": sum(check["size"] for check in checks),
        "checks": checks,
    }


def _assert_semantic(document: dict, assertion: dict) -> dict:
    observed = json_pointer(document, assertion["json_pointer"])
    operator = assertion.get("operator", "equals")
    expected = assertion.get("expected")
    if operator == "equals":
        passed = observed == expected
    elif operator == "is_true":
        passed = observed is True
        expected = True
    elif operator == "less_equal":
        passed = observed <= expected
    elif operator == "greater_equal":
        passed = observed >= expected
    else:
        raise RuntimeError(f"unsupported semantic assertion operator: {operator}")
    return {
        "json_pointer": assertion["json_pointer"],
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def verify_replay_contracts(manifest: dict) -> dict:
    files = manifest["evidence_files"]
    validations = []
    for record_id, record in manifest["execution_records"].items():
        result_registration = files[record["result"]]
        policy = record["replay_policy"]
        if policy["exact"]["expected_sha256"] != result_registration["sha256"]:
            raise RuntimeError(f"exact replay hash is not bound to result: {record_id}")
        document = load_json(_safe_repo_path(result_registration["path"]))
        assertions = [
            _assert_semantic(document, assertion)
            for assertion in policy["semantic"]["assertions"]
        ]
        if not assertions or not all(assertion["passed"] for assertion in assertions):
            raise RuntimeError(f"semantic replay contract failed: {record_id}")
        validations.append(
            {"record_id": record_id, "passed": True, "assertions": assertions}
        )
    return {"passed": True, "records": validations}


def verify_report_view(manifest: dict, view: dict | None = None) -> dict:
    view = load_json(REPORT_VIEW_PATH) if view is None else view
    if view.get("schema_version") != 1:
        raise RuntimeError("report-view schema mismatch")
    if set(view.get("claim_order", [])) != set(manifest["claim_nodes"]):
        raise RuntimeError("report view omits a claim node")
    files = manifest["evidence_files"]
    cache = {}
    checks = []
    figures = view.get("figures", [])
    if [figure["id"] for figure in figures] != view.get("table_order"):
        raise RuntimeError("report figure and table order differ")
    if len(figures) != 4:
        raise RuntimeError("Liu mainline requires exactly four report figures")
    for figure in figures:
        for metric in figure.get("metrics", []):
            source_id = metric["source_id"]
            registration = files.get(source_id)
            if registration is None:
                raise RuntimeError(f"report metric has unknown source: {source_id}")
            if source_id not in cache:
                cache[source_id] = load_json(_safe_repo_path(registration["path"]))
            value = json_pointer(cache[source_id], metric["json_pointer"])
            checks.append(
                {
                    "figure_id": figure["id"],
                    "label": metric["label"],
                    "source_id": source_id,
                    "json_pointer": metric["json_pointer"],
                    "value_type": type(value).__name__,
                    "passed": True,
                }
            )
    return {
        "passed": True,
        "figures": len(figures),
        "metrics": len(checks),
        "checks": checks,
    }


def verify_mainline() -> dict:
    manifest = load_json(MANIFEST_PATH)
    result = {
        "schema": validate_manifest_structure(manifest),
        "freeze": verify_freeze_attestation(manifest),
        "evidence": verify_evidence_files(manifest),
        "historical_executions": verify_historical_executions(manifest),
        "artifacts": verify_artifact_bundle(),
        "replay_contracts": verify_replay_contracts(manifest),
        "report_view": verify_report_view(manifest),
    }
    result["passed"] = True
    return result


def _format_value(value: Any, format_spec: str) -> str:
    if format_spec == "s":
        return str(value)
    if format_spec == "d":
        return format(int(value), "d")
    return format(value, format_spec)


def _write_exclusive(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def _svg_figure(figure: dict, rows: list[dict]) -> str:
    width, height = 1200, 720
    card_width, card_height = 540, 125
    colors = ["#E8F1F8", "#F3ECE4", "#E9F3EC", "#F5EAF0"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="720" fill="#FAFAF8"/>',
        '<text x="60" y="70" font-family="DejaVu Sans, sans-serif" font-size="32" font-weight="700" fill="#1F2933">'
        + html.escape(figure["title"])
        + "</text>",
        '<text x="60" y="104" font-family="DejaVu Sans, sans-serif" font-size="15" fill="#52606D">Liu Mainline v1 · frozen-result view</text>',
    ]
    for index, row in enumerate(rows):
        column = index % 2
        line = index // 2
        x = 60 + column * 570
        y = 135 + line * 140
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="12" fill="{color}" stroke="#CBD5E1"/>'
        )
        label_lines = textwrap.wrap(row["label"], width=48) or [row["label"]]
        for offset, label in enumerate(label_lines[:2]):
            parts.append(
                f'<text x="{x + 24}" y="{y + 30 + offset * 19}" font-family="DejaVu Sans, sans-serif" font-size="15" fill="#334E68">{html.escape(label)}</text>'
            )
        value_lines = textwrap.wrap(row["display"], width=54) or [row["display"]]
        for offset, value in enumerate(value_lines[:2]):
            parts.append(
                f'<text x="{x + 24}" y="{y + 78 + offset * 21}" font-family="DejaVu Sans Mono, monospace" font-size="18" font-weight="700" fill="#102A43">{html.escape(value)}</text>'
            )
        provenance = f"{row['source_id']}{row['json_pointer']}"
        parts.append(
            f'<text x="{x + 24}" y="{y + 114}" font-family="DejaVu Sans Mono, monospace" font-size="9" fill="#627D98">{html.escape(provenance[:86])}</text>'
        )
    parts.append(
        '<text x="60" y="694" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#7B8794">Generated only from frozen JSON fields; no checkpoint load, resampling, or model execution.</text>'
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def summarize_mainline(output_dir: Path) -> dict:
    """Create report tables and SVGs from frozen JSON fields only."""

    manifest = load_json(MANIFEST_PATH)
    validate_manifest_structure(manifest)
    view = load_json(REPORT_VIEW_PATH)
    verify_report_view(manifest, view)
    files = manifest["evidence_files"]
    cache = {}
    figures = []
    table_rows = []
    for figure in view["figures"]:
        rows = []
        for metric in figure["metrics"]:
            source_id = metric["source_id"]
            if source_id not in cache:
                cache[source_id] = load_json(_safe_repo_path(files[source_id]["path"]))
            value = json_pointer(cache[source_id], metric["json_pointer"])
            row = {
                "figure_id": figure["id"],
                "label": metric["label"],
                "value": value,
                "display": _format_value(value, metric["format"]),
                "source_id": source_id,
                "source_path": files[source_id]["path"],
                "source_sha256": files[source_id]["sha256"],
                "json_pointer": metric["json_pointer"],
            }
            rows.append(row)
            table_rows.append(row)
        svg_name = f"{figure['id']}.svg"
        figures.append(
            {
                "id": figure["id"],
                "title": figure["title"],
                "claim_nodes": figure["claim_nodes"],
                "asset": svg_name,
                "metrics": rows,
            }
        )
    summary = {
        "schema_version": 1,
        "mainline_id": manifest["mainline_id"],
        "manifest_status": manifest["status"],
        "generation_contract": "frozen_json_field_extraction_only",
        "figures": figures,
    }
    targets = [
        output_dir / "liu_mainline_summary.json",
        output_dir / "liu_mainline_tables.csv",
        *[output_dir / figure["asset"] for figure in figures],
    ]
    if any(target.exists() or target.is_symlink() for target in targets):
        raise RuntimeError("summarize refuses to overwrite an existing output")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_exclusive(
        targets[0],
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    with targets[1].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "figure_id",
                "label",
                "display",
                "source_id",
                "source_path",
                "source_sha256",
                "json_pointer",
            ),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(table_rows)
    for figure, target in zip(figures, targets[2:], strict=True):
        _write_exclusive(target, _svg_figure(figure, figure["metrics"]))
    return {
        "passed": True,
        "output_dir": str(output_dir),
        "figures": len(figures),
        "metrics": len(table_rows),
        "files": [
            {"path": str(target), "sha256": file_sha256(target)} for target in targets
        ],
    }


def doctor_mainline(stage: str | None = None) -> dict:
    environment = load_json(ENVIRONMENT_PATH)
    snapshot = environment["host_snapshot"]
    package_map = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "matplotlib": "matplotlib",
        "jsonschema": "jsonschema",
    }
    package_checks = {}
    for key, package in package_map.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        package_checks[key] = {
            "expected": snapshot[key],
            "observed": observed,
            "passed": observed == snapshot[key],
        }
    python_observed = platform.python_version()
    torch_observed = None
    cuda_available = False
    try:
        import torch

        torch_observed = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        pass
    torch_ready = torch_observed == snapshot["torch"]
    gpu_required = stage in environment["runtime_contract"]["gpu_required_for"]
    core_ready = bool(
        python_observed == snapshot["python"]
        and torch_ready
        and all(check["passed"] for check in package_checks.values())
        and shutil.which("git")
        and shutil.which("tar")
        and shutil.which("zstd")
        and file_sha256(_safe_repo_path(environment["dependency_lock"]["path"]))
        == environment["dependency_lock"]["sha256"]
    )
    stage_ready = core_ready and (not gpu_required or cuda_available)
    return {
        "passed": stage_ready,
        "stage": stage,
        "core_ready": core_ready,
        "gpu_required": gpu_required,
        "cuda_available": cuda_available,
        "python": {
            "expected": snapshot["python"],
            "observed": python_observed,
            "passed": python_observed == snapshot["python"],
        },
        "torch": {
            "expected": snapshot["torch"],
            "observed": torch_observed,
            "passed": torch_ready,
        },
        "packages": package_checks,
    }


def _worktree(commit: str) -> Path:
    base = Path(os.environ.get("FSRL_MAINLINE_TMP", "/tmp/fsrl-mainline"))
    worktree = base / "worktrees" / commit
    if worktree.exists():
        completed = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or completed.stdout.strip() != commit:
            raise RuntimeError(
                f"existing replay worktree has wrong identity: {worktree}"
            )
        tracked_diff = subprocess.run(
            [
                "git",
                "-C",
                str(worktree),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked_diff.returncode != 0 or tracked_diff.stdout:
            raise RuntimeError(
                f"existing replay worktree has tracked changes: {worktree}"
            )
        return worktree
    worktree.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), commit],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not create detached replay worktree: {completed.stderr}"
        )
    return worktree


def _extract_required_artifacts(
    worktree: Path, tags: list[str], *, active_runtime_layout: bool = False
) -> list[str]:
    if not tags:
        return []
    artifacts = load_json(ARTIFACTS_PATH)
    bundle = _safe_repo_path(artifacts["bundle"]["path"])
    selected = [
        member
        for member in artifacts["members"]
        if set(tags).intersection(member["required_for"])
    ]
    if not selected:
        raise RuntimeError(f"no bundled artifact satisfies replay tags: {tags}")
    for member in selected:
        bundle_member = PurePosixPath(member["bundle_member"])
        if active_runtime_layout:
            if bundle_member.parts[0] != "output":
                raise RuntimeError(
                    "active runtime artifact must use the frozen output/ prefix"
                )
            target = worktree / "artifacts" / "runs" / Path(*bundle_member.parts[1:])
            extraction_root = worktree / "artifacts" / "runs"
        else:
            target = worktree.joinpath(*bundle_member.parts)
            extraction_root = worktree
        if target.exists():
            if (
                target.is_symlink()
                or not target.is_file()
                or file_sha256(target) != member["sha256"]
            ):
                raise RuntimeError(f"existing replay artifact has wrong hash: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "tar",
            "--zstd",
            "-xf",
            str(bundle),
            "-C",
            str(extraction_root),
        ]
        if active_runtime_layout:
            command.append("--strip-components=1")
        command.append(member["bundle_member"])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"artifact extraction failed: {completed.stderr}")
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(
                f"extracted replay artifact is not a regular file: {target}"
            )
        if file_sha256(target) != member["sha256"]:
            raise RuntimeError(f"extracted replay artifact has wrong hash: {target}")
    return [member["logical_name"] for member in selected]


def restore_test_artifacts() -> dict:
    """Restore only the six registered artifacts needed by the CPU test suite."""

    verify_artifact_bundle()
    restored = _extract_required_artifacts(
        ROOT, ["cpu_test_suite"], active_runtime_layout=True
    )
    if len(restored) != 6:
        raise RuntimeError("CPU test artifact profile must contain exactly six files")
    return {
        "passed": True,
        "profile": "cpu_test_suite",
        "restored_artifacts": restored,
    }


def replay_stage(stage: str, output: Path | None = None) -> dict:
    manifest = load_json(MANIFEST_PATH)
    verify_mainline()
    if stage not in manifest["replay_stages"]:
        raise RuntimeError(f"unknown replay stage: {stage}")
    record_id = manifest["replay_stages"][stage]
    record = manifest["execution_records"][record_id]
    worktree = _worktree(record["execution_commit"])
    extracted = _extract_required_artifacts(
        worktree, record.get("artifact_requirement_tags", [])
    )
    if output is None:
        base = Path(os.environ.get("FSRL_MAINLINE_TMP", "/tmp/fsrl-mainline"))
        output = base / "outputs" / stage / "result.json"
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("replay refuses to overwrite an existing result")
    output.parent.mkdir(parents=True, exist_ok=True)
    replacements = {"{python}": sys.executable, "{result}": str(output)}
    argv = [replacements.get(argument, argument) for argument in record["argv"]]
    replay_environment = os.environ.copy()
    replay_environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(worktree),
        }
    )
    completed = subprocess.run(
        argv,
        cwd=worktree,
        env=replay_environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"historical replay failed with exit code {completed.returncode}"
        )
    if not output.is_file():
        raise RuntimeError("historical replay did not create its registered result")
    observed_sha = file_sha256(output)
    expected_sha = record["replay_policy"]["exact"]["expected_sha256"]
    exact = observed_sha == expected_sha
    document = load_json(output)
    assertions = [
        _assert_semantic(document, assertion)
        for assertion in record["replay_policy"]["semantic"]["assertions"]
    ]
    semantic = bool(assertions and all(assertion["passed"] for assertion in assertions))
    if not semantic:
        raise RuntimeError(
            f"historical replay failed semantic assertions: {assertions}"
        )
    return {
        "passed": True,
        "stage": stage,
        "record_id": record_id,
        "execution_commit": record["execution_commit"],
        "worktree": str(worktree),
        "output": str(output),
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "exact_replay": exact,
        "semantic_replay": semantic,
        "replay_outcome": "exact_and_semantic" if exact else "semantic_only",
        "semantic_assertions": assertions,
        "extracted_artifacts": extracted,
    }


def status_mainline() -> dict:
    manifest = load_json(MANIFEST_PATH)
    schema = validate_manifest_structure(manifest)
    return {
        "passed": True,
        "mainline_id": manifest["mainline_id"],
        "status": manifest["status"],
        "claim_nodes": list(manifest["claim_nodes"]),
        "replay_stages": list(manifest["replay_stages"]),
        "schema": schema,
        "freeze": manifest.get("freeze"),
    }


def _parser() -> argparse.ArgumentParser:
    manifest = load_json(MANIFEST_PATH)
    replay_choices = sorted(manifest.get("replay_stages", {}))
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--stage", choices=replay_choices)
    subparsers.add_parser("restore-test-artifacts")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/fsrl-mainline/summary"),
    )
    replay = subparsers.add_parser("replay")
    replay.add_argument("--stage", required=True, choices=replay_choices)
    replay.add_argument("--output", type=Path)
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = _parser().parse_args(args)
    if parsed.command == "status":
        result = status_mainline()
    elif parsed.command == "verify":
        result = verify_mainline()
    elif parsed.command == "doctor":
        result = doctor_mainline(parsed.stage)
    elif parsed.command == "restore-test-artifacts":
        result = restore_test_artifacts()
    elif parsed.command == "summarize":
        result = summarize_mainline(parsed.output_dir)
    elif parsed.command == "replay":
        result = replay_stage(parsed.stage, parsed.output)
    else:
        raise AssertionError(f"unhandled Liu mainline command: {parsed.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
