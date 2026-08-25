"""Schema validation and human rendering for maintained research workflows."""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

from fsrl.infra.study_registry import (
    load_registry,
    load_studies,
    registered_file_sha256,
)
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
WORKFLOW_ID = re.compile(r"[a-z][a-z0-9_]*")
EVIDENCE_USES = {"defines", "supports", "constrains", "closes"}
RESOURCES = {"cpu", "gpu", "mixed"}
REQUIRED_STAGE_FIELDS = (
    "id",
    "title",
    "question",
    "method",
    "finding",
    "boundary",
    "depends_on",
    "implementation",
    "tests",
    "studies",
    "verification",
    "evidence",
)


def load_workflow(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _safe_repo_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"workflow path must be repository-relative: {value!r}")
    return ROOT.joinpath(*pure.parts)


def _json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise TypeError(f"JSON pointer crosses a scalar: {pointer!r}")
    return value


def _study_record(
    studies: dict[str, dict[str, Any]], study_id: str, record_path: str
) -> tuple[dict[str, Any], Path]:
    study = studies.get(study_id)
    if study is None:
        raise ValueError(f"workflow references an unknown study: {study_id}")
    matches = [record for record in study["records"] if record["path"] == record_path]
    if len(matches) != 1:
        raise ValueError(
            f"workflow evidence must identify one registered record: "
            f"{study_id}:{record_path}"
        )
    record = matches[0]
    path = ROOT / "studies" / study_id / record_path
    return record, path


def _validate_evidence(
    stage: dict[str, Any], studies: dict[str, dict[str, Any]]
) -> int:
    evidence = stage["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"stage {stage['id']} requires non-empty evidence")
    referenced_studies: set[str] = set()
    for entry in evidence:
        required = {"study", "record", "use", "description"}
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(
                f"stage {stage['id']} evidence is missing fields: {missing}"
            )
        study_id = entry["study"]
        if study_id not in stage["studies"]:
            raise ValueError(
                f"stage {stage['id']} evidence study is absent from studies: {study_id}"
            )
        if entry["use"] not in EVIDENCE_USES:
            raise ValueError(
                f"stage {stage['id']} has invalid evidence use: {entry['use']}"
            )
        if (
            not isinstance(entry["description"], str)
            or not entry["description"].strip()
        ):
            raise ValueError(
                f"stage {stage['id']} evidence description must be non-empty"
            )
        record, path = _study_record(studies, study_id, entry["record"])
        observed = registered_file_sha256(path, record["sha256"], resolved_path=path)
        if observed != record["sha256"]:
            raise ValueError(f"workflow evidence hash mismatch: {path}")
        pointer = entry.get("json_pointer")
        if pointer is not None:
            if path.suffix != ".json":
                raise ValueError(f"JSON pointer targets a non-JSON record: {path}")
            try:
                _json_pointer(json.loads(path.read_text(encoding="utf-8")), pointer)
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"workflow evidence pointer does not resolve: "
                    f"{study_id}:{entry['record']}#{pointer}"
                ) from error
        referenced_studies.add(study_id)
    if referenced_studies != set(stage["studies"]):
        missing = sorted(set(stage["studies"]) - referenced_studies)
        raise ValueError(
            f"stage {stage['id']} has studies without exact evidence: {missing}"
        )
    return len(evidence)


def _validate_verification(stage: dict[str, Any]) -> int:
    verifications = stage["verification"]
    if not isinstance(verifications, list) or not verifications:
        raise ValueError(f"stage {stage['id']} requires non-empty verification")
    ids: set[str] = set()
    for entry in verifications:
        required = {"id", "resource", "argv"}
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(
                f"stage {stage['id']} verification is missing fields: {missing}"
            )
        verification_id = entry["id"]
        if (
            not isinstance(verification_id, str)
            or WORKFLOW_ID.fullmatch(verification_id) is None
        ):
            raise ValueError(
                f"stage {stage['id']} has invalid verification id: {verification_id!r}"
            )
        if verification_id in ids:
            raise ValueError(
                f"stage {stage['id']} has duplicate verification id: {verification_id}"
            )
        if entry["resource"] not in RESOURCES:
            raise ValueError(
                f"stage {stage['id']} has invalid verification resource: "
                f"{entry['resource']}"
            )
        argv = entry["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
        ):
            raise ValueError(
                f"stage {stage['id']} verification argv must be non-empty strings"
            )
        ids.add(verification_id)
    return len(verifications)


def _validate_figures(stage: dict[str, Any]) -> int:
    figures = stage.get("figures", [])
    if not isinstance(figures, list):
        raise TypeError(f"stage {stage['id']} figures must be a list")
    for entry in figures:
        required = {"specification", "figure", "description"}
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f"stage {stage['id']} figure is missing fields: {missing}")
        specification = _safe_repo_path(entry["specification"])
        if not specification.is_file():
            raise FileNotFoundError(
                f"workflow figure specification does not exist: {specification}"
            )
        payload = json.loads(specification.read_text(encoding="utf-8"))
        figure_ids = {figure["id"] for figure in payload.get("figures", [])}
        if entry["figure"] not in figure_ids:
            raise ValueError(
                f"workflow figure is absent from {entry['specification']}: "
                f"{entry['figure']}"
            )
    return len(figures)


def validate_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Validate the current claim graph and every code/evidence/figure locator."""

    if workflow.get("schema_version") != 2:
        raise ValueError("workflow schema_version must be 2")
    workflow_id = workflow.get("id")
    if not isinstance(workflow_id, str) or WORKFLOW_ID.fullmatch(workflow_id) is None:
        raise ValueError("workflow id must use lowercase snake_case")
    if workflow.get("status") not in {"working", "frozen", "retired"}:
        raise ValueError("workflow status must be working, frozen, or retired")
    for field in ("title", "purpose", "working_claim", "boundary"):
        if not isinstance(workflow.get(field), str) or not workflow[field].strip():
            raise ValueError(f"workflow field is empty: {field}")

    paths = [
        workflow.get("study_registry"),
        workflow.get("synthesis"),
        workflow.get("figure_root"),
    ]
    if any(not isinstance(value, str) for value in paths):
        raise ValueError("workflow registry, synthesis, and figure paths are required")

    registry = load_registry()
    studies = load_studies(registry)
    stages = workflow.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("workflow must contain at least one stage")
    stage_ids = [stage.get("id") for stage in stages]
    if any(not isinstance(stage_id, str) for stage_id in stage_ids):
        raise ValueError("every stage requires a string id")
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("workflow stage ids must be unique")

    seen: set[str] = set()
    implementation_paths: set[str] = set()
    test_paths: set[str] = set()
    evidence_count = 0
    verification_count = 0
    figure_count = 0
    for stage in stages:
        missing = [field for field in REQUIRED_STAGE_FIELDS if field not in stage]
        if missing:
            raise ValueError(f"stage {stage.get('id')} is missing fields: {missing}")
        if WORKFLOW_ID.fullmatch(stage["id"]) is None:
            raise ValueError(f"invalid stage id: {stage['id']}")
        dependencies = stage["depends_on"]
        if not isinstance(dependencies, list) or any(
            dependency not in seen for dependency in dependencies
        ):
            raise ValueError(
                f"stage {stage['id']} must depend only on preceding stages"
            )
        for field in ("implementation", "tests", "studies"):
            values = stage[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"stage {stage['id']} requires non-empty {field}")
            if len(values) != len(set(values)):
                raise ValueError(f"stage {stage['id']} has duplicate {field}")
        implementation_paths.update(stage["implementation"])
        test_paths.update(stage["tests"])
        evidence_count += _validate_evidence(stage, studies)
        verification_count += _validate_verification(stage)
        figure_count += _validate_figures(stage)
        seen.add(stage["id"])

    registered_paths = [*paths, *implementation_paths, *test_paths]
    missing_paths = [
        value
        for value in sorted(set(registered_paths))
        if not _safe_repo_path(value).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"workflow paths do not exist: {missing_paths}")
    invalid_implementation = sorted(
        path for path in implementation_paths if not path.startswith("fsrl/")
    )
    invalid_tests = sorted(path for path in test_paths if not path.startswith("tests/"))
    if invalid_implementation:
        raise ValueError(
            f"workflow implementation paths must live under fsrl/: "
            f"{invalid_implementation}"
        )
    if invalid_tests:
        raise ValueError(f"workflow test paths must live under tests/: {invalid_tests}")

    return {
        "passed": True,
        "workflow_id": workflow_id,
        "stages": len(stages),
        "dependency_edges": sum(len(stage["depends_on"]) for stage in stages),
        "implementation_paths": len(implementation_paths),
        "test_paths": len(test_paths),
        "studies": len({study for stage in stages for study in stage["studies"]}),
        "evidence": evidence_count,
        "verifications": verification_count,
        "figures": figure_count,
    }


def _record_link(study_id: str, record: str, pointer: str | None) -> str:
    target = f"../../studies/{study_id}/{record}"
    label = f"{study_id}:{record}"
    suffix = f" `{pointer}`" if pointer is not None else ""
    return f"[{label}]({target}){suffix}"


def render_workflow(workflow: dict[str, Any]) -> str:
    """Render the exact workflow schema as the human mainline."""

    validate_workflow(workflow)
    lines = [
        f"# {workflow['title']}",
        "",
        "> This page is generated from `workflow.toml`. The TOML is the machine-readable",
        "> claim, evidence, implementation, verification, and figure contract.",
        "",
        workflow["purpose"],
        "",
        f"**Current working claim.** {workflow['working_claim']}",
        "",
        f"**Claim boundary.** {workflow['boundary']}",
        "",
        "## How to read this mainline",
        "",
        "| Stage | Scientific question | Current result |",
        "| --- | --- | --- |",
    ]
    for index, stage in enumerate(workflow["stages"], start=1):
        lines.append(
            f"| {index}. [{stage['title']}](#{stage['id'].replace('_', '-')}) "
            f"| {stage['question']} | {stage['finding']} |"
        )

    for index, stage in enumerate(workflow["stages"], start=1):
        lines.extend(
            [
                "",
                f"## {index}. {stage['title']}",
                "",
                f'<a id="{stage["id"].replace("_", "-")}"></a>',
                "",
                f"**Question.** {stage['question']}",
                "",
                f"**Method.** {stage['method']}",
                "",
                f"**Result.** {stage['finding']}",
                "",
                f"**Boundary.** {stage['boundary']}",
                "",
                "Implementation:",
                "",
            ]
        )
        lines.extend(f"- [`{path}`](../../{path})" for path in stage["implementation"])
        lines.extend(["", "Tests:", ""])
        lines.extend(f"- [`{path}`](../../{path})" for path in stage["tests"])
        lines.extend(["", "Exact evidence:", ""])
        for entry in stage["evidence"]:
            link = _record_link(
                entry["study"], entry["record"], entry.get("json_pointer")
            )
            lines.append(f"- `{entry['use']}` — {link} — {entry['description']}")
        if stage.get("figures"):
            lines.extend(["", "Figures:", ""])
            for entry in stage["figures"]:
                specification = entry["specification"]
                figure_root = Path(specification).parent
                figure = entry["figure"]
                target = f"../../{figure_root.as_posix()}/{figure}/{figure}.svg"
                lines.append(
                    f"- [{figure}]({target}) — {entry['description']} "
                    f"([specification](../../{specification}))"
                )
        lines.extend(["", "Verification:", ""])
        for entry in stage["verification"]:
            command = shlex.join(entry["argv"])
            lines.append(f"- `{entry['id']}` (`{entry['resource']}`): `{command}`")

    lines.extend(
        [
            "",
            "## Evidence and figures",
            "",
            f"- [Study registry](../../{workflow['study_registry']})",
            f"- [Cross-study synthesis](../../{workflow['synthesis']})",
            f"- [Report and paper figures](../../{workflow['figure_root']}/README.md)",
            "",
            "Runtime outputs remain outside this workflow. A result enters the evidence",
            "registry only through a study-owned contract, result, report, and provenance",
            "record; a report-facing figure additionally requires source data and a panel",
            "manifest. Historical replay remains a separate snapshot-level operation.",
            "",
        ]
    )
    return "\n".join(lines)


def check_rendered_readme(workflow_path: Path | str) -> dict[str, Any]:
    workflow_path = Path(workflow_path)
    expected = render_workflow(load_workflow(workflow_path))
    readme_path = workflow_path.with_name("README.md")
    observed = readme_path.read_text(encoding="utf-8")
    if observed != expected:
        raise ValueError(f"generated workflow README is stale: {readme_path}")
    return {"passed": True, "readme": str(readme_path)}
