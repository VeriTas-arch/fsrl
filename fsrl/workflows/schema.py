"""Schema validation and human rendering for maintained research workflows."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ID = re.compile(r"[a-z][a-z0-9_]*")
REQUIRED_STAGE_FIELDS = (
    "id",
    "title",
    "question",
    "method",
    "finding",
    "boundary",
    "depends_on",
    "implementation",
    "studies",
    "verify",
)


def load_workflow(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _safe_repo_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"workflow path must be repository-relative: {value!r}")
    return ROOT.joinpath(*pure.parts)


def validate_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Validate structure, dependency order, and every repository locator."""

    if workflow.get("schema_version") != 1:
        raise ValueError("workflow schema_version must be 1")
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
    study_paths: set[str] = set()
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
        for field in ("implementation", "studies", "verify"):
            values = stage[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"stage {stage['id']} requires non-empty {field}")
        implementation_paths.update(stage["implementation"])
        study_paths.update(f"studies/{study}/study.toml" for study in stage["studies"])
        seen.add(stage["id"])

    registered_paths = [*paths, *implementation_paths, *study_paths]
    missing_paths = [
        value
        for value in sorted(set(registered_paths))
        if not _safe_repo_path(value).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"workflow paths do not exist: {missing_paths}")

    return {
        "passed": True,
        "workflow_id": workflow_id,
        "stages": len(stages),
        "dependency_edges": sum(len(stage["depends_on"]) for stage in stages),
        "implementation_paths": len(implementation_paths),
        "studies": len({study for stage in stages for study in stage["studies"]}),
    }


def render_workflow(workflow: dict[str, Any]) -> str:
    """Render the same schema as a concise human research route."""

    validate_workflow(workflow)
    lines = [
        f"# {workflow['title']}",
        "",
        "> This page is generated from `workflow.toml`. The TOML is the machine-readable",
        "> contract; this page is the human reading route.",
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
        lines.extend(["", "Evidence:", ""])
        lines.extend(
            f"- [{study}](../../studies/{study}/README.md)"
            for study in stage["studies"]
        )
        lines.extend(["", "Verification:", ""])
        lines.extend(f"- `{command}`" for command in stage["verify"])

    lines.extend(
        [
            "",
            "## Evidence and figures",
            "",
            f"- [Study registry](../../{workflow['study_registry']})",
            f"- [Cross-study synthesis](../../{workflow['synthesis']})",
            f"- [Report and paper figures](../../{workflow['figure_root']})",
            "",
            "Runtime outputs remain outside this workflow. A result enters the evidence",
            "registry only through a study-owned contract, result, report, and provenance",
            "record; a report-facing figure additionally requires source data and a panel",
            "manifest.",
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
