"""Human-facing catalog for the frozen Liu research record.

The catalog is a locator layer.  It never rewrites or relocates canonical
reports, contracts, locks, results, or artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "research" / "liu" / "catalog.json"
GENERATED_PATHS = (
    Path("research/liu/README.md"),
    Path("research/liu/STUDY_LEDGER.md"),
    Path("docs/INDEX.md"),
    Path("benchmarks/INDEX.md"),
    Path("results/INDEX.md"),
)
INVENTORY_ROOTS = ("docs", "benchmarks", "results")
REQUIRED_STUDY_FIELDS = (
    "id",
    "chapter",
    "order",
    "title",
    "status",
    "question",
    "finding",
    "boundary",
    "prefixes",
    "paths",
)


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_inventory() -> list[Path]:
    generated = set(GENERATED_PATHS)
    files: list[Path] = []
    for root_name in INVENTORY_ROOTS:
        for path in (ROOT / root_name).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if relative not in generated:
                files.append(relative)
    return sorted(files, key=lambda path: path.as_posix())


def _matches_study(relative: Path, study: dict[str, Any]) -> bool:
    relative_string = relative.as_posix()
    if relative_string in study["paths"]:
        return True
    return any(relative.name.startswith(prefix) for prefix in study["prefixes"])


def file_role(relative: Path) -> str:
    path = relative.as_posix()
    name = relative.name
    if path.startswith("mainlines/liu_v1/"):
        if name == "manifest.json":
            return "canonical_mainline_manifest"
        if name == "report_view.json":
            return "presentation_view"
        if name == "artifacts.json":
            return "artifact_registry"
        if name in {"environment.json", "requirements-lock.txt"}:
            return "environment_contract"
        if name == "validation.json":
            return "freeze_validation"
        return "mainline_guide"
    if path.startswith("docs/assets/"):
        return "presentation_asset"
    if path.startswith("docs/"):
        return "source_context" if relative.suffix == ".pdf" else "report"
    if path.startswith("benchmarks/"):
        if "artifact_lock" in name:
            return "artifact_lock"
        if ".repair" in name and ".lock" in name:
            return "repair_lock"
        if ".repair" in name:
            return "repair_contract"
        if ".lock" in name:
            return "execution_lock"
        if ".parameters" in name:
            return "frozen_parameters"
        if "raw.schema" in name:
            return "data_schema"
        if "randomization" in name:
            return "randomization_record"
        return "registered_contract"
    if path.startswith("results/"):
        if "noninterpretable" in name:
            return "noninterpretable_attempt"
        if ".attempt" in name:
            return "superseded_repair_source"
        if relative.suffix == ".npz":
            return "supporting_artifact"
        if "collection_readiness" in name:
            return "readiness_result"
        if "validation" in name:
            return "validation_result"
        return "frozen_result"
    return "supporting_file"


def validate_catalog(
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = load_catalog() if catalog is None else catalog
    errors: list[str] = []

    if catalog.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if (
        catalog.get("canonical_path_policy", {}).get("mode")
        != "legacy_pinned_capsule_native"
    ):
        errors.append("canonical_path_policy.mode must be legacy_pinned_capsule_native")

    chapters = catalog.get("chapters", [])
    chapter_ids = [chapter.get("id") for chapter in chapters]
    if len(chapter_ids) != len(set(chapter_ids)):
        errors.append("chapter ids must be unique")
    chapter_set = set(chapter_ids)

    statuses = set(catalog.get("status_legend", {}))
    studies = catalog.get("studies", [])
    study_ids = [study.get("id") for study in studies]
    if len(study_ids) != len(set(study_ids)):
        errors.append("study ids must be unique")
    study_by_id = {study.get("id"): study for study in studies}

    for study in studies:
        study_id = study.get("id", "<missing>")
        missing = [field for field in REQUIRED_STUDY_FIELDS if field not in study]
        if missing:
            errors.append(f"{study_id}: missing fields {missing}")
            continue
        for field in ("id", "title", "question", "finding", "boundary"):
            if not isinstance(study[field], str) or not study[field].strip():
                errors.append(f"{study_id}: {field} must be a non-empty string")
        if study["chapter"] not in chapter_set:
            errors.append(f"{study_id}: unknown chapter {study['chapter']!r}")
        if study["status"] not in statuses:
            errors.append(f"{study_id}: unknown status {study['status']!r}")
        if not isinstance(study["order"], int):
            errors.append(f"{study_id}: order must be an integer")
        if not isinstance(study["prefixes"], list) or not all(
            isinstance(prefix, str) and prefix for prefix in study["prefixes"]
        ):
            errors.append(f"{study_id}: prefixes must be non-empty strings")
        if not isinstance(study["paths"], list) or not all(
            isinstance(path, str) and path for path in study["paths"]
        ):
            errors.append(f"{study_id}: paths must be non-empty strings")

    for view in catalog.get("views", []):
        references = view.get("studies", [])
        if len(references) != len(set(references)):
            errors.append(f"view {view.get('id')}: duplicate study references")
        unknown = sorted(set(references) - set(study_ids))
        if unknown:
            errors.append(f"view {view.get('id')}: unknown studies {unknown}")

    explicit_owners: dict[str, list[str]] = defaultdict(list)
    for study in studies:
        for path_string in study.get("paths", []):
            explicit_owners[path_string].append(study.get("id", "<missing>"))
            if not (ROOT / path_string).is_file():
                errors.append(f"{study.get('id')}: missing explicit path {path_string}")
    for path_string, owners in sorted(explicit_owners.items()):
        if len(owners) > 1:
            errors.append(f"explicit path {path_string} has multiple owners {owners}")

    inventory = discover_inventory()
    files_by_study: dict[str, list[Path]] = defaultdict(list)
    unassigned: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for relative in inventory:
        owners = [study["id"] for study in studies if _matches_study(relative, study)]
        if not owners:
            unassigned.append(relative.as_posix())
        elif len(owners) > 1:
            ambiguous[relative.as_posix()] = owners
        else:
            files_by_study[owners[0]].append(relative)

    if unassigned:
        errors.append(f"unassigned inventory files: {unassigned}")
    for path_string, owners in ambiguous.items():
        errors.append(f"ambiguous inventory file {path_string}: {owners}")

    for study_id, study in study_by_id.items():
        matched = files_by_study.get(study_id, [])
        explicit = [Path(path) for path in study.get("paths", [])]
        if not matched and not explicit:
            errors.append(f"{study_id}: study owns no files")

    for relative in inventory:
        if (
            relative.as_posix().startswith("results/")
            and "noninterpretable" in relative.name
            and file_role(relative) != "noninterpretable_attempt"
        ):
            errors.append(f"attempt result is not visibly classified: {relative}")

    return {
        "passed": not errors,
        "errors": errors,
        "inventory_files": len(inventory),
        "studies": len(studies),
        "chapters": len(chapters),
        "files_by_study": {
            study_id: [path.as_posix() for path in sorted(paths)]
            for study_id, paths in sorted(files_by_study.items())
        },
        "unassigned": unassigned,
        "ambiguous": ambiguous,
    }


def _ordered_chapters(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(catalog["chapters"], key=lambda chapter: chapter["order"])


def _ordered_studies(
    catalog: dict[str, Any], chapter_id: str | None = None
) -> list[dict[str, Any]]:
    chapter_order = {chapter["id"]: chapter["order"] for chapter in catalog["chapters"]}
    studies = catalog["studies"]
    if chapter_id is not None:
        studies = [study for study in studies if study["chapter"] == chapter_id]
    return sorted(
        studies,
        key=lambda study: (
            chapter_order[study["chapter"]],
            study["order"],
            study["id"],
        ),
    )


def _all_study_files(study: dict[str, Any], validation: dict[str, Any]) -> list[Path]:
    matched = [Path(path) for path in validation["files_by_study"].get(study["id"], [])]
    explicit = [Path(path) for path in study["paths"]]
    return sorted(set(matched + explicit), key=lambda path: path.as_posix())


def _link(
    relative: Path,
    output_path: Path,
    *,
    full_label: bool = True,
    label: str | None = None,
) -> str:
    target = Path(os.path.relpath(ROOT / relative, start=(ROOT / output_path).parent))
    target_text = target.as_posix()
    if " " in target_text:
        target_text = f"<{target_text}>"
    display = label or (relative.as_posix() if full_label else relative.name)
    return f"[{display}]({target_text})"


def _generated_notice() -> list[str]:
    return [
        "> [!NOTE]",
        "> This page is generated from `research/liu/catalog.json`. Edit the",
        "> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.",
        "> Historical files remain canonical at their original paths.",
        "",
    ]


def _study_page_path(study_id: str) -> Path:
    return Path("research") / "liu" / "studies" / study_id / "README.md"


def _render_readme(catalog: dict[str, Any], validation: dict[str, Any]) -> str:
    study_by_id = {study["id"]: study for study in catalog["studies"]}
    lines = ["# Liu research guide", ""] + _generated_notice()
    lines.extend(
        [
            "This is the human-facing companion to the frozen machine-readable Liu",
            "mainline. It organizes the same evidence by scientific question, finding,",
            "negative constraint, and reading purpose. It does not create a second copy",
            "of the evidence or change any registered outcome.",
            "",
            "```text",
            "canonical historical files -> frozen Liu Mainline v1 -> human catalog views",
            "```",
            "",
            "```mermaid",
            "flowchart TD",
            "    T[Task fidelity] --> B[Behavioral competence]",
            "    T --> G[P_T global assembly]",
            "    T --> L[a_T direct fidelity]",
            "    G --> A[Algorithmic asymmetry]",
            "    L --> A",
            "    G --> X[One-factor transport]",
            "    L --> X",
            "    B --> F[Liu Mainline v1]",
            "    A --> F",
            "    X --> F",
            "```",
            "",
            "## Start here",
            "",
            f"- Frozen evidence object: {_link(Path('mainlines/liu_v1/README.md'), GENERATED_PATHS[0])}",
            f"- Current presentation package: {_link(Path('docs/liu_presentation_package_v2.md'), GENERATED_PATHS[0])}",
            "- Physical study capsules: [studies/](studies/)",
            "- Complete one-page ledger: [STUDY_LEDGER.md](STUDY_LEDGER.md)",
            f"- Report library: {_link(Path('docs/INDEX.md'), GENERATED_PATHS[0])}",
            f"- Contract and lock library: {_link(Path('benchmarks/INDEX.md'), GENERATED_PATHS[0])}",
            f"- Result library: {_link(Path('results/INDEX.md'), GENERATED_PATHS[0])}",
            "",
            "## Reading routes",
            "",
        ]
    )
    for view in catalog["views"]:
        lines.extend([f"### {view['title']}", "", view["purpose"], ""])
        for index, study_id in enumerate(view["studies"], start=1):
            study = study_by_id[study_id]
            lines.append(
                f"{index}. [{study['title']}](studies/{study_id}/README.md) "
                f"— `{study['status']}` — {study['finding']}"
            )
        lines.append("")

    lines.extend(["## Evidence chapters", ""])
    for chapter in _ordered_chapters(catalog):
        chapter_studies = _ordered_studies(catalog, chapter["id"])
        lines.append(f"### {chapter['title']}")
        lines.append("")
        lines.append(chapter["purpose"])
        lines.append("")
        for study in chapter_studies:
            lines.append(
                f"- [{study['title']}](studies/{study['id']}/README.md) "
                f"— `{study['status']}`"
            )
        lines.append("")

    lines.extend(["## Status vocabulary", ""])
    for status, meaning in catalog["status_legend"].items():
        lines.append(f"- `{status}`: {meaning}")
    lines.extend(
        [
            "",
            "## Maintenance contract",
            "",
            (
                "- The physical-path audit at commit `"
                f"{catalog['canonical_path_policy']['migration_audit']['base_commit'][:7]}` "
                "found "
                f"{catalog['canonical_path_policy']['migration_audit']['explicit_legacy_path_reference_occurrences']} "
                "explicit legacy-path references."
            ),
            "- Add or revise a study entry in `research/liu/catalog.json`.",
            "- Frozen legacy paths remain pinned because contracts, locks, runners, and",
            "  reports refer to them by path and hash. Each physical study capsule gathers",
            "  their human-facing entry points without changing those identities.",
            "- A future study may place canonical files inside its capsule from the start.",
            "  Moving a historical file later is a versioned provenance migration, not a",
            "  navigation edit.",
            "- Mark failed executions as `noninterpretable_attempt`; never let them appear",
            "  as canonical frozen results.",
            "- Run `direnv exec . python -m fsrl.liu_catalog check` before commit.",
            "- A scientific change belongs in a new registered experiment or Liu mainline",
            "  version, not in this navigation layer.",
            "",
            (
                f"Catalog coverage: {validation['inventory_files']} historical files "
                f"across {validation['studies']} study units and "
                f"{validation['chapters']} chapters."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_study_page(
    catalog: dict[str, Any],
    study: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    output_path = _study_page_path(study["id"])
    chapter = next(
        chapter for chapter in catalog["chapters"] if chapter["id"] == study["chapter"]
    )
    lines = [f"# {study['title']}", ""] + _generated_notice()
    lines.extend(
        [
            _link(
                Path("research/liu/README.md"),
                output_path,
                label="Back to Liu research guide",
            ),
            "",
            f"- **Status:** `{study['status']}`",
            f"- **Study ID:** `{study['id']}`",
            f"- **Chapter:** {chapter['title']}",
            "",
            "## Scientific role",
            "",
            f"**Question.** {study['question']}",
            "",
            f"**Finding.** {study['finding']}",
            "",
            f"**Claim boundary.** {study['boundary']}",
            "",
            "## Canonical files",
            "",
        ]
    )
    for relative in _all_study_files(study, validation):
        lines.append(f"- `{file_role(relative)}` — {_link(relative, output_path)}")
    lines.extend(
        [
            "",
            "## Path policy",
            "",
            "The files above remain canonical at their registered historical paths. This",
            "capsule is the stable human-facing home for the study. A future study may put",
            "its canonical files inside its capsule from inception, but relocating these",
            "frozen files would require a separately versioned provenance migration.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_ledger(catalog: dict[str, Any], validation: dict[str, Any]) -> str:
    output_path = GENERATED_PATHS[1]
    lines = ["# Liu study ledger", ""] + _generated_notice()
    lines.extend(
        [
            "Each study unit joins the scientific question to its exact reports, contracts,",
            "locks, results, repairs, and attempts. Status applies to the study-level",
            "conclusion; file roles distinguish canonical evidence from provenance-only",
            "attempts.",
            "",
        ]
    )
    for chapter in _ordered_chapters(catalog):
        lines.extend([f"## {chapter['title']}", "", chapter["purpose"], ""])
        for study in _ordered_studies(catalog, chapter["id"]):
            lines.extend(
                [
                    f'<a id="{study["id"]}"></a>',
                    f"### [{study['title']}](studies/{study['id']}/README.md)",
                    "",
                    f"- **Status:** `{study['status']}`",
                    f"- **Study ID:** `{study['id']}`",
                    "",
                    f"**Question.** {study['question']}",
                    "",
                    f"**Finding.** {study['finding']}",
                    "",
                    f"**Claim boundary.** {study['boundary']}",
                    "",
                    "**Files.**",
                    "",
                ]
            )
            files = _all_study_files(study, validation)
            for relative in files:
                lines.append(
                    f"- `{file_role(relative)}` — {_link(relative, output_path)}"
                )
            lines.append("")
    return "\n".join(lines)


def _render_directory_index(
    catalog: dict[str, Any],
    validation: dict[str, Any],
    directory: str,
    title: str,
    introduction: list[str],
) -> str:
    output_path = Path(directory) / "INDEX.md"
    lines = [f"# {title}", ""] + _generated_notice() + introduction + [""]
    for chapter in _ordered_chapters(catalog):
        rows: list[tuple[dict[str, Any], list[Path]]] = []
        for study in _ordered_studies(catalog, chapter["id"]):
            files = [
                path
                for path in _all_study_files(study, validation)
                if path.parts[0] == directory
            ]
            if files:
                rows.append((study, files))
        if not rows:
            continue
        lines.extend([f"## {chapter['title']}", "", chapter["purpose"], ""])
        for study, files in rows:
            capsule_link = _link(
                _study_page_path(study["id"]),
                output_path,
                label="study capsule",
            )
            lines.extend(
                [
                    f"### {study['title']}",
                    "",
                    f"`{study['status']}` · {capsule_link}",
                    "",
                    f"{study['finding']}",
                    "",
                ]
            )
            for relative in files:
                lines.append(
                    f"- `{file_role(relative)}` — "
                    f"{_link(relative, output_path, full_label=False)}"
                )
            lines.append("")
    return "\n".join(lines)


def render_indexes(
    catalog: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[Path, str]:
    catalog = load_catalog() if catalog is None else catalog
    validation = validate_catalog(catalog) if validation is None else validation
    if not validation["passed"]:
        raise RuntimeError(
            "catalog validation failed: " + "; ".join(validation["errors"])
        )
    rendered = {
        GENERATED_PATHS[0]: _render_readme(catalog, validation),
        GENERATED_PATHS[1]: _render_ledger(catalog, validation),
        GENERATED_PATHS[2]: _render_directory_index(
            catalog,
            validation,
            "docs",
            "Liu report library",
            [
                "Reports remain at their historical paths. This index presents them in",
                "scientific order rather than filename or commit order.",
            ],
        ),
        GENERATED_PATHS[3]: _render_directory_index(
            catalog,
            validation,
            "benchmarks",
            "Liu contract and provenance library",
            [
                "A `registered_contract` defines a scientific test. Locks bind source or",
                "artifacts; repair files preserve prospective corrections. None is silently",
                "treated as interchangeable with a result.",
            ],
        ),
        GENERATED_PATHS[4]: _render_directory_index(
            catalog,
            validation,
            "results",
            "Liu result library",
            [
                "Canonical results, supporting arrays, validation outputs, and failed",
                "executions are listed separately. A `noninterpretable_attempt` is provenance",
                "only and never contributes scientific support. A",
                "`superseded_repair_source` remains interpretable repair provenance but is",
                "not the canonical post-repair outcome.",
            ],
        ),
    }
    for study in catalog["studies"]:
        rendered[_study_page_path(study["id"])] = _render_study_page(
            catalog, study, validation
        )
    return rendered


def build_indexes() -> dict[str, Any]:
    validation = validate_catalog()
    rendered = render_indexes(validation=validation)
    changed: list[str] = []
    for relative, content in rendered.items():
        path = ROOT / relative
        expected = content.rstrip() + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
            changed.append(relative.as_posix())
    return {"passed": True, "changed": changed, **validation}


def check_indexes() -> dict[str, Any]:
    validation = validate_catalog()
    if not validation["passed"]:
        return validation
    rendered = render_indexes(validation=validation)
    stale: list[str] = []
    for relative, content in rendered.items():
        path = ROOT / relative
        expected = content.rstrip() + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale.append(relative.as_posix())
    studies_root = ROOT / "research" / "liu" / "studies"
    existing_capsules = (
        {path.relative_to(ROOT).as_posix() for path in studies_root.glob("*/README.md")}
        if studies_root.exists()
        else set()
    )
    expected_capsules = {
        path.as_posix()
        for path in rendered
        if path.as_posix().startswith("research/liu/studies/")
    }
    unexpected = sorted(existing_capsules - expected_capsules)
    return {
        **validation,
        "passed": not stale and not unexpected,
        "stale_generated_files": stale,
        "unexpected_study_capsules": unexpected,
    }


def audit_summary() -> dict[str, Any]:
    catalog = load_catalog()
    validation = validate_catalog(catalog)
    status_counts = Counter(study["status"] for study in catalog["studies"])
    role_counts = Counter(file_role(path) for path in discover_inventory())
    directory_counts = Counter(path.parts[0] for path in discover_inventory())
    return {
        "passed": validation["passed"],
        "errors": validation["errors"],
        "inventory_files": validation["inventory_files"],
        "studies": validation["studies"],
        "chapters": validation["chapters"],
        "status_counts": dict(sorted(status_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "directory_counts": dict(sorted(directory_counts.items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "build", "check"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = audit_summary()
    elif args.command == "build":
        result = build_indexes()
    else:
        result = check_indexes()
    display = {key: value for key, value in result.items() if key != "files_by_study"}
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
