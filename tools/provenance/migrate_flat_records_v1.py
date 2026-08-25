"""One-time byte-preserving migration from flat research roots to study bundles.

This importer is retained as migration provenance.  It is intentionally tied to
the pre-migration human catalog at commit ``fb32095`` and refuses partial or
ambiguous ownership.  After application, ongoing validation belongs to
``python -m fsrl.infrastructure.study_registry check``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = REPO_ROOT
SOURCE_COMMIT = "fb32095df70b1265b5d14b8eda3be6cb65036c6a"
SOURCE_REF = "refs/tags/liu-mainline-v1"
SYNTHESIS_OWNER = "liu_mainline_v1"
MIGRATION_PATH = ROOT / "studies" / "migrations" / "flat-records-v1.json"


def load_catalog() -> dict[str, Any]:
    payload = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:research/liu/catalog.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(payload)


def file_role(relative: Path) -> str:
    path = relative.as_posix()
    name = relative.name
    if path.startswith("mainlines/liu_v1/"):
        roles = {
            "manifest.json": "canonical_mainline_manifest",
            "report_view.json": "presentation_view",
            "artifacts.json": "artifact_registry",
            "environment.json": "environment_contract",
            "requirements-lock.txt": "environment_contract",
            "validation.json": "freeze_validation",
        }
        return roles.get(name, "mainline_guide")
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


def validate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    generated = {
        Path("docs/INDEX.md"),
        Path("benchmarks/INDEX.md"),
        Path("results/INDEX.md"),
    }
    inventory = sorted(
        relative
        for root_name in ("docs", "benchmarks", "results")
        for path in (ROOT / root_name).rglob("*")
        if path.is_file()
        for relative in (path.relative_to(ROOT),)
        if relative not in generated
    )
    files_by_study: dict[str, list[str]] = {
        study["id"]: [] for study in catalog["studies"]
    }
    errors: list[str] = []
    for relative in inventory:
        value = relative.as_posix()
        owners = [
            study["id"]
            for study in catalog["studies"]
            if value in study["paths"]
            or any(relative.name.startswith(prefix) for prefix in study["prefixes"])
        ]
        if len(owners) != 1:
            errors.append(f"{value}: expected one owner, found {owners}")
        else:
            files_by_study[owners[0]].append(value)
    return {
        "passed": not errors,
        "errors": errors,
        "files_by_study": files_by_study,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _string_list(values: list[str]) -> str:
    return "[" + ", ".join(_quote(value) for value in values) + "]"


def _record_lines(record: dict[str, Any], *, local_path: str) -> list[str]:
    return [
        "[[records]]",
        f"path = {_quote(local_path)}",
        f"legacy_path = {_quote(record['legacy_path'])}",
        f"role = {_quote(record['role'])}",
        f"sha256 = {_quote(record['sha256'])}",
        f"bytes = {record['bytes']}",
        f"source_ref = {_quote(record['source_ref'])}",
        "",
    ]


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob(reference: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{reference}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _record(
    *,
    legacy_path: Path,
    path: Path,
    owner_kind: str,
    owner_id: str,
    role: str,
) -> dict[str, Any]:
    source = ROOT / legacy_path
    if not source.is_file():
        raise RuntimeError(f"missing migration source: {legacy_path}")
    payload = _git_blob(SOURCE_REF, legacy_path.as_posix())
    source_hash = _sha256(source)
    if source.stat().st_size != len(payload) or source_hash != _sha256_bytes(payload):
        raise RuntimeError(f"migration source differs from {SOURCE_REF}: {legacy_path}")
    return {
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "legacy_path": legacy_path.as_posix(),
        "path": path.as_posix(),
        "role": role,
        "sha256": source_hash,
        "bytes": source.stat().st_size,
        "source_ref": SOURCE_REF,
    }


def build_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    if _git_output("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise RuntimeError("migration importer must start at the audited source commit")
    catalog = load_catalog()
    validation = validate_catalog(catalog)
    if not validation["passed"]:
        raise RuntimeError(
            "source catalog is invalid: " + "; ".join(validation["errors"])
        )

    owner_by_path = {
        path: study_id
        for study_id, paths in validation["files_by_study"].items()
        for path in paths
    }
    records: list[dict[str, Any]] = []
    for legacy_text, owner_id in sorted(owner_by_path.items()):
        legacy_path = Path(legacy_text)
        if owner_id == SYNTHESIS_OWNER:
            destination = Path("synthesis") / "records" / legacy_path
            owner_kind = "synthesis"
            final_owner_id = "current-model-synthesis"
        else:
            destination = Path("studies") / owner_id / "records" / legacy_path
            owner_kind = "study"
            final_owner_id = owner_id
        records.append(
            _record(
                legacy_path=legacy_path,
                path=destination,
                owner_kind=owner_kind,
                owner_id=final_owner_id,
                role=file_role(legacy_path),
            )
        )

    mainline_root = ROOT / "mainlines" / "liu_v1"
    for source in sorted(path for path in mainline_root.rglob("*") if path.is_file()):
        legacy_path = source.relative_to(ROOT)
        relative = source.relative_to(mainline_root)
        records.append(
            _record(
                legacy_path=legacy_path,
                path=Path("synthesis") / "frozen" / relative,
                owner_kind="synthesis",
                owner_id="current-model-synthesis",
                role=(
                    "artifact_bundle"
                    if relative.parts[:1] == ("artifacts",)
                    else file_role(legacy_path)
                ),
            )
        )

    destinations = [record["path"] for record in records]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("migration destinations collide")
    migration = {
        "schema_version": 1,
        "id": "flat-records-v1",
        "source_commit": SOURCE_COMMIT,
        "source_ref": SOURCE_REF,
        "mode": "byte_preserving_physical_relocation",
        "record_count": len(records),
        "records": sorted(records, key=lambda record: record["legacy_path"]),
    }
    return catalog, migration


def _registry_toml(catalog: dict[str, Any]) -> str:
    studies = [study for study in catalog["studies"] if study["id"] != SYNTHESIS_OWNER]
    views = []
    for view in catalog["views"]:
        converted = dict(view)
        converted["studies"] = [
            study_id for study_id in view["studies"] if study_id != SYNTHESIS_OWNER
        ]
        views.append(converted)
    lines = [
        "schema_version = 1",
        'id = "study-registry-v1"',
        'review_state = "indexed"',
        f"source_import_commit = {_quote(SOURCE_COMMIT)}",
        f"evidence_source_ref = {_quote(SOURCE_REF)}",
        'migration = "migrations/flat-records-v1.json"',
        "",
        "[status_legend]",
    ]
    for status, meaning in catalog["status_legend"].items():
        if status == "frozen_mainline":
            continue
        lines.append(f"{status} = {_quote(meaning)}")
    lines.append("")
    for chapter in catalog["chapters"]:
        lines.extend(
            [
                "[[chapters]]",
                f"id = {_quote(chapter['id'])}",
                f"order = {chapter['order']}",
                f"title = {_quote(chapter['title'])}",
                f"purpose = {_quote(chapter['purpose'])}",
                "",
            ]
        )
    for view in views:
        lines.extend(
            [
                "[[views]]",
                f"id = {_quote(view['id'])}",
                f"title = {_quote(view['title'])}",
                f"purpose = {_quote(view['purpose'])}",
                f"studies = {_string_list(view['studies'])}",
                "",
            ]
        )
    chapter_order = {chapter["id"]: chapter["order"] for chapter in catalog["chapters"]}
    for study in sorted(
        studies,
        key=lambda value: (
            chapter_order[value["chapter"]],
            value["order"],
            value["id"],
        ),
    ):
        manifest_path = f"studies/{study['id']}/study.toml"
        lines.extend(
            [
                "[[studies]]",
                f"id = {_quote(study['id'])}",
                f"path = {_quote(manifest_path)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def audit_migration() -> dict[str, Any]:
    migration = json.loads(MIGRATION_PATH.read_text(encoding="utf-8"))
    locator_path = ROOT / "studies" / "migrations" / "runtime-locators-v1.json"
    locator_migration = (
        json.loads(locator_path.read_text(encoding="utf-8"))
        if locator_path.is_file()
        else {"records": []}
    )
    locator_records = {
        record["path"]: record for record in locator_migration.get("records", [])
    }
    errors: list[str] = []
    for record in migration.get("records", []):
        legacy_path = record["legacy_path"]
        current = ROOT / record["path"]
        try:
            source_payload = _git_blob(record["source_ref"], legacy_path)
        except subprocess.CalledProcessError:
            errors.append(f"source ref lacks {legacy_path}")
            continue
        expected_hash = record["sha256"]
        expected_bytes = record["bytes"]
        if (
            len(source_payload) != expected_bytes
            or _sha256_bytes(source_payload) != expected_hash
        ):
            errors.append(f"source ref provenance differs for {legacy_path}")
        rewrite = locator_records.get(record["path"])
        if rewrite is not None and (
            rewrite.get("before_sha256") != expected_hash
            or rewrite.get("before_bytes") != expected_bytes
        ):
            errors.append(f"runtime-locator source differs for {record['path']}")
        current_hash = rewrite["after_sha256"] if rewrite is not None else expected_hash
        current_bytes = (
            rewrite["after_bytes"] if rewrite is not None else expected_bytes
        )
        if not current.is_file():
            errors.append(f"current record is missing: {record['path']}")
        elif (
            current.stat().st_size != current_bytes or _sha256(current) != current_hash
        ):
            errors.append(f"current record differs: {record['path']}")
    if migration.get("record_count") != len(migration.get("records", [])):
        errors.append("migration record_count does not match records")
    return {
        "passed": not errors,
        "records": len(migration.get("records", [])),
        "errors": errors,
    }


def _study_toml(study: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "schema_version = 1",
        f"id = {_quote(study['id'])}",
        f"title = {_quote(study['title'])}",
        f"chapter = {_quote(study['chapter'])}",
        f"order = {study['order']}",
        f"status = {_quote(study['status'])}",
        'review_state = "indexed"',
        f"question = {_quote(study['question'])}",
        f"finding = {_quote(study['finding'])}",
        f"boundary = {_quote(study['boundary'])}",
        "",
    ]
    prefix = f"studies/{study['id']}/"
    for record in sorted(records, key=lambda value: value["legacy_path"]):
        local_path = record["path"].removeprefix(prefix)
        lines.extend(_record_lines(record, local_path=local_path))
    return "\n".join(lines).rstrip() + "\n"


def _synthesis_toml(records: list[dict[str, Any]]) -> str:
    lines = [
        "schema_version = 1",
        'id = "current-model-synthesis"',
        'title = "Current model evidence synthesis"',
        'review_state = "indexed"',
        'scope = "A provisional human-readable organization of the frozen model evidence, diagnostic lineage, closed candidate families, and remaining claim boundaries."',
        'working_claim = "Sparse relational evidence feeds a meta-learned global assembly state and a causally distinct direct-fidelity state with broader evidence admission; the current record also preserves unresolved global-policy and transport boundaries."',
        'boundary = "This first refactor changes organization and provenance paths only. It does not promote a new scientific estimand, erase negative results, or assert that the current reading order is the final paper argument."',
        'registry = "../studies/registry.toml"',
        'history = "history.toml"',
        'frozen_reference = "frozen/manifest.json"',
        'figure_root = "figures"',
        "",
    ]
    for record in sorted(records, key=lambda value: value["legacy_path"]):
        local_path = record["path"].removeprefix("synthesis/")
        lines.extend(_record_lines(record, local_path=local_path))
    return "\n".join(lines).rstrip() + "\n"


def _history_toml() -> str:
    return """schema_version = 1

[[releases]]
id = "frozen-evidence-overlay-v1"
status = "frozen"
git_tag = "liu-mainline-v1"
commit = "2e0bdd86f4ee8f247157df5eb748131470246317"
manifest = "frozen/manifest.json"
note = "The internal Liu v1 identity and content remain historical provenance; the active repository no longer uses a Liu-named structural namespace."

[[migrations]]
id = "flat-records-v1"
source_commit = "fb32095df70b1265b5d14b8eda3be6cb65036c6a"
map = "../studies/migrations/flat-records-v1.json"
mode = "byte_preserving_physical_relocation"
"""


def apply_migration() -> dict[str, Any]:
    catalog, migration = build_plan()
    for record in migration["records"]:
        source = ROOT / record["legacy_path"]
        destination = ROOT / record["path"]
        if destination.exists():
            raise RuntimeError(
                f"migration destination already exists: {record['path']}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        if _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"post-move hash mismatch: {record['path']}")

    study_records: dict[str, list[dict[str, Any]]] = {}
    synthesis_records: list[dict[str, Any]] = []
    for record in migration["records"]:
        if record["owner_kind"] == "study":
            study_records.setdefault(record["owner_id"], []).append(record)
        else:
            synthesis_records.append(record)

    studies_by_id = {study["id"]: study for study in catalog["studies"]}
    for study_id, records in study_records.items():
        path = ROOT / "studies" / study_id / "study.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_study_toml(studies_by_id[study_id], records), encoding="utf-8")

    (ROOT / "studies").mkdir(parents=True, exist_ok=True)
    (ROOT / "studies" / "registry.toml").write_text(
        _registry_toml(catalog), encoding="utf-8"
    )
    MIGRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIGRATION_PATH.write_text(
        json.dumps(migration, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / "synthesis").mkdir(parents=True, exist_ok=True)
    (ROOT / "synthesis" / "manifest.toml").write_text(
        _synthesis_toml(synthesis_records), encoding="utf-8"
    )
    (ROOT / "synthesis" / "history.toml").write_text(_history_toml(), encoding="utf-8")

    generated_old = [
        ROOT / "docs" / "INDEX.md",
        ROOT / "benchmarks" / "INDEX.md",
        ROOT / "results" / "INDEX.md",
    ]
    for path in generated_old:
        if path.exists():
            path.unlink()
    old_catalog_root = ROOT / "research" / "liu"
    if old_catalog_root.exists():
        shutil.rmtree(old_catalog_root)
    for directory in (
        ROOT / "research",
        ROOT / "mainlines" / "liu_v1" / "artifacts",
        ROOT / "mainlines" / "liu_v1",
        ROOT / "mainlines",
        ROOT / "docs" / "assets" / "liu_mainline_v1",
        ROOT / "docs" / "assets",
        ROOT / "docs",
        ROOT / "benchmarks",
        ROOT / "results",
    ):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()

    return {
        "passed": True,
        "records": len(migration["records"]),
        "studies": len(study_records),
        "synthesis_records": len(synthesis_records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "audit"))
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_migration()
    elif args.command == "plan":
        _, migration = build_plan()
        result = {
            "passed": True,
            "records": len(migration["records"]),
            "studies": len(
                {
                    record["owner_id"]
                    for record in migration["records"]
                    if record["owner_kind"] == "study"
                }
            ),
            "synthesis_records": sum(
                record["owner_kind"] == "synthesis" for record in migration["records"]
            ),
        }
    else:
        result = apply_migration()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
