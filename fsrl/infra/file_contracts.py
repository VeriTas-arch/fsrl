"""Typed contracts for analysis files without rewriting scientific payloads."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fsrl.infra.provenance import file_sha256
from fsrl.paths import EXTERNAL_DATA_ROOT, REPO_ROOT, RUNS_ROOT

_FORMAT_RULES = (
    (".schema.json", "json_schema", "application/schema+json", None),
    (".json.gz", "gzip_json", "application/gzip", ".json.gz"),
    (".tar.zst", "tar_zstd", "application/zstd", ".tar.zst"),
    (".jsonl", "json_lines", "application/x-ndjson", ".jsonl"),
    (".json", "json", "application/json", ".json"),
    (".npz", "numpy_npz", "application/vnd.numpy.npz", ".npz"),
    (".dat", "legacy_pytorch_state_dict", "application/x-pytorch", ".pth"),
    (".pth", "pytorch_state_dict", "application/x-pytorch", ".pth"),
    (".pt", "pytorch_program", "application/x-pytorch", ".pt"),
    (".csv", "csv", "text/csv", ".csv"),
    (".toml", "toml", "application/toml", ".toml"),
    (".md", "markdown", "text/markdown", ".md"),
    (".pdf", "pdf", "application/pdf", ".pdf"),
    (".svg", "svg", "image/svg+xml", ".svg"),
    (".png", "png", "image/png", ".png"),
    (".txt", "text", "text/plain", None),
)
_ID_COMPONENT = re.compile(r"[^a-z0-9]+")


def safe_relative_path(value: str) -> PurePosixPath:
    """Return one normalized repository-relative locator or fail closed."""

    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"path must be a safe relative locator: {value}")
    return path


def classify_path(path: Path | str) -> dict[str, str | bool | None]:
    """Classify a supported analysis file from its explicit compound suffix."""

    name = Path(path).name.lower()
    for suffix, format_name, media_type, prospective_suffix in _FORMAT_RULES:
        if name.endswith(suffix):
            return {
                "format": format_name,
                "media_type": media_type,
                "legacy_format": format_name == "legacy_pytorch_state_dict",
                "prospective_suffix": prospective_suffix,
            }
    raise ValueError(f"unsupported analysis-file format: {path}")


def stable_record_id(owner_kind: str, owner_id: str, legacy_path: str) -> str:
    """Derive a stable logical ID from the immutable legacy identifier."""

    legacy = safe_relative_path(legacy_path).as_posix().lower()
    slug = _ID_COMPONENT.sub("_", legacy).strip("_")
    return f"{owner_kind}.{owner_id}.{slug}"


def describe_file(path: Path, *, relative_to: Path) -> dict[str, Any]:
    """Describe one materialized file with portable identity metadata."""

    relative = path.resolve().relative_to(relative_to.resolve()).as_posix()
    classification = classify_path(path)
    return {
        "path": relative,
        **classification,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_dataset_manifest(path: Path | str) -> dict[str, Any]:
    """Load a machine-readable external-dataset manifest."""

    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def dataset_file(manifest: dict[str, Any], file_id: str) -> dict[str, Any]:
    """Resolve one dataset member by its stable logical ID."""

    matches = [
        entry for entry in manifest.get("files", []) if entry.get("id") == file_id
    ]
    if len(matches) != 1:
        raise KeyError(f"dataset file ID must resolve exactly once: {file_id}")
    return matches[0]


def _loads_strict_json(value: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(value, parse_constant=reject_constant)


def _strict_json(path: Path) -> None:
    _loads_strict_json(path.read_text(encoding="utf-8"))


def _csv_contract(path: Path) -> tuple[int, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return sum(1 for _ in reader), header


def _pytorch_zip_contract(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if not any(name.endswith("/data.pkl") for name in archive.namelist()):
            raise ValueError(f"checkpoint is not a PyTorch state-dict archive: {path}")


def _pytorch_program_contract(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"PyTorch program is not a ZIP archive: {path}")


def validate_payload(path: Path | str) -> dict[str, Any]:
    """Validate the structural envelope appropriate for one supported format."""

    source = Path(path)
    classification = classify_path(source)
    format_name = classification["format"]
    detail: dict[str, Any] = {}
    if format_name in {"json", "json_schema"}:
        _strict_json(source)
    elif format_name == "json_lines":
        records = 0
        with source.open(encoding="utf-8") as handle:
            for records, line in enumerate(handle, start=1):
                _loads_strict_json(line)
        detail["records"] = records
    elif format_name == "gzip_json":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            _loads_strict_json(handle.read())
    elif format_name == "csv":
        rows, columns = _csv_contract(source)
        detail.update({"rows": rows, "columns": columns})
    elif format_name == "numpy_npz":
        import numpy as np

        arrays = {}
        with np.load(source, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        detail["arrays"] = arrays
    elif format_name in {"legacy_pytorch_state_dict", "pytorch_state_dict"}:
        _pytorch_zip_contract(source)
    elif format_name == "pytorch_program":
        _pytorch_program_contract(source)
    return {"passed": True, **classification, **detail}


def validate_dataset_manifest(path: Path | str) -> dict[str, Any]:
    """Verify an external dataset manifest against immutable local source bytes."""

    manifest_path = Path(path)
    manifest = load_dataset_manifest(manifest_path)
    errors: list[str] = []
    if manifest.get("document_type") != "fsrl.external_dataset_manifest":
        errors.append("invalid dataset document_type")
    if manifest.get("schema_version") != 1:
        errors.append("dataset schema_version must be 1")
    entries = manifest.get("files", [])
    ids = [entry.get("id") for entry in entries]
    if len(ids) != len(set(ids)):
        errors.append("dataset file IDs must be unique")
    paths: list[str] = []
    checked = 0
    for entry in entries:
        try:
            relative = safe_relative_path(entry["path"])
            paths.append(relative.as_posix())
            source = manifest_path.parent / relative
            if not source.is_file():
                errors.append(f"missing dataset file: {relative.as_posix()}")
                continue
            observed = describe_file(source, relative_to=manifest_path.parent)
            validate_payload(source)
            if observed["sha256"] != entry.get("sha256"):
                errors.append(f"dataset hash mismatch: {relative.as_posix()}")
            if observed["bytes"] != entry.get("bytes"):
                errors.append(f"dataset byte count mismatch: {relative.as_posix()}")
            expected_format = entry.get("format")
            if observed["format"] != expected_format:
                errors.append(f"dataset format mismatch: {relative.as_posix()}")
            if expected_format == "csv":
                rows, columns = _csv_contract(source)
                if rows != entry.get("rows"):
                    errors.append(f"dataset row count mismatch: {relative.as_posix()}")
                if columns != entry.get("columns"):
                    errors.append(f"dataset columns mismatch: {relative.as_posix()}")
            checked += 1
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"invalid dataset entry {entry.get('id')}: {error}")
    if len(paths) != len(set(paths)):
        errors.append("dataset file paths must be unique")
    return {
        "passed": not errors,
        "errors": errors,
        "dataset_id": manifest.get("id"),
        "files": len(entries),
        "checked_files": checked,
    }


def _validate_prospective_lifecycle(
    manifest: dict[str, Any], errors: list[str]
) -> None:
    lifecycle_state = manifest.get("lifecycle_state")
    if lifecycle_state not in {
        "running",
        "complete",
        "failed",
        "materialized_compatibility_view",
    }:
        errors.append("prospective run manifest has invalid lifecycle_state")
    if lifecycle_state == "failed" and not isinstance(manifest.get("error"), dict):
        errors.append("failed prospective run manifest requires error")
    if lifecycle_state != "failed" and "error" in manifest:
        errors.append("only failed prospective runs may record error")


def _validate_prospective_run_header(
    manifest: dict[str, Any], errors: list[str]
) -> None:
    for field in ("workflow_id", "execution_id", "lifecycle_state"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            errors.append(f"prospective run manifest requires {field}")
    for field in ("producer", "resolved_config"):
        if not isinstance(manifest.get(field), dict):
            errors.append(f"prospective run manifest requires {field}")
    _validate_prospective_lifecycle(manifest, errors)


def _validate_legacy_run_header(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("lifecycle_state") != "historical_unclassified":
        errors.append("legacy run manifest lifecycle_state mismatch")
    if not isinstance(manifest.get("conversion_contract"), dict):
        errors.append("legacy run manifest requires conversion_contract")


def _validate_run_header(manifest: dict[str, Any], errors: list[str]) -> None:
    document_type = manifest.get("document_type")
    if document_type not in {"fsrl.run_manifest", "fsrl.legacy_run_manifest"}:
        errors.append("invalid run-manifest document_type")
    if manifest.get("schema_version") != 1:
        errors.append("run-manifest schema_version must be 1")
    if document_type == "fsrl.run_manifest":
        _validate_prospective_run_header(manifest, errors)
    elif document_type == "fsrl.legacy_run_manifest":
        _validate_legacy_run_header(manifest, errors)


def _validate_run_entries(
    manifest_path: Path,
    entries: list[dict[str, Any]],
    errors: list[str],
) -> tuple[list[str], int]:
    paths: list[str] = []
    checked = 0
    for entry in entries:
        try:
            relative = safe_relative_path(entry["path"])
            paths.append(relative.as_posix())
            source = manifest_path.parent / relative
            observed = describe_file(source, relative_to=manifest_path.parent)
            validate_payload(source)
            for field in ("format", "media_type", "bytes", "sha256"):
                if observed[field] != entry.get(field):
                    errors.append(f"run file {field} mismatch: {relative.as_posix()}")
            checked += 1
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"invalid run file entry: {error}")
    return paths, checked


def _materialized_run_paths(manifest_path: Path, manifest: dict[str, Any]) -> set[str]:
    if manifest.get("layout") == "legacy_loose_files":
        payload_files = [
            candidate
            for candidate in manifest_path.parent.iterdir()
            if candidate.is_file() and not candidate.name.endswith(".run.json")
        ]
    else:
        payload_files = [
            candidate
            for candidate in manifest_path.parent.rglob("*")
            if candidate.is_file()
            and candidate.name != "run.json"
            and not candidate.name.endswith(".run.json")
            and candidate.name != ".run.json.tmp"
        ]
    return {
        candidate.relative_to(manifest_path.parent).as_posix()
        for candidate in payload_files
    }


def _validate_run_file_set(
    manifest_path: Path,
    manifest: dict[str, Any],
    declared_paths: set[str],
    errors: list[str],
) -> None:
    if manifest.get("lifecycle_state") == "running":
        return
    actual_paths = _materialized_run_paths(manifest_path, manifest)
    for relative in sorted(actual_paths - declared_paths):
        errors.append(f"run file is not declared: {relative}")
    for relative in sorted(declared_paths - actual_paths):
        errors.append(f"declared run file is absent: {relative}")


def validate_run_manifest(path: Path | str) -> dict[str, Any]:
    """Verify a prospective or backfilled run manifest against local bytes."""

    manifest_path = Path(path)
    errors: list[str] = []
    try:
        manifest = _loads_strict_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"passed": False, "errors": [str(error)], "files": 0}
    if not isinstance(manifest, dict):
        return {
            "passed": False,
            "errors": ["run manifest must be a JSON object"],
            "files": 0,
        }
    _validate_run_header(manifest, errors)
    entries = manifest.get("files", [])
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        return {
            "passed": False,
            "errors": [*errors, "run manifest files must be a list of objects"],
            "execution_id": manifest.get("execution_id"),
            "files": 0,
            "checked_files": 0,
        }
    paths, checked = _validate_run_entries(manifest_path, entries, errors)
    if len(paths) != len(set(paths)):
        errors.append("run file paths must be unique")
    if manifest.get("file_count") != len(entries):
        errors.append("run file_count mismatch")
    if manifest.get("bytes") != sum(entry.get("bytes", 0) for entry in entries):
        errors.append("run byte total mismatch")
    _validate_run_file_set(manifest_path, manifest, set(paths), errors)
    return {
        "passed": not errors,
        "errors": errors,
        "execution_id": manifest.get("execution_id"),
        "files": len(entries),
        "checked_files": checked,
    }


def find_unmanifested_run_roots(runs_root: Path = RUNS_ROOT) -> list[Path]:
    """Find materialized run roots that have no owning manifest."""

    if not runs_root.is_dir():
        return []

    def contains_payload(directory: Path) -> bool:
        return any(
            path.is_file()
            and path.name != "run.json"
            and not path.name.endswith(".run.json")
            for path in directory.rglob("*")
        )

    missing = []
    root_payloads = [
        path
        for path in runs_root.iterdir()
        if path.is_file() and not path.name.endswith(".run.json")
    ]
    root_manifests = [
        path
        for path in runs_root.iterdir()
        if path.is_file() and path.name.endswith(".run.json")
    ]
    if root_payloads and not root_manifests:
        missing.append(runs_root)

    for workflow_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if (workflow_root / "run.json").is_file():
            continue
        execution_roots = [
            path
            for path in workflow_root.iterdir()
            if path.is_dir()
            and (contains_payload(path) or (path / "run.json").is_file())
        ]
        has_manifested_execution = any(
            (path / "run.json").is_file() for path in execution_roots
        )
        if not has_manifested_execution:
            if contains_payload(workflow_root):
                missing.append(workflow_root)
            continue
        if any(
            path.is_file()
            and path.name != "run.json"
            and not path.name.endswith(".run.json")
            for path in workflow_root.iterdir()
        ):
            missing.append(workflow_root)
        missing.extend(
            path for path in execution_roots if not (path / "run.json").is_file()
        )
    return missing


def audit_repository_file_contracts() -> dict[str, Any]:
    """Audit tracked dataset/catalog contracts and any materialized run manifests."""

    from fsrl.infra.record_catalog import CATALOG_PATH, check_record_catalog

    dataset = validate_dataset_manifest(EXTERNAL_DATA_ROOT / "liu2026" / "dataset.toml")
    catalog = check_record_catalog(CATALOG_PATH)
    run_paths = []
    if RUNS_ROOT.is_dir():
        run_paths = sorted(
            {
                *RUNS_ROOT.rglob("run.json"),
                *RUNS_ROOT.rglob("*.run.json"),
            }
        )
    run_checks = [validate_run_manifest(path) for path in run_paths]
    run_errors = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "errors": check["errors"],
        }
        for path, check in zip(run_paths, run_checks, strict=True)
        if not check["passed"]
    ]
    missing_run_roots = [
        path.relative_to(REPO_ROOT).as_posix() for path in find_unmanifested_run_roots()
    ]
    return {
        "passed": (
            dataset["passed"]
            and catalog["passed"]
            and not run_errors
            and not missing_run_roots
        ),
        "dataset": dataset,
        "record_catalog": catalog,
        "run_manifests": len(run_paths),
        "run_manifest_files": sum(check["files"] for check in run_checks),
        "run_errors": run_errors,
        "unmanifested_run_roots": missing_run_roots,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    return parser


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    result = audit_repository_file_contracts()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
