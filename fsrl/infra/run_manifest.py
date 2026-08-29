"""Prospective run-directory ownership and manifest lifecycle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from fsrl.infra.file_contracts import describe_file


@dataclass
class ProspectiveRun:
    """Own one new execution directory and its atomically updated manifest."""

    output_dir: Path
    workflow_id: str
    execution_id: str
    producer: dict[str, Any]
    resolved_config: dict[str, Any]

    @classmethod
    def start(
        cls,
        output_dir: Path | str,
        *,
        workflow_id: str,
        execution_id: str,
        producer: dict[str, Any],
        resolved_config: dict[str, Any],
    ) -> ProspectiveRun:
        destination = Path(output_dir)
        if not workflow_id or not execution_id:
            raise ValueError("workflow_id and execution_id must be non-empty")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"prospective run directory already exists: {destination}"
            ) from error
        run = cls(
            output_dir=destination,
            workflow_id=workflow_id,
            execution_id=execution_id,
            producer=producer,
            resolved_config=resolved_config,
        )
        run._write_manifest("running")
        return run

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run.json"

    def complete(self) -> None:
        self._write_manifest("complete")

    def fail(self, error: BaseException) -> None:
        self._write_manifest(
            "failed",
            error={"type": type(error).__name__, "message": str(error)},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if error is None:
            self.complete()
        else:
            self.fail(error)
        return False

    def _files(self) -> list[dict[str, Any]]:
        return [
            describe_file(path, relative_to=self.output_dir)
            for path in sorted(self.output_dir.rglob("*"))
            if path.is_file()
            and path != self.manifest_path
            and path.name != ".run.json.tmp"
        ]

    def _write_manifest(
        self, lifecycle_state: str, *, error: dict[str, str] | None = None
    ) -> None:
        files = self._files()
        payload: dict[str, Any] = {
            "document_type": "fsrl.run_manifest",
            "schema_version": 1,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "lifecycle_state": lifecycle_state,
            "producer": self.producer,
            "resolved_config": self.resolved_config,
            "file_count": len(files),
            "bytes": sum(int(file["bytes"]) for file in files),
            "files": files,
        }
        if error is not None:
            payload["error"] = error
        temporary = self.output_dir / ".run.json.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, self.manifest_path)
