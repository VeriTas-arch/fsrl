"""Audit the execution and ownership refactor against its pinned baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit_refactor_equivalence_v1 import (
    load_contract as _load_contract,
)
from .audit_refactor_equivalence_v1 import (
    main as _main,
)
from .audit_refactor_equivalence_v1 import (
    run_audit as _run_audit,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT / "workflows" / "relational_model" / "refactor_equivalence_v2.json"
)


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    return _load_contract(path)


def run_audit(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    return _run_audit(contract_path)


def main(argv: list[str] | None = None) -> int:
    return _main(argv, default_contract=DEFAULT_CONTRACT)


if __name__ == "__main__":
    raise SystemExit(main())
