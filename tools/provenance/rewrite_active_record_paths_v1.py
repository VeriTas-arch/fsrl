"""Rewrite active Python path constructors to the migration-aware resolver.

Historical contracts and reports are deliberately excluded: their bytes are
frozen.  This script only updates executable source and tests that previously
constructed ``docs/``, ``benchmarks/``, or ``results/`` paths directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from fsrl.infra.study_registry import resolve_record

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = ROOT / "studies" / "migrations" / "flat-records-v1.json"
ROOT_EXPRESSION = re.compile(
    r'\bROOT\s*/\s*"(?P<root>docs|benchmarks|results)"\s*/\s*'
    r'"(?P<name>[^"\n]+)"'
)
PATH_EXPRESSION = re.compile(
    r'\bPath\(\s*"(?P<path>(?:docs|benchmarks|results)/[^"\n]+)"\s*\)'
)
TEST_LITERAL = re.compile(r'"(?P<path>(?:docs|benchmarks|results)/[^"\n]+)"')
RESOLVED_LITERAL = re.compile(
    r'resolve_record\(\s*"(?P<path>(?:docs|benchmarks|results)/[^"\n]+)"\s*\)'
)


def _import_resolver(content: str, path: Path) -> str:
    if "resolve_record(" not in content:
        return content
    if path.is_relative_to(ROOT / "fsrl"):
        statement = "from fsrl.infra.study_registry import resolve_record"
        if statement in content:
            return content
        marker = "from __future__ import annotations\n"
        if marker not in content:
            raise RuntimeError(f"cannot place resolver import in {path}")
        return content.replace(marker, marker + "\n" + statement + "\n", 1)

    statement = "from fsrl.infra.study_registry import resolve_record"
    if statement in content:
        return content
    first_fsrl = re.search(r"^from fsrl\.", content, flags=re.MULTILINE)
    if first_fsrl is not None:
        return (
            content[: first_fsrl.start()]
            + statement
            + "\n"
            + content[first_fsrl.start() :]
        )
    marker = "import unittest\n"
    if marker not in content:
        raise RuntimeError(f"cannot place resolver import in {path}")
    return content.replace(marker, marker + statement + "\n", 1)


def _rewrite(content: str, *, test_literals: bool) -> str:
    content = ROOT_EXPRESSION.sub(
        lambda match: f'resolve_record("{match.group("root")}/{match.group("name")}")',
        content,
    )
    content = PATH_EXPRESSION.sub(
        lambda match: f'resolve_record("{match.group("path")}")', content
    )
    if test_literals:
        protected: dict[str, str] = {}

        def protect_resolved(match: re.Match[str]) -> str:
            marker = f"__RESOLVED_RECORD_{len(protected)}__"
            protected[marker] = match.group(0)
            return marker

        content = RESOLVED_LITERAL.sub(protect_resolved, content)
        content = TEST_LITERAL.sub(
            lambda match: f'resolve_record("{match.group("path")}")', content
        )
        for marker, resolved in protected.items():
            content = content.replace(marker, resolved)
    return content


def rewrite(*, apply: bool) -> dict[str, object]:
    migration = json.loads(MIGRATION_PATH.read_text(encoding="utf-8"))
    registered = {record["legacy_path"] for record in migration["records"]}
    changed: list[str] = []
    unknown: list[str] = []
    for root in (ROOT / "fsrl", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path.name == "liu_catalog.py":
                continue
            original = path.read_text(encoding="utf-8")
            rewritten = _rewrite(
                original,
                test_literals=(
                    path.parent.name == "tests"
                    and path.name != "test_study_registry.py"
                ),
            )
            if rewritten == original:
                continue
            rewritten = _import_resolver(rewritten, path)
            for value in re.findall(
                r'resolve_record\("((?:docs|benchmarks|results)/[^"]+)"\)',
                rewritten,
            ):
                if value not in registered:
                    unknown.append(f"{path.relative_to(ROOT)}:{value}")
            changed.append(path.relative_to(ROOT).as_posix())
            if apply:
                path.write_text(rewritten, encoding="utf-8")
    for path in (ROOT / "AGENTS.md",):
        original = path.read_text(encoding="utf-8")
        rewritten = original
        for record in sorted(
            migration["records"],
            key=lambda value: len(value["legacy_path"]),
            reverse=True,
        ):
            current = resolve_record(record["path"]).relative_to(ROOT).as_posix()
            rewritten = re.sub(
                rf"(?<!records/){re.escape(record['legacy_path'])}",
                current,
                rewritten,
            )
        if rewritten != original:
            changed.append(path.relative_to(ROOT).as_posix())
            if apply:
                path.write_text(rewritten, encoding="utf-8")
    if unknown:
        raise RuntimeError(f"rewritten paths are absent from migration: {unknown}")
    return {"passed": apply or not changed, "apply": apply, "changed": changed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    args = parser.parse_args(argv)
    result = rewrite(apply=args.command == "apply")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
