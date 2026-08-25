"""Validate or render a schema-driven research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import (
    check_rendered_readme,
    load_workflow,
    render_workflow,
    validate_workflow,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "render"])
    parser.add_argument("workflow", type=Path)
    return parser.parse_args(args)


def main(args=None) -> None:
    parsed = parse_args(args)
    workflow = load_workflow(parsed.workflow)
    validation = validate_workflow(workflow)
    if parsed.action == "render":
        parsed.workflow.with_name("README.md").write_text(
            render_workflow(workflow), encoding="utf-8"
        )
    else:
        validation["render"] = check_rendered_readme(parsed.workflow)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
