# Repository tests and static validation

Tests mirror the maintained package ownership and include unit, architecture,
scientific-contract, provenance, workflow, packaging, and repository-layout
checks. They validate execution and contracts; a passing suite does not promote
a scientific claim.

## Complete bounded suite

Run the complete suite through the process-group-owning runtime:

```bash
direnv exec . python -m fsrl.infra.test_runtime
```

The runner applies a timeout, owns an independent process group, and cleans its
children on timeout or interruption.

Run the static and source-quality gates separately:

```bash
direnv exec . basedpyright
direnv exec . python -m tools.quality.complexity_budget
direnv exec . ruff check fsrl tests tools reproductions
direnv exec . ruff format --check fsrl tests tools reproductions
git diff --check
```

BasedPyright and Ruff read their configuration from `pyproject.toml`.

## Focused tests

Use the same bounded runner for a selected unittest module:

```bash
direnv exec . python -m fsrl.infra.test_runtime --timeout 60 \
  --framework unittest -- tests.infra.test_study_registry -v
```

The documentation contract checks AGENTS scope inheritance, generated-page
markers, heading and fence structure, local fragments, and top-level navigation:

```bash
direnv exec . python -m unittest tests.infra.test_documentation_contract -v
```

While iterating, run the smallest test module that exercises the changed
contract. A completed structural change still requires the repository-wide
checks in the [repository guide](../AGENTS.md#validation-boundary).
