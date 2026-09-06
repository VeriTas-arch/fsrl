# Repository tests and static validation

Tests mirror the maintained package ownership and include unit, architecture,
scientific-contract, provenance, workflow, packaging, and repository-layout
checks. They validate execution and contracts; a passing suite does not promote
a scientific claim.

## Complete bounded suite

For changes requiring full engineering validation under the
[repository guide](../AGENTS.md#validation-boundary), run the complete suite
through the process-group-owning runtime:

```bash
direnv exec . python -m fsrl.infra.test_runtime
```

The runner applies a timeout, owns an independent process group, and cleans its
children on timeout or interruption.

Full engineering validation also includes these static and source-quality gates:

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
direnv exec . python -m fsrl.infra.test_runtime --timeout 60 \
  --framework unittest -- tests.infra.test_documentation_contract \
  tests.infra.test_study_registry.StudyRegistryTests.test_active_human_docs_have_live_local_links_and_python_modules -v
```

While iterating, run the smallest tests that exercise the changed contract.
Choose broader checks and affected registry/workflow/figure validators using
the repository guide. A documentation-only change needs the documentation/link
checks above and `git diff --check`. Reuse a passing check when its relevant
files, inputs, dependencies, and environment are unchanged; a broader suite can
cover an identical standalone check without another invocation.
