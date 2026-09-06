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

Select checks from the [root validation boundary](../AGENTS.md#validation-boundary)
and reuse successful checks under its unchanged-input rule. Run the smallest
affected tests while iterating and any required broader checks once the change
is ready. A documentation-only change needs the documentation/link checks above
and `git diff --check`. Commit, push, and handoff are not test triggers; inspect
Git status and references for delivery without a second closing audit.
