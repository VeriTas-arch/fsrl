# Infrastructure and execution guide

This file applies to `fsrl/infra/`.

Navigation: [package guide](../AGENTS.md) · [study guide](../../studies/AGENTS.md)
· [provenance-tool guide](../../tools/AGENTS.md)

## Runtime policy

- Formal workflows use `python -m fsrl.infra.formal_runtime` as required by
  their registered execution contracts. Preserve the locked device and thread
  requirements.
- Use GPU execution when it materially accelerates neural work. CPU is
  appropriate for lightweight tests, exact enumeration, data checks, and
  bootstrap summaries when more efficient.
- Current training profiles are documented in the [training guide](../training/README.md).
  Record effective compiler settings, iteration boundaries, device/thread
  configuration, runtime snapshot, and source identities in existing execution
  provenance. Profile changes require scoped parity and performance checks;
  never substitute current defaults for a frozen execution lock.
- Tests run through `python -m fsrl.infra.test_runtime`, which owns an
  independent process group and cleans it on timeout or interruption.
- Diagnose repeated failures and orphaned CPU use from the exact command,
  working directory, parent/process group, age, and trigger parameters before
  terminating or changing lifecycle code.
- One formal-runtime command owns one process-local validation session. Pure
  validators may reuse exact results within that session, but caches must not
  cross commands or hide changes to branch, HEAD, or worktree cleanliness.
  Network and Git-blob witnesses may be reused after those local checks pass.
  This runtime-cache boundary does not require rerunning completed engineering
  checks; reuse those under the [root validation boundary](../../AGENTS.md#validation-boundary).

## Registry and provenance

- `study_registry.py` validates evidence structure and renders navigation; it
  must not silently repair, reinterpret, or rewrite scientific records.
- Historical record identifiers resolve through the ordered migration chain.
  Preserve both the original legacy identifier and the one current locator.
- Generated navigation must be deterministic and must derive current claims
  from the workflow instead of maintaining a second prose authority.
- Frozen source identities are `(path, sha256)` pairs verified from Git blobs
  and witness commits. Never copy historical source back into the live package
  merely to satisfy a locator.

Use the [root validation boundary](../../AGENTS.md#validation-boundary).
Runtime/process behavior or shared provenance-contract changes require focused
infra tests and the complete engineering suite. Exercise destructive or timeout
paths in temporary outputs, never on registered artifacts.
