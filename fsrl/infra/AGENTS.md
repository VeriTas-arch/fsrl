# Infrastructure and execution guide

This file applies to `fsrl/infra/`.

Navigation: [package guide](../AGENTS.md) · [study guide](../../studies/AGENTS.md)
· [provenance-tool guide](../../tools/AGENTS.md)

## Runtime policy

- Formal training and scientific evaluation run through
  `python -m fsrl.infra.formal_runtime`. It requires a visible GPU and bounds
  PyTorch intra-op and inter-op work to one CPU thread.
- Use GPU execution when it materially accelerates neural work. CPU is
  appropriate for lightweight tests, exact enumeration, data checks, and
  bootstrap summaries when more efficient.
- Formal meta-training uses one contiguous host-to-device transfer per trial
  and `torch.compile(..., fullgraph=True, mode="default")`. Compiler settings,
  device identity, and source hashes are part of the execution lock.
- Tests run through `python -m fsrl.infra.test_runtime`, which owns an
  independent process group and cleans it on timeout or interruption.
- Diagnose repeated failures and orphaned CPU use from the exact command,
  working directory, parent/process group, age, and trigger parameters before
  terminating or changing lifecycle code.

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

Runtime, provenance, or process-lifecycle changes require focused infra tests
plus the complete bounded test suite. Exercise destructive or timeout paths in
temporary outputs, never on registered artifacts.
