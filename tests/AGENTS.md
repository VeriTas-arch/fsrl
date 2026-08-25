# Test suite guide

This file applies to `tests/`. Tests also follow the nearest guide for the code
under test.

Navigation: [repository guide](../AGENTS.md) · [package guide](../fsrl/AGENTS.md)

- Mirror maintained source ownership: core, tasks, training, evaluation,
  analysis, experiments, infrastructure, and workflows have corresponding test
  subtrees. Do not recreate a flat root test collection.
- Test scientific invariants and frozen boundaries, not only successful
  execution. Preserve checkpoint ABI, exact intervention identities, control
  conditions, seed contracts, numerical tolerances, and negative outcomes.
- Repository tests should validate authorities, links, migration chains,
  ownership, dependency direction, and generated-view determinism without
  treating structural PASS as scientific support.
- Historical records and snapshot prose are immutable inputs. Active-document
  checks may exclude those trees but must cover current guides and generated
  navigation.
- Keep fixtures small and deterministic. Never launch formal training or write
  into registered studies from a unit test; use temporary directories.
- A test that can hang or spawn children must run under the bounded test
  runtime and clean the complete process group on interruption.

Run focused unittest modules while iterating. Before a cross-package or
repository-structure change is complete, run
`direnv exec . python -m fsrl.infra.test_runtime` and verify no child process
remains.
