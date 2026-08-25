# External-paper reproduction guide

This file applies to `reproductions/`.

Navigation: [repository guide](../AGENTS.md) ·
[relational-learning capsule](relational_learning_2024/README.md)

- Each capsule isolates byte-locked upstream source, supplied checkpoints,
  provenance, and maintained teaching adapters from the active `fsrl` package.
- Do not import upstream notebooks or scripts into maintained model code. Patch
  a teaching adapter outside `upstream/` when a current runtime needs a fix.
- Treat `upstream/` and supplied checkpoints as immutable. Verify them against
  the capsule manifest after structural changes.
- Generated teaching figures and training outputs belong under ignored
  `artifacts/reproductions/<capsule>/`; they are not automatically registered
  scientific evidence or report figures.
- Exact historical-paper replay may require a detached historical worktree.
  Do not recreate obsolete root modules or source trees to make that replay
  look current.
- Ruff and current tests apply to maintained capsule adapters. Byte-preserved
  upstream source is checked for provenance and concrete defects, not silently
  reformatted or modernized.

Run the capsule verification and focused reproduction tests after changes.
