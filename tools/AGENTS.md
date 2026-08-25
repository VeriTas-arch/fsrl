# Provenance tool guide

This file applies to `tools/`.

Navigation: [repository guide](../AGENTS.md) · [study guide](../studies/AGENTS.md)
· [infrastructure guide](../fsrl/infra/AGENTS.md)

- Provenance tools audit or perform explicitly versioned migrations. Default to
  read-only `check` or `audit` behavior; never mutate registered evidence as a
  side effect of inspection.
- Migration tools resolve the ordered migration chain, verify source commits,
  hashes, byte counts, and final locators, and preserve legacy identifiers.
- Active-locator rewrites operate only on maintained code and current prose.
  Frozen records and historical snapshots remain byte-preserved.
- Source-provenance indexing records historical `(path, sha256)` identities and
  witness commits; it must not copy redundant source into the active package.
- Keep each migration tool narrow and rerunnable. Do not add a generic cleanup
  framework or infer scientific ownership from filenames alone.

Run every affected tool in non-mutating check/audit mode, the registry and
active-document tests, Ruff, and `git diff --check` before committing.
