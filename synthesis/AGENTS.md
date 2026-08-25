# Human synthesis and reporting guide

This file applies to `synthesis/`. Figure and snapshot work also follows the
nearest guide under [`figures/`](figures/AGENTS.md) or
[`snapshots/`](snapshots/AGENTS.md).

Navigation: [repository guide](../AGENTS.md) · [synthesis portal](README.md) ·
[current mainline](../workflows/relational_model/README.md) ·
[study portal](../studies/README.md) · [snapshot guide](snapshots/AGENTS.md)

## Responsibilities

- `manifest.toml` owns synthesis navigation, diagnostic groupings, review
  state, and snapshot references. It derives the current working claim and
  boundary from the workflow rather than duplicating them.
- `README.md` is a generated human view. It may organize diagnostic lineage,
  closed families, unresolved boundaries, and deferred routes, but it must not
  replace study-level estimands or the workflow claim graph.
- `history.toml` records reporting releases and migrations.
- `source-provenance.toml` maps historical source identities to immutable Git
  blobs and witness commits.
- `snapshots/<id>/` is immutable historical reporting state and follows the
  snapshot-specific preservation rules.

Human-facing prose should let a new researcher locate the method, evidence,
result, uncertainty, and claim boundary without an AI summary. It need not be
the final manuscript argument during the indexed review stage, and it must not
hide diagnostic or negative evidence for narrative neatness.

The human-mechanism and new-data program remains deferred. Presentation work
must not imply that the current model establishes human neural implementation.

Rebuild synthesis through the registry builder, then run registry, workflow,
snapshot, figure, and active-link checks. A reporting snapshot migration also
requires a byte-preserving versioned migration ledger.
