# Historical reporting snapshot guide

This file applies to `synthesis/snapshots/`.

Navigation: [synthesis guide](../AGENTS.md) · [snapshot portal](README.md) ·
[current mainline](../../workflows/relational_model/README.md)

- Every `snapshots/<id>/` directory is an immutable reporting object. Do not
  edit internal names, commands, links, prose, manifests, locks, or artifacts
  to match the current checkout.
- Maintain current replay instructions in `snapshots/README.md`, outside each
  frozen object. Historical inconsistencies remain provenance unless a
  separately versioned audit records corruption.
- A new reporting freeze gets a new sibling ID and a complete manifest; it
  never replaces or incrementally edits an existing snapshot.
- Moving a snapshot requires a byte-preserving migration ledger with source
  commit, hash, byte count, old locator, and new locator for every file.
- Do not treat the historical snapshot's internal namespace as the current
  scientific or package namespace.

Run frozen-evidence status and verification, registry and migration audits,
and active-link checks after snapshot-index or locator changes.
