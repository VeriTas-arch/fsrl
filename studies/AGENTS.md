# Registered study guide

This file applies to `studies/`.

Navigation: [repository guide](../AGENTS.md) · [study portal](README.md) ·
[registry](registry.toml) · [current mainline](../workflows/relational_model/README.md)

## Authorities and layout

- `registry.toml` owns study discovery, presentation order, allowed statuses,
  storage policy, and the ordered migration sequence.
- Each flat `studies/<study-id>/study.toml` owns that experiment's title,
  question, status, finding, claim boundary, source reference, implementation,
  and record hashes.
- Each generated `README.md` is a human capsule. Edit `study.toml` and rebuild;
  do not hand-edit generated prose.
- `records/` contains byte-preserved protocols, locks, results, reports, and
  compact assets. Status and narrative role are metadata, not directory names,
  so promotion or rejection must not move a study.

## Evidence rules

1. Every record has one study owner and an exact repository-relative locator,
   byte count, and hash.
2. Never rewrite a frozen record to update a path, command, wording, threshold,
   or outcome. Current navigation explains historical commands separately.
3. Preserve positive, negative, mixed, unresolved, transported, and deferred
   results. Registry validity is structural and does not itself establish a
   scientific claim.
4. New evidence records must identify the protocol, estimand, controls, seed
   scope, artifacts, result commit, uncertainty, and exact claim boundary.
5. Shared maintained code may appear in multiple study contracts, but an
   artifact remains owned by exactly one study.

## Migration and storage

- Physical evidence moves occur only through a new append-only migration file
  under `migrations/`. Verify every source against its recorded source commit
  and preserve bytes exactly.
- Active code and prose use the final locator resolved through the complete
  migration chain. Legacy identifiers remain searchable provenance.
- Track contracts, reports, compact tables, and canonical results in Git.
  Records above the registry review threshold require explicit storage review;
  new payloads above the hard limit require a registered content-addressed
  bundle, release asset, or Git LFS backend.

Any manifest, registry, migration, or generated-view change must pass the
registry check, active-link checks, migration audits, and `git diff --check`.
