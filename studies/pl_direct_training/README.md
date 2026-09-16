# Direct training of the cleaned no-time P/L model

<!-- fsrl-doc role=generated-navigation source=studies/pl_direct_training/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/pl_direct_training/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `frozen_contract`
- **Review state:** `indexed`
- **Study ID:** `pl_direct_training`

## Scientific role

**Question.** Can optimizer-equivalent direct training recover the cleaned P/L
organization, and does removing normalized episode time preserve its competence, causal
links, and inherited behavior profile?

**Finding.** All six paired development runs completed under the source-locked protocol
and their checkpoints, logs, optimizer counters, and stream fingerprints are
artifact-locked before evaluation. No scientific evaluation has been exposed.

**Claim boundary.** This design isolates normalized time within one fixed
direct-training recipe. Development cannot establish network-population confirmation,
failure cannot prove universal time necessity, and no historical checkpoint deletion is
authorized.

## Frozen records

- `registered_contract` —
  [studies/pl_direct_training/records/benchmarks/pl_direct_training_v1.json](records/benchmarks/pl_direct_training_v1.json)
  (`sha256:49b02c9275cd`)
- `repair_contract` —
  [studies/pl_direct_training/records/benchmarks/pl_direct_training_v1.repair1.json](records/benchmarks/pl_direct_training_v1.repair1.json)
  (`sha256:3cbd18ac36ce`)
- `execution_lock` —
  [studies/pl_direct_training/records/benchmarks/pl_direct_training_v1.execution_lock.json](records/benchmarks/pl_direct_training_v1.execution_lock.json)
  (`sha256:afc4db445acb`)
- `artifact_lock` —
  [studies/pl_direct_training/records/benchmarks/pl_direct_training_v1.development_artifact_lock.json](records/benchmarks/pl_direct_training_v1.development_artifact_lock.json)
  (`sha256:6cf9d0e2d61d`)

## Provenance rule

Files under `records/` are byte-preserving relocations. Their former paths,
hashes, sizes, and source ref are recorded in `study.toml` and the global
migration map. New interpretation belongs in this capsule or `synthesis/`;
the frozen records themselves are not rewritten.
Commands and relative links inside a frozen report describe its historical
checkout. Use the maintained workflow for current commands, or the snapshot
replay guide for an exact detached-worktree replay.

Add a `figures/` directory only when this study has a promoted, reproducible
study-level figure. Cross-study paper figures belong in `synthesis/figures/`.
