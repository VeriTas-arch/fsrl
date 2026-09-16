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

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `pl_direct_training`

## Scientific role

**Question.** Can optimizer-equivalent direct training recover the cleaned P/L
organization, and does removing normalized episode time preserve its competence, causal
links, and inherited behavior profile?

**Finding.** Fresh direct training made both conditions competent in all three
development seeds, and the no-time candidate passed paired noninferiority on all six
registered endpoints. Both conditions nevertheless failed retained direct-local fidelity
in every seed (bootstrap lower bounds 0.00383--0.00505 versus the registered 0.01
threshold) and failed the frozen quantitative behavior gate. The registered outcome is
training_parameterization_failure, so confirmation stops without tuning.

**Claim boundary.** This rejects the fixed direct-training recipe as a route to a
reportable P/L mainline. It does not identify normalized time as the cause because the
no-time candidate was noninferior and both conditions failed the same mandatory
mechanism link; it does not reject the no-time architecture generally, authorize
protocol tuning, or authorize historical checkpoint deletion.

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
- `frozen_result` —
  [studies/pl_direct_training/records/results/pl_direct_training_v1.development.json](records/results/pl_direct_training_v1.development.json)
  (`sha256:386beeb82e30`)
- `report` —
  [studies/pl_direct_training/records/reports/pl_direct_training_v1.development.md](records/reports/pl_direct_training_v1.development.md)
  (`sha256:5048c359bd49`)

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
