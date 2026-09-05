# Matched single-stage versus staged training

<!-- fsrl-doc role=generated-navigation source=studies/joint_training_strategy/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/joint_training_strategy/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `frozen_contract`
- **Review state:** `indexed`
- **Study ID:** `joint_training_strategy`

## Scientific role

**Question.** Can a shared query objective preserve global assembly, direct fidelity,
and Liu behavior without staged optimization?

**Finding.** Protocol and complete implementation are prospectively locked after CPU
contract tests and non-Liu CUDA output, gradient, optimizer-update, and query-readout
parity checks. The three paired seeds, equal-episode-budget comparator, fixed dual
evidence admission, and separate competence, noninferiority, behavior, and cost
decisions are unchanged. Final training artifacts and scientific evaluation are pending.

**Claim boundary.** This new candidate tests a fixed training recipe under imposed P/L
structural priors. It does not replace frozen v2.4 evidence, establish minimal
architecture or human neural learning, or identify an order-only effect at equal FLOPs.

## Frozen records

- `registered_contract` —
  [studies/joint_training_strategy/records/benchmarks/joint_training_strategy_v1.json](records/benchmarks/joint_training_strategy_v1.json)
  (`sha256:af6fe7ccd078`)
- `execution_lock` —
  [studies/joint_training_strategy/records/benchmarks/joint_training_strategy_v1.execution_lock.json](records/benchmarks/joint_training_strategy_v1.execution_lock.json)
  (`sha256:c2baf711692c`)

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
