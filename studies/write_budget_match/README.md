# Actual stage write budgets versus modulation organization

<!-- fsrl-doc role=generated-navigation source=studies/write_budget_match/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/write_budget_match/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `unresolved`
- **Review state:** `indexed`
- **Study ID:** `write_budget_match`

## Scientific role

**Question.** After matching actual closed-loop L1 writes in each active phase, do stage
constants recover the frozen affine RNN prediction loss?

**Finding.** All six generic-only calibrations meet the four phase/observation budget
targets; gains and raw calibration records locked before behavior.

**Claim boundary.** Frozen affine networks and exposed panels; budget-only calibration.
No from-scratch training, reliability-gate necessity, main-model promotion or compact
requalification.

## Frozen records

- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2531-clean.npz](records/artifacts/calibration/2531-clean.npz)
  (`sha256:5eb2c077e573`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2531-noisy.npz](records/artifacts/calibration/2531-noisy.npz)
  (`sha256:fcf79c679eb5`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2532-clean.npz](records/artifacts/calibration/2532-clean.npz)
  (`sha256:2dbe32d46115`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2532-noisy.npz](records/artifacts/calibration/2532-noisy.npz)
  (`sha256:d1d73ec131a8`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2533-clean.npz](records/artifacts/calibration/2533-clean.npz)
  (`sha256:abc56c51feb4`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/2533-noisy.npz](records/artifacts/calibration/2533-noisy.npz)
  (`sha256:4431336f19ff`)
- `frozen_result` —
  [studies/write_budget_match/records/artifacts/calibration/result.json](records/artifacts/calibration/result.json)
  (`sha256:ee79eac7ac6c`)
- `supporting_artifact` —
  [studies/write_budget_match/records/artifacts/calibration/run.json](records/artifacts/calibration/run.json)
  (`sha256:bc20667bdc7e`)
- `artifact_lock` —
  [studies/write_budget_match/records/benchmarks/calibration_lock.json](records/benchmarks/calibration_lock.json)
  (`sha256:8c2a8938600f`)
- `registered_contract` —
  [studies/write_budget_match/records/benchmarks/protocol.json](records/benchmarks/protocol.json)
  (`sha256:9dc0bacebc5b`)
- `supporting_artifact` —
  [studies/write_budget_match/records/benchmarks/qualification.json](records/benchmarks/qualification.json)
  (`sha256:5d00b2873a05`)
- `execution_lock` —
  [studies/write_budget_match/records/benchmarks/source_lock.json](records/benchmarks/source_lock.json)
  (`sha256:339f8a8cd1bc`)

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
