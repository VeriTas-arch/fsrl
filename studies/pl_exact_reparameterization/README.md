# Exact P/L structural reparameterization

<!-- fsrl-doc role=generated-navigation source=studies/pl_exact_reparameterization/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/pl_exact_reparameterization/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `frozen_contract`
- **Review state:** `indexed`
- **Study ID:** `pl_exact_reparameterization`

## Scientific role

**Question.** Can the confirmed P/L computation be expressed through a clean 32-channel
task interface and exact packed state without changing its supported computation?

**Finding.** A prospective deterministic mapping, timestep-accounting repair, and source
lock are frozen for the two confirmed P/L networks. They retain the legacy time drive,
four-step support microcircuit, and complete active episode horizon while removing only
redundant task-interface and state representation; qualification has not yet run.

**Claim boundary.** This study can establish checkpoint-preserving structural
equivalence only. It does not establish that episode time is unnecessary, authorize
historical checkpoint deletion, or expose a no-time training outcome.

## Frozen records

- `registered_contract` —
  [studies/pl_exact_reparameterization/records/benchmarks/pl_exact_reparameterization_v1.json](records/benchmarks/pl_exact_reparameterization_v1.json)
  (`sha256:799fd9a4ca18`)
- `repair_contract` —
  [studies/pl_exact_reparameterization/records/benchmarks/pl_exact_reparameterization_v1.repair1.json](records/benchmarks/pl_exact_reparameterization_v1.repair1.json)
  (`sha256:fbf85c720119`)
- `execution_lock` —
  [studies/pl_exact_reparameterization/records/benchmarks/pl_exact_reparameterization_v1.execution_lock.json](records/benchmarks/pl_exact_reparameterization_v1.execution_lock.json)
  (`sha256:3497e685b5a3`)

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
