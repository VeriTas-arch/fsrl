# Operator-output semantics

<!-- fsrl-doc role=generated-navigation source=studies/operator_output_semantics/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/operator_output_semantics/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `operator_output_semantics`

## Scientific role

**Question.** At what stage does a correctness-aligned local operator signal become
misexpressed?

**Finding.** A and J_b A are correctness-aligned, but exact finite-amplitude tanh
degrades the signal and reverses H>A, localizing the missing link to nonlinear
expression.

**Claim boundary.** The result does not license a new readout, activation change, or
global scalar gain.

## Frozen records

- `registered_contract` —
  [benchmarks/operator_output_semantics_v1.json](records/benchmarks/operator_output_semantics_v1.json)
  (`sha256:1ed8a134268f`)
- `report` —
  [docs/operator_output_semantics_v1.md](records/docs/operator_output_semantics_v1.md)
  (`sha256:0a74602f28d9`)
- `frozen_result` —
  [results/operator_output_semantics_v1.json](records/results/operator_output_semantics_v1.json)
  (`sha256:b38f577ad83f`)

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
