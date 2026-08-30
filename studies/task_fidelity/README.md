# Liu task and human benchmark contract

<!-- fsrl-doc role=generated-navigation source=studies/task_fidelity/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/task_fidelity/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `frozen_contract`
- **Review state:** `indexed`
- **Study ID:** `task_fidelity`

## Scientific role

**Question.** What information and response boundary must every model-facing Liu
analysis preserve?

**Finding.** The task contract preserves item identity, signed displayed magnitude,
passive four-presentation learning, all-pair testing, and no test feedback.

**Claim boundary.** Classic rewarded adjacent-pair TI and list linking are context, not
descriptions of the Liu observation interface.

## Frozen records

- `registered_contract` —
  [benchmarks/liu_human_exact_v1.json](records/benchmarks/liu_human_exact_v1.json)
  (`sha256:c1e3efbbbc03`)
- `registered_contract` — [benchmarks/liu_v1.json](records/benchmarks/liu_v1.json)
  (`sha256:91cd3358b159`)
- `registered_contract` — [benchmarks/liu_v2.json](records/benchmarks/liu_v2.json)
  (`sha256:7428f87476c6`)

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
