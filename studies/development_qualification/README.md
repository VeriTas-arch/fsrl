# Development and qualification record

<!-- fsrl-doc role=generated-navigation source=studies/development_qualification/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/development_qualification/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `historical`
- **Review state:** `indexed`
- **Study ID:** `development_qualification`

## Scientific role

**Question.** Did the initial candidate satisfy enough competence and causal gates to
justify formal study?

**Finding.** The early candidate established development evidence and qualification
infrastructure that motivated the frozen pilot and formal contracts.

**Claim boundary.** Single-seed development output is provenance, not formal multi-seed
confirmation or the current headline model.

## Frozen records

- `report` — [docs/development_plan.md](records/docs/development_plan.md)
  (`sha256:0a1722fc8053`)
- `frozen_result` —
  [results/dev_v2_seed1801_step1000.json](records/results/dev_v2_seed1801_step1000.json)
  (`sha256:7d76d7262fe4`)

## Retired historical assets

These files are intentionally absent from the current worktree. Their
exact bytes remain recoverable from the recorded Git source ref.

- `development_checkpoint_config` — `checkpoints/dev-v2-seed1801-step1000/config.json`
  (`sha256:fb6faae8f21a`, source `refs/tags/liu-mainline-v1`) — The configuration
  belongs to a superseded single-seed development candidate and is retained through Git
  provenance rather than as a root-level active asset.
- `development_checkpoint` — `checkpoints/dev-v2-seed1801-step1000/net.dat`
  (`sha256:0fb9f063ba8e`, source `refs/tags/liu-mainline-v1`) — The weights belong to a
  superseded single-seed development candidate and are not loaded by the maintained
  model, tests, workflows, or current evidence chain.

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
