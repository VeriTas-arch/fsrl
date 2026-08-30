# Local behavioral non-rescue attribution

<!-- fsrl-doc role=generated-navigation source=studies/local_behavior_attribution/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/local_behavior_attribution/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `local_behavior_attribution`

## Scientific role

**Question.** Why can a large direct-causal local rescue produce only a tiny sampled
learned-accuracy change?

**Finding.** Stable-omitted cells dominate learned error mass, retained cells are mostly
near ceiling, and the local trace removes retained exact error while sampled accuracy
remains endpoint-sensitive.

**Claim boundary.** The historical 0.65 rule still misses narrowly; the result cannot be
relabeled as the original pilot PASS or repaired by increasing gain.

## Frozen records

- `registered_contract` —
  [benchmarks/local_behavior_attribution_v2_3.json](records/benchmarks/local_behavior_attribution_v2_3.json)
  (`sha256:ea892ce00c85`)
- `execution_lock` —
  [benchmarks/local_behavior_attribution_v2_3.lock.json](records/benchmarks/local_behavior_attribution_v2_3.lock.json)
  (`sha256:d9b477faddef`)
- `report` —
  [docs/local_behavior_attribution_v2_3.md](records/docs/local_behavior_attribution_v2_3.md)
  (`sha256:ac6345651dc2`)
- `frozen_result` —
  [results/local_behavior_attribution_v2_3.json](records/results/local_behavior_attribution_v2_3.json)
  (`sha256:599b379d1d1a`)

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
