# Frozen v2.4 behavioral reproduction map

> [!NOTE]
> This navigation page is generated from `studies/behavior_reproduction_map/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `behavior_reproduction_map`

## Scientific role

**Question.** Which Liu behavioral phenomena are quantitatively reproduced and which remain mismatched by the final frozen model?

**Finding.** Six of nine registered phenomena are reproduced and three are qualitatively reproduced but quantitatively mismatched; none is absent.

**Claim boundary.** This is a competence and phenomenology map, not evidence that humans use the same internal mechanism or that the fit is complete.

## Frozen records

- `registered_contract` — [benchmarks/model_behavior_reproduction_map_v1.json](records/benchmarks/model_behavior_reproduction_map_v1.json) (`sha256:62f2176314b9`)
- `execution_lock` — [benchmarks/model_behavior_reproduction_map_v1.lock.json](records/benchmarks/model_behavior_reproduction_map_v1.lock.json) (`sha256:466604fb507f`)
- `report` — [docs/model_behavior_reproduction_map_v1.md](records/docs/model_behavior_reproduction_map_v1.md) (`sha256:11d7ef2f5c07`)
- `frozen_result` — [results/model_behavior_reproduction_map_v1.json](records/results/model_behavior_reproduction_map_v1.json) (`sha256:6c6096443519`)

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
