# Scalar-history reduction v2

> [!NOTE]
> This navigation page is generated from `studies/dual_state_reduction_v2/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `dual_state_reduction_v2`

## Scientific role

**Question.** Does adding low-capacity scalar history close global update amount?

**Finding.** Scalar history is insufficient, while preserving evidence that update amount depends on history.

**Claim boundary.** The negative does not deny history dependence; it rejects the registered scalar closure.

## Frozen records

- `registered_contract` — [benchmarks/dual_state_reduced_algorithm_v2.json](records/benchmarks/dual_state_reduced_algorithm_v2.json) (`sha256:e8e5497a607b`)
- `execution_lock` — [benchmarks/dual_state_reduced_algorithm_v2.lock.json](records/benchmarks/dual_state_reduced_algorithm_v2.lock.json) (`sha256:0e12b87e99ba`)
- `report` — [docs/dual_state_reduced_algorithm_v2.md](records/docs/dual_state_reduced_algorithm_v2.md) (`sha256:6c51d516bdf8`)
- `frozen_result` — [results/dual_state_reduced_algorithm_v2.json](records/results/dual_state_reduced_algorithm_v2.json) (`sha256:4326ec769951`)

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
