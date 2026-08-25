# Presentation-order transport

> [!NOTE]
> This navigation page is generated from `studies/presentation_order_transport/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `transported`
- **Review state:** `indexed`
- **Study ID:** `presentation_order_transport`

## Scientific role

**Question.** Does the mechanism persist when the same physical support trials are reordered?

**Finding.** All registered links transport across random, clustered, and reversed schedules; the local ledger is exactly commutative while P_T remains quantitatively order-sensitive.

**Claim boundary.** Mechanism transport is not behavioral or P_T invariance.

## Frozen records

- `registered_contract` — [benchmarks/liu_presentation_order_transport_v1.json](records/benchmarks/liu_presentation_order_transport_v1.json) (`sha256:b294a8cf23d3`)
- `execution_lock` — [benchmarks/liu_presentation_order_transport_v1.lock.json](records/benchmarks/liu_presentation_order_transport_v1.lock.json) (`sha256:2d9d2ed6f55c`)
- `report` — [docs/liu_presentation_order_transport_v1.md](records/docs/liu_presentation_order_transport_v1.md) (`sha256:7b7a88f32a04`)
- `frozen_result` — [results/liu_presentation_order_transport_v1.json](records/results/liu_presentation_order_transport_v1.json) (`sha256:d13f0e1ff448`)

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
