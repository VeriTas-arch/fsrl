# Item-count transport

> [!NOTE]
> This navigation page is generated from `studies/item_count_transport/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `transported`
- **Review state:** `indexed`
- **Study ID:** `item_count_transport`

## Scientific role

**Question.** Does the functional asymmetry transport out of distribution to six- and ten-item Liu-style cycles?

**Finding.** All primary links pass at N=6, 8, and 10 across all three development backbones, including exact local ledger reconstruction and P/a double dissociation.

**Claim boundary.** N covaries with support duration, query count, and direct-query fraction; global policy degrades with size, so this is not arbitrary-size scaling.

## Frozen records

- `registered_contract` — [benchmarks/liu_item_count_transport_v1.json](records/benchmarks/liu_item_count_transport_v1.json) (`sha256:236c4076a421`)
- `execution_lock` — [benchmarks/liu_item_count_transport_v1.lock.json](records/benchmarks/liu_item_count_transport_v1.lock.json) (`sha256:a29b72cf573d`)
- `report` — [docs/liu_item_count_transport_v1.md](records/docs/liu_item_count_transport_v1.md) (`sha256:3e134573bed7`)
- `frozen_result` — [results/liu_item_count_transport_v1.json](records/results/liu_item_count_transport_v1.json) (`sha256:2429ef71c849`)

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
