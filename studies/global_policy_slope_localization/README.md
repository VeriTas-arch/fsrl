# Global-policy slope localization

> [!NOTE]
> This navigation page is generated from `studies/global_policy_slope_localization/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `global_policy_slope_localization`

## Scientific role

**Question.** Which algebraic part of the pure-global neural policy carries the excessive nonlearned distance slope?

**Finding.** The positive neural margin slope is already in the additive potential; the Hodge residual and fixed sigmoid are compressive rather than amplifying.

**Claim boundary.** The later common-unit audit rejects the stronger claim that neural additive norm is globally over-sharp relative to the posterior comparator.

## Frozen records

- `registered_contract` — [benchmarks/global_policy_slope_localization_v1.json](records/benchmarks/global_policy_slope_localization_v1.json) (`sha256:a32878f67132`)
- `execution_lock` — [benchmarks/global_policy_slope_localization_v1.lock.json](records/benchmarks/global_policy_slope_localization_v1.lock.json) (`sha256:4d2ddb28f275`)
- `report` — [docs/global_policy_slope_localization_v1.md](records/docs/global_policy_slope_localization_v1.md) (`sha256:cef52c9efe1d`)
- `frozen_result` — [results/global_policy_slope_localization_v1.json](records/results/global_policy_slope_localization_v1.json) (`sha256:a10060d6bd87`)

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
