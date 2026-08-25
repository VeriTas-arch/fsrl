# Functional fast-weight latent sufficiency

> [!NOTE]
> This navigation page is generated from `studies/functional_fast_weight_latent_sufficiency/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `functional_fast_weight_latent_sufficiency`

## Scientific role

**Question.** Does alpha times P contain a stable cross-episode linear coordinate sufficient beyond current potential and evidence?

**Finding.** The full-P oracle fails held-out generalization and worsens prediction in every backbone; the registered linear sufficiency audit is negative.

**Claim boundary.** This is a readout upper bound, not an autonomous latent rollout or proof that no nonlinear episode-specific coordinate exists.

## Frozen records

- `registered_contract` — [benchmarks/functional_fast_weight_latent_sufficiency_v1.json](records/benchmarks/functional_fast_weight_latent_sufficiency_v1.json) (`sha256:8c6edfc57a15`)
- `execution_lock` — [benchmarks/functional_fast_weight_latent_sufficiency_v1.lock.json](records/benchmarks/functional_fast_weight_latent_sufficiency_v1.lock.json) (`sha256:47cd34715be2`)
- `repair_contract` — [benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.json](records/benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.json) (`sha256:ec0ed94d8cef`)
- `repair_lock` — [benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.lock.json](records/benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.lock.json) (`sha256:1e67af184b81`)
- `report` — [docs/functional_fast_weight_latent_sufficiency_v1.md](records/docs/functional_fast_weight_latent_sufficiency_v1.md) (`sha256:f2cbd9ccddaa`)
- `supporting_artifact` — [results/functional_fast_weight_latent_sufficiency_v1.fit.npz](records/results/functional_fast_weight_latent_sufficiency_v1.fit.npz) (`sha256:5f3b7096879e`)
- `frozen_result` — [results/functional_fast_weight_latent_sufficiency_v1.json](records/results/functional_fast_weight_latent_sufficiency_v1.json) (`sha256:e0f3bc7e2366`)

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
