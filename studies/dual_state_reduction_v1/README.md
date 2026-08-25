# Potential-transition reduction v1

> [!NOTE]
> This navigation page is generated from `studies/dual_state_reduction_v1/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `dual_state_reduction_v1`

## Scientific role

**Question.** Can current global potential and evidence close the next potential transition?

**Finding.** The registered rank-two potential-transition reduction is insufficient.

**Claim boundary.** This rejects the tested reduction, not every nonlinear or trajectory-dependent compact global state.

## Frozen records

- `registered_contract` — [benchmarks/dual_state_reduced_algorithm_v1.json](records/benchmarks/dual_state_reduced_algorithm_v1.json) (`sha256:1de79c278f3d`)
- `execution_lock` — [benchmarks/dual_state_reduced_algorithm_v1.lock.json](records/benchmarks/dual_state_reduced_algorithm_v1.lock.json) (`sha256:739da286667d`)
- `repair_lock` — [benchmarks/dual_state_reduced_algorithm_v1.repair1.lock.json](records/benchmarks/dual_state_reduced_algorithm_v1.repair1.lock.json) (`sha256:2561287af962`)
- `repair_lock` — [benchmarks/dual_state_reduced_algorithm_v1.repair2.lock.json](records/benchmarks/dual_state_reduced_algorithm_v1.repair2.lock.json) (`sha256:cff6df9310e6`)
- `repair_lock` — [benchmarks/dual_state_reduced_algorithm_v1.repair3.lock.json](records/benchmarks/dual_state_reduced_algorithm_v1.repair3.lock.json) (`sha256:79ee811e463f`)
- `report` — [docs/dual_state_reduced_algorithm_v1.md](records/docs/dual_state_reduced_algorithm_v1.md) (`sha256:ae952b871695`)
- `frozen_result` — [results/dual_state_reduced_algorithm_v1.json](records/results/dual_state_reduced_algorithm_v1.json) (`sha256:573b56ea2128`)
- `supporting_artifact` — [results/dual_state_reduced_algorithm_v1.trajectories.npz](records/results/dual_state_reduced_algorithm_v1.trajectories.npz) (`sha256:3bc5f874b459`)

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
