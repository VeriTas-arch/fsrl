# Exact P/L retained cross-talk decomposition

<!-- fsrl-doc role=generated-navigation source=studies/pl_crosstalk_decomposition/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/pl_crosstalk_decomposition/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `pl_crosstalk_decomposition`

## Scientific role

**Question.** Can the frozen retained dual-minus-shared probability cost in clean P/L
seeds 3004--3006 be reconstructed exactly from omitted-source address overlap, scalar L
gain, and each network's fixed operating point?

**Finding.** The repaired exact decomposition passed every locked integrity check.
Gain-free omitted-write cross-talk was identical across seeds; seed 3006's threshold
crossing was a joint gain-by-operating-point boundary, with operating point accounting
for 84.8% and 97.9% of its retained-mean difference from seeds 3004 and 3005. The 3004
gain rescued the 3006 gate while the 3005 gain did not, and the 3006 gain passed at both
other operating points. Cross-talk was concentrated in few source collisions (median
top-one share 0.768, top-two share 1.000, effective source count 1.522).

**Claim boundary.** The result exactly attributes the frozen retained contrast to
omitted writes, fixed nonorthogonal addresses, scalar gain, and shared-margin operating
point. It does not revise seed 3006 or the parent all-seed failure, explain why joint
training learned those parameters, authorize P/L repair or transport, establish
biological stores, or turn a deterministic network comparison into population
prevalence.

## Frozen records

- `registered_contract` —
  [studies/pl_crosstalk_decomposition/records/benchmarks/pl_crosstalk_decomposition_v1.json](records/benchmarks/pl_crosstalk_decomposition_v1.json)
  (`sha256:0a8aa21c8798`)
- `execution_lock` —
  [studies/pl_crosstalk_decomposition/records/benchmarks/pl_crosstalk_decomposition_v1.execution_lock.json](records/benchmarks/pl_crosstalk_decomposition_v1.execution_lock.json)
  (`sha256:b9d5c6d62e46`)
- `noninterpretable_attempt` —
  [studies/pl_crosstalk_decomposition/records/results/pl_crosstalk_decomposition_v1.attempt1.json](records/results/pl_crosstalk_decomposition_v1.attempt1.json)
  (`sha256:c6515344b1bb`)
- `supporting_artifact` —
  [studies/pl_crosstalk_decomposition/records/artifacts/pl_crosstalk_decomposition_v1.attempt1.npz](records/artifacts/pl_crosstalk_decomposition_v1.attempt1.npz)
  (`sha256:4f1d36aad5e9`)
- `report` —
  [studies/pl_crosstalk_decomposition/records/reports/pl_crosstalk_decomposition_v1.attempt1.md](records/reports/pl_crosstalk_decomposition_v1.attempt1.md)
  (`sha256:f2cfc273bc0d`)
- `repair_contract` —
  [studies/pl_crosstalk_decomposition/records/benchmarks/pl_crosstalk_decomposition_v1.repair1.json](records/benchmarks/pl_crosstalk_decomposition_v1.repair1.json)
  (`sha256:5fbc3dde30e0`)
- `repair_lock` —
  [studies/pl_crosstalk_decomposition/records/benchmarks/pl_crosstalk_decomposition_v1.source_repair1.json](records/benchmarks/pl_crosstalk_decomposition_v1.source_repair1.json)
  (`sha256:b6e456597f25`)
- `frozen_result` —
  [studies/pl_crosstalk_decomposition/records/results/pl_crosstalk_decomposition_v1.json](records/results/pl_crosstalk_decomposition_v1.json)
  (`sha256:2302d9ddacc4`)
- `supporting_artifact` —
  [studies/pl_crosstalk_decomposition/records/artifacts/pl_crosstalk_decomposition_v1.npz](records/artifacts/pl_crosstalk_decomposition_v1.npz)
  (`sha256:3d9b51c8a850`)
- `report` —
  [studies/pl_crosstalk_decomposition/records/reports/pl_crosstalk_decomposition_v1.md](records/reports/pl_crosstalk_decomposition_v1.md)
  (`sha256:025f8c4619db`)

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
