# First-order policy residual

> [!NOTE]
> This navigation page is generated from `studies/policy_residual/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `policy_residual`

## Scientific role

**Question.** Is preserving the existing first-order branch through one low-capacity policy correction sufficient for local fidelity?

**Finding.** The natural residual causally moves H>A and F>A toward correctness with state-query specificity, but aggregate rescue and control specificity fail and six correct relations decline.

**Claim boundary.** Preserve the causal residual fingerprint, but close low-capacity response-expression correction as a sufficient mechanism.

## Frozen records

- `artifact_lock` — [benchmarks/policy_residual_pilot_v2_2.artifact_lock.json](records/benchmarks/policy_residual_pilot_v2_2.artifact_lock.json) (`sha256:f3c34e12b8e1`)
- `registered_contract` — [benchmarks/policy_residual_pilot_v2_2.json](records/benchmarks/policy_residual_pilot_v2_2.json) (`sha256:d53b98b2159d`)
- `execution_lock` — [benchmarks/policy_residual_pilot_v2_2.lock.json](records/benchmarks/policy_residual_pilot_v2_2.lock.json) (`sha256:6cf77a375d69`)
- `report` — [docs/policy_residual_pilot_v2_2.md](records/docs/policy_residual_pilot_v2_2.md) (`sha256:e777de862045`)
- `frozen_result` — [results/policy_residual_pilot_v2_2.json](records/results/policy_residual_pilot_v2_2.json) (`sha256:74d8150e2e53`)

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
