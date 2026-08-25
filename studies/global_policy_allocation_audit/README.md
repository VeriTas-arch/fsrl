# Equal-energy allocation audit

> [!NOTE]
> This navigation page is generated from `studies/global_policy_allocation_audit/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `confirmed`
- **Review state:** `indexed`
- **Study ID:** `global_policy_allocation_audit`

## Scientific role

**Question.** Where does equal-norm neural-versus-posterior additive allocation differ, and which cells exactly carry Q_shape?

**Finding.** The exact delta-g to delta-p to per-pair q bridge localizes a replicated, policy-effective allocation fingerprint.

**Claim boundary.** The first execution is noninterpretable and preserved as such; comparator-relative localization is not evidence that the neural network incorrectly encodes uncertainty.

## Frozen records

- `registered_contract` — [benchmarks/global_policy_allocation_audit_v1.json](records/benchmarks/global_policy_allocation_audit_v1.json) (`sha256:b14836fbd523`)
- `execution_lock` — [benchmarks/global_policy_allocation_audit_v1.lock.json](records/benchmarks/global_policy_allocation_audit_v1.lock.json) (`sha256:68bcd6bace7b`)
- `repair_lock` — [benchmarks/global_policy_allocation_audit_v1.repair1.lock.json](records/benchmarks/global_policy_allocation_audit_v1.repair1.lock.json) (`sha256:2d163fc7c999`)
- `report` — [docs/global_policy_allocation_audit_v1.md](records/docs/global_policy_allocation_audit_v1.md) (`sha256:229378dc536a`)
- `frozen_result` — [results/global_policy_allocation_audit_v1.json](records/results/global_policy_allocation_audit_v1.json) (`sha256:82147bad41b0`)
- `noninterpretable_attempt` — [results/global_policy_allocation_audit_v1_attempt1_noninterpretable.json](records/results/global_policy_allocation_audit_v1_attempt1_noninterpretable.json) (`sha256:8949aa755fbe`)

## Provenance rule

Files under `records/` are byte-preserving relocations. Their former paths,
hashes, sizes, and source ref are recorded in `study.toml` and the global
migration map. New interpretation belongs in this capsule or `synthesis/`;
the frozen records themselves are not rewritten.

Add a `figures/` directory only when this study has a promoted, reproducible
study-level figure. Cross-study paper figures belong in `synthesis/figures/`.
