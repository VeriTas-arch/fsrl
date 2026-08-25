# Hidden learned-relation residual audit

> [!NOTE]
> This navigation page is generated from `studies/hidden_residual_audit/study.toml`. The current
> `review_state = "indexed"` means the records are organized and checked,
> but the prose is intentionally provisional pending the second synthesis pass.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `hidden_residual_audit`

## Scientific role

**Question.** Is an existing cross-relation correctness direction present in response hidden state but suppressed by W_out?

**Finding.** A small direct-enriched residual exists, but no shared correctness direction is hidden from the current readout; the simplest readout-only route is rejected.

**Claim boundary.** The result does not show that persistent local information is absent at earlier operator states.

## Frozen records

- `registered_contract` — [benchmarks/hidden_residual_audit_v1.json](records/benchmarks/hidden_residual_audit_v1.json) (`sha256:cf7f10b3c152`)
- `report` — [docs/hidden_residual_audit_v1.md](records/docs/hidden_residual_audit_v1.md) (`sha256:e0dc5ec20c7a`)
- `frozen_result` — [results/hidden_residual_audit_v1.json](records/results/hidden_residual_audit_v1.json) (`sha256:627577ed0ea2`)

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
