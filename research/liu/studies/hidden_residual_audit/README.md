# Hidden learned-relation residual audit

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `valid_negative`
- **Study ID:** `hidden_residual_audit`
- **Chapter:** Direct local fidelity

## Scientific role

**Question.** Is an existing cross-relation correctness direction present in response hidden state but suppressed by W_out?

**Finding.** A small direct-enriched residual exists, but no shared correctness direction is hidden from the current readout; the simplest readout-only route is rejected.

**Claim boundary.** The result does not show that persistent local information is absent at earlier operator states.

## Canonical files

- `registered_contract` — [benchmarks/hidden_residual_audit_v1.json](../../../../benchmarks/hidden_residual_audit_v1.json)
- `report` — [docs/hidden_residual_audit_v1.md](../../../../docs/hidden_residual_audit_v1.md)
- `frozen_result` — [results/hidden_residual_audit_v1.json](../../../../results/hidden_residual_audit_v1.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
