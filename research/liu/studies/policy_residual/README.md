# First-order policy residual

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `valid_negative`
- **Study ID:** `policy_residual`
- **Chapter:** Direct local fidelity

## Scientific role

**Question.** Is preserving the existing first-order branch through one low-capacity policy correction sufficient for local fidelity?

**Finding.** The natural residual causally moves H>A and F>A toward correctness with state-query specificity, but aggregate rescue and control specificity fail and six correct relations decline.

**Claim boundary.** Preserve the causal residual fingerprint, but close low-capacity response-expression correction as a sufficient mechanism.

## Canonical files

- `artifact_lock` — [benchmarks/policy_residual_pilot_v2_2.artifact_lock.json](../../../../benchmarks/policy_residual_pilot_v2_2.artifact_lock.json)
- `registered_contract` — [benchmarks/policy_residual_pilot_v2_2.json](../../../../benchmarks/policy_residual_pilot_v2_2.json)
- `execution_lock` — [benchmarks/policy_residual_pilot_v2_2.lock.json](../../../../benchmarks/policy_residual_pilot_v2_2.lock.json)
- `report` — [docs/policy_residual_pilot_v2_2.md](../../../../docs/policy_residual_pilot_v2_2.md)
- `frozen_result` — [results/policy_residual_pilot_v2_2.json](../../../../results/policy_residual_pilot_v2_2.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
