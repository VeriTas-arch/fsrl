# Operator-output semantics

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `supporting`
- **Study ID:** `operator_output_semantics`
- **Chapter:** Direct local fidelity

## Scientific role

**Question.** At what stage does a correctness-aligned local operator signal become misexpressed?

**Finding.** A and J_b A are correctness-aligned, but exact finite-amplitude tanh degrades the signal and reverses H>A, localizing the missing link to nonlinear expression.

**Claim boundary.** The result does not license a new readout, activation change, or global scalar gain.

## Canonical files

- `registered_contract` — [benchmarks/operator_output_semantics_v1.json](../../../../benchmarks/operator_output_semantics_v1.json)
- `report` — [docs/operator_output_semantics_v1.md](../../../../docs/operator_output_semantics_v1.md)
- `frozen_result` — [results/operator_output_semantics_v1.json](../../../../results/operator_output_semantics_v1.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
