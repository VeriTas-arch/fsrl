# Scalar-history reduction v2

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `valid_negative`
- **Study ID:** `dual_state_reduction_v2`
- **Chapter:** Algorithmic compression

## Scientific role

**Question.** Does adding low-capacity scalar history close global update amount?

**Finding.** Scalar history is insufficient, while preserving evidence that update amount depends on history.

**Claim boundary.** The negative does not deny history dependence; it rejects the registered scalar closure.

## Canonical files

- `registered_contract` — [benchmarks/dual_state_reduced_algorithm_v2.json](../../../../benchmarks/dual_state_reduced_algorithm_v2.json)
- `execution_lock` — [benchmarks/dual_state_reduced_algorithm_v2.lock.json](../../../../benchmarks/dual_state_reduced_algorithm_v2.lock.json)
- `report` — [docs/dual_state_reduced_algorithm_v2.md](../../../../docs/dual_state_reduced_algorithm_v2.md)
- `frozen_result` — [results/dual_state_reduced_algorithm_v2.json](../../../../results/dual_state_reduced_algorithm_v2.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
