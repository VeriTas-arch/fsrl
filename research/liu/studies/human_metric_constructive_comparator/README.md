# Human-only metric constructive comparator

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `valid_negative`
- **Study ID:** `human_metric_constructive_comparator`
- **Chapter:** Global-policy mismatch and comparator boundary

## Scientific role

**Question.** Can a task-faithful, metric-preserving constructive comparator generalize from preregistered humans to the held-out replication cohort?

**Finding.** The candidate reproduces held-out human distance slope but fails the reliable distance-residualized pair field and cannot become a neural target.

**Claim boundary.** Comparator search on the same responses is closed; the first failed execution is noninterpretable, and human-mechanism validation is deferred.

## Canonical files

- `registered_contract` — [benchmarks/human_metric_constructive_comparator_v1.json](../../../../benchmarks/human_metric_constructive_comparator_v1.json)
- `execution_lock` — [benchmarks/human_metric_constructive_comparator_v1.lock.json](../../../../benchmarks/human_metric_constructive_comparator_v1.lock.json)
- `frozen_parameters` — [benchmarks/human_metric_constructive_comparator_v1.parameters.json](../../../../benchmarks/human_metric_constructive_comparator_v1.parameters.json)
- `execution_lock` — [benchmarks/human_metric_constructive_comparator_v1.parameters.lock.json](../../../../benchmarks/human_metric_constructive_comparator_v1.parameters.lock.json)
- `repair_contract` — [benchmarks/human_metric_constructive_comparator_v1.repair1.json](../../../../benchmarks/human_metric_constructive_comparator_v1.repair1.json)
- `repair_lock` — [benchmarks/human_metric_constructive_comparator_v1.repair1.lock.json](../../../../benchmarks/human_metric_constructive_comparator_v1.repair1.lock.json)
- `report` — [docs/human_metric_constructive_comparator_v1.md](../../../../docs/human_metric_constructive_comparator_v1.md)
- `frozen_result` — [results/human_metric_constructive_comparator_v1.json](../../../../results/human_metric_constructive_comparator_v1.json)
- `noninterpretable_attempt` — [results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json](../../../../results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
