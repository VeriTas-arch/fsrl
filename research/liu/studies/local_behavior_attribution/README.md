# Local behavioral non-rescue attribution

> [!NOTE]
> This page is generated from `research/liu/catalog.json`. Edit the
> catalog, then run `direnv exec . python -m fsrl.liu_catalog build`.
> Historical files remain canonical at their original paths.

[Back to Liu research guide](../../README.md)

- **Status:** `supporting`
- **Study ID:** `local_behavior_attribution`
- **Chapter:** Direct local fidelity

## Scientific role

**Question.** Why can a large direct-causal local rescue produce only a tiny sampled learned-accuracy change?

**Finding.** Stable-omitted cells dominate learned error mass, retained cells are mostly near ceiling, and the local trace removes retained exact error while sampled accuracy remains endpoint-sensitive.

**Claim boundary.** The historical 0.65 rule still misses narrowly; the result cannot be relabeled as the original pilot PASS or repaired by increasing gain.

## Canonical files

- `registered_contract` — [benchmarks/local_behavior_attribution_v2_3.json](../../../../benchmarks/local_behavior_attribution_v2_3.json)
- `execution_lock` — [benchmarks/local_behavior_attribution_v2_3.lock.json](../../../../benchmarks/local_behavior_attribution_v2_3.lock.json)
- `report` — [docs/local_behavior_attribution_v2_3.md](../../../../docs/local_behavior_attribution_v2_3.md)
- `frozen_result` — [results/local_behavior_attribution_v2_3.json](../../../../results/local_behavior_attribution_v2_3.json)

## Path policy

The files above remain canonical at their registered historical paths. This
capsule is the stable human-facing home for the study. A future study may put
its canonical files inside its capsule from inception, but relocating these
frozen files would require a separately versioned provenance migration.
