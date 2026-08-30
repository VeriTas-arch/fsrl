# Human-only metric constructive comparator

<!-- fsrl-doc role=generated-navigation source=studies/human_metric_constructive_comparator/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/human_metric_constructive_comparator/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `human_metric_constructive_comparator`

## Scientific role

**Question.** Can a task-faithful, metric-preserving constructive comparator generalize
from preregistered humans to the held-out replication cohort?

**Finding.** The candidate reproduces held-out human distance slope but fails the
reliable distance-residualized pair field and cannot become a neural target.

**Claim boundary.** Comparator search on the same responses is closed; the first failed
execution is noninterpretable, and human-mechanism validation is deferred.

## Frozen records

- `registered_contract` —
  [benchmarks/human_metric_constructive_comparator_v1.json](records/benchmarks/human_metric_constructive_comparator_v1.json)
  (`sha256:3ad0587dd1da`)
- `execution_lock` —
  [benchmarks/human_metric_constructive_comparator_v1.lock.json](records/benchmarks/human_metric_constructive_comparator_v1.lock.json)
  (`sha256:dd5661b026ec`)
- `frozen_parameters` —
  [benchmarks/human_metric_constructive_comparator_v1.parameters.json](records/benchmarks/human_metric_constructive_comparator_v1.parameters.json)
  (`sha256:12fa39b79c2c`)
- `execution_lock` —
  [benchmarks/human_metric_constructive_comparator_v1.parameters.lock.json](records/benchmarks/human_metric_constructive_comparator_v1.parameters.lock.json)
  (`sha256:3082307b32d2`)
- `repair_contract` —
  [benchmarks/human_metric_constructive_comparator_v1.repair1.json](records/benchmarks/human_metric_constructive_comparator_v1.repair1.json)
  (`sha256:bb5747e48e80`)
- `repair_lock` —
  [benchmarks/human_metric_constructive_comparator_v1.repair1.lock.json](records/benchmarks/human_metric_constructive_comparator_v1.repair1.lock.json)
  (`sha256:7a9b3460021e`)
- `report` —
  [docs/human_metric_constructive_comparator_v1.md](records/docs/human_metric_constructive_comparator_v1.md)
  (`sha256:32d34311533b`)
- `frozen_result` —
  [results/human_metric_constructive_comparator_v1.json](records/results/human_metric_constructive_comparator_v1.json)
  (`sha256:8c77ee6c4020`)
- `noninterpretable_attempt` —
  [results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json](records/results/human_metric_constructive_comparator_v1_attempt1_noninterpretable.json)
  (`sha256:81cf13087ca7`)

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
