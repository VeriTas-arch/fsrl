# Minimal metric-error relational learner

<!-- fsrl-doc role=generated-navigation source=studies/minimal_relational_learner/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/minimal_relational_learner/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `confirmed`
- **Review state:** `indexed`
- **Study ID:** `minimal_relational_learner`

## Scientific role

**Question.** Can a compact online score learner, with or without direct relational
memory, explain the Liu behavioral map?

**Finding.** All three paired training streams yield score_only_sufficient: the
independently fitted two-scalar, 15-state learner passes competence and all nine
qualitative behavior rules. All six fits pass those rules, but each matches only three
of nine frozen quantitative classifiers. The trace passes all five local-support rules
and improves omitted probability, while reducing nonlearned probability; it is useful
but not required for this registered qualitative sufficiency. The fixed comparison is
complete and frozen.

**Claim boundary.** Sufficiency for the frozen qualitative criteria under imposed
additive geometry and inherited encoding/temperature, not quantitative Liu equivalence,
universal minimality, old-network trajectory compression, or human confirmation. Three
paired training streams are not independent random backbones; no post-result tuning or
seed expansion.

## Frozen records

- `registered_contract` —
  [studies/minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.json](records/benchmarks/minimal_relational_learner_v1.json)
  (`sha256:6586f2e51eef`)
- `execution_lock` —
  [studies/minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.source_lock.json](records/benchmarks/minimal_relational_learner_v1.source_lock.json)
  (`sha256:ee28e3d49b1f`)
- `artifact_lock` —
  [studies/minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.artifact_lock.json](records/benchmarks/minimal_relational_learner_v1.artifact_lock.json)
  (`sha256:0b0e64ae8d1e`)
- `report` —
  [studies/minimal_relational_learner/records/reports/minimal_relational_learner_v1.interpretation.md](records/reports/minimal_relational_learner_v1.interpretation.md)
  (`sha256:0fbe87ee75e5`)
- `report` —
  [studies/minimal_relational_learner/records/reports/minimal_relational_learner_v1.md](records/reports/minimal_relational_learner_v1.md)
  (`sha256:5162b90db488`)
- `frozen_result` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.json](records/results/minimal_relational_learner_v1.json)
  (`sha256:35530ca80c95`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2111.score_only.npz](records/results/minimal_relational_learner_v1.seed-2111.score_only.npz)
  (`sha256:a41b5d063b77`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2111.score_trace.npz](records/results/minimal_relational_learner_v1.seed-2111.score_trace.npz)
  (`sha256:8137be61dcaf`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2112.score_only.npz](records/results/minimal_relational_learner_v1.seed-2112.score_only.npz)
  (`sha256:9147f5951fcc`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2112.score_trace.npz](records/results/minimal_relational_learner_v1.seed-2112.score_trace.npz)
  (`sha256:ff84a6effbaa`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2113.score_only.npz](records/results/minimal_relational_learner_v1.seed-2113.score_only.npz)
  (`sha256:dfde651510a8`)
- `supporting_artifact` —
  [studies/minimal_relational_learner/records/results/minimal_relational_learner_v1.seed-2113.score_trace.npz](records/results/minimal_relational_learner_v1.seed-2113.score_trace.npz)
  (`sha256:cd0a64593c92`)

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
