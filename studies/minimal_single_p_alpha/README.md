# Paired dense-alpha add-back to minimal single-P M2

<!-- fsrl-doc role=generated-navigation source=studies/minimal_single_p_alpha/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/minimal_single_p_alpha/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `minimal_single_p_alpha`

## Scientific role

**Question.** Holding the frozen M2 architecture, generic training streams,
optimization, observation process, and evaluation arrays fixed, does adding one
trainable dense expression-gain matrix alpha recover constrained Liu morphology without
error inflation?

**Finding.** Frozen M2-alpha result: 19/20 networks were stable_constructive, but 0/60
noisy units met constrained morphology and 0/60 passed all nine qualitative rows;
registered outcome no_constrained_rescue.

**Claim boundary.** Alpha is a slow trainable expression gain initialized exactly to
one; P remains the only episode-persistent plastic state. The paired complete-recipe
result cannot identify alpha and P separately or establish a biological mechanism.

## Frozen records

- `registered_contract` —
  [studies/minimal_single_p_alpha/records/benchmarks/minimal_single_p_alpha_v1.json](records/benchmarks/minimal_single_p_alpha_v1.json)
  (`sha256:1b8fd50b3e7a`)
- `artifact_lock` —
  [studies/minimal_single_p_alpha/records/benchmarks/model_lock.json](records/benchmarks/model_lock.json)
  (`sha256:5c64c9121b6a`)
- `validation_result` —
  [studies/minimal_single_p_alpha/records/benchmarks/qualification.json](records/benchmarks/qualification.json)
  (`sha256:aa406095814f`)
- `execution_lock` —
  [studies/minimal_single_p_alpha/records/benchmarks/source_input_lock.json](records/benchmarks/source_input_lock.json)
  (`sha256:c0baf1ab0aeb`)
- `report` —
  [studies/minimal_single_p_alpha/records/reports/minimal_single_p_alpha_v1.generic.md](records/reports/minimal_single_p_alpha_v1.generic.md)
  (`sha256:17264b30f560`)
- `report` —
  [studies/minimal_single_p_alpha/records/reports/minimal_single_p_alpha_v1.md](records/reports/minimal_single_p_alpha_v1.md)
  (`sha256:5d865096378c`)
- `frozen_result` —
  [studies/minimal_single_p_alpha/records/results/minimal_single_p_alpha_v1.generic.json](records/results/minimal_single_p_alpha_v1.generic.json)
  (`sha256:4f8ca7fd408b`)
- `frozen_result` —
  [studies/minimal_single_p_alpha/records/results/minimal_single_p_alpha_v1.json](records/results/minimal_single_p_alpha_v1.json)
  (`sha256:9cd55f19d5e1`)
- `supporting_artifact` —
  [studies/minimal_single_p_alpha/records/results/minimal_single_p_alpha_v1.pairs.npz](records/results/minimal_single_p_alpha_v1.pairs.npz)
  (`sha256:288afe5c113c`)
- `supporting_artifact` —
  [studies/minimal_single_p_alpha/records/results/minimal_single_p_alpha_v1.parameters.npz](records/results/minimal_single_p_alpha_v1.parameters.npz)
  (`sha256:5201218648a2`)

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
