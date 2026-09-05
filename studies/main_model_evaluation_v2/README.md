# Claim-relative single-stage main-model evaluation

<!-- fsrl-doc role=generated-navigation source=studies/main_model_evaluation_v2/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/main_model_evaluation_v2/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `supporting`
- **Review state:** `indexed`
- **Study ID:** `main_model_evaluation_v2`

## Scientific role

**Question.** How should core Liu-mechanism adequacy be distinguished from strict full
quantitative fidelity, and what do the frozen Resampled outputs imply for the next
minimal single-stage learner?

**Finding.** All three frozen Resampled fits retrospectively satisfy the added
core-behavior layer while retaining their legacy 8/9 quantitative pilot status. Internal
strict correctness is only about 3.4%, and sampled choice has no directional net effect
on correct-ranker prevalence; ranking-composition TV is about 0.118. This supports
targeting the update rule before the readout, while retaining sampling inconsistency as
a separate boundary.

**Claim boundary.** Retrospective evaluation of exposed simulations, not prospective
validation of the new 0.90 threshold, full quantitative fidelity, human latent-state
measurement or main-model promotion. No training, calibration, candidate repair or human
fitting occurred.

## Frozen records

- `registered_contract` —
  [studies/main_model_evaluation_v2/records/benchmarks/main_model_evaluation_v2.json](records/benchmarks/main_model_evaluation_v2.json)
  (`sha256:bbdedc413849`)
- `readiness_result` —
  [studies/main_model_evaluation_v2/records/benchmarks/main_model_evaluation_v2.qualification.json](records/benchmarks/main_model_evaluation_v2.qualification.json)
  (`sha256:4bf2f8420344`)
- `execution_lock` —
  [studies/main_model_evaluation_v2/records/benchmarks/main_model_evaluation_v2.execution_lock.json](records/benchmarks/main_model_evaluation_v2.execution_lock.json)
  (`sha256:ccecca1a2a6f`)
- `frozen_result` —
  [studies/main_model_evaluation_v2/records/results/main_model_evaluation_v2.json](records/results/main_model_evaluation_v2.json)
  (`sha256:c134122bb6a6`)
- `supporting_artifact` —
  [studies/main_model_evaluation_v2/records/results/main_model_evaluation_v2.npz](records/results/main_model_evaluation_v2.npz)
  (`sha256:5f9d9b10435d`)
- `report` —
  [studies/main_model_evaluation_v2/records/reports/main_model_evaluation_v2.md](records/reports/main_model_evaluation_v2.md)
  (`sha256:2507e6033c71`)
- `report` —
  [studies/main_model_evaluation_v2/records/reports/main_model_evaluation_v2.interpretation.md](records/reports/main_model_evaluation_v2.interpretation.md)
  (`sha256:5ce1b39433b7`)

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
