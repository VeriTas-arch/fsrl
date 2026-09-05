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

- **Status:** `frozen_contract`
- **Review state:** `indexed`
- **Study ID:** `main_model_evaluation_v2`

## Scientific role

**Question.** How should core Liu-mechanism adequacy be distinguished from strict full
quantitative fidelity, and what do the frozen Resampled outputs imply for the next
minimal single-stage learner?

**Finding.** The claim-relative evaluation contract, qualified read-only evaluator and
all-input execution lock are frozen. It preserves the complete legacy 9/9 metric while
adding a core-mechanism label, a joint ranking-composition profile and an
internal-to-sampled localization. No new result or model promotion is registered yet.

**Claim boundary.** Evaluation and read-only diagnostic authority only. The current
Resampled result is exposed and may be classified retrospectively but not confirmed
under the new threshold. No training, calibration, candidate repair, human fitting or
main-model promotion is authorized.

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
