# Anytime presentation-horizon training for clean single-P

<!-- fsrl-doc role=generated-navigation source=studies/single_p_anytime/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/single_p_anytime/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `mixed`
- **Review state:** `indexed`
- **Study ID:** `single_p_anytime`

## Scientific role

**Question.** Under the inherited terminal-P regularizer, does variable
presentation-horizon training improve held-out short- and long-horizon predictive
quality while preserving the clean no-time single-P model's historical B=4 phenotype?

**Finding.** Across all three development seeds and both clean/noisy arms,
variable-horizon training preserved the historical B=4 phenotype and absolute competence
at B=1..7, and improved the registered B=1 CE endpoint without probability harm, but
failed the B=7 CE-superiority gate. The frozen V1 outcome is horizon_tradeoff and is not
admitted to confirmation.

**Claim boundary.** This rejects admission of the frozen V1 presentation-horizon recipe
under the inherited terminal-P regularizer; it does not reject variable-horizon training
in general. The study does not test accumulation of new relation information, explicit
time or evidence-count inference, online uncertainty inference, confirmation-level
reliability, or a human/biological mechanism.

## Frozen records

- `registered_contract` —
  [studies/single_p_anytime/records/benchmarks/single_p_anytime_v1.json](records/benchmarks/single_p_anytime_v1.json)
  (`sha256:3d70b51bc076`)
- `validation_result` —
  [studies/single_p_anytime/records/benchmarks/qualification.json](records/benchmarks/qualification.json)
  (`sha256:8b24bfff5e7f`)
- `execution_lock` —
  [studies/single_p_anytime/records/benchmarks/source_lock.json](records/benchmarks/source_lock.json)
  (`sha256:a4e16a009525`)
- `execution_lock` —
  [studies/single_p_anytime/records/benchmarks/model_lock.json](records/benchmarks/model_lock.json)
  (`sha256:1924fb8d95a4`)
- `frozen_result` —
  [studies/single_p_anytime/records/results/single_p_anytime_v1.json](records/results/single_p_anytime_v1.json)
  (`sha256:bcfac478425b`)
- `report` —
  [studies/single_p_anytime/records/reports/single_p_anytime_v1.md](records/reports/single_p_anytime_v1.md)
  (`sha256:4cdfe7026c8a`)

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
