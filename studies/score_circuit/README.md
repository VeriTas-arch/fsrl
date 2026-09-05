# Score-only: finite-time opponent circuit realization

<!-- fsrl-doc role=generated-navigation source=studies/score_circuit/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/score_circuit/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `confirmed`
- **Review state:** `indexed`
- **Study ID:** `score_circuit`

## Scientific role

**Question.** Can local centered activity and compartment mismatch implement the frozen
score learner with bounded nonnegative efficacies at finite time scales?

**Finding.** All three exposed frozen score-only fits satisfy
conditional_circuit_sufficiency across all locked scales and step sizes. Actual bounded
nonnegative efficacies, finite-time compartment mismatch and pooled error gain preserve
the original task function; teaching-off and mismatch-clamp give no writes, and shuffled
teaching loses competence. All 798 endpoint/interval estimates independently
reconstructed. Historical qualitative 9/9 and quantitative 3/9 behavior remain
unchanged. Complete and frozen; no training, calibration or main-model promotion.

**Claim boundary.** Conditional rate-level circuit sufficiency under explicit
centered-activity, opponent-teaching, pooled-gain and external admission assumptions;
not full conductance/spike implementation, human confirmation or main-model promotion.

## Frozen records

- `registered_contract` —
  [studies/score_circuit/records/benchmarks/score_circuit_v1.json](records/benchmarks/score_circuit_v1.json)
  (`sha256:c01cf52ee9a4`)
- `readiness_result` —
  [studies/score_circuit/records/benchmarks/score_circuit_v1.qualification.json](records/benchmarks/score_circuit_v1.qualification.json)
  (`sha256:a17543070062`)
- `execution_lock` —
  [studies/score_circuit/records/benchmarks/score_circuit_v1.execution_lock.json](records/benchmarks/score_circuit_v1.execution_lock.json)
  (`sha256:79ab40516947`)
- `report` —
  [studies/score_circuit/records/reports/score_circuit_v1.interpretation.md](records/reports/score_circuit_v1.interpretation.md)
  (`sha256:be12264d5330`)
- `report` —
  [studies/score_circuit/records/reports/score_circuit_v1.md](records/reports/score_circuit_v1.md)
  (`sha256:9d4ec51e9823`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/run.json](records/results/run.json)
  (`sha256:34776df744e1`)
- `frozen_result` —
  [studies/score_circuit/records/results/score_circuit_v1.json](records/results/score_circuit_v1.json)
  (`sha256:740e4694c060`)
- `validation_result` —
  [studies/score_circuit/records/results/score_circuit_v1.validation.json](records/results/score_circuit_v1.validation.json)
  (`sha256:638540491809`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-fast-4096.npz](records/results/seed-2111-fast-4096.npz)
  (`sha256:b4aa6d1c251a`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-fast-8192.npz](records/results/seed-2111-fast-8192.npz)
  (`sha256:10cfe9058557`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-mismatch_clamp.npz](records/results/seed-2111-mismatch_clamp.npz)
  (`sha256:e5723a479c56`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-primary-4096.npz](records/results/seed-2111-primary-4096.npz)
  (`sha256:9edee232936d`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-primary-8192.npz](records/results/seed-2111-primary-8192.npz)
  (`sha256:d8f20f720751`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-reference-checks.npz](records/results/seed-2111-reference-checks.npz)
  (`sha256:b9d5405b3f9a`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-reference.npz](records/results/seed-2111-reference.npz)
  (`sha256:051b60df75bc`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-sampled-behavior.json](records/results/seed-2111-sampled-behavior.json)
  (`sha256:85b138feb9cf`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-slow-4096.npz](records/results/seed-2111-slow-4096.npz)
  (`sha256:27bd6866f718`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-slow-8192.npz](records/results/seed-2111-slow-8192.npz)
  (`sha256:403f2cedf441`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-teacher_off.npz](records/results/seed-2111-teacher_off.npz)
  (`sha256:1759d97bb8b7`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2111-teaching_shuffle.npz](records/results/seed-2111-teaching_shuffle.npz)
  (`sha256:fcde97ce1880`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-fast-4096.npz](records/results/seed-2112-fast-4096.npz)
  (`sha256:32492657365b`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-fast-8192.npz](records/results/seed-2112-fast-8192.npz)
  (`sha256:826f94243119`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-mismatch_clamp.npz](records/results/seed-2112-mismatch_clamp.npz)
  (`sha256:e5723a479c56`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-primary-4096.npz](records/results/seed-2112-primary-4096.npz)
  (`sha256:b878e3eb3c18`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-primary-8192.npz](records/results/seed-2112-primary-8192.npz)
  (`sha256:820a3701b676`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-reference-checks.npz](records/results/seed-2112-reference-checks.npz)
  (`sha256:56d1d8b29a58`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-reference.npz](records/results/seed-2112-reference.npz)
  (`sha256:0bc63f6facc6`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-sampled-behavior.json](records/results/seed-2112-sampled-behavior.json)
  (`sha256:4806dd6140ac`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-slow-4096.npz](records/results/seed-2112-slow-4096.npz)
  (`sha256:d48bbe0beef9`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-slow-8192.npz](records/results/seed-2112-slow-8192.npz)
  (`sha256:87a541bf2a69`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-teacher_off.npz](records/results/seed-2112-teacher_off.npz)
  (`sha256:1759d97bb8b7`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2112-teaching_shuffle.npz](records/results/seed-2112-teaching_shuffle.npz)
  (`sha256:5bbaff443b70`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-fast-4096.npz](records/results/seed-2113-fast-4096.npz)
  (`sha256:d4a99a0b8054`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-fast-8192.npz](records/results/seed-2113-fast-8192.npz)
  (`sha256:5f4f66f7f0ac`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-mismatch_clamp.npz](records/results/seed-2113-mismatch_clamp.npz)
  (`sha256:e5723a479c56`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-primary-4096.npz](records/results/seed-2113-primary-4096.npz)
  (`sha256:4a0067cc5c94`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-primary-8192.npz](records/results/seed-2113-primary-8192.npz)
  (`sha256:b44b7eab6ddc`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-reference-checks.npz](records/results/seed-2113-reference-checks.npz)
  (`sha256:d7156d2e2725`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-reference.npz](records/results/seed-2113-reference.npz)
  (`sha256:7de311875543`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-sampled-behavior.json](records/results/seed-2113-sampled-behavior.json)
  (`sha256:13100d4ddb0f`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-slow-4096.npz](records/results/seed-2113-slow-4096.npz)
  (`sha256:5c39ae3dc225`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-slow-8192.npz](records/results/seed-2113-slow-8192.npz)
  (`sha256:28ec1af1e53c`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-teacher_off.npz](records/results/seed-2113-teacher_off.npz)
  (`sha256:1759d97bb8b7`)
- `supporting_artifact` —
  [studies/score_circuit/records/results/seed-2113-teaching_shuffle.npz](records/results/seed-2113-teaching_shuffle.npz)
  (`sha256:95dfd7f56282`)

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
