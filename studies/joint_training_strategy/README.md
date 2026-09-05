# Matched single-stage versus staged training

<!-- fsrl-doc role=generated-navigation source=studies/joint_training_strategy/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/joint_training_strategy/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `mixed`
- **Review state:** `indexed`
- **Study ID:** `joint_training_strategy`

## Scientific role

**Question.** Can a shared query objective preserve global assembly, direct fidelity,
and Liu behavior without staged optimization?

**Finding.** All six matched staged/joint models are competent; all three joint networks
preserve nine qualitative behavior rows and improve nonlearned correct probability.
Registered outcome: competent_but_not_noninferior, because seed 2108 omitted NI has
lower bound -0.0270 versus -0.02. Both schedules fail the retained local-benefit
materiality gate in every seed, despite positive effects; the other four causal links
and omitted rescue pass. Joint preserves all six named quantitative behavior rows in
2108/2110, not 2109, and is slower at equal episode exposure. The complete comparison is
frozen with no automatic tuning or seed expansion.

**Claim boundary.** This new candidate tests a fixed training recipe under imposed P/L
structural priors. It does not replace frozen v2.4 evidence, establish minimal
architecture or human neural learning, or identify an order-only effect at equal FLOPs.

## Frozen records

- `registered_contract` —
  [studies/joint_training_strategy/records/benchmarks/joint_training_strategy_v1.json](records/benchmarks/joint_training_strategy_v1.json)
  (`sha256:af6fe7ccd078`)
- `execution_lock` —
  [studies/joint_training_strategy/records/benchmarks/joint_training_strategy_v1.execution_lock.json](records/benchmarks/joint_training_strategy_v1.execution_lock.json)
  (`sha256:c2baf711692c`)
- `artifact_lock` —
  [studies/joint_training_strategy/records/benchmarks/joint_training_strategy_v1.artifact_lock.json](records/benchmarks/joint_training_strategy_v1.artifact_lock.json)
  (`sha256:b423816701e7`)
- `report` —
  [studies/joint_training_strategy/records/reports/joint_training_strategy_v1.interpretation.md](records/reports/joint_training_strategy_v1.interpretation.md)
  (`sha256:436788e7d32c`)
- `report` —
  [studies/joint_training_strategy/records/reports/joint_training_strategy_v1.md](records/reports/joint_training_strategy_v1.md)
  (`sha256:cfa9d4b8713d`)
- `frozen_result` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.json](records/results/joint_training_strategy_v1.json)
  (`sha256:08a6509f246a`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2108.joint.npz](records/results/joint_training_strategy_v1.seed-2108.joint.npz)
  (`sha256:b44e8819eced`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2108.matched_staged.npz](records/results/joint_training_strategy_v1.seed-2108.matched_staged.npz)
  (`sha256:09fc65753edd`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2109.joint.npz](records/results/joint_training_strategy_v1.seed-2109.joint.npz)
  (`sha256:de4842a5a6da`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2109.matched_staged.npz](records/results/joint_training_strategy_v1.seed-2109.matched_staged.npz)
  (`sha256:0cfafcbf1c43`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2110.joint.npz](records/results/joint_training_strategy_v1.seed-2110.joint.npz)
  (`sha256:c9f0f4c12357`)
- `supporting_artifact` —
  [studies/joint_training_strategy/records/results/joint_training_strategy_v1.seed-2110.matched_staged.npz](records/results/joint_training_strategy_v1.seed-2110.matched_staged.npz)
  (`sha256:7691cf7cfa2c`)

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
