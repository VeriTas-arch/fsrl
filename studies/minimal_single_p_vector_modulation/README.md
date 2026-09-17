# Postsynaptic vector modulation in minimal single-P M2

<!-- fsrl-doc role=generated-navigation source=studies/minimal_single_p_vector_modulation/study.toml -->

> [!NOTE]
> **Generated navigation.**
>
> - **Authority:** `studies/minimal_single_p_vector_modulation/study.toml`
> - **Rebuild:** `direnv exec . python -m fsrl.infra.study_registry build`
> - **Edit:** do not edit this README directly.
>
> `review_state = "indexed"` means the records are organized and
> structurally checked. This page is navigation, not reviewed cross-study
> synthesis or independent scientific evidence.

[Back to the study registry](../README.md)

- **Status:** `valid_negative`
- **Review state:** `indexed`
- **Study ID:** `minimal_single_p_vector_modulation`

## Scientific role

**Question.** Holding M2 inputs, persistent state, readout, task streams, optimization,
observation process, and evaluation arrays fixed, does replacing the shared scalar write
modulator with a postsynaptic-neuron-specific instantaneous vector recover constrained
Liu morphology without damaging competence?

**Finding.** Frozen development result: all 3 networks were stable_constructive and used
strongly non-scalar writes without competence damage, but 0/9 noisy units passed all
nine rows or constrained morphology. Difficult-pair bimodality failed in 9/9 and stable
within-subject errors failed in 6/9; outcome used_without_rescue.

**Claim boundary.** P remains the only episode-persistent plastic state. This exposed
three-seed development study tests heterogeneous write allocation, not confirmation,
biological implementation, or main-model promotion.

## Frozen records

- `registered_contract` —
  [studies/minimal_single_p_vector_modulation/records/benchmarks/minimal_single_p_vector_modulation_v1.json](records/benchmarks/minimal_single_p_vector_modulation_v1.json)
  (`sha256:11c11848906d`)
- `artifact_lock` —
  [studies/minimal_single_p_vector_modulation/records/benchmarks/model_lock.json](records/benchmarks/model_lock.json)
  (`sha256:ee9474a99daf`)
- `validation_result` —
  [studies/minimal_single_p_vector_modulation/records/benchmarks/qualification.json](records/benchmarks/qualification.json)
  (`sha256:36283b49a9ed`)
- `execution_lock` —
  [studies/minimal_single_p_vector_modulation/records/benchmarks/source_input_lock.json](records/benchmarks/source_input_lock.json)
  (`sha256:ac053207f110`)
- `report` —
  [studies/minimal_single_p_vector_modulation/records/reports/minimal_single_p_vector_modulation_v1.generic.md](records/reports/minimal_single_p_vector_modulation_v1.generic.md)
  (`sha256:60bb7be5af86`)
- `report` —
  [studies/minimal_single_p_vector_modulation/records/reports/minimal_single_p_vector_modulation_v1.md](records/reports/minimal_single_p_vector_modulation_v1.md)
  (`sha256:b1412e73ce2a`)
- `frozen_result` —
  [studies/minimal_single_p_vector_modulation/records/results/minimal_single_p_vector_modulation_v1.generic.json](records/results/minimal_single_p_vector_modulation_v1.generic.json)
  (`sha256:9caca316f72c`)
- `frozen_result` —
  [studies/minimal_single_p_vector_modulation/records/results/minimal_single_p_vector_modulation_v1.json](records/results/minimal_single_p_vector_modulation_v1.json)
  (`sha256:d3c964a400e4`)
- `supporting_artifact` —
  [studies/minimal_single_p_vector_modulation/records/results/minimal_single_p_vector_modulation_v1.pairs.npz](records/results/minimal_single_p_vector_modulation_v1.pairs.npz)
  (`sha256:302f3160e62d`)
- `supporting_artifact` —
  [studies/minimal_single_p_vector_modulation/records/results/minimal_single_p_vector_modulation_v1.writes.npz](records/results/minimal_single_p_vector_modulation_v1.writes.npz)
  (`sha256:ed3e60ce7a90`)

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
