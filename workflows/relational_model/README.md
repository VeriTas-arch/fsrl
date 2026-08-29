# Relational model mainline

> This page is generated from `workflow.toml`. The TOML is the machine-readable
> claim, evidence, implementation, verification, and figure contract.

A reproducible route from the task information boundary through the maintained global/local computation to registered evidence and report-facing figures.

**Current working claim.** Sparse signed evidence feeds a meta-learned fast-weight state for global assembly and a query-addressed local trace with broader evidence admission for direct fidelity.

**Claim boundary.** This workflow organizes the current model and its evidence; it does not turn the model into a claim about human neural implementation, erase valid negative studies, or supersede frozen study contracts.

## How to read this mainline

| Stage | Scientific question | Current result |
| --- | --- | --- |
| 1. [Task and evidence contract](#task-contract) | What information enters the model, and what is deliberately withheld? | The maintained task adapter preserves the experimental information boundary while allowing explicit subject-level evidence admission. |
| 2. [Global fast-weight backbone](#global-backbone) | Can a shared recurrent learner assemble unseen sparse relational graphs? | Held-out transfer and fast-weight necessity establish a competent global relational assembly substrate. |
| 3. [Global assembly mechanism](#global-mechanism) | How does the fast-weight state transform retained evidence into a coherent global field? | Evidence-specific writes accumulate through a meta-learned recurrent sensitivity landscape toward an expected-rank-like global potential. |
| 4. [Query-addressed local fidelity](#local-mechanism) | What preserves direct experience when nonlinear global expression loses local correctness? | Independent backbones replicate a direct-enriched local rescue and a causal P/L double dissociation. |
| 5. [Differential evidence admission](#differential-admission) | Can weak observed evidence persist locally when it is not admitted to global assembly? | Four independent backbones confirm omitted direct-fidelity rescue with evidence and query specificity while global inference remains P-dependent. |
| 6. [Algorithmic form and transport](#algorithm-and-transport) | Which parts of the working model compress exactly, and where does it transport? | The local computation has an exact reduction and the structural mechanism transports broadly; tested low-dimensional global reductions remain insufficient. |
| 7. [Evidence synthesis and figures](#reporting) | How are frozen estimands converted into a report without rewriting diagnostic history? | The repository exposes one current claim graph, a complete evidence ledger, historical snapshots, and paper-aligned behavioral figures. |

## 1. Task and evidence contract

<a id="task-contract"></a>

**Question.** What information enters the model, and what is deliberately withheld?

**Method.** Represent each support event by item identity plus signed displayed magnitude, with passive presentation and no query label or test feedback in the input.

**Result.** The maintained task adapter preserves the experimental information boundary while allowing explicit subject-level evidence admission.

**Boundary.** Any omission, attenuation, or broader local admission is a model hypothesis rather than a restatement of the task.

Implementation:

- [`fsrl/tasks/protocol.py`](../../fsrl/tasks/protocol.py)
- [`fsrl/tasks/sparse_ranking.py`](../../fsrl/tasks/sparse_ranking.py)
- [`fsrl/tasks/evidence.py`](../../fsrl/tasks/evidence.py)
- [`fsrl/core/inputs.py`](../../fsrl/core/inputs.py)

Tests:

- [`tests/tasks/test_protocol.py`](../../tests/tasks/test_protocol.py)
- [`tests/tasks/test_meta_tasks.py`](../../tests/tasks/test_meta_tasks.py)

Exact evidence:

- `defines` — [task_fidelity:records/benchmarks/liu_v2.json](../../studies/task_fidelity/records/benchmarks/liu_v2.json) — Frozen task interface and information boundary used by the maintained model evaluation.

Verification:

- `task_contract_tests` (`cpu`): `direnv exec . python -m unittest tests.tasks.test_protocol tests.tasks.test_meta_tasks`

## 2. Global fast-weight backbone

<a id="global-backbone"></a>

**Question.** Can a shared recurrent learner assemble unseen sparse relational graphs?

**Method.** Meta-train the plastic recurrent backbone on generic connected ranking graphs while prospectively holding out the evaluation graph family.

**Result.** Held-out transfer and fast-weight necessity establish a competent global relational assembly substrate.

**Boundary.** Competence does not by itself identify the algorithm or reproduce every human behavioral statistic.

Implementation:

- [`fsrl/core/plastic_rnn.py`](../../fsrl/core/plastic_rnn.py)
- [`fsrl/training/backbone.py`](../../fsrl/training/backbone.py)
- [`fsrl/training/checkpoints.py`](../../fsrl/training/checkpoints.py)
- [`fsrl/evaluation/contracts.py`](../../fsrl/evaluation/contracts.py)
- [`fsrl/evaluation/execution.py`](../../fsrl/evaluation/execution.py)
- [`fsrl/evaluation/frozen_fast_weight.py`](../../fsrl/evaluation/frozen_fast_weight.py)
- [`fsrl/evaluation/sampling.py`](../../fsrl/evaluation/sampling.py)
- [`fsrl/evaluation/subject_encoding.py`](../../fsrl/evaluation/subject_encoding.py)
- [`fsrl/infra/run_manifest.py`](../../fsrl/infra/run_manifest.py)
- [`fsrl/infra/runtime.py`](../../fsrl/infra/runtime.py)

Tests:

- [`tests/training/test_backbone.py`](../../tests/training/test_backbone.py)
- [`tests/evaluation/test_frozen_fast_weight.py`](../../tests/evaluation/test_frozen_fast_weight.py)

Exact evidence:

- `supports` — [development_qualification:records/results/dev_v2_seed1801_step1000.json](../../studies/development_qualification/records/results/dev_v2_seed1801_step1000.json) `/causal_qualification/status` — The development checkpoint passes the held-out causal qualification gate.
- `supports` — [formal_behavioral_confirmation:records/results/confirmation_v1.json](../../studies/formal_behavioral_confirmation/records/results/confirmation_v1.json) `/pass_rates/causal_qualification_pass_proportion` — All ten declared formal networks pass causal qualification without seed filtering.
- `constrains` — [formal_behavioral_confirmation:records/results/confirmation_v1.json](../../studies/formal_behavioral_confirmation/records/results/confirmation_v1.json) `/pass_rates/behavioral_confirmation_pass_proportion` — The full behavioral conjunction fails in every formal network and remains a model boundary.

Verification:

- `global_backbone_tests` (`cpu`): `direnv exec . python -m unittest tests.training.test_backbone tests.evaluation.test_frozen_fast_weight`

## 3. Global assembly mechanism

<a id="global-mechanism"></a>

**Question.** How does the fast-weight state transform retained evidence into a coherent global field?

**Method.** Use registered causal ablations, write-factor swaps, trajectory analyses, and expected-rank versus MAP geometry without changing the frozen evaluator.

**Result.** Evidence-specific writes accumulate through a meta-learned recurrent sensitivity landscape toward an expected-rank-like global potential.

**Boundary.** The complete DA direction-preservation chain is heterogeneous, and the model is not a sequential Bayesian updater.

Implementation:

- [`fsrl/evaluation/frozen_fast_weight.py`](../../fsrl/evaluation/frozen_fast_weight.py)
- [`fsrl/evaluation/registered.py`](../../fsrl/evaluation/registered.py)
- [`fsrl/analysis/hodge.py`](../../fsrl/analysis/hodge.py)
- [`fsrl/analysis/statistics.py`](../../fsrl/analysis/statistics.py)
- [`fsrl/experiments/assembly/trajectory.py`](../../fsrl/experiments/assembly/trajectory.py)
- [`fsrl/experiments/assembly/factor_swap.py`](../../fsrl/experiments/assembly/factor_swap.py)
- [`fsrl/experiments/confirmation/mechanism.py`](../../fsrl/experiments/confirmation/mechanism.py)

Tests:

- [`tests/experiments/assembly/test_trajectory.py`](../../tests/experiments/assembly/test_trajectory.py)
- [`tests/experiments/assembly/test_factor_swap.py`](../../tests/experiments/assembly/test_factor_swap.py)
- [`tests/experiments/confirmation/test_mechanism.py`](../../tests/experiments/confirmation/test_mechanism.py)

Exact evidence:

- `supports` — [assembly_trajectory:records/results/assembly_trajectory_v1.json](../../studies/assembly_trajectory/records/results/assembly_trajectory_v1.json) `/overall_diagnosis/episode_specific_global_reassembly_replicated_across_pilot_seeds` — Episode-specific global reassembly replicates across both diagnostic pilots.
- `supports` — [support_factor_swap:records/results/support_factor_swap_v1.json](../../studies/support_factor_swap/records/results/support_factor_swap_v1.json) `/overall_diagnosis/eligibility_identity_transfer_replicated_across_pilot_seeds` — Matched eligibility transfers evidence-specific direction across both pilots.
- `supports` — [mechanism_confirmation:records/results/mechanism_confirmation_v1.json](../../studies/mechanism_confirmation/records/results/mechanism_confirmation_v1.json) `/linkwise_confirmation/immediate_and_episode_global_reassembly/status` — Immediate and episode-level global reassembly confirms across all formal seeds.
- `constrains` — [mechanism_confirmation:records/results/mechanism_confirmation_v1.json](../../studies/mechanism_confirmation/records/results/mechanism_confirmation_v1.json) `/linkwise_confirmation/da_gain_with_direction_preservation/status` — DA magnitude confirms but direction preservation remains unresolved because of seed heterogeneity.

Verification:

- `global_mechanism_tests` (`cpu`): `direnv exec . python -m unittest tests.experiments.assembly.test_trajectory tests.experiments.assembly.test_factor_swap tests.experiments.confirmation.test_mechanism`

## 4. Query-addressed local fidelity

<a id="local-mechanism"></a>

**Question.** What preserves direct experience when nonlinear global expression loses local correctness?

**Method.** Accumulate a fixed antisymmetric conjunctive trace and read it only through the matching query address, with explicit P-off and L-off interventions.

**Result.** Independent backbones replicate a direct-enriched local rescue and a causal P/L double dissociation.

**Boundary.** The evidence supports the computation, not a unique tensor-product code or biologically separate memory store.

Implementation:

- [`fsrl/core/local_trace.py`](../../fsrl/core/local_trace.py)
- [`fsrl/core/relational_system.py`](../../fsrl/core/relational_system.py)
- [`fsrl/core/state.py`](../../fsrl/core/state.py)
- [`fsrl/evaluation/relational_query.py`](../../fsrl/evaluation/relational_query.py)
- [`fsrl/experiments/local_fidelity/trace_replication.py`](../../fsrl/experiments/local_fidelity/trace_replication.py)
- [`fsrl/experiments/local_fidelity/behavior_attribution.py`](../../fsrl/experiments/local_fidelity/behavior_attribution.py)

Tests:

- [`tests/core/test_local_trace.py`](../../tests/core/test_local_trace.py)
- [`tests/core/test_relational_system.py`](../../tests/core/test_relational_system.py)
- [`tests/experiments/local_fidelity/test_trace_replication.py`](../../tests/experiments/local_fidelity/test_trace_replication.py)
- [`tests/experiments/local_fidelity/test_behavior_attribution.py`](../../tests/experiments/local_fidelity/test_behavior_attribution.py)

Exact evidence:

- `supports` — [conjunctive_local_trace_replication:records/results/conjunctive_local_trace_replication_v2_3.json](../../studies/conjunctive_local_trace_replication/records/results/conjunctive_local_trace_replication_v2_3.json) `/decision/outcome` — All four local/global causal links replicate independently on seeds 2102 and 2103.
- `supports` — [local_behavior_attribution:records/results/local_behavior_attribution_v2_3.json](../../studies/local_behavior_attribution/records/results/local_behavior_attribution_v2_3.json) `/decision/outcome` — Exact-probability attribution resolves the sampled-endpoint sensitivity without changing the mechanism.

Verification:

- `local_core_tests` (`cpu`): `direnv exec . python -m unittest tests.core.test_local_trace tests.core.test_relational_system`
- `local_evidence_tests` (`cpu`): `direnv exec . python -m unittest tests.experiments.local_fidelity.test_trace_replication tests.experiments.local_fidelity.test_behavior_attribution`

## 5. Differential evidence admission

<a id="differential-admission"></a>

**Question.** Can weak observed evidence persist locally when it is not admitted to global assembly?

**Method.** Keep the P and L update/readout mechanisms fixed, use selective effective evidence for P, and use z + (1-z)p for the local write.

**Result.** Four independent backbones confirm omitted direct-fidelity rescue with evidence and query specificity while global inference remains P-dependent.

**Boundary.** The broader local writes have a small replicated retained cross-talk cost and do not repair the excessive global distance slope.

Implementation:

- [`fsrl/tasks/evidence.py`](../../fsrl/tasks/evidence.py)
- [`fsrl/core/relational_system.py`](../../fsrl/core/relational_system.py)
- [`fsrl/experiments/local_fidelity/evidence_access_pilot.py`](../../fsrl/experiments/local_fidelity/evidence_access_pilot.py)
- [`fsrl/experiments/local_fidelity/evidence_access_confirmation.py`](../../fsrl/experiments/local_fidelity/evidence_access_confirmation.py)

Tests:

- [`tests/core/test_relational_system.py`](../../tests/core/test_relational_system.py)
- [`tests/experiments/local_fidelity/test_evidence_access_pilot.py`](../../tests/experiments/local_fidelity/test_evidence_access_pilot.py)
- [`tests/experiments/local_fidelity/test_evidence_access_confirmation.py`](../../tests/experiments/local_fidelity/test_evidence_access_confirmation.py)

Exact evidence:

- `supports` — [dual_evidence_access_pilot:records/results/dual_evidence_access_pilot_v2_4.json](../../studies/dual_evidence_access_pilot/records/results/dual_evidence_access_pilot_v2_4.json) `/decision/outcome` — The zero-parameter differential-admission rule passes all four development links.
- `supports` — [dual_evidence_access_confirmation:records/results/dual_evidence_access_confirmation_v2_4.json](../../studies/dual_evidence_access_confirmation/records/results/dual_evidence_access_confirmation_v2_4.json) `/decision/outcome` — The unchanged rule confirms on two fresh backbones after joint artifact locking.
- `constrains` — [dual_evidence_access_confirmation:records/results/dual_evidence_access_confirmation_v2_4.json](../../studies/dual_evidence_access_confirmation/records/results/dual_evidence_access_confirmation_v2_4.json) `/claim_boundary` — The result preserves the retained cross-talk cost, slope mismatch, and non-population claim boundary.

Verification:

- `differential_admission_tests` (`cpu`): `direnv exec . python -m unittest tests.experiments.local_fidelity.test_evidence_access_pilot tests.experiments.local_fidelity.test_evidence_access_confirmation tests.core.test_relational_system`

## 6. Algorithmic form and transport

<a id="algorithm-and-transport"></a>

**Question.** Which parts of the working model compress exactly, and where does it transport?

**Method.** Reduce the local state to an exact edge ledger plus Gram geometry and test the unchanged causal links across support topology, order, sparsity, and item count.

**Result.** The local computation has an exact reduction and the structural mechanism transports broadly; tested low-dimensional global reductions remain insufficient.

**Boundary.** Evidence-density prevalence has an unresolved boundary, and no tested compact global state is yet sufficient.

Implementation:

- [`fsrl/core/relational_system.py`](../../fsrl/core/relational_system.py)
- [`fsrl/analysis/hodge.py`](../../fsrl/analysis/hodge.py)
- [`fsrl/experiments/reduction/dual_state_v1.py`](../../fsrl/experiments/reduction/dual_state_v1.py)
- [`fsrl/experiments/reduction/dual_state_v2.py`](../../fsrl/experiments/reduction/dual_state_v2.py)
- [`fsrl/experiments/reduction/dual_state_v3.py`](../../fsrl/experiments/reduction/dual_state_v3.py)
- [`fsrl/experiments/reduction/functional_latent.py`](../../fsrl/experiments/reduction/functional_latent.py)
- [`fsrl/experiments/transport/topology.py`](../../fsrl/experiments/transport/topology.py)
- [`fsrl/experiments/transport/presentation_order.py`](../../fsrl/experiments/transport/presentation_order.py)
- [`fsrl/experiments/transport/evidence_sparsity.py`](../../fsrl/experiments/transport/evidence_sparsity.py)
- [`fsrl/experiments/transport/sparsity_individualization.py`](../../fsrl/experiments/transport/sparsity_individualization.py)
- [`fsrl/experiments/transport/item_count.py`](../../fsrl/experiments/transport/item_count.py)

Tests:

- [`tests/analysis/test_algorithmic.py`](../../tests/analysis/test_algorithmic.py)
- [`tests/experiments/reduction/test_dual_state_v1.py`](../../tests/experiments/reduction/test_dual_state_v1.py)
- [`tests/experiments/reduction/test_dual_state_v2.py`](../../tests/experiments/reduction/test_dual_state_v2.py)
- [`tests/experiments/reduction/test_dual_state_v3.py`](../../tests/experiments/reduction/test_dual_state_v3.py)
- [`tests/experiments/reduction/test_functional_latent.py`](../../tests/experiments/reduction/test_functional_latent.py)
- [`tests/experiments/transport/test_topology.py`](../../tests/experiments/transport/test_topology.py)
- [`tests/experiments/transport/test_presentation_order.py`](../../tests/experiments/transport/test_presentation_order.py)
- [`tests/experiments/transport/test_evidence_sparsity.py`](../../tests/experiments/transport/test_evidence_sparsity.py)
- [`tests/experiments/transport/test_sparsity_individualization.py`](../../tests/experiments/transport/test_sparsity_individualization.py)
- [`tests/experiments/transport/test_item_count.py`](../../tests/experiments/transport/test_item_count.py)

Exact evidence:

- `closes` — [dual_state_reduction_v1:records/results/dual_state_reduced_algorithm_v1.json](../../studies/dual_state_reduction_v1/records/results/dual_state_reduced_algorithm_v1.json) `/decision/outcome` — The potential-transition reduction is insufficient.
- `closes` — [dual_state_reduction_v2:records/results/dual_state_reduced_algorithm_v2.json](../../studies/dual_state_reduction_v2/records/results/dual_state_reduced_algorithm_v2.json) `/decision/outcome` — The scalar-history reduction is insufficient.
- `closes` — [dual_state_reduction_v3:records/results/dual_state_reduced_algorithm_v3.json](../../studies/dual_state_reduction_v3/records/results/dual_state_reduced_algorithm_v3.json) `/decision/outcome` — The item-history reduction is insufficient.
- `closes` — [functional_fast_weight_latent_sufficiency:records/results/functional_fast_weight_latent_sufficiency_v1.json](../../studies/functional_fast_weight_latent_sufficiency/records/results/functional_fast_weight_latent_sufficiency_v1.json) `/decision/outcome` — The registered fixed linear functional-P latent fails held-out sufficiency.
- `supports` — [asymmetric_algorithmic_organization:records/benchmarks/asymmetric_algorithmic_organization_v1.json](../../studies/asymmetric_algorithmic_organization/records/benchmarks/asymmetric_algorithmic_organization_v1.json) `/three_level_theory/algorithmic_asymmetry` — The locked cross-study synthesis states the exact local reduction and unresolved global closure boundary.
- `supports` — [support_topology_transport:records/results/liu_support_topology_transport_v1.json](../../studies/support_topology_transport/records/results/liu_support_topology_transport_v1.json) `/decision/outcome` — The causal mechanism transports across all registered support topologies.
- `supports` — [presentation_order_transport:records/results/liu_presentation_order_transport_v1.json](../../studies/presentation_order_transport/records/results/liu_presentation_order_transport_v1.json) `/decision/outcome` — The mechanism transports across all registered presentation schedules.
- `constrains` — [evidence_sparsity_transport:records/results/liu_evidence_sparsity_transport_v1.json](../../studies/evidence_sparsity_transport/records/results/liu_evidence_sparsity_transport_v1.json) `/decision/outcome` — The E=10 stable-error prevalence gate and density-allocation prediction remain unresolved or negative.
- `constrains` — [sparsity_individualization_localization:records/results/liu_sparsity_individualization_localization_v1.json](../../studies/sparsity_individualization_localization/records/results/liu_sparsity_individualization_localization_v1.json) `/decision/outcome` — Order convergence transports, but the registered stable-error loss does not replicate.
- `supports` — [item_count_transport:records/results/liu_item_count_transport_v1.json](../../studies/item_count_transport/records/results/liu_item_count_transport_v1.json) `/decision/outcome` — All registered links pass at N=6, N=8, and N=10 on all three development backbones.

Verification:

- `algorithmic_reduction_tests` (`cpu`): `direnv exec . python -m unittest tests.analysis.test_algorithmic tests.experiments.reduction.test_dual_state_v1 tests.experiments.reduction.test_dual_state_v2 tests.experiments.reduction.test_dual_state_v3 tests.experiments.reduction.test_functional_latent`
- `transport_tests` (`cpu`): `direnv exec . python -m unittest tests.experiments.transport.test_topology tests.experiments.transport.test_presentation_order tests.experiments.transport.test_evidence_sparsity tests.experiments.transport.test_sparsity_individualization tests.experiments.transport.test_item_count`

## 7. Evidence synthesis and figures

<a id="reporting"></a>

**Question.** How are frozen estimands converted into a report without rewriting diagnostic history?

**Method.** Keep study records byte-preserved, organize the cross-study claim in the mainline workflow, and require source data plus panel provenance for promoted figures.

**Result.** The repository exposes one current claim graph, a complete evidence ledger, historical snapshots, and paper-aligned behavioral figures.

**Boundary.** The current synthesis remains indexed rather than final; a later scientific curation pass may reorder the argument without changing frozen records.

Implementation:

- [`fsrl/infra/study_registry.py`](../../fsrl/infra/study_registry.py)
- [`fsrl/workflows/schema.py`](../../fsrl/workflows/schema.py)
- [`fsrl/workflows/frozen_evidence.py`](../../fsrl/workflows/frozen_evidence.py)
- [`fsrl/workflows/paper_figures.py`](../../fsrl/workflows/paper_figures.py)

Tests:

- [`tests/infra/test_study_registry.py`](../../tests/infra/test_study_registry.py)
- [`tests/workflows/test_schema.py`](../../tests/workflows/test_schema.py)
- [`tests/workflows/test_frozen_evidence.py`](../../tests/workflows/test_frozen_evidence.py)
- [`tests/workflows/test_paper_figures.py`](../../tests/workflows/test_paper_figures.py)

Exact evidence:

- `constrains` — [behavior_reproduction_map:records/results/model_behavior_reproduction_map_v1.json](../../studies/behavior_reproduction_map/records/results/model_behavior_reproduction_map_v1.json) `/summary/status_counts` — Six phenomena reproduce and three remain quantitative mismatches; none is absent.

Figures:

- [figure_01_group_behavior](../../synthesis/figures/paper_alignment/figure_01_group_behavior/figure_01_group_behavior.svg) — Matched group-level human and model behavioral estimands. ([specification](../../synthesis/figures/paper_alignment/figure_spec.json))
- [figure_02_pair_structure](../../synthesis/figures/paper_alignment/figure_02_pair_structure/figure_02_pair_structure.svg) — Pair-level accuracy and stable-error structure. ([specification](../../synthesis/figures/paper_alignment/figure_spec.json))
- [figure_02h_error_fingerprints](../../synthesis/figures/paper_alignment/figure_02h_error_fingerprints/figure_02h_error_fingerprints.svg) — Subject-by-pair stable-error fingerprints. ([specification](../../synthesis/figures/paper_alignment/figure_spec.json))
- [figure_03_global_rankings](../../synthesis/figures/paper_alignment/figure_03_global_rankings/figure_03_global_rankings.svg) — Reconstructed individualized global rankings. ([specification](../../synthesis/figures/paper_alignment/figure_spec.json))

Verification:

- `registry_check` (`cpu`): `direnv exec . python -m fsrl.infra.study_registry check`
- `workflow_check` (`cpu`): `direnv exec . python -m fsrl.workflows check workflows/relational_model/workflow.toml`
- `figure_check` (`cpu`): `direnv exec . python -m fsrl.workflows.paper_figures check`

## Evidence and figures

- [Study registry](../../studies/README.md)
- [Cross-study synthesis](../../synthesis/README.md)
- [Report and paper figures](../../synthesis/figures/README.md)

Runtime outputs remain outside this workflow. A result enters the evidence
registry only through a study-owned contract, result, report, and provenance
record; a report-facing figure additionally requires source data and a panel
manifest. Historical replay remains a separate snapshot-level operation.
