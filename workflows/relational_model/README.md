# Relational model mainline

> This page is generated from `workflow.toml`. The TOML is the machine-readable
> contract; this page is the human reading route.

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
| 7. [Evidence synthesis and figures](#reporting) | How are frozen estimands converted into a report without rewriting diagnostic history? | The repository exposes a short human route, a machine-verifiable registry, and paper-aligned behavioral figures from the same evidence ownership model. |

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

Evidence:

- [task_fidelity](../../studies/task_fidelity/README.md)

Verification:

- `direnv exec . python -m unittest tests.tasks.test_protocol tests.tasks.test_meta_tasks`

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
- [`fsrl/infra/runtime.py`](../../fsrl/infra/runtime.py)

Evidence:

- [development_qualification](../../studies/development_qualification/README.md)
- [formal_behavioral_confirmation](../../studies/formal_behavioral_confirmation/README.md)

Verification:

- `direnv exec . python -m unittest tests.training.test_backbone tests.evaluation.test_frozen_fast_weight`

## 3. Global assembly mechanism

<a id="global-mechanism"></a>

**Question.** How does the fast-weight state transform retained evidence into a coherent global field?

**Method.** Use registered causal ablations, write-factor swaps, trajectory analyses, and expected-rank versus MAP geometry without changing the frozen evaluator.

**Result.** Evidence-specific writes accumulate through a meta-learned recurrent sensitivity landscape toward an expected-rank-like global potential.

**Boundary.** The complete DA direction-preservation chain is heterogeneous, and the model is not a sequential Bayesian updater.

Implementation:

- [`fsrl/evaluation/frozen_fast_weight.py`](../../fsrl/evaluation/frozen_fast_weight.py)
- [`fsrl/analysis/hodge.py`](../../fsrl/analysis/hodge.py)
- [`fsrl/analysis/statistics.py`](../../fsrl/analysis/statistics.py)

Evidence:

- [assembly_trajectory](../../studies/assembly_trajectory/README.md)
- [support_factor_swap](../../studies/support_factor_swap/README.md)
- [mechanism_confirmation](../../studies/mechanism_confirmation/README.md)

Verification:

- `direnv exec . python -m unittest tests.experiments.assembly.test_trajectory tests.experiments.assembly.test_factor_swap tests.experiments.confirmation.test_mechanism`

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

Evidence:

- [conjunctive_local_trace_replication](../../studies/conjunctive_local_trace_replication/README.md)
- [local_behavior_attribution](../../studies/local_behavior_attribution/README.md)

Verification:

- `direnv exec . python -m unittest tests.core.test_local_trace tests.core.test_relational_system`

## 5. Differential evidence admission

<a id="differential-admission"></a>

**Question.** Can weak observed evidence persist locally when it is not admitted to global assembly?

**Method.** Keep the P and L update/readout mechanisms fixed, use selective effective evidence for P, and use z + (1-z)p for the local write.

**Result.** Four independent backbones confirm omitted direct-fidelity rescue with evidence and query specificity while global inference remains P-dependent.

**Boundary.** The broader local writes have a small replicated retained cross-talk cost and do not repair the excessive global distance slope.

Implementation:

- [`fsrl/tasks/evidence.py`](../../fsrl/tasks/evidence.py)
- [`fsrl/core/relational_system.py`](../../fsrl/core/relational_system.py)

Evidence:

- [dual_evidence_access_pilot](../../studies/dual_evidence_access_pilot/README.md)
- [dual_evidence_access_confirmation](../../studies/dual_evidence_access_confirmation/README.md)

Verification:

- `direnv exec . python -m unittest tests.experiments.local_fidelity.test_evidence_access_pilot tests.experiments.local_fidelity.test_evidence_access_confirmation tests.core.test_relational_system`

## 6. Algorithmic form and transport

<a id="algorithm-and-transport"></a>

**Question.** Which parts of the working model compress exactly, and where does it transport?

**Method.** Reduce the local state to an exact edge ledger plus Gram geometry and test the unchanged causal links across support topology, order, sparsity, and item count.

**Result.** The local computation has an exact reduction and the structural mechanism transports broadly; tested low-dimensional global reductions remain insufficient.

**Boundary.** Evidence-density prevalence has an unresolved boundary, and no tested compact global state is yet sufficient.

Implementation:

- [`fsrl/core/relational_system.py`](../../fsrl/core/relational_system.py)
- [`fsrl/analysis/hodge.py`](../../fsrl/analysis/hodge.py)

Evidence:

- [asymmetric_algorithmic_organization](../../studies/asymmetric_algorithmic_organization/README.md)
- [support_topology_transport](../../studies/support_topology_transport/README.md)
- [presentation_order_transport](../../studies/presentation_order_transport/README.md)
- [evidence_sparsity_transport](../../studies/evidence_sparsity_transport/README.md)
- [item_count_transport](../../studies/item_count_transport/README.md)

Verification:

- `direnv exec . python -m fsrl.infra.study_registry check`

## 7. Evidence synthesis and figures

<a id="reporting"></a>

**Question.** How are frozen estimands converted into a report without rewriting diagnostic history?

**Method.** Keep study records byte-preserved, organize the cross-study claim in synthesis, and require source data plus panel provenance for promoted figures.

**Result.** The repository exposes a short human route, a machine-verifiable registry, and paper-aligned behavioral figures from the same evidence ownership model.

**Boundary.** The current synthesis remains indexed rather than final; a later scientific curation pass may reorder the argument without changing frozen records.

Implementation:

- [`fsrl/infra/study_registry.py`](../../fsrl/infra/study_registry.py)
- [`fsrl/workflows/paper_figures.py`](../../fsrl/workflows/paper_figures.py)
- [`synthesis/manifest.toml`](../../synthesis/manifest.toml)

Evidence:

- [behavior_reproduction_map](../../studies/behavior_reproduction_map/README.md)

Verification:

- `direnv exec . python -m fsrl.infra.study_registry check`
- `direnv exec . python -m fsrl.workflows check workflows/relational_model/workflow.toml`

## Evidence and figures

- [Study registry](../../studies/README.md)
- [Cross-study synthesis](../../synthesis/README.md)
- [Report and paper figures](../../synthesis/figures/README.md)

Runtime outputs remain outside this workflow. A result enters the evidence
registry only through a study-owned contract, result, report, and provenance
record; a report-facing figure additionally requires source data and a panel
manifest.
