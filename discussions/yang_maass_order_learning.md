# Yang and Maass (2026): local order learning and the FSRL claim boundary

- **Status:** maintained research discussion
- **Reviewed:** 2026-08-30
- **Authority:** none; follow the linked workflow and study records for project
  claims
- **External sources:** [Nature Communications article](https://doi.org/10.1038/s41467-026-76102-5)
  and [archived source code v1.0.0](https://doi.org/10.5281/zenodo.20729243)

## Question

Does the local order-learning rule of Yang and Maass provide a simpler
explanation of the relational behavior attributed to the frozen FSRL model,
and does it require a new comparator experiment?

The short answer is asymmetric:

- it is a serious sufficiency result for constructing a global rank potential
  with a simple local update;
- it does not invalidate the registered causal division between `P_T` and
  `L_T` inside the frozen FSRL architecture;
- it does limit any claim that recurrence or the FSRL two-path implementation
  is generally required for transitive inference;
- a new formal comparator on the existing Liu responses would cross a frozen
  project boundary and is not recommended.

## External result

Yang and Maass analyze a linear rate unit with an item code. To avoid a
notation clash with the FSRL ledger `a_T`, write that code here as `x_i`. Its
scalar rate is

$$
r_i = w^\top x_i,
$$

and an error-triggered local update for an observed order relation between
items `i` and `j`:

$$
w_{t+1} = w_t + \eta e_t(x_j-x_i).
$$

The theory and simulations show that such a rule can extract ranks, support
transitive inference, learn partial orders, and rapidly reorganize after new
evidence. The paper also demonstrates the computation on Loihi 2.

"Single neuron" in this result denotes a theoretical linear rate unit and its
local plasticity rule. It is not a new single-cell recording result and should
not be cited as evidence that one biological neuron implements the human Liu
phenotype.

## Algebraic relation to FSRL

Let the relation feature be

$$
\phi_r = x_j-x_i.
$$

After an episode, the Yang--Maass state has the form

$$
w_T = w_0 + \sum_r \alpha_r\phi_r,
$$

so the margin for query `q` is

$$
\phi_q^\top w_T
= \phi_q^\top w_0 + \sum_r K_{qr}\alpha_r,
\qquad K_{qr}=\phi_q^\top\phi_r.
$$

This is a ledger-plus-Gram computation in relation coordinates. That algebra
resembles the exact reduction of the FSRL local trace,

$$
L_T=\sum_r a_r k_r,
\qquad \ell_q=k_q^\top L_T=(Ka)_q,
$$

but the computational roles differ:

- the Yang--Maass update writes one rank state whose output is a global scalar
  potential over items;
- the implemented FSRL `L_T`/`a_T` path is query-addressed and preserves direct
  evidence;
- the FSRL `P_T` path performs the registered global and remote assembly.

The relevant analogy is therefore not "Yang--Maass equals `L_T`". It is that a
local update can generate a global potential, while FSRL assigns global
assembly and direct fidelity to causally distinct paths.

## Verified project evidence

The current model-level claim is defined by the
[relational-model workflow](../workflows/relational_model/README.md) and the
[frozen manuscript](../synthesis/manuscript/relational_model/main.typ). It says
that `P_T` supplies global and remote construction while `L_T`, exactly reduced
to `a_T`, supplies query-addressed direct fidelity. It does not establish a
unique compact global algorithm or a human neural implementation.

Several existing records constrain what a new Yang--Maass analysis could add:

1. The prospective
   [human metric constructive comparator](../studies/human_metric_constructive_comparator/records/docs/human_metric_constructive_comparator_v1.md)
   reproduced the held-out symbolic-distance gradient but failed the reliable
   distance-residualized pair field. Its registered decision closes comparator
   search on the same 37 replication responses.
2. The
   [presentation-order transport study](../studies/presentation_order_transport/records/docs/liu_presentation_order_transport_v1.md)
   already compares blockwise-random, relation-clustered, and reversed versions
   of the same 32 virtual support observations. `P_T` is quantitatively
   order-sensitive and the additive `a_T` ledger is exactly order-invariant.
   This is model-side evidence, not human presentation-order evidence.
3. The public Liu
   [dataset manifest](../data/external/liu2026/dataset.toml) contains query
   responses but no participant-level support trials or learning presentation
   sequences. A Yang--Maass analysis cannot use each participant's "actual
   learning sequence" from the current source data.
4. A prospectively registered
   [magnitude-placement human program](../studies/magnitude_placement_human_program/README.md)
   already addresses the decisive unresolved human-data question by changing
   magnitude placement while holding the signed graph and magnitude multiset
   fixed. Its implementation readiness passes, but collection remains deferred
   and `NO_GO` pending the registered external requirements and explicit user
   authorization.

## Interpretation

| Claim | Effect of Yang and Maass |
| --- | --- |
| A simple local rule can produce a coherent global rank potential | Directly supported by the external theory |
| Near-potential geometry uniquely identifies recurrent global assembly | Rejected as a general inference |
| `P_T` is causally required for global behavior inside the frozen FSRL model | Unchanged |
| `L_T` provides direct-fidelity rescue inside the frozen FSRL model | Unchanged |
| The complete Liu phenotype generally requires a `P_T`/`L_T` decomposition | Not established by current evidence |
| The external paper establishes a biological single-neuron mechanism for Liu behavior | Not supported |

The contribution that remains distinctive is not global ordering alone. It is
the coexistence, within one frozen model, of history-dependent global assembly
and query-specific preservation of direct evidence, together with causal
interventions that separate those roles.

## Experiment decision

### Current reporting

Do not add a new confirmatory Yang--Maass comparator on the existing Liu
responses. Add the paper to related work and state explicitly that the FSRL
causal necessity is architecture-specific rather than a proof of general
computational necessity.

The manuscript sentence

> A pure edge store retains observations but does not construct unseen
> relations.

would be safer as:

> The implemented conjunctive edge ledger preserves direct evidence but does
> not, under its fixed Gram readout, reproduce the global recurrent assembly
> observed in this model.

### Optional model-side stress test

If explicitly authorized later, a small external-algorithm stress test could
quantify source-faithful transfer without reopening human comparator search.
It should be registered with these boundaries:

- use the original Yang--Maass rule as the sole primary model and pin the
  public v1.0.0 source;
- freeze initialization, item-code geometry, normalization, learning rate,
  margin threshold, presentation budget, and output map before execution;
- use task-visible support evidence and the already frozen virtual schedule
  conditions, but do not pass the FSRL-specific participant variables
  `z_sr` or `p_sr` into the competitor;
- do not fit the 37-person replication responses, tune a response temperature,
  scan a Pareto frontier, or use slope and endpoint discrepancies as tuning
  targets;
- report global/Hodge structure, nonlearned inference, presentation-order
  sensitivity, and relation-LOO direct-versus-remote coupling;
- treat participant-level individualization and human pair fields as
  descriptive limits, not new confirmation gates.

A magnitude-aware variant is a new model rather than the published
Yang--Maass rule. It is admissible only as one prospectively fixed secondary
sensitivity analysis, with a single declared magnitude-to-update equation and
no parameter sweep.

### Decisive extension

If the intended claim is that the joint human phenotype generally requires
metric-dependent global construction plus local direct fidelity, the next
decisive evidence is new human behavior. The existing magnitude-placement
program is the registered route. It should not be altered through this
discussion document, and real collection remains unauthorized until its own
external gates are satisfied.

## Current decision

For the frozen report, cite Yang and Maass and narrow the generality claim; do
not rerun FSRL and do not reopen comparator search on the existing human
responses. Retain the model-side Yang--Maass transfer as an optional
supplementary diagnostic. If one new empirical study is prioritized, use the
already registered magnitude-placement human program.
