# Main computational mechanism synthesis v1

## Scope

This document compresses already confirmed results into the current model-level
mechanism. It introduces no new estimand, fit, intervention, checkpoint replay,
or human-mechanism claim. Each arrow below inherits the seed scope, controls,
and claim boundary of its source experiment.

The model question is:

> How does a meta-learned plastic network transform sparse relational evidence
> into a stable global structure while preserving direct experience?

## Main mechanism figure

```mermaid
flowchart LR
    O[Observed signed support evidence<br/>m_t, reliability p_sr]
    A{Differential admission}
    G[s_G = m_t z_sr<br/>selective global evidence]
    W[State-dependent recurrent write<br/>E_t, d_t, alpha ⊙ delta P_t]
    P[P_T<br/>interacting fast-weight state]
    H[Near-additive Hodge field<br/>g_ij ≈ s_i - s_j]
    L[s_L = m_t z_sr + m_t(1-z_sr)p_sr<br/>broader weak local evidence]
    T[L_T = Σ s_L k_t<br/>persistent addressed trace]
    R[ell_ij = L_T · k_ij<br/>query-matched local read]
    M[Total pair margin<br/>m_ij = m^P_ij + lambda_L ell_ij]
    C[Fixed sigmoid choice policy]

    O --> A
    A --> G --> W --> P --> H --> M
    A --> L --> T --> R --> M
    M --> C
```

The upper branch is a global relational computation. Retained writes interact
with the accumulated fast-weight state, progressively form an almost additive
pair field, and terminate closer to a distributional expected-rank/Hodge
potential than to a hard MAP order. The lower branch is a direct-experience
computation. It stores a signed value on a fixed conjunctive address and
contributes only when the query address matches.

The core division of labor is therefore

\[
\text{selective global admission}
\rightarrow P_T
\rightarrow \text{relational abstraction and remote assembly},
\]

\[
\text{broader weak local admission}
\rightarrow L_T
\rightarrow \text{experience-specific direct fidelity}.
\]

## Global chain: evidence to coherent relational assembly

At support step (k), the frozen recurrent mechanism is

\[
h_k=f\!\left(x_k,h_{k-1};W+\alpha\odot P_k\right),
\qquad d_k=w_{DA}^{\mathsf T}h_k,
\]

\[
P_{k+1}=\operatorname{clip}(P_k+d_kE_k).
\]

The supported computational reading is not “one observation performs one
correct Bayesian update.” It is a state-dependent iterative relaxation:

\[
e_t
\rightarrow (E_t,d_t)
\rightarrow \Delta P_t
\rightarrow P_T
\rightarrow g_{ij}\simeq s_i-s_j
\rightarrow \pi(i,j).
\]

| Link | Causal or structural evidence | Confirmed scope | Exact boundary |
|---|---|---|---|
| Episode content requires fast weights | Reset, write-off, and `alpha=0` remove episode-specific structure | All formal seeds 2001--2010 pass necessity | Fast weights are functionally necessary; no individual synapse is identified |
| One support write has global reach | Matched zero-magnitude and relation-LOO interventions change disjoint pairs and third-party potentials | Ten-seed formal remote links confirmed | Remote magnitude is robust; isolated remote correctness is negative or unresolved |
| Eligibility carries relation-specific direction | Matched factor swaps transfer donor potential direction | Strong in pilots and population-typical in 9/10 formal seeds | Not a universal implementation invariant because seed 2009 differs |
| Modeled DA modulates magnitude | High/low swaps change effective-write and policy norm | Both magnitude links confirmed across ten formal seeds | DA direction preservation fails the formal universal threshold; modeled DA is not biological dopamine or exact surprise |
| `alpha` is a sensitivity map | Natural placement beats all norm-matched permutation nulls | Confirmed across ten formal seeds | High functional gain, not unique posterior-direction mapping |
| Accumulated state changes expression | History baseline effect and matched interaction are positive | Confirmed across ten formal seeds | Not reducible to eligibility or a scalar learning rate alone |
| Global form develops during support | Hodge gradient fraction is near one by early prefixes and commitment strengthens through support | Frozen trajectory pilots 1901/1902; terminal link confirmed formally | Additive query form is not hidden-state dimensionality and is not constructed de novo at query time |
| Terminal target is distributional | Expected-rank/Hodge cosine exceeds MAP cosine | Confirmed across ten formal seeds | Posterior-like is not exact sequential Bayesian inference |
| `P_T` is necessary for global inference | `P`-off collapses remote influence and leaves nonlearned choice near chance | v2.3/v2.4 networks 2102--2105 | Does not imply that every `P_T` component is global or uniquely interpretable |

The invariant global description is:

> `P_T` implements state-dependent iterative relational reassembly.

Parameter labels such as eligibility and DA describe one common
factorization, not the mechanism's universal identity.

## Local chain: direct observation to query-addressed fidelity

For the confirmed v2.4 branch,

\[
s_t^G=m_tz_{sr},
\qquad
s_t^L=m_t\left[z_{sr}+(1-z_{sr})p_{sr}\right].
\]

The fixed normalized antisymmetric conjunctive address is

\[
k(c_l,c_r)=
\frac{\operatorname{vec}(c_lc_r^{\mathsf T}-c_rc_l^{\mathsf T})}
{\max\!\left(\left\|\operatorname{vec}(c_lc_r^{\mathsf T}-c_rc_l^{\mathsf T})\right\|_2,10^{-8}\right)},
\]

with

\[
L_{t+1}=L_t+s_t^L k_t,
\qquad
\ell_e=L_T\cdot k_e.
\]

| Link | Causal evidence | Confirmed scope | Exact boundary |
|---|---|---|---|
| Persistent local state improves direct fidelity | Exact probability and relation-LOO direct correctness increase | v2.3 replicated on 2102/2103; v2.4 confirmed on 2104/2105 | It is not an unconstrained aggregate gain |
| Readout is query-addressed | Natural query key beats a canonical derangement | Independently in 2102--2105 | Supports the fixed address function, not unique tensor-product coding |
| Evidence routing is relation-specific | Natural weak evidence beats a scalar-multiset derangement | Independently in v2.4 seeds 2102--2105 | The existing `p_sr` is sufficient, not uniquely human-correct |
| Local-only computation is direct, not transitive | `P`-off retains learned/omitted direct benefit while remote reassembly collapses and nonlearned choice stays near chance | Independently in 2102--2105 | `L_T` is not a second global learner |
| Global-only identity is preserved | `L`-off restores v1 logits exactly and every global gate remains intact | Independently in 2102--2105 | Functional separation does not establish biological storage separation |

The double dissociation identifies distinct computations within one model:

```text
P off, L on  -> direct experience survives; remote/global inference collapses
P on,  L off -> v1 global inference survives; local fidelity benefit disappears
```

## What the combined mechanism explains

The frozen reproduction map shows that the dual-state mechanism coexists with
six independently reproduced Liu phenomena:

- learned and nonlearned accuracy in the human range;
- difficult-pair bimodality;
- stable within-subject errors;
- Hodge-reconstructed coherent but individualized rankings; and
- human-range inter-subject ranking diversity.

This closes the model-level explanatory chain

\[
\text{sparse observed evidence}
\rightarrow
\begin{cases}
P_T & \text{global coherence and generalization},\\
L_T & \text{direct-experience fidelity},
\end{cases}
\rightarrow
\text{individualized pairwise behavior}.
\]

It also explains why a pure global channel was insufficient: an almost
additive potential can assemble coherent novel relations while suppressing or
redistributing direct relation-specific value. A separately persistent local
trace restores that value without becoming transitive.

## Negative constraints carried forward

The synthesis must preserve all of the following.

1. An isolated retained write has remote causal reach but is not an
   independently correct transitive message.
2. The eligibility-direction/DA-gain factorization is population-typical, not
   universal; seed 2009 is a competent counterexample.
3. Unsigned and signed scalar amplitude gates failed, and low-capacity
   first-order residual correction was insufficient. Those families remain
   closed.
4. v2.4 broader local admission has a small replicated cross-talk cost on
   retained probabilities. Retained behavior is not literally unchanged.
5. Symbolic-distance slope is too steep, serial-position endpoint advantage is
   too weak, and one fresh network overproduces self-inconsistent subjects.
   The opposite distance/serial directions rule out a single global gain or
   temperature repair.
6. Neither `P_T`, `L_T`, the conjunctive address, nor `p_sr` is established as
   the unique human or biological implementation.

## Bridge to the reduced algorithm

The next deliverable should remove implementation-specific tensors while
preserving the causal invariants above. Its minimal form is

\[
P_{t+1}=F_\theta(P_t,e_t),
\qquad
L_{t+1}=L_t+s_t^L k_t,
\]

\[
m_{ij}=\underbrace{s_i(P_T)-s_j(P_T)+c_{ij}(P_T)}_{\text{global field}}
+\underbrace{\lambda_L\,L_T\cdot k_{ij}}_{\text{local direct field}},
\qquad
p_{ij}=\sigma(m_{ij}/T).
\]

That abstraction is a target for the next read-only compression test, not yet
a new confirmed mechanism. It should be judged by whether it preserves the
registered `P`/`L` interventions, global Hodge field, direct-only local effect,
and the behavioral reproduction map on one to three development artifacts.
Only after that compression is frozen should the project test item-count,
support-topology, sparsity/order/magnitude-placement, adjacent transitive
inference, and list-linking generalization.

## Evidence sources

- `docs/formal_confirmation_v1.md`
- `docs/assembly_trajectory_v1.md`
- `docs/support_write_localization_v1.md`
- `docs/support_factor_swap_v1.md`
- `docs/history_state_factorial_v1.md`
- `docs/conjunctive_local_trace_replication_v2_3.md`
- `docs/dual_evidence_access_confirmation_v2_4.md`
- `docs/model_behavior_reproduction_map_v1.md`
