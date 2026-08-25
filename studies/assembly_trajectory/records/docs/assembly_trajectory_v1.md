# Assembly trajectory and relation-LOO diagnostic v1

## Status and provenance

This registered read-only diagnostic is complete for frozen pilot seeds 1901
and 1902. The protocol was committed and pushed as `50424a1` before the
analysis implementation was run. No training, checkpoint selection,
architecture change, formal-seed access, or temperature refit occurred.

The complete subject-level result is
`results/assembly_trajectory_v1.json` (SHA-256
`0d47b8b5e61c3b05e0e8cbfe20b3c3f92d89f2d8a6f26388fbff13437a676b3c`).
Neural forward passes used CUDA on an RTX 5090. The exact 40,320-order prefix
posterior used batched NumPy enumeration on CPU.

The question was when and where the frozen networks transform sparse,
subject-specific evidence into the near-pure additive logit potential found by
the preceding assembly diagnostic.

## Registered estimands and two necessary refinements

The natural support trajectory contains `P_0`, after the registered two blank
initialization steps, through `P_32`, after every support trial. All 28 pairs
were queried in both orientations from every prefix with hidden state and
eligibility reset before each query.

For trial `t`, the primary causal branch was

\[
\Delta m_{t,q}=m_q(P_t^+)-m_q(P_t^0),
\]

where both states start from the same natural `P_(t-1)`. The zero branch keeps
the trial cue identities, orientation, time, and four recurrent steps and sets
only signed magnitude to zero. Natural `P_t-P_(t-1)` was not substituted for
this estimand.

For relation `e`, leave-one-relation-out replay kept all 32 support slots and
zeroed magnitude for all four presentations of `e`:

\[
I_{e\rightarrow q}=m_q(P_{32})-m_q(P_{32}^{(-e)}).
\]

Two mathematical refinements were fixed before execution:

1. On a complete pair graph, the centered normalized Hodge potential of the
   exact posterior pair field is algebraically identical to centered negative
   posterior expected rank. They are one distributional target, not two
   independent competitors. Their maximum numerical discrepancy was
   `1.06e-15`. The identifiable contrast is distributional potential versus a
   hard MAP order.
2. The requested raw third-party potential energy can be inflated by the
   sum-zero gauge. The analysis therefore retains it and adds
   `R_third_rel`, which removes the common shift among third-party items.
   Remote pair margins and `R_third_rel` are the primary gauge-invariant
   evidence for third-party reassembly.

Uniform-posterior potentials with norm at or below `1e-12` are treated as zero,
so prefixes with no retained evidence have undefined rather than arbitrary
normalized directions.

## Results

### Additive content forms rapidly and strengthens throughout support

| Seed | `G_0` | `G_4` | `G_32` | `A_0` | `A_32` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1901 | 0.92360 | 0.99052 | 0.99580 | 0.53696 | 4.82834 |
| 1902 | 0.72689 | 0.98924 | 0.99686 | 0.22451 | 4.94521 |

`G_t` is each subject's neural-logit Hodge gradient-energy fraction and `A_t`
is the norm of its centered potential. By four support trials, the mean field
was already about 99% additive. Commitment strength then continued to rise,
and the prefix potential approached its own final direction: final-potential
cosine was `0.67327/0.67602` at `P_4`, `0.90795/0.91314` at `P_8`, and
`0.96530/0.96520` at `P_16` for seeds 1901/1902.

This supports progressive commitment in the support-written state. It does not
mean that the entire fast-weight matrix or hidden state is one-dimensional.

### The final neural potential tracks a distribution, not a MAP commitment

| Seed | Exact distributional cosine | MAP cosine | True-order cosine | Distributional minus MAP, 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 0.88721 | 0.84121 | 0.81660 | [0.02324, 0.07207] |
| 1902 | 0.89077 | 0.84156 | 0.82413 | [0.02466, 0.07774] |

The registered distributional-over-MAP direction passes in both seeds. The
network is better described as projecting distributed posterior pair evidence,
equivalently posterior expected rank, into one additive potential than as
selecting a single MAP ranking. The cosine remains below one and the previous
slope diagnostic showed neural over-sharpening, so this is not exact Bayesian
inference.

### One retained trial causally changes disjoint pairs

Subject-averaged absolute matched-branch effects were:

| Seed | Direct | Shares one endpoint | Remote/disjoint | Remote/direct |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 1.74991 | 0.87449 | 0.28045 | 0.16027 |
| 1902 | 1.86108 | 0.92603 | 0.32301 | 0.17356 |

Remote effects were present from the first retained support trial and remained
present at every registered prefix. Stable-omitted presentations produced
exactly zero effect in both seeds. Thus a support observation is not written as
an isolated pair-only scalar: it immediately changes queries sharing neither
endpoint.

However, the mean correctness-aligned remote effect was unresolved:
`-0.00017` with interval `[-0.01889, 0.01896]` for seed 1901 and `-0.00472`
with interval `[-0.02587, 0.01761]` for seed 1902. Causal reach is therefore
not equivalent to uniformly correct transitive propagation.

### Relation LOO confirms global reassembly and exposes redistribution

| Seed | Direct | Shares one endpoint | Remote/disjoint | Influence-field `G` | `R_third` | `R_third_rel` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1901 | 2.39564 | 1.17282 | 0.42410 | 0.99222 | 0.19381 | 0.19099 |
| 1902 | 2.47982 | 1.20746 | 0.44675 | 0.99474 | 0.19762 | 0.19534 |

All values except `G` and the `R` fractions are mean absolute logit-margin
influences. Remote influence was 17.7% and 18.0% of direct influence. About 19%
of the relation-induced potential energy remained among third-party items even
after removing their common gauge shift. Influence fields themselves were
almost purely additive. Leaving out an already stable-omitted relation again
gave exactly zero pair influence.

The negative result is informative: mean correctness-aligned remote LOO
influence was `-0.06764`, interval `[-0.07867, -0.05671]`, and `-0.06961`,
interval `[-0.08087, -0.05857]`. A retained relation globally rebalances the
potential, but its isolated remote contribution can reduce other true-order
margins. The final correct global structure arises from interacting evidence,
not a sum of independently correctness-propagating messages.

### Fast weights carry content; additive form is already available at query onset

Reset, write-off, and `alpha=0` all reproduced the content-free baseline. Their
mean output-field gradient fractions were `0.92360/0.72689`, commitment
strengths were `0.53696/0.22451`, and exact-content cosines were near zero. In
contrast, intact `P_32` had `G=0.99580/0.99686`, strength
`4.82834/4.94521`, and exact-content cosine `0.88721/0.89077`. Intact-minus-
control exact-alignment intervals were wholly positive in both seeds.

At the first query cue step, intact vector-hidden fields were already highly
additive: `G_h^0=0.99090/0.99083`. At the registered response step they were
slightly less additive, not more:

| Seed | `G_h^0` | response `G_h^1` | response minus first, 95% interval | response `G_l^1` | `G_l^1-G_h^1`, 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1901 | 0.99090 | 0.98465 | [-0.00660, -0.00591] | 0.99580 | [0.01090, 0.01142] |
| 1902 | 0.99083 | 0.98563 | [-0.00550, -0.00492] | 0.99686 | [0.01098, 0.01150] |

This rejects query-time recurrent commitment as the source of additive form.
The content-bearing potential is already accessible to the first query-step
hidden dynamics, while the fixed output direction selects a modestly more
additive component at the response step. High `G_h` also occurs at `P_0`, but
without episode-specific content; it is a form prior, not a stored order.

## Revised mechanism

The combined evidence now supports the following working route:

\[
D_s
\longrightarrow
\{\Delta P_t^{(s)}\}_{t=1}^{32}
\longrightarrow
\text{interacting global potential updates}
\longrightarrow
s_{32}^{(s)}
\approx
\text{posterior expected-rank/Hodge potential}
\longrightarrow
w_{out}^{\mathsf T}h^-.
\]

The fast-weight pathway is necessary for episode-specific content. Individual
updates have broad, almost additive causal influence, including differential
third-party changes, but they are not independent correct transitive messages.
Across the episode, their interactions commit progressively to a coherent
distributional potential. Query recurrence exposes that structure immediately;
the output direction further suppresses its small non-additive component.

This revises two candidate links without shrinking the project question:

- replace hard MAP selection with a learned distributional-potential
  projection;
- replace query-time construction with support-time fast-weight assembly plus
  query-time readout selection.

The remaining mismatch is the learned transformation's over-sharpening and its
suppression of the learned-pair residual present in human choices.

## Claim boundary and next decisive test

- These are two frozen pilot seeds, not formal confirmation seeds.
- Global causal reach does not imply that every remote change is correct or
  that the network implements the registered exact likelihood.
- `G_h` is vector-field form across queries, not hidden-manifold dimensionality.
- The result localizes episode content to the fast-weight pathway, not to an
  individual synapse, eligibility trace, or neuromodulatory unit.

Before architecture changes or formal scaling, the next registered diagnostic
should localize the support-time update itself. For every retained trial it
should relate DA, eligibility, `alpha * Delta P_t`, and the resulting neural
potential change to the exact posterior expected-rank update, while testing
whether the late decline in matched-branch magnitude reflects saturation or
evidence interaction. This directly addresses how bidirectional remote
redistribution accumulates into the final correctly aligned potential.

## Reproduction

```bash
direnv exec . python -m fsrl.assembly_trajectory
direnv exec . python -m json.tool results/assembly_trajectory_v1.json >/dev/null
```
