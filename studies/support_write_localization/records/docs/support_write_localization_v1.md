# Support-write localization diagnostic v1

## Status and provenance

This registered read-only diagnostic is complete for frozen pilot seeds 1901
and 1902. The specification was committed and pushed as `bbd4fc8` before the
analysis implementation was run. No training, checkpoint selection,
architecture change, formal-seed access, or parameter refit occurred.

The complete subject-level result is
`results/support_write_localization_v1.json` (SHA-256
`61c9d877876e6ecead4f5d9f17df65ba36fc3e92e1a38cfc0bd7703e06c789dc`).
Neural evaluation used CUDA; exact 40,320-order posterior innovations and
bootstrap summaries used CPU NumPy.

Two robustness checks were added only after inspecting the first registered
output: DA correlations residualized within subject by relation exposure, and
functional-amplification contrasts for the alpha controls. The result labels
these separately as post-result exploratory analyses. They refine the working
theory but are not retroactively treated as registered decision rules.

The diagnostic asks how one local support observation enters the episode-local
fast weights and changes the globally accessible item potential.

## Registered estimands

For every support step, the traced model order is

\[
h_k=f\left(x_k,h_{k-1};W+\alpha\odot P_k\right),\qquad
d_k=w_{DA}^{\mathsf T}h_k,
\]

\[
P_{k+1}=\operatorname{clip}(P_k+d_k E_k),
\]

followed by the eligibility update. Each support trial starts with zero hidden
state and zero eligibility. Consequently, nonzero DA at steps 0 and 1 cannot
write because the preceding eligibility is zero; relation-specific writes can
first occur at steps 2 and 3.

The retained-evidence branch and its matched zero-magnitude branch start from
the same natural pre-trial state. Before clipping, their write difference is
decomposed exactly as

\[
\Delta U=
\underbrace{\tfrac12(d^+-d^0)(E^++E^0)}_{\Delta U_{DA}}+
\underbrace{\tfrac12(d^++d^0)(E^+-E^0)}_{\Delta U_E}.
\]

Each component is also replayed through the frozen query policy. The alpha
test compares the observed effective increment with norm-matched scalar-alpha
and shuffled-alpha increments around the same zero-branch state. Trial-level
neural potential changes are compared with the exact posterior expected-rank
innovation and a fixed true-order direction. Clipping, write generation,
downstream functional gain, and removal of earlier same-relation evidence
separate candidate explanations of late attenuation.

## Results

### Eligibility carries the effective write direction

The registered timing check passed exactly: effective write was zero at steps
0 and 1 in both seeds and nonzero at steps 2 and 3.

| Seed | Step | Total effective write | DA component | Eligibility component |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 2 | 0.50690 | 0.01917 | 0.50916 |
| 1901 | 3 | 0.23919 | 0.01302 | 0.23688 |
| 1902 | 2 | 0.45141 | 0.01616 | 0.45233 |
| 1902 | 3 | 0.25498 | 0.01017 | 0.25397 |

Across each retained trial, the summed eligibility-component effective norm
was `0.65296/0.61238`, whereas the DA-component norm was only
`0.02626/0.01992` for seeds 1901/1902. The paired eligibility-minus-DA 95%
bootstrap intervals were `[0.61073, 0.64263]` and `[0.57666, 0.60811]`.
Functional replay gave the same assignment: eligibility-only policy-update
norm was `1.35956/1.45717`, while DA-only norm was `0.02309/0.01902`.

Thus relation-specific, high-dimensional write content enters primarily
through the evidence-dependent eligibility pattern. DA is not the source of
that direction.

### DA has a modest gain signal, not an exact scalar innovation code

The registered raw within-subject association of summed `abs(d+)` with exact
information gain passed strongly (`rho=0.9323/0.9210`). Much of this magnitude
is shared exposure structure: both DA and exact innovation decline across the
four presentations.

The post-result exposure-adjusted check retained a smaller positive association
with information gain: `rho=0.1973`, interval `[0.1493, 0.2431]`, and
`rho=0.1741`, interval `[0.1218, 0.2228]`. Its association with exact
expected-rank update norm was weaker: `0.0649 [0.0204, 0.1094]` and
`0.0498 [0.0009, 0.0973]`.

The evidence-specific matched DA difference was not a unitary surprise code.
After exposure adjustment, it correlated positively with expected-rank update
norm (`rho=0.1509/0.1513`) but was unresolved or negative for information gain
(`rho=-0.0382/-0.0522`). Together with its very small causal replay effect,
this supports a coarse state- and innovation-sensitive gain role, not the
proposed identity `d_t = exact Bayesian information gain`.

### Alpha amplifies functional expression but does not uniquely set direction

The registered alpha structural-mapping rule failed. Actual-alpha alignment to
the exact innovation was `0.31768/0.30974`; scalar-control alignment was
`-0.31575/0.31125`, and shuffled-control alignment was `0.32026/0.30454`.
Actual alpha did not beat both controls with positive paired intervals in
either seed.

The exploratory norm contrast was nevertheless consistent. All effective
increments were norm matched before query replay, yet actual-alpha policy
updates exceeded the scalar control by `0.2365 [0.2268, 0.2459]` and
`0.2320 [0.2225, 0.2414]`, and exceeded the shuffled control by
`0.9443 [0.9228, 0.9659]` and `0.9610 [0.9364, 0.9861]`.

Alpha therefore appears to place the eligibility-derived write in a
functionally sensitive recurrent subspace. It cannot currently be assigned the
stronger role of mapping the write into its unique posterior-update direction.

### Trial updates are innovation-aligned, but not exact stepwise Bayes

The registered positive-alignment rule passed. Mean neural-to-exact innovation
cosine was `0.31768`, interval `[0.29054, 0.34450]`, and `0.30974`, interval
`[0.28264, 0.33679]`. This is causal trial-level evidence connecting fast-weight
writes to the distributional potential found in the final state.

The effect is concentrated early:

| Exposure | Exact information gain | Exact update norm | Neural/exact cosine, seed 1901 | Neural/exact cosine, seed 1902 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.95806 | 1.99268 | 0.72448 | 0.71409 |
| 2 | 0.05947 | 0.12960 | 0.27013 | 0.26235 |
| 3 | 0.01967 | 0.06554 | 0.16327 | 0.15508 |
| 4 | 0.00889 | 0.04138 | 0.11284 | 0.10745 |

The fixed true-order cosine stayed near `0.33`. Across all trials, exact minus
true-order alignment was unresolved in both seeds: `[-0.04355, 0.01026]` and
`[-0.05047, 0.00833]`. The positive result is therefore that each write has a
component along the exact posterior innovation, especially on first exposure.
It does not establish that every update implements the exact Bayesian step or
that exact innovation is preferred to a fixed order direction throughout the
episode.

### Late attenuation is generated before readout and becomes relation-specific

No traced entry reached the `[-50, 50]` clip bound. From exposure 1 to 4,
effective-write norm declined by `-0.2317 [-0.2474, -0.2171]` and
`-0.1208 [-0.1333, -0.1090]`; policy-update norm declined by
`-0.4155 [-0.4484, -0.3832]` and `-0.2961 [-0.3303, -0.2638]`.
Functional gain increased in seed 1901 but decreased in seed 1902. Literal
clipping and a replicated downstream saturation mechanism are therefore
rejected; attenuation is already present in write generation.

Removing earlier observations of the current relation increased the current
policy update over exposures 2--4 on average, with intervals
`[0.00780, 0.04000]` and `[0.00146, 0.03311]`, satisfying the registered
assimilation rule. The exposure-resolved result is more informative: the
effect was negative or unresolved at exposure 2, mixed at exposure 3, and
positive in both seeds at exposure 4 (`[0.03419, 0.08100]` and
`[0.02594, 0.07254]`). Relation-specific redundancy is thus a robust late
effect, not a monotonic law beginning with the second presentation.

Stable-omitted observations produced exactly zero exact innovation, write, and
policy update. Traced forward passes, incremental endpoints, final endpoints,
and explicit effective-connectivity readouts reproduced the frozen evaluator;
the maximum DA/eligibility decomposition residual was `2.38e-7`.

## Revised mechanism

The joint positive and negative pattern supports the following factorization:

\[
D_{s,t}
\longrightarrow
\underbrace{E_t}_{\text{relation-specific write direction}},
\qquad
\underbrace{d_t}_{\text{coarse state/innovation-sensitive gain}},
\]

\[
d_t E_t
\longrightarrow
\underbrace{\alpha\odot\Delta P_t}_{\text{functionally amplified expression}}
\longrightarrow
\Delta s_t
\longrightarrow
P_T\mapsto s_T\approx-\mathbb E[\operatorname{rank}\mid D_s].
\]

`P` remains the persistent integrated episode state. Eligibility supplies most
of the content and direction, DA weakly scales it according to state and
innovation context, and alpha makes the write effective for the frozen
recurrent policy. The network strongly resembles an expected-rank update on
the first presentation, then continues smaller globally structured corrections
whose direction is not identical to each vanishing exact Bayesian innovation.
Late relation-specific assimilation contributes to attenuation by the fourth
presentation.

This replaces, rather than merely weakens, three parts of the initial
three-factor proposal:

- DA is not an exact scalar Bayesian surprise variable.
- Alpha is not identified as the unique posterior-direction map.
- Final posterior-like alignment does not arise from uniformly exact and
  independently correct trial updates.

The supported backbone remains: stable effective evidence enters an
eligibility-dominated plastic write; the trained recurrent system expresses
that write as a broad additive potential update; interacting, state-dependent
updates progressively assemble the final individualized global potential.

## Claim boundary and next decisive test

- These are two frozen pilot seeds, not formal confirmation seeds.
- The modeled scalar DA signal has no established biological identity.
- The exposure-adjusted DA and alpha-amplification checks are explicitly
  post-result exploratory robustness analyses.
- Positive cosine is directional overlap, not equality of update magnitude or
  proof of exact Bayesian computation.
- The history control identifies a late relation-specific state dependence; it
  does not exclude all other causes of write attenuation.

Before formal seeds or architecture changes, the next registered read-only
test should causally separate gain, direction, and history within exposure. A
matched factor-swap design can exchange DA scalars and eligibility matrices
among trials at the same exposure while holding the pre-query state and alpha
map fixed. The prediction is that eligibility swaps transfer relation-specific
direction, whereas DA swaps primarily rescale magnitude. Natural versus
no-prior-relation histories should then determine whether the fourth-exposure
reduction is generated in eligibility, DA, or their interaction. Multiple
preregistered alpha permutations or a local sensitivity analysis should test
whether the observed norm amplification reflects systematic alignment with a
high-gain recurrent subspace rather than one favorable shuffle.

## Reproduction

```bash
direnv exec . python -m fsrl.support_write_localization
direnv exec . python -m unittest tests.test_support_write_localization -v
direnv exec . python -m json.tool results/support_write_localization_v1.json >/dev/null
```
