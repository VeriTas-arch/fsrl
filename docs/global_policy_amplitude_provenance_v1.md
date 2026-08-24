# Global-policy amplitude provenance v1

## Conclusion

The registered read-only audit returns
`comparator_sensitive_unresolved`. It rejects the stronger assumption that the
excessive nonlearned probability-distance slope is caused by a globally
over-large neural policy-potential amplitude relative to the exact posterior.

The prior Track-B result remains intact: **within the neural policy**, the
additive Hodge potential carries more than the full positive distance slope,
while the neural residual and fixed sigmoid compress it. The new result shows
that this internal decomposition does not license a cross-comparator claim of
global over-amplitude. In the preregistered common policy-margin units, neural
additive policy-potential amplitude is substantially *smaller* than the exact-
posterior additive comparator in both networks, and the additive probability-
potential contrast is unresolved.

The working correction is therefore:

```text
supported: P_T receives nearly all signed allocation along the intact additive-potential direction
supported: that neural additive field carries the neural distance slope
rejected: the neural additive-potential norm is too large relative to exact posterior
unresolved: how pairwise confidence is allocated to produce the slope mismatch
```

No scalar calibration, state normalization, readout normalization, temperature
fit, or other scale intervention is authorized.

## Frozen protocol and execution

- The original protocol was registered and pushed as `b7daef8` before the
  runner was written.
- Independent static review found that cumulative elasticities could hide
  cancellation. The superseding two-axis/four-atomic-term decision repair was
  committed as `4229fd3` before any checkpoint was loaded for this audit; the
  timing clarification was committed as `0ed9dcf`.
- Runner, tests, formal dispatch, and implementation/source lock were committed
  and pushed as `fba1418` before neural replay.
- The only networks are the jointly locked fresh backbones 2104 and 2105. Each
  uses all 77 participants, all 28 canonical edges, and both orientations.
- The primary condition is pure `L`-off with intact terminal `P_T`; `P=0` is an
  exact query-time control on the same learned support state.
- No training, adaptation, checkpoint selection, parameter mutation,
  temperature fit, participant filtering, network pooling, or new Liu-derived
  variable occurred.
- Execution used the NVIDIA GeForce RTX 5090 through
  `python -m fsrl.formal_runtime`, with one PyTorch intra-op and one inter-op
  CPU thread.
- A second complete GPU execution to `/tmp` is byte-identical. Result SHA-256:
  `b94a7ff154d66553fedf48469eb509dc7f041e7a9454143228c9a7963b1e61b4`.

Every source, artifact, runtime, parameter-immutability, reconstruction,
Hodge, posterior, denominator, subject-count, and bootstrap-finiteness gate
passes separately in both networks.

## 1. The Track-B premise replicates exactly

The frozen participant-level neural-minus-posterior probability-slope mismatch
is reproduced to at most `3.47e-17` and remains robustly positive:

| Seed | Mean mismatch | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.05774 | [+0.05098, +0.06444] |
| 2105 | +0.05876 | [+0.05295, +0.06445] |

Thus the earlier empirical problem has not disappeared. What changes is its
mechanistic interpretation.

## 2. Common-unit additive-potential amplitude reverses the proposed direction

For each participant, the exact posterior pair probability was converted
without clipping to the margin that would produce it under the same frozen
choice temperature:

```text
m_post,ij = T_choice [LSE(log w_pi : i>j) - LSE(log w_pi : j>i)]
s_post    = B_plus m_post
a_post    = ||s_post||_2
```

The neural comparator is the additive Hodge potential of its actual response
margin, `a_N=||s_N||_2`. These norms do not include either edge field's Hodge
residual. The primary mismatch is

```text
Y = log(a_N) - log(a_post).
```

| Seed | `a_N` | `a_post` | Mean `Y` | 95% bootstrap for `Y` |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 5.11052 | 13.02152 | -0.87606 | [-0.94103, -0.80612] |
| 2105 | 5.28474 | 13.02152 | -0.84235 | [-0.90962, -0.76925] |

The geometric-mean neural/posterior amplitude ratios are approximately 0.416
and 0.431, not greater than one. This fails the registered positive-`Y`
premise in the opposite direction.

The mandatory probability-space check does not supply an alternative PASS:

| Seed | Mean `d_prob` | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.01797 | [-0.00813, +0.04855] |
| 2105 | +0.02089 | [-0.00646, +0.05486] |

Both intervals cross zero. The preregistered outcome is therefore
`comparator_sensitive_unresolved`; neither final-comparator nor layer-
elasticity attribution is evaluated.

## 3. The descriptive scalar fit fails its registered rule

The frozen through-origin fit

```text
s_N approximately equals c s_post
```

does not satisfy either the direction or shape rule:

| Seed | Scale `c` | 95% bootstrap | Energy explained | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 0.34296 | [0.33112, 0.35648] | 0.82376 | [0.79806, 0.84860] |
| 2105 | 0.35450 | [0.34089, 0.36999] | 0.82288 | [0.79655, 0.84759] |

The scale is below one, not above one, and the lower confidence bounds for
explained energy are far below the registered 0.90 rule. Normalized vector
cosine remains high (about 0.918 in both networks), but approximately 18% of
neural-potential energy is not captured by one scalar posterior map. Because
the comparator gate stopped first, no final-comparator axis was evaluated;
descriptively, the scalar fit also fails the 0.90 rule and cannot support a
constant calibration fingerprint.

The mismatch becomes more negative as admitted coverage and posterior
certainty increase. The registered `Y~coverage` slopes are -1.6500 and
-1.6150, and `Y~certainty` slopes are -2.2703 and -2.2110; all four 95%
intervals exclude zero negatively. The neural additive-potential amplitude
therefore undertracks the comparator's growth with evidence/certainty rather
than overshooting it.

## 4. Positive `P_T` provenance is preserved

The query-time `P=0` subtraction characterizes episode-state provenance along
the intact neural additive-potential direction:

| Seed | Mean `phi_P` | 95% bootstrap | Mean `phi_0` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 1.00086 | [0.99576, 1.00611] | -0.00086 | [-0.00611, 0.00424] |
| 2105 | 0.99827 | [0.99348, 1.00340] | +0.00173 | [-0.00340, 0.00652] |

These are signed squared-amplitude allocations, not probabilities or norm
shares. Nearly all signed allocation along the intact additive-potential
direction is assigned to the `P_T`-induced contrast; the baseline allocation
is unresolved around zero and was not tested for equivalence. This preserves,
rather than weakens, the established `P_T` global-assembly mechanism.

The exact amplitude ledger is also numerically valid:

```text
a_N = a_P g_rec g_out g_mix
g_out = ||w|| rho_W
```

Maximum ledger error is `1.78e-15`, all required amplitudes are nonzero, and
the frozen parameter hashes are unchanged. Because the comparator gate stops
the decision tree, the observed atomic elasticity pattern is reported but not
promoted to a source claim: `e_P` is materially negative, `e_out` materially
positive, and `e_rec/e_mix` equivalent to zero in both networks. It cannot be
called drive-, readout-, or co-adapted provenance under this contract.

## Revised theory and claim boundary

The full evidence now supports the following distinction:

```text
selective evidence -> P_T -> P-dominant additive neural policy field
                               -> excessive probability-distance slope

exact posterior -> larger common-unit additive-potential amplitude
                   with a different confidence allocation/shape
```

Track B identified where the slope resides *inside the neural field*. The new
audit rejects the additional assumption that “where the neural slope resides”
is the same as “which additive-potential norm is too large relative to the
comparator.”
The excessive slope can coexist with a smaller additive-potential norm when
confidence is distributed differently across pairs/distances and undergoes a
different pattern of compression after the fixed sigmoid. The probability-
space allocation itself remains unresolved.

This conclusion remains conditional on the frozen exact posterior (uniform
prior, squared-residual energy, posterior temperature 0.05) and the shared
choice temperature 0.25. It does not establish the human posterior, identify a
biological store, infer network-population prevalence, or causally identify a
state/readout source. The replicated v2.4 `P_T/L_T` learning mechanism is
unchanged.

## Next decisive test

Do not perform a scalar scale correction. Before any model change, a separate
protocol should freeze a same-unit **neural/posterior policy-field factorial**:

```text
m_N    = g_N    + c_N
m_post = g_post + c_post
```

where `g` is the complete-graph Hodge/additive field and `c` its residual. The
read-only 2-by-2 source swap `(g_N/g_post) x (c_N/c_post)`, evaluated through
the same fixed sigmoid on the same 20 nonlearned pairs, would be a field-level
reassembly/sufficiency test, not network-internal `P_T` or `W_out` causality.
Its frozen outcomes could distinguish within this fixed-field reassembly:

- an additive replacement that determines the fixed slope contrast;
- a residual replacement that determines the fixed slope contrast;
- a registered additive-by-residual interaction;
- an unresolved field mismatch under the frozen exact-posterior comparator.

Let `S_ab` denote the probability-distance slope with additive source `a` and
residual source `b`, where each source is neural `N` or posterior `P`. The
contract must first replicate `D=S_NN-S_PP>0`, then freeze paired replacement
contrasts and equivalence margins. The factorial interaction is

```text
I = (S_NN - S_PN) - (S_NP - S_PP).
```

It cannot be inferred merely because neither one-component swap reaches the
posterior anchor. Conservative outcomes are additive-source sufficient,
residual-source sufficient, both components required with interaction status
reported separately, or registered contrasts unresolved.

A norm-matched additive control should separate shape/allocation from global
amplitude. The contract must freeze the norm-match target and direction,
natural anchors, 20-pair mask, participant-within-network bootstrap without
pooling, exact `g+c` reconstruction, factorial interaction, and closure and
equivalence margins. It must not be appended post hoc to the present result.
Until then, preserve `comparator_sensitive_unresolved` and make no scale
intervention.
