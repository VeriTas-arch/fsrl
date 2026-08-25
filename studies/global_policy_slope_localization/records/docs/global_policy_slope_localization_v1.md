# Global-policy symbolic-distance slope localization v1

## Conclusion

The frozen read-only diagnostic localizes the excessive nonlearned symbolic-
distance dependence to the **scale of the additive global policy potential**,
not to over-distance-like normalized geometry, a positive Hodge residual, or
amplification by the fixed sigmoid choice link.

This pattern replicates independently in frozen fresh backbones 2104 and 2105:

```text
P_T
  -> additive policy potential with large natural amplitude  [slope source]
  -> small negative Hodge residual                           [compression]
  -> fixed sigmoid at T=0.25                                 [compression]
  -> excessive nonlearned exact-probability slope
```

The result narrows the next question to where `P_T -> policy potential` acquires
its over-sharp scale. It does not show that normalized neural and posterior
geometry are equivalent, identify whether scale originates in recurrent state
or frozen readout gain, authorize a temperature fit, or change the confirmed
v2.4 local/global learning mechanism.

## Frozen protocol and execution

- Registration was committed and pushed as `6fda137` before implementation.
- The runner, tests, formal-runtime dispatch, and exact source lock were
  committed and pushed as `c7d3f93` before neural replay.
- Both mandatory networks were evaluated in one command through
  `python -m fsrl.formal_runtime global-policy-slope-localization`.
- The primary condition is pure `L`-off with intact `P_T` and frozen `W_out`.
  No local trace was constructed or read, and no training, gain adaptation,
  checkpoint selection, temperature refit, or model mutation occurred.
- Participants were bootstrapped separately within network and never pooled.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, one
  PyTorch intra-op thread and one inter-op thread.
- Result SHA-256:
  `a10060d6bd87751ea0b9303b6228c3059914c58c5234ddeeed3f8c2096eb4330`.
  An independent complete GPU replay to `/tmp` is byte-identical.

All 18 registered source checks, the joint checkpoint/artifact lock, both
77-participant evaluations, and every exact numerical identity pass.

## 1. The slope is already in the additive potential

For the 20 nonlearned pairs, the correct-signed global margin was decomposed
subject by subject as

```text
m = g + c,
g = B s,
beta_m = beta_g + beta_c.
```

| Seed | `beta_m` | `beta_g` | `beta_c` |
| --- | ---: | ---: | ---: |
| 2104 | 0.63833 [0.60239, 0.67411] | 0.65944 [0.62102, 0.69778] | -0.02112 [-0.02628, -0.01607] |
| 2105 | 0.65856 [0.62060, 0.69652] | 0.68437 [0.64362, 0.72529] | -0.02581 [-0.03139, -0.02028] |

The registered potential-dominance contrast
`beta_g - 0.9 beta_m` is positive in both networks:

| Seed | Contrast | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.08495 | [+0.07744, +0.09245] |
| 2105 | +0.09166 | [+0.08363, +0.09990] |

The additive potential supplies more than the full positive margin slope;
the nonadditive residual is small and reliably counteracts it. A residual or
readout-corruption account is therefore rejected for this frozen estimand.

## 2. Normalized geometry is not excessively distance-like

The potential was separated exactly as

```text
s = a hat_s,
a = ||s||_2,
beta_g = a beta_hat_s.
```

| Seed | Natural amplitude `a` | Neural `beta_hat_s` | Posterior `beta_hat_s` | Neural minus posterior |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 5.11052 [4.91444, 5.30355] | 0.12806 [0.12338, 0.13246] | 0.15142 [0.14884, 0.15351] | -0.02336 [-0.02775, -0.01930] |
| 2105 | 5.28474 [5.08109, 5.48558] | 0.12841 [0.12348, 0.13316] | 0.15142 [0.14879, 0.15352] | -0.02300 [-0.02790, -0.01838] |

Neural and exact-posterior normalized potentials remain strongly aligned
(mean cosine 0.87086 and 0.87243), but they are not equivalent. Crucially, the
neural normalized geometry is reliably **less**, not more, distance-dependent
than the posterior expected-rank comparator. The excessive raw slope must
therefore arise downstream of normalized shape; the large natural potential
amplitude is the leading source under the frozen decision tree.

This is a policy-scale conclusion. The present diagnostic does not yet split
recurrent `P_T` state magnitude from the gain with which frozen `W_out` maps
that state into margins.

## 3. The sigmoid compresses rather than amplifies

At the frozen choice temperature `T=0.25`, the neural nonlearned exact-
probability slope substantially exceeds the exact-posterior comparator:

| Seed | Neural `beta_p` | Posterior `beta_p` | Neural minus posterior |
| --- | ---: | ---: | ---: |
| 2104 | 0.06840 [0.06177, 0.07508] | 0.01066 [0.00708, 0.01473] | +0.05774 [+0.05122, +0.06440] |
| 2105 | 0.06942 [0.06336, 0.07557] | 0.01066 [0.00708, 0.01467] | +0.05876 [+0.05306, +0.06459] |

The registered within-subject projection

```text
p = intercept + kappa (y m) + e,
beta_p = kappa beta_m + beta_e
```

shows that the nonlinear remainder is negative:

| Seed | Linearized contribution | Nonlinear remainder `beta_e` | 95% bootstrap for `beta_e` |
| --- | ---: | ---: | ---: |
| 2104 | 0.07803 | -0.00963 | [-0.01380, -0.00542] |
| 2105 | 0.07974 | -0.01031 | [-0.01422, -0.00629] |

Thus the fixed sigmoid is already attenuating the upstream distance structure.
It does not create the excessive slope. Fitting a new temperature would mask
the upstream scale problem rather than explain it.

## 4. v2.4 preservation check

The already frozen dual-intact v2.4 result retains the same qualitative
fingerprint and was not mixed into the primary decomposition:

| Seed | Dual-intact exact slope | Nonlearned contribution | Fraction of total |
| --- | ---: | ---: | ---: |
| 2104 | 0.06068 | 0.04933 | 81.30% |
| 2105 | 0.05989 | 0.04901 | 81.83% |

Nonlearned policy remains the largest exact slope source in both networks.
This preserves the separation between confirmed differential local evidence
access and the unresolved global confidence-scale mechanism.

## Integrity

- Maximum margin-field identity error: exactly zero.
- Maximum `beta_m = beta_g + beta_c` error: `3.89e-16`.
- Maximum `beta_g = a beta_hat_s` error: `4.44e-16`.
- Maximum sigmoid-link slope identity error: `1.21e-16`.
- Maximum exact-posterior Hodge/expected-rank equivalence error: `6.18e-16`.
- Both networks contain all 77 participants and all 20 nonlearned pairs.
- Two full executions are byte-identical; no participant, pair, relation,
  checkpoint, or network was filtered.

## Revised theory and next decisive test

The working model is now more specific:

```text
selective global admission -> P_T -> coherent, posterior-like geometry
                                  -> over-sharp policy-potential scale
                                  -> excessive nonlearned confidence slope

broader weak local admission -> L_T -> addressed direct fidelity
```

Before any intervention, the next contract should be a read-only amplitude-
provenance audit. It should separate recurrent `P_T` state magnitude from
`W_out` projection gain and test whether subjectwise policy amplitude tracks
registered evidence coverage and exact-posterior uncertainty. Possible
outcomes have distinct implications:

- If `P_T` magnitude is already over-sharp or insensitive to uncertainty,
  target global confidence/state normalization while preserving direction.
- If `P_T` scale is appropriate but `W_out` projection gain is excessive,
  target the learned state-to-policy calibration mechanism.
- If both track posterior uncertainty but a constant units mismatch remains,
  register a generic-only scale-calibration sufficiency test; do not fit on Liu
  or call a post-hoc temperature a mechanism.

No such intervention is authorized by the present read-only contract.
