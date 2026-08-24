# Global-policy field reassembly v1

## Conclusion

The registered read-only neural/posterior additive-by-residual reassembly
returns `mixed_or_unresolved` independently in seeds 2104 and 2105. No natural
single-component replacement closes the frozen nonlearned probability-distance
slope mismatch.

The result nevertheless contains three cross-network-reproduced field-level
constraints:

1. The neural-versus-posterior additive-source main effect is materially
   positive.
2. At the neural full-field additive norm and with the neural residual fixed,
   posterior additive shape materially reduces the slope, but does not close
   it.
3. The fixed-sigmoid additive-by-residual interaction is materially negative.

Thus Track B's within-neural additive-field localization remains supported,
and the stronger scalar-overgain account remains rejected. The cross-comparator
discrepancy is not closed by natural additive replacement, norm-matched
additive shape alone, or residual replacement alone. It is a comparator-
relative field-allocation problem with probability-link context dependence and
a remaining closure gap.

No model intervention, temperature fit, `P_T` normalization, `W_out`
normalization, new hybrid cell, or alternative comparator is authorized.

## Frozen protocol and execution provenance

- The original 2-by-2 contract was committed and pushed as `be265eb` before
  this diagnostic runner was written or either checkpoint was loaded.
- Static implementation review then found that the direct-margin Hodge bridge
  had incorrectly required exact reproduction of the prior hidden-derived
  amplitude. Before any checkpoint replay, the contract was transparently
  repaired in `4b1c2a5`: direct-margin `a_N_from_margin`, `a_post`, and
  internally reconstructed `Y_margin` use the exact `1e-10` gate, while the
  historical hidden/readout `a_N` and `Y` retain the already registered GPU
  bridge tolerance of `1e-5`. No factorial cell, threshold, norm match, or
  outcome rule changed.
- Runner, tests, formal dispatch, and implementation/source lock were committed
  and pushed as `2b5ea5a` before formal replay.
- The only networks are the jointly locked fresh v2.4 backbones 2104 and 2105.
  Each uses all 77 participants, all 28 canonical edges, both neural query
  orientations, and the same fixed 20 nonlearned-pair mask as Track B.
- The neural condition is pure `L`-off with intact terminal `P_T` and frozen
  `W_out`. No local trace is constructed or read.
- The posterior field is the same-unit log-odds margin of the frozen exact
  posterior at posterior temperature 0.05 and choice temperature 0.25. It is a
  scientific comparator, not the human posterior or ground-truth neural
  geometry.
- Execution used the NVIDIA GeForce RTX 5090 through
  `python -m fsrl.formal_runtime`, with one PyTorch intra-op and one inter-op
  CPU thread.
- A second complete GPU execution to `/tmp` is byte-identical. Result SHA-256:
  `de1419e1dc6135a8ad92212d04b5c9d6fb1ecc3ca400f7db9ad3e0867e2b61fe`.

All 29 source checks, artifact locks, parameter hashes, endpoint
reproductions, Hodge identities, norm-match identities, participant and
bootstrap factorial identities, and finiteness gates pass in both networks.
All 10,000 bootstrap draws are retained separately within each network; no
participant or network pooling occurs.

## 1. Natural endpoint mismatch is reproduced

For each participant,

```text
m_N = g_N + c_N
m_P = g_P + c_P
S_ab = slope_d sigmoid[y(g_a + c_b)/T]
```

where the first index selects the additive source and the second selects the
residual source. The four frozen cells are:

| Seed | `S_NN` | `S_PN` | `S_NP` | `S_PP` |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 0.06840 | 0.02760 | 0.07821 | 0.01066 |
| 2105 | 0.06942 | 0.02750 | 0.08062 | 0.01066 |

The registered anchor `D=S_NN-S_PP` exactly reproduces Track B and remains
positive:

| Seed | `D` | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | 0.05774 | [0.05117, 0.06441] |
| 2105 | 0.05876 | [0.05312, 0.06451] |

Raw participant endpoint errors are at most `4.60e-17`. This is a sequential
localization on the same backbones, not an independent confirmation.

## 2. Neither natural single-component replacement is sufficient

Posterior-additive replacement with the neural residual fixed produces a large
material reduction:

| Seed | `Delta_A=S_NN-S_PN` | 95% bootstrap | `C_A=S_PN-S_PP` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2104 | 0.04080 | [0.03443, 0.04708] | 0.01694 | [0.01124, 0.02342] |
| 2105 | 0.04192 | [0.03561, 0.04818] | 0.01684 | [0.01111, 0.02318] |

`Delta_A` is materially positive, but `C_A` is also materially positive rather
than equivalent to zero under the frozen `+/-0.005` margin. Posterior-additive
replacement therefore reduces but does not close the discrepancy. Because the
natural replacement changes both additive norm and direction, this result
alone is not a shape-specific claim.

Posterior-residual replacement does not supply the missing closure:

| Seed | `Delta_R=S_NN-S_NP` | 95% bootstrap | `C_R=S_NP-S_PP` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2104 | -0.00980 | [-0.02572, 0.00641] | 0.06754 | [0.05310, 0.08143] |
| 2105 | -0.01119 | [-0.02610, 0.00376] | 0.06996 | [0.05604, 0.08351] |

`Delta_R` is unresolved and `C_R` remains materially positive. Do not call the
point-estimate increase from `S_NN` to `S_NP` a confirmed adverse residual
effect, and do not call residual replacement sufficient.

The preregistered field-source fingerprint is therefore
`mixed_or_unresolved` in each network and jointly. This is not permission to
select the numerically larger additive contrast and relabel it as sufficient.

## 3. Main effects and the fixed-sigmoid interaction

The dependent symmetric summaries are:

| Seed | `A` | 95% bootstrap | `R` | 95% bootstrap | `I` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2104 | 0.05417 | [0.04480, 0.06321] | 0.00357 | [-0.00679, 0.01431] | -0.02674 | [-0.03878, -0.01462] |
| 2105 | 0.05594 | [0.04673, 0.06485] | 0.00282 | [-0.00692, 0.01273] | -0.02804 | [-0.03946, -0.01681] |

Under the frozen mutually exclusive rule, `A` is materially positive, `R` is
unresolved, and `I` is materially negative in both networks. `A` and `R`
partition `D`; `I` is a separately registered contrast of the same four cells.
None constitutes an independent evidence source or confirmation link.

The negative interaction has a precise field-level meaning:

```text
I = (S_NN - S_PN) - (S_NP - S_PP) < 0.
```

The neural-versus-posterior additive-source effect is smaller under `c_N` than
under `c_P`. Equivalently, the residual-source contrast depends on which
additive field enters the fixed sigmoid. The corresponding pre-sigmoid margin
interaction is exactly zero (maximum numerical error `8.88e-16`), so this is a
probability-link field-reassembly interaction. It is not recurrent, circuit,
or biological coupling evidence.

## 4. Norm-matched posterior shape is informative but not sufficient

The sole registered control scales the posterior additive field to the neural
full-28-edge additive norm within participant:

```text
g_P_tilde = g_P ||g_N||_2 / ||g_P||_2
```

and holds `c_N` fixed. It exactly matches both additive norm and complete
pre-sigmoid edge-field energy; it does not match the 20-pair subset norm or
post-sigmoid probability amplitude.

| Seed | `Q_shape=S_NN-S_tildePN` | 95% bootstrap | `C_shape=S_tildePN-S_PP` | 95% bootstrap | `Q_amp=S_tildePN-S_PN` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2104 | 0.03368 | [0.02796, 0.03932] | 0.02406 | [0.01888, 0.02998] | 0.00712 | [0.00553, 0.00871] |
| 2105 | 0.03474 | [0.02903, 0.04031] | 0.02403 | [0.01887, 0.02978] | 0.00718 | [0.00575, 0.00867] |

All three contrasts are materially positive in both networks. Therefore:

- At the neural additive norm and fixed neural residual, posterior additive
  shape/global allocation materially reduces the neural slope.
- It does not close the posterior anchor because `C_shape` is materially
  positive rather than equivalent.
- Along this one fixed posterior-shape path, moving from the participant-wise
  neural-norm target to each posterior's natural norm reduces the slope on
  average. The scale direction is not uniform across participants, so this is
  incompatible with a simple monotone global-overgain explanation but is not
  a general amplitude intervention result.

The identity `Delta_A=Q_shape+Q_amp` holds participant-wise and in every
bootstrap draw. A positive `Q_shape` is a shape/allocation effect at one frozen
operating point; it does not show that additive shape alone is sufficient.

## Revised theory and claim boundary

The full evidence now supports:

```text
P_T-dependent neural global policy
    -> additive field carrying the within-neural distance slope
    -> neural/posterior additive allocation difference
    x residual context through the fixed sigmoid
    -> comparator-relative pairwise probability-slope mismatch
```

This replaces neither the replicated `P_T/L_T` causal decomposition nor the
v2.4 differential evidence-access result. It also does not reinstate scalar
overgain: posterior additive norm is larger at the registered cohort level,
matched posterior shape lowers the slope at neural norm, and moving to the
natural posterior norm lowers it again on average along the registered path.

The exact posterior remains a frozen comparator. The result licenses only the
statement that the neural field allocates confidence differently relative to
that comparator. It does not establish that the neural field is incorrect,
that the comparator is the human posterior, or that the hybrid fields are
realizable recurrent states.

## Stop/go and next decisive stage

Honor the registered `mixed_or_unresolved` outcome:

- Do not add more hybrid cells, norm directions, masks, temperatures, or
  comparators on seeds 2104/2105.
- Do not perform a scalar calibration or select additive shape as a sufficient
  intervention target.
- Preserve the material additive main effect, material matched-norm shape
  effect, material negative probability-link interaction, unresolved residual
  main effect, and both failed closure rules together.

Before any implementable or causal model change, a separate contract should
use one to three new development backbones to test whether this unchanged
three-part fingerprint replicates: material positive additive-source effect,
material positive matched-norm shape reduction without assumed closure, and
material negative fixed-sigmoid interaction. Only a fresh replicated field
fingerprint could justify a later network-internal question about how
`P_T -> g_N` allocates confidence. A comparator-adequacy test would require its
own prospective adequacy criterion and cannot be appended to this result.
