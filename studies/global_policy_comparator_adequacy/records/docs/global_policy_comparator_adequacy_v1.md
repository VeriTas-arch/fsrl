# Global-policy comparator adequacy v1

## Conclusion

The prospectively frozen external-reference audit returns
`comparator_externally_inadequate`. Both necessary primary criteria fail:

- the current exact-posterior comparator has a fixed 20-pair symbolic-distance
  slope of `0.01066`, far below the human participant-bootstrap 95% interval
  `[0.04731, 0.06069]` around `S_H=0.05398`;
- the distance-residualized human pair field is highly reliable, but the
  posterior field does not capture it. `r_HH=0.96976`, the Spearman-Brown
  corrected reliability is `0.98465`, while `r_PH=-0.04214` and the corrected
  ceiling ratio is `eta_pair=-0.04247` with a 90% interval
  `[-0.32253, 0.36888]`, well below the frozen `0.80` adequacy floor.

The stable neural-versus-posterior `A/Q_shape/I` and pair/distance allocation
fingerprints remain valid facts relative to this exact comparator. They are not
valid targets for a `P_T`, `W_out`, temperature, or other neural intervention,
because the reference itself fails both prospectively fixed human axes. No
neural field or alternative comparator entered this audit.

The next scientific stage is prospective comparator-theory reassessment, not
network modification or comparator search against the neural result.

## Frozen protocol and provenance

- The scientific contract was committed and pushed as `3023072` before the new
  20-pair human slope, split-half reliability, posterior-human pair
  correlation, corrected ceiling ratio, or distance profile was calculated.
- The first static edge-contract test found that the protocol had transcribed
  the inherited distance-level counts as `[4,5,4,3,2,2]` instead of the counts
  deterministically implied by its already frozen 20 pair labels,
  `[6,5,2,3,2,2]`. No human or posterior adequacy outcome had been loaded.
  Repair 1 preserves the original specification and changes only that
  descriptive completeness vector; it was committed and pushed as `485c542`.
- The runner, synthetic tests, formal-runtime dispatch, and exact source lock
  were committed and pushed as `44004aa` before formal evaluation.
- The audit used all 40 preregistered and 37 retained replication participants.
  Every participant contributes all 280 trials, ten observations per pair, and
  five observations to each fixed odd/even block half. Correct rankers remain
  in the choice-field cohort.
- Human and virtual subjects are not matched. Human participants alone are
  resampled; the current 77-subject exact-posterior cohort remains fixed.
- Execution used `python -m fsrl.formal_runtime` on an NVIDIA GeForce RTX 5090,
  with one PyTorch intra-op and one inter-op CPU thread. There was no training,
  checkpoint loading, neural replay, parameter update, temperature fit, or
  alternative comparator.

All 23 source checks, 12 allocation-prerequisite checks, raw-trial identities,
human-benchmark reproduction checks, posterior anchors, reliability gates, and
10,000 bootstrap draws pass. The reconstructed posterior subject slopes match
the independently frozen seed-2106 and seed-2107 `S_PP` arrays to
`7.63e-17`; those two anchors are exactly identical.

The first `/tmp` replay request was rejected before calculation because the new
canonical result made the worktree non-clean. The canonical file was moved
temporarily to `/tmp`, the same `44004aa` clean-worktree gate was re-established,
the replay completed, and the original file was moved back unchanged. The two
complete result files are byte-identical. SHA-256:

```text
72e7e0beb0715fa4dcff6f5f74e7594cc8b8d51999db4f453ec8696e9ed3585f
```

## External fields and primary estimands

For the same fixed 20 nonlearned pairs used throughout Track B, the human field
is

```text
h[e] = mean participant accuracy for pair e across ten query blocks.
```

The posterior field is

```text
p_P[e] = mean P_post(source-correct orientation | D_s)
         over the fixed 77 virtual evidence states.
```

No participant-wise human-model mapping is assumed. With the inherited fixed
design,

```text
d_bar = 2.8
V_d   = 57.2
w[e]  = (d[e] - 2.8) / 57.2

S_H = sum_e w[e] h[e]
S_P = sum_e w[e] p_P[e].
```

The human slope point estimate exactly equals the mean participant slope to
`6.94e-18`. The independently recomputed raw-array audit reproduced all primary
point estimates and bootstrap limits to floating-point error.

## 1. Distance adequacy fails below the human interval

| Estimand | Point | Registered interval/rule | Status |
| --- | ---: | ---: | --- |
| Human `S_H` | 0.05398 | 95% [0.04731, 0.06069] | external interval |
| Posterior `S_P` | 0.01066 | must lie inside human 95% interval | `inadequate_below` |

The current comparator is not merely somewhat flatter than the human field;
its fixed slope lies well outside the external interval. The direction was
historically suggested by earlier all-pair summaries, so this is a formal
prospective adequacy decision rather than a blind discovery.

The registered secondary ledger localizes the discrepancy without creating a
new near/far primary:

| Distance | Pairs | Human field [95%] | Posterior field | Human minus posterior [95%] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.70844 [0.67706, 0.73896] | 0.92812 | -0.21968 [-0.25106, -0.18916] |
| 2 | 5 | 0.80545 [0.77221, 0.83818] | 0.95074 | -0.14528 [-0.17853, -0.11256] |
| 3 | 2 | 0.86039 [0.81494, 0.90195] | 0.96601 | -0.10562 [-0.15107, -0.06406] |
| 4 | 3 | 0.90476 [0.87359, 0.93333] | 0.97196 | -0.06720 [-0.09837, -0.03863] |
| 5 | 2 | 0.96688 [0.94221, 0.98701] | 0.97863 | -0.01175 [-0.03643, 0.00838] |
| 6 | 2 | 0.95390 [0.92338, 0.98052] | 0.97578 | -0.02189 [-0.05241, 0.00474] |

Thus the posterior field is already close to ceiling for distance-1 and
distance-2 pairs, where the human field retains the largest room for graded
confidence. Distances 5 and 6 do not show a resolved human-posterior difference
under the registered secondary intervals. This directly explains why the
previous exact `q` ledger concentrated comparator-relative slope effects at
distances 1 and 2, but it does not retrospectively turn those levels into a
separate primary family.

The source cohorts point in the same descriptive direction: the preregistered
human slope is `0.05634` and the replication slope is `0.05143`. These were not
separate primary tests and do not replace the combined-cohort inference.

## 2. Pair adequacy fails despite a high human noise ceiling

Only an intercept and linear symbolic distance are removed from each 20-vector:

```text
r(x) = [I - X pinv(X)] x,  X = [1, d].
```

Odd blocks `1,3,5,7,9` and even blocks `2,4,6,8,10` provide the pre-registered
human halves. The resulting primary quantities are:

| Estimand | Point | 90% participant bootstrap | 95% participant bootstrap |
| --- | ---: | ---: | ---: |
| Human split-half `r_HH` | 0.96976 | [0.91001, 0.98284] | [0.89499, 0.98535] |
| Corrected reliability `rho_H` | 0.98465 | [0.95289, 0.99134] | [0.94458, 0.99262] |
| Posterior-human `r_PH` | -0.04214 | [-0.31962, 0.36317] | [-0.35495, 0.44883] |
| Corrected ratio `eta_pair` | -0.04247 | [-0.32253, 0.36888] | [-0.35721, 0.45773] |

The human residual pair structure is therefore highly reproducible, while the
posterior-human correlation is unresolved around zero. The pair criterion
fails because the `eta_pair` 90% lower bound is not at least `0.80`. This does
not establish a robust negative posterior-human correlation; it establishes
failure to capture a reliable human pair fingerprint under the frozen adequacy
rule.

All 20 human, posterior, and distance-residualized pair vectors are retained in
the canonical result. No individual pair significance tests, pair selection,
or neural-informed comparator optimization were performed.

## Revised theory and stop/go

The supported Track-B chain is now:

```text
P_T -> g_N carries the neural global distance slope

g_N versus this exact posterior
  -> stable A/Q_shape/I field fingerprint across four networks
  -> stable pair/distance allocation at the exact policy bridge

this exact posterior versus human choices
  -> distance inadequacy
  -> distance-residualized pair inadequacy.
```

The new negative result changes the route rather than erasing the earlier
positive results. The comparator-relative fingerprints are real and
reproducible, but at least part of their apparent mechanistic discrepancy is a
limitation of the reference used to define departure. In particular, the
current posterior is much too confident on short-distance nonlearned pairs and
does not recover the reliable human residual pair field.

Close neural intervention under this comparator. Do not inspect or modify
`P_T`, `W_out`, temperature, evidence admission, norm matching, or the v2.4
local mechanism in response to this result. Do not search posterior variants
for the one that minimizes neural `Delta g` or `q`; that would make comparator
selection circular.

A future comparator must first have independent theoretical motivation and a
prospectively frozen human-only adequacy contract. Only after it passes that
external contract may it be used in a new neural-mechanism question. The
present audit neither selects such a comparator nor authorizes tuning the
current one.
