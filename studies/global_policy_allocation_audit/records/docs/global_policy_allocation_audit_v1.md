# Global-policy allocation audit v1

## Conclusion

The prospectively frozen read-only audit returns
`policy_effective_allocation_localized` on the already locked seed-2106 and
seed-2107 backbones. The registered localization scope is
`structural_only`:

- distance-residualized pair identity is stable in both the equal-energy
  additive mismatch and the exact probability-slope bridge;
- symbolic distance is likewise stable at both levels;
- posterior uncertainty and effective evidence coverage each have stable
  coefficients, but their field and bridge directions are opposite, so
  neither is a policy-effective state-dependent axis under the frozen rule.

The only registered next step is therefore
`prospective_comparator_adequacy`. This result does not authorize a
`P_T`-generation analysis, neural decomposition, parameter audit, or model
intervention.

The positive result is narrower than a population or causal claim. It shows
that, relative to the frozen exact-posterior comparator and at equal full-field
additive norm, the neural allocation difference has reproducible pair and
distance structure, and that structure occurs where the already replicated
`Q_shape` probability-slope difference is carried. The audit is sequential on
two previously evaluated development networks, uses no family-wise error or
minimum-effect criterion, and makes no network-population inference.

## Frozen protocol, execution repair, and provenance

- The allocation contract was committed and pushed as `46bfa6a` before the
  runner was implemented or either checkpoint was loaded for this audit.
- The runner, tests, formal-runtime dispatch, and initial implementation/source
  lock were committed and pushed as `9d9bd2a` before Attempt 1.
- Attempt 1 completed both frozen seed replays but strict final JSON encoding
  rejected one `numpy.bool_` prerequisite check. The attempt is
  noninterpretable; no scientific outcome was inspected or used to change an
  estimand, threshold, axis, or decision. Its failure record is preserved at
  `results/global_policy_allocation_audit_v1_attempt1_noninterpretable.json`.
  The invalid partial output has SHA-256
  `6d42baa1b5a987e01e75abcddd26358c0c3f7e16dc65507a7a60234688653d73`.
- The superseding Repair 1 lock and regression were committed and pushed as
  `d216bdb`. The only semantic change was conversion of the same truth-valued
  completeness predicate to a built-in `bool`; the writer now serializes the
  complete payload before exclusive file creation. The repair changed no
  scientific calculation.
- The formal git gate explicitly verified the protocol, Repair 1 lock, initial
  lock, and Attempt 1 record as tracked and unchanged at clean
  `dev == origin/dev == d216bdb` before checkpoint validation.
- Both mandatory networks use all 77 participants, all 28 canonical edges,
  the fixed 20 nonlearned pairs, and separate 10,000-draw participant
  bootstraps. Participants and networks are never pooled.
- Execution used the NVIDIA GeForce RTX 5090 through
  `python -m fsrl.formal_runtime`, with one PyTorch intra-op and one inter-op
  CPU thread. No training, adaptation, compilation, checkpoint write, or
  parameter update occurred.
- A complete GPU replay to `/tmp` is byte-identical to the independently
  recomputed canonical result. Result SHA-256:
  `82147bad41b0cbdbb812217f383555f166e5f96492a7b00de752bec14ef867f6`.

All 31 source/provenance checks, all 10 inherited artifact checks, the frozen
fingerprint prerequisite, both seed integrity batteries, and the cross-network
integrity gate pass. Both backbone tensor maps are unchanged. The largest
registered numerical identity error is `1.71e-13`, below the frozen `1e-10`
tolerance. The recomputed `Q_shape` vectors match the prior fingerprint result
exactly.

## Exact allocation-to-policy bridge

For participant `s` and canonical edge `e`, the audit uses

```text
g_P_tilde = g_P ||g_N||_2 / ||g_P||_2
Delta g    = g_N - g_P_tilde
delta      = y Delta g

Delta p = sigmoid[y(g_N + c_N)/T]
        - sigmoid[y(g_P_tilde + c_N)/T]

q_e = (d_e - 2.8) Delta p_e / 57.2
```

and verifies separately for every participant and every bootstrap draw that

```text
sum_e q_e = Q_shape.
```

The maximum participant identity error is `7.63e-17`; the maximum bootstrap
identity error is `3.47e-17`. Thus `Delta g` identifies where the equal-energy
pre-sigmoid allocation differs, while `q` identifies where those differences
actually contribute to the replicated probability-slope reduction at the
frozen sigmoid operating point.

The cohort mean reconstructed `Q_shape` is `0.03454` for seed 2106 and
`0.03325` for seed 2107, exactly reproducing the prior fingerprint.

## 1. Pair-specific allocation survives linear-distance removal

The registered pair fingerprint first removes only an intercept and linear
symbolic-distance term from the 20-dimensional cohort-mean vectors. It does
not remove categorical or nonlinear distance structure and does not test 20
pairwise hypotheses.

| Registered cross-network vector | Correlation | 95% independent participant-product bootstrap | Status |
| --- | ---: | ---: | --- |
| `r_delta` | 0.99477 | [0.40845, 0.97464] | resolved positive |
| `r_q` | 0.95800 | [0.33947, 0.93268] | resolved positive |

Both field and exact-bridge requirements pass, so pair identity is
`policy_effective` under the frozen developmental rule. There are no
zero-norm or nonfinite draws.

The raw, non-primary vector correlations are also positive: `mu_delta`
`0.99436`, `mu_Delta_p` `0.98185`, and `mu_q` `0.99324`. These diagnostics
are reported only as consistency checks; they do not replace the registered
distance-residualized metrics.

## 2. The exact slope contribution has a stable distance profile

The correct-signed pre-sigmoid allocation difference has the same negative
linear distance slope in both networks:

| Seed | `beta_delta_distance` | 95% participant bootstrap | Status |
| --- | ---: | ---: | --- |
| 2106 | -0.05472 | [-0.07229, -0.03830] | resolved negative |
| 2107 | -0.05395 | [-0.07359, -0.03600] | resolved negative |

The six-level exact-bridge vectors are almost identical across networks:

```text
corr(Q_by_distance_2106, Q_by_distance_2107) = 0.99986
95% bootstrap = [0.99267, 0.99979].
```

Every registered distance-level contribution is retained:

| Distance | Seed 2106 `q` contribution [95%] | Seed 2107 `q` contribution [95%] |
| ---: | ---: | ---: |
| 1 | 0.03356 [0.02925, 0.03781] | 0.03200 [0.02753, 0.03641] |
| 2 | 0.00903 [0.00729, 0.01089] | 0.00906 [0.00732, 0.01084] |
| 3 | -0.00079 [-0.00111, -0.00050] | -0.00083 [-0.00114, -0.00054] |
| 4 | -0.00435 [-0.00616, -0.00273] | -0.00437 [-0.00599, -0.00288] |
| 5 | -0.00140 [-0.00290, -0.00021] | -0.00138 [-0.00292, -0.00023] |
| 6 | -0.00150 [-0.00339, -0.00004] | -0.00123 [-0.00297, -0.00015] |

The positive total `Q_shape` is therefore concentrated in the distance-1 and
distance-2 terms and partially offset by distances 3 through 6. This is an
exact slope ledger, not a post-hoc near-versus-far hypothesis test. The
negative pre-sigmoid distance coefficient and the positive total probability-
slope discrepancy must not be collapsed into a sign-preserving scalar story;
the fixed residual and sigmoid operating context remain consequential.

## 3. Uncertainty and coverage are stable but directionally discordant

`U` and `C` were standardized once over the full cohort and entered together
in the frozen pair-fixed-effect model. Their correlation is `-0.89086`, so the
coefficients are conditional, pair-invariant associations rather than total or
causal effects.

| Axis | Seed | Field coefficient | 95% bootstrap | Bridge coefficient | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Posterior uncertainty `U` | 2106 | +0.07073 | [+0.00052, +0.14413] | -0.001234 | [-0.001785, -0.000664] |
| Posterior uncertainty `U` | 2107 | +0.08719 | [+0.02130, +0.15907] | -0.001373 | [-0.001977, -0.000765] |
| Effective coverage `C` | 2106 | +0.12588 | [+0.05733, +0.19953] | -0.000726 | [-0.001244, -0.000170] |
| Effective coverage `C` | 2107 | +0.13289 | [+0.06873, +0.20507] | -0.000761 | [-0.001386, -0.000111] |

All eight coefficients are resolved in the displayed direction. The result is
not a null U/C finding. It is a reproducible sign reversal: greater `U` or `C`
is associated with a more positive correct-signed pre-sigmoid allocation
difference after conditioning on the other predictor, but with a smaller
`Q_shape` contribution at the fixed policy link. Because the frozen rule
requires field and bridge directions to agree, neither state axis is
`policy_effective`.

The exact balanced-design identity gives `20 beta_q` as the corresponding
`Q_shape` coefficient. These are negative for both `U` (-0.02468 and
-0.02747) and `C` (-0.01452 and -0.01523). Posterior uncertainty remains
comparator-derived; this cannot be described as incorrect neural uncertainty
coding. Effective coverage is not randomized and cannot be described as the
cause of the allocation.

## Revised theory and claim boundary

The four-network field fingerprint remains supported:

```text
P_T-dependent g_N carries the neural distance slope
and differs from the frozen posterior additive allocation at equal energy.
```

This audit adds the narrower localization:

```text
equal-energy Delta g
  -> stable residual pair structure
  -> stable symbolic-distance structure
  -> stable exact probability-slope contribution at the frozen policy link.
```

It does not support the stronger state-dependent route

```text
U or C -> concordant Delta g -> Q_shape,
```

because both registered state axes reverse sign between field and bridge. The
negative finding revises the route without erasing the structural positive:
the comparator-relative discrepancy is reproducibly organized across pair and
distance, but the two prospectively allowed subject-state variables do not
provide a concordant online explanation for how `P_T` generates it.

The exact posterior is still one frozen comparator, not the human posterior,
behavioral ground truth, or uniquely correct neural geometry. The result is a
sequential two-network developmental localization, not an independent
replication, population-prevalence estimate, family-wise-error-controlled
discovery, component-sufficiency result, or causal intervention.

## Stop/go and next decisive stage

The frozen decision tree requires a comparator-adequacy question next. Close
further neural decomposition and intervention on these backbones: do not inspect
`P_T` parameters, add state interactions, introduce a fifth allocation axis,
change the norm match, refit either temperature, or target the stable pair/
distance pattern inside the network.

The next protocol must prospectively define an external adequacy criterion for
the exact-posterior comparator before introducing any alternative comparator.
It should ask whether the structural pair/distance fingerprint remains when
the reference field is judged against held-out human behavior, while keeping
network fields, participants, masks, temperatures, and the current comparator
result frozen. If the exact posterior is externally adequate yet the same
structural discrepancy remains, a new neural mechanism question may be
registered. If it is not externally adequate on the same pair/distance
structure, the current mismatch should be attributed to comparator limitation
rather than targeted inside `P_T`. No comparator search, temperature tuning,
or best-fitting alternative is authorized by this result.
