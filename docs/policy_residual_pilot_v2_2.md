# First-order policy-residual pilot v2.2

## Conclusion

The registered one-backbone policy-residual test is a valid mixed negative.
The natural first-order residual has a real, state-query-matched causal effect
on the two hardest retained relations, but one generic-trained scalar mixing
coefficient is not sufficient to restore local fidelity without degrading the
relations that v1 already expresses correctly.

The candidate moves H>A toward the correct direction by `+0.03596` with a
strictly positive participant-bootstrap interval, and moves F>A by `+0.00779`.
Neither a balanced matched-magnitude null nor a within-subject shuffled
residual reproduces the H>A improvement. These are positive mechanistic
results: the signed first-order residual and its state-query match contain
relation-specific causal information.

Nevertheless, H>A remains robustly wrong-sign, the aggregate direct-
correctness increase is not significant, the other seven relations decline on
average, and the candidate does not beat either control on the registered
aggregate specificity estimand. Under the prospective stop rule, the result
closes the family in which the existing first-order signal is sufficient when
recovered by a low-capacity response-expression correction.

Do not run seeds 2102/2103, select a larger eta from Liu, add relation-specific
residual scalars, change `tanh`, or begin end-to-end co-adaptation to rescue
this candidate. The next justified mechanism family is a separately
persistent local component that preserves direct experience alongside the
confirmed global fast-weight assembly channel.

## Frozen provenance

- Protocol: `benchmarks/policy_residual_pilot_v2_2.json`, committed as
  `f88732c` before implementation or execution.
- Implementation and source lock:
  `benchmarks/policy_residual_pilot_v2_2.lock.json`, committed as `c961499`
  before eta adaptation.
- Post-adaptation, pre-Liu artifact lock:
  `benchmarks/policy_residual_pilot_v2_2.artifact_lock.json`, committed as
  `24bdd9f` before any Liu evaluation.
- Frozen backbone:
  `output/curvature-gate-pilot-v2/seed-2101/backbone/net.dat`, SHA-256
  `3671582a3d0f638f9b383e9bea20b966824462217566edf8b15f8152d2a2c78d`.
- Frozen eta artifact:
  `output/policy-residual-pilot-v2-2/seed-2101/residual/eta.json`, SHA-256
  `20a64e540780083213d850d69f5e9f52daba0dd1a761f1fc2afa961037ce0b3e`.
- Final result: `results/policy_residual_pilot_v2_2.json`, SHA-256
  `74d8150e2e539907d72a54a01930618b81b90303840a6e45b4c1f9e8f495b628`.
- A complete independent evaluation to `/tmp` was byte-identical to the
  final result.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, one
  PyTorch intra-op and one inter-op CPU thread. Eta adaptation used
  `torch.compile(..., fullgraph=True, mode="default")`.

## Registered intervention

The intervention acts only on the two choice logits. The exact frozen-v1
hidden activation, value, DA, eligibility, fast weights, support computation,
and readout parameters are unchanged.

```text
b = i2h(x_response) + W h0
u = (alpha * P_T) h0
delta_h_exact = tanh(b + u) - tanh(b)
delta_h_lin = J_b u = (1 - tanh(b)^2) * u
r_h = delta_h_lin - delta_h_exact
W_margin = W_out[class_1] - W_out[class_0]
c_res = W_margin^T r_h
m_eta = m_v1 + eta c_res
```

The implementation preserves the common-mode logit exactly:

```text
output_eta[class_0] = output_v1[class_0] - eta c_res / 2
output_eta[class_1] = output_v1[class_1] + eta c_res / 2
```

Thus eta=0 is exact v1, eta=1 restores the current-operating-point first-order
fast-weight policy increment, and intermediate eta mixes only the response
expression. Hidden-space and policy-space mixing with the same eta are
algebraically equivalent at this linear readout; only the minimal policy form
was tested.

`eta=sigmoid(raw_eta)` was the only trainable parameter. Generic-only
adaptation moved eta from 0.5 to:

```text
eta = 0.0939782858
```

The backbone was byte-identical before and after adaptation. No Liu subject,
relation, label, correctness value, LOO state, human target, or control entered
adaptation.

## Controls

Four conditions were frozen:

1. `original_v1`: eta=0.
2. `policy_residual`: natural online `c_res` with generic-trained eta.
3. `matched_magnitude_null`: `sigma |c_res|` with the same eta. Every subject
   has exactly 28 positive and 28 negative seeded signs across 56 oriented
   queries; intact and every LOO replay reuse the sign table.
4. `shuffled_residual`: within-subject permutation of signed `c_res`, using the
   same query permutation for intact and every LOO replay.

The magnitude null preserves each state-query correction magnitude and removes
its policy direction. The shuffled control preserves each replay's full signed
correction distribution and removes state-query assignment. Both reuse the
candidate eta and add no parameter.

## Integrity and competence

Every declared source, implementation, backbone, and eta-artifact hash passes.
Registered numerical checks are:

| Check | Maximum error |
| --- | ---: |
| Eta-zero versus v1 logits | 0 |
| Policy-margin residual identity | `7.14e-7` |
| Matched-null absolute magnitude | 0 |
| Matched-null 28/28 sign balance | 0 |
| Shuffled signed multiset | 0 |
| Unchanged nonconstant shuffled rows | 0 |
| Already omitted relation influence | 0 |

Both original-v1 and policy-residual qualification suites pass, including all
fast-weight interventions and query-order invariance. The inherited
query-binding estimator emits NumPy empty-slice warnings for zero-norm,
already omitted state rows; the prospectively fixed retained mask excludes
those rows, and all registered retained summaries are finite.

## Registered primary result

Intervals are frozen 95% participant bootstraps.

| Condition | Retained direct correctness | H>A direct correctness | Other seven |
| --- | ---: | ---: | ---: |
| Original v1 | 0.04909 [0.04025, 0.05796] | -0.26379 [-0.31653, -0.21349] | 0.09310 [0.08451, 0.10191] |
| Policy residual | 0.04970 [0.04160, 0.05787] | -0.22782 [-0.27648, -0.18144] | 0.08866 [0.08072, 0.09687] |
| Matched magnitude | 0.04985 [0.04116, 0.05852] | -0.26301 [-0.31858, -0.20975] | 0.09379 [0.08517, 0.10267] |
| Shuffled residual | 0.04920 [0.04048, 0.05796] | -0.26520 [-0.31733, -0.21479] | 0.09339 [0.08473, 0.10232] |

The registered paired contrasts are:

| Contrast | Mean | 95% bootstrap interval | Rule |
| --- | ---: | ---: | --- |
| Residual - original, aggregate | +0.000611 | [-0.000231, +0.001444] | FAIL |
| Residual - original, H>A | +0.035964 | [+0.031737, +0.040359] | Direction improves, but H>A remains wrong-sign: FAIL |
| Residual - original, other seven | -0.004442 | [-0.005201, -0.003714] | PASS frozen `-0.01` preservation threshold, negative scientifically |
| Residual - matched magnitude, aggregate | -0.000152 | [-0.001468, +0.001194] | FAIL specificity |
| Residual - shuffled, aggregate | +0.000500 | [-0.000399, +0.001414] | FAIL specificity |

Seven of ten primary flags pass: original competence plus the other-seven
tolerance, nonlearned inference, fast-weight necessity, global reassembly,
query binding, and terminal projection. Local rescue, H>A rescue, and control
specificity fail. The registered outcome is
`valid_local_or_specificity_failure`.

## Positive relation-specific fingerprint

The aggregate failure conceals a structured causal effect:

| Relation | Policy residual - v1 direct correctness |
| --- | ---: |
| F>A | +0.00779 |
| C>B | -0.00535 |
| E>B | -0.00768 |
| G>C | -0.00535 |
| F>D | -0.00654 |
| G>D | -0.00491 |
| H>E | -0.01000 |
| H>A | +0.03596 |

For H>A, candidate-minus-matched-magnitude is `+0.03519` with bootstrap
interval `[+0.02548,+0.04565]`, and candidate-minus-shuffled is `+0.03738`
with interval `[+0.03152,+0.04351]`. These relation-specific comparisons were
not the registered aggregate specificity gate and cannot turn the candidate
into a PASS. They do show that the natural residual's sign and state-query
assignment are causally relevant to H>A.

F>A also improves over v1 by `+0.00779 [0.00482,0.01085]` and over shuffled by
`+0.00856 [0.00436,0.01298]`, but not robustly over the matched-magnitude null.
The six relations already expressed correctly by v1 all decline. A single
shared eta therefore exposes a hard-relation residual while producing a
systematic tradeoff rather than a coherent local-fidelity rescue.

This is stronger than saying the residual has no effect. It supports:

```text
existing J_b u residual
  -> state-query-matched causal movement of H>A/F>A toward correctness
```

It rejects the sufficiency link:

```text
one generic-trained scalar eta
  -> restore H>A to nonnegative fidelity
  -> improve retained relations jointly beyond matched controls
```

## Preserved global backbone and behavior

The candidate leaves the confirmed mechanism intact:

- remote absolute reassembly is 0.47405 [0.45265, 0.49612], versus 0.47218
  under v1;
- third-party relational fraction is 0.21426 [0.20418, 0.22463], versus
  0.21389 under v1;
- matched-minus-shared query binding is 0.26216 and matched-minus-disjoint is
  0.30827, both positive and exactly equal to the pre-expression v1 estimand;
- terminal expected-rank-over-MAP alignment is 0.02998 with bootstrap lower
  bound 0.01400;
- fast-weight necessity and all qualification gates pass;
- nonlearned accuracy is unchanged.

Behavioral summaries remain nearly identical:

| Condition | Learned | Nonlearned | Overall | Distance slope |
| --- | ---: | ---: | ---: | ---: |
| Original v1 | 0.91006 | 0.81526 | 0.84235 | 0.04843 |
| Policy residual | 0.90942 | 0.81526 | 0.84216 | 0.04834 |
| Matched magnitude | 0.91023 | 0.81513 | 0.84230 | 0.04835 |
| Shuffled residual | 0.91006 | 0.81474 | 0.84198 | 0.04843 |

The symbolic-distance slope is not rescued.

## Mechanistic decision and next route

The evidence now supports a sharper theory:

```text
same fast-weight substrate
  -> global expected-rank assembly remains intact
  -> first-order residual contains matched local information for H>A/F>A
  -> shared low-capacity policy mixing cannot express it without harming
     already-correct relations
```

Thus “global versus local” is not explained by a scalar choice between two
expression regimes. A persistent local component is now justified, but it
must be designed to preserve direct experience selectively while leaving the
confirmed global channel and the newly demonstrated H>A/F>A residual
fingerprint intact. The first-order residual may remain a diagnostic or
routing signal; this pilot does not justify discarding it.

Before implementation, the next contract must specify where local content is
stored, how it is written from retained support, how query identity retrieves
it, and selective causal ablations distinguishing it from the global
fast-weight expected-rank channel. Start with the same frozen seed-2101
backbone and one development seed. Do not tune a persistent component merely
to cross the failed Liu scalars.

## Reproduction

```bash
direnv exec . python -m fsrl.policy_residual_pilot evaluate
direnv exec . python -m pytest \
  tests/test_policy_residual.py \
  tests/test_policy_residual_pilot.py -q
```

Eta adaptation is intentionally non-repeatable in the frozen output directory.
The eta artifact is already frozen; reproduce evaluation only unless a new
protocol explicitly authorizes a separate output root.
