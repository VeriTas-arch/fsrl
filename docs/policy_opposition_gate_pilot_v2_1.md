# Policy-relative curvature-opposition gate v2.1

## Conclusion

The registered one-backbone test is a valid negative result. Selective scalar
attenuation based on the sign of the local second-order policy projection is
not sufficient to rescue relation-specific fidelity on the frozen seed-2101
backbone.

All source, artifact, competence, causal-necessity, global-reassembly,
query-binding, nonlearned-inference, and terminal-projection gates pass. The
primary local rescue, H>A rescue, and control-specificity rules fail. Under the
prospective stop rule, this closes the scalar amplitude-gating family. Seeds
2102/2103, end-to-end co-adaptation, post-hoc risk rescaling, and additional
risk metrics are not authorized for this family.

The next candidate must be separately registered on the same frozen backbone
and explicitly preserve a first-order `J_b u` branch through residualized or
near-linear policy expression. This result does not motivate changing `tanh`.

## Frozen provenance

- Protocol: `benchmarks/policy_opposition_gate_pilot_v2_1.json`, committed as
  `b9902c5` before implementation or execution.
- Initial implementation and source lock:
  `benchmarks/policy_opposition_gate_pilot_v2_1.lock.json`, committed as
  `6c7cfaa` before gate adaptation.
- Reporting-only superseding lock:
  `benchmarks/policy_opposition_gate_pilot_v2_1.lock_v2.json`, committed as
  `5b541a2` before the final evaluation. It changes only the descriptive
  per-relation opposed-fraction reduction from an orientation-averaged sign to
  counting the two oriented online states separately. Gate adaptation, logits,
  estimands, bootstrap rules, and PASS/FAIL are unchanged.
- Final execution-integrity lock:
  `benchmarks/policy_opposition_gate_pilot_v2_1.lock_v3.json`, committed as
  `53a096f` before the formal result was regenerated. It makes evaluation
  actively reject a mismatch to the already frozen gate-artifact hash and
  changes no computation.
- Frozen backbone:
  `output/curvature-gate-pilot-v2/seed-2101/backbone/net.dat`, SHA-256
  `3671582a3d0f638f9b383e9bea20b966824462217566edf8b15f8152d2a2c78d`.
- Frozen v2.1 gate artifact:
  `output/policy-opposition-gate-pilot-v2-1/seed-2101/gate/gate.json`, SHA-256
  `a1b51eb565d37e64d086c4bde348b90adfa07c151a677978ecf36aa0d481ddc0`.
- Final result: `results/policy_opposition_gate_pilot_v2_1.json`, SHA-256
  `8b0cc636f344b11c440b7a38b34f8775689b16f3bd32c6db6708cb8477764011`.
- The final evaluation was repeated independently to `/tmp`; the two result
  files were byte-identical.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, one
  PyTorch intra-op and one inter-op CPU thread. Gate adaptation used
  `torch.compile(..., fullgraph=True, mode="default")`.

The first evaluation under the initial implementation lock was also repeated
byte-for-byte. Comparing it with the lock-v2 evaluation after removing source
lock metadata and the repaired opposed-fraction display showed that every
other result field was identical.

## Registered controller

For the first fast-weight-sensitive response transition,

```text
b = i2h(x_response) + W h0
u = (alpha * P_T) h0
J_b u = (1 - tanh(b)^2) * u
K2(b,u) = -tanh(b) * (1 - tanh(b)^2) * u^2
W_margin = W_out[class_1] - W_out[class_0]
j = W_margin^T J_b u
k = W_margin^T K2(b,u)
s^2 = ||W_margin||^2 ||J_b u||^2
d = j^2 + 0.01 s^2 + 1e-8
r_opp = relu(-j k) / d
r_support = relu(+j k) / d
gamma(r) = 1 / (1 + beta r)
```

Only `raw_beta`, with `beta=softplus(raw_beta)`, was adapted. The backbone was
byte-identical before and after adaptation. The sign-reversed control used the
same beta and denominator and added no parameter.

Generic-only adaptation produced:

```text
beta                         1.0515683889
generic calibration gamma   0.9981706973
generic mean r_opp           0.0018056191
```

No Liu subject, relation, label, correctness value, LOO state, crossing, or
human target entered adaptation or calibration.

## Integrity and competence

All registered integrity errors are exactly zero:

| Check | Maximum error |
| --- | ---: |
| Replayed generic calibration | 0 |
| `gamma=1` versus frozen-v1 logits | 0 |
| Signed-risk identity | 0 |
| Shuffled-gamma multiset | 0 |
| Already omitted relation influence | 0 |

Both original-v1 and opposition-gated qualification suites pass, including
fast-weight interventions and order invariance. The inherited query-binding
estimator emits NumPy empty-slice warnings for zero-norm, already omitted
state rows; these rows are excluded by the prospectively fixed retained mask.
The registered retained-state summaries are finite and unchanged from v1.

## Primary local result

Participant-bootstrap intervals are the frozen 95% intervals.

| Condition | Retained direct correctness | H>A direct correctness | Other seven |
| --- | ---: | ---: | ---: |
| Original v1 | 0.04909 [0.04025, 0.05796] | -0.26379 [-0.31653, -0.21349] | 0.09310 [0.08451, 0.10191] |
| Opposition gate | 0.04930 [0.04049, 0.05814] | -0.26344 [-0.31583, -0.21316] | 0.09330 [0.08481, 0.10201] |
| Matched global | 0.04906 [0.04025, 0.05791] | -0.26231 [-0.31488, -0.21217] | 0.09286 [0.08430, 0.10164] |
| Shuffled opposition | 0.04903 [0.04026, 0.05779] | -0.26116 [-0.31406, -0.21083] | 0.09262 [0.08395, 0.10145] |
| Sign-reversed support | 0.02668 [0.01865, 0.03494] | -0.22582 [-0.27185, -0.18048] | 0.06170 [0.05409, 0.06958] |

The registered paired contrasts are:

| Contrast | Mean | 95% bootstrap interval | Rule |
| --- | ---: | ---: | --- |
| Opposition - original, aggregate | +0.000218 | [-0.000041, +0.000475] | FAIL |
| Opposition - original, H>A | +0.000348 | [-0.000727, +0.001257] | FAIL |
| Opposition - original, other seven | +0.000194 | [-0.000100, +0.000472] | PASS preservation |
| Opposition - matched global | +0.000243 | [-0.000013, +0.000497] | FAIL specificity |
| Opposition - shuffled | +0.000277 | [-0.000609, +0.001225] | FAIL specificity |
| Opposition - sign-reversed | +0.022625 | [+0.018045, +0.027116] | PASS this control only |

Thus the opposition statistic makes only a very small, non-robust change. H>A
remains robustly wrong-sign. The opposition gate does not beat matched-global
or shuffled controls under the prospectively strict bootstrap rule.

The sign-reversed control is informative but not a rescue. It attenuates the
much more common `j*k>0` states, substantially harms aggregate and other-seven
direct correctness, and still leaves H>A robustly negative. Its H>A movement
toward zero shows that attenuation can move this relation, while the large
loss elsewhere shows that this is nonspecific gain manipulation.

## Why the signed statistic failed

Across all intact Liu query states, only 7.42% have `j*k<0`; 92.58% have
`j*k>0`. On the two oriented direct-query states of retained relations:

| Relation | Opposed fraction | Mean opposition gamma |
| --- | ---: | ---: |
| F>A | 2.68% | 0.99949 |
| C>B | 9.68% | 0.99748 |
| E>B | 3.57% | 0.99935 |
| G>C | 2.46% | 0.99943 |
| F>D | 5.08% | 0.99938 |
| G>D | 0.85% | 0.99993 |
| H>E | 2.73% | 0.99896 |
| H>A | 2.50% | 0.99944 |

The statistic therefore does not identify the already-localized H>A failure:
it assigns almost no attenuation to H>A or F>A. This is not a reason to refit
`tau`, cap the denominator, or increase beta after seeing Liu. It is evidence
that the sign of the local quadratic output projection is not the missing
online state variable. The failure can involve higher-order finite-amplitude
terms, relation-LOO residual interactions, or expression geometry not captured
by a scalar pointwise `j*k` test.

The diagnostic-only crossing association is weak and non-significant:

```text
Spearman(gamma, crossing midpoint) = 0.0518
p = 0.6279
crossing cases = 90
```

As registered, this correlation does not enter PASS/FAIL.

## Preserved positive backbone

The negative result does not erase the supported global mechanism:

- remote absolute reassembly is 0.47217 [0.45085, 0.49421] under opposition,
  versus 0.47218 under v1;
- third-party relational fraction is 0.21389 [0.20381, 0.22427], versus
  0.21389 under v1;
- matched-minus-shared query binding is 0.26216, and matched-minus-disjoint is
  0.30827; both are positive and exactly equal to the pre-gate v1 estimand;
- terminal expected-rank-over-MAP alignment remains positive at 0.02995 with
  bootstrap lower bound 0.01401;
- the opposition-gated causal suite passes fast-weight necessity and all other
  qualification rules;
- nonlearned accuracy is preserved.

Behavior is correspondingly almost unchanged:

| Condition | Learned | Nonlearned | Overall | Distance slope |
| --- | ---: | ---: | ---: | ---: |
| Original v1 | 0.91006 | 0.81526 | 0.84235 | 0.04843 |
| Opposition gate | 0.91023 | 0.81506 | 0.84225 | 0.04849 |
| Matched global | 0.90958 | 0.81532 | 0.84225 | 0.04834 |
| Shuffled opposition | 0.91006 | 0.81513 | 0.84225 | 0.04847 |
| Sign-reversed support | 0.90795 | 0.81409 | 0.84091 | 0.04873 |

## Mechanistic decision and next test

The supported chain remains:

```text
subject-specific effective evidence
  -> fast-weight global expected-rank assembly
  -> query-bound first-order policy value
```

The rejected link is:

```text
sign of local W_margin-projected K2 relative to J_b u
  -> sufficient selective scalar attenuation
  -> human-like retained-relation fidelity
```

The next decisive family is not another gate statistic. It should use the same
frozen seed-2101 backbone and the same readout, and prospectively test whether
an explicit first-order residual or near-linear expression path can preserve
`J_b u` while retaining the exact nonlinear/global branch. The exact mixing
equation, whether the residual acts in hidden or policy space, matched
controls, and outcome-contingent interpretations must be frozen before
implementation. If that route fails, the missing local-fidelity mechanism is
not already recoverable by a low-capacity response-expression correction and
a separately persistent local component becomes the next justified family.

Do not run seeds 2102/2103, re-adapt beta, tune `tau`, add a risk cap, change
activation, or begin end-to-end co-adaptation for the closed amplitude-gating
family.

## Reproduction

```bash
direnv exec . python -m fsrl.policy_opposition_gate_pilot adapt-gate
direnv exec . python -m fsrl.policy_opposition_gate_pilot evaluate
direnv exec . python -m pytest \
  tests/test_policy_opposition_gate.py \
  tests/test_policy_opposition_gate_pilot.py -q
```

The gate-adaptation command is intentionally non-repeatable in the same output
directory. The frozen gate artifact already exists; reproduce evaluation only
unless intentionally using a separate output root under a newly registered
contract.
