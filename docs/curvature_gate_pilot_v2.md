# Curvature-conditioned expression gate pilot v2

## Outcome

The registered one-new-seed gate-only sufficiency test is complete on seed
2101. The candidate is rejected: it preserves the v1 global relational
mechanism but does not rescue local fidelity, does not outperform its matched
attenuation controls, and does not assign smaller amplitudes to cases with
earlier offline crossings.

This is a valid negative result, not a competence failure. It rejects the
unsigned hidden-space norm ratio

```text
r = ||K2(b,u)||2 / (||J_b u||2 + epsilon)
```

as a sufficient online control variable for a frozen v1 backbone. It does not
reject state-conditioned expression control in general, and it does not erase
the preserved global-mechanism results or the descriptive H>A improvement.

## Provenance and execution

- Protocol: `benchmarks/curvature_gate_pilot_v2.json`, committed as `4d79270`
  before implementation or seed-2101 training.
- Initial implementation lock: `benchmarks/curvature_gate_pilot_v2.lock.json`,
  committed with implementation `14895ee`.
- Evaluation repair lock: `benchmarks/curvature_gate_pilot_v2.lock_v2.json`,
  committed as `a56d659` before the complete evaluation retry.
- Backbone checkpoint SHA-256:
  `3671582a3d0f638f9b383e9bea20b966824462217566edf8b15f8152d2a2c78d`.
- Gate artifact SHA-256:
  `5233a73074b35300ed1e7bd7a191ffa696f2aaaf15c34b5ac7402b4980cb2f6e`.
- Result SHA-256:
  `35d251ba919bdea7ab87963c75e1e2d6d55fdc1b4057ed22fa03343c0d401fd6`.
- A second complete evaluation produced a byte-identical result with the same
  SHA-256.

Training and neural replay used the NVIDIA GeForce RTX 5090 with PyTorch
intra-op and inter-op work both bounded to one CPU thread. The v1 backbone used
one contiguous CPU-to-GPU input transfer per trial and
`torch.compile(..., fullgraph=True)` in the default mode. The gate-only
adaptation used the same compiler contract. No NumPy/BLAS thread override was
introduced.

The first evaluation attempt stopped before writing a result because NumPy
boolean indexing transposed the selected-query and subject axes in the
query-binding summary. The repair changed only
`normalized[state_index, :, mask]` to
`normalized[state_index][:, mask]`, added a subject-axis regression test, and
passed 137 repository tests. It did not change a checkpoint, gate parameter,
estimand, threshold, subject, relation, or scientific outcome.

The complete evaluation emits an empty-slice warning for stable-omitted
subject-relation cells whose operator norm is structurally zero. These cells
are subsequently excluded by the prospectively registered retained-evidence
mask. Stable-omitted pair influence remains exactly zero.

## Registered training sequence

The pilot used exactly one new development seed and did not inspect old
localization seeds 1901/1902 or formal seeds 2001--2010.

1. Train a new v1 backbone at seed 2101 for exactly 1000 outer steps on the
   generic sparse-graph distribution, excluding the Liu graph and its rank-axis
   reflection.
2. Freeze every v1 tensor.
3. Train only `raw_beta` for exactly 500 outer steps on an independently seeded
   generic stream.
4. Freeze beta and estimate the matched global scalar on 64 independent
   generic batches before Liu evaluation.
5. Evaluate original v1, conditioned gate, matched global scalar, and
   within-subject shuffled gate on the same 77-subject Liu cohort.

The final gate parameter was `beta=0.10788777`. The independent generic
calibration contained 57,344 subject-query states and fixed
`gamma_global=0.87860796`. Every frozen backbone tensor had the same SHA before
and after gate adaptation.

## Integrity and competence

All integrity errors were exactly zero:

| Check | Maximum error |
| --- | ---: |
| Replayed matched-global calibration | 0 |
| Gamma-one v1 logit reproduction | 0 |
| Shuffled per-subject gamma multiset | 0 |
| Stable-omitted pair influence | 0 |

The original seed-2101 backbone and the conditioned gate both pass every
`qualification_v2` rule. Under the conditioned gate, intact deterministic
accuracy is 0.8516 overall, 0.9229 on learned pairs, and 0.8231 on nonlearned
pairs. Write-off, alpha-zero, reset, and cross-subject fast-weight shuffle all
reduce nonlearned accuracy to about chance and reduce decision agreement to the
registered range. Query order invariance also passes.

## Primary local-fidelity result

| Condition | Retained direct correctness | H>A direct correctness | Other seven relations |
| --- | ---: | ---: | ---: |
| Original v1 | 0.04909 [0.04025, 0.05796] | -0.26379 [-0.31653, -0.21349] | 0.09310 [0.08451, 0.10191] |
| Conditioned gate | 0.04482 [0.03757, 0.05202] | -0.15123 [-0.19083, -0.11362] | 0.07199 [0.06506, 0.07910] |
| Matched global scalar | 0.04670 [0.03997, 0.05335] | -0.17225 [-0.21329, -0.13299] | 0.07743 [0.07085, 0.08428] |
| Shuffled gate | 0.04796 [0.03984, 0.05614] | -0.17116 [-0.21558, -0.12863] | 0.07806 [0.06968, 0.08654] |

Intervals are the registered participant-bootstrap 95% intervals. H>A does
move substantially toward zero under the conditioned gate, which is a useful
descriptive positive result. It nevertheless remains robustly wrong-sign.
More importantly, the gate reduces the retained-relation aggregate rather than
improving it:

```text
conditioned - original = -0.00427
95% interval            = [-0.00767, -0.00090]
```

The damage is concentrated in the other seven relations:

```text
conditioned - original = -0.02111
95% interval            = [-0.02519, -0.01717]
```

Conditioned state-query matching also fails the specificity rule. Its
aggregate direct correctness is below both the matched-global and shuffled
controls. The paired conditioned-minus-matched interval includes zero, and the
conditioned-minus-shuffled interval also includes zero.

## Global mechanism preservation

The gate does not obtain its result by destroying the global channel.

| Estimand | Original v1 | Conditioned gate |
| --- | ---: | ---: |
| Remote absolute LOO influence | 0.47218 [0.45087, 0.49419] | 0.43311 [0.41368, 0.45312] |
| Gauge-invariant third-party fraction | 0.21389 [0.20383, 0.22425] | 0.21581 [0.20576, 0.22611] |
| Expected-rank minus MAP alignment | 0.02996 [0.01401, 0.04947] | 0.03005 [0.01404, 0.04950] |

Both registered query-binding contrasts are unchanged exactly because the
candidate acts after operator binding:

- matched minus shared-endpoint normalized gain: 0.26216
  [0.25739, 0.26690];
- matched minus disjoint normalized gain: 0.30827
  [0.30378, 0.31261].

Thus fast-weight necessity, remote/third-party reassembly, query binding,
nonlearned inference, and terminal expected-rank-over-MAP projection all remain
supported on the new seed.

## Behavioral consequences

The conditioned gate does not move the symbolic-distance phenotype in the
desired direction.

| Condition | Overall accuracy | Learned accuracy | Nonlearned accuracy | Distance slope |
| --- | ---: | ---: | ---: | ---: |
| Original v1 | 0.84235 | 0.91006 | 0.81526 | 0.04843 |
| Conditioned gate | 0.84017 | 0.90633 | 0.81370 | 0.04847 |
| Matched global scalar | 0.83980 | 0.90698 | 0.81292 | 0.04900 |
| Shuffled gate | 0.84017 | 0.90714 | 0.81338 | 0.04890 |

These values are consequences, not the primary decision rule. They agree with
the mechanistic failure: the candidate neither rescues local fidelity nor
reduces the excessive slope.

## Why the candidate failed

The online statistic is an unsigned hidden-space magnitude ratio. It knows how
large curvature is relative to the local Jacobian drive, but not whether that
curvature opposes or supports the network's own first-order output direction.
The result shows that this distinction is necessary.

The gate's allocation is also opposite to the offline susceptibility ordering.
Across 90 retained subject-relation crossing cases,

```text
Spearman(gamma, crossing midpoint) = -0.142
p = 0.182
```

H>A has 48 crossing subjects with a mean fixed-grid midpoint of 0.559, yet its
mean online gamma is 0.905, among the least attenuated relation means and above
the matched global value 0.879. F>A similarly has 29 crossings with mean
midpoint 0.623 and mean gamma 0.904. The unsigned norm risk therefore does not
recover the safe-amplitude ordering found offline.

The strongest scientific conclusion is not simply that attenuation failed.
Attenuation improved the worst H>A response while preserving the global
backbone, but the chosen state variable attenuated beneficial expression across
the other relations and its state-query pairing was not causally helpful in the
registered aggregate. The missing information is the *direction* of curvature
relative to useful first-order expression, not curvature magnitude alone.

## Revised theory and next decisive test

Preserve the supported chain:

```text
fast-weight storage
-> query-addressed operator binding
-> correct first-order local value
-> global expected-rank assembly
```

Replace the failed v2 link:

```text
unsigned hidden-space curvature magnitude
-> safe state-conditioned amplitude
```

with the hypothesis that amplitude control must be sensitive to *opposition*
between the first-order drive and finite-amplitude curvature in the trained
output geometry. A minimal next candidate should remain online and low
capacity. One example to register before implementation is

```text
j = W_margin^T J_bu
k = W_margin^T K2(b,u)
r_opp = relu(-j k) / (j^2 + epsilon)
gamma = 1 / (1 + beta r_opp)
```

This uses the frozen network's own output-margin direction, not the true label,
relation identity, H>A indicator, LOO state, or crossing bracket. It leaves
gamma at one when curvature supports the first-order output and attenuates only
when the two oppose.

Before running it, freeze a new protocol that specifies numerical safeguards,
generic-only beta adaptation, the same matched-global and shuffled controls,
and the same local/global causal gates. Reuse the frozen seed-2101 backbone for
the next sufficiency test so that the only changed causal factor is the online
risk statistic. Do not run seeds 2102/2103, begin end-to-end co-adaptation, or
change the failed pilot's thresholds. Independent seeds become appropriate
only after a new candidate passes its registered one-backbone sufficiency test.
