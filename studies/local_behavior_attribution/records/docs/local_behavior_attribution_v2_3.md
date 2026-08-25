# Local-behavior attribution v2.3

## Conclusion

The registered read-only attribution resolves the apparent mismatch between
the v2.3 direct causal rescue and its nearly unchanged sampled learned
accuracy. Three results jointly explain it:

1. Stable-omitted learned cells contain `71.36%` of the frozen v1 exact error
   mass, although they are only `296/1232` learned orientation cells. Their own
   local trace is prospectively and exactly zero.
2. Of the `936` retained cells, `827` (`88.35%`) already have exact v1 correct
   probability at least `0.99`. The local trace significantly improves exact
   retained probability and removes `20.15%` of retained error mass, but this
   continuous improvement rarely changes the already near-ceiling sampled
   endpoint.
3. The excessive exact-probability distance slope is dominated by nonlearned
   queries, which contribute `81.05%` of the v1 slope. Improvements in both
   learned groups are offset by a small worsening of the nonlearned
   contribution, so the total slope remains essentially unchanged.

The frozen decision outcome is `confirmation_estimand_sensitivity`. Exact
retained probability improves robustly while the sampled learned-accuracy
contrast remains unresolved. Address cross-talk does not reach the registered
materiality threshold, and retained value conversion is not identified as the
limiting route.

The stronger `dual_evidence_access` branch does not formally pass: although
omission dominance is decisive and the `P`-off/local-intact retained point
estimate is `0.66768`, its bootstrap lower bound is `0.64435`, narrowly below
the prospectively fixed `0.65` threshold. This threshold must not be moved.
Pure-local diagnostics are strong, but they do not replace the failed
registered conjunction.

Do not change v2.3, increase its gain, train a value transform, alter stable
omission, or immediately run 2102/2103. The next admissible step is to freeze a
new replication contract before execution, keeping the mechanism unchanged
and using exact retained probability plus the causal direct/double-
dissociation estimands as prospectively defined confirmation outcomes. The
failed v2.3 sampled endpoint remains preserved and reportable; it cannot be
retroactively relabeled.

## Frozen provenance

- Attribution protocol: `benchmarks/local_behavior_attribution_v2_3.json`,
  committed as `e0a2d87` before implementation or cellwise replay.
- Implementation and source lock:
  `benchmarks/local_behavior_attribution_v2_3.lock.json`, committed as
  `45fb20c` before execution.
- Frozen backbone SHA-256:
  `3671582a3d0f638f9b383e9bea20b966824462217566edf8b15f8152d2a2c78d`.
- Frozen v2.3 gain artifact SHA-256:
  `32a0750f3e4ce08703da0f3edca506d649e0e7d123aa67cc0bc3fc91fd059cc8`.
- Result: `results/local_behavior_attribution_v2_3.json`, SHA-256
  `599b379d1d1a9ae5a0f7f4b4bd8f8a99b30120163f5cee89cf0680a0d2e953d2`.
- A complete independent execution to `/tmp` is byte-identical to the final
  result.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, with one
  PyTorch intra-op and one inter-op CPU thread.

No training, fitting, parameter update, threshold change, new seed, or new
choice sampling occurred. The existing sampled endpoint was deterministically
replayed only as an integrity check.

## Cellwise estimand

The unit is subject by learned relation by query orientation. Retaining both
orientations avoids applying the sigmoid after an inappropriate logit average.
For correct-orientation sign `y` and frozen temperature `T=0.25`:

```text
g_sro   = y_ro * m_v1,sro
ell_sro = y_ro * m_local,sro
d_sro   = g_sro + ell_sro

pG_sro = sigmoid(g_sro / T)
pD_sro = sigmoid(d_sro / T)
```

`R_sr=1` denotes a relation retained by the frozen stable-omission bottleneck;
the same status labels both orientations. The primary attribution uses these
exact probabilities rather than sampled choices.

Every registered identity passes:

| Integrity check | Maximum error |
| --- | ---: |
| Dual = global + local margin | 0 |
| Full local = self + cross | 0 |
| Stable-omitted self contribution | 0 |
| Frozen sampled endpoint reproduction | 0 |
| Additive slope contribution identity | `2.78e-17` |

## Error-mass source

For the frozen v1 exact probability:

```text
E_R = sum R(1-pG)
E_O = sum (1-R)(1-pG)
```

The attribution is:

| Group | Cells | Error mass | Mean error per cell |
| --- | ---: | ---: | ---: |
| Learned retained | 936 | 31.1969 | 0.03333 |
| Learned stable-omitted | 296 | 77.7280 | 0.26259 |

The omitted error-mass fraction is:

```text
E_O / (E_R + E_O) = 0.71359
95% bootstrap interval = [0.62810, 0.79256]
```

The registered lower-bound-above-`0.50` omission-dominance rule therefore
passes. An omitted cell carries almost eight times as much mean exact error as
a retained cell. This is the main reason an intervention that is correct and
large on retained relations has little leverage over aggregate learned
behavior.

## Retained ceiling and exact rescue

The retained v1 cells occupy the following exact-probability regimes:

| v1 exact probability | Cells | Mean local `delta p` |
| --- | ---: | ---: |
| `<0.50` | 28 | +0.13358 |
| `[0.50,0.90)` | 26 | +0.06135 |
| `[0.90,0.99)` | 55 | +0.01468 |
| `[0.99,1.00]` | 827 | +0.00017 |

Across all retained cells, dual-minus-v1 exact probability is:

```text
+0.00682 [0.00437, 0.00950]
```

This removes `20.15% [14.07%,27.35%]` of retained exact error mass. Among the
28 retained cells initially on the wrong side, 9 cross to the correct side;
only 1 of the 908 initially correct retained cells crosses in the harmful
direction. The self trace alone also rescues 9 cells.

The sampled endpoint nevertheless remains exactly the frozen v2.3 result:

```text
learned accuracy change = +0.00146 [-0.00244,+0.00536]
```

Thus the registered `exact_rescue_with_sampled_insensitivity` rule passes. The
continuous mechanism is effective, but the sampled aggregate is insensitive
because most writable cells are already saturated and most remaining error is
unwritable under stable omission.

## Self trace versus cross-talk

For retained cells:

| Component | Mean | 95% bootstrap interval |
| --- | ---: | ---: |
| Signed self | +0.33691 | [+0.32798, +0.34546] |
| Signed cross | -0.02974 | [-0.04306, -0.01701] |
| Absolute self | 0.33691 | [0.32798, 0.34546] |
| Absolute cross | 0.09571 | [0.08793, 0.10345] |

Every retained self contribution is positive. Absolute cross/self is
`0.28409`, below the registered one-third materiality threshold. Cross-talk is
therefore a modest, directionally negative dilution, not the primary failure
source.

For stable-omitted cells, self is exactly zero. Cross-talk alone changes exact
probability by `-0.01359 [-0.02478,-0.00280]` and increases omitted error mass
by about `5.13%`. A larger global local gain would therefore amplify both the
useful retained self signal and harmful omitted cross-talk; it is not a valid
repair.

## Local-only retained versus omitted

The registered `P`-off/local-intact partition is:

| Group | Exact probability | Hard accuracy |
| --- | ---: | ---: |
| Retained | 0.66768 [0.64435, 0.69076] | 0.70318 [0.67093, 0.73450] |
| Stable-omitted | 0.48788 [0.44055, 0.53478] | 0.50169 [0.42500, 0.57747] |

Retained-minus-omitted exact probability is
`+0.18675 [0.14060,0.23408]`. The omitted upper bound and between-group
difference pass, but the retained lower bound misses the frozen `0.65`
threshold by `0.00565`; consequently `retained_local_sufficient` is formally
false.

The pure local contribution, reported prospectively as a diagnostic, is much
cleaner:

| Group | Exact probability | Hard accuracy |
| --- | ---: | ---: |
| Retained | 0.74721 [0.73719, 0.75700] | 0.95628 [0.93862, 0.97205] |
| Stable-omitted | 0.46626 [0.44385, 0.48865] | 0.44082 [0.35507, 0.52696] |

This supports the content and direction of the local state, but it cannot be
substituted for the failed registered `P`-off conjunction after observing the
result.

## Exact symbolic-distance slope decomposition

The exact-probability OLS slope decomposes additively:

| Group | v1 contribution | Dual contribution | Dual - v1 |
| --- | ---: | ---: | ---: |
| Learned retained | 0.00783 | 0.00752 | -0.00031 |
| Learned omitted | 0.00352 | 0.00320 | -0.00032 |
| Nonlearned | 0.04856 | 0.04926 | +0.00070 |
| Total | 0.05991 | 0.05998 | +0.00007 |

Nonlearned queries contribute `81.05%` of the v1 slope. The local trace makes
both learned contributions slightly smaller, but its cross-talk makes the
nonlearned contribution slightly larger, cancelling the improvement. The
excessive distance slope is therefore primarily a global/nonlearned policy
problem, not a behavioral criterion that this direct-memory module should be
expected to solve alone.

## Mechanistic decision and next route

The registered flags are:

| Flag | Result |
| --- | --- |
| Omission dominant | PASS |
| Retained local sufficient under the full `P`-off rule | FAIL narrowly |
| Retained value conversion limited | FAIL |
| Exact rescue with sampled insensitivity | PASS |
| Address interference material | FAIL |

According to the frozen outcome hierarchy, the result is
`confirmation_estimand_sensitivity`, not `dual_evidence_access`. The supported
working theory is:

```text
retained evidence -> strong, correctly signed persistent local trace
stable-omitted evidence -> no self trace and most remaining learned error
nonlearned/global policy -> most of the excessive distance slope
sampled learned accuracy -> insensitive to retained continuous rescue at this ceiling
```

The next contract should keep the candidate exactly unchanged and ask whether
the direct causal rescue, exact retained probability gain, self/query
specificity, and `P`/`L` double dissociation replicate on independent
development seeds. Sampled learned accuracy and distance slope must remain
reported as the original failed endpoints, not silently removed or converted
into success criteria. Only after that replication should a distinct
local/global evidence-access hypothesis be considered.

## Reproduction

```bash
direnv exec . python -m fsrl.local_behavior_attribution
direnv exec . python -m pytest -q
```
