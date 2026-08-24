# Conjunctive persistent local-trace pilot v2.3

## Conclusion

The registered one-backbone v2.3 test is a valid mixed result. It provides the
first strong causal double dissociation between the frozen fast-weight global
channel and a separately persistent direct-experience trace, but it does not
pass the complete behavioral sufficiency contract.

Nine of ten registered primary flags pass. The natural local trace increases
retained-relation direct correctness by `+0.30039 [0.29207,0.30860]`, makes
H>A robustly positive, beats the query-address derangement, is more than three
times as direct as remote, and remains sufficient for above-chance learned-pair
performance when `P_T` is removed. Conversely, `P_T` removal collapses remote
reassembly and leaves nonlearned performance near chance, while local removal
exactly restores v1 and its confirmed global mechanism.

The sole failed primary rule is sampled learned-accuracy rescue. Dual-intact
learned accuracy changes by only `+0.00146 [-0.00244,+0.00536]`; nonlearned
accuracy decreases by `-0.00604` but remains inside the registered preservation
bound. The symbolic-distance slope also does not improve. Therefore this pilot
supports a causally separable local storage computation on seed 2101, but does
not show that the minimal fixed-address trace is sufficient to repair the
formal behavioral fingerprint.

The serialized outcome is the contract's catch-all
`valid_local_or_specificity_failure`; the flag vector shows that neither local
rescue nor specificity failed. The actual failure is the sole learned-accuracy
flag, and that more precise interpretation governs this report.

Do not run seeds 2102/2103, choose a larger gain from Liu, relax the learned-
accuracy rule, train relation-specific values, change the tensor address, or
begin end-to-end co-adaptation. Preserve the double dissociation as a positive
mechanistic result and treat the full candidate as a registered mixed negative.
Before registering another trainable mechanism, use a separately frozen
read-only attribution to determine whether the behavioral non-rescue is
concentrated in stable-omitted relations, where this trace is prospectively
required to remain zero, or in retained low-margin relations that would require
a different shared value/expression transform.

## Frozen provenance

- Protocol: `benchmarks/conjunctive_local_trace_pilot_v2_3.json`, committed as
  `9b78c4d` before implementation, adaptation, or evaluation.
- Implementation and source lock:
  `benchmarks/conjunctive_local_trace_pilot_v2_3.lock.json`, committed as
  `9005e34` before gain adaptation.
- Post-adaptation, pre-Liu artifact lock:
  `benchmarks/conjunctive_local_trace_pilot_v2_3.artifact_lock.json`, committed
  as `404b8a3` before any Liu evaluation.
- Frozen backbone:
  `output/curvature-gate-pilot-v2/seed-2101/backbone/net.dat`, SHA-256
  `3671582a3d0f638f9b383e9bea20b966824462217566edf8b15f8152d2a2c78d`.
- Frozen gain artifact:
  `output/conjunctive-local-trace-pilot-v2-3/seed-2101/local/gain.json`,
  SHA-256
  `32a0750f3e4ce08703da0f3edca506d649e0e7d123aa67cc0bc3fc91fd059cc8`.
- Final result: `results/conjunctive_local_trace_pilot_v2_3.json`, SHA-256
  `5176802b437795a7b609472fb2969c3ba33d7bf74f8d05b9450dbef812d3cc4d`.
- A complete independent evaluation to `/tmp` is byte-identical to the final
  result.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, one
  PyTorch intra-op and one inter-op CPU thread. Adaptation used
  `torch.compile(..., fullgraph=True, mode="default")` for the backbone, local
  write, and local read/policy operations.

The complete 500-step generic-only adaptation was deterministically replayed;
the two training logs are byte-identical with SHA-256
`730c0129f1de283cdb7a8c17694136d544e98ae0dd1f6d62c552ca48e88758ac`.
No Liu value entered either run.

## Registered local mechanism

At the normal step-0 pair input, let `c_l` and `c_r` be the two existing item
cue vectors. The local address is the normalized antisymmetric tensor product:

```text
z(c_l,c_r) = vec(c_l c_r^T - c_r c_l^T)
k(c_l,c_r) = z / max(||z||_2, 1e-8)
k(c_r,c_l) = -k(c_l,c_r)
```

The scalar-value persistent trace is:

```text
L_0 = 0
L_(t+1) = L_t + s_t k(c_l,c_r)
ell_e = L_T dot k(c_l,c_r)
m_e = m_v1,e + lambda_L ell_e
```

`s_t` is the signed support magnitude after the existing subject encoding
bottleneck. Reversing a presentation changes both `s_t` and `k_t` signs, so
the stored relation is presentation invariant. `rho=1` and `omega=1` are
fixed. The module receives no item index, Liu relation identity, label,
correctness, residual target, posterior, or LOO state.

`lambda_L=softplus(raw_lambda_L)` is the only trainable parameter. Generic-only
adaptation selects:

```text
lambda_L = 0.1694634557
```

Every backbone tensor is byte-identical before and after adaptation. This is
not a second transitive network: its address and value transforms are fixed,
and it only binds the normal item-cue conjunction to normally encoded signed
support evidence.

## Frozen conditions and integrity

Four conditions were evaluated:

1. `original_v1_local_off`: exact v1 with the local margin zeroed.
2. `dual_intact`: frozen `P_T` plus the natural local trace.
3. `local_query_key_shuffle`: the natural trace with a fixed within-subject
   derangement of all 28 canonical query addresses, preserving orientation and
   reusing the same map for intact and every LOO replay.
4. `global_P_off_local_intact`: `P_T=0` at query while the natural local trace
   remains intact.

All source, artifact, backbone, and control checks pass:

| Integrity check | Result |
| --- | ---: |
| Local-off versus v1 maximum logit error | 0 |
| Total-margin minus global-plus-local identity error | `7.08e-7` |
| Stable-omitted local influence | 0 |
| Query-control maps | all derangements |
| Trainable v2.3 parameters | `raw_lambda_L` only |

The original backbone passes qualification and all registered fast-weight and
query-order checks. The inherited query-binding estimator emits the known
NumPy empty-slice warnings on zero-norm, already omitted rows; the frozen
retained mask excludes these rows and all registered summaries are finite.

## Direct local rescue

Participant-bootstrap intervals are frozen 95% intervals.

| Condition | Retained direct correctness | H>A | Remote absolute |
| --- | ---: | ---: | ---: |
| Original v1 / local off | 0.04909 [0.04030, 0.05790] | -0.26379 [-0.31598, -0.21256] | 0.47218 |
| Dual intact | 0.34947 [0.33907, 0.35981] | 0.33976 [0.28583, 0.39317] | 0.46719 |
| Query-key shuffle | 0.03901 [0.02821, 0.05021] | -0.29813 [-0.35045, -0.24628] | 0.47946 |
| `P` off / local intact | 0.30039 [0.29207, 0.30860] | 0.60354 [0.59488, 0.61279] | 0.01998 |

Registered contrasts are:

| Contrast | Mean | 95% bootstrap interval |
| --- | ---: | ---: |
| Dual - v1, aggregate direct correctness | +0.30039 | [+0.29207, +0.30860] |
| Dual - v1, H>A | +0.60354 | [+0.59488, +0.61279] |
| Dual - shuffled, aggregate direct correctness | +0.31046 | [+0.29947, +0.32174] |

All eight relations move toward positive direct correctness; this is not an
H>A/F>A-specific training effect:

| Relation | v1 | Dual intact | Dual - v1 |
| --- | ---: | ---: | ---: |
| F>A | -0.01551 | 0.42137 | +0.43688 |
| C>B | 0.07278 | 0.15876 | +0.08598 |
| E>B | 0.13830 | 0.39706 | +0.25876 |
| G>C | 0.11075 | 0.45390 | +0.34315 |
| F>D | 0.10516 | 0.27833 | +0.17317 |
| G>D | 0.09607 | 0.35398 | +0.25791 |
| H>E | 0.15618 | 0.41155 | +0.25538 |
| H>A | -0.26379 | 0.33976 | +0.60354 |

The natural local branch has mean absolute direct influence `0.33691`, versus
only `0.01998` on disjoint remote pairs. The registered
`direct - 3 * remote` lower bound is `0.26969`, well above zero. Thus the
trace is selectively content addressed rather than a second global relational
field.

## Causal double dissociation

The critical `P`-off/`L`-intact cell passes all three local-only rules when all
77 participants are included:

| Measure | Mean | 95% bootstrap interval |
| --- | ---: | ---: |
| Learned accuracy | 0.62127 | [0.59838, 0.64448] |
| Nonlearned accuracy | 0.46701 | [0.44870, 0.48494] |
| Learned - nonlearned | +0.15425 | [+0.13081, +0.17786] |

Its remote absolute influence is `0.01998`; the registered contrast against
25% of the v1 remote influence has upper bound `-0.09318`, so global remote
reassembly collapses. In the opposite intervention, local-off is exactly v1
and therefore retains the formally confirmed global channel. Query-address
derangement removes the local direct rescue while leaving the frozen global
branch intact.

This supports the functional decomposition:

```text
P_T -> nonlearned inference and remote/global reassembly
L_T -> persistent, query-addressed direct-experience fidelity
```

It does not establish that tensor-product binding is the unique implementation
or that the two states are biologically distinct stores.

## Preserved global mechanism and failed behavioral sufficiency

Dual-intact retains `98.94%` of v1 remote absolute reassembly and `98.42%` of
the gauge-invariant third-party relational fraction. Query-binding contrasts
remain positive, and terminal expected-rank-over-MAP alignment remains
positive with bootstrap lower bound `0.01363`.

However, the registered sampled behavioral rescue fails:

| Condition | Learned | Nonlearned | Overall | Distance slope |
| --- | ---: | ---: | ---: | ---: |
| Original v1 / local off | 0.91006 | 0.81526 | 0.84235 | 0.04843 |
| Dual intact | 0.91153 | 0.80922 | 0.83845 | 0.04910 |
| Query-key shuffle | 0.90601 | 0.81539 | 0.84128 | 0.04853 |
| `P` off / local intact | 0.62127 | 0.46701 | 0.51109 | 0.02310 |

Dual-minus-v1 learned accuracy is only
`+0.00146 [-0.00244,+0.00536]`, so the registered lower-bound-above-zero rule
fails. Nonlearned accuracy changes by
`-0.00604 [-0.00870,-0.00344]`; this is scientifically negative but remains
inside the frozen `-0.02` preservation bound. The distance slope slightly
increases instead of improving.

The result therefore separates two claims that should not be collapsed:

```text
supported:
  a minimal persistent conjunctive state can causally restore retained-edge
  local fidelity and dissociate from P_T-mediated global inference

not supported:
  the same fixed-address, one-gain candidate is sufficient to repair the
  full sampled learned-accuracy / distance-slope behavioral fingerprint
```

## Residual fingerprint and next decisive analysis

The descriptive correlation between absolute local margin and the frozen v1
absolute first-order residual is `0.30611`. Cells in the top residual quartile
receive mean absolute local margin `0.21605`, versus `0.14739` elsewhere. This
connects the earlier residual localization to when the persistent trace has a
larger causal contribution, but it was not a training target or decision rule.

The next step should be read-only and prospectively separated from this result:

1. partition sampled learned errors and signed margins by retained versus
   stable-omitted relation state;
2. measure how often the frozen local correction crosses a decision or choice-
   probability threshold for retained relations;
3. distinguish an upstream evidence-retention limit from an insufficient
   shared local value/expression transform.

If errors concentrate in stable omissions, increasing the current gain or
changing the address cannot solve the problem without violating the direct-
memory evidence contract; the theory must explain how local persistence can
retain weak evidence that the global channel does not encode. If retained
low-margin errors dominate, the next candidate should target a shared value or
expression transform, not replace the now-supported conjunctive storage and
query-address mechanism.

## Reproduction

```bash
direnv exec . python -m fsrl.conjunctive_local_trace_pilot evaluate
direnv exec . python -m pytest -q
```
