# Dual evidence access v2.4 pilot

## Conclusion

The registered zero-parameter dual-evidence-access candidate passes all four
primary mechanism links independently on both mandatory frozen development
backbones, seeds 2102 and 2103. Every competence, source, artifact, estimator,
and integrity gate passes after the prospectively constrained execution repair
described below. The registered cross-seed outcome is `all_links_pass`.
Participants were bootstrapped separately within each network and were never
pooled across networks; no network-population inference was performed.

The result supports the following working mechanism:

```text
observed support evidence
  -> selective global admission -> P_T -> remote/global relational assembly
  -> broader weak local admission -> L_T -> query-addressed direct fidelity
```

The tested distinction is an evidence-admission rule, not a new memory update
or readout. The global path remains

```text
s_G = m_t z_sr,
```

the exact v2.3 shared-access control uses

```text
s_L,shared = m_t z_sr,
```

and the registered candidate uses the existing subject relation reliability
without a new parameter:

```text
s_L,dual = m_t [z_sr + (1 - z_sr) p_sr].
```

Thus every retained relation's own write is exactly v2.3, while a relation
omitted from `P_T` receives a weaker `m_t p_sr` local write. The address,
`L_T` update, frozen gain, query read, `P_T`, recurrent network, activation,
and output readout do not change.

This two-development-backbone sufficiency result does not establish network-
population prevalence, a uniquely correct human evidence model, biological
memory stores, unique tensor-product coding, or access to information that was
never presented. It shows only that weak evidence from normally presented
support trials can remain locally useful when the same evidence is excluded
from global assembly by the frozen model bottleneck.

## Frozen provenance and execution repair

- Protocol: `benchmarks/dual_evidence_access_pilot_v2_4.json`; the initial
  registration and its two pre-implementation clarifications were committed
  and pushed as `51c2f74`, `b625f77`, and `81762d3`.
- Initial implementation/source lock:
  `benchmarks/dual_evidence_access_pilot_v2_4.lock.json`; committed and pushed
  as `cc64d60` before any v2.4 condition was evaluated.
- The first execution was non-interpretable because the shared-access exact-
  probability identity differed from frozen v2.3 by `2.89e-8` and `2.42e-8`.
  It used a once-materialized float32 corrected margin, whereas frozen v2.3
  defines the estimand as the separately exported
  `global_logits + applied_local_margins`. All other integrity gates passed.
- That output is preserved unchanged in
  `results/dual_evidence_access_pilot_v2_4_attempt1_noninterpretable.json`,
  SHA-256
  `3e2f7e5c63e95cd98a151db1f8e361469563b1083e0548eb66c8be2cc712409e`.
  None of its primary effects was interpreted or used to change a candidate,
  control, seed, bootstrap, threshold, or outcome rule.
- Superseding repair lock:
  `benchmarks/dual_evidence_access_pilot_v2_4.repair1.lock.json`; committed and
  pushed as `86e8983` before reexecution. The only repair was exact reuse of
  the frozen v2.3 component-sum probability estimator, backed by a regression
  test. The full suite passed (`178 passed`) before reexecution.
- Final result: `results/dual_evidence_access_pilot_v2_4.json`, SHA-256
  `16f09acb1ff0caa6eae2821bb278a2e73afc5f3470825959a026807248d9b717`.
  An independent complete replay to `/tmp` is byte-identical.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, with one
  PyTorch intra-op and one inter-op CPU thread. v2.4 performs frozen replay
  only: it trains no backbone, adapts no gain, and compiles no new model.
- Frozen checkpoint SHA-256 values are `83efbfee...09eb` for seed 2102 and
  `72046a49...f63d` for seed 2103. The exact reused gains are `0.18592523` and
  `0.16086961`.

## Primary mechanism links

### 1. Stable-omitted direct-fidelity rescue

The candidate improves exact correct probability on stable-omitted learned
cells in both networks:

| Seed | Candidate minus v2.3 | 95% participant bootstrap |
| --- | ---: | ---: |
| 2102 | +0.07432 | [+0.05961, +0.09059] |
| 2103 | +0.06024 | [+0.04732, +0.07490] |

The corresponding relation-LOO, antisymmetrized, Hodge-residual direct causal
correctness effects are also positive:

| Seed | Candidate minus v2.3 | 95% participant bootstrap |
| --- | ---: | ---: |
| 2102 | +0.23711 | [+0.21662, +0.25839] |
| 2103 | +0.20515 | [+0.18729, +0.22340] |

Omitted exact probability rises from `0.72234` to `0.79667` in seed 2102 and
from `0.73373` to `0.79397` in seed 2103. This is a probability-level and
causal direct-effect rescue, not only a sampled-accuracy movement.

### 2. Retained fidelity is noninferior, with a qualified negative

Every retained relation's own local scalar and tensor write are exactly
unchanged (maximum error `0`), and the change in retained direct causal
correctness is numerically zero. Exact retained probability nevertheless
decreases slightly because newly admitted omitted writes can cross-talk into
retained queries:

| Seed | Candidate minus v2.3 | 95% participant bootstrap | Frozen lower rule |
| --- | ---: | ---: | ---: |
| 2102 | -0.00182 | [-0.00324, -0.00068] | >= -0.005 |
| 2103 | -0.00135 | [-0.00257, -0.00029] | >= -0.005 |

Both seeds pass the prospectively frozen noninferiority rule, but the small,
reproducible probability cost must be retained as a negative constraint. The
result supports preservation of the existing own-write mechanism, not literal
absence of all effects on retained decisions.

### 3. Evidence and query binding are necessary

Natural evidence-to-support routing beats a within-subject, within-block
derangement that preserves the exact candidate scalar multiset:

| Seed | Omitted exact advantage | 95% bootstrap | Omitted direct advantage | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2102 | +0.06973 | [+0.04463, +0.09633] | +0.23660 | [+0.21649, +0.25696] |
| 2103 | +0.05814 | [+0.03584, +0.08253] | +0.20471 | [+0.18736, +0.22257] |

Natural query addressing also beats the canonical query-address derangement
on omitted direct causal correctness:

| Seed | Natural minus query shuffle | 95% participant bootstrap |
| --- | ---: | ---: |
| 2102 | +0.24931 | [+0.22404, +0.27624] |
| 2103 | +0.21571 | [+0.19370, +0.23846] |

Extra local signal energy is therefore insufficient. The weak scalar must be
bound to the support relation that generated it and retrieved through the
matching query address.

### 4. Local-only weak access does not become transitive inference

With `P_T` removed, the candidate still rescues stable-omitted direct cells:

| Seed | `P`-off omitted exact probability | 95% bootstrap | Candidate minus shared `P`-off |
| --- | ---: | ---: | ---: |
| 2102 | 0.64322 | [0.60707, 0.67887] | +0.20151 [+0.18100, +0.22320] |
| 2103 | 0.63369 | [0.58632, 0.68002] | +0.14581 [+0.13013, +0.16195] |

At the same time, sampled nonlearned accuracy remains near chance, with upper
bounds well below the frozen `0.55` gate (`0.46487` and `0.47429`). The
registered local-only remote contrast
`remote_Poff - 0.25 remote_shared-intact` is negative in both networks:

| Seed | Remote contrast | 95% participant bootstrap |
| --- | ---: | ---: |
| 2102 | -0.05660 | [-0.06080, -0.05240] |
| 2103 | -0.06754 | [-0.07227, -0.06269] |

Conversely, `L`-off exactly restores v1 logits, and the frozen qualification,
fast-weight necessity, query binding, remote/third-party reassembly, and
expected-rank-over-MAP gates remain intact. Broader local access therefore
does not turn `L_T` into a second global relational learner.

## Secondary behavior and remaining negatives

Sampled learned accuracy improves, while nonlearned accuracy declines slightly:

| Seed | v2.3 learned -> candidate | v2.3 nonlearned -> candidate | v2.3 overall -> candidate |
| --- | ---: | ---: | ---: |
| 2102 | 0.90341 -> 0.92224 | 0.80201 -> 0.80032 | 0.83098 -> 0.83516 |
| 2103 | 0.91656 -> 0.92841 | 0.80143 -> 0.80052 | 0.83432 -> 0.83706 |

The symbolic-distance discrepancy is not repaired and becomes slightly larger:

| Seed | v2.3 sampled slope -> candidate | v2.3 exact slope -> candidate |
| --- | ---: | ---: |
| 2102 | 0.04772 -> 0.05087 | 0.05969 -> 0.06169 |
| 2103 | 0.04864 -> 0.05033 | 0.05913 -> 0.06076 |

The exact increase is concentrated primarily in the newly rescued learned-
omitted contribution (`0.00216 -> 0.00395` and `0.00284 -> 0.00434`), while
the nonlearned contribution changes little and remains the dominant source
(`0.04969` and `0.04947` under the candidate). This preserves the registered
separation of scientific questions: v2.4 resolves local evidence access but
does not resolve excessive global/nonlearned distance dependence.

## Integrity

- Shared-access local state, exact probabilities, raw direct causal values,
  and sampled behavior reproduce frozen v2.3 exactly in both seeds.
- Candidate retained own-write error, omitted `m_t p_sr` scalar error, global
  condition logit error, `P`-off global identity error, local-off v1 error,
  and evidence-shuffle multiset error are all exactly zero.
- Every evidence map and query map is a derangement. Support reversal leaves
  the value-key product invariant, and query reversal negates the key, both
  with maximum error zero.
- Backbone tensor hashes are unchanged and every condition uses the same
  frozen gain within seed.
- Exact slope components reconstruct the total with maximum error
  `2.78e-17`.
- The final result is byte-identical across two complete GPU executions.

## Revised theory and next decisive test

The convergent v2.3 and v2.4 evidence now supports two causally distinct
computations fed by different evidence-admission rules:

```text
selectively admitted effective evidence -> P_T -> coherent global inference
broader weak observed evidence          -> L_T -> direct-experience fidelity
```

The strongest remaining qualification is scope. Seeds 2102 and 2103 are
independent networks for the v2.3 mechanism, but they were already
characterized when the exact v2.4 admission equation was proposed. Before a
network-population claim, freeze the unchanged v2.4 rule on new independent
backbones, train and generic-adapt all mandatory artifacts before inspecting
any Liu outcome, and require the same four links separately within each seed.
Do not refit `p_sr`, the local gain, the noninferiority margin, or the routing
controls.

The next separate scientific family is the `P_T`/nonlearned policy source of
the excessive distance slope. Its first step should be a read-only localization
of how terminal expected-rank-like potentials are converted into distance-
dependent nonlearned choice confidence. v2.4 should remain fixed and should
not be altered to solve that global-policy problem.
