# Dual evidence access v2.4 fresh-backbone confirmation

## Conclusion

The unchanged zero-parameter v2.4 differential-admission mechanism confirms on
both mandatory fresh backbones, seeds 2104 and 2105. Every source, artifact,
competence, shared-reference, routing, parameter, and numerical-integrity gate
passes. All four primary links pass independently within both networks, and
the registered outcome is `fresh_backbone_confirmation_pass`.

Participants were bootstrapped separately within each network and were never
pooled across networks. This is blind fresh-backbone mechanism confirmation,
not a network-population prevalence estimate.

Combined with the characterized-network v2.3/v2.4 results, this promotes the
following decomposition into the working main computational model:

```text
observed support evidence
  -> selective global admission -> P_T -> coherent global relational assembly
  -> broader weak local admission -> L_T -> query-addressed direct fidelity
```

The confirmed candidate remains exactly

```text
s_G        = m_t z_sr
s_L,shared = m_t z_sr
s_L,dual   = m_t [z_sr + (1-z_sr) p_sr].
```

No address, `L_T` update, gain rule, query read, `P_T`, recurrent computation,
`W_out`, `tanh`, subject reliability, or control changed.

## Frozen execution chain

- Protocol registration: `511e549`; the pre-implementation historical source
  aliases were added transparently in `1664306`.
- Confirmation runner, tests, and source lock: `56b94c3`, pushed before either
  fresh artifact was generated.
- Both 1000-step backbones and both 500-step generic-only gains were completed
  before the joint artifact lock. The lock was pushed as `741dc35` before
  either Liu evaluation.
- Seed-2104 checkpoint SHA-256:
  `cf62bd2488e18fc7d6b776ef63dcbe549d90722183fede66f95d07be7559e92a`;
  frozen generic gain `lambda_L=0.15213929`.
- Seed-2105 checkpoint SHA-256:
  `10931fa7e11e6af04a20e93562dbf5f808204c1d037e3b9609c13d099553bf92`;
  frozen generic gain `lambda_L=0.16323349`.
- Final result: `results/dual_evidence_access_confirmation_v2_4.json`, SHA-256
  `ca7e48e3b4ac3a5a75ef4d876bd56565e4a226e9bc6d8e197675610655643991`.
  A complete independent GPU replay to `/tmp` is byte-identical.
- Runtime: NVIDIA GeForce RTX 5090, CUDA 13.0, PyTorch 2.13.0+cu130, with one
  PyTorch intra-op and one inter-op CPU thread. Backbone and gain adaptation
  used `torch.compile(..., fullgraph=True, mode="default")`.

The frozen query-binding diagnostic emits its existing NumPy empty-slice
warning for rows lacking a mismatch class. All registered finite summaries,
decisions, and integrity values are unchanged; no warning, row, subject,
relation, or seed was filtered.

## Primary links

### 1. Stable-omitted direct fidelity

| Seed | Omitted exact candidate minus shared | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.06118 | [+0.04918, +0.07395] |
| 2105 | +0.05241 | [+0.04302, +0.06254] |

| Seed | Omitted direct causal candidate minus shared | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.19402 | [+0.17741, +0.21167] |
| 2105 | +0.20817 | [+0.19021, +0.22694] |

Both exact-probability and relation-LOO direct-correctness gates therefore
confirm without changing an endpoint or threshold.

### 2. Retained fidelity and the replicated trade-off

Retained own writes are exactly identical between candidate and shared access,
and retained direct causal changes are numerically zero. Exact retained
probability again decreases slightly:

| Seed | Retained exact candidate minus shared | 95% participant bootstrap | Rule |
| --- | ---: | ---: | ---: |
| 2104 | -0.00114 | [-0.00219, -0.00026] | lower >= -0.005 |
| 2105 | -0.00143 | [-0.00266, -0.00043] | lower >= -0.005 |

Both networks pass the frozen noninferiority margin. The negative is now
replicated across all four v2.4 networks: broader omitted writes rescue omitted
queries while causing small cross-talk into retained decisions. Do not call
retained behavior literally unchanged.

### 3. Evidence and query specificity

Matched evidence routing beats the blockwise signed-scalar derangement:

| Seed | Omitted exact advantage | 95% bootstrap | Omitted direct advantage | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2104 | +0.05450 | [+0.03496, +0.07534] | +0.19360 | [+0.17727, +0.21063] |
| 2105 | +0.04987 | [+0.03018, +0.07030] | +0.20772 | [+0.19030, +0.22572] |

Natural query addressing also beats the canonical query derangement on omitted
direct causal correctness:

| Seed | Natural minus query shuffle | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | +0.20401 | [+0.18378, +0.22581] |
| 2105 | +0.21888 | [+0.19700, +0.24206] |

The confirmation therefore identifies matched evidence-to-relation and
trace-to-query routing, not an undirected increase in local energy.

### 4. Local-only access remains nontransitive

| Seed | `P`-off omitted exact | 95% bootstrap | Candidate minus shared `P`-off |
| --- | ---: | ---: | ---: |
| 2104 | 0.62335 | [0.59275, 0.65429] | +0.16172 [+0.14611, +0.17819] |
| 2105 | 0.64438 | [0.61432, 0.67428] | +0.18031 [+0.16368, +0.19787] |

`P`-off sampled nonlearned accuracy remains near chance, with upper bounds
`0.46896` and `0.46942`, below the frozen `0.55` gate. The registered remote
contrast also remains negative:

| Seed | `remote_Poff - 0.25 remote_shared` | 95% participant bootstrap |
| --- | ---: | ---: |
| 2104 | -0.06767 | [-0.07235, -0.06304] |
| 2105 | -0.06860 | [-0.07344, -0.06374] |

`L`-off exactly restores v1, and all qualification, fast-weight necessity,
query-binding, remote/third-party reassembly, and expected-rank-over-MAP gates
remain intact. `L_T` is not acting as a second transitive learner.

## Secondary behavior and remaining discrepancy

Sampled learned accuracy improves under the candidate, with a small
nonlearned cost:

| Seed | Shared learned -> dual | Shared nonlearned -> dual | Shared overall -> dual |
| --- | ---: | ---: | ---: |
| 2104 | 0.91120 -> 0.92516 | 0.80435 -> 0.80357 | 0.83488 -> 0.83831 |
| 2105 | 0.91753 -> 0.93149 | 0.80695 -> 0.80526 | 0.83854 -> 0.84133 |

The symbolic-distance discrepancy remains:

| Seed | Shared sampled slope -> dual | Shared exact slope -> dual |
| --- | ---: | ---: |
| 2104 | 0.04840 -> 0.04949 | 0.05924 -> 0.06068 |
| 2105 | 0.04874 -> 0.04954 | 0.05884 -> 0.05989 |

As in 2102/2103, the exact increase is mainly the newly rescued learned-
omitted contribution. The nonlearned contribution changes little and remains
dominant (`0.04933` and `0.04901`). Local access and global distance dependence
are therefore still separate mechanism questions.

## Integrity

- Shared-access local state, exact probabilities, direct causal values, and
  sampled behavior reproduce their same-process v2.3 references exactly.
- Retained own-write, omitted `m_t p_sr`, global-logit, `P`-off global,
  local-off v1, and evidence-multiset errors are zero in both seeds.
- Every evidence and query map is a derangement; support and query reversal
  checks have zero error.
- Backbone tensors remain unchanged during gain adaptation and Liu replay.
- Local margin identity errors are `7.12e-7`, below the frozen `1e-6` rule.
- Exact slope decomposition errors are at most `5.55e-17`.
- Two complete confirmation evaluations are byte-identical.

## Revised theory and next test

The main computational model can now be frozen as:

```text
selective global admission -> P_T -> expected-rank-like global assembly
broader weak local admission -> L_T -> addressed direct-experience fidelity
```

This is supported by the seed-2101 discovery chain, v2.3 mechanism replication
on 2102/2103, characterized-network v2.4 sufficiency on 2102/2103, and blind
fresh-backbone confirmation on 2104/2105. It still does not establish the
unique human `p_sr`, biological stores, or network-population prevalence.

The next route is now Track B only: a separately frozen read-only analysis of
the pure global (`L`-off) chain

```text
P_T -> Hodge potential s -> pair margin m_ij -> sigmoid probability p_ij
```

on one to two existing competent networks. It must distinguish potential
geometry, potential amplitude, residual margin, and fixed-choice-link
contributions to excessive nonlearned symbolic-distance dependence. v2.4 is
now immutable and must not be altered to repair that separate discrepancy.
