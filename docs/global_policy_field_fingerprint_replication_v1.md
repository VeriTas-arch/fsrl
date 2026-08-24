# Global-policy field fingerprint replication v1

## Conclusion

The prospectively frozen read-only replication returns
`replicated_field_fingerprint` independently in both mandatory fresh backbones,
seeds 2106 and 2107. The prerequisite neural-versus-posterior endpoint slope
difference is positive in each network, and all three registered fingerprint
links reproduce with their frozen materiality rules:

1. the additive-source main effect `A` is materially positive;
2. the neural-norm posterior-shape reduction `Q_shape` is materially positive;
3. the fixed-sigmoid field-reassembly interaction `I` is materially negative.

This promotes the 2104/2105 sequential result into a four-network-stable
comparator-relative field fingerprint. It does not establish network-population
prevalence, a sufficient field component, or a realizable recurrent mechanism.

The secondary boundaries also reproduce. Both natural posterior-additive
replacement and neural-norm posterior-shape replacement remain materially
above the posterior anchor. The residual-source main effect `R` remains
unresolved in both networks and was not a primary replication gate. It must not
be described as zero, small, or equivalent.

## Frozen protocol and execution provenance

- The replication contract was committed and pushed as `ff2212bc` before the
  runner was implemented, either mandatory backbone was generated, or either
  Liu evaluation ran.
- The runner, tests, formal dispatch, and implementation/source lock were
  committed and pushed as `08db0a82` before training.
- Both complete 1000-step v1 backbones were trained in mandatory order, 2106
  then 2107, without Liu evaluation. Their complete artifact sets were then
  jointly hash-locked, committed, and pushed as `54a48ffc` before either Liu
  evaluation.
- Checkpoint SHA-256 values are
  `4c1e0261080f37782472082479f95b04290edc395ad9696e85d0abbf4db8aa9c`
  for seed 2106 and
  `fdd8017e01f5f3b74e1a8126016166e63509a45e07218eb4369c8e04e05b3363`
  for seed 2107.
- The only evaluated condition is the pure `L`-off v1 global branch with
  intact `P_T` and frozen `W_out`. No local trace or gain is constructed,
  trained, or read.
- Each network uses all 77 participants, all 28 canonical edges, both neural
  query orientations, and the frozen 20-pair nonlearned mask. Each network has
  its own 10,000-draw participant bootstrap. Participants and networks are
  never pooled.
- Execution used the NVIDIA GeForce RTX 5090 through
  `python -m fsrl.formal_runtime`, with one PyTorch intra-op and one inter-op
  CPU thread.
- A second complete GPU evaluation to `/tmp` is byte-identical to the canonical
  result. Result SHA-256:
  `21592db1f9dc1a35bdca05a0eddc62c5b8a7bcec2a6abf90d103fac33b138600`.

All 30 source checks, all 10 artifact checks, both competence batteries, and
both numerical-integrity batteries pass. The natural neural and posterior
probability endpoints reconstruct directly to at most `3.89e-16`; all
participant and bootstrap factorial identities hold to floating-point error.

## 1. The prerequisite endpoint mismatch replicates

For each participant, the four fixed cells remain

```text
m_N = g_N + c_N
m_P = g_P + c_P
S_ab = slope_d sigmoid[y(g_a + c_b)/T]
```

where the first index selects the additive source and the second selects the
residual source.

| Seed | `S_NN` | `S_PN` | `S_NP` | `S_PP` |
| --- | ---: | ---: | ---: | ---: |
| 2106 | 0.06892 | 0.02734 | 0.08003 | 0.01066 |
| 2107 | 0.06805 | 0.02753 | 0.07765 | 0.01066 |

The frozen prerequisite anchor is positive independently in both networks:

| Seed | `D=S_NN-S_PP` | 95% participant bootstrap |
| --- | ---: | ---: |
| 2106 | 0.05826 | [0.05255, 0.06412] |
| 2107 | 0.05739 | [0.05103, 0.06349] |

This is a fresh-backbone replication of the endpoint fingerprint, not a
network-population estimate.

## 2. All three primary fingerprint links replicate

The registered primary contrasts are:

```text
A = 0.5[(S_NN-S_PN) + (S_NP-S_PP)]
I = (S_NN-S_PN) - (S_NP-S_PP)
Q_shape = S_NN-S_tildePN
```

Here `S_tildePN` uses posterior additive shape rescaled participant-wise to the
neural full-28-edge additive norm, while holding the neural residual fixed.

| Seed | `A` | 95% bootstrap | `Q_shape` | 95% bootstrap | `I` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2106 | 0.05548 | [0.04593, 0.06424] | 0.03454 | [0.02860, 0.04040] | -0.02779 | [-0.03992, -0.01527] |
| 2107 | 0.05375 | [0.04397, 0.06314] | 0.03325 | [0.02665, 0.03953] | -0.02648 | [-0.03827, -0.01461] |

The frozen `0.005` materiality margin is cleared in the registered direction
for every link within each network. Thus the outcome does not depend on pooling
or on a non-significant result being counted as a successful replication.

The corresponding pre-sigmoid additive-by-residual interaction is exactly zero
up to numerical error: the maximum margin-field interaction error is
`3.55e-15`, and the maximum margin-`I` error is `8.88e-16`. The negative
probability-space `I` therefore identifies context dependence introduced by
the fixed sigmoid reassembly. It is not evidence for a recurrent, circuit, or
biological interaction.

## 3. The nonclosure boundaries also reproduce

The natural posterior-additive replacement and the matched-norm
posterior-shape replacement both remain materially above the posterior anchor:

| Seed | `C_A=S_PN-S_PP` | 95% bootstrap | `C_shape=S_tildePN-S_PP` | 95% bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2106 | 0.01667 | [0.01116, 0.02282] | 0.02372 | [0.01868, 0.02936] |
| 2107 | 0.01687 | [0.01116, 0.02340] | 0.02414 | [0.01909, 0.02990] |

Both are materially positive in both networks. These secondary results support
`replicated_nonclosure`: additive allocation/shape contributes to the mismatch
but neither registered replacement closes the comparator gap. The replication
contract did not retest a new sufficiency tree, and this boundary must not be
converted into evidence for a single sufficient component.

The residual-source main effect remains unresolved:

| Seed | `R` | 95% participant bootstrap | Registered status |
| --- | ---: | ---: | --- |
| 2106 | 0.00278 | [-0.00723, 0.01331] | unresolved |
| 2107 | 0.00363 | [-0.00632, 0.01377] | unresolved |

`R` was deliberately excluded from the primary conjunction because repeated
failure to reject zero is not a replicable positive proposition. No
equivalence result is available.

## Revised working theory and claim boundary

Together with the independent `P_T` provenance result, the replicated field
fingerprint supports the constrained chain

```text
P_T-dependent global policy
    -> neural additive field g_N carrying the neural distance slope
    -> comparator-relative additive confidence allocation
    x residual context through the fixed sigmoid
    -> pairwise probability-field slope discrepancy
```

The promoted object is the joint fingerprint `A>0`, `Q_shape>0`, and `I<0`,
conditional on `D>0`. It is not the stronger claim that `g_N` alone, `c_N`
alone, or their natural replacement is sufficient. The exact posterior remains
a frozen comparator rather than the human posterior, correct neural geometry,
or behavioral ground truth. The hybrid fields are offline reassemblies, not
states shown to be realizable by the network.

The evidence is now stable across four development networks, but two newly
replicated backbones are not a network-population prevalence study. Preserve
all network-wise estimates and refrain from network pooling or a population
bootstrap.

## Stop/go and next decisive stage

The registered GO condition for a separately frozen read-only allocation audit
is met. Before any `P_T`, recurrent, readout, temperature, or model
intervention, the next contract should localize

```text
Delta g = g_N - g_P_tilde
```

along prospectively fixed axes: pair identity, symbolic distance, posterior
uncertainty, and observed/effective evidence coverage. On seeds 2106/2107 this
would be sequential localization of a replicated fingerprint, not another
independent confirmation.

The next audit must preserve the full result rather than select only the
largest contrast: the unresolved `R`, negative fixed-sigmoid context effect,
and material closure gaps remain constraints. A stable allocation fingerprint
could justify a later targeted question about how `P_T -> g_N` produces that
allocation. Failure to localize one would redirect the project toward a
prospectively defined comparator-adequacy test rather than model tuning.

Do not modify the current model, fit a scalar correction, expand posterior
hybrids, or claim component sufficiency before that audit.
