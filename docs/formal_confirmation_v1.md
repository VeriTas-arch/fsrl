# Formal confirmation v1 results

The two frozen contracts were evaluated on all ten declared independently
trained networks, seeds 2001--2010, without seed filtering. The behavioral
contract is complete but does not pass its all-scalar conjunction. The
mechanism contract confirms six of seven registered links; its complete-chain
conjunction is therefore unresolved rather than confirmed.

The machine-readable records are
[`results/confirmation_v1.json`](../results/confirmation_v1.json) and
[`results/mechanism_confirmation_v1.json`](../results/mechanism_confirmation_v1.json).
Raw checkpoints and per-seed artifacts remain under the ignored
`output/confirmation-v1/` tree.

## Integrity and competence

- All seeds 2001--2010 are present and each has exactly 1,000 training records.
- Every checkpoint and mechanism input, adapter, component, and orchestration
  hash matches its recorded SHA-256.
- The ten artifacts share one runtime record and one formal runtime, training,
  and orchestration source version.
- All ten networks pass the frozen fast-weight content-necessity qualification.
- All source-reproduction checks pass the registered
  `3.814697265625e-6` tolerance; the largest observed error is
  `1.9073486328125e-6`.
- Training used the source-locked GPU runtime, one PyTorch intra-op and
  inter-op CPU thread, one contiguous trial-input transfer, and
  `torch.compile(net, fullgraph=True)` with the default mode and Inductor
  backend.

The scientific outcomes are therefore interpretable. The failed conjunctions
below are not competence, provenance, or numerical-reproduction failures.

## Behavioral and geometry contract

All ten seeds pass the causal-qualification and antisymmetric-geometry gates.
None passes the deliberately strict behavioral rule requiring every registered
scalar to lie within its human participant-bootstrap 95% interval, so the
joint pass proportion is zero.

| Registered result | Formal result |
| --- | --- |
| Causal qualification | 10/10 pass |
| Antisymmetric subjective-rank geometry | 10/10 pass |
| All-scalar human behavior | 0/10 pass |
| Overall accuracy | mean `0.8317`; 6/10 inside human interval |
| Learned accuracy | mean `0.8929`; 6/10 inside human interval |
| Nonlearned accuracy | mean `0.8072`; 10/10 inside human interval |
| Symbolic-distance slope | mean `0.04881`, range `0.04736--0.05013`; 0/10 inside human interval `0.03475--0.04490` |
| Self-consistent incorrect subjects | mean `0.8494`; 10/10 inside human interval |
| Fully stable errors | mean `0.8022`; 10/10 inside human interval |

The formal population therefore transports the positive evidence backbone:
held-out graph transfer, episode-specific fast-weight necessity, stable
individual differences, coherent errors, and antisymmetric subjective-rank
geometry. It also reproduces the pilot's negative fingerprint. The network is
slightly low in overall and learned-pair accuracy and is systematically too
sensitive to symbolic distance. This pattern is consistent with the earlier
read-only localization: the neural policy is dominated by an additive global
potential, whereas human choices retain a stronger learned-pair residual.

The algorithmic comparison also disfavors a hard MAP-order account. The neural
MAP proportion is only `0.052--0.130`, while closest-MAP Kendall tau remains
`0.732--0.750`. The mechanism test below directly confirms an expected-rank
rather than MAP terminal advantage.

## Mechanism contract

The interval in this table is the registered 95% bootstrap interval across the
ten network-seed means. Except for DA direction, confirmation requires a lower
bound above zero.

| Link and primary estimand | Mean [95% interval] | Status |
| --- | --- | --- |
| Immediate remote absolute effect | `0.3177 [0.3005, 0.3319]` | confirmed |
| Episode LOO remote absolute effect | `0.4339 [0.4258, 0.4414]` | confirmed |
| LOO third-party relational fraction | `0.1915 [0.1878, 0.1952]` | confirmed |
| Eligibility donor-identity advantage | `0.7674 [0.5933, 0.8550]` | confirmed |
| DA high-minus-low write norm | `0.1157 [0.0989, 0.1274]` | confirmed |
| DA high-minus-low policy norm | `0.2341 [0.1804, 0.2684]` | confirmed |
| DA high/low direction cosine | `0.8836 [0.6523, 0.9998]`; frozen lower-bound threshold `0.99` | unresolved |
| Actual-alpha minus permutation-null gain | `1.5199 [1.4878, 1.5562]` | confirmed |
| History baseline-expression effect | `0.03005 [0.02386, 0.03434]` | confirmed |
| History-matched interaction | `0.00493 [0.00375, 0.00660]` | confirmed |
| Expected-rank minus MAP terminal cosine | `0.04834 [0.04739, 0.04918]` | confirmed |

Thus six of the seven linkwise mechanisms are confirmed, but the complete
seven-link chain is not. This preserves the confirmed links rather than
discarding them because one conjunction failed.

The DA result contains scientifically important between-network heterogeneity.
Nine seeds have direction cosines from `0.9950` to `0.99996`. Seed 2009 has a
mean cosine of `-0.1556`, while still passing all competence gates, attaining
the highest behavioral overall accuracy (`0.8353`), and retaining global
reassembly, alpha-gain placement, history effects, and terminal distributional
projection. Its high- and low-DA policy vectors are not near-zero
(`1.1667` and `1.1504` mean norms), so the failed cosine is not a numerical
zero-vector artifact. In the same seed, a matched eligibility transfer has a
small synthetic policy norm (`0.0365`, versus about `1.28--1.43` in the other
nine seeds) and no donor-identity advantage. This supports an implementation-
regime hypothesis: the trained family usually factorizes relation direction
into eligibility and magnitude into modeled DA, but a competent network can
co-adapt DA, eligibility, baseline state, and recurrent sensitivity so that
this isolated factorization is not transportable. That hypothesis is an
inference from the formal heterogeneity, not a newly confirmed primary link.

Two registered nonprimary diagnostics further constrain interpretation.
Immediate and LOO correctness-aligned remote effects are negative,
`-0.00344 [-0.00485, -0.00198]` and
`-0.06640 [-0.06926, -0.06294]`. Retained evidence therefore has robust global
causal reach, but an individual update is not an independently
correctness-propagating step. Conversely, the previously unresolved total
history factor-generation and first-exposure alignment-generation diagnostics
have intervals above zero in the formal population. They remain nonprimary and
must not be promoted post hoc, but they motivate a future registered test.

## Revised working theory

The cross-network invariant core is a meta-learned, state-dependent iterative
relaxation process. Episode-local fast weights are necessary; retained evidence
immediately redistributes remote relations; alpha places writes in recurrently
high-gain directions; accumulated history changes their expression; and the
terminal solution is closer to a distributional expected-rank potential than
to a MAP order.

The formal result rejects the stronger claim that every trained solution uses
the same separable `eligibility = relation direction` and `DA = scalar gain`
implementation. Those factors remain a strongly supported population-typical
decomposition, but not a universal necessary decomposition. The mechanistic
target should therefore move upward from parameter labels to invariant causal
operators, while retaining the parameter-level heterogeneity as an object to
explain.

The behavioral negatives point to a second missing component rather than a
reason to shrink the global-assembly claim. A purely additive global potential
can explain coherent novel-pair judgments and the confirmed reassembly links,
but it cannot by itself explain the stronger human learned-pair residual. The
next candidate should combine:

1. the confirmed fast-weight global expected-rank channel; and
2. an episode-local relation-specific channel that directly preserves learned
   pair evidence at policy readout.

This dual-channel account makes the already-supported global phenomena and the
human local/conjunctive structure compatible. It predicts that strengthening
the local channel raises learned-pair accuracy and flattens symbolic-distance
dependence while preserving nonlearned inference, fast-weight necessity,
subjective geometry, global causal reach, and terminal expected-rank
projection. Merely changing temperature or selecting checkpoints would not
test this explanation.

## Next decisive test

Before a new multi-seed run, use one to three development seeds and freeze a v2
protocol with two separable interventions:

- **Implementation-regime localization:** compare transfer of the complete
  natural effective write with separately transferred DA and eligibility
  factors under matched baseline and alpha. Rescue by the complete write but
  not either factor supports a co-adapted multiplicative code; failure of the
  complete write with preserved end-to-end LOO reach instead locates content in
  baseline-dependent recurrent expression.
- **Dual-channel necessity:** expose global-potential and learned-pair residual
  readouts separately, then ablate each while holding the other fixed. Global
  ablation should selectively damage nonlearned coherent inference and remote
  reassembly; local-channel ablation should selectively restore the excessive
  distance slope and lower learned-pair accuracy.

Advance only if both interventions are competent and the dual-channel candidate
retains the complete positive backbone. If local ablation is not selective,
the learned-pair-residual account is rejected. If a direct complete-write swap
does not rescue seed-2009-like networks, the separable write-content account
must be replaced by a state-distributed mechanism rather than weakened through
seed exclusion.

## Reproduction

Validate frozen sources, then aggregate all mandatory artifacts:

```bash
direnv exec . python -m fsrl.formal_runtime mechanism validate
direnv exec . python -m fsrl.formal_runtime mechanism validate-development
direnv exec . python -m fsrl.formal_runtime confirmation aggregate \
  --output results/confirmation_v1.json
direnv exec . python -m fsrl.formal_runtime mechanism aggregate
```

The v1 contracts and results are immutable evidence. Any architecture,
estimand, threshold, or causal-chain revision belongs in a separately frozen
v2 contract and new seed population.
