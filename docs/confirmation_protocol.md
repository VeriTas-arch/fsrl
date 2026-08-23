# Frozen confirmation protocol

`07591af56ccd693307ab90cc72b03335cb4cd6c9` is the development freeze. Its
single checkpoint and all `liu-constructive-ranking-v1` outputs remain
developmental evidence; they are not eligible confirmation seeds.

## Source correction and human gate

Liu, Wang, and Luo define the objective order as `A < ... < H`. The development
protocol reversed that order without reflecting the retained graph. The
confirmation protocol therefore uses `benchmarks/liu_v2.json`, with `H` highest
and the eight source relations directed accordingly. Meta-training excludes both
the source-correct rank graph and its rank-axis reflection. This is stricter than
the v1 held-out declaration and prevents the reflected development graph from
entering the confirmation distribution.

`benchmarks/liu_human_exact_v1.json` is regenerated from the two public OSF
trial files before any confirmation training. It records SHA-256 checks for the
raw trials and the released Figure 2d/3b source data, exact combined accuracies,
the published 8/64/5 rank classes, stable-error prevalence, and a fixed 10,000
sample participant bootstrap.

Two source-level ambiguities are preserved rather than silently tuned away:

- A transparent majority-vote circular-triad reconstruction matches 75/77
  released Figure 3b labels. IDs 30 and 38 contain majority ties and are labeled
  inconsistent in the released figure, but the paper does not specify the tie
  rule that produces those two labels. Released labels are the reproduction
  target; reconstructed labels remain in the artifact as a diagnostic.
- The released Figure 2d Beta parameters for 27/28 pairs exactly match clipping
  endpoints to 0.01/0.99 before fixed-support MLE. B-H instead exactly matches
  0.001/0.999. Both the uniform recomputation and the released classifications
  are retained.

The human gate must pass before training:

```bash
direnv exec . python -m fsrl.human_benchmark
direnv exec . python -m fsrl.confirmation validate
```

## Frozen neural procedure

The machine-readable formal contract is `benchmarks/confirmation_v1.json`. It
was frozen only after the two-seed workflow pilot described below. It fixes:

- fresh formal seeds 2001–2010, all mandatory and never filtered;
- the existing architecture and optimizer configuration;
- exactly 1,000 outer steps and only the final checkpoint;
- stable omission during meta-training;
- the v2 causal and antisymmetric-geometry gates;
- a common 77-subject evaluation cohort and common evaluation seeds;
- query temperature 0.25, with no seed-specific refit;
- exact-posterior temperature 0.05 for probabilities (MAP itself is invariant
  to this positive temperature).

Run one registered seed on a CUDA-visible host with:

```bash
direnv exec . python -m fsrl.confirmation run-seed --seed 2001
```

After all ten registered seeds finish, aggregate without optional omissions:

```bash
direnv exec . python -m fsrl.confirmation aggregate \
  --output results/confirmation_v1.json
```

Aggregation fails if any seed is absent. It reports causal, behavioral,
geometry, and joint pass proportions plus full seed-level distributions. The
behavioral criterion is fixed before training: every registered scalar estimate
must lie inside its human participant-bootstrap 95% interval. Standardized
discrepancies and Beta class counts are also reported.

## Two-seed workflow pilot

Seeds 1901 and 1902 are registered separately in `benchmarks/pilot_v1.json`.
They were deliberately limited to two after resource and protocol validation.
The pilot exposed and fixed a v1 assumption in deterministic causal scoring:
canonical item order had been treated as true rank order. Because this was
corrected after pilot training began, neither pilot seed may be counted toward
formal confirmation.

Both pilot seeds passed the frozen causal and antisymmetric-geometry gates.
Both failed the deliberately strict behavioral rule for exactly one reason:
symbolic-distance slopes 0.0466 and 0.0476 exceeded the human bootstrap upper
bound 0.0449. The other eight registered scalar comparisons fell within their
human intervals. The full, unfiltered pilot aggregate is
`results/pilot_v1.json`.

The observed workload consists of many small sequential recurrent CUDA kernels.
A single process used about 5 GB on an RTX 5090 but remained dispatch-bound;
four concurrent seeds consumed about 18.6 GB without improving aggregate
throughput. Future work should therefore validate changes with one to three
pilot seeds and run the formal set sequentially unless a numerically equivalent
batched implementation is separately validated.

## Matched controls and algorithmic interpretation

Each checkpoint is evaluated under stable omission, presentationwise omission,
blockwise omission, and a uniform no-bottleneck control. The last uses the
cohort's mean registered reliability as one deterministic gain for every subject
and relation. Because every relation occurs exactly once in each support block,
blockwise and presentationwise resampling have the same statistical granularity
in this task. They are retained to satisfy the planned control family but cannot
be treated as two independent mechanistic contrasts.

For stable omission, `fsrl.algorithmic` enumerates all 40,320 rankings from the
exact realized evidence and compares the frozen neural order with the MAP set,
posterior pair probabilities, and posterior-supported neural choices. This is a
descriptive algorithmic analysis, not a qualification gate. A close match would
support approximate global inference; a systematic mismatch would identify a
learned neural inductive bias. Neither result alone changes the fast-weight
necessity claim.

## Claim boundary

The confirmation tests whether a shared meta-learned relational prior, combined
with a stable subject-specific evidence bottleneck, constructs an episode-local
fast-weight state that is necessary for subject-specific query-policy selection.
It does not test whether fast plasticity creates transitivity from nothing, nor
whether the full hidden-state manifold represents subjective rank. The registered
geometry claim remains limited to the antisymmetric relational component.
